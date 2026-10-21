"""Computational engine: quantile classes, the features of T_V(d, s), their exact null
covariance, rank Moran statistics and the exact permutation reference table.

Notation follows the paper. For a centre v with neighbours N(v) and a feature
tau = (c, mu), where c is the label of the basis function carried by the centre
(0: not involved) and mu[a] is the number of neighbours carrying e_{a+1},

    f_{v,tau} = e_c(delta_v) |O_v(tau)|^{-1/2} sum_{A in O_v(tau)} prod_{(u,a) in A} e_a(delta_u),
    W_tau     = K_tau^{-1/2} sum_v f_{v,tau},     T_V(d, s) = sum_tau W_tau^2,

over the features of order 2..s. The neighbour sums are monomial symmetric functions of the
neighbours' basis values; they are computed from power sums P_S = A @ prod_{l in S} X_l with
the sparse adjacency matrix A, for many rank arrangements at once.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import sparse

from .graphs import Graph

Feature = Tuple[int, Tuple[int, ...]]


def basis_matrix(d: int) -> np.ndarray:
    """E[class, a]: discrete orthonormal polynomials on {0..d-1} (uniform weights), e_0 = 1,
    e_a of degree a, normalized so that e_a(highest class) > 0."""
    V = np.vander(np.arange(d), d, increasing=True).astype(float)
    Q, _ = np.linalg.qr(V)
    E = Q * np.sqrt(d)
    E[:, 0] = 1.0
    for a in range(1, d):
        if E[d - 1, a] < 0:
            E[:, a] = -E[:, a]
    return E


def multisets(d: int, size: int) -> List[Tuple[int, ...]]:
    out = []
    for comb in itertools.combinations_with_replacement(range(d - 1), size):
        mu = [0] * (d - 1)
        for a in comb:
            mu[a] += 1
        out.append(tuple(mu))
    return out


def feature_list(d: int, s_max: int) -> List[Feature]:
    feats = []
    for c in range(d):
        for s_nb in range(0, s_max + 1):
            s = (c > 0) + s_nb
            if 2 <= s <= s_max:
                for mu in multisets(d, s_nb):
                    feats.append((c, mu))
    return feats


LAB = {1: "lin", 2: "quad", 3: "cub", 4: "quart", 5: "quint"}


def feature_name(f: Feature) -> str:
    c, mu = f
    parts = []
    for a, k in enumerate(mu):
        if k:
            parts.append(LAB.get(a + 1, f"e{a + 1}") + (f"^{k}" if k > 1 else ""))
    nbp = "n." + "*".join(parts) if parts else ""
    if c:
        return f"c.{LAB.get(c, f'e{c}')}|{nbp}" if nbp else f"c.{LAB.get(c, f'e{c}')}"
    return nbp


def orbit_size(k: int, mu: Sequence[int]) -> int:
    s = sum(mu)
    if s > k:
        return 0
    return math.factorial(k) // (math.prod(math.factorial(x) for x in mu) * math.factorial(k - s))


def ranks(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """0-based ranks; ties are broken at random, which keeps the tests exact for discrete data."""
    order = np.lexsort((rng.random(x.size), x))
    r = np.empty(x.size, dtype=np.int64)
    r[order] = np.arange(x.size)
    return r


class TVEngine:
    """Features of T_V(d, s_max) on a graph."""

    def __init__(self, graph: Graph, d: int, s_max: int):
        if d < 2 or s_max not in (2, 3):
            raise ValueError("d must be at least 2 and s_max 2 or 3")
        self.graph, self.d, self.s_max = graph, d, s_max
        self.n = graph.n
        self.A = graph.adjacency()
        self.E = basis_matrix(d)
        deg = graph.degree
        feats = feature_list(d, s_max)
        size = np.array([[orbit_size(int(k), mu) for k in deg] for (c, mu) in feats], float).reshape(len(feats), -1)
        keep = (size > 0).any(1)
        self.feats: List[Feature] = [f for f, k in zip(feats, keep) if k]
        self.size = size[keep]
        self.Kt = (self.size > 0).sum(1).astype(float)
        self.scale = np.where(self.size > 0, 1.0 / np.sqrt(np.maximum(self.size, 1)), 0.0)
        self.names = [feature_name(f) for f in self.feats]

    # -- neighbour sums ---------------------------------------------------
    def _msym(self, X: Dict[int, np.ndarray], cache: dict, mu: Tuple[int, ...]) -> np.ndarray:
        labels = [a for a, k in enumerate(mu) for _ in range(k)]

        def P(*ls):
            key = tuple(sorted(ls))
            if key not in cache:
                prod = X[key[0]].copy()
                for l in key[1:]:
                    prod *= X[l]
                cache[key] = self.A @ prod
            return cache[key]

        s = len(labels)
        if s == 0:
            return None
        if s == 1:
            return P(labels[0])
        if s == 2:
            a, b = labels
            if a == b:
                return 0.5 * (P(a) ** 2 - P(a, a))
            return P(a) * P(b) - P(a, b)
        a, b, c = labels
        if a == b == c:
            return (P(a) ** 3 - 3.0 * P(a) * P(a, a) + 2.0 * P(a, a, a)) / 6.0
        if a == b or b == c:
            if b == c:
                a, c = c, a                       # now labels (a, a, c)
            return 0.5 * (P(a) ** 2 * P(c) - P(a, a) * P(c) - 2.0 * P(a, c) * P(a) + 2.0 * P(a, a, c))
        return (P(a) * P(b) * P(c) - P(a, b) * P(c) - P(a, c) * P(b) - P(b, c) * P(a) + 2.0 * P(a, b, c))

    def local(self, delta: np.ndarray) -> np.ndarray:
        """Local contributions f_{v,tau}: array (features, n) for one class vector."""
        return self._local(np.asarray(delta)[:, None])[..., 0]

    def _local(self, delta: np.ndarray) -> np.ndarray:
        X = {a: self.E[delta, a + 1] for a in range(self.d - 1)}
        cache: dict = {}
        out = np.empty((len(self.feats), self.n, delta.shape[1]))
        for i, (c, mu) in enumerate(self.feats):
            m = self._msym(X, cache, mu)
            f = self.E[delta, c] if m is None else (m if c == 0 else self.E[delta, c] * m)
            out[i] = f * self.scale[i][:, None]
        return out

    def W(self, delta: np.ndarray) -> np.ndarray:
        """Feature vector(s): delta is (n,) or (n, B) class array; returns (p,) or (p, B)."""
        one = delta.ndim == 1
        D = delta[:, None] if one else delta
        F = self._local(D).sum(1) / np.sqrt(self.Kt)[:, None]
        return F[:, 0] if one else F

    def T(self, delta: np.ndarray) -> np.ndarray:
        w = self.W(delta)
        return np.sum(w * w, axis=0)

    # -- exact null covariance (Proposition 4 of the paper) ---------------
    def sigma(self) -> np.ndarray:
        d, smax = self.d, self.s_max
        fidx = {f: i for i, f in enumerate(self.feats)}
        keys, fid, wts = [], [], []
        for v in range(self.n):
            nbv = sorted(int(u) for u in self.graph.neighbors[v])
            for s_nb in range(0, smax + 1):
                for sub in itertools.combinations(nbv, s_nb):
                    for labs in itertools.product(range(1, d), repeat=s_nb):
                        mu = [0] * (d - 1)
                        for a in labs:
                            mu[a - 1] += 1
                        mu = tuple(mu)
                        for c in range(d):
                            i = fidx.get((c, mu))
                            if i is None:
                                continue
                            sites = list(sub) + ([v] if c else [])
                            lab = list(labs) + ([c] if c else [])
                            o = np.argsort(sites)
                            key = tuple(sites[t] for t in o) + (-1,) * (smax - len(sites)) \
                                + tuple(lab[t] for t in o) + (-1,) * (smax - len(sites))
                            keys.append(key)
                            fid.append(i)
                            wts.append(self.scale[i, v] / math.sqrt(self.Kt[i]))
        keys = np.array(keys, np.int64)
        _, g = np.unique(keys, axis=0, return_inverse=True)
        g = np.asarray(g).ravel()
        C = sparse.coo_matrix((np.array(wts), (g, np.array(fid))), shape=(g.max() + 1, len(self.feats))).tocsr()
        return (C.T @ C).toarray()


# ----------------------------------------------------------------------
# Rank Moran statistics
# ----------------------------------------------------------------------
class RankMoran:
    """RM = |I(ranks) + 1/(n-1)| and RS = |I(squared standardized ranks) + 1/(n-1)|."""

    def __init__(self, graph: Graph):
        self.W = graph.row_standardized()
        self.S0 = float(self.W.sum())
        self.n = graph.n

    def _I(self, Y: np.ndarray) -> np.ndarray:
        Yc = Y - Y.mean(0, keepdims=True)
        num = np.sum(Yc * (self.W @ Yc), axis=0)
        den = np.sum(Yc * Yc, axis=0)
        return self.n / self.S0 * num / den

    def stats(self, R: np.ndarray):
        """R: (n,) or (n, B) array of 0-based ranks. Returns (RM, RS)."""
        one = R.ndim == 1
        R2 = (R[:, None] if one else R).astype(float)
        Z = (R2 - (self.n - 1) / 2.0) / R2.std(0, keepdims=True)
        e = 1.0 / (self.n - 1)
        rm = np.abs(self._I(R2) + e)
        rs = np.abs(self._I(Z * Z) + e)
        return (rm[0], rs[0]) if one else (rm, rs)


# ----------------------------------------------------------------------
# Weighted chi-square limits
# ----------------------------------------------------------------------
def _group(lam: np.ndarray, tol: float = 1e-8):
    lam = np.sort(lam[lam > tol])[::-1]
    w, df = [], []
    for x in lam:
        if w and abs(x - w[-1]) <= 1e-6 * max(1.0, x):
            df[-1] += 1
        else:
            w.append(x); df.append(1)
    return np.array(w), np.array(df, float)


def weighted_chi2_sf(t: float, lam) -> float:
    """P(sum_i lam_i chi2_1 > t) by numerical inversion of the characteristic function (Imhof, 1961)."""
    from scipy import integrate, stats
    w, df = _group(np.asarray(lam, float))
    if w.size == 0:
        return 0.0 if t > 0 else 1.0
    if w.size == 1:
        return float(stats.chi2.sf(t / w[0], df[0]))
    mean, var = float(np.sum(w * df)), float(2 * np.sum(w * w * df))
    if t > mean + 12.0 * math.sqrt(var):
        th = 0.25 / w.max()
        return float(min(1.0, math.exp(-th * t - 0.5 * np.sum(df * np.log(1 - 2 * th * w)))))

    def integrand(u):
        theta = 0.5 * np.sum(df * np.arctan(w * u)) - 0.5 * t * u
        rho = np.prod((1.0 + (w * u) ** 2) ** (df / 4.0))
        return np.sin(theta) / (u * rho)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", integrate.IntegrationWarning)
        val, err = integrate.quad(integrand, 0.0, np.inf, limit=500)
    p = 0.5 + val / math.pi
    if p < 1e-6 or err > 1e-4:
        # far tail: Chernoff bound, an upper bound for the P-value
        th = 0.25 / w.max()
        bound = math.exp(-th * t - 0.5 * np.sum(df * np.log(1 - 2 * th * w)))
        p = min(bound, max(p, 0.0)) if p > 0 else min(bound, 1e-6)
    return float(min(max(p, 0.0), 1.0))


def weighted_chi2_quantile(alpha: float, lam) -> float:
    from scipy.optimize import brentq
    w, df = _group(np.asarray(lam, float))
    mean, var = float(np.sum(w * df)), float(2 * np.sum(w * w * df))
    return float(brentq(lambda t: weighted_chi2_sf(t, lam) - alpha, 1e-9, mean + 30 * math.sqrt(var)))

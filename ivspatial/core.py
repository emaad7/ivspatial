"""A nonparametric test of spatial independence on neighbourhood graphs.

Implements the procedure of Ashtari Nezhad, "A nonparametric test of spatial independence on neighbourhood graphs":

* the variable-degree statistics T_V(d, s) (features of order 2..s with d quantile
  classes), which work for lattices, polygons, irregular points and networks,
* the rank Moran statistics RM (level) and RS (scale),
* the combined test C4 = min(p_V(4,3), p_V(4,2), p_RM, p_RS), calibrated exactly
  from one table of random rank arrangements per graph,
* asymptotic P-values from the explicit weighted chi-square limits,
* the standardized features of T_V(3,2), which show whether the dependence is in the
  level, in the scale or between neighbours, and local feature contributions for maps,
* a data-driven choice of the resolution (minimum-P or Schwarz rule) with exact
  post-selection calibration, statistics at several spatial lags, and tests for the
  residuals of regressions with exogenous regressors.

All statistics depend on the data only through the ranks: under the null hypothesis of
independent and identically distributed observations the permutation P-values are exact for
every sample size and every continuous distribution (ties are broken at random).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .engine import RankMoran, TVEngine, ranks, weighted_chi2_quantile, weighted_chi2_sf
from .graphs import Graph, lag_graph

__all__ = ["spatial_test", "SpatialTestResult", "screen", "residual_test", "local_features",
           "null_limit", "ReferenceTable", "RESOLUTION_FAMILY", "benjamini_hochberg"]

RESOLUTION_FAMILY = [(d, s) for s in (2, 3) for d in (2, 3, 4, 5, 6)]


def _fmt_p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def benjamini_hochberg(p) -> np.ndarray:
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


# ----------------------------------------------------------------------
# Components
# ----------------------------------------------------------------------
class _Component:
    def __init__(self, name: str, kind: str, graph: Graph, d: int = 0, s: int = 0, lag: int = 1):
        self.name, self.kind, self.graph, self.d, self.s, self.lag = name, kind, graph, d, s, lag
        if kind == "TV":
            self.eng = TVEngine(graph, d, s)
            self.dim = len(self.eng.feats)
        else:
            self.rm = RankMoran(graph)
            self.dim = 1

    def values(self, R: np.ndarray) -> np.ndarray:
        """R: (n, B) 0-based ranks."""
        n = R.shape[0]
        if self.kind == "TV":
            return self.eng.T((self.d * R) // n)
        rm, rs = self.rm.stats(R)
        return rm if self.kind == "RM" else rs

    def describe(self) -> str:
        lag = f", lag {self.lag}" if self.lag > 1 else ""
        if self.kind == "TV":
            order = "2" if self.s == 2 else f"2..{self.s}"
            return f"T_V({self.d},{self.s}){lag}: {self.dim} features of order {order}, {self.d} classes"
        return ("RM" if self.kind == "RM" else "RS") + lag + (": rank Moran (level)" if self.kind == "RM"
                                                             else ": Moran of squared ranks (scale)")


def _components(graph: Graph, spec: str, lags: int) -> List[_Component]:
    comps = []
    graphs = [graph] + [lag_graph(graph, l) for l in range(2, lags + 1)]
    for l, gr in enumerate(graphs, start=1):
        suf = "" if l == 1 else f"@lag{l}"
        if spec == "C4":
            comps += [_Component("V43" + suf, "TV", gr, 4, 3, l), _Component("V42" + suf, "TV", gr, 4, 2, l),
                      _Component("RM" + suf, "RM", gr, lag=l), _Component("RS" + suf, "RS", gr, lag=l)]
        elif spec == "family":
            comps += [_Component(f"V{d}{s}" + suf, "TV", gr, d, s, l) for d, s in RESOLUTION_FAMILY]
        else:
            raise ValueError(spec)
    return comps


class ReferenceTable:
    """Statistics of B random rank arrangements on a graph; data-free, reusable for every
    variable observed on the same graph."""

    def __init__(self, graph: Graph, spec: str = "C4", lags: int = 1, B: int = 9999, seed=None,
                 chunk: int = 250, progress=None):
        self.graph, self.spec, self.lags, self.B = graph, spec, lags, int(B)
        self.comps = _components(graph, spec, lags)
        self.names = [c.name for c in self.comps]
        rng = np.random.default_rng(seed)
        n = graph.n
        out = np.empty((self.B, len(self.comps)))
        done = 0
        while done < self.B:
            b = min(chunk, self.B - done)
            R = np.argsort(rng.random((n, b)), axis=0)
            for j, c in enumerate(self.comps):
                out[done:done + b, j] = c.values(R)
            done += b
            if progress:
                progress(done / self.B)
        self.ref = out
        # reference P-values of every arrangement (for exact minimum-P combinations)
        P = np.empty_like(out)
        for j in range(out.shape[1]):
            s = np.sort(out[:, j])
            P[:, j] = (self.B - np.searchsorted(s, out[:, j] - 1e-9 * np.abs(out[:, j]), side="left")) / self.B
        self.refP = P
        self.rng = rng

    def p_values(self, T: np.ndarray) -> np.ndarray:
        return np.array([(1 + np.sum(self.ref[:, j] >= T[j] - 1e-9 * abs(T[j]))) / (self.B + 1)
                         for j in range(len(T))])

    def p_min(self, p: np.ndarray, cols: Sequence[int]) -> float:
        mref = self.refP[:, cols].min(1)
        return float((1 + np.sum(mref <= p[cols].min() + 1e-12)) / (self.B + 1))


# ----------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------
@dataclass
class Component:
    name: str
    description: str
    statistic: float
    p_value: float
    p_asymptotic: Optional[float] = None


@dataclass
class SpatialTestResult:
    label: str
    n: int
    graph: str
    B: int
    alpha: float
    method: str
    p_value: float
    components: List[Component]
    features: Dict[str, float] = field(default_factory=dict)
    selected: Optional[str] = None
    n_removed: int = 0
    residuals: str = ""

    @property
    def reject(self) -> bool:
        return self.p_value <= self.alpha

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("label", "n", "graph", "B", "alpha", "method", "p_value", "selected",
                                           "n_removed", "residuals")}
        d["reject"] = self.reject
        d["components"] = [c.__dict__ for c in self.components]
        d["features"] = dict(self.features)
        return d

    def summary(self) -> str:
        title = "Nonparametric test of spatial independence" + (f"  [{self.label}]" if self.label else "")
        L = [title, "=" * len(title), f"Observations n = {self.n}" +
             (f"  ({self.n_removed} missing values removed)" if self.n_removed else "")]
        if self.residuals:
            L.append(f"Data: {self.residuals}")
        L += [f"Neighbourhood graph: {self.graph}", f"Random rank arrangements B = {self.B}", ""]
        L.append(f"{'Test':<56}{'Statistic':>11}{'P-value':>11}{'asympt.':>11}")
        L.append(f"{self.method + '  [proposed]':<56}{'':>11}{_fmt_p(self.p_value):>11}{'':>11}")
        for c in self.components:
            pa = "" if c.p_asymptotic is None else _fmt_p(c.p_asymptotic)
            L.append(f"{'  ' + c.description[:54]:<56}{c.statistic:>11.3f}{_fmt_p(c.p_value):>11}{pa:>11}")
        L.append("")
        L.append(f"Spatial independence is {'REJECTED' if self.reject else 'not rejected'} at the "
                 f"{self.alpha:g} level ({self.method}).")
        if self.selected:
            L.append(f"Resolution selected by the data: {self.selected} (exactly calibrated after selection).")
        if self.features:
            L += ["", "Standardized features of T_V(3,2) (approximately N(0,1) under independence):"]
            expl = {"c.lin|n.lin": "level: site and neighbours similar",
                    "c.quad|n.quad": "scale: extremes next to extremes",
                    "n.lin^2": "level: neighbours similar to each other",
                    "n.quad^2": "scale: neighbours jointly extreme",
                    "c.lin|n.quad": "low/high site among extreme neighbours",
                    "c.quad|n.lin": "extreme site among low/high neighbours",
                    "n.lin*quad": "mixed level-scale between neighbours"}
            for k, v in self.features.items():
                L.append(f"  {k:<16}{v:>8.2f}   {expl.get(k, '')}")
            L.append("  |value| > 3 is notable; the sign of mixed features depends on the coding direction.")
        return "\n".join(L)

    __str__ = summary


# ----------------------------------------------------------------------
# Main functions
# ----------------------------------------------------------------------
_ASY_SIGMA_LIMIT = 6_000_000      # maximum number of labelled sets enumerated for an asymptotic P-value


def _n_sets(graph: Graph, d: int, s: int) -> float:
    k = graph.degree.astype(float)
    tot = 0.0
    for j in range(0, s + 1):
        tot += np.sum([math.comb(int(x), j) for x in k]) * (d - 1) ** j * d
    return tot


def _clean(x, n_expected: int):
    x = np.asarray(x, float).ravel()
    if x.size != n_expected:
        raise ValueError(f"the variable has {x.size} values but the graph has {n_expected} sites")
    ok = np.isfinite(x)
    return x, ok


def _standardized_features(graph: Graph, R: np.ndarray, cache: dict) -> Dict[str, float]:
    key = ("feat", id(graph))
    if key not in cache:
        eng = TVEngine(graph, 3, 2)
        cache[key] = (eng, np.sqrt(np.diag(eng.sigma())))
    eng, sd = cache[key]
    w = eng.W((3 * R) // R.size)
    return {nm: float(wi / si) for nm, wi, si in zip(eng.names, w, sd)}


def _asymptotic(comp: _Component, T: float, cache: dict, asymptotic) -> Optional[float]:
    if not asymptotic or comp.kind != "TV":
        return None
    if asymptotic != "all" and _n_sets(comp.graph, comp.d, comp.s) > _ASY_SIGMA_LIMIT:
        return None
    key = ("lam", id(comp.graph), comp.d, comp.s)
    if key not in cache:
        lam = np.linalg.eigvalsh(comp.eng.sigma())
        cache[key] = lam[lam > 1e-10]
    return weighted_chi2_sf(T, cache[key])


def spatial_test(x: Sequence[float], graph: Graph, B: int = 9999, seed: Optional[int] = None,
                 alpha: float = 0.05, resolution: str = "C4", lags: int = 1, label: str = "",
                 features: bool = True, asymptotic=True, reference: Optional[ReferenceTable] = None,
                 _cache: Optional[dict] = None, residuals: str = "") -> SpatialTestResult:
    """Nonparametric test of spatial independence.

    Parameters
    ----------
    x : sequence of numbers
        The variable at the n sites of ``graph`` (same order). Missing values are not allowed
        (remove the sites and build the graph for the remaining ones).
    graph : Graph
        Neighbourhood graph (see knn_graph, delaunay_graph, lattice_graph, polygon_contiguity_graph).
    B : int
        Number of random rank arrangements (default 9999). The P-values are exact for every B.
    seed : int, optional
        Seed for reproducible P-values.
    resolution : {"C4", "auto", "schwarz"}
        "C4" (default): the combined test of the paper. "auto": T_V(d, s) for d = 2..6, s = 2, 3,
        combined by the minimum-P rule, which also reports the resolution chosen by the data.
        "schwarz": the Schwarz-type penalized choice among the same statistics. Both data-driven
        rules are calibrated exactly, the selection being repeated in every rank arrangement.
    lags : int
        1 (default) or more: add the same components on the designs at graph distance 2..lags and
        combine all of them by the minimum-P rule.
    features : bool
        Report the standardized features of T_V(3,2).
    asymptotic : bool or "all"
        Also report asymptotic P-values of the T_V components (skipped for very large designs
        unless "all").
    """
    cache = {} if _cache is None else _cache
    x, ok = _clean(x, graph.n)
    if not ok.all():
        raise ValueError(f"{int((~ok).sum())} missing or non-finite values: remove those sites and rebuild the graph")
    n = graph.n
    if n < 20:
        raise ValueError("at least 20 sites are needed")
    if np.unique(x).size < 3:
        raise ValueError("the variable is (almost) constant")
    if graph.degree.min() == 0:
        warnings.warn(f"{int(np.sum(graph.degree == 0))} isolated site(s) contribute nothing")
    spec = "C4" if resolution == "C4" else "family"
    if resolution not in ("C4", "auto", "schwarz"):
        raise ValueError("resolution must be 'C4', 'auto' or 'schwarz'")
    ref = reference
    if ref is None or ref.spec != spec or ref.lags != lags or ref.graph is not graph:
        ref = ReferenceTable(graph, spec, lags, B, seed)
    rng = np.random.default_rng(None if seed is None else seed + 7919)
    r = ranks(x, rng)
    R = r[:, None]
    T = np.array([c.values(R)[0] for c in ref.comps])
    p = ref.p_values(T)
    comps = [Component(c.name, c.describe(), float(T[j]), float(p[j]), _asymptotic(c, T[j], cache, asymptotic))
             for j, c in enumerate(ref.comps)]
    selected = None
    if resolution == "C4" or resolution == "auto":
        pC = ref.p_min(p, list(range(len(ref.comps))))
        method = ("C4 = min-P(V43, V42, RM, RS)" if resolution == "C4" else "min-P over T_V(d,s), d=2..6, s=2,3") + \
                 (f", lags 1..{lags}" if lags > 1 else "")
        if resolution == "auto":
            cand = np.flatnonzero(p <= p.min() + 1e-12)
            # ties (e.g. all P-values at 1/(B+1)): the largest standardized statistic decides
            z = (T[cand] - ref.ref[:, cand].mean(0)) / ref.ref[:, cand].std(0)
            j = cand[int(np.argmax(z))]
            selected = ref.comps[j].describe().split(":")[0]
    else:
        dims = np.array([c.dim for c in ref.comps], float)
        pen = math.log(n)

        def sel(Tv):
            crit = Tv - dims * pen
            best = np.flatnonzero(crit >= crit.max() - 1e-12)
            return best[np.argmin(dims[best])]
        refsel = np.array([ref.ref[b, sel(ref.ref[b])] for b in range(ref.B)])
        j = sel(T)
        pC = float((1 + np.sum(refsel >= T[j] - 1e-9 * abs(T[j]))) / (ref.B + 1))
        selected = ref.comps[j].describe().split(":")[0]
        method = "Schwarz-selected T_V" + (f", lags 1..{lags}" if lags > 1 else "")
    feats = _standardized_features(graph, r, cache) if features else {}
    return SpatialTestResult(label=label, n=n, graph=graph.summary(), B=ref.B, alpha=alpha, method=method,
                             p_value=pC, components=comps, features=feats, selected=selected, residuals=residuals)


def screen(table: Dict[str, Sequence[float]], graph: Graph, B: int = 9999, seed: Optional[int] = None,
           alpha: float = 0.05, **kw):
    """Test several variables observed on the same graph; one reference table is shared, and the
    P-values are adjusted by the Benjamini-Hochberg procedure. Returns (results, adjusted P-values)."""
    resolution = kw.get("resolution", "C4")
    lags = kw.get("lags", 1)
    ref = ReferenceTable(graph, "C4" if resolution == "C4" else "family", lags, B, seed)
    cache: dict = {}
    res = [spatial_test(v, graph, B=B, seed=seed, alpha=alpha, label=k, reference=ref, _cache=cache, **kw)
           for k, v in table.items()]
    return res, benjamini_hochberg([r.p_value for r in res])


def residual_test(y: Sequence[float], X, graph: Graph, intercept: bool = True, **kw) -> SpatialTestResult:
    """Test the OLS residuals of y on the columns of X. Valid without correction when the regressors
    are exogenous (Proposition 6 of the paper); not for spatial-lag models (y regressed on W y)."""
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if intercept:
        X = np.column_stack([np.ones(y.size), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    r2 = 1 - e.var() / y.var()
    return spatial_test(e, graph, residuals=f"OLS residuals on {X.shape[1] - intercept} regressor(s)"
                                            f"{' and an intercept' if intercept else ''}, R^2 = {r2:.3f}", **kw)


def local_features(x: Sequence[float], graph: Graph, d: int = 4, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Local contributions f_{v,tau} of the features of T_V(d,2) at every site (mean 0, variance 1
    under independence); their scaled sum is the feature W_tau. Useful for maps; not tests."""
    x, ok = _clean(x, graph.n)
    if not ok.all():
        raise ValueError("missing values are not allowed")
    r = ranks(x, np.random.default_rng(seed))
    eng = TVEngine(graph, d, 2)
    F = eng.local((d * r) // graph.n)
    return {nm: F[i] for i, nm in enumerate(eng.names)}


def null_limit(graph: Graph, d: int = 4, s: int = 2, value: Optional[float] = None) -> dict:
    """Null limit of T_V(d, s) on the graph: weights, mean, variance, critical values, P-value of a value."""
    eng = TVEngine(graph, d, s)
    lam = np.linalg.eigvalsh(eng.sigma())
    lam = lam[lam > 1e-10]
    out = {"d": d, "s": s, "features": len(eng.feats), "eigenvalues": lam, "mean": float(lam.sum()),
           "variance": float(2 * np.sum(lam ** 2)),
           "critical": {a: weighted_chi2_quantile(a, lam) for a in (0.10, 0.05, 0.01)}}
    if value is not None:
        out["p_value"] = weighted_chi2_sf(float(value), lam)
    return out

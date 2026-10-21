"""Neighbourhood graphs: points, lattices, polygons, edge lists and spatial lags.

A :class:`Graph` stores, for every site v, the array of its neighbours N(v). The
neighbourhood of a centre is v together with N(v); graphs may be directed (k nearest
neighbours) or undirected (contiguity, Delaunay, lattices).
"""
from __future__ import annotations

import warnings
from typing import Iterable, List, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

__all__ = ["Graph", "knn_graph", "delaunay_graph", "distance_band_graph", "lattice_graph",
           "lattice_from_indices", "graph_from_edges", "graph_from_neighbors", "polygon_contiguity_graph", "lag_graph"]


class Graph:
    """Neighbourhood graph on n sites.

    Parameters
    ----------
    neighbors : list of sequences of int
        ``neighbors[v]`` are the neighbours of site v (0-based, without v itself).
    kind : str
        Short description, shown in reports.
    coords : array (n, 2), optional
        Coordinates of the sites, used only for maps.
    """

    def __init__(self, neighbors: Sequence[Iterable[int]], kind: str = "graph", coords=None):
        nb = [np.unique(np.asarray(list(a), dtype=np.int64)) for a in neighbors]
        n = len(nb)
        for v, a in enumerate(nb):
            if a.size and (a.min() < 0 or a.max() >= n):
                raise ValueError(f"neighbour index out of range at site {v}")
            if np.any(a == v):
                nb[v] = a[a != v]
        self.neighbors: List[np.ndarray] = nb
        self.n = n
        self.kind = kind
        self.coords = None if coords is None else np.asarray(coords, float)
        self.degree = np.array([a.size for a in nb], dtype=np.int64)

    # ------------------------------------------------------------------
    def adjacency(self) -> sparse.csr_matrix:
        """Binary matrix A with A[v, u] = 1 if u is a neighbour of v."""
        rows = np.repeat(np.arange(self.n), self.degree)
        cols = np.concatenate(self.neighbors) if self.n else np.zeros(0, np.int64)
        return sparse.csr_matrix((np.ones(cols.size), (rows, cols)), shape=(self.n, self.n))

    def row_standardized(self) -> sparse.csr_matrix:
        A = self.adjacency()
        deg = np.maximum(self.degree, 1).astype(float)
        return sparse.diags(1.0 / deg) @ A

    def is_symmetric(self) -> bool:
        A = self.adjacency()
        return (A != A.T).nnz == 0

    def symmetrized(self) -> "Graph":
        A = self.adjacency()
        A = ((A + A.T) > 0).tocsr()
        return Graph([A.indices[A.indptr[v]:A.indptr[v + 1]] for v in range(self.n)], self.kind + ", symmetrized",
                     self.coords)

    def summary(self) -> str:
        deg = self.degree
        iso = int(np.sum(deg == 0))
        s = (f"{self.kind}: n = {self.n} sites, degree mean {deg.mean():.2f} (min {deg.min()}, max {deg.max()})"
             + ("" if self.is_symmetric() else ", directed"))
        if iso:
            s += f", {iso} isolated site(s)"
        return s

    def __repr__(self):
        return f"<Graph {self.summary()}>"


# ----------------------------------------------------------------------
# Constructors
# ----------------------------------------------------------------------
def _coords(coords) -> np.ndarray:
    xy = np.asarray(coords, float)
    if xy.ndim != 2 or xy.shape[1] < 1:
        raise ValueError("coords must be an (n, 2) array")
    if not np.all(np.isfinite(xy)):
        raise ValueError("coordinates must be finite")
    return xy


def knn_graph(coords, k: int = 4) -> Graph:
    """Each site and its k nearest neighbours (directed graph; ties in distance broken by index)."""
    xy = _coords(coords)
    n = xy.shape[0]
    if not 1 <= k < n:
        raise ValueError("k must be between 1 and n - 1")
    if np.unique(xy, axis=0).shape[0] < n:
        warnings.warn("duplicate coordinates: their order among nearest neighbours is arbitrary")
    tree = cKDTree(xy)
    dist, idx = tree.query(xy, k=k + 1)
    nb = []
    for v in range(n):
        # all candidates up to the k-th distance, ordered by (distance, index): ties broken by index
        dk = dist[v, k]
        cand = np.array([u for u in tree.query_ball_point(xy[v], dk * (1 + 1e-12) + 1e-300) if u != v], np.int64)
        dd = np.sum((xy[cand] - xy[v]) ** 2, axis=1)
        nb.append(cand[np.lexsort((cand, dd))][:k].tolist())
    return Graph(nb, f"{k}-nearest-neighbour graph", xy)


def delaunay_graph(coords) -> Graph:
    """Delaunay triangulation of the points (the contiguity graph of their Voronoi cells)."""
    from scipy.spatial import Delaunay
    xy = _coords(coords)
    n = xy.shape[0]
    if n < 4:
        raise ValueError("at least 4 points are needed")
    if np.unique(xy, axis=0).shape[0] < n:
        raise ValueError("duplicate coordinates: Delaunay triangulation needs distinct points "
                         "(use a k-nearest-neighbour graph or jitter the points)")
    tri = Delaunay(xy)
    nb = [set() for _ in range(n)]
    for s in tri.simplices:
        for a in s:
            nb[a].update(int(b) for b in s if b != a)
    return Graph([sorted(s) for s in nb], "Delaunay graph", xy)


def distance_band_graph(coords, radius: float) -> Graph:
    """Sites within Euclidean distance ``radius`` of each other."""
    xy = _coords(coords)
    lists = cKDTree(xy).query_ball_point(xy, r=radius)
    return Graph([[u for u in a if u != v] for v, a in enumerate(lists)], f"distance band {radius:g}", xy)


def lattice_graph(nrow: int, ncol: int, kind: str = "rook") -> Graph:
    """Regular grid stored row by row (site index = row * ncol + column); rook or queen contiguity."""
    if nrow < 1 or ncol < 1:
        raise ValueError("nrow and ncol must be positive")
    if kind not in ("rook", "queen"):
        raise ValueError("kind must be 'rook' or 'queen'")
    off = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    if kind == "queen":
        off += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    nb = []
    for i in range(nrow):
        for j in range(ncol):
            nb.append([(i + a) * ncol + j + b for a, b in off if 0 <= i + a < nrow and 0 <= j + b < ncol])
    ii, jj = np.divmod(np.arange(nrow * ncol), ncol)
    return Graph(nb, f"{nrow} x {ncol} lattice, {kind} contiguity", np.c_[jj, nrow - 1 - ii])


def lattice_from_indices(rows, cols, kind: str = "rook") -> Graph:
    """Lattice graph for sites given by integer row and column indices (cells may be missing);
    the order of the sites is that of the input."""
    rows = np.asarray(rows).astype(np.int64)
    cols = np.asarray(cols).astype(np.int64)
    if kind not in ("rook", "queen"):
        raise ValueError("kind must be 'rook' or 'queen'")
    pos = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    if len(pos) < rows.size:
        raise ValueError("duplicate (row, column) cells")
    off = [(0, 1), (1, 0), (0, -1), (-1, 0)] + ([(1, 1), (1, -1), (-1, 1), (-1, -1)] if kind == "queen" else [])
    nb = [[pos[(int(r) + a, int(c) + b)] for a, b in off if (int(r) + a, int(c) + b) in pos]
          for r, c in zip(rows, cols)]
    return Graph(nb, f"lattice ({rows.max() - rows.min() + 1} x {cols.max() - cols.min() + 1} cells), {kind} contiguity",
                 np.c_[cols, -rows])


def graph_from_edges(i, j, n: Optional[int] = None, symmetric: bool = True, base: int = 0) -> Graph:
    """Graph from an edge list (i[e], j[e]); ``base=1`` for 1-based indices."""
    i = np.asarray(i, dtype=np.int64) - base
    j = np.asarray(j, dtype=np.int64) - base
    n = int(max(i.max(), j.max()) + 1) if n is None else int(n)
    A = sparse.csr_matrix((np.ones(i.size), (i, j)), shape=(n, n))
    if symmetric:
        A = A + A.T
    A = (A > 0).tocsr()
    return Graph([A.indices[A.indptr[v]:A.indptr[v + 1]] for v in range(n)], "graph from edge list")


def graph_from_neighbors(neighbors, kind: str = "graph", coords=None) -> Graph:
    return Graph(neighbors, kind, coords)


def polygon_contiguity_graph(polygons, kind: str = "queen", tol: float = 1e-9) -> Graph:
    """Contiguity of polygons given as lists of rings (each ring an (m, 2) array of vertices).

    Queen: polygons sharing at least one vertex; rook: sharing at least one edge. Vertices are
    matched after rounding to ``tol``; for polygons whose boundaries do not share vertices
    exactly, use a GIS package to build the graph and pass it with graph_from_edges().
    """
    if kind not in ("queen", "rook"):
        raise ValueError("kind must be 'queen' or 'rook'")
    owners = {}
    centroids = []
    for p, rings in enumerate(polygons):
        pts_all = []
        for ring in rings:
            ring = np.asarray(ring, float)
            if ring.shape[0] < 3:
                continue
            key = np.round(ring / tol).astype(np.int64)
            pts_all.append(ring)
            if kind == "queen":
                items = {tuple(k) for k in key}
            else:
                items = set()
                for a, b in zip(key[:-1], key[1:]):
                    ta, tb = tuple(a), tuple(b)
                    if ta != tb:
                        items.add((ta, tb) if ta < tb else (tb, ta))
            for it in items:
                owners.setdefault(it, set()).add(p)
        allp = np.vstack(pts_all) if pts_all else np.zeros((1, 2))
        centroids.append(allp.mean(0))
    n = len(polygons)
    nb = [set() for _ in range(n)]
    for own in owners.values():
        if len(own) > 1:
            for a in own:
                nb[a].update(own - {a})
    return Graph([sorted(s) for s in nb], f"polygons, {kind} contiguity", np.array(centroids))


def lag_graph(graph: Graph, lag: int) -> Graph:
    """Sites at graph distance exactly ``lag`` (in the symmetrized graph)."""
    if lag < 1:
        raise ValueError("lag must be at least 1")
    A = graph.adjacency()
    A = ((A + A.T) > 0).astype(float).tocsr()
    n = graph.n
    reach = sparse.identity(n, format="csr")
    prev = reach.copy()
    ring = None
    for _ in range(lag):
        new = ((prev @ A) > 0).astype(float)
        ring = ((new - new.multiply(reach)) > 0).astype(float).tocsr()
        reach = ((reach + new) > 0).astype(float)
        prev = ring
    return Graph([ring.indices[ring.indptr[v]:ring.indptr[v + 1]] for v in range(n)],
                 f"{graph.kind}, lag {lag}", graph.coords)

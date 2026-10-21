"""Building the neighbourhood graph from a data file (shared by the command line and the window)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .graphs import (Graph, delaunay_graph, distance_band_graph, graph_from_edges, knn_graph, lattice_from_indices,
                     lattice_graph, polygon_contiguity_graph)
from .io import read_polygons, read_table

GRAPH_TYPES = ("knn", "delaunay", "band", "lattice", "grid", "polygons", "edges")


@dataclass
class DataSet:
    columns: Dict[str, np.ndarray]
    polygons: Optional[list] = None
    path: str = ""

    @property
    def n(self) -> int:
        return len(next(iter(self.columns.values()))) if self.columns else 0


def load(path: str) -> DataSet:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".shp", ".geojson", ".json"):
        polys, att = read_polygons(path)
        return DataSet(att, polys, path)
    return DataSet(read_table(path), None, path)


def build_graph(ds: DataSet, kind: str, x: str = "", y: str = "", k: int = 4, radius: float = 0.0,
                row: str = "", col: str = "", nrow: int = 0, ncol: int = 0, contiguity: str = "rook",
                edges: str = "", edge_base: int = 1) -> Graph:
    """kind: knn | delaunay | band (coordinates x, y); lattice (row/col index columns); grid (nrow x ncol, rows
    stored row by row); polygons (polygon file); edges (CSV file with two columns of site numbers)."""
    c = ds.columns
    if kind in ("knn", "delaunay", "band"):
        for v in (x, y):
            if v not in c:
                raise ValueError(f"coordinate column '{v}' not found; available: {list(c)}")
        xy = np.c_[c[x], c[y]]
        if not np.all(np.isfinite(xy)):
            raise ValueError("missing coordinates")
        if kind == "knn":
            return knn_graph(xy, k)
        if kind == "delaunay":
            return delaunay_graph(xy)
        if radius <= 0:
            raise ValueError("give a positive distance for the distance band")
        return distance_band_graph(xy, radius)
    if kind == "lattice":
        for v in (row, col):
            if v not in c:
                raise ValueError(f"index column '{v}' not found; available: {list(c)}")
        return lattice_from_indices(c[row], c[col], contiguity)
    if kind == "grid":
        if nrow * ncol != ds.n:
            raise ValueError(f"nrow x ncol = {nrow * ncol} but the file has {ds.n} rows")
        return lattice_graph(nrow, ncol, contiguity)
    if kind == "polygons":
        if ds.polygons is None:
            raise ValueError("polygon contiguity needs a .shp or .geojson file")
        g = polygon_contiguity_graph(ds.polygons, "queen" if contiguity == "queen" else "rook")
        return g
    if kind == "edges":
        tab = read_table(edges)
        cols: List[np.ndarray] = list(tab.values())
        if len(cols) < 2:
            raise ValueError("the edge file needs two numeric columns (site numbers)")
        return graph_from_edges(cols[0], cols[1], n=ds.n, symmetric=True, base=edge_base)
    raise ValueError(f"unknown graph type '{kind}'; use one of {GRAPH_TYPES}")

"""Reading data: CSV/TXT/Excel tables and polygon files (GeoJSON; shapefiles with pyshp or geopandas)."""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np

__all__ = ["read_table", "read_polygons"]


def _to_float(s: str):
    s = s.strip().replace(",", ".") if s.count(",") == 1 and "." not in s else s.strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def read_table(path: str) -> Dict[str, np.ndarray]:
    """Return {column name: numeric array} for every column that is mostly numeric.

    One row per site; columns may hold coordinates, lattice row/column indices and variables."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("reading Excel files needs pandas and openpyxl: pip install pandas openpyxl") from e
        df = pd.read_excel(path)
        cols = {}
        for c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            if np.isfinite(v).mean() > 0.5:
                cols[str(c)] = v
        return cols
    with open(path, newline="", encoding="utf-8-sig") as f:
        text = f.read()
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("the file is empty")
    try:
        delim = csv.Sniffer().sniff("\n".join(lines[:50]), delimiters=",;\t |").delimiter
    except csv.Error:
        delim = None
    rows: List[List[str]] = [l.split() if delim in (None, " ") else next(csv.reader([l], delimiter=delim))
                             for l in lines]
    first = rows[0]
    has_header = any(np.isnan(_to_float(c)) for c in first if c.strip())
    header = [c.strip() or f"column {i + 1}" for i, c in enumerate(first)] if has_header else \
             [f"column {i + 1}" for i in range(len(first))]
    body = rows[1:] if has_header else rows
    cols = {}
    for j in range(len(header)):
        v = np.array([_to_float(r[j]) if j < len(r) else np.nan for r in body], float)
        if v.size and np.isfinite(v).mean() > 0.5:
            cols[header[j]] = v
    if not cols:
        raise ValueError("no numeric column found")
    return cols


def _rings_from_geojson_geom(geom) -> List[np.ndarray]:
    t = geom.get("type")
    if t == "Polygon":
        return [np.asarray(r, float)[:, :2] for r in geom["coordinates"]]
    if t == "MultiPolygon":
        return [np.asarray(r, float)[:, :2] for poly in geom["coordinates"] for r in poly]
    raise ValueError(f"unsupported geometry type {t}")


def _attributes(records: List[dict]) -> Dict[str, np.ndarray]:
    keys = list(records[0]) if records else []
    out = {}
    for k in keys:
        v = np.array([_to_float(str(r.get(k, ""))) if not isinstance(r.get(k), (int, float)) else float(r.get(k))
                      for r in records], float)
        if np.isfinite(v).mean() > 0.5:
            out[k] = v
    return out


def read_polygons(path: str) -> Tuple[List[List[np.ndarray]], Dict[str, np.ndarray]]:
    """Read polygons and their numeric attributes.

    Returns (polygons, attributes): polygons[i] is a list of rings ((m, 2) arrays), attributes maps
    column names to numeric arrays. GeoJSON is read directly; shapefiles need the package pyshp
    (pip install pyshp) or geopandas."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".geojson", ".json"):
        with open(path, encoding="utf-8") as f:
            gj = json.load(f)
        feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
        polys = [_rings_from_geojson_geom(ft["geometry"]) for ft in feats]
        return polys, _attributes([ft.get("properties") or {} for ft in feats])
    if ext == ".shp":
        try:
            import shapefile  # pyshp
        except ImportError:
            try:
                import geopandas as gpd
            except ImportError as e:
                raise ImportError("reading shapefiles needs pyshp (pip install pyshp) or geopandas") from e
            gdf = gpd.read_file(path)
            polys = [_rings_from_geojson_geom(g.__geo_interface__) for g in gdf.geometry]
            recs = gdf.drop(columns="geometry").to_dict("records")
            return polys, _attributes(recs)
        sf = shapefile.Reader(path)
        polys = [_rings_from_geojson_geom(s.__geo_interface__) for s in sf.shapes()]
        names = [f[0] for f in sf.fields[1:]]
        recs = [dict(zip(names, r)) for r in sf.records()]
        return polys, _attributes(recs)
    raise ValueError("polygon files must be .geojson, .json or .shp")

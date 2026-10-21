"""Command line interface.

Examples
--------
  ivspatial houses.csv --x X --y Y --graph knn --k 4 --column PRICE --seed 1
  ivspatial houses.csv --x X --y Y --graph delaunay --all            # every numeric column, FDR adjusted
  ivspatial wheat.csv --graph lattice --row row --col col --column grain
  ivspatial counties.shp --graph polygons --contiguity queen --column HR90 --residuals-on RD90,PS90,UE90
  ivspatial data.csv --graph edges --edges edges.csv --column y
  ivspatial houses.csv --x X --y Y --column PRICE --resolution auto --lags 2 --local local.csv
  ivspatial null houses.csv --x X --y Y --graph knn --k 4 --d 4 --s 2 --value 30
  ivspatial gui                                                      # graphical interface
  ivspatial guide                                                    # user guide in the browser
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings

import numpy as np

from . import __version__
from .core import local_features, null_limit, residual_test, screen, spatial_test
from .design import GRAPH_TYPES, build_graph, load


def open_guide():
    import os
    import webbrowser
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GUIDE.html")
    webbrowser.open("file://" + path)
    print(path)


def _graph_args(ap):
    ap.add_argument("file", help="CSV, TXT, Excel (one row per site), or a .shp / .geojson polygon file")
    ap.add_argument("--graph", "-g", choices=GRAPH_TYPES, default=None,
                    help="neighbourhood graph (default: polygons for polygon files, otherwise knn)")
    ap.add_argument("--x", default="x", help="x coordinate column (knn, delaunay, band)")
    ap.add_argument("--y", default="y", help="y coordinate column")
    ap.add_argument("--k", type=int, default=4, help="number of nearest neighbours (default 4)")
    ap.add_argument("--radius", type=float, default=0.0, help="distance of the distance band")
    ap.add_argument("--row", default="row", help="row index column (lattice)")
    ap.add_argument("--col", default="col", help="column index column (lattice)")
    ap.add_argument("--nrow", type=int, default=0, help="number of rows (grid stored row by row)")
    ap.add_argument("--ncol", type=int, default=0, help="number of columns (grid)")
    ap.add_argument("--contiguity", choices=["rook", "queen"], default=None,
                    help="rook or queen contiguity (default: rook for lattices, queen for polygons)")
    ap.add_argument("--edges", default="", help="CSV file with two columns of site numbers (graph edges)")
    ap.add_argument("--edge-base", type=int, default=1, help="1 if site numbers in the edge file start at 1")


def _make_graph(a):
    ds = load(a.file)
    kind = a.graph or ("polygons" if ds.polygons is not None else "knn")
    cont = a.contiguity or ("queen" if kind == "polygons" else "rook")
    g = build_graph(ds, kind, x=a.x, y=a.y, k=a.k, radius=a.radius, row=a.row, col=a.col, nrow=a.nrow,
                    ncol=a.ncol, contiguity=cont, edges=a.edges, edge_base=a.edge_base)
    return ds, g


def _null(argv):
    ap = argparse.ArgumentParser(prog="ivspatial null", description="Null limit of T_V(d, s) on a graph.")
    _graph_args(ap)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--s", type=int, default=2, choices=[2, 3])
    ap.add_argument("--value", type=float, default=None, help="value of T_V: report its asymptotic P-value")
    a = ap.parse_args(argv)
    _, g = _make_graph(a)
    s = null_limit(g, a.d, a.s, a.value)
    print(g.summary())
    print(f"T_V({a.d},{a.s}): {s['features']} features; limit = weighted sum of {len(s['eigenvalues'])} chi2(1)")
    print(f"mean = {s['mean']:.3f}, variance = {s['variance']:.3f}")
    for lev, c in s["critical"].items():
        print(f"critical value at level {lev:.2f}: {c:.3f}")
    if a.value is not None:
        print(f"P(limit > {a.value:g}) = {s['p_value']:.5f}")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "null":
        return _null(argv[1:])
    if argv and argv[0] == "guide":
        open_guide()
        return 0
    if argv and argv[0] == "gui":
        from .gui import main as gui_main
        gui_main(argv[1] if len(argv) > 1 else None)
        return 0
    ap = argparse.ArgumentParser(prog="ivspatial", description="Nonparametric test of spatial independence.")
    _graph_args(ap)
    ap.add_argument("--column", "-c", action="append", help="variable(s) to test")
    ap.add_argument("--all", action="store_true", help="test every numeric column except coordinates/indices")
    ap.add_argument("--transform", "-t", choices=["none", "log"], default="none",
                    help="log: test the logarithm of the variable(s) (positive values)")
    ap.add_argument("--residuals-on", default="", help="comma-separated regressors: test the OLS residuals")
    ap.add_argument("--B", type=int, default=9999, help="number of random rank arrangements (default 9999)")
    ap.add_argument("--seed", type=int, default=None, help="random seed for reproducible P-values")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--resolution", choices=["C4", "auto", "schwarz"], default="C4",
                    help="C4 (default), auto (minimum-P choice of (d, s)) or schwarz")
    ap.add_argument("--lags", type=int, default=1, help="spatial lags 1..L combined (default 1)")
    ap.add_argument("--local", default="", help="write local feature contributions of T_V(4,2) to this CSV")
    ap.add_argument("--out", help="write the report to this text file")
    ap.add_argument("--json", help="write all results to this JSON file")
    ap.add_argument("--version", action="version", version=f"ivspatial {__version__}")
    a = ap.parse_args(argv)

    ds, g = _make_graph(a)
    skip = {a.x, a.y, a.row, a.col}
    names = a.column or ([c for c in ds.columns if c not in skip] if a.all else [])
    if not names:
        ap.error(f"give --column NAME or --all; numeric columns: {list(ds.columns)}")
    missing = [c for c in names if c not in ds.columns]
    if missing:
        ap.error(f"column(s) not found: {missing}; available: {list(ds.columns)}")
    regs = [s.strip() for s in a.residuals_on.split(",") if s.strip()]
    for r in regs:
        if r not in ds.columns:
            ap.error(f"regressor '{r}' not found")
    kw = dict(B=a.B, seed=a.seed, alpha=a.alpha, resolution=a.resolution, lags=a.lags)
    if a.transform == "log":
        for nm in names:
            v = ds.columns[nm]
            if np.any(v[np.isfinite(v)] <= 0):
                ap.error(f"log transform needs positive values ('{nm}')")
            ds.columns[nm] = np.log(v)
    reports, results = [g.summary()], []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if regs:
            X = np.column_stack([ds.columns[r] for r in regs])
            for nm in names:
                res = residual_test(ds.columns[nm], X, g, label=nm, **kw)
                results.append(res)
        elif len(names) > 1:
            results, q = screen({nm: ds.columns[nm] for nm in names}, g, **kw)
        else:
            results = [spatial_test(ds.columns[names[0]], g, label=names[0], **kw)]
    reports += [r.summary() for r in results]
    if len(results) > 1:
        from .core import benjamini_hochberg
        q = benjamini_hochberg([r.p_value for r in results])
        lines = ["Screening summary (Benjamini-Hochberg adjustment over the variables tested)",
                 f"{'variable':<24}{'p':>10}{'adj. p':>10}  decision"]
        for r, qq in zip(results, q):
            lines.append(f"{r.label[:23]:<24}{r.p_value:>10.4f}{qq:>10.4f}  {'reject' if qq <= a.alpha else '-'}")
        reports.append("\n".join(lines))
    if w:
        reports.append("Notes: " + "; ".join(sorted({str(x.message) for x in w})))
    text = "\n\n".join(reports)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=1)
    if a.local:
        lf = local_features(ds.columns[names[0]], g, seed=a.seed)
        with open(a.local, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["site"] + list(lf))
            for i in range(g.n):
                wr.writerow([i + 1] + [f"{lf[k][i]:.5f}" for k in lf])
        print(f"local feature contributions of '{names[0]}' written to {a.local}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

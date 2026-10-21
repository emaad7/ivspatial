"""Graphical interface (tkinter, included with Python).

Start with:  ivspatial-gui,  python -m ivspatial,  or double-click  ivspatial_gui.pyw

Tabs:
  Test               - load a file, define the neighbourhood graph, choose variables and options, run
  Null distribution  - the weighted chi-square limit of T_V(d, s) on the current graph
"""
from __future__ import annotations

import json
import os
import threading
import warnings

import numpy as np

from . import __version__
from .core import benjamini_hochberg, local_features, null_limit, residual_test, screen, spatial_test
from .design import build_graph, load

HELP = (
    "1. Open a data file: CSV, TXT or Excel with one row per site (coordinates, lattice indices and variables "
    "as columns), or a polygon file (.shp or .geojson) whose attributes are the variables.\n"
    "2. Define the neighbourhood graph: k nearest neighbours, Delaunay triangulation or a distance band "
    "(choose the coordinate columns); a lattice (row and column index columns, or nrow x ncol with rows "
    "stored row by row); polygon contiguity; or an edge list file. Press 'Build graph' to check it.\n"
    "3. Choose the variable (or tick 'test all' to screen every variable with false discovery rate control), "
    "optionally the log transform, and optionally regressors: the OLS residuals are then tested.\n"
    "4. Press Run test. 'Save report' writes a text or JSON file; 'Map' shows the data and the local "
    "contributions of the level and scale features (needs matplotlib).\n\n"
    "C4 is the proposed test. A P-value below the level means that the values are not independent and "
    "identically distributed across the sites. The standardized features show whether the dependence is in the "
    "level (c.lin|n.lin), in the scale (c.quad|n.quad) or between neighbours.\n\n"
    "Resolution 'auto' lets the data choose among T_V(d, s), d = 2..6, s = 2, 3, with exact calibration; "
    "lags = 2 adds the neighbours at graph distance two.")


def main(path=None):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText

    root = tk.Tk()
    root.title(f"ivspatial {__version__} - nonparametric test of spatial independence")
    root.geometry("1040x780")
    state = {"ds": None, "graph": None, "results": [], "last_var": None}

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=6, pady=6)
    tab_test, tab_null = ttk.Frame(nb, padding=10), ttk.Frame(nb, padding=10)
    nb.add(tab_test, text="  Test  ")
    nb.add(tab_null, text="  Null distribution  ")

    def entry(parent, label, var, width=8):
        ttk.Label(parent, text=label).pack(side="left")
        ttk.Entry(parent, textvariable=var, width=width).pack(side="left", padx=(4, 12))

    def combo(parent, label, var, width=14):
        ttk.Label(parent, text=label).pack(side="left")
        cb = ttk.Combobox(parent, textvariable=var, state="readonly", width=width)
        cb.pack(side="left", padx=(4, 12))
        return cb

    def output_box(parent):
        box = ScrolledText(parent, font=("Courier", 10), wrap="none")
        box.pack(fill="both", expand=True, pady=(10, 0))
        return box

    def write(box, text):
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("1.0", text)

    def as_int(s, name, default=None):
        s = s.strip()
        if s in ("", "auto", "none") and default is not None:
            return default
        try:
            return int(s)
        except ValueError:
            raise ValueError(f"{name} must be an integer")

    # ---------------------------------------------------------------- file
    top = ttk.Frame(tab_test)
    top.pack(fill="x")
    file_var = tk.StringVar(value="No file selected")
    ttk.Button(top, text="Open data file...", command=lambda: open_file()).pack(side="left")
    ttk.Label(top, textvariable=file_var, foreground="#444").pack(side="left", padx=8)

    # ---------------------------------------------------------------- graph
    gf = ttk.LabelFrame(tab_test, text="Neighbourhood graph", padding=8)
    gf.pack(fill="x", pady=(10, 0))
    kind_var = tk.StringVar(value="knn")
    kinds = [("k nearest neighbours", "knn"), ("Delaunay", "delaunay"), ("distance band", "band"),
             ("lattice (row, col)", "lattice"), ("grid nrow x ncol", "grid"), ("polygons", "polygons"),
             ("edge list", "edges")]
    rowk = ttk.Frame(gf)
    rowk.pack(fill="x")
    for txt, val in kinds:
        ttk.Radiobutton(rowk, text=txt, variable=kind_var, value=val).pack(side="left", padx=(0, 10))
    rowc = ttk.Frame(gf)
    rowc.pack(fill="x", pady=(6, 0))
    x_var, y_var, r_var, c_var = (tk.StringVar() for _ in range(4))
    cb_x = combo(rowc, "x", x_var, 10)
    cb_y = combo(rowc, "y", y_var, 10)
    k_var, rad_var = tk.StringVar(value="4"), tk.StringVar(value="")
    entry(rowc, "k", k_var, 4)
    entry(rowc, "radius", rad_var, 7)
    cb_r = combo(rowc, "row", r_var, 9)
    cb_c = combo(rowc, "col", c_var, 9)
    rowc2 = ttk.Frame(gf)
    rowc2.pack(fill="x", pady=(6, 0))
    nr_var, nc_var = tk.StringVar(value=""), tk.StringVar(value="")
    entry(rowc2, "nrow", nr_var, 5)
    entry(rowc2, "ncol", nc_var, 5)
    cont_var = tk.StringVar(value="rook")
    ttk.Label(rowc2, text="contiguity").pack(side="left")
    for t in ("rook", "queen"):
        ttk.Radiobutton(rowc2, text=t, variable=cont_var, value=t).pack(side="left", padx=(4, 6))
    edge_var = tk.StringVar(value="")
    ttk.Button(rowc2, text="Edge file...", command=lambda: edge_var.set(
        filedialog.askopenfilename(title="Edge list (two columns of site numbers, starting at 1)") or edge_var.get())
               ).pack(side="left", padx=(12, 4))
    ttk.Label(rowc2, textvariable=edge_var, foreground="#444").pack(side="left")
    ttk.Button(rowc2, text="Build graph", command=lambda: build()).pack(side="right")
    graph_info = tk.StringVar(value="")
    ttk.Label(gf, textvariable=graph_info, foreground="#225").pack(anchor="w", pady=(6, 0))

    # ---------------------------------------------------------------- variables
    vf = ttk.LabelFrame(tab_test, text="Variable and options", padding=8)
    vf.pack(fill="x", pady=(10, 0))
    rv = ttk.Frame(vf)
    rv.pack(fill="x")
    var_var = tk.StringVar()
    cb_v = combo(rv, "Variable", var_var, 18)
    all_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(rv, text="test all (screening)", variable=all_var).pack(side="left", padx=(0, 12))
    log_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(rv, text="log transform", variable=log_var).pack(side="left", padx=(0, 12))
    res_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(rv, text="test OLS residuals on the selected regressors:", variable=res_var).pack(side="left")
    reg_box = tk.Listbox(rv, selectmode="multiple", height=4, exportselection=False, width=22)
    reg_box.pack(side="left", padx=6)
    ro = ttk.Frame(vf)
    ro.pack(fill="x", pady=(6, 0))
    B_var, seed_var, a_var, lag_var = (tk.StringVar(value=v) for v in ("9999", "1", "0.05", "1"))
    entry(ro, "Rank arrangements B", B_var)
    entry(ro, "Seed", seed_var, 6)
    entry(ro, "Level", a_var, 6)
    entry(ro, "Lags", lag_var, 4)
    resol_var = tk.StringVar(value="C4")
    cb_res = combo(ro, "Resolution", resol_var, 9)
    cb_res["values"] = ["C4", "auto", "schwarz"]

    btn = ttk.Frame(tab_test)
    btn.pack(fill="x", pady=(10, 0))
    run_btn = ttk.Button(btn, text="Run test", command=lambda: run())
    run_btn.pack(side="left")
    ttk.Button(btn, text="Save report...", command=lambda: save()).pack(side="left", padx=6)
    ttk.Button(btn, text="Map", command=lambda: show_map()).pack(side="left")
    ttk.Button(btn, text="Help", command=lambda: messagebox.showinfo("How to use", HELP)).pack(side="left", padx=6)

    def _guide():
        from .cli import open_guide
        open_guide()
    ttk.Button(btn, text="User guide", command=_guide).pack(side="left")
    status = tk.StringVar(value="Open a data file.")
    ttk.Label(btn, textvariable=status, foreground="#225").pack(side="left", padx=12)
    out = output_box(tab_test)

    def open_file(path=None):
        path = path or filedialog.askopenfilename(title="Open data file", filetypes=[
            ("Data files", "*.csv *.txt *.dat *.xlsx *.xls *.shp *.geojson *.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            ds = load(path)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Cannot read file", str(e))
            return
        state.update(ds=ds, graph=None)
        cols = list(ds.columns)
        file_var.set(f"{os.path.basename(path)}   ({ds.n} sites, {len(cols)} numeric column(s)"
                     + (", polygons" if ds.polygons is not None else "") + ")")
        for cb in (cb_x, cb_y, cb_r, cb_c, cb_v):
            cb["values"] = cols
        low = {c.lower(): c for c in cols}
        for var, cands in ((x_var, ("x", "lon", "longitude", "easting")), (y_var, ("y", "lat", "latitude", "northing")),
                           (r_var, ("row", "i")), (c_var, ("col", "column", "j"))):
            var.set(next((low[c] for c in cands if c in low), ""))
        var_var.set(next((c for c in cols if c not in (x_var.get(), y_var.get(), r_var.get(), c_var.get())), cols[0]))
        reg_box.delete(0, "end")
        for c in cols:
            reg_box.insert("end", c)
        if ds.polygons is not None:
            kind_var.set("polygons")
            cont_var.set("queen")
        elif r_var.get() and c_var.get():
            kind_var.set("lattice")
        graph_info.set("")
        status.set("Define the graph (Build graph), choose the variable, then Run test.")

    def build(silent=False):
        if state["ds"] is None:
            messagebox.showinfo("No data", "Open a data file first.")
            return None
        try:
            rad = float(rad_var.get()) if rad_var.get().strip() else 0.0
            g = build_graph(state["ds"], kind_var.get(), x=x_var.get(), y=y_var.get(), k=as_int(k_var.get(), "k", 4),
                            radius=rad, row=r_var.get(), col=c_var.get(), nrow=as_int(nr_var.get(), "nrow", 0),
                            ncol=as_int(nc_var.get(), "ncol", 0), contiguity=cont_var.get(), edges=edge_var.get())
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Cannot build the graph", str(e))
            return None
        state["graph"] = g
        graph_info.set(g.summary())
        return g

    def run():
        g = build()
        if g is None:
            return
        ds = state["ds"]
        try:
            B = as_int(B_var.get(), "B", 9999)
            seed = None if not seed_var.get().strip() else as_int(seed_var.get(), "Seed")
            alpha = float(a_var.get())
            lags = as_int(lag_var.get(), "Lags", 1)
        except ValueError as e:
            messagebox.showerror("Invalid option", str(e))
            return
        skip = {x_var.get(), y_var.get(), r_var.get(), c_var.get()} if kind_var.get() != "polygons" else set()
        names = [c for c in ds.columns if c not in skip] if all_var.get() else [var_var.get()]
        regs = [reg_box.get(i) for i in reg_box.curselection()] if res_var.get() else []
        if res_var.get() and not regs:
            messagebox.showinfo("Regressors", "Select at least one regressor, or untick the residual option.")
            return
        names = [n_ for n_ in names if n_ not in regs]
        data = {}
        for nm in names:
            v = ds.columns[nm].astype(float)
            if log_var.get():
                if np.any(v[np.isfinite(v)] <= 0):
                    messagebox.showerror("Log transform", f"'{nm}' has values <= 0")
                    return
                v = np.log(v)
            data[nm] = v
        state["last_var"] = data[names[0]] if names else None
        kw = dict(B=B, seed=seed, alpha=alpha, resolution=resol_var.get(), lags=lags)
        run_btn.configure(state="disabled")
        status.set("Running ... (the reference table is computed once for the graph)")

        def work():
            reports, results = [g.summary()], []
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    if regs:
                        X = np.column_stack([ds.columns[r] for r in regs])
                        results = [residual_test(v, X, g, label=nm, **kw) for nm, v in data.items()]
                    elif len(data) > 1:
                        results, _ = screen(data, g, **kw)
                    else:
                        nm, v = next(iter(data.items()))
                        results = [spatial_test(v, g, label=nm, **kw)]
                reports += [r.summary() for r in results]
                if len(results) > 1:
                    q = benjamini_hochberg([r.p_value for r in results])
                    lines = ["Screening summary (Benjamini-Hochberg adjustment)",
                             f"{'variable':<24}{'p':>10}{'adj. p':>10}  decision"]
                    for r, qq in zip(results, q):
                        lines.append(f"{r.label[:23]:<24}{r.p_value:>10.4f}{qq:>10.4f}  {'reject' if qq <= alpha else '-'}")
                    reports.insert(1, "\n".join(lines))
                if w:
                    reports.append("Notes: " + "; ".join(sorted({str(x.message) for x in w})))
                text = "\n\n".join(reports)
            except Exception as e:  # noqa: BLE001
                text = f"Error: {e}"

            def done():
                state["results"] = results
                write(out, text)
                run_btn.configure(state="normal")
                status.set("Done." if results else "No variable could be tested.")
            root.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def save():
        text = out.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Nothing to save", "Run the test first.")
            return
        path = filedialog.asksaveasfilename(title="Save report", defaultextension=".txt",
                                            filetypes=[("Text report", "*.txt"), ("JSON results", "*.json")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            if path.lower().endswith(".json"):
                json.dump([r.to_dict() for r in state["results"]], f, indent=1)
            else:
                f.write(text + "\n")
        status.set(f"Saved to {os.path.basename(path)}")

    def show_map():
        g, v = state["graph"], state["last_var"]
        if g is None or v is None:
            messagebox.showinfo("Map", "Run a test first.")
            return
        try:
            import matplotlib.pyplot as plt
            from matplotlib.collections import PolyCollection
        except ImportError:
            messagebox.showerror("Map", "The map needs matplotlib: python -m pip install matplotlib")
            return
        if g.coords is None:
            messagebox.showinfo("Map", "This graph has no coordinates (edge list).")
            return
        lf = local_features(v, g)
        rk = (np.argsort(np.argsort(v)) + 0.5) / v.size - 0.5
        panels = [("ranks of the variable", rk, (-0.5, 0.5)),
                  ("local level feature c.lin|n.lin", np.clip(lf["c.lin|n.lin"], -3, 3), (-3, 3)),
                  ("local scale feature c.quad|n.quad", np.clip(lf["c.quad|n.quad"], -3, 3), (-3, 3))]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
        polys = state["ds"].polygons if kind_var.get() == "polygons" else None
        for ax, (title, val, lim) in zip(axes, panels):
            if polys is not None:
                rings = [r for p in polys for r in p[:1]]
                pc = PolyCollection(rings, array=val, cmap="RdBu_r", edgecolor="white", linewidth=0.1)
                pc.set_clim(*lim)
                ax.add_collection(pc)
                ax.autoscale_view()
                fig.colorbar(pc, ax=ax, shrink=0.7)
            else:
                sc = ax.scatter(g.coords[:, 0], g.coords[:, 1], c=val, cmap="RdBu_r", vmin=lim[0], vmax=lim[1],
                                s=max(4, 4000 / g.n), marker="s" if "lattice" in g.kind else "o")
                fig.colorbar(sc, ax=ax, shrink=0.7)
            ax.set_title(title, fontsize=10)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Local feature contributions (mean 0, variance 1 under independence; descriptive, not tests)",
                     fontsize=10)
        fig.tight_layout()
        plt.show()

    # ======================================================================
    # Null distribution tab
    # ======================================================================
    ttk.Label(tab_null, wraplength=960, justify="left", text=(
        "Under independence, T_V(d, s) on the current graph converges to a weighted sum of chi-square(1) "
        "variables whose weights are the eigenvalues of a matrix computed exactly from the graph (Proposition 4 "
        "of the paper). The permutation tests are exact; this tab gives the limit. Build the graph in the Test "
        "tab first. Order s = 3 can take a while on large graphs.")).pack(anchor="w")
    nf = ttk.Frame(tab_null)
    nf.pack(fill="x", pady=(10, 0))
    nd_var, ns_var, nv_var = tk.StringVar(value="4"), tk.StringVar(value="2"), tk.StringVar(value="")
    entry(nf, "classes d", nd_var, 4)
    entry(nf, "order s (2 or 3)", ns_var, 4)
    entry(nf, "value of T_V (optional)", nv_var, 10)
    ttk.Button(nf, text="Compute", command=lambda: null_calc()).pack(side="left", padx=8)
    null_out = output_box(tab_null)

    def null_calc():
        g = state["graph"] or build()
        if g is None:
            return
        try:
            d, s = as_int(nd_var.get(), "d"), as_int(ns_var.get(), "s")
            val = float(nv_var.get()) if nv_var.get().strip() else None
            res = null_limit(g, d, s, val)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Invalid input", str(e))
            return
        L = [g.summary(), "", f"T_V({d},{s}): {res['features']} features; limit = weighted sum of "
             f"{len(res['eigenvalues'])} chi-square(1) variables",
             f"mean = {res['mean']:.3f}, variance = {res['variance']:.3f}", "",
             f"{'level':>8}{'critical value':>18}"]
        for a, c in res["critical"].items():
            L.append(f"{a:>8.2f}{c:>18.3f}")
        if val is not None:
            L += ["", f"P(limit > {val:g}) = {res['p_value']:.5f}"]
        lam = np.sort(res["eigenvalues"])[::-1]
        L += ["", "largest weights: " + ", ".join(f"{x:.3f}" for x in lam[:10])]
        write(null_out, "\n".join(L))

    if path:
        root.after(200, lambda: open_file(path))
        if os.environ.get("IVSPATIAL_GUI_SELFTEST"):
            def selftest():
                B_var.set("499")
                run()
            root.after(1200, selftest)
    root.mainloop()


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)

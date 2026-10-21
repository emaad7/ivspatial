"""Basic checks: graphs, exact size under independence, power under dependence, API."""
import numpy as np
from ivspatial import (delaunay_graph, knn_graph, lattice_graph, lattice_from_indices, local_features, null_limit,
                       residual_test, screen, spatial_test)


def _sar(graph, rho, rng):
    from scipy.sparse import identity
    from scipy.sparse.linalg import spsolve
    W = graph.row_standardized()
    return spsolve((identity(graph.n) - rho * W).tocsc(), rng.standard_normal(graph.n))


def test_graphs():
    g = lattice_graph(4, 5)
    assert g.n == 20 and g.degree.max() == 4 and g.is_symmetric()
    rows, cols = np.divmod(np.arange(20), 5)
    h = lattice_from_indices(rows, cols)
    assert all((a == b).all() for a, b in zip(g.neighbors, h.neighbors))
    xy = np.random.default_rng(0).random((50, 2))
    assert knn_graph(xy, 3).degree.min() == 3 and delaunay_graph(xy).is_symmetric()


def test_api_and_reproducibility():
    rng = np.random.default_rng(1)
    g = delaunay_graph(rng.random((150, 2)))
    x = rng.standard_normal(150)
    a, b = spatial_test(x, g, B=299, seed=3), spatial_test(x, g, B=299, seed=3)
    assert a.p_value == b.p_value and 0 < a.p_value <= 1 and len(a.components) == 4
    assert "c.lin|n.lin" in a.features


def test_size_under_independence():
    rng = np.random.default_rng(2)
    g = knn_graph(rng.random((120, 2)), 4)
    res, _ = screen({f"v{i}": rng.standard_t(3, 120) for i in range(150)}, g, B=199, seed=5)
    rej = np.mean([r.p_value <= 0.05 for r in res])
    assert 0.01 <= rej <= 0.11


def test_power_sar_and_residuals():
    rng = np.random.default_rng(3)
    g = delaunay_graph(rng.random((400, 2)))
    assert spatial_test(_sar(g, 0.5, rng), g, B=499, seed=1).reject
    z = rng.standard_normal(400)
    y = 1 + 2 * z + rng.standard_normal(400)
    assert residual_test(y, z, g, B=499, seed=1).p_value > 0.001


def test_auto_resolution_and_lags():
    rng = np.random.default_rng(4)
    g = lattice_graph(15, 15)
    r = spatial_test(_sar(g, 0.6, rng), g, B=199, seed=2, resolution="auto", lags=2, features=False)
    assert r.reject and r.selected


def test_limit_and_local():
    g = lattice_graph(12, 12)
    s = null_limit(g, 3, 2)
    assert 9 < s["mean"] < 14 and s["critical"][0.05] > s["mean"]
    lf = local_features(np.random.default_rng(5).standard_normal(144), g)
    assert lf["c.lin|n.lin"].shape == (144,)


if __name__ == "__main__":
    for f in [test_graphs, test_api_and_reproducibility, test_size_under_independence, test_power_sar_and_residuals,
              test_auto_resolution_and_lags, test_limit_and_local]:
        f(); print("ok", f.__name__)


def test_knn_ties_by_index_and_agreement_with_r():
    import os
    tab = __import__("ivspatial").read_table(os.path.join(os.path.dirname(__file__), "..", "examples", "baltimore_houses.csv"))
    g = knn_graph(np.c_[tab["X"], tab["Y"]], 4)
    x = np.log(tab["PRICE"])
    r = np.lexsort((np.arange(x.size), x)).argsort()
    from ivspatial.engine import TVEngine, RankMoran
    assert abs(TVEngine(g, 4, 3).T((4 * r) // g.n) - 1167.4123222749) < 1e-6   # same value in the R package
    assert abs(RankMoran(g).stats(r)[0] - 0.49913761534301637) < 1e-9

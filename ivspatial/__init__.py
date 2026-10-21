"""ivspatial: nonparametric tests of spatial independence on neighbourhood graphs.

    >>> from ivspatial import knn_graph, spatial_test
    >>> g = knn_graph(coords, k=4)
    >>> res = spatial_test(values, g, seed=1)
    >>> print(res)
"""
__version__ = "1.0.0"

from .graphs import (Graph, knn_graph, delaunay_graph, distance_band_graph, lattice_graph, lattice_from_indices, graph_from_edges,
                     graph_from_neighbors, polygon_contiguity_graph, lag_graph)
from .core import (spatial_test, SpatialTestResult, screen, residual_test, local_features, null_limit,
                   ReferenceTable, benjamini_hochberg)
from .io import read_table, read_polygons

__all__ = ["Graph", "knn_graph", "delaunay_graph", "distance_band_graph", "lattice_graph", "lattice_from_indices", "graph_from_edges",
           "graph_from_neighbors", "polygon_contiguity_graph", "lag_graph", "spatial_test", "SpatialTestResult",
           "screen", "residual_test", "local_features", "null_limit", "ReferenceTable", "benjamini_hochberg",
           "read_table", "read_polygons", "__version__"]

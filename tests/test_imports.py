"""Smoke tests that all public package modules can be imported."""


def test_package_imports():
    import graph_dl
    import graph_dl.datasets
    import graph_dl.graph_learn
    import graph_dl.metrics
    import graph_dl.models
    import graph_dl.sparse_coding
    import graph_dl.utils


def test_graph_learn_classes_available():
    from graph_dl.graph_learn import (
        BCD, GAMA, GLAMA, HomogeneousSelfDualEmbedding,
        StationaryADMM, ProjADMM2, LaplacianIPM, LaplacianActiveSet,
    )
    assert BCD is not None
    assert GAMA is not None
    assert GLAMA is not None

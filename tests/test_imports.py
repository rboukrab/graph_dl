"""Smoke tests that all public package modules can be imported."""


def test_package_imports():
    import core
    import core.datasets
    import core.graph_learn
    import core.metrics
    import core.models
    import core.sparse_coding
    import core.utils


def test_graph_learn_classes_available():
    from core.graph_learn import (
        BCD, GAMA, GLAMA, HomogeneousSelfDualEmbedding,
        StationaryADMM, ProjADMM, ProjActiveSet,
    )
    assert BCD is not None
    assert GAMA is not None
    assert GLAMA is not None

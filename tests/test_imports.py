"""Smoke tests that all public package modules can be imported."""


def test_package_imports():
    import joint_ol_gl
    import joint_ol_gl.datasets
    import joint_ol_gl.graph_learn
    import joint_ol_gl.metrics
    import joint_ol_gl.models
    import joint_ol_gl.sparse_coding
    import joint_ol_gl.utils


def test_graph_learn_classes_available():
    from joint_ol_gl.graph_learn import (
        BCD, GAMA, GLAMA, HomogeneousSelfDualEmbedding,
        StationaryADMM, ProjADMM2, LaplacianIPM, LaplacianActiveSet,
    )
    assert BCD is not None
    assert GAMA is not None
    assert GLAMA is not None

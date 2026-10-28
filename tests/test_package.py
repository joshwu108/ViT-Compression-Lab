import importlib


def test_package_imports():
    assert importlib.import_module("vit_compression_lab") is not None

import importlib


def test_package_imports():
    assert importlib.import_module("qwen_vision_compression") is not None

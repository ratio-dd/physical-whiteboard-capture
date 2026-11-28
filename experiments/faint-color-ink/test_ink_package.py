def test_module_is_importable():
    import ink_package
    assert callable(ink_package.build)

def test_v4_threshold_is_optional_for_legacy_cli_compatibility():
    import inspect, ink_package
    assert inspect.signature(ink_package.build).parameters['threshold'].default == 0.7972984313964844

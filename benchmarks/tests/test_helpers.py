"""Quick check that the shared benchmark helpers work."""


def test_machine_specs_keys():
    from nanomlx_bench import machine_specs

    specs = machine_specs()
    assert specs["chip"]
    assert specs["python"]

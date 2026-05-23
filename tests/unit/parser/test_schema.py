import pytest

from mrm_trace.parser.schema import (
    OpType,
    Phase,
    RegionType,
    RetentionClass,
    RunValidity,
)


def test_region_type_values():
    assert RegionType.WEIGHT.value == "weight"
    assert RegionType.KV_CACHE.value == "kv_cache"
    assert RegionType.ACTIVATION.value == "activation"
    assert RegionType.RUNTIME.value == "runtime"
    assert RegionType.UNKNOWN.value == "unknown"


def test_op_type_values():
    assert OpType.LOAD.value == "load"
    assert OpType.STORE.value == "store"


def test_phase_values():
    assert Phase.WEIGHT_LOAD.value == "weight_load"
    assert Phase.GENERATION.value == "generation"
    assert Phase.TEARDOWN.value == "teardown"


def test_run_validity_values():
    assert RunValidity.CLEAN.value == "clean"
    assert RunValidity.MARGINAL.value == "marginal"
    assert RunValidity.CONTAMINATED.value == "contaminated"


def test_retention_class_values():
    assert RetentionClass.SECONDS.value == "seconds"
    assert RetentionClass.MINUTES.value == "minutes"
    assert RetentionClass.HOURS.value == "hours"
    assert RetentionClass.SESSION.value == "session"


def test_region_type_is_string_enum():
    assert isinstance(RegionType.WEIGHT, str)
    assert RegionType.WEIGHT == "weight"


def test_op_type_is_string_enum():
    assert isinstance(OpType.LOAD, str)
    assert OpType.LOAD == "load"


def test_phase_is_string_enum():
    assert isinstance(Phase.GENERATION, str)
    assert Phase.GENERATION == "generation"


def test_region_type_from_string():
    rt = RegionType("kv_cache")
    assert rt == RegionType.KV_CACHE


def test_invalid_region_type_raises():
    with pytest.raises(ValueError):
        RegionType("not_a_region")


def test_invalid_op_type_raises():
    with pytest.raises(ValueError):
        OpType("read")  # should be "load"


def test_all_region_types_count():
    assert len(RegionType) == 5


def test_all_phases_count():
    assert len(Phase) == 3


def test_all_op_types_count():
    assert len(OpType) == 2

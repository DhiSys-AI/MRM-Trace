"""
Tier 3 — provenance of GPU-path runs.

Two things a GPU run must state rather than leave to inference:

1. Why weight/kv_cache attribution is absent.  A CUDA ``data_ptr()`` is a
   device address, so an empty pytorch_regions.json is the correct result for a
   GPU-resident model — but identical, in the output, to the extractor having
   silently failed.  parameter_devices.json makes the difference checkable.
2. Whether the recorded GPU power draw is a measurement.  nvidia-smi on mobile
   Ampere reported 752.67 W against a 115 W enforced limit; that number must
   not reach metadata.json as machine context.
"""
import json

import pytest

from mrm_trace.reporter.run_audit import Audit
from mrm_trace.telemetry.system_monitor import gpu_section

pytestmark = pytest.mark.validity


# --------------------------------------------------------------------------- #
# Power draw gating
# --------------------------------------------------------------------------- #

class _Stats:
    name = "NVIDIA GeForce RTX 3080 Laptop GPU"
    vram_total_mb = 16384
    vram_start_used_mb = 15
    vram_peak_used_mb = 1191
    vram_end_used_mb = 15
    gpu_util_pct_mean = 15.14
    n_samples = 10


def test_implausible_draw_is_not_published_as_a_measurement():
    """The observed mobile-Ampere artifact must not appear as power_draw_w."""
    ctx = {
        "gpu_power_draw_w": None,          # validate_machine rejected it
        "gpu_power_limit_w": 115.0,
        "gpu_power_draw_status":
            "implausible: driver reported 752.67 W against a 115.00 W enforced limit",
    }
    section = gpu_section(_Stats(), ctx)
    assert "power_draw_w" not in section
    assert section["power_limit_w"] == 115.0
    assert "752.67" in section["power_draw_status"]


def test_plausible_draw_is_published_without_a_status():
    ctx = {"gpu_power_draw_w": 98.4, "gpu_power_limit_w": 115.0,
           "gpu_power_draw_status": "ok"}
    section = gpu_section(_Stats(), ctx)
    assert section["power_draw_w"] == 98.4
    assert section["power_limit_w"] == 115.0
    assert "power_draw_status" not in section, "ok status is noise, not provenance"


def test_gpu_section_is_empty_without_a_gpu():
    assert gpu_section(None, None) == {}


def test_vram_fields_survive_the_shared_helper():
    """Both demo scripts depend on these keys; the helper must not rename them."""
    section = gpu_section(_Stats(), None)
    for key in ("name", "vram_total_mb", "vram_start_used_mb",
                "vram_peak_used_mb", "vram_end_used_mb", "gpu_util_pct_mean"):
        assert key in section, key
    assert section["gpu_util_pct_mean"] == 15.1, "must stay rounded to 1 dp"


# --------------------------------------------------------------------------- #
# Absence reasons
# --------------------------------------------------------------------------- #

def _audit_with_census(tmp_path, census):
    run = tmp_path / "run"
    run.mkdir()
    if census is not None:
        (run / "parameter_devices.json").write_text(json.dumps(census))
    return Audit(run)


def test_gpu_resident_absence_is_explained_from_evidence(tmp_path):
    audit = _audit_with_census(tmp_path, {"cuda:0": {"tensors": 290,
                                                     "bytes": 988_123_136}})
    reason = audit._absence_reason("weight")
    assert reason.startswith("expected:")
    assert "290 tensors" in reason and "cuda:0" in reason
    assert not reason.startswith("UNEXPECTED")


def test_host_resident_absence_is_a_failure_not_a_note(tmp_path):
    """Weights in host memory with no attribution is a pipeline bug."""
    audit = _audit_with_census(tmp_path, {"cpu": {"tensors": 290,
                                                  "bytes": 988_123_136}})
    assert audit._absence_reason("weight").startswith("UNEXPECTED")


def test_split_placement_is_partially_expected(tmp_path):
    audit = _audit_with_census(tmp_path, {
        "cpu": {"tensors": 10, "bytes": 1_000_000},
        "cuda:0": {"tensors": 280, "bytes": 980_000_000},
    })
    reason = audit._absence_reason("weight")
    assert reason.startswith("partially expected:")


def test_missing_census_degrades_to_the_old_hedge(tmp_path):
    """Pre-census runs must still audit, just without the justification."""
    audit = _audit_with_census(tmp_path, None)
    reason = audit._absence_reason("weight")
    assert "no device census recorded" in reason
    assert not reason.startswith("UNEXPECTED"), \
        "an old run must not be failed for lacking a field it never wrote"


def test_empty_census_is_not_treated_as_host_resident(tmp_path):
    audit = _audit_with_census(tmp_path, {})
    assert not audit._absence_reason("weight").startswith("UNEXPECTED")

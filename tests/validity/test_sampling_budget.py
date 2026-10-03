"""
Sample count must track the sampling budget, not the run's wall time.

perf mem's -c is a period — one sample per N memory accesses — so sample count
scales with how long a run takes, not with model size.  A fixed per-model
period therefore blows up on long runs: measured at 368k samples/s (-c 32000),
the period that gives a usable 1B run yields ~2 billion samples on an 8B run,
which no amount of streaming analyses on a 32 GB machine.

These tests pin the budget, the floor, and the two flags that carry the
acquisition parameters into the run and into its provenance.
"""

import importlib.util
import pathlib

import pytest

pytestmark = pytest.mark.validity

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RUNNER = _ROOT / "notebooks" / "scripts" / "experiment_runner.py"
_DEMO = _ROOT / "notebooks" / "scripts" / "demo_perf_mem.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load(_RUNNER, "experiment_runner")


@pytest.fixture(scope="module")
def demo():
    return _load(_DEMO, "demo_perf_mem")


# ── the budget ───────────────────────────────────────────────────────────────

def test_every_model_lands_inside_the_sample_budget(runner):
    """
    The whole point: a 40-minute 8B run must produce roughly as many samples as
    a 5-minute 1B run, so both are analysable on the same machine.
    """
    for model_id, _, _, tok_per_s in runner.CPU_MODELS:
        for ctx in runner.CONTEXT_LENGTHS:
            period = runner.sample_period_for(tok_per_s, ctx)
            est_s = runner.estimate_run_seconds(tok_per_s, ctx)
            samples = runner._ACCESS_RATE_PER_S * est_s / period
            assert samples <= runner.SAMPLE_BUDGET * 1.05, (
                f"{model_id} ctx{ctx} projects {samples / 1e6:.0f}M, over budget"
            )


def test_a_longer_run_gets_a_coarser_period(runner):
    """Period must rise with run length; a constant would defeat the budget."""
    fast = runner.sample_period_for(11.0)
    slow = runner.sample_period_for(1.5)
    assert slow > fast
    # 8B is ~7x slower than 1B, so its period must be ~7x coarser.
    assert slow / fast == pytest.approx(11.0 / 1.5, rel=0.1)


def test_more_output_tokens_coarsens_the_period(runner):
    assert (runner.sample_period_for(11.0, output_tokens=1024)
            > runner.sample_period_for(11.0, output_tokens=512))


def test_a_longer_context_coarsens_the_period(runner):
    """Decode slows as context grows — measured 11.2 tok/s at ctx 512 vs 8.38
    at 2048 — so the same model needs a coarser period at a longer context."""
    at_2k = runner.sample_period_for(8.38, 2048)
    at_8k = runner.sample_period_for(8.38, 8192)
    at_32k = runner.sample_period_for(8.38, 32768)
    assert at_2k < at_8k < at_32k


def test_a_very_fast_run_is_floored_not_made_absurdly_dense(runner):
    assert runner.sample_period_for(10_000.0) == runner.MIN_SAMPLE_PERIOD


def test_the_budget_fits_the_measured_memory_ceiling(runner):
    """
    30M rows was measured end to end at 15 GB peak RSS.  If the budget is
    raised without re-measuring, the analysis stage OOMs again.
    """
    assert runner.SAMPLE_BUDGET <= 40_000_000


# ── the flags that carry it into the run ─────────────────────────────────────

def test_the_run_command_passes_both_acquisition_parameters(runner):
    """
    A run labelled at the 4 KB default contradicts workplan §5.1.2, and a run
    at the demo's default period contradicts the budget above.  Both have to be
    on the command line.
    """
    spec = next(s for s in runner.build_matrix(cpu_only=True) if s.path_type == "cpu")
    cmd = runner._build_command(spec, pathlib.Path("models"), "m")
    assert "--sample-period" in cmd
    assert cmd[cmd.index("--sample-period") + 1] == str(spec.sample_period)
    assert "--region-granularity" in cmd
    assert cmd[cmd.index("--region-granularity") + 1] == str(spec.region_granularity)


def test_granularity_override_reaches_the_spec(runner):
    specs = runner.build_matrix(cpu_only=True, region_granularity=1_048_576)
    assert all(s.region_granularity == 1_048_576
               for s in specs if s.path_type == "cpu")


def test_demo_accepts_the_flags_the_runner_sends(runner, demo):
    """Guards against the runner and the demo drifting apart."""
    spec = next(s for s in runner.build_matrix(cpu_only=True) if s.path_type == "cpu")
    cmd = runner._build_command(spec, pathlib.Path("models"), "m")
    # Parse the runner's own argv through the demo's parser.
    import sys
    saved = sys.argv
    try:
        sys.argv = ["demo_perf_mem.py"] + cmd[2:]
        parsed = demo.parse_args()
    finally:
        sys.argv = saved
    assert parsed.sample_period == spec.sample_period
    assert parsed.region_granularity == spec.region_granularity


# ── provenance ───────────────────────────────────────────────────────────────

def test_metadata_records_the_acquisition_parameters(demo):
    """
    A sample count is uninterpretable without the period it was taken at, and
    a retention figure is uninterpretable without the granule.  The data card
    (workplan §12.2) has to be able to state both from metadata.json alone.
    """
    from mrm_trace.collector.base import CollectorMeta
    from mrm_trace.config.schema import EngineType, ModelConfig

    meta = demo.build_metadata(
        ModelConfig(id="m", path="models/x.gguf", engine=EngineType.LLAMACPP,
                    family="llama", params_billions=1.0, quantisation="Q4_K_M"),
        "run-1",
        {"engine_name": "llamacpp", "engine_version": "0.3.34"},
        [],
        CollectorMeta(
            collector_type="perf_mem", available=True, used=True,
            event_strings=["cpu/mem-loads,ldlat=30/P"], sample_period=128_000,
            output_file="raw/perf.data", output_file_size_bytes=1,
            parse_success=True, warnings=[],
        ),
        region_granularity=65_536,
    )
    collector = meta["collector"]
    assert collector["sample_period"] == 128_000
    assert collector["event_strings"] == ["cpu/mem-loads,ldlat=30/P"]
    assert collector["region_granularity_bytes"] == 65_536


# ── disk reclaim ─────────────────────────────────────────────────────────────

def test_perf_script_is_reclaimed_but_perf_data_is_kept(demo, tmp_path, capsys):
    """
    perf_script.txt is intermediate state — regenerable from perf.data, and the
    largest artifact of a run.  perf.data is the primary raw artifact, cited by
    metadata.json and bound for Zenodo, so it must survive.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "perf_script.txt").write_text("x" * 1024)
    (raw / "perf.data").write_bytes(b"\0" * 512)

    demo._reclaim_perf_script(raw)

    assert not (raw / "perf_script.txt").exists()
    assert (raw / "perf.data").exists(), "perf.data must never be deleted"
    assert "Reclaimed" in capsys.readouterr().out


def test_keep_flag_preserves_perf_script(demo, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "perf_script.txt").write_text("x")
    demo._reclaim_perf_script(raw, keep=True)
    assert (raw / "perf_script.txt").exists()


def test_reclaim_is_a_noop_when_there_is_nothing_to_remove(demo, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    demo._reclaim_perf_script(raw)   # must not raise


# ── graceful shutdown ────────────────────────────────────────────────────────

def test_the_child_run_is_isolated_from_terminal_sigint(runner):
    """
    _handle_sigint promises to finish the current run before exiting, but a
    child in the terminal's foreground process group receives the same Ctrl+C
    and dies with exit -2 — recorded as `failed`, losing a 30-minute run.
    start_new_session keeps the SIGINT with the runner.
    """
    import inspect
    src = inspect.getsource(runner.run_one)
    assert "start_new_session=True" in src


def test_reclassify_rewrites_verdicts_and_reasons(runner, tmp_path, monkeypatch):
    """
    Validity is a label over numbers the run already recorded, so correcting
    the rule must never require re-collecting a trace.
    """
    import json
    run_dir = tmp_path / "machine" / "model" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({
        "run_id": "run-1",
        "thermal": {"peak_temp_c": 73.0, "throttle_events": 0,
                    "run_validity": "contaminated"},
        "observer_effect": {
            "observer_cpu_overhead_pct_mean": 4.0,
            "observer_memory_overhead_mb_peak": 2512.0,
            "observer_memory_overhead_frac": 1.70,
            "baseline_cpu_pct": 0.06, "io_contention_detected": False,
        },
        "hardware": {"mem_total_mb": 31742.0},
    }))
    state = {"version": 1, "runs": {"k": {"run_id": "run-1",
                                          "run_validity": "contaminated",
                                          "contaminated": True}}}
    monkeypatch.setattr(runner, "_save_state", lambda s: None)

    counts = runner.reclassify_runs(tmp_path, state, lambda *_: None)

    assert counts == {"clean": 1}
    thermal = json.loads((run_dir / "metadata.json").read_text())["thermal"]
    assert thermal["run_validity"] == "clean"
    assert thermal["validity_failure_reasons"] == []
    # --status must agree with the metadata afterwards.
    assert state["runs"]["k"]["run_validity"] == "clean"
    assert state["runs"]["k"]["contaminated"] is False


def test_reclassify_dry_run_changes_nothing(runner, tmp_path, monkeypatch):
    import json
    run_dir = tmp_path / "m" / "mo" / "r"
    run_dir.mkdir(parents=True)
    meta = {
        "run_id": "r",
        "thermal": {"peak_temp_c": 73.0, "throttle_events": 0,
                    "run_validity": "contaminated"},
        "observer_effect": {"observer_memory_overhead_mb_peak": 2512.0},
        "hardware": {"mem_total_mb": 31742.0},
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta))
    monkeypatch.setattr(runner, "_save_state", lambda s: None)

    runner.reclassify_runs(tmp_path, {"version": 1, "runs": {}},
                           lambda *_: None, dry_run=True)

    after = json.loads((run_dir / "metadata.json").read_text())
    assert after["thermal"]["run_validity"] == "contaminated"


def test_reclassify_is_idempotent(runner, tmp_path, monkeypatch):
    import json
    run_dir = tmp_path / "m" / "mo" / "r"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({
        "run_id": "r",
        "thermal": {"peak_temp_c": 73.0, "throttle_events": 0},
        "observer_effect": {"observer_memory_overhead_mb_peak": 2512.0,
                            "io_contention_detected": False},
        "hardware": {"mem_total_mb": 31742.0},
    }))
    monkeypatch.setattr(runner, "_save_state", lambda s: None)
    log = []
    runner.reclassify_runs(tmp_path, {"version": 1, "runs": {}}, log.append)
    first = (run_dir / "metadata.json").read_text()
    runner.reclassify_runs(tmp_path, {"version": 1, "runs": {}}, log.append)
    assert (run_dir / "metadata.json").read_text() == first


def test_state_sync_matches_on_output_dir_when_run_id_is_missing(runner, tmp_path,
                                                                 monkeypatch):
    """
    run_one() scrapes run_id from the child's stdout and in practice always
    gets None — every real state entry identifies its run only by output_dir.
    The sync must fall back to that, or --status keeps reporting stale verdicts
    after a reclassification.
    """
    import json
    run_dir = tmp_path / "machine" / "model" / "run-xyz"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({
        "run_id": "run-xyz",
        "thermal": {"peak_temp_c": 73.0, "throttle_events": 0,
                    "run_validity": "contaminated"},
        "observer_effect": {"observer_memory_overhead_mb_peak": 2512.0,
                            "io_contention_detected": False},
        "hardware": {"mem_total_mb": 31742.0},
    }))
    state = {"version": 1, "runs": {"some__key__cpu": {
        "status": "done",
        "run_id": None,                                   # as written in practice
        "output_dir": str(run_dir),
        "run_validity": "contaminated",
        "contaminated": True,
    }}}
    monkeypatch.setattr(runner, "_save_state", lambda s: None)

    runner.reclassify_runs(tmp_path, state, lambda *_: None)

    entry = state["runs"]["some__key__cpu"]
    assert entry["run_validity"] == "clean"
    assert entry["contaminated"] is False


def test_a_run_without_metadata_is_left_alone(runner, tmp_path, monkeypatch):
    """An interrupted run has no metadata.json; reclassify must not invent one."""
    import json
    good = tmp_path / "m" / "mo" / "good"
    good.mkdir(parents=True)
    (good / "metadata.json").write_text(json.dumps({
        "run_id": "good",
        "thermal": {"peak_temp_c": 73.0, "throttle_events": 0},
        "observer_effect": {"observer_memory_overhead_mb_peak": 100.0},
        "hardware": {"mem_total_mb": 31742.0},
    }))
    (tmp_path / "m" / "mo" / "killed").mkdir(parents=True)   # raw/ only, no metadata
    state = {"version": 1, "runs": {"k": {
        "status": "failed", "run_id": None,
        "output_dir": str(tmp_path / "m" / "mo" / "killed"),
        "run_validity": "unknown", "contaminated": False,
    }}}
    monkeypatch.setattr(runner, "_save_state", lambda s: None)

    counts = runner.reclassify_runs(tmp_path, state, lambda *_: None)

    assert counts == {"clean": 1}                     # only the exported run
    assert state["runs"]["k"]["run_validity"] == "unknown"


# ── calibration: measure, don't guess ────────────────────────────────────────

def test_measurement_overrides_the_estimate(runner):
    """
    The estimate put the first 8 runs 35% over budget: 40.5M samples against a
    30M target, because it assumed 11.0 tok/s where the real figure was 8.38.
    A measured run must correct the next one proportionally, with no duration
    model involved.
    """
    key = runner._calibration_key("m", 2048, runner.OUTPUT_TOKENS,
                                  runner.PROMPTS_PER_RUN)
    cal = {key: {"period": 128_000, "samples": 40_500_000}}
    corrected = runner.sample_period_for(11.0, 2048, model_id="m", calibration=cal)
    # 128000 * 40.5/30 = 172,800, rounded up to 3 significant figures.
    assert corrected == 173_000
    projected = 40_500_000 * 128_000 / corrected
    assert abs(projected - runner.SAMPLE_BUDGET) / runner.SAMPLE_BUDGET < 0.01


def test_calibration_is_keyed_by_model_and_context(runner):
    """A measurement at one context must not be applied to another."""
    key = runner._calibration_key("m", 2048, runner.OUTPUT_TOKENS,
                                  runner.PROMPTS_PER_RUN)
    cal = {key: {"period": 128_000, "samples": 40_500_000}}
    assert (runner.sample_period_for(8.38, 8192, model_id="m", calibration=cal)
            == runner.sample_period_for(8.38, 8192))


def test_estimate_is_used_when_nothing_has_been_measured(runner):
    assert (runner.sample_period_for(8.38, 2048, model_id="unseen", calibration={})
            == runner.sample_period_for(8.38, 2048))


def test_a_nonsense_calibration_entry_is_ignored(runner):
    key = runner._calibration_key("m", 2048, runner.OUTPUT_TOKENS,
                                  runner.PROMPTS_PER_RUN)
    for bad in ({"samples": 0, "period": 1000}, {"samples": 10, "period": 0}, {}):
        cal = {key: bad}
        assert (runner.sample_period_for(8.38, 2048, model_id="m", calibration=cal)
                == runner.sample_period_for(8.38, 2048))


def test_record_calibration_round_trips(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CALIBRATION_FILE", tmp_path / ".sample_calibration.json")
    runner.record_calibration("m", 2048, 128_000, 40_500_000)
    key = runner._calibration_key("m", 2048, runner.OUTPUT_TOKENS,
                                  runner.PROMPTS_PER_RUN)
    assert runner._load_calibration()[key]["samples"] == 40_500_000
    assert runner.sample_period_for(11.0, 2048, model_id="m") == 173_000


def test_record_calibration_ignores_empty_measurements(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CALIBRATION_FILE", tmp_path / "cal.json")
    runner.record_calibration("m", 2048, 128_000, 0)
    assert runner._load_calibration() == {}


# ── timeout ──────────────────────────────────────────────────────────────────

def test_no_run_in_the_matrix_is_killed_by_its_own_timeout(runner):
    """
    A flat 2 h timeout hard-killed 27 of the 135 CPU runs — every ctx-32768 run
    of phi, mistral and llama-3.1-8b, whose estimates reach 5.2 h — for a
    reason unrelated to the science.
    """
    for spec in runner.build_matrix(cpu_only=True):
        if spec.path_type != "cpu":
            continue
        tok_per_s = next(t for m, _, _, t in runner.CPU_MODELS if m == spec.model_id)
        est = runner.estimate_run_seconds(tok_per_s, spec.ctx)
        assert spec.max_runtime > est, (
            f"{spec.model_id} ctx{spec.ctx}: {spec.max_runtime}s timeout "
            f"vs {est:.0f}s estimate"
        )


def test_timeout_still_catches_a_hang(runner):
    """Generous, not absent: a short run keeps the 2 h floor, nothing exceeds 12 h."""
    assert runner.run_timeout_for(60) == runner.MAX_RUN_SECONDS
    assert runner.run_timeout_for(10 ** 7) == runner.MAX_RUN_SECONDS_CEILING
    assert runner.run_timeout_for(3600) == 3600 * runner.RUN_TIMEOUT_FACTOR


def test_calibration_is_invalidated_by_a_prompt_count_change(runner):
    """
    Dropping 7 prompts to 4 halves run length.  Reusing a 7-prompt measurement
    would coarsen the period by 7/4 and deliver half the sample budget, so the
    key has to include the prompt count.
    """
    stale = {runner._calibration_key("m", 2048, runner.OUTPUT_TOKENS, 7):
             {"period": 128_000, "samples": 40_500_000}}
    assert runner.PROMPTS_PER_RUN != 7
    assert (runner.sample_period_for(8.38, 2048, model_id="m", calibration=stale)
            == runner.sample_period_for(8.38, 2048))


def test_calibration_is_invalidated_by_a_token_count_change(runner):
    stale = {runner._calibration_key("m", 2048, 256, runner.PROMPTS_PER_RUN):
             {"period": 128_000, "samples": 40_500_000}}
    assert (runner.sample_period_for(8.38, 2048, output_tokens=512,
                                     model_id="m", calibration=stale)
            == runner.sample_period_for(8.38, 2048, output_tokens=512))


def test_the_run_command_fixes_the_prompt_count(runner):
    """Run length is linear in prompts, so it cannot be left to the script default."""
    import pathlib as _p
    spec = next(s for s in runner.build_matrix(cpu_only=True) if s.path_type == "cpu")
    cmd = runner._build_command(spec, _p.Path("models"), "m")
    assert "--num-prompts" in cmd
    assert cmd[cmd.index("--num-prompts") + 1] == str(runner.PROMPTS_PER_RUN)

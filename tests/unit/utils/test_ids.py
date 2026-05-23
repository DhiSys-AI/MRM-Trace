from mrm_trace.utils.ids import generate_run_id

TS = "20260101T000000Z"


def test_run_id_contains_model_id():
    rid = generate_run_id("llama3-q4", 2048, 4, 128, 0, ts=TS)
    assert "llama3-q4" in rid


def test_run_id_contains_context_length():
    rid = generate_run_id("model", 2048, 4, 128, 0, ts=TS)
    assert "ctx2048" in rid


def test_run_id_contains_batch_size():
    rid = generate_run_id("model", 512, 4, 128, 0, ts=TS)
    assert "batch4" in rid


def test_run_id_contains_output_tokens():
    rid = generate_run_id("model", 512, 1, 256, 0, ts=TS)
    assert "out256" in rid


def test_run_id_contains_rep_index():
    rid = generate_run_id("model", 512, 1, 128, 2, ts=TS)
    assert "rep2" in rid


def test_warmup_run_id_contains_warmup():
    rid = generate_run_id("model", 512, 1, 128, 0, warmup=True, ts=TS)
    assert "warmup" in rid
    assert "rep" not in rid


def test_warmup_false_has_rep_prefix():
    rid = generate_run_id("model", 512, 1, 128, 0, warmup=False, ts=TS)
    assert "rep0" in rid
    assert "warmup" not in rid


def test_run_ids_differ_by_rep():
    rid0 = generate_run_id("model", 512, 1, 128, 0, ts=TS)
    rid1 = generate_run_id("model", 512, 1, 128, 1, ts=TS)
    assert rid0 != rid1


def test_run_ids_differ_by_context():
    rid512 = generate_run_id("model", 512, 1, 128, 0, ts=TS)
    rid2048 = generate_run_id("model", 2048, 1, 128, 0, ts=TS)
    assert rid512 != rid2048


def test_run_id_contains_timestamp():
    rid = generate_run_id("model", 512, 1, 128, 0, ts=TS)
    assert TS in rid


def test_run_id_auto_timestamp_when_none():
    rid = generate_run_id("model", 512, 1, 128, 0)
    assert isinstance(rid, str)
    assert len(rid) > 0
    assert "ctx512" in rid


def test_run_id_is_string():
    rid = generate_run_id("model", 512, 1, 128, 0, ts=TS)
    assert isinstance(rid, str)

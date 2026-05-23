"""Unit tests for the memray parser."""

from unittest.mock import MagicMock, patch

import pytest

from mrm_trace.parser.memray_parser import parse_memray


def _make_record(address=0x7F00_0000_0000, allocator="MALLOC", thread_id=1, ts=0):
    rec = MagicMock()
    rec.address = address
    rec.allocator = allocator
    rec.thread_id = thread_id
    rec.timestamp = ts
    frame = MagicMock()
    frame.function_name = "test_func"
    frame.filename = "/test/module.py"
    rec.stack_trace.return_value = [frame]
    return rec


def _mock_memray_with(records):
    mock_memray = MagicMock()
    mock_reader = MagicMock()
    mock_reader.get_allocation_records.return_value = records
    mock_memray.FileReader.return_value = mock_reader
    return mock_memray


# ── availability ──────────────────────────────────────────────────────────────

def test_raises_importerror_when_memray_unavailable(tmp_path):
    f = tmp_path / "trace.bin"
    f.write_bytes(b"fake")
    with patch("mrm_trace.parser.memray_parser._try_import_memray", return_value=None):
        with pytest.raises(ImportError):
            list(parse_memray(f, run_id="r1"))


# ── basic parsing ─────────────────────────────────────────────────────────────

def test_yields_one_row_per_record(tmp_path):
    f = tmp_path / "trace.bin"
    f.write_bytes(b"fake")
    mock = _mock_memray_with([_make_record(), _make_record()])
    with patch("mrm_trace.parser.memray_parser._try_import_memray", return_value=mock):
        rows = list(parse_memray(f, run_id="r1"))
    assert len(rows) == 2


def test_empty_trace_yields_nothing(tmp_path):
    f = tmp_path / "trace.bin"
    f.write_bytes(b"fake")
    mock = _mock_memray_with([])
    with patch("mrm_trace.parser.memray_parser._try_import_memray", return_value=mock):
        rows = list(parse_memray(f, run_id="r1"))
    assert rows == []


# ── op_type mapping ───────────────────────────────────────────────────────────

def test_malloc_maps_to_store(tmp_path):
    f = tmp_path / "trace.bin"
    f.write_bytes(b"fake")
    mock = _mock_memray_with([_make_record(allocator="MALLOC")])
    with patch("mrm_trace.parser.memray_parser._try_import_memray", return_value=mock):
        rows = list(parse_memray(f, run_id="r1"))
    assert rows[0]["op_type"] == "store"


def test_free_maps_to_load(tmp_path):
    f = tmp_path / "trace.bin"
    f.write_bytes(b"fake")
    mock = _mock_memray_with([_make_record(allocator="FREE")])
    with patch("mrm_trace.parser.memray_parser._try_import_memray", return_value=mock):
        rows = list(parse_memray(f, run_id="r1"))
    assert rows[0]["op_type"] == "load"


def test_delete_maps_to_load(tmp_path):
    f = tmp_path / "trace.bin"
    f.write_bytes(b"fake")
    mock = _mock_memray_with([_make_record(allocator="DELETE")])
    with patch("mrm_trace.parser.memray_parser._try_import_memray", return_value=mock):
        rows = list(parse_memray(f, run_id="r1"))
    assert rows[0]["op_type"] == "load"


# ── address and page ──────────────────────────────────────────────────────────

def test_address_preserved(tmp_path):
    addr = 0x7F00_0123_4567
    f = tmp_path / "trace.bin"
    f.write_bytes(b"fake")
    mock = _mock_memray_with([_make_record(address=addr)])
    with patch("mrm_trace.parser.memray_parser._try_import_memray", return_value=mock):
        rows = list(parse_memray(f, run_id="r1"))
    assert rows[0]["address"] == addr


def test_address_page_computed(tmp_path):
    addr = 0x7F00_0123_4567
    f = tmp_path / "trace.bin"
    f.write_bytes(b"fake")
    mock = _mock_memray_with([_make_record(address=addr)])
    with patch("mrm_trace.parser.memray_parser._try_import_memray", return_value=mock):
        rows = list(parse_memray(f, run_id="r1"))
    assert rows[0]["address_page"] == addr >> 12


# ── schema ────────────────────────────────────────────────────────────────────

def test_all_canonical_columns_present(tmp_path):
    f = tmp_path / "trace.bin"
    f.write_bytes(b"fake")
    mock = _mock_memray_with([_make_record()])
    with patch("mrm_trace.parser.memray_parser._try_import_memray", return_value=mock):
        rows = list(parse_memray(f, run_id="r1"))
    required = {
        "run_id", "sample_id", "timestamp_ns", "phase", "address", "address_page",
        "op_type", "symbol", "dso", "region_type", "region_id", "access_latency_ns",
        "cache_level", "numa_node", "cpu_id", "thread_id", "token_index", "context_tokens",
    }
    assert required.issubset(rows[0].keys())


def test_sample_id_sequential(tmp_path):
    f = tmp_path / "trace.bin"
    f.write_bytes(b"fake")
    mock = _mock_memray_with([_make_record(), _make_record(), _make_record()])
    with patch("mrm_trace.parser.memray_parser._try_import_memray", return_value=mock):
        rows = list(parse_memray(f, run_id="r1"))
    assert [r["sample_id"] for r in rows] == [0, 1, 2]


def test_run_id_in_every_row(tmp_path):
    f = tmp_path / "trace.bin"
    f.write_bytes(b"fake")
    mock = _mock_memray_with([_make_record(), _make_record()])
    with patch("mrm_trace.parser.memray_parser._try_import_memray", return_value=mock):
        rows = list(parse_memray(f, run_id="myrun"))
    assert all(r["run_id"] == "myrun" for r in rows)

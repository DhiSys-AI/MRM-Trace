from pathlib import Path

from mrm_trace.utils.files import ensure_dir, get_run_dir, sha256_file

HELLO_SHA256 = "b94d27b9934d3e08a52e52d7da7dabfac484efe04294e576ca576d5b6b0aa4e4"


def test_sha256_file_returns_hex_string(tmp_path):
    f = tmp_path / "test.txt"
    f.write_bytes(b"hello world")
    digest = sha256_file(f)
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_sha256_file_correct_value(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"hello world")
    import hashlib
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert sha256_file(f) == expected


def test_sha256_same_content_same_hash(tmp_path):
    content = b"identical content"
    f1 = tmp_path / "a.bin"
    f2 = tmp_path / "b.bin"
    f1.write_bytes(content)
    f2.write_bytes(content)
    assert sha256_file(f1) == sha256_file(f2)


def test_sha256_different_content_different_hash(tmp_path):
    f1 = tmp_path / "a.bin"
    f2 = tmp_path / "b.bin"
    f1.write_bytes(b"content A")
    f2.write_bytes(b"content B")
    assert sha256_file(f1) != sha256_file(f2)


def test_sha256_empty_file(tmp_path):
    f = tmp_path / "empty.bin"
    f.write_bytes(b"")
    import hashlib
    expected = hashlib.sha256(b"").hexdigest()
    assert sha256_file(f) == expected


def test_sha256_large_file(tmp_path):
    content = b"x" * (3 * 1024 * 1024)  # 3 MB, crosses chunk boundary
    f = tmp_path / "large.bin"
    f.write_bytes(content)
    import hashlib
    expected = hashlib.sha256(content).hexdigest()
    assert sha256_file(f) == expected


def test_ensure_dir_creates_directory(tmp_path):
    new_dir = tmp_path / "new"
    assert not new_dir.exists()
    ensure_dir(new_dir)
    assert new_dir.is_dir()


def test_ensure_dir_creates_nested(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    ensure_dir(nested)
    assert nested.is_dir()


def test_ensure_dir_idempotent(tmp_path):
    d = tmp_path / "dir"
    ensure_dir(d)
    ensure_dir(d)  # must not raise
    assert d.is_dir()


def test_get_run_dir_structure():
    base = Path("/results")
    run_dir = get_run_dir(base, "llama3-q4", "run_001")
    assert run_dir == Path("/results/llama3-q4/run_001")


def test_get_run_dir_uses_model_id():
    base = Path("/out")
    run_dir = get_run_dir(base, "mistral-7b", "run_abc")
    assert "mistral-7b" in str(run_dir)


def test_get_run_dir_uses_run_id():
    base = Path("/out")
    run_dir = get_run_dir(base, "model", "myrun123")
    assert "myrun123" in str(run_dir)

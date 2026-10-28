import pytest

from vit_compression_lab.run_checkpoint import (
    ResumeError,
    append_record,
    check_resume,
    load_records,
    write_json,
)

FINGERPRINT = {
    "revision": "abc",
    "prompt": {"text": "q"},
    "manifests": [{"sha256": "1"}],
}


def start_run(tmp_path, completed_ids):
    write_json(tmp_path / "run_config.json", FINGERPRINT)
    for example_id in completed_ids:
        append_record(tmp_path / "results.jsonl", {"id": example_id, "answer": "x"})


def test_resume_returns_completed_ids_without_duplicates(tmp_path):
    start_run(tmp_path, ["a", "b", "c"])
    completed, repaired = check_resume(tmp_path, dict(FINGERPRINT))
    assert completed == {"a", "b", "c"} and repaired is False


def test_resume_repairs_partial_last_line_from_interrupted_write(tmp_path):
    start_run(tmp_path, ["a", "b"])
    with open(tmp_path / "results.jsonl", "a") as f:
        f.write('{"id": "c", "answer": "tru')
    completed, repaired = check_resume(tmp_path, dict(FINGERPRINT))
    assert completed == {"a", "b"} and repaired is True
    records, repaired_again = load_records(tmp_path / "results.jsonl")
    assert [r["id"] for r in records] == ["a", "b"] and repaired_again is False
    append_record(tmp_path / "results.jsonl", {"id": "c"})
    assert [r["id"] for r in load_records(tmp_path / "results.jsonl")[0]] == [
        "a",
        "b",
        "c",
    ]


@pytest.mark.parametrize(
    "changed",
    [
        {"revision": "other"},
        {"prompt": {"text": "different"}},
        {"manifests": [{"sha256": "2"}]},
        {"generation_config": {"do_sample": True}},
    ],
)
def test_resume_rejects_changed_settings(tmp_path, changed):
    start_run(tmp_path, ["a"])
    with pytest.raises(ResumeError, match=next(iter(changed))):
        check_resume(tmp_path, {**FINGERPRINT, **changed})


def test_resume_rejects_non_run_directory_and_duplicates(tmp_path):
    with pytest.raises(ResumeError, match="run_config.json"):
        check_resume(tmp_path, FINGERPRINT)
    start_run(tmp_path, ["a", "a"])
    with pytest.raises(ResumeError, match="duplicate"):
        check_resume(tmp_path, FINGERPRINT)


def test_load_records_on_missing_file_is_empty(tmp_path):
    assert load_records(tmp_path / "none.jsonl") == ([], False)


def test_complete_last_line_without_newline_is_kept_and_terminated(tmp_path):
    start_run(tmp_path, ["a"])
    with open(tmp_path / "results.jsonl", "a") as f:
        f.write('{"id": "b"}')  # complete record, newline lost
    completed, repaired = check_resume(tmp_path, dict(FINGERPRINT))
    assert completed == {"a", "b"} and repaired is False
    append_record(tmp_path / "results.jsonl", {"id": "c"})
    records, _ = load_records(tmp_path / "results.jsonl")
    assert [r["id"] for r in records] == ["a", "b", "c"]


def test_corruption_before_the_last_line_is_an_error_not_a_repair(tmp_path):
    start_run(tmp_path, ["a"])
    with open(tmp_path / "results.jsonl", "a") as f:
        f.write("GARBAGE\n")
    append_record(tmp_path / "results.jsonl", {"id": "b"})
    with pytest.raises(ResumeError, match="line 2"):
        check_resume(tmp_path, dict(FINGERPRINT))
    # Nothing was truncated.
    assert (tmp_path / "results.jsonl").read_text().count("\n") == 3

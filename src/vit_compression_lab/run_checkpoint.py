"""Append-only JSONL checkpointing and resume checks for long evaluation runs.

A run directory holds ``run_config.json`` (the settings that must not change
between a run and its resumption) and ``results.jsonl`` (one completed
example per line). Resuming re-reads the completed ids and skips them.
"""

import hashlib
import json
import os
from pathlib import Path


class ResumeError(Exception):
    """The output directory cannot be resumed with the given settings."""


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_record(path, record):
    """Append one JSON line and flush it to disk before returning."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_records(path):
    """Records in a JSONL file.

    Only the last line may be partial (an interrupted write); it is removed
    from the file and reported via the second return value. Invalid JSON
    anywhere else raises ValueError so no completed record is ever dropped.
    A complete last line without its newline gets one appended.
    """
    path = Path(path)
    if not path.exists():
        return [], False
    data = path.read_bytes()
    lines = data.split(b"\n")
    if data.endswith(b"\n"):
        lines.pop()
    records, offset = [], 0
    for index, line in enumerate(lines):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise ValueError(
                        f"{path}: line {index + 1} is not valid JSON; inspect before resuming"
                    ) from None
                path.write_bytes(data[:offset])
                return records, True
        offset += len(line) + 1
    if lines and not data.endswith(b"\n"):
        with open(path, "ab") as f:
            f.write(b"\n")
    return records, False


def differing_keys(expected, actual):
    return sorted(
        key
        for key in set(expected) | set(actual)
        if json.dumps(expected.get(key), sort_keys=True)
        != json.dumps(actual.get(key), sort_keys=True)
    )


def check_resume(output_dir, fingerprint):
    """Validate a resume and return the ids already completed.

    Raises ResumeError if the directory is not a run, the settings differ, or
    results.jsonl contains duplicate ids.
    """
    output_dir = Path(output_dir)
    config_path = output_dir / "run_config.json"
    if not config_path.exists():
        raise ResumeError(f"{output_dir} has no run_config.json; not a resumable run")
    stored = json.loads(config_path.read_text())
    mismatched = differing_keys(stored, fingerprint)
    if mismatched:
        raise ResumeError(
            "Run settings differ from the stored run_config.json in: "
            + ", ".join(mismatched)
            + ". Start a new output directory instead."
        )
    try:
        records, repaired = load_records(output_dir / "results.jsonl")
    except ValueError as error:
        raise ResumeError(str(error)) from None
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ResumeError(
            "results.jsonl contains duplicate ids; inspect before resuming"
        )
    return set(ids), repaired

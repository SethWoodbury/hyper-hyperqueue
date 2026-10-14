import hashlib
import json
import multiprocessing
from pathlib import Path

import pytest

from hyper_hyperqueue.ingest import import_campaign_id, ingest_file, parse_file
from hyper_hyperqueue.state import StateStore, atomic_json, ensure_directory, read_json


def test_snapshot_duplicates_idempotence_and_explicit_append(tmp_path):
    source = tmp_path / "commands"
    source.write_text("# comment\n\n  printf '%s' '$HOME; quote'  \nprintf '%s' repeat\nprintf '%s' repeat")
    store = StateStore(tmp_path / "state")
    first = ingest_file(store, source, mode="direct", cwd=tmp_path, import_key="test")
    ids = first["task_ids"]
    assert len(ids) == len(set(ids)) == 3
    assert store.task(first["campaign_id"], ids[0]).command == "  printf '%s' '$HOME; quote'  "
    assert store.task(first["campaign_id"], ids[0]).source_line == 3
    assert (
        store.task(first["campaign_id"], ids[1]).command_digest
        == store.task(first["campaign_id"], ids[2]).command_digest
    )
    assert ingest_file(store, source, mode="direct", cwd=tmp_path, import_key="test") == first
    snapshot = Path(first["imports"][0]["snapshot"])
    assert snapshot.read_bytes() == source.read_bytes()
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == first["imports"][0]["sha256"]
    appended = ingest_file(
        store, source, mode="direct", cwd=tmp_path, campaign_id=first["campaign_id"], import_key="second"
    )
    assert len(appended["task_ids"]) == len(set(appended["task_ids"])) == 6
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="Import key already exists"):
        ingest_file(store, source, mode="direct", cwd=tmp_path, import_key="test")
    assert snapshot.read_text().startswith("# comment")


def test_literal_jsonl_and_continuation(tmp_path):
    source = tmp_path / "commands"
    source.write_text("# comment\n\ntrue\n")
    assert len(parse_file(source)) == 1
    assert len(parse_file(source, literal=True)) == 3
    source.write_text("echo one \\\ntwo\n")
    with pytest.raises(ValueError, match="JSONL"):
        parse_file(source)
    source.write_text(
        json.dumps({"command": "printf 'hello\\n'\nprintf world"})
        + "\n"
        + json.dumps({"argv": ["echo", "a;b"]})
    )
    records = parse_file(source, format="jsonl")
    assert records[0]["command"].endswith("\nprintf world")
    assert records[1]["argv"] == ["echo", "a;b"]
    source.write_text("printf 'literal\vcharacter'\r\ntrue")
    assert parse_file(source)[0]["command"] == "printf 'literal\vcharacter'"


def _events(root, n):
    store = StateStore(root)
    for i in range(n):
        store.event("concurrent", item=i)


def test_parallel_writers_and_interrupted_snapshot(tmp_path):
    root = tmp_path / "state"
    processes = [multiprocessing.Process(target=_events, args=(root, 8)) for _ in range(3)]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(30)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(5)
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    assert [event["sequence"] for event in events] == list(range(1, 25))
    # Simulate a completed journal append whose snapshot did not commit, then a partial write.
    atomic_json(root / "snapshot.json", {"last_sequence": 0, "journal_offset": 0})
    with (root / "events.jsonl").open("ab") as handle:
        handle.write(b'{"sequence":25')
    assert StateStore(root).event("recovered")["sequence"] == 25
    assert len((root / "events.jsonl").read_text().splitlines()) == 25
    assert read_json(root / "snapshot.json")["last_sequence"] == 25


def test_task_identity_immutable_and_intent_identity(tmp_path):
    source = tmp_path / "commands"
    source.write_text("true")
    store = StateStore(tmp_path / "state")
    campaign = ingest_file(store, source, mode="direct", cwd=tmp_path)
    task = store.task(campaign["campaign_id"], campaign["task_ids"][0])
    with pytest.raises(ValueError, match="Immutable"):
        store.add_task(task.model_copy(update={"command": "false"}))
    store.put_intent({"token": "same", "backend": "direct", "task_ids": [task.task_id], "state": "unknown"})
    with pytest.raises(ValueError, match="identity"):
        store.put_intent({"token": "same", "backend": "direct", "task_ids": [], "state": "submitted"})
    assert store.intents()[0]["state"] == "unknown"
    with pytest.raises(ValueError, match="Unsafe"):
        store.task("../escape", "task")
    with store.lock():
        store.update_task_state(task.campaign_id, task.task_id, {"state": "submitted"})
    state = store.task_state(task.campaign_id, task.task_id)
    assert state["state"] == state["status"] == "submitted"


def test_import_precreates_remote_runner_parents_and_mkdir_recovers(tmp_path, monkeypatch):
    source = tmp_path / "commands"
    source.write_text("true")
    store = StateStore(tmp_path / "state")
    campaign = ingest_file(store, source, mode="direct", cwd=tmp_path)
    directory = store.campaign_dir(campaign["campaign_id"])
    assert (directory / "attempts" / campaign["task_ids"][0]).is_dir()
    assert (directory / "task_locks").is_dir()
    assert (directory / "completions").is_dir()
    real = Path.mkdir
    transient = tmp_path / "negative_cache"
    calls = []

    def mkdir(path, *args, **kwargs):
        if path == transient and not calls:
            calls.append(True)
            raise FileExistsError("simulated concurrent mkdir with stale negative lookup")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    assert ensure_directory(transient).is_dir()
    assert calls


def test_literal_empty_file_has_no_line_but_blank_line_is_an_occurrence():
    from hyper_hyperqueue.ingest import parse_bytes

    assert parse_bytes(b"", literal=True) == []
    assert len(parse_bytes(b"\n", literal=True)) == 1


def test_initial_import_identity_remains_compatible_with_saved_campaigns(tmp_path):
    # Known schema-1 identity, independent of the implementation under test.
    expected = "c_9f86d081884c7d659a2feaa0"
    assert import_campaign_id("test") == expected
    source = tmp_path / "commands.txt"
    source.write_text("true\n")
    campaign = ingest_file(
        StateStore(tmp_path / "state"), source, mode="direct", cwd=tmp_path, import_key="test"
    )
    assert campaign["campaign_id"] == expected

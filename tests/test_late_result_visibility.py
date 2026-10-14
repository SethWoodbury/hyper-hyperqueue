"""A terminal HQ observation may beat final result visibility over NFS."""
import pytest

from hyper_hyperqueue.ingest import ingest_file
from hyper_hyperqueue.state import StateStore, atomic_json, fingerprint


def setup(tmp_path, state='interrupted', reason='worker_lost'):
    store = StateStore(tmp_path / 'state')
    source = tmp_path / 'commands'
    source.write_text('exit 1\n')
    campaign = ingest_file(store, source, mode='pooled', cwd=tmp_path)
    cid, tid = campaign['campaign_id'], campaign['task_ids'][0]
    task_fp = fingerprint(store.task(cid, tid).model_dump(mode='json'))
    attempt_id = '000001_test'
    directory = store.campaign_dir(cid) / 'attempts' / tid / attempt_id
    atomic_json(directory / 'attempt.json', {'attempt_id': attempt_id, 'task_fingerprint': task_fp})
    store.update_task_state(cid, tid, {'state': state, 'status': state, 'reason': reason,
        'after_attempt_id': attempt_id, 'hq_terminal_evidence': {'state': 'failed', 'id': 0}})
    return store, cid, tid, directory, {'attempt_id': attempt_id, 'task_fingerprint': task_fp,
                                      'status': 'failed', 'reason': 'application_exit'}


def test_late_matching_result_supersedes_provisional_hq_interruption(tmp_path):
    store, cid, tid, directory, result = setup(tmp_path)
    assert store.task_state(cid, tid)['state'] == 'interrupted'
    atomic_json(directory / 'result.json', result)
    current = store.task_state(cid, tid)
    assert current['state'] == current['status'] == 'failed'
    assert current['reason'] == 'application_exit'
    assert current['hq_terminal_evidence']['state'] == 'failed'  # Keep the original observation.


@pytest.mark.parametrize('state,reason', [('retry_wait', 'retry_authorized'), ('canceled', 'user_cancelled'),
                                         ('blocked', 'operator'), ('failed', 'retry_exhausted')])
def test_explicit_coordinator_decisions_remain_authoritative(tmp_path, state, reason):
    store, cid, tid, directory, result = setup(tmp_path, state, reason)
    atomic_json(directory / 'result.json', result)
    current = store.task_state(cid, tid)
    assert current['state'] == state and current['reason'] == reason


@pytest.mark.parametrize('key', ['task_fingerprint', 'attempt_id'])
def test_mismatched_late_result_cannot_override_provisional_state(tmp_path, key):
    store, cid, tid, directory, result = setup(tmp_path)
    result[key] = 'other'
    atomic_json(directory / 'result.json', result)
    assert store.task_state(cid, tid)['state'] == 'interrupted'

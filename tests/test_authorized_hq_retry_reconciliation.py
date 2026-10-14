import json
from pathlib import Path

import pytest

from hyper_hyperqueue import controller
from hyper_hyperqueue.ingest import ingest_file
from hyper_hyperqueue.models import Config
from hyper_hyperqueue.runner import authorize_retry, run_task
from hyper_hyperqueue.state import StateStore, atomic_json


def retry_fixture(tmp_path, monkeypatch, *, reason='application_exit', no_result=False):
    store = StateStore(tmp_path / 'state')
    marker = tmp_path / 'once'
    command = f'if [ ! -f {marker} ]; then touch {marker}; echo "first failure" >&2; exit 1; fi'
    commands = tmp_path / 'commands.txt'
    commands.write_text(command + '\ntrue\n')
    campaign = ingest_file(store, commands, mode='pooled', cwd=tmp_path,
                           retry={'max_attempts': 2, 'reasons': []},
                           output={'repeatable': True, 'mode': 'in_place'})
    cid = campaign['campaign_id']
    tid, sibling = campaign['task_ids']
    for key, value in {'HQ_JOB_ID': '7', 'HQ_TASK_ID': '0', 'HHQ_SERVER_EPOCH': '1'}.items():
        monkeypatch.setenv(key, value)
    assert run_task(store.root, cid, tid) == 1
    attempt = store.attempts(cid, tid)[0]
    result_path = Path(attempt['attempt_dir']) / 'result.json'
    if no_result:
        result_path.unlink()
    else:
        result = json.loads(result_path.read_text())
        result['reason'] = reason
        atomic_json(result_path, result)
    for task_id, index in [(tid, 0), (sibling, 1)]:
        store.update_task_state(cid, task_id, {'state': 'submitted', 'submission_token': 'oldtoken',
                                             'server_epoch': 1, 'hq_job_id': 7, 'hq_task_id': index})
    store.put_intent({'token': 'oldtoken', 'backend': 'hq', 'campaign_id': cid,
                      'task_ids': [tid, sibling], 'hq_job_id': 7, 'state': 'submitted', 'epoch': 1})
    proof = None
    if no_result:
        proof = tmp_path / 'synthetic_fencing_proof.json'
        atomic_json(proof, {'quiescent': True, 'previous_epoch': '1', 'attempt_id': attempt['attempt_id']})
    authorize_retry(store, cid, tid, manual=True, terminal_evidence=proof)
    return store, campaign, attempt


class Scheduler:
    def query_jobs(self):
        return []


class HQ:
    def __init__(self):
        self.submissions = []
        self.terminal = {'id': 0, 'state': 'failed', 'error': 'original attempt exited'}

    def tasks(self, job_id):
        if job_id == 7:
            return {'7': [self.terminal, {'id': 1, 'state': 'running'}]}
        return {str(job_id): [{'id': 0, 'state': 'waiting'}]}

    def submit(self, tasks, token, runner, **kwargs):
        self.submissions.append([task.task_id for task in tasks])
        return {'job_id': 8, 'tasks': {task.task_id: index for index, task in enumerate(tasks)}}


@pytest.mark.parametrize('reason,no_result', [('application_exit', False), ('gpu_oom', False), ('worker_lost', True)])
def test_authorized_retry_survives_full_tick_and_runs_once(tmp_path, monkeypatch, reason, no_result):
    store, campaign, attempt = retry_fixture(tmp_path, monkeypatch, reason=reason, no_result=no_result)
    cid = campaign['campaign_id']
    tid, sibling = campaign['task_ids']
    cfg = Config(state_dir=str(store.root))
    hq = HQ()
    monkeypatch.setattr(controller, 'assert_epoch', lambda *_: None)
    controller.tick(cfg, epoch=1, slurm=Scheduler(), hq=hq)
    state = store.task_state(cid, tid)
    assert state['state'] == 'submitted' and state['hq_job_id'] == 8
    assert state['after_attempt_id'] == attempt['attempt_id'] and state['retry_authorized']
    original = next(intent for intent in store.intents() if intent['token'] == 'oldtoken')
    assert original['retired_task_ids'] == [tid]
    assert original['retired_task_evidence'][tid]['hq_terminal_evidence'] == hq.terminal
    assert store.task_state(cid, sibling)['hq_job_id'] == 7
    # Repeated reconciliation of old terminal evidence cannot overwrite/resubmit the retry.
    controller.tick(cfg, epoch=1, slurm=Scheduler(), hq=hq)
    assert hq.submissions == [[tid]] and store.task_state(cid, tid)['hq_job_id'] == 8
    monkeypatch.setenv('HQ_JOB_ID', '8')
    assert run_task(store.root, cid, tid) == 0
    assert run_task(store.root, cid, tid) == 0  # Duplicate delivery reuses completion.
    assert len(store.attempts(cid, tid)) == 2
    assert store.task_state(cid, tid)['state'] == 'succeeded'
    assert len(store.attempts(cid, sibling)) == 0


def test_unbound_old_occurrence_does_not_replay_or_overwrite_retry(tmp_path, monkeypatch):
    store, campaign, _ = retry_fixture(tmp_path, monkeypatch)
    tid = campaign['task_ids'][0]
    intent = store.intents()[0]
    intent['hq_job_id'] = 999
    store.put_intent(intent)
    hq = HQ()
    hq.tasks = lambda _: [{'id': 0, 'state': 'failed'}]
    controller.reconcile_hq(store, hq)
    controller.admit_hq(store, Config(), hq, campaign, epoch=None)
    assert store.task_state(campaign['campaign_id'], tid)['state'] == 'retry_wait'
    assert not hq.submissions and not store.intents()[0].get('retired_task_ids')


@pytest.mark.parametrize('terminal_state', ['canceled', 'blocked'])
def test_later_explicit_cancellation_or_retirement_still_wins(tmp_path, monkeypatch, terminal_state):
    store, campaign, _ = retry_fixture(tmp_path, monkeypatch)
    tid = campaign['task_ids'][0]
    store.update_task_state(campaign['campaign_id'], tid, {'state': terminal_state, 'reason': 'operator_decision'})
    hq = HQ()
    controller.reconcile_hq(store, hq)
    controller.admit_hq(store, Config(), hq, campaign, epoch=None)
    assert store.task_state(campaign['campaign_id'], tid)['state'] == terminal_state
    assert not hq.submissions


def test_old_terminal_occurrence_cannot_overwrite_a_different_submission(tmp_path, monkeypatch):
    store, campaign, _ = retry_fixture(tmp_path, monkeypatch)
    tid = campaign['task_ids'][0]
    store.update_task_state(campaign['campaign_id'], tid, {
        'state': 'submitted', 'submission_token': 'newtoken', 'hq_job_id': 8, 'hq_task_id': 0,
    })
    store.put_intent({'token': 'newtoken', 'backend': 'hq', 'campaign_id': campaign['campaign_id'],
                      'task_ids': [tid], 'hq_job_id': 8, 'state': 'submitted', 'epoch': 1})
    controller.reconcile_hq(store, HQ())
    state = store.task_state(campaign['campaign_id'], tid)
    assert state['state'] == 'submitted' and state['submission_token'] == 'newtoken'
    assert state['hq_job_id'] == 8

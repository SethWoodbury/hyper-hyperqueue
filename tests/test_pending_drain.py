"""Release unused pending allocations after assignment without ending active work."""
from types import SimpleNamespace

import pytest

from hyper_hyperqueue.backends.slurm import CommandResult
from hyper_hyperqueue.ingest import ingest_file
from hyper_hyperqueue.models import Config
from hyper_hyperqueue.service.completion import reconcile_pending_drain
from hyper_hyperqueue.state import StateStore, read_json


def setup(tmp_path):
    store = StateStore(tmp_path / 'state')
    commands = tmp_path / 'commands'
    commands.write_text('true\n')
    campaign = ingest_file(store, commands, mode='pooled', cwd=tmp_path)
    tid = campaign['task_ids'][0]
    store.update_task_state(campaign['campaign_id'], tid, {'state': 'running'})
    store.put_intent({'backend': 'hq', 'token': 'hq_token', 'state': 'submitted',
                      'hq_job_id': 7, 'task_ids': [tid]})
    hq = SimpleNamespace(jobs=lambda: [{'id': 7, 'name': 'hhq:hq_token'}],
                         tasks=lambda _: [{'id': 0, 'state': 'running'}])
    config = Config(state_dir=str(store.root), policy={'cancel_pending_when_assigned': True})
    calls = []
    slurm = SimpleNamespace(user='owner', cancel_pending=lambda ids, expected_token:
        calls.append((ids, expected_token)) or CommandResult(['scancel', '--state=PENDING', *ids], 0, '', ''))
    jobs = []
    for i, state in [(1, 'PENDING'), (2, 'RUNNING')]:
        store.put_intent({'backend': 'worker', 'token': f'w{i}', 'state': 'submitted',
                          'epoch': 1, 'job_id': str(i), 'profile': 'cpu'})
        jobs.append({'job_id': str(i), 'state': state, 'user': 'owner', 'comment': f'hhq:w{i}'})
    return store, campaign, config, slurm, jobs, hq, calls


def test_assigned_work_releases_only_pending_and_keeps_running(tmp_path):
    store, campaign, config, slurm, jobs, hq, calls = setup(tmp_path)
    result = reconcile_pending_drain(store, config, slurm, jobs, hq, 1, [])
    assert result['all_work_assigned']
    assert calls == [(['1'], 'w1')]
    assert store.task_state(campaign['campaign_id'], campaign['task_ids'][0])['state'] == 'running'
    assert read_json(store.root / 'service/pending_drain_cancellations/w1.json')['reason'] == 'all_registered_work_assigned'
    assert not result['workers']['settled']


@pytest.mark.parametrize('state', ['ready', 'submitted', 'retry_wait', 'unknown', 'blocked'])
def test_unassigned_or_uncertain_work_blocks_cleanup(tmp_path, state):
    store, campaign, config, slurm, jobs, hq, calls = setup(tmp_path)
    store.update_task_state(campaign['campaign_id'], campaign['task_ids'][0], {'state': state})
    assert not reconcile_pending_drain(store, config, slurm, jobs, hq, 1, [])['all_work_assigned']
    assert not calls


@pytest.mark.parametrize('state', ['waiting', 'unknown'])
def test_actual_hq_waiting_blocks_despite_early_store_state(tmp_path, state):
    store, _, config, slurm, jobs, hq, calls = setup(tmp_path)
    hq.tasks = lambda _: [{'id': 0, 'state': state}]
    assert not reconcile_pending_drain(store, config, slurm, jobs, hq, 1, [])['all_work_assigned']
    assert not calls


def test_unknown_allocation_and_default_disable_cleanup(tmp_path):
    store, _, config, slurm, jobs, hq, calls = setup(tmp_path)
    store.put_intent({'token': 'unknown', 'backend': 'worker', 'state': 'unknown'})
    assert not reconcile_pending_drain(store, config, slurm, jobs, hq, 1, [])['all_work_assigned']
    assert not calls
    config.policy.cancel_pending_when_assigned = False
    assert reconcile_pending_drain(store, config, slurm, jobs, hq, 1, []) == {'enabled': False}


def test_foreign_worker_is_not_canceled_and_new_retry_blocks_drain(tmp_path):
    store, campaign, config, slurm, jobs, hq, calls = setup(tmp_path)
    jobs[0]['comment'] = 'foreign'
    result = reconcile_pending_drain(store, config, slurm, jobs, hq, 1, [])
    assert result['all_work_assigned'] and not calls
    assert result['workers']['blockers'][0]['reason'] == 'worker_ownership_unproven'
    store.update_task_state(campaign['campaign_id'], campaign['task_ids'][0], {'state': 'retry_wait'})
    assert not reconcile_pending_drain(store, config, slurm, jobs, hq, 1, [])['all_work_assigned']

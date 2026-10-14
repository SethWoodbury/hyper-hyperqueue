"""End-of-work ownership and re-opening regressions; no scheduler mutations."""
import time
from datetime import UTC, datetime

import pytest

from hyper_hyperqueue import api, controller
from hyper_hyperqueue.backends.slurm import CommandResult
from hyper_hyperqueue.config import save_config
from hyper_hyperqueue.ingest import ingest_file
from hyper_hyperqueue.models import Config
from hyper_hyperqueue.runner import run_task
from hyper_hyperqueue.service import supervisor
from hyper_hyperqueue.service.completion import reconcile_completion
from hyper_hyperqueue.state import StateStore, atomic_json


def finished(tmp_path):
    store = StateStore(tmp_path / 'state')
    source = tmp_path / 'commands'
    source.write_text('true\n')
    campaign = ingest_file(store, source, mode='pooled', cwd=tmp_path)
    assert run_task(store.root, campaign['campaign_id'], campaign['task_ids'][0]) == 0
    config = Config(state_dir=str(store.root), control={'stop_when_complete': True},
                    profiles={'cpu': {'partition': 'cpu'}})
    return store, campaign, config


class HQ:
    def jobs(self):
        return []


class Slurm:
    user = 'owner'

    def __init__(self):
        self.calls = []

    def cancel_pending(self, ids, expected_token):
        self.calls.append((ids, expected_token))
        return CommandResult(['scancel', '--ctld', '--state=PENDING', *ids], 0, '', '')


def test_pending_cleanup_waits_for_terminal_proof_and_never_changes_tasks(tmp_path):
    store, campaign, config = finished(tmp_path)
    token = 'worker_token'
    store.put_intent({'backend': 'worker', 'token': token, 'state': 'submitted',
                      'epoch': 1, 'job_id': '12', 'profile': 'cpu'})
    slurm = Slurm()
    pending = {'job_id': '12', 'state': 'PENDING', 'user': 'owner', 'comment': f'hhq:{token}'}
    with store.lock():
        status = reconcile_completion(store, config, slurm, [pending], HQ(), 1, [])
    assert not status['ready']
    assert slurm.calls == [(['12'], token)]
    assert store.intents()[0]['state'] == 'submitted'
    intent = store.intents()[0]
    intent.update(state='terminal', terminal_evidence=[{**pending, 'state': 'CANCELLED'}])
    store.put_intent(intent)
    with store.lock():
        status = reconcile_completion(store, config, slurm, [], HQ(), 1, [])
    assert status['ready']
    assert status['work']['counts'] == {'succeeded': 1}
    assert store.task_state(campaign['campaign_id'], campaign['task_ids'][0])['state'] == 'succeeded'


def test_completion_disabled_preserves_existing_service_behavior(tmp_path):
    store, _, config = finished(tmp_path)
    config.control.stop_when_complete = False
    assert reconcile_completion(store, config, object(), [], object(), 1, []) == {'enabled': False}
    assert Config().policy.reconcile_interval_seconds == 15
    assert not Config().control.stop_when_complete


def stopped_service(store, config):
    directory = store.root / 'service'
    path = directory / 'config.json'
    save_config(config, path)
    atomic_json(directory / 'deployment.json', {
        'deployment_id': 'deployment', 'desired_state': 'STOPPED', 'config': str(path),
        'hosting': 'slurm', 'allocations': [], 'created_at': time.time(),
    })
    atomic_json(directory / 'leader.json', {'epoch': 1, 'owner_id': 'owner', 'state': 'STOPPED'})
    atomic_json(directory / 'completion.json', {
        'deployment_id': 'deployment', 'epoch': 1, 'phase': 'children_stopped',
    })


def test_append_after_auto_shutdown_reopens_and_starts_only_new_work(tmp_path, monkeypatch):
    store, old, config = finished(tmp_path)
    stopped_service(store, config)
    proof, starts = [], []
    monkeypatch.setattr(supervisor, '_terminal_generation_proof',
                        lambda *a, **kw: proof.append(True))
    monkeypatch.setattr(supervisor, 'start_service',
                        lambda *a, **kw: starts.append(True) or {'status': 'started'})
    source = tmp_path / 'more'
    source.write_text('echo more\n')
    result = api.append(old['campaign_id'], source, config)
    assert result.backend_result == {'status': 'started'}
    assert proof == starts == [True]
    assert store.task_state(old['campaign_id'], old['task_ids'][0])['state'] == 'succeeded'
    assert result.task_count == 2
    new_tid = next(t for t in store.task_ids(old['campaign_id']) if t not in old['task_ids'])
    assert store.task_state(old['campaign_id'], new_tid)['state'] == 'ready'


def test_unknown_old_allocation_blocks_append_before_import(tmp_path, monkeypatch):
    store, old, config = finished(tmp_path)
    stopped_service(store, config)

    def unresolved(*a, **kw):
        raise RuntimeError('accounting missing')

    monkeypatch.setattr(supervisor, '_terminal_generation_proof', unresolved)
    source = tmp_path / 'more'
    source.write_text('echo more\n')
    with pytest.raises(ValueError, match='not yet positively terminal'):
        api.append(old['campaign_id'], source, config)
    assert store.task_ids(old['campaign_id']) == old['task_ids']


def test_direct_import_and_idempotent_old_import_ignore_closing_pooled_service(tmp_path):
    store, old, config = finished(tmp_path)
    stopped_service(store, config)
    assert ingest_file(store, tmp_path / 'commands', mode='pooled', cwd=tmp_path) == old
    source = tmp_path / 'direct'
    source.write_text('echo direct\n')
    assert ingest_file(store, source, mode='direct', cwd=tmp_path)['mode'] == 'direct'


def test_controller_checks_after_retry_authorization(tmp_path, monkeypatch):
    store = StateStore(tmp_path / 'state')
    source = tmp_path / 'commands'
    source.write_text('exit 3\n')
    campaign = ingest_file(store, source, mode='pooled', cwd=tmp_path,
                           retry={'max_attempts': 2, 'reasons': ['application_exit']},
                           output={'repeatable': True, 'mode': 'in_place'})
    assert run_task(store.root, campaign['campaign_id'], campaign['task_ids'][0]) == 3
    config = Config(state_dir=str(store.root), control={'stop_when_complete': True})
    monkeypatch.setattr(controller, 'assert_epoch', lambda *a: None)
    monkeypatch.setattr(controller, 'admit_hq', lambda *a: None)
    monkeypatch.setattr(controller, 'reconcile_hq', lambda *a: None)
    monkeypatch.setattr(controller, 'capacity_plan', lambda *a, **kw: {'actions': [], 'reasons': []})

    class Scheduler(Slurm):
        def query_jobs(self):
            return []

    result = controller.tick(config, epoch=1, slurm=Scheduler(), hq=HQ())
    assert result['retry_decisions'][0]['decision'] == 'retry_authorized'
    assert result['completion']['work']['counts'] == {'retry_wait': 1}
    assert not result['completion']['ready']


def test_failed_exhausted_task_is_finished_without_becoming_success(tmp_path):
    store = StateStore(tmp_path / 'state')
    source = tmp_path / 'commands'
    source.write_text('exit 3\n')
    campaign = ingest_file(store, source, mode='pooled', cwd=tmp_path)
    assert run_task(store.root, campaign['campaign_id'], campaign['task_ids'][0]) == 3
    config = Config(state_dir=str(store.root), control={'stop_when_complete': True})
    result = reconcile_completion(store, config, Slurm(), [], HQ(), 1, [])
    assert result['ready'] and result['work']['counts'] == {'failed': 1}


def test_stale_healthy_completion_report_cannot_stop_service(tmp_path):
    store, _, config = finished(tmp_path)
    atomic_json(store.root / 'controller_status.json', {
        'state': 'healthy', 'at': datetime.now(UTC).isoformat(),
        'completion': {'ready': True, 'epoch': 1, 'observed_at': time.time() - 120},
    })
    guard = type('Guard', (), {'epoch': 1})()
    assert not supervisor.finish_completed_service(config, guard, object(), lambda: pytest.fail('stop'))

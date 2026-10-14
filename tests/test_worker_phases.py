"""Same-allocation fallback is explicit, single-phase and conservative."""
import os
import signal
import sys
import time
from datetime import UTC, datetime

import pytest

from hyper_hyperqueue.ingest import ingest_file
from hyper_hyperqueue.models import Config
from hyper_hyperqueue.runner import run_task
from hyper_hyperqueue.state import StateStore, atomic_json
from hyper_hyperqueue.worker import run_worker_phase, select_fallback


def phase(command):
    return [sys.executable, '-c', command, '--time-limit', '1s']


def test_worker_receipt_distinguishes_natural_deadline_and_normal_exit():
    ticks = iter([100, 101])
    result = run_worker_phase(phase('pass'), clock=lambda: next(ticks))
    assert result['termination_reason'] == 'time_limit'
    assert result['configured_end_time'] == 101 and result['returncode'] == 0
    ticks = iter([100, 100.5])
    assert run_worker_phase(phase('pass'), clock=lambda: next(ticks))['termination_reason'] == 'normal_exit'


def test_forwarded_signal_is_never_natural_timeout():
    result = run_worker_phase(phase('import os,signal,time; os.kill(os.getppid(),signal.SIGUSR1); time.sleep(0.05)'))
    assert result['termination_reason'] == 'signal'
    assert result['received_signals'] == [signal.SIGUSR1]


def setup(tmp_path):
    store = StateStore(tmp_path / 'state')
    config = Config(state_dir=str(store.root), control={'hosting': 'local'}, profiles={
        'large': {'partition': 'train', 'memory_mib': 2048, 'allocation_seconds': 300, 'fallback_profiles': ['small']},
        'small': {'partition': 'train', 'memory_mib': 1024, 'allocation_seconds': 300},
    })
    commands = tmp_path / 'commands'
    commands.write_text('true\n')
    primary = ingest_file(store, commands, mode='pooled', cwd=tmp_path,
                          resources={'memory_mib': 512, 'runtime_seconds': 1}, profile_names=['large'])
    assert run_task(store.root, primary['campaign_id'], primary['task_ids'][0]) == 0
    fallback = ingest_file(store, commands, mode='pooled', cwd=tmp_path, import_key='fallback',
                           resources={'memory_mib': 512, 'runtime_seconds': 1}, profile_names=['small'])
    store.update_task_state(fallback['campaign_id'], fallback['task_ids'][0], {'state': 'submitted'})
    atomic_json(store.root / 'service/endpoint.json', {'epoch': 1, 'status': 'ACTIVE'})
    atomic_json(store.root / 'service/deployment.json', {'desired_state': 'RUNNING'})
    atomic_json(store.root / 'controller_status.json', {'epoch': 1, 'state': 'healthy',
                 'at': datetime.now(UTC).isoformat(), 'worker_capacity': []})
    assigned = {'cpus': 1, 'memory_mib': 2048, 'gpu': None}
    return store, config, assigned, primary, fallback


def choose(store, config, assigned):
    return select_fallback(store, config, 'large', assigned, 1, 'self', 200)


def test_fallback_preserves_original_tasks_and_requires_idle_primary_capacity(tmp_path):
    store, config, assigned, _, fallback = setup(tmp_path)
    original = store.task(fallback['campaign_id'], fallback['task_ids'][0]).model_dump()
    assert choose(store, config, assigned) == 'small'
    assert store.task(fallback['campaign_id'], fallback['task_ids'][0]).model_dump() == original
    store.put_intent({'token': 'another', 'backend': 'worker', 'profile': 'small', 'state': 'submitted'})
    # A missing/uncertain worker observation blocks opportunistic switching.
    assert choose(store, config, assigned) is None
    report = {'epoch': 1, 'state': 'healthy', 'at': datetime.now(UTC).isoformat(),
              'worker_capacity': [{'token': 'another', 'profile': 'small', 'state': 'RUNNING'}]}
    atomic_json(store.root / 'controller_status.json', report)
    assert choose(store, config, assigned) is None
    report['worker_capacity'][0]['state'] = 'PENDING'
    atomic_json(store.root / 'controller_status.json', report)
    assert choose(store, config, assigned) == 'small'
    intent = store.intents()[0]
    intent['dispatch_profile'] = 'small'  # Another phase claim wins before the next controller tick.
    store.put_intent(intent)
    assert choose(store, config, assigned) is None


@pytest.mark.parametrize('block', ['running_primary', 'stale', 'draining', 'epoch', 'paused', 'ram', 'time'])
def test_fallback_fails_closed(tmp_path, block):
    store, config, assigned, _primary, fallback = setup(tmp_path)
    if block == 'running_primary':
        # Completion records are authoritative: create a new unfinished primary occurrence.
        source = tmp_path / 'new'
        source.write_text('echo later\n')
        ingest_file(store, source, mode='pooled', cwd=tmp_path, profile_names=['large'])
    elif block in {'stale', 'epoch'}:
        atomic_json(store.root / 'controller_status.json', {'epoch': 2 if block == 'epoch' else 1,
            'state': 'healthy', 'at': datetime.fromtimestamp(time.time() - 100, UTC).isoformat()})
    elif block == 'draining':
        atomic_json(store.root / 'service/endpoint.json', {'epoch': 1, 'status': 'DRAINING'})
    elif block == 'paused':
        fallback['pause_dispatch'] = True
        store.save_campaign(fallback)
    elif block == 'ram':
        assigned['memory_mib'] = 400
    else:
        assert select_fallback(store, config, 'large', assigned, 1, 'self', 1) is None
        return
    assert choose(store, config, assigned) is None


@pytest.mark.integration
def test_real_hq_idle_large_worker_switches_to_waiting_small_task(tmp_path):
    import contextlib
    import subprocess

    from hyper_hyperqueue.backends.hyperqueue import HQBackend, HQError, locate_binary
    from hyper_hyperqueue.config import save_config
    from hyper_hyperqueue.controller import runner_argv
    from hyper_hyperqueue.state import read_json

    try:
        binary = locate_binary()
        __import__('hyperqueue')
    except (RuntimeError, ImportError):
        pytest.skip('Pinned optional HQ runtime not installed')
    store, config, _, _, fallback = setup(tmp_path)
    config.policy.worker_idle_seconds = 1
    config.policy.cleanup_seconds = 1
    config_path = tmp_path / 'config.json'
    save_config(config, config_path)
    directory = store.root / 'service/hq'
    log = (tmp_path / 'hq.log').open('w')
    server = subprocess.Popen([binary, '--server-dir', str(directory), 'server', 'start', '--host', '127.0.0.1'],
                              stdout=log, stderr=log, start_new_session=True)
    worker = None
    try:
        backend = HQBackend(directory, binary)
        for _ in range(100):
            try:
                backend.json('server', 'info')
                break
            except HQError:
                time.sleep(0.1)
        else:
            pytest.fail('local HQ server did not start')
        task = store.task(fallback['campaign_id'], fallback['task_ids'][0])
        submitted = backend.submit([task], 'fallback_test', lambda t: runner_argv(store, t), epoch=1)
        store.put_intent({'backend': 'hq', 'token': 'fallback_test', 'state': 'submitted',
                          'hq_job_id': submitted['job_id'], 'task_ids': [task.task_id]})
        store.put_intent({'backend': 'worker', 'token': 'self', 'profile': 'large', 'state': 'running', 'epoch': 1})
        env = {key: value for key, value in os.environ.items() if not key.startswith('SLURM_')}
        env['HHQ_ALLOCATION_TOKEN'] = 'self'
        worker = subprocess.Popen([sys.executable, '-m', 'hyper_hyperqueue.worker', '--config', str(config_path),
            '--state-dir', str(store.root), '--epoch', '1', '--profile', 'large'],
            env=env, stdout=log, stderr=log, start_new_session=True)
        assert worker.wait(timeout=20) == 0
        record = read_json(store.root / f'service/worker_exits/local-{worker.pid}.json')
        assert [p['dispatch_profile'] for p in record['phases']] == ['large', 'small']
        assert store.task_state(task.campaign_id, task.task_id)['state'] == 'succeeded'
        assert len(store.attempts(task.campaign_id, task.task_id)) == 1
        assert len(backend.workers()) == 2  # Sequential HQ workers inside ONE allocation parent.
    finally:
        for process in (worker, server):
            if process and process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
        log.close()


@pytest.mark.parametrize('record', ['stop', 'paused_pool', 'missing_report'])
def test_operator_stop_pause_and_missing_evidence_prevent_fallback(tmp_path, record):
    store, config, assigned, _, _ = setup(tmp_path)
    if record == 'stop':
        atomic_json(store.root / 'service/deployment.json', {'desired_state': 'STOPPED'})
    elif record == 'paused_pool':
        atomic_json(store.root / 'controls.json', {'paused_pools': ['small']})
    else:
        (store.root / 'controller_status.json').unlink()
    assert choose(store, config, assigned) is None

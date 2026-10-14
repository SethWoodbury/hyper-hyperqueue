from hyper_hyperqueue.models import Config
from hyper_hyperqueue.worker_capacity import admission_dependencies


def setup(states):
    config = Config(budgets={'train': {'partitions': ['gpu-train'], 'max_running': 2,
                                      'max_pending': 1, 'max_submitted': 3}},
                    profiles={'beam': {'partition': 'gpu-train', 'budget': 'train'}})
    owned, jobs = [], []
    for index, state in enumerate(states, 10):
        owned.append({'backend': 'worker', 'token': f'w{index}', 'job_id': str(index),
                      'profile': 'beam', 'state': 'submitted'})
        if state != 'UNKNOWN':
            jobs.append({'job_id': str(index), 'state': state, 'comment': f'hhq:w{index}'})
    return config, owned, jobs


def test_two_running_train_workers_get_one_or_dependency_reserve():
    config, owned, jobs = setup(['RUNNING', 'RUNNING'])
    report = admission_dependencies(config, config.profiles['beam'], jobs, owned)
    assert report['allowed']
    assert report['extra_args'] == ['--dependency', 'afterany:10?afterany:11']
    owned.append({'backend': 'worker', 'token': 'new', 'profile': 'beam', 'state': 'intent'})
    assert not admission_dependencies(config, config.profiles['beam'], jobs, owned)['allowed']


def test_less_than_limit_can_start_and_pending_blocks_another_reserve():
    for states, allowed in [([], True), (['RUNNING'], True), (['RUNNING', 'PENDING'], False),
                            (['UNKNOWN'], False), (['RUNNING'] * 3, False)]:
        config, owned, jobs = setup(states)
        result = admission_dependencies(config, config.profiles['beam'], jobs, owned)
        assert result['allowed'] is allowed
        assert not result['extra_args']


def test_wrong_token_is_not_running_proof_and_default_has_no_dependencies():
    config, owned, jobs = setup(['RUNNING', 'RUNNING'])
    jobs[0]['comment'] = 'foreign'
    assert not admission_dependencies(config, config.profiles['beam'], jobs, owned)['allowed']
    config.budgets['train'].max_running = None
    assert admission_dependencies(config, config.profiles['beam'], jobs, owned)['extra_args'] == []


def test_ended_or_canceled_predecessor_is_no_longer_a_dependency():
    config, owned, jobs = setup(['RUNNING', 'RUNNING'])
    owned[0]['state'] = 'terminal'
    jobs.pop(0)
    result = admission_dependencies(config, config.profiles['beam'], jobs, owned)
    assert result['allowed'] and not result['extra_args']


def test_controller_emits_dependency_on_exact_owned_running_workers(tmp_path, monkeypatch):
    from hyper_hyperqueue import controller
    from hyper_hyperqueue.ingest import ingest_file
    from hyper_hyperqueue.state import StateStore

    config, owned, jobs = setup(['RUNNING', 'RUNNING'])
    config.state_dir = str(tmp_path / 'state')
    store = StateStore(config.state_dir)
    for intent in owned:
        store.put_intent(intent)
    commands = tmp_path / 'commands'
    commands.write_text('true\n')
    campaign = ingest_file(store, commands, mode='pooled', cwd=tmp_path,
                           resources={'memory_mib': 512, 'runtime_seconds': 1}, profile_names=['beam'])
    config.control.hosting = 'local'
    monkeypatch.setattr(controller, 'assert_epoch', lambda *a: None)
    monkeypatch.setattr(controller, 'admit_hq', lambda *a: None)
    monkeypatch.setattr(controller, 'reconcile_hq', lambda *a: None)
    monkeypatch.setattr(controller, 'retry_ready_failures', lambda *a: [])
    monkeypatch.setattr(controller, 'capacity_plan', lambda *a, **k: {
        'actions': [{'profile': 'beam', 'reason': 'reserve', 'covered_task_ids': campaign['task_ids']}], 'reasons': []})
    class Slurm:
        def query_jobs(self):
            return jobs
        def submit(self, script, **kwargs):
            self.args = kwargs['extra_args']
            return {'status': 'accepted', 'job_id': '12'}
    slurm = Slurm()
    controller.tick(config, epoch=1, slurm=slurm, hq=object())
    assert slurm.args[slurm.args.index('--dependency') + 1] == 'afterany:10?afterany:11'
    intent = next(row for row in store.intents() if row.get('job_id') == '12')
    assert intent['concurrency_admission']['counts']['train']['running'] == 2


def test_zero_pending_reserve_cannot_create_extra_allocation_at_running_limit():
    config, owned, jobs = setup(['RUNNING', 'RUNNING'])
    config.budgets['train'].max_pending = 0
    assert not admission_dependencies(config, config.profiles['beam'], jobs, owned)['allowed']


def test_controller_sizes_memory_for_each_action_not_whole_retry_envelope(tmp_path, monkeypatch):
    from hyper_hyperqueue import adaptive_recovery, controller
    from hyper_hyperqueue.ingest import ingest_file
    from hyper_hyperqueue.state import StateStore, read_json

    config = Config(state_dir=str(tmp_path / 'state'), control={'hosting': 'local'},
                    budgets={'cpu': {'partitions': ['cpu'], 'max_submitted': 4}},
                    profiles={'cpu': {'partition': 'cpu', 'budget': 'cpu', 'memory_mib': 2048}})
    store = StateStore(config.state_dir)
    task_ids = []
    for memory in (1024, 4096):
        commands = tmp_path / f'commands_{memory}'
        commands.write_text('true\n')
        campaign = ingest_file(store, commands, mode='pooled', cwd=tmp_path,
            resources={'memory_mib': memory, 'runtime_seconds': 1}, profile_names=['cpu'])
        task_ids.extend(campaign['task_ids'])
    def envelope(_store, base, _tasks):
        copy = base.model_copy(deep=True)
        copy.profiles['cpu'].memory_mib = 8192
        return copy
    monkeypatch.setattr(adaptive_recovery, 'effective_config', envelope)
    monkeypatch.setattr(controller, 'assert_epoch', lambda *a: None)
    monkeypatch.setattr(controller, 'admit_hq', lambda *a: None)
    monkeypatch.setattr(controller, 'reconcile_hq', lambda *a: None)
    monkeypatch.setattr(controller, 'retry_ready_failures', lambda *a: [])
    monkeypatch.setattr(controller, 'capacity_plan', lambda *a, **k: {'actions': [
        {'profile': 'cpu', 'reason': 'test', 'covered_task_ids': [tid]} for tid in task_ids], 'reasons': []})
    class Slurm:
        def __init__(self):
            self.requests = []
        def query_jobs(self):
            return []
        def submit(self, script, **kwargs):
            self.requests.append(kwargs['extra_args'])
            return {'status': 'accepted', 'job_id': str(10 + len(self.requests))}
    slurm = Slurm()
    controller.tick(config, epoch=1, slurm=slurm, hq=object())
    assert [args[args.index('--mem') + 1] for args in slurm.requests] == ['2048M', '4352M']
    snapshots = sorted((row for row in store.intents() if row.get('backend') == 'worker'), key=lambda row: row['job_id'])
    assert [read_json(row['config_snapshot'])['profiles']['cpu']['memory_mib'] for row in snapshots] == [2048, 4352]
    assert config.profiles['cpu'].memory_mib == 2048

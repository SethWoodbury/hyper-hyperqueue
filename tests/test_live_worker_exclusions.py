import json
from pathlib import Path

import pytest

from hyper_hyperqueue import controller
from hyper_hyperqueue.ingest import ingest_file
from hyper_hyperqueue.models import Config
from hyper_hyperqueue.state import StateStore, atomic_json


def configuration(tmp_path):
    return Config(
        state_dir=str(tmp_path / 'state'),
        profiles={
            'beam': {'partition': 'gpu', 'exclude': ['g[100-102]'], 'fallback_profiles': ['gradient']},
            'gradient': {'partition': 'gpu', 'exclude': ['g[100-102]']},
            'cpu': {'partition': 'cpu'},
        },
    )


def test_additive_global_exclusions_preserve_base_and_expressions(tmp_path):
    config = configuration(tmp_path)
    before = config.model_dump()
    derived, applied = controller.worker_exclusions(config, {'excluded_nodes': ['g2507', 'g1108', 'g2507']})
    assert config.model_dump() == before
    assert applied == {name: ['g1108', 'g2507'] for name in config.profiles}
    assert derived.profiles['beam'].exclude == ['g[100-102]', 'g1108', 'g2507']
    assert derived.profiles['gradient'].exclude == derived.profiles['beam'].exclude
    assert derived.profiles['cpu'].exclude == ['g1108', 'g2507']
    Config.model_validate(derived.model_dump())


@pytest.mark.parametrize('profile', ['beam', 'gradient'])
def test_scoped_exclusions_follow_fallback_in_both_directions(tmp_path, profile):
    config = configuration(tmp_path)
    derived, applied = controller.worker_exclusions(config, {'excluded_nodes_by_profile': {profile: ['bad-node']}})
    assert applied == {'beam': ['bad-node'], 'gradient': ['bad-node']}
    assert derived.profiles['cpu'].exclude == []
    assert derived.profiles['beam'].exclude == derived.profiles['gradient'].exclude
    Config.model_validate(derived.model_dump())


def test_no_override_preserves_legacy_object(tmp_path):
    config = configuration(tmp_path)
    derived, applied = controller.worker_exclusions(config, {'paused_pools': []})
    assert derived is config and applied == {}


@pytest.mark.parametrize('controls', [
    [], {'excluded_nodes': 'g2507'}, {'excluded_nodes': [None]}, {'excluded_nodes': [True]},
    {'excluded_nodes': ['--exclude=x']}, {'excluded_nodes': ['g1,g2']},
    {'excluded_nodes': ['g[1-2]']}, {'excluded_nodes': ['g1\n--partition=x']},
    {'excluded_nodes_by_profile': {'missing': ['g1']}}, {'excluded_nodes_by_profile': []},
])
def test_invalid_controls_block_before_scheduler_or_task_admission(tmp_path, controls):
    config = configuration(tmp_path)
    store = StateStore(config.state_dir)
    atomic_json(store.root / 'controls.json', controls)

    class NoCalls:
        def query_jobs(self):
            pytest.fail('invalid controls must stop before any scheduler call')

    with pytest.raises((ValueError, TypeError)):
        controller.tick(config, slurm=NoCalls())
    assert store.intents() == []


def test_tick_snapshots_exclusions_without_changing_task_or_startup_circuit(tmp_path, monkeypatch):
    config = Config(state_dir=str(tmp_path / 'state'), profiles={'cpu': {'partition': 'cpu', 'budget': 'cpu'}},
                    budgets={'cpu': {'running_target': 1, 'max_submitted': 2}},
                    policy={'submit_cooldown_seconds': 0})
    store = StateStore(config.state_dir)
    source = tmp_path / 'commands.txt'
    source.write_text('true\n')
    campaign = ingest_file(store, source, mode='pooled', cwd=tmp_path, profile_names=['cpu'],
                           resources={'cpus': 1, 'memory_mib': 512, 'runtime_seconds': 10})
    task_id = campaign['task_ids'][0]
    task_before = store.task(campaign['campaign_id'], task_id)
    atomic_json(store.root / 'controls.json', {'excluded_nodes': ['bad-node']})
    monkeypatch.setattr(controller, 'assert_epoch', lambda *_: None)
    monkeypatch.setattr(controller, 'control_deadline', lambda *_: None)
    monkeypatch.setattr(controller, 'reconcile_hq', lambda *_: None)
    monkeypatch.setattr(controller, 'admit_hq', lambda *_: None)

    class Scheduler:
        def query_jobs(self):
            return []

        def submit(self, script, token, job_name, extra_args):
            assert extra_args[extra_args.index('--exclude') + 1] == 'bad-node'
            snap = json.loads((Path(script).parent / 'config.json').read_text())
            assert snap['profiles']['cpu']['exclude'] == ['bad-node']
            return {'status': 'accepted', 'job_id': '100', 'job_name': job_name + '-' + token}

    result = controller.tick(config, epoch=1, slurm=Scheduler(), hq=object())
    assert result['plan']['worker_node_exclusions'] == {'cpu': ['bad-node']}
    assert config.profiles['cpu'].exclude == []
    assert store.task(campaign['campaign_id'], task_id) == task_before
    assert json.loads((store.root / 'controls.json').read_text()) == {'excluded_nodes': ['bad-node']}
    assert len([intent for intent in store.intents() if intent['backend'] == 'worker']) == 1

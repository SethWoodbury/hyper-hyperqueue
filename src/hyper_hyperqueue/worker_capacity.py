"""Optional strict owned-worker concurrency through Slurm reserve dependencies."""
from .planner import ACTIVE_INTENT_STATES, _positive_worker_job, _profile_budgets, job_dict


def admission_dependencies(config, profile, jobs, owned):
    """Caller holds the submission lock; one pending reserve per limited budget.

    No allocation is canceled. An OR-afterany dependency makes the reserve wait
    for a running slot, even when Slurm can otherwise start it between polls.
    External jobs still consume the separate aggregate/site budgets.
    """
    observed = [job_dict(job) for job in jobs]
    dependencies, counts = [], {}
    for name in _profile_budgets(config, profile):
        budget = config.budgets[name]
        if budget.max_running is None:
            continue
        running, pending, uncertain = [], 0, 0
        for allocation in owned:
            if allocation.get('backend') != 'worker' or allocation.get('state') not in ACTIVE_INTENT_STATES:
                continue
            source = config.profiles.get(allocation.get('profile'))
            if source is None:
                uncertain += 1
                continue
            if name not in _profile_budgets(config, source):
                continue
            matches = [job for job in observed if _positive_worker_job(job, allocation)]
            states = {job.get('state') for job in matches}
            if states == {'RUNNING'} and len(matches) == 1:
                jid = str(matches[0]['job_id'])
                if not jid.isdigit():
                    uncertain += 1
                else:
                    running.append(jid)
            elif states == {'PENDING'}:
                pending += 1
            else:
                uncertain += 1
        counts[name] = {'running': len(running), 'pending': pending, 'unknown': uncertain,
                        'max_running': budget.max_running}
        reason = None
        if uncertain:
            reason = 'STRICT_RUNNING_OWNERSHIP_UNKNOWN'
        elif pending:
            reason = 'STRICT_RUNNING_RESERVE_ALREADY_PENDING'
        elif len(running) > budget.max_running:
            reason = 'STRICT_RUNNING_LIMIT_ALREADY_EXCEEDED'
        elif len(running) == budget.max_running and budget.max_pending == 0:
            reason = 'STRICT_RUNNING_LIMIT_NO_PENDING_RESERVE'
        if reason:
            return {'allowed': False, 'reason': f'{name}: {reason}', 'counts': counts, 'extra_args': []}
        if len(running) == budget.max_running:
            dependencies.append('?'.join(f'afterany:{jid}' for jid in sorted(running, key=int)))
    if len(set(dependencies)) > 1:
        return {'allowed': False, 'reason': 'OVERLAPPING_RUNNING_LIMIT_DEPENDENCIES', 'counts': counts, 'extra_args': []}
    return {'allowed': True, 'reason': None, 'counts': counts,
            'extra_args': ['--dependency', dependencies[0]] if dependencies else []}

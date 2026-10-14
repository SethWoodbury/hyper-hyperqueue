from hyper_hyperqueue.accounting import FIELDS, allocation_report, parse_allocations, summarize
from hyper_hyperqueue.backends.slurm import CommandResult


def fixture(*values):
    return "|".join(FIELDS) + "\n" + "\n".join("|".join(row) for row in values) + "\n"


def row(job="123", start="2026-10-14T01:00:00", elapsed="3600", tres=None):
    return [
        "42",
        job,
        "user",
        "gpu",
        "normal",
        "COMPLETED",
        "2026-10-14T00:00:00",
        start,
        "2026-10-14T02:00:00",
        elapsed,
        "480",
        "cpu=2,mem=18G,gres/gpu=1,gres/gpu:large=1,billing=18" if tres is None else tres,
        "node1",
    ]


def test_actual_elapsed_not_requested_and_no_double_gpu_count():
    result = parse_allocations(fixture(row()), usage_factors={"normal": 0.5})[0]
    assert result["gpu_hours"] == 1
    assert result["cpu_hours"] == 2
    assert result["memory_gib_hours"] == 18
    assert result["billing_unit_hours"] == 18
    assert result["weighted_billing_unit_hours"] == 9
    assert result["submit_to_start_seconds"] == 3600


def test_pending_zero_unknown_not_zero_and_no_steps():
    rows = parse_allocations(fixture(row(start="Unknown", elapsed="0", tres=""), row(job="123.batch")))
    assert len(rows) == 1 and rows[0]["gpu_hours"] == 0
    missing = parse_allocations(fixture(row(tres="cpu=2")))
    result = summarize(missing)
    assert result["totals"]["billing_unit_hours"] is None
    assert result["totals"]["billing_unit_hours_missing_records"] == 1


def test_requeue_attempts_preserved_exact_duplicates_dropped():
    rows = parse_allocations(fixture(row(), row(), row(start="2026-10-14T04:00:00")))
    assert len(rows) == 2
    assert summarize(rows)["totals"]["gpu_hours"] == 2


def test_report_rejects_injection_and_reports_missing():
    import pytest

    with pytest.raises(ValueError):
        allocation_report(["123; touch bad"])
    calls = []

    def query(argv):
        calls.append(argv)
        return CommandResult(argv, 0, fixture(row()))

    report = allocation_report(["123", "456"], command=query)
    assert report["missing_job_ids"] == ["456"]
    assert len(calls) == 1

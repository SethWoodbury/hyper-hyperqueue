from hyper_hyperqueue.estimates import summarize_measurements


def test_failures_and_hardware_not_mixed():
    rows = [
        {"state": "succeeded", "gpu_model": "A", "wall_seconds": x, "peak_memory_mib": 100}
        for x in range(10, 15)
    ]
    rows += [{"state": "failed", "gpu_model": "A", "wall_seconds": 0.1, "peak_memory_mib": 50000}]
    rows += [{"state": "succeeded", "gpu_model": "B", "wall_seconds": 50}]
    result = summarize_measurements(rows)
    assert result[0]["samples"] == 5
    assert result[0]["suggested_memory_mib"] == 2148
    assert result[0]["suggested_runtime_seconds"] == 74
    assert result[1]["suggested_runtime_seconds"] is None

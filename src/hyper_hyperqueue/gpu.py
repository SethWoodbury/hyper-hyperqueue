"""Identify CUDA-visible devices without guessing host/NVML numeric indices."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import uuid


def cuda_visible_devices(expected_count: int, *, library=None) -> list[dict]:
    """Driver metadata only: no CUDA context, allocation, or visibility override."""
    driver = library if library is not None else ctypes.CDLL("libcuda.so.1")

    def invoke(name, argtypes, *args):
        function = getattr(driver, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
        code = function(*args)
        if code != 0:
            raise ValueError(f"{name} failed with CUDA driver error {code}")

    invoke("cuInit", [ctypes.c_uint], 0)
    count = ctypes.c_int()
    invoke("cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)], ctypes.byref(count))
    if count.value != expected_count:
        raise ValueError(
            f"CUDA-visible device count {count.value} differs from assigned count {expected_count}"
        )
    records = []
    for ordinal in range(count.value):
        device = ctypes.c_int()
        invoke("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.byref(device), ordinal)
        name = ctypes.create_string_buffer(256)
        invoke("cuDeviceGetName", [ctypes.c_char_p, ctypes.c_int, ctypes.c_int], name, len(name), device)
        memory = ctypes.c_size_t()
        invoke(
            "cuDeviceTotalMem_v2",
            [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int],
            ctypes.byref(memory),
            device,
        )
        identity = (ctypes.c_ubyte * 16)()
        uuid_function = "cuDeviceGetUuid_v2" if hasattr(driver, "cuDeviceGetUuid_v2") else "cuDeviceGetUuid"
        invoke(uuid_function, [ctypes.c_void_p, ctypes.c_int], ctypes.byref(identity), device)
        model = name.value.decode("utf-8", errors="strict")
        if "MIG" in model.upper():
            raise ValueError("MIG device identity is not qualified in this release")
        if not any(identity) or memory.value <= 0 or not model:
            raise ValueError("CUDA driver returned incomplete assigned-device identity")
        records.append(
            {
                "uuid": "GPU-" + str(uuid.UUID(bytes=bytes(identity))),
                "model": model,
                "memory_mib": memory.value // (1024 * 1024),
                "cuda_ordinal": ordinal,
                "identity_source": uuid_function,
            }
        )
    if len({record["uuid"] for record in records}) != len(records):
        raise ValueError("CUDA-visible devices did not have unique identities")
    return records


def query_assigned_devices(expected_count: int, *, environment=None, query=None) -> list[dict]:
    """Bounded shared startup check used by both workers and task runners.

    `query` is a test seam taking a CUDA-visible ordinal, never a host GPU ID.
    Production probes run in a fresh process so driver initialization observes
    precisely the inherited visibility and cannot hold up startup indefinitely.
    """
    env = os.environ if environment is None else environment
    visible = [part.strip() for part in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
    if len(visible) != expected_count or any(value in {"-1", "NoDevFiles"} for value in visible):
        raise ValueError("Declared GPU count must match explicitly assigned CUDA_VISIBLE_DEVICES")
    if len(set(visible)) != len(visible) or any(value.startswith("MIG-") for value in visible):
        raise ValueError("Duplicate or unqualified MIG CUDA device assignment")
    if query is not None:
        devices = [dict(query(ordinal)) for ordinal in range(expected_count)]
    else:
        process = subprocess.run(
            [sys.executable, "-m", "hyper_hyperqueue.gpu", "--count", str(expected_count)],
            env=dict(env),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if process.returncode:
            detail = (process.stderr or process.stdout).strip()[-2000:]
            raise ValueError(f"Assigned CUDA device query exited {process.returncode}: {detail}")
        devices = json.loads(process.stdout)
    if len(devices) != expected_count:
        raise ValueError("Assigned CUDA identity response has wrong device count")
    return [
        {
            **device,
            "cuda_ordinal": ordinal,
            "visible_identifier": visible[ordinal],
            "assigned_identifier": device["uuid"],
            "slurm_step_gpus": env.get("SLURM_STEP_GPUS"),
            "slurm_job_gpus": env.get("SLURM_JOB_GPUS"),
        }
        for ordinal, device in enumerate(devices)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", required=True, type=int)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    print(json.dumps(cuda_visible_devices(args.count)))


if __name__ == "__main__":
    main()

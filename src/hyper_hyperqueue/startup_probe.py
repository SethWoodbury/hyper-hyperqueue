"""Optional bounded application check before a pooled worker accepts any tasks."""

from __future__ import annotations

import contextlib
import os
import selectors
import signal
import subprocess
import time

OUTPUT_TAIL_BYTES = 4096


class StartupProbeError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(f"Worker startup probe failed: {result['reason']}; "
                         f"returncode={result.get('returncode')}; {result.get('error', '')}")


def run_startup_probe(profile):
    """Run argv without a shell; drain pipes into bounded tails, never log full output.

    The probe inherits the worker's allocation environment. It must be a foreground
    check, not a daemon launcher: any remaining process-group children are killed.
    An empty argv is disabled and changes no worker behavior.
    """
    if not profile.startup_probe_argv:
        return None
    result = {"argv": profile.startup_probe_argv, "timeout_seconds": profile.startup_probe_timeout_seconds,
              "started": time.time(), "status": "failed", "reason": "startup_error"}
    tails, counts, previous = {"stdout": b"", "stderr": b""}, {"stdout": 0, "stderr": 0}, {}
    process = None
    started = time.monotonic()

    def interrupted(signum, _frame):
        result.update(reason="signal", signal=signum)
        raise RuntimeError(f"Startup probe interrupted by signal {signum}")

    try:
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR1):
            previous[signum] = signal.signal(signum, interrupted)
        process = subprocess.Popen(profile.startup_probe_argv, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map() or process.poll() is None:
                remaining = profile.startup_probe_timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    result["reason"] = "timeout"
                    break
                for key, _ in selector.select(min(remaining, 0.1)):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    counts[key.data] += len(chunk)
                    tails[key.data] = (tails[key.data] + chunk)[-OUTPUT_TAIL_BYTES:]
            else:
                result.update(returncode=process.wait(), reason="exit")
                if result["returncode"] == 0:
                    result["status"] = "passed"
    except (OSError, ValueError, RuntimeError) as error:
        result["error"] = f"{type(error).__name__}: {error}"[-OUTPUT_TAIL_BYTES:]
    finally:
        if process is not None:
            # Also remove descendants left behind by a probe that exited early.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            try:
                result["returncode"] = process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                result.update(status="failed", reason="cleanup_timeout")
            for stream in (process.stdout, process.stderr):
                stream.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    result.update(finished=time.time(), elapsed_seconds=time.monotonic() - started)
    for stream, tail in tails.items():
        result[f"{stream}_tail"] = tail.decode("utf-8", errors="replace")
        result[f"{stream}_bytes"] = counts[stream]
        result[f"{stream}_truncated"] = counts[stream] > OUTPUT_TAIL_BYTES
    if result["status"] != "passed":
        raise StartupProbeError(result)
    return result

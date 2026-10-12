import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from hyper_hyperqueue.ingest import ingest_file
from hyper_hyperqueue.metrics import measurements
from hyper_hyperqueue.models import Config
from hyper_hyperqueue.runner import run_task
from hyper_hyperqueue.state import StateStore
from hyper_hyperqueue.ui import detail_page, make_server, tail


def populated(tmp_path):
    source = tmp_path / "commands"
    source.write_text("printf '<private>quoted</private>'\nexit 7\n")
    store = StateStore(tmp_path / "state")
    campaign = ingest_file(store, source, mode="direct", cwd=tmp_path)
    for tid in campaign["task_ids"]:
        run_task(store.root, campaign["campaign_id"], tid)
    return Config(state_dir=str(store.root)), store, campaign


def test_measurements_and_detail_pages_use_real_attempts(tmp_path):
    config, store, campaign = populated(tmp_path)
    report = measurements(config, minimum_samples=1)
    assert report["attempt_outcomes"] == {"succeeded": 1, "failed": 1}
    assert report["groups"][0]["samples"] == 1
    assert report["groups"][0]["suggested_runtime_seconds"] >= 60
    tid = campaign["task_ids"][0]
    detail = detail_page("/task/" + tid, {}, config)
    assert "stdout" in detail and "&lt;private&gt;" in detail
    attempt = store.attempts(campaign["campaign_id"], tid)[0]
    content = detail_page(f"/log/{tid}/{attempt['attempt_id']}/stdout", {}, config)
    assert "&lt;private&gt;quoted&lt;/private&gt;" in content
    with pytest.raises(ValueError, match="Unknown attempt"):
        detail_page(f"/log/{tid}/../stdout", {}, config)
    big = tmp_path / "biglog"
    big.write_bytes(b"x" * 1000000)
    assert len(tail(big, max_bytes=1000)) == 1000


def test_dashboard_rejects_cross_site_reads_and_mutations(tmp_path):
    config, _, _ = populated(tmp_path)
    server = make_server(config, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/status", timeout=5) as response:
            assert json.load(response)["campaigns"][0]["tasks"] == 2
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base, headers={"Host": "untrusted.example"}), timeout=5)
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/action", data=b"action=submit&csrf=wrong"), timeout=5)
        assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_remote_bind_requires_explicit_transport(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        make_server(Config(state_dir=str(tmp_path)), host="0.0.0.0")

"""Per-launch session authentication and bounded, non-blocking detail reads."""

import json
import os
import threading
from http.cookiejar import CookieJar
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, HTTPRedirectHandler, Request, build_opener

import pytest
from test_ui_dashboard import SESSION, hosted, request

from hyper_hyperqueue import api, ui
from hyper_hyperqueue.dashboard import DashboardSnapshot
from hyper_hyperqueue.ingest import ingest_file
from hyper_hyperqueue.models import Config, Profile
from hyper_hyperqueue.state import StateStore, atomic_json

ROUTES = (
    "/",
    "/assets/dashboard.js",
    "/assets/dashboard.css",
    "/api/dashboard",
    "/api/status",
    "/submit",
    "/events",
    "/pools",
    "/allocations",
    "/metrics",
    "/campaign/missing",
    "/task/missing",
    "/script/missing",
    "/log/missing/missing/stdout",
)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def raw(base, path, headers=None):
    try:
        response = build_opener(NoRedirect).open(Request(base + path, headers=headers or {}), timeout=5)
    except HTTPError as error:
        response = error
    with response:
        return response.status, response.headers, response.read().decode()


def campaign(tmp_path):
    source = tmp_path / "commands.txt"
    source.write_text("printf 'private command'\n")
    store = StateStore(tmp_path / "state")
    record = ingest_file(store, source, mode="direct", name="private-name", cwd=tmp_path)
    return Config(state_dir=str(store.root), profiles={"cpu": Profile(partition="cpu")}), store, record


@pytest.mark.parametrize("read_only", [True, False])
def test_every_route_requires_the_session(tmp_path, read_only):
    config, _, _ = campaign(tmp_path)
    with hosted(config, read_only=read_only) as base:
        for path in ROUTES:
            status, _, content = request(base, path, auth=False)
            assert status == 401, path
            assert "private-name" not in content and SESSION not in content
        for headers in ({"Cookie": "hhq_viewer_1=wrong"}, {"Authorization": "Bearer wrong"}):
            assert request(base, "/api/dashboard", auth=False, headers=headers)[0] == 401
        assert request(base, "/?token=wrong", auth=False)[0] == 401
        assert request(base, "/api/dashboard")[0] == 200


def test_unauthenticated_post_never_reaches_submission(tmp_path, monkeypatch):
    config, _, _ = campaign(tmp_path)
    monkeypatch.setattr(api, "submit", lambda *a, **k: pytest.fail("unauthenticated submit"))
    monkeypatch.setattr(api, "plan", lambda *a, **k: pytest.fail("unauthenticated plan"))
    with hosted(config, read_only=False) as base:
        status, _, content = request(
            base, "/action", form={"action": "submit", "csrf": "x"}, auth=False
        )
        assert status == 401 and 'name="csrf"' not in content


def test_bootstrap_sets_strict_httponly_cookie_and_redirects_to_clean_url(tmp_path):
    config, _, _ = campaign(tmp_path)
    with hosted(config) as base:
        status, headers, _ = raw(base, f"/metrics?token={SESSION}&page=2")
        assert status == 303
        assert headers["Location"] == "/metrics?page=2"
        assert headers["Referrer-Policy"] == "no-referrer"
        cookie = headers["Set-Cookie"]
        assert "HttpOnly" in cookie and "SameSite=Strict" in cookie and "Path=/" in cookie
        for hostile in ("//evil.example/", "/\\evil.example", "///evil.example"):
            status, headers, _ = raw(base, f"{hostile}?token={SESSION}")
            location = headers["Location"]
            assert status == 303 and location.startswith("/"), hostile
            assert not location.startswith("//") and "\\" not in location[:2], hostile
        pair = cookie.split(";", 1)[0]
        status, headers, content = raw(base, "/api/dashboard", {"Cookie": pair})
        assert status == 200 and SESSION not in content
        assert headers["Referrer-Policy"] == "no-referrer"
        for asset in ("dashboard.html", "dashboard.js", "dashboard.css"):
            assert SESSION not in ui.asset(asset)


def test_default_server_is_protected_with_random_per_launch_secret(tmp_path):
    config = Config(state_dir=str(tmp_path / "missing"))
    first = ui.make_server(config, port=0)
    second = ui.make_server(config, port=0)
    try:
        assert first.session_token != second.session_token and len(first.session_token) >= 32
        url = ui.bootstrap_url(first)
        assert url.startswith(f"http://127.0.0.1:{first.server_port}/?token=")
    finally:
        first.server_close()
        second.server_close()


def test_two_forwarded_viewers_keep_independent_browser_sessions(tmp_path):
    """Two remote default-port viewers can be forwarded to different local ports."""
    config = Config(state_dir=str(tmp_path / "missing"))
    servers = [ui.make_server(config, port=0) for _ in range(2)]
    bases = [f"http://127.0.0.1:{server.server_port}" for server in servers]
    threads = []
    try:
        for server in servers:
            # The sockets keep distinct local ports; both remote viewers report
            # their default internal port, as with separate SSH forwards.
            server.server_port = 8765
            thread = threading.Thread(target=lambda s=server: s.serve_forever(poll_interval=0.02), daemon=True)
            thread.start()
            threads.append(thread)
        jar = CookieJar()
        browser = build_opener(HTTPCookieProcessor(jar))
        for base, server in zip(bases, servers):
            with browser.open(f"{base}/?token={server.session_token}", timeout=5) as response:
                assert response.status == 200
        for base in bases:
            with browser.open(f"{base}/api/dashboard", timeout=5) as response:
                assert response.status == 200
        assert len(jar) == 2
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()


def test_viewer_prints_host_state_and_private_access_instructions(tmp_path, monkeypatch, capsys):
    config = Config(state_dir=str(tmp_path / "state"))
    server = ui.make_server(config, port=0)
    monkeypatch.setattr(ui, "make_server", lambda *args, **kwargs: server)
    monkeypatch.setattr(server, "serve_forever", lambda: None)
    monkeypatch.setattr(ui.socket, "gethostname", lambda: "compute-example")
    monkeypatch.setattr(ui.getpass, "getuser", lambda: "example-user")
    ui.serve(config)
    printed = capsys.readouterr()
    assert printed.out == ""
    assert "Viewer host: compute-example | mode: read-only" in printed.err
    assert str(tmp_path / "state") in printed.err
    assert ui.bootstrap_url(server) in printed.err
    assert "example-user@compute-example" in printed.err
    assert "if SSH to this host is permitted" in printed.err


def test_fifo_log_fails_without_blocking(tmp_path):
    config, store, record = campaign(tmp_path)
    cid, tid = record["campaign_id"], record["task_ids"][0]
    attempt_dir = store.campaign_dir(cid) / "attempts" / tid / "000001_fifo"
    attempt_dir.mkdir(parents=True)
    atomic_json(
        attempt_dir / "attempt.json",
        {"attempt_id": "000001_fifo", "task_id": tid, "campaign_id": cid, "attempt_dir": str(attempt_dir)},
    )
    os.mkfifo(attempt_dir / "stdout")
    outcome = []

    def read():
        try:
            ui.detail_page(f"/log/{tid}/000001_fifo/stdout", {}, config)
        except ValueError as error:
            outcome.append(str(error))

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    thread.join(5)
    assert not thread.is_alive(), "reading a FIFO log blocked"
    assert outcome == ["not_a_regular_file"]


def test_detail_pages_escape_and_label_unfinished(tmp_path):
    config, store, record = campaign(tmp_path)
    cid, tid = record["campaign_id"], record["task_ids"][0]
    spec = store.task(cid, tid)
    from hyper_hyperqueue.state import fingerprint

    attempt_dir = store.campaign_dir(cid) / "attempts" / tid / "000001_open"
    atomic_json(
        attempt_dir / "attempt.json",
        {
            "attempt_id": "000001_open",
            "task_id": tid,
            "campaign_id": cid,
            "attempt_dir": str(attempt_dir),
            "task_fingerprint": fingerprint(spec.model_dump(mode="json")),
        },
    )
    page = ui.detail_page("/campaign/" + cid, {}, config)
    assert "unfinished (no result; liveness not probed)" in page
    assert "running" not in page and "failed" not in page
    assert f'href="/task/{tid}"' in page
    task_page = ui.detail_page("/task/" + tid, {}, config)
    assert "unfinished" in task_page
    with pytest.raises(ValueError, match="Unknown task"):
        ui.detail_page("/task/%3Cscript%3E", {}, config)


def test_bad_controller_plan_is_a_diagnostic_not_a_crash(tmp_path):
    config, store, _ = campaign(tmp_path)
    atomic_json(store.root / "controller_status.json", {"state": "healthy", "plan": "garbage"})
    data = DashboardSnapshot(config).get()
    assert data["pending_reserves"]["budgets"] == {}
    assert {"path": "controller_status.json", "kind": "invalid_plan"} in data["diagnostics"]["errors"]
    with hosted(config) as base:
        status, _, content = request(base, "/api/dashboard")
        assert status == 200 and json.loads(content)["diagnostics"]["partial"]
        assert request(base, "/pools")[0] == 200

"""HTTP and offline dashboard boundaries without scheduler or model execution."""

import json
import re
import threading
from contextlib import contextmanager
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest
from typer.testing import CliRunner

from hyper_hyperqueue import api, ui
from hyper_hyperqueue.backends.slurm import SlurmClient
from hyper_hyperqueue.cli import app
from hyper_hyperqueue.ingest import ingest_file
from hyper_hyperqueue.models import Config, Plan, Profile
from hyper_hyperqueue.state import StateStore


@pytest.fixture(autouse=True)
def no_scheduler_or_inherited_state(monkeypatch):
    monkeypatch.delenv("HHQ_STATE_DIR", raising=False)
    monkeypatch.delenv("HHQ_CONFIG", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Viewing a dashboard must not contact or mutate Slurm")

    for method in ("query_jobs", "accounting", "submit", "cancel"):
        monkeypatch.setattr(SlurmClient, method, forbidden)


SESSION = "test-session-secret-not-for-production"


@contextmanager
def hosted(config, **kwargs):
    server = ui.make_server(config, port=0, session_token=SESSION, **kwargs)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()


def session_cookie(base):
    return f"{ui.session_cookie_name(base.rsplit(':', 1)[1], SESSION)}={SESSION}"


def request(base, path="/", *, form=None, headers=None, auth=True):
    data = urlencode(form).encode() if form is not None else None
    headers = {**({"Cookie": session_cookie(base)} if auth else {}), **(headers or {})}
    req = Request(base + path, data=data, headers=headers)
    try:
        response = urlopen(req, timeout=5)
    except HTTPError as exc:
        response = exc
    with response:
        return response.status, response.headers, response.read().decode()


def private_campaign(tmp_path):
    source = tmp_path / "commands.txt"
    source.write_text("printf 'do not run this task'\n")
    store = StateStore(tmp_path / "state")
    name = "</script><script>globalThis.dashboardInjected=true</script><b>&private</b>"
    record = ingest_file(store, source, mode="direct", name=name, cwd=tmp_path)
    config = Config(state_dir=str(store.root), profiles={"cpu": Profile(partition="cpu")})
    return config, record, name


def tree(root):
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_read_only_default_has_no_mutation_form_or_scheduler_effects(tmp_path, monkeypatch):
    config, _, _ = private_campaign(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("A read-only viewer must not invoke plan or submit")

    monkeypatch.setattr(api, "plan", forbidden)
    monkeypatch.setattr(api, "submit", forbidden)
    with hosted(config) as base:
        status, _, content = request(base, "/api/dashboard")
        assert status == 200
        assert json.loads(content)["capabilities"]["submission_enabled"] is False
        status, _, content = request(base, "/submit")
        assert status == 200 and "Read-only dashboard" in content
        assert 'name="csrf"' not in content
        status, _, _ = request(base, "/action", form={"action": "submit", "csrf": "wrong"})
        assert status == 403


def test_allow_submit_requires_session_token_and_same_origin(tmp_path, monkeypatch):
    config, _, _ = private_campaign(tmp_path)
    calls = []

    def action(name):
        def fake(commands, selected_config, **kwargs):
            calls.append((name, commands, selected_config, kwargs))
            return Plan(mode="direct", task_count=1, profiles=["cpu"], reasons=["synthetic preview"])

        return fake

    monkeypatch.setattr(api, "plan", action("plan"))
    monkeypatch.setattr(api, "submit", action("submit"))
    with hosted(config, read_only=False) as base:
        _, _, content = request(base, "/api/dashboard")
        assert json.loads(content)["capabilities"]["submission_enabled"] is True
        status, _, content = request(base, "/submit")
        assert status == 200
        token = re.search(r'name="csrf" value="([^"]+)"', content).group(1)
        fields = {
            "commands": str(tmp_path / "commands.txt"),
            "profile": "cpu",
            "mode": "direct",
            "action": "plan",
            "csrf": token,
            "cpus": "2",
            "memory_mib": "512",
            "runtime_seconds": "30",
        }
        assert request(base, "/action", form={**fields, "csrf": "wrong"})[0] == 403
        assert (
            request(base, "/action", form=fields, headers={"Origin": "https://untrusted.example"})[0] == 403
        )
        assert not calls
        assert request(base, "/action", form=fields, headers={"Origin": base})[0] == 200
        assert request(base, "/action", form={**fields, "action": "submit"})[0] == 200
    assert [call[0] for call in calls] == ["plan", "submit"]
    for _, commands, selected_config, kwargs in calls:
        assert commands == str(tmp_path / "commands.txt")
        assert selected_config is config
        resources = kwargs["resources"]
        assert (resources.cpus, resources.memory_mib, resources.runtime_seconds) == (2, 512, 30)
        assert resources.runtime_confidence == "declared"


def test_assets_are_local_allowlisted_and_csp_protected(tmp_path):
    config = Config(state_dir=str(tmp_path / "missing"))
    with hosted(config) as base:
        status, headers, content = request(base)
        assert status == 200 and "/assets/dashboard.js" in content
        csp = headers["Content-Security-Policy"]
        assert "default-src 'none'" in csp and "script-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp and "base-uri 'none'" in csp
        assert headers["Cache-Control"] == "no-store"
        for name, content_type in (("dashboard.css", "text/css"), ("dashboard.js", "text/javascript")):
            status, headers, content = request(base, "/assets/" + name)
            assert status == 200 and content
            assert headers["Content-Type"].startswith(content_type)
            assert headers["X-Content-Type-Options"] == "nosniff"
        for path in ("/assets/dashboard.html", "/assets/../ui.py", "/assets/missing.js"):
            assert request(base, path)[0] == 404
        assert request(base, headers={"Host": "untrusted.example"})[0] == 403
    with pytest.raises(ValueError, match="Unknown dashboard asset"):
        ui.asset("../ui.py")


def test_export_is_self_contained_escaped_and_never_overwrites(tmp_path):
    config, record, injection = private_campaign(tmp_path)
    destination = tmp_path / "snapshot.html"
    assert ui.export_dashboard(config, destination) == destination.resolve()
    content = destination.read_text()
    assert injection not in content
    assert "\\u003c/script>" in content
    assert '<link rel="stylesheet" href="/assets/dashboard.css">' not in content
    assert 'src="/assets/dashboard.js"' not in content
    payload = re.search(r"window\.HHQ_SNAPSHOT=(.*?);</script>", content, re.DOTALL)
    assert payload is not None
    snapshot = json.loads(payload.group(1))
    assert snapshot["capabilities"]["submission_enabled"] is False
    assert snapshot["campaigns"][0]["name"] == injection
    assert snapshot["campaigns"][0]["campaign_id"] == record["campaign_id"]
    assert destination.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        ui.export_dashboard(config, destination)


def test_cli_export_does_not_serve_or_initialize_state(tmp_path, monkeypatch):
    root = tmp_path / "missing-state"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"state_dir": str(root)}))
    destination = tmp_path / "report.html"

    def forbidden(*args, **kwargs):
        pytest.fail("HTML export must not start a server")

    monkeypatch.setattr(ui, "serve", forbidden)
    result = CliRunner().invoke(app, ["--config", str(config), "ui", "--export", str(destination)])
    assert result.exit_code == 0, result.output
    assert str(destination) in result.output and destination.is_file()
    assert not root.exists()


@pytest.mark.parametrize(
    "flags,read_only", [([], True), (["--read-only"], True), (["--allow-submit"], False)]
)
def test_cli_requires_explicit_opt_in_to_submission(tmp_path, monkeypatch, flags, read_only):
    calls = []
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"state_dir": str(tmp_path / "missing-state")}))
    monkeypatch.setattr(ui, "serve", lambda selected, **kwargs: calls.append(kwargs))
    result = CliRunner().invoke(app, ["--config", str(config), "ui", "--port", "8766", *flags])
    assert result.exit_code == 0, result.output
    assert calls == [{"host": "127.0.0.1", "port": 8766, "read_only": read_only}]


def test_missing_state_is_not_created_by_any_read_only_page(tmp_path):
    root = tmp_path / "missing-state"
    config = Config(state_dir=str(root))
    with hosted(config) as base:
        for path in (
            "/",
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
        ):
            assert request(base, path)[0] in {200, 400, 404}
            assert not root.exists(), path


def test_populated_read_only_pages_do_not_write_locks_or_state(tmp_path):
    config, record, injection = private_campaign(tmp_path)
    root = tmp_path / "state"
    before = tree(root)
    with hosted(config) as base:
        for path in (
            "/api/dashboard",
            "/events",
            "/pools",
            "/allocations",
            "/metrics",
            "/campaign/" + record["campaign_id"],
            "/task/" + record["task_ids"][0],
        ):
            status, _, content = request(base, path)
            assert status == 200, content
            assert injection not in content or path == "/api/dashboard"
            assert tree(root) == before, path

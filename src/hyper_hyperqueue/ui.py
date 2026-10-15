"""Small loopback-only dashboard; scheduling stays in the public API.

Every HTTP route requires a random per-launch session secret. The printed
bootstrap URL sets an HttpOnly, SameSite=Strict cookie and redirects to a clean
URL. Submission additionally needs a separate CSRF token.
"""

from __future__ import annotations

import getpass
import hashlib
import html
import itertools
import json
import os
import re
import secrets
import shlex
import socket
import sys
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse

from . import api

DETAIL_ATTEMPTS = 50
DETAIL_SCAN_LIMIT = 1000
INTENT_SCAN_LIMIT = 10000
LOOPBACK = ("127.0.0.1", "localhost")


def asset(name):
    if name not in {"dashboard.html", "dashboard.css", "dashboard.js"}:
        raise ValueError("Unknown dashboard asset")
    return files("hyper_hyperqueue").joinpath("ui_assets", name).read_text(encoding="utf-8")


def export_dashboard(config, destination):
    """Write a private (0600), standalone snapshot without starting a server or scheduler."""
    from .dashboard import DashboardSnapshot

    data = DashboardSnapshot(config).get()
    data = {**data, "capabilities": {"submission_enabled": False}}
    payload = json.dumps(data, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    document = asset("dashboard.html")
    document = re.sub(
        r'<link\b[^>]*href="/assets/dashboard\.css"[^>]*>',
        lambda _: "<style>" + asset("dashboard.css") + "</style>",
        document,
    )
    document = re.sub(
        r'<script\b[^>]*src="/assets/dashboard\.js"[^>]*>\s*</script>',
        lambda _: (
            "<script>window.HHQ_SNAPSHOT="
            + payload
            + ";</script><script>"
            + asset("dashboard.js")
            + "</script>"
        ),
        document,
    )
    destination = Path(destination).expanduser()
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(document)
    return destination.resolve()


def page_shell(body, *, title="hyper-hyperqueue"):
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{html.escape(title)}</title><link rel="stylesheet" href="/assets/dashboard.css">'
        "<style>.detail-view{max-width:1100px;margin:0 auto;padding:32px;line-height:1.65}"
        ".detail-view nav{display:flex;flex-wrap:wrap;gap:20px;margin-bottom:40px}"
        ".detail-view pre{padding:20px;background:#fff;border:1px solid #e6e8ed;border-radius:16px;"
        "white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}"
        ".detail-view input,.detail-view select,.detail-view button{padding:10px;margin:6px;"
        "border:1px solid #d8dce5;border-radius:8px;font:inherit}"
        ".detail-view label{display:block}.detail-view h1{letter-spacing:-.04em;font-size:32px}"
        '</style></head><body><main class="detail-view"><nav>'
        '<a href="/">← Overview</a><a href="/pools">Pools</a><a href="/allocations">Allocations</a>'
        '<a href="/events">Events</a><a href="/metrics">Measurements</a></nav>'
        + body
        + "</main></body></html>"
    )


def submission_form(config, csrf):
    profiles = "".join(
        f"<option>{html.escape(n)}</option>" for n in list(config.profiles) + list(config.workloads)
    )
    return f'''<h1>Submit a command file</h1>
    <p>Preview resources before submitting. Each nonempty line is one command.
    RFD4 sweeps use the notebook or <code>hhq rfd4</code> adapter.</p>
    <form method="post" action="/action"><input type="hidden" name="csrf" value="{html.escape(csrf)}">
    <label>Commands file <input name="commands" placeholder="/shared/commands.txt" required size="45"></label>
    <label>Execution <select name="mode"><option>auto</option><option>direct</option><option>pooled</option></select></label>
    <label>Profile <select name="profile">{profiles}</select></label>
    <details><summary>Task resources</summary>
    <label>CPUs <input name="cpus" type="number" min="1"></label>
    <label>Host RAM, MiB <input name="memory_mib" type="number" min="1"></label>
    <label>GPUs <input name="gpus" type="number" min="0"></label>
    <label>Expected seconds <input name="runtime_seconds" type="number" min="1"></label>
    <label>Direct max parallel <input name="max_parallel" type="number" min="1"></label></details>
    <button name="action" value="plan">Preview</button><button name="action" value="submit">Submit</button></form>'''


def tail(path, limit=80, max_bytes=131072):
    """Bound disk reads even for a single huge log line; never block on a FIFO."""
    from .dashboard import open_regular

    with open_regular(path) as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - max_bytes))
        data = stream.read(max_bytes)
    return "\n".join(data.decode("utf-8", errors="replace").splitlines()[-limit:])


def pretty(value):
    return "<pre>" + html.escape(json.dumps(value, indent=2, default=str)) + "</pre>"


def _href(*parts):
    return html.escape("/" + "/".join(quote(str(part), safe="") for part in parts))


def _text(value):
    return html.escape(str(value))


def _safe_id(value, message):
    from .state import identifier

    try:
        return identifier(unquote(value))
    except (TypeError, ValueError):
        raise ValueError(message) from None


def _bounded_paths(directory, pattern, limit):
    paths = list(itertools.islice(Path(directory).glob(pattern), limit + 1))
    return sorted(paths[:limit]), len(paths) > limit


def _locate_task(root, tid):
    matches = list(itertools.islice(root.glob(f"campaigns/*/tasks/{tid}.json"), 2))
    if len(matches) != 1:
        raise ValueError("Task ID must identify exactly one imported task")
    return matches[0].parent.parent.name


def _attempt(root, cid, tid, aid):
    from .dashboard import read_json_object

    directory = root / "campaigns" / cid / "attempts" / tid / aid
    record = read_json_object(directory / "attempt.json")
    if record.get("attempt_id") != aid or record.get("task_id") != tid:
        raise ValueError("Attempt identity mismatch")
    return record, directory


def _state_label(state):
    from .dashboard import UNFINISHED_REASON

    if state.get("state") == "unknown" and state.get("reason") == UNFINISHED_REASON:
        return "unfinished (no result; liveness not probed)"
    return f"{state.get('state', 'unknown')} {state.get('reason') or ''}".strip()


def detail_page(route, params, config, *, snapshot=None):
    """Read bounded durable records only; never create state or query the scheduler."""
    from .dashboard import DashboardSnapshot, read_json_object, read_task_state

    if route in {"/pools", "/metrics"}:
        data = (snapshot or DashboardSnapshot(config)).get()
        if route == "/metrics":
            return pretty({key: data[key] for key in ("timing", "memory", "groups", "diagnostics")})
        return pretty(
            {
                "budgets": {n: b.model_dump() for n, b in config.budgets.items()},
                "profiles": {n: p.model_dump() for n, p in config.profiles.items()},
                "pending_reserves": data["pending_reserves"],
                "health": data["health"],
            }
        )
    root = Path(config.state_dir).expanduser()
    if not root.is_dir():
        return "<h1>No workspace records yet</h1><p>Import or submit tasks with the notebook or CLI.</p>"
    try:
        page = max(0, int(params.get("page", ["0"])[0]))
    except ValueError:
        raise ValueError("Invalid page number") from None
    if route.startswith("/campaign/"):
        cid = _safe_id(route.removeprefix("/campaign/"), "Unknown campaign")
        campaign = read_json_object(root / "campaigns" / cid / "campaign.json", 32 * 1024 * 1024)
        task_ids = campaign.get("task_ids")
        if campaign.get("campaign_id") != cid or not isinstance(task_ids, list):
            raise ValueError("Campaign identity mismatch")
        rows = []
        for tid in task_ids[page * 50 : (page + 1) * 50]:
            try:
                label = _state_label(read_task_state(root, cid, tid))
            except (OSError, ValueError, TypeError) as error:
                label = f"unreadable record ({type(error).__name__})"
            rows.append(f'<li><a href="{_href("task", tid)}">{_text(tid)}</a>: {_text(label)}</li>')
        more = f'<a href="?page={page + 1}">Next 50</a>' if (page + 1) * 50 < len(task_ids) else ""
        return f"<h1>{_text(campaign.get('name', cid))}</h1><ul>{''.join(rows)}</ul>{more}"
    if route.startswith("/task/"):
        tid = _safe_id(route.removeprefix("/task/"), "Unknown task")
        cid = _locate_task(root, tid)
        paths, truncated = _bounded_paths(
            root / "campaigns" / cid / "attempts" / tid, "*/attempt.json", DETAIL_SCAN_LIMIT
        )
        attempts = []
        for path in paths[-DETAIL_ATTEMPTS:]:
            record = read_json_object(path)
            result = path.parent / "result.json"
            if result.exists():
                record["result"] = read_json_object(result)
            attempts.append(record)
        links = "".join(
            f"<li>{_text(a.get('attempt_id'))}: "
            f'<a href="{_href("log", tid, a.get("attempt_id"), "stdout")}">stdout</a> · '
            f'<a href="{_href("log", tid, a.get("attempt_id"), "stderr")}">stderr</a></li>'
            for a in attempts
        )
        intent_paths, intents_truncated = _bounded_paths(root / "intents", "*.json", DETAIL_SCAN_LIMIT)
        submissions = []
        for path in intent_paths:
            try:
                intent = read_json_object(path, 1024 * 1024)
            except (OSError, ValueError, TypeError):
                continue
            if tid in (intent.get("task_ids") or []):
                submissions.append(intent)
        state = read_task_state(root, cid, tid)
        notes = [f"Showing the latest {len(attempts)} recorded attempts."]
        if truncated:
            notes.append(f"Attempt scan stopped at {DETAIL_SCAN_LIMIT} records.")
        if intents_truncated:
            notes.append(f"Submission records were scanned up to {DETAIL_SCAN_LIMIT} files.")
        return (
            f"<h1>{_text(tid)}</h1><p>State: <strong>{_text(_state_label(state))}</strong></p>"
            f"<p>{_text(' '.join(notes))} Live HQ eligibility: <code>hhq explain {_text(tid)}</code></p>"
            f"<ul>{links}</ul>"
            + pretty(
                {
                    "task": read_json_object(root / "campaigns" / cid / "tasks" / f"{tid}.json"),
                    "state": state,
                    "attempts": attempts,
                    "submissions": submissions,
                }
            )
        )
    if route.startswith("/log/"):
        pieces = route.strip("/").split("/")
        if len(pieces) != 4 or pieces[3] not in ("stdout", "stderr"):
            raise ValueError("Unknown log")
        _, tid, aid, stream = pieces
        tid = _safe_id(tid, "Unknown task")
        aid = _safe_id(aid, "Unknown attempt")
        cid = _locate_task(root, tid)
        try:
            attempt, _ = _attempt(root, cid, tid, aid)
        except FileNotFoundError:
            raise ValueError("Unknown attempt") from None
        from .logs import attempt_log_path

        return (
            "<p>Last 80 lines, at most 128 KiB.</p><pre>"
            + html.escape(tail(attempt_log_path(attempt, stream)))
            + "</pre>"
        )
    if route == "/allocations":
        paths, truncated = _bounded_paths(root / "intents", "*.json", INTENT_SCAN_LIMIT)
        shown = paths[page * 30 : (page + 1) * 30]
        blocks = []
        for path in shown:
            try:
                intent = read_json_object(path, 1024 * 1024)
            except (OSError, ValueError, TypeError) as error:
                blocks.append(f"<p>{_text(path.name)}: unreadable ({_text(type(error).__name__)})</p>")
                continue
            summary = (
                f"{intent.get('backend')} {intent.get('job_id', intent.get('hq_job_id', '?'))}"
                f" · {intent.get('state')} · {intent.get('profile', '')}"
            )
            script = (
                f'<a href="{_href("script", path.stem)}">Rendered script</a>' if intent.get("script") else ""
            )
            blocks.append(f"<details><summary>{_text(summary)}</summary>{script}{pretty(intent)}</details>")
        total = f"{len(paths)}+" if truncated else str(len(paths))
        more = f'<p><a href="?page={page + 1}">Next 30</a></p>' if (page + 1) * 30 < len(paths) else ""
        return (
            f"<h1>Allocation records</h1><p>Saved submission intents {page * 30 + 1 if shown else 0}–"
            f"{page * 30 + len(shown)} of {total}; not a live Slurm queue.</p>" + "".join(blocks) + more
        )
    if route.startswith("/script/"):
        key = _safe_id(route.removeprefix("/script/"), "Unknown owned script")
        try:
            intent = read_json_object(root / "intents" / f"{key}.json", 1024 * 1024)
        except FileNotFoundError:
            raise ValueError("Unknown owned script") from None
        if not intent.get("script"):
            raise ValueError("Unknown owned script")
        return "<pre>" + html.escape(tail(intent["script"], 300)) + "</pre>"
    if route == "/events":
        path = root / "events.jsonl"
        return (
            "<p>Last 80 events, at most 128 KiB. These contain private campaign metadata.</p><pre>"
            + html.escape(tail(path) if path.exists() else "No events yet.")
            + "</pre>"
        )
    return None


def session_cookie_name(port, session_token):
    # Browser cookies are not port-scoped. Separate remote viewers may share
    # their default internal port while using different local SSH forwards.
    launch_id = hashlib.sha256(session_token.encode()).hexdigest()[:24]
    return f"hhq_viewer_{port}_{launch_id}"


def bootstrap_url(server, host="127.0.0.1"):
    """The private URL that opens a browser session; treat it like a password."""
    return f"http://{host}:{server.server_port}/?{urlencode({'token': server.session_token})}"


def make_server(config, *, host="127.0.0.1", port=8765, read_only=True, session_token=None):
    """Create a loopback server whose every route requires the per-launch session secret."""
    if host not in LOOPBACK:
        raise ValueError("Dashboard binds loopback only. Use SSH forwarding for cluster access.")
    from .dashboard import DashboardSnapshot, DashboardUnavailable

    snapshot = DashboardSnapshot(config)
    session = session_token or secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(32)

    def matches(candidate, secret):
        return isinstance(candidate, str) and secrets.compare_digest(
            candidate.encode("utf-8", "surrogateescape"), secret.encode()
        )

    class Handler(BaseHTTPRequestHandler):
        def respond(self, text, code=200, content_type="text/html", headers=()):
            data = text.encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'self'; connect-src 'self'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                "form-action 'self'; frame-ancestors 'none'; base-uri 'none'",
            )
            for name, value in headers:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        def host_allowed(self):
            return urlparse("http://" + self.headers.get("Host", "")).hostname in LOOPBACK

        def authenticated(self):
            scheme, _, value = self.headers.get("Authorization", "").partition(" ")
            if scheme.lower() == "bearer" and matches(value.strip(), session):
                return True
            try:
                jar = SimpleCookie(self.headers.get("Cookie", ""))
            except CookieError:
                return False
            morsel = jar.get(session_cookie_name(self.server.server_port, session))
            return morsel is not None and matches(morsel.value, session)

        def unauthorized(self):
            return self.respond(
                page_shell(
                    "<h1>Session required</h1><p>Open the private URL printed when this viewer "
                    "started. Restarting the viewer issues a new one.</p>"
                ),
                401,
            )

        def do_GET(self):
            if not self.host_allowed():
                return self.respond("Host rejected", 403)
            route = urlparse(self.path)
            query = parse_qs(route.query, keep_blank_values=True)
            if "token" in query:
                if len(query["token"]) != 1 or not matches(query["token"][0], session):
                    return self.unauthorized()
                rest = urlencode([(k, v) for k, vs in query.items() if k != "token" for v in vs], doseq=True)
                cookie = (
                    f"{session_cookie_name(self.server.server_port, session)}={session}; "
                    "HttpOnly; SameSite=Strict; Path=/"
                )
                # Always a same-origin path; "//host" would be a cross-site redirect.
                location = "/" + route.path.lstrip("/\\") +("?" + rest if rest else "")
                return self.respond(
                    "", 303, headers=(("Location", location), ("Set-Cookie", cookie))
                )
            if not self.authenticated():
                return self.unauthorized()
            try:
                if route.path == "/":
                    return self.respond(asset("dashboard.html"))
                if route.path.startswith("/assets/"):
                    name = route.path.removeprefix("/assets/")
                    content_types = {"dashboard.css": "text/css", "dashboard.js": "text/javascript"}
                    if name not in content_types:
                        return self.respond("Not found", 404)
                    return self.respond(asset(name), content_type=content_types[name])
                if route.path in {"/api/dashboard", "/api/status"}:
                    try:
                        data = {**snapshot.get(), "capabilities": {"submission_enabled": not read_only}}
                    except DashboardUnavailable as error:
                        body = json.dumps({"error": str(error)})
                        return self.respond(body, 503, content_type="application/json")
                    if route.path == "/api/status":
                        data = {
                            "schema_version": 1,
                            "campaigns": [{**c, "tasks": c["total"]} for c in data["campaigns"]],
                            "controller": data["health"]["controller"],
                            "state_dir": str(Path(config.state_dir).expanduser()),
                            "note": "Cached records only; unfinished attempt liveness is not probed.",
                        }
                    return self.respond(json.dumps(data, allow_nan=False), content_type="application/json")
                if route.path == "/submit":
                    if read_only:
                        return self.respond(
                            page_shell(
                                "<h1>Read-only dashboard</h1><p>Use the notebook or CLI to submit work. "
                                "To enable the command-file form, start this viewer with <code>--allow-submit</code>.</p>"
                            )
                        )
                    return self.respond(page_shell(submission_form(config, csrf), title="Submit · HHQ"))
                detail = detail_page(route.path, parse_qs(route.query), config, snapshot=snapshot)
                if detail is None:
                    return self.respond("Not found", 404)
                return self.respond(page_shell(detail))
            except DashboardUnavailable as exc:
                return self.respond(page_shell("<h1>Records unavailable</h1><p>" + _text(exc) + "</p>"), 503)
            except (ValueError, KeyError, OSError, TypeError) as exc:
                return self.respond(
                    page_shell("<h1>Unable to read this view</h1><p>" + _text(exc) + "</p>"), 400
                )
            except Exception as exc:  # noqa: BLE001 -- keep the viewer up; show the kind only
                return self.respond(
                    page_shell("<h1>Unable to read this view</h1><p>" + _text(type(exc).__name__) + "</p>"),
                    500,
                )

        def do_POST(self):
            if not self.host_allowed():
                return self.respond("Host rejected", 403)
            if not self.authenticated():
                return self.unauthorized()
            if self.path != "/action":
                return self.respond("Not found", 404)
            if read_only:
                return self.respond("Read-only dashboard; use the CLI or start with --allow-submit.", 403)
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self.respond("Origin rejected", 403)
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self.respond("Invalid request length", 400)
            if size < 0 or size > 16384:
                return self.respond("Request too large", 413)
            form = {k: v[0] for k, v in parse_qs(self.rfile.read(size).decode()).items()}
            if not matches(form.get("csrf", ""), csrf):
                return self.respond("Token rejected", 403)
            try:
                if form.get("action") not in ("plan", "submit"):
                    raise ValueError("Unknown action")
                from .config import profile_names
                from .models import Resources

                resources = api.default_resources(config, profile_names(config, form["profile"])).model_dump()
                for key in ("cpus", "memory_mib", "gpus", "runtime_seconds"):
                    if form.get(key):
                        resources[key] = int(form[key])
                if form.get("runtime_seconds"):
                    resources["runtime_confidence"] = "declared"
                method = api.plan if form["action"] == "plan" else api.submit
                result = method(
                    form["commands"],
                    config,
                    profile=form["profile"],
                    mode=form["mode"],
                    resources=Resources.model_validate(resources),
                    max_parallel=int(form["max_parallel"]) if form.get("max_parallel") else None,
                )
                self.respond(page_shell(pretty(result.model_dump())))
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                self.respond(html.escape(str(exc)), 400)

        def log_message(self, *args):
            pass  # Request lines can contain the bootstrap secret; never log them.

    server = ThreadingHTTPServer((host, port), Handler)
    server.session_token = session
    return server


def serve(config, *, host="127.0.0.1", port=8765, read_only=True):
    server = make_server(config, host=host, port=port, read_only=read_only)
    hostname = socket.gethostname()
    port = server.server_port
    tunnel = shlex.join([
        "ssh", "-N", "-o", "ExitOnForwardFailure=yes",
        "-L", f"127.0.0.1:{port}:127.0.0.1:{port}", f"{getpass.getuser()}@{hostname}",
    ])
    print(
        f"Viewer host: {hostname} | mode: {'read-only' if read_only else 'generic submit form enabled'}\n"
        f"State directory: {Path(config.state_dir).expanduser().resolve()}\n"
        f"Dashboard (loopback only): {bootstrap_url(server, host)}\n"
        f"From another computer, if SSH to this host is permitted: {tunnel}\n"
        "This URL contains a per-launch secret. Keep it private; it is not stored on disk by HHQ.",
        flush=True, file=sys.stderr,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()

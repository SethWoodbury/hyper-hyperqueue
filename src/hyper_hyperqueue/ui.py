"""Small loopback-only dashboard; scheduling stays in the public API."""

from __future__ import annotations

import html
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import api
from .state import StateStore


def tail(path, limit=80, max_bytes=131072):
    """Bound disk reads even for a single huge log line."""
    from pathlib import Path

    with Path(path).open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - max_bytes))
        data = stream.read(max_bytes)
    return "\n".join(data.decode("utf-8", errors="replace").splitlines()[-limit:])


def pretty(value):
    return "<pre>" + html.escape(json.dumps(value, indent=2, default=str)) + "</pre>"


def detail_page(route, params, config):
    """Read durable records only; refreshing pages never queries the scheduler."""
    store = StateStore(config.state_dir)
    page = max(0, int(params.get("page", ["0"])[0]))
    if route.startswith("/campaign/"):
        cid = route.removeprefix("/campaign/")
        campaign = store.campaign(cid)
        rows = [
            (tid, store.task_state(cid, tid)) for tid in campaign["task_ids"][page * 50 : (page + 1) * 50]
        ]
        return (
            "<h1>"
            + html.escape(campaign["name"])
            + "</h1><ul>"
            + "".join(
                f'<li><a href="/task/{tid}">{tid}</a>: {html.escape(s.get("state", "ready"))} {html.escape(str(s.get("reason", "")))}</li>'
                for tid, s in rows
            )
            + f'</ul><a href="?page={page + 1}">Next 50</a>'
        )
    if route.startswith("/task/"):
        tid = route.removeprefix("/task/")
        cid = api.find_task(store, tid)
        attempts = store.attempts(cid, tid)
        links = "".join(
            f'<li>{html.escape(a["attempt_id"])}: <a href="/log/{tid}/{a["attempt_id"]}/stdout">stdout</a> · <a href="/log/{tid}/{a["attempt_id"]}/stderr">stderr</a></li>'
            for a in attempts
        )
        return (
            "<p>Live HQ eligibility: <code>hhq explain "
            + tid
            + "</code></p><ul>"
            + links
            + "</ul>"
            + pretty(
                {
                    "task": store.task(cid, tid).model_dump(),
                    "state": store.task_state(cid, tid),
                    "attempts": attempts,
                    "submissions": [i for i in store.intents() if tid in i.get("task_ids", [])],
                }
            )
        )
    if route.startswith("/log/"):
        pieces = route.strip("/").split("/")
        if len(pieces) != 4 or pieces[3] not in ("stdout", "stderr"):
            raise ValueError("Unknown log")
        _, tid, aid, stream = pieces
        cid = api.find_task(store, tid)
        if not any(a["attempt_id"] == aid for a in store.attempts(cid, tid)):
            raise ValueError("Unknown attempt")
        return (
            "<p>Last 80 lines, at most 128 KiB.</p><pre>"
            + html.escape(tail(store.campaign_dir(cid) / "attempts" / tid / aid / stream))
            + "</pre>"
        )
    if route == "/allocations":
        rows = store.intents()[page * 30 : (page + 1) * 30]
        return (
            "".join(
                "<details><summary>"
                + html.escape(
                    f"{i.get('backend')} {i.get('job_id', i.get('hq_job_id', '?'))} · {i.get('state')} · {i.get('profile', '')}"
                )
                + "</summary>"
                + (f'<a href="/script/{i["token"]}">Rendered script</a>' if i.get("script") else "")
                + pretty(i)
                + "</details>"
                for i in rows
            )
            + f'<p><a href="?page={page + 1}">Next 30</a></p>'
        )
    if route.startswith("/script/"):
        token = route.removeprefix("/script/")
        matches = [i for i in store.intents() if i.get("token") == token and i.get("script")]
        if len(matches) != 1:
            raise ValueError("Unknown owned script")
        return "<pre>" + html.escape(tail(matches[0]["script"], 300)) + "</pre>"
    if route == "/pools":
        return pretty(
            {
                "budgets": {n: b.model_dump() for n, b in config.budgets.items()},
                "profiles": {n: p.model_dump() for n, p in config.profiles.items()},
                "controller": api.status(config)["controller"],
            }
        )
    if route == "/events":
        path = store.root / "events.jsonl"
        return (
            "<p>Last 80 events, at most 128 KiB. These contain private campaign metadata.</p><pre>"
            + html.escape(tail(path) if path.exists() else "No events yet.")
            + "</pre>"
        )
    if route == "/metrics":
        from .metrics import measurements

        return pretty(measurements(config))
    return None


def make_server(config, *, host="127.0.0.1", port=8765):
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("Dashboard binds loopback only. Use SSH forwarding for cluster access.")
    csrf = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, text, code=200, content_type="text/html"):
            data = text.encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if urlparse("http://" + self.headers.get("Host", "")).hostname not in ("127.0.0.1", "localhost"):
                return self.respond("Host rejected", 403)
            route = urlparse(self.path)
            if route.path == "/api/status":
                return self.respond(json.dumps(api.status(config)), content_type="application/json")
            if route.path != "/":
                try:
                    detail = detail_page(route.path, parse_qs(route.query), config)
                    if detail is None:
                        return self.respond("Not found", 404)
                    return self.respond(
                        '<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:2rem}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><nav><a href="/">Campaigns</a> · <a href="/pools">Pools</a> · <a href="/allocations">Allocations</a> · <a href="/events">Events</a> · <a href="/metrics">Measurements</a></nav>'
                        + detail
                    )
                except (ValueError, KeyError, FileNotFoundError) as exc:
                    return self.respond(html.escape(str(exc)), 400)
            data = api.status(config)
            params = parse_qs(route.query)
            try:
                page = max(0, int(params.get("page", ["0"])[0]))
            except ValueError:
                return self.respond("Page must be an integer", 400)
            rows = data["campaigns"][page * 50 : (page + 1) * 50]
            table = "".join(
                '<tr><td><a href="/campaign/'
                + c["campaign_id"]
                + '">'
                + c["campaign_id"]
                + "</a></td>"
                + "".join(f"<td>{html.escape(str(c[k]))}</td>" for k in ["name", "mode", "tasks", "counts"])
                + "</tr>"
                for c in rows
            )
            profiles = "".join(
                f"<option>{html.escape(n)}</option>" for n in list(config.profiles) + list(config.workloads)
            )
            self.respond(f'''<!doctype html><html><title>hyper-hyperqueue</title><style>body{{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:1rem}}td,th{{padding:.6rem;text-align:left;border-bottom:1px solid #ddd}}input,select,button{{padding:.5rem;margin:.3rem}}pre{{white-space:pre-wrap}}</style>
            <h1>hyper-hyperqueue</h1><nav><a href="/pools">Pools</a> · <a href="/allocations">Allocations</a> · <a href="/events">Events</a> · <a href="/metrics">Measurements</a></nav><p>Refresh to update. Controller timestamps show snapshot age; refresh does not query Slurm. Commands remain private on this host.</p>
            <form method="post" action="/action"><input type="hidden" name="csrf" value="{csrf}">
            <input name="commands" placeholder="Absolute commands-file path" required size="45"><select name="mode"><option>auto</option><option>direct</option><option>pooled</option></select><select name="profile">{profiles}</select>
            <details><summary>Task resources (optional; must fit selected allocation)</summary><label>CPUs <input name="cpus" type="number" min="1"></label><label>RAM MiB <input name="memory_mib" type="number" min="1"></label><label>GPUs <input name="gpus" type="number" min="0"></label><label>Expected seconds <input name="runtime_seconds" type="number" min="1"></label><label>Direct max parallel <input name="max_parallel" type="number" min="1"></label></details>
            <button name="action" value="plan">Preview</button><button name="action" value="submit">Submit</button></form>
            <table><tr><th>Campaign</th><th>Name</th><th>Mode</th><th>Tasks</th><th>States</th></tr>{table}</table><p><a href="/?page={page + 1}">Next page</a></p><pre>{html.escape(json.dumps(data["controller"], indent=2))}</pre><p>{html.escape(data["note"])}</p></html>''')

        def do_POST(self):
            if urlparse("http://" + self.headers.get("Host", "")).hostname not in ("127.0.0.1", "localhost"):
                return self.respond("Host rejected", 403)
            if self.path != "/action":
                return self.respond("Not found", 404)
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
            if not secrets.compare_digest(form.get("csrf", ""), csrf):
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
                self.respond(
                    '<a href="/">Dashboard</a><pre>'
                    + html.escape(result.model_dump_json(indent=2))
                    + "</pre>"
                )
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                self.respond(html.escape(str(exc)), 400)

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve(config, *, host="127.0.0.1", port=8765):
    server = make_server(config, host=host, port=port)
    print(f"Dashboard: http://{host}:{server.server_port} (loopback only)", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()

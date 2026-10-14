/* Read-only dashboard. All record-derived content is inserted as text or SVG attributes. */
function initializeDashboard() {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const saved = Object.prototype.hasOwnProperty.call(window, "HHQ_SNAPSHOT");
  const integer = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });
  let snapshot = null, visibleCampaigns = 15, timer = null, pending = null, lastError = null;
  let flowState = null, motionEnabled = true, freshnessTimer = null;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const number = (v) => typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : null;
  const count = (v) => number(v) === null ? "—" : integer.format(v);
  const entries = (value) => value && typeof value === "object" && !Array.isArray(value) ? Object.entries(value) : [];
  const list = (value) => Array.isArray(value) ? value : [];
  const stateCount = (counts, key) => number(counts && counts[key]) ?? 0;
  const sumStates = (counts, keys) => keys.reduce((sum, key) => sum + stateCount(counts, key), 0);
  const attentionKeys = ["failed", "blocked", "unknown", "interrupted"];
  // Unfinished attempts keep backend state "unknown" but are shown neutrally:
  // no result yet, liveness not probed, neither running nor failed.
  const unfinishedOf = (c) => number(c?.unfinished) ?? 0;
  const attentionOf = (c) => Math.max(0, sumStates(c?.counts, attentionKeys) - unfinishedOf(c));
  function duration(value) {
    const v = number(value);
    if (v === null) return "—";
    if (v < 60) return `${v < 10 ? v.toFixed(1) : Math.round(v)}s`;
    if (v < 3600) return `${(v / 60).toFixed(v < 600 ? 1 : 0)}m`;
    if (v < 86400) return `${(v / 3600).toFixed(1)}h`;
    return `${(v / 86400).toFixed(1)}d`;
  }
  function memory(value) {
    const v = number(value);
    if (v === null) return "—";
    return v >= 1024 ? `${(v / 1024).toFixed(1)} GiB` : `${Math.round(v)} MiB`;
  }
  function date(value) {
    if (value === null || value === undefined || value === "") return null;
    const d = new Date(typeof value === "number" ? value * 1000 : value);
    return Number.isFinite(d.getTime()) ? d : null;
  }
  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
  }
  function link(kind, id, className, text) {
    const node = element("a", className, text);
    node.href = `/${kind}/${encodeURIComponent(String(id))}`;
    if (saved) disableLink(node);
    return node;
  }
  function disableLink(node) {
    node.removeAttribute("href");
    node.setAttribute("aria-disabled", "true");
    node.tabIndex = -1;
    node.title = "Open the live dashboard for details";
  }
  function put(id, value) { $(id).textContent = value; }
  function statusBadge(label, state) {
    put("connection-label", label);
    $("connection-badge").className = `connection-badge${state ? ` is-${state}` : ""}`;
  }
  function snapshotTiming() {
    const stamp = date(snapshot?.generated_at);
    const elapsed = stamp ? (Date.now() - stamp.getTime()) / 1000 : null;
    const limit = Math.max(30, (number(snapshot?.cache?.ttl_seconds) ?? 10) * 3);
    return { stamp, age: elapsed === null ? null : Math.max(0, elapsed), limit, stale: elapsed === null || elapsed < -30 || elapsed > limit };
  }
  function syncFlowMotion() {
    const panel = $("flow-panel"), button = $("flow-motion");
    if (!panel || !button) return;
    const timing = snapshotTiming(), offline = navigator.onLine === false;
    const staleHealth = ["controller", "service"].some((key) => snapshot?.health?.[key]?.stale === true);
    const live = Boolean(snapshot && !saved && !lastError && !offline && !document.hidden && !timing.stale && !staleHealth);
    const paths = { ready: flowState?.admitted ?? 0, admitted: flowState?.running ?? 0, running: flowState?.running ?? 0, attention: 0 };
    // Populated queues alone do not imply dispatch. Historical completions and
    // failures never drive current; the paths depict lifecycle, not throughput.
    panel.querySelectorAll("[data-flow]").forEach((node) => {
      node.classList.toggle("is-active", live && (paths[node.dataset.flow] ?? 0) > 0);
    });
    const hasWork = (flowState?.admitted ?? 0) + (flowState?.running ?? 0) > 0;
    const moving = live && hasWork && motionEnabled && !reducedMotion.matches;
    panel.classList.toggle("is-moving", moving);
    button.disabled = saved || !snapshot || reducedMotion.matches;
    button.setAttribute("aria-pressed", String(motionEnabled && !saved && !reducedMotion.matches));
    button.setAttribute("aria-label", "Circuit motion");
    button.textContent = motionEnabled ? "Pause motion" : "Resume motion";
    if (saved || reducedMotion.matches) button.textContent = "Motion off";
    button.title = saved ? "Saved snapshots are static" : reducedMotion.matches ? "Your reduced-motion preference is enabled" : "Toggle schematic motion; task records are unchanged";
    const label = !snapshot ? "Waiting for records" : saved ? "Saved snapshot · static" :
      offline || lastError ? "Offline · static" : document.hidden ? "Hidden tab · static" :
      timing.stale ? "Stale or undated · static" : staleHealth ? "Coordinator status stale · static" :
      reducedMotion.matches ? "Reduced motion · static" :
      !motionEnabled ? "Motion paused" : !hasWork ? "No admitted or running records · static" : "Fresh records · schematic motion";
    put("flow-freshness", label);
    panel.dataset.motion = moving ? "moving" : "static";
  }
  function renderFlow() {
    if (!$("flow-panel")) return;
    const counts = snapshot.totals.counts || {};
    flowState = {
      ready: sumStates(counts, ["ready", "retry_wait"]), admitted: stateCount(counts, "submitted"),
      running: stateCount(counts, "running"), succeeded: stateCount(counts, "succeeded"),
      attention: attentionOf(snapshot.totals),
    };
    const unfinished = unfinishedOf(snapshot.totals);
    for (const [key, value] of Object.entries(flowState)) put(`flow-${key}`, count(key === "running" ? value + unfinished : value));
    const covered = ["ready", "retry_wait", "submitted", "running", "succeeded", "canceled", ...attentionKeys];
    const other = entries(counts).reduce((total, [key, value]) => total + (covered.includes(key) ? 0 : number(value) ?? 0), 0);
    const details = [
      "Counts from saved records; paths show lifecycle, not throughput.",
      `Ready includes ${count(stateCount(counts, "retry_wait"))} waiting for retry.`,
      `${count(stateCount(counts, "canceled"))} canceled, shown separately from attention.`,
      `Started includes ${count(unfinished)} unfinished attempts with no result; neither running nor failure is claimed.`,
      "Recorded running is not a live process probe.",
    ];
    if (other) details.push(`${count(other)} tasks have other recorded states and are outside these nodes.`);
    if (snapshot.diagnostics?.partial || snapshot.partial) details.push("Coverage is partial; see Snapshot details.");
    put("flow-note", details.join(" "));
    const description = $("flow-svg-description");
    if (description) {
      description.textContent = `${count(flowState.ready)} ready or waiting for retry; ${count(flowState.admitted)} admitted; ${count(flowState.running)} recorded running; ${count(unfinished)} unfinished; ${count(flowState.succeeded)} succeeded; ${count(flowState.attention)} need attention. ${details.join(" ")}`;
    }
    syncFlowMotion();
  }
  function freshness() {
    clearTimeout(freshnessTimer);
    syncFlowMotion();
    if (!snapshot) return;
    const { stamp, age, limit, stale } = snapshotTiming();
    put("freshness", saved ? (stamp ? `Captured ${stamp.toLocaleString()}` : "Capture time not recorded") :
      (age === null ? "Snapshot time not recorded" : `Snapshot ${age < 5 ? "just now" : `${duration(age)} ago`}`));
    $("freshness").title = stamp ? stamp.toISOString() : "";
    if (saved) statusBadge("Saved snapshot", "saved");
    else if (lastError || navigator.onLine === false) statusBadge("Connection lost", "stale");
    else if (document.hidden) statusBadge("Refresh paused", "paused");
    else if (stale) statusBadge("Stale snapshot", "stale");
    else statusBadge("Live view", "");
    // This one-shot deadline freezes stale current even while fetch is stalled.
    // Static exports and hidden tabs never schedule it.
    if (!saved && !document.hidden && !lastError && navigator.onLine !== false && !stale) {
      freshnessTimer = setTimeout(freshness, Math.min(2147483647, Math.max(100, (limit - age) * 1000 + 100)));
    }
  }
  function notices() {
    const messages = [];
    const scanError = snapshot?.cache?.error;
    if (scanError && !saved) messages.push(`The latest records scan failed (${String(scanError.kind || "error")}). Showing the previous snapshot; retrying after a short pause.`);
    if (lastError?.status === 401) messages.push("Session required. Reopen the private URL printed when the viewer started.");
    else if (lastError) messages.push(snapshot ? "Refresh failed. Showing the last available snapshot; task states may have changed." : "The workspace could not be loaded. Use Refresh to try again.");
    if (snapshot?.partial || snapshot?.diagnostics?.partial) messages.push("This snapshot is partial: some records are unreadable or a display/scan limit was reached. Open Snapshot details below for coverage.");
    const stale = ["controller", "service"].filter((key) => snapshot?.health?.[key]?.stale === true);
    if (stale.length) messages.push(`${stale.map((v) => v[0].toUpperCase() + v.slice(1)).join(" and ")} heartbeat is old. This view does not establish whether its process is still running.`);
    $("notice").hidden = !messages.length;
    put("notice", messages.join(" "));
  }
  function campaignState(c) {
    if (attentionOf(c)) return { label: "Needs attention", className: "attention" };
    if (stateCount(c.counts, "running")) return { label: "In progress", className: "running" };
    if (unfinishedOf(c)) return { label: "Unfinished", className: "" };
    if (c.total > 0 && c.completed >= c.total) return { label: "Finished", className: "finished" };
    if (c.draining) return { label: "Draining", className: "" };
    return { label: c.total ? "Waiting" : "Empty", className: "" };
  }
  function renderCampaigns() {
    if (!snapshot) return;
    const campaigns = list(snapshot.campaigns), query = $("campaign-search").value.trim().toLowerCase(), filter = $("campaign-filter").value;
    const selected = campaigns.filter((c) => {
      const state = campaignState(c);
      const matchesText = `${c.name} ${c.campaign_id}`.toLowerCase().includes(query);
      const matchesFilter = filter === "all" || filter === c.mode ||
        (filter === "attention" && state.className === "attention") ||
        (filter === "finished" && c.total > 0 && c.completed >= c.total) ||
        (filter === "active" && c.remaining > 0);
      return matchesText && matchesFilter;
    });
    const rows = selected.slice(0, visibleCampaigns).map((c) => {
      const row = element("tr"), first = element("td"), title = element("div", "campaign-name");
      title.append(element("span", "campaign-icon", String(c.mode || "?").slice(0, 1).toUpperCase()));
      const identity = element("div"), name = link("campaign", c.campaign_id, "campaign-title", c.name || c.campaign_id);
      name.title = String(c.name || c.campaign_id);
      identity.append(name, element("div", "campaign-id", c.campaign_id)); title.append(identity); first.append(title);
      const mode = element("td"); mode.append(element("span", "mode-pill", c.mode || "Unknown"));
      const progressCell = element("td", "progress-cell"), progressLabel = element("div", "progress-label");
      const fraction = number(c.completion_fraction), percent = fraction === null ? null : Math.min(100, fraction * 100);
      progressLabel.append(element("strong", "", `${count(c.completed)} / ${count(c.total)}`), element("span", "", percent === null ? "—" : `${Math.round(percent)}%`));
      const progress = element("progress"); progress.max = 100; progress.value = percent ?? 0;
      progress.setAttribute("aria-label", `${c.name || c.campaign_id}: ${count(c.completed)} of ${count(c.total)} tasks finished`);
      if (percent === null) progress.hidden = true;
      progressCell.append(progressLabel, progress);
      const state = campaignState(c), stateCell = element("td"); stateCell.append(element("span", `state-pill ${state.className}`, state.label));
      const details = element("td"), arrow = link("campaign", c.campaign_id, "row-arrow", "↗"); arrow.setAttribute("aria-label", `Open ${c.name || c.campaign_id}`); details.append(arrow);
      row.append(first, mode, progressCell, element("td", "numeric", count(stateCount(c.counts, "running") + unfinishedOf(c))), element("td", "numeric", count(c.remaining)), stateCell, details);
      return row;
    });
    $("campaign-rows").replaceChildren(...rows);
    put("campaign-count", count(campaigns.length));
    const empty = $("campaign-empty"); empty.hidden = selected.length > 0;
    if (!selected.length) {
      empty.replaceChildren(element("div", "empty-mark", "◫"), element("h3", "", campaigns.length ? "No matching campaigns" : "A clear view of what comes next"),
        element("p", "", campaigns.length ? "Try a different name or filter." : "Submit commands with the CLI, then follow their progress here."));
      if (!campaigns.length) empty.append(element("code", "", "hhq submit --help"));
    }
    put("campaign-footer", `${count(Math.min(selected.length, visibleCampaigns))} of ${count(selected.length)} campaigns · Finished includes failed and canceled tasks.`);
    $("show-more").hidden = visibleCampaigns >= selected.length;
    const total = snapshot.totals || {};
    put("campaigns-subtitle", total.unscanned_tasks > 0 ? `${count(total.observed_tasks)} of ${count(total.tasks)} task records scanned; remaining records are unknown.` : "Progress follows logical tasks, not allocation walltime.");
  }
  function svgElement(tag, attributes, text) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attributes || {})) node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = String(text);
    return node;
  }
  function chart(id, records, key, format, comparison) {
    const values = records.map((row) => number(row[key]));
    const comparisons = comparison ? records.map((row) => number(row[comparison])) : [];
    if (!values.some((value) => value !== null)) {
      $(id).replaceChildren(element("div", "chart-empty", "No recorded measurements for recent successful attempts.")); return;
    }
    const width = 500, left = 45, right = 485, top = 16, bottom = 125;
    const max = Math.max(1, ...values.filter((v) => v !== null), ...comparisons.filter((v) => v !== null));
    const x = (i) => values.length === 1 ? (left + right) / 2 : left + i * (right - left) / (values.length - 1);
    const y = (v) => bottom - v / max * (bottom - top);
    const svg = svgElement("svg", { viewBox: `0 0 ${width} 150`, role: "img", "aria-label": `${id === "runtime-chart" ? "Task duration" : "Sampled host RAM and requested RAM"} across recent successful attempts in completion order` });
    for (const fraction of [0, 0.5, 1]) {
      const yy = y(max * fraction);
      svg.append(svgElement("line", { x1: left, x2: right, y1: yy, y2: yy, class: "chart-gridline" }));
      svg.append(svgElement("text", { x: left - 9, y: yy + 3, "text-anchor": "end", class: "chart-label" }, format(max * fraction)));
    }
    function series(data, className) {
      let path = "", connected = false;
      data.forEach((value, i) => {
        if (value === null) { connected = false; return; }
        path += `${connected ? " L" : " M"}${x(i).toFixed(2)},${y(value).toFixed(2)}`; connected = true;
      });
      svg.append(svgElement("path", { d: path.trim(), class: className }));
    }
    if (comparison) series(comparisons, "chart-request");
    series(values, "chart-line");
    values.forEach((value, i) => {
      if (value === null) return;
      const point = svgElement("circle", { cx: x(i), cy: y(value), r: values.length < 20 ? 2.8 : 1.8, class: "chart-point" });
      point.append(svgElement("title", {}, `${String(records[i].task_id || "Task")}: ${format(value)}${comparison ? `; requested ${format(comparisons[i])}${records[i].ram_override === true ? ` (retry override; task declared ${format(records[i].declared_mib)})` : ""}` : ""}`)); svg.append(point);
    });
    svg.append(svgElement("text", { x: left, y: 146, class: "chart-label" }, "Earlier completions"), svgElement("text", { x: right, y: 146, "text-anchor": "end", class: "chart-label" }, "Latest"));
    $(id).replaceChildren(svg);
  }
  function renderMetrics() {
    const timing = snapshot.timing || {}, mem = snapshot.memory || {}, recent = list(snapshot.recent_runs);
    put("runtime-value", duration(timing.median_seconds)); put("runtime-p90", `P90 ${duration(timing.p90_seconds)}`);
    put("runtime-samples", `Mean ${duration(timing.mean_seconds)} · ${count(timing.samples)} successful samples · ${count(recent.filter((r) => number(r.duration_seconds) !== null).length)} shown`);
    put("memory-value", memory(mem.max_peak_mib));
    put("memory-samples", `${count(mem.samples)} measured attempts · chart shows recent successes`);
    chart("runtime-chart", recent, "duration_seconds", duration);
    chart("memory-chart", recent, "peak_rss_mib", memory, "requested_mib");
    const groups = list(snapshot.groups).slice(0, 12);
    $("workload-rows").replaceChildren(...groups.map((g) => {
      const row = element("tr"), name = element("td"), hardware = g.gpu_model === "cpu" ? "CPU" : g.gpu_model === "unknown" ? "GPU not recorded" : String(g.gpu_model || "Hardware not recorded");
      name.append(element("div", "workload-name", g.workload || "Unknown workload"), element("div", "hardware-name", `${hardware}${number(g.cpus) !== null ? ` · ${count(g.cpus)} CPU` : ""}`));
      name.title = `Software: ${g.software_fingerprint || "not recorded"}`;
      row.append(name, element("td", "numeric", count(g.timing?.samples)), element("td", "numeric", duration(g.timing?.median_seconds)), element("td", "numeric", memory(g.memory?.max_peak_mib)));
      return row;
    }));
    $("workload-empty").hidden = groups.length > 0;
  }
  function renderActivity() {
    const active = list(snapshot.active_tasks);
    put("active-count", count(snapshot.active_attempt_candidates ?? active.length));
    $("active-tasks").replaceChildren(...active.slice(0, 8).map((task) => {
      const row = element("li"), identity = element("div", "activity-task");
      const title = link("task", task.task_id, "activity-title", task.task_id); title.title = String(task.task_id);
      const shown = task.display_state || task.state || "unknown", unfinished = shown === "unfinished";
      identity.append(title, element("div", "activity-meta", `${task.workload || "Unknown workload"} · ${unfinished ? "unfinished · no result yet" : String(shown).replaceAll("_", " ")}${number(task.attempt_number) !== null ? ` · attempt ${task.attempt_number}` : ""}`));
      row.append(element("span", `activity-dot${shown === "running" ? "" : unfinished ? " unfinished" : " uncertain"}`), identity, element("span", "activity-age", `Age ${duration(task.elapsed_seconds)}`));
      row.title = unfinished ? "No result recorded yet. Liveness has not been probed; this is neither running nor a failure." : String(task.reason || "Liveness has not been probed"); return row;
    }));
    $("active-empty").hidden = active.length > 0;
    document.querySelector(".activity-note").textContent = `${active.length > 8 ? `Showing 8 of ${count(snapshot.active_attempt_candidates ?? active.length)} recorded attempts. ` : ""}Attempt age does not prove a process is still running.`;
  }
  function renderHealth() {
    for (const key of ["controller", "service"]) {
      const health = snapshot.health?.[key] || {};
      put(`health-${key}`, String(health.status || "Not recorded").replaceAll("_", " "));
      $(`health-${key}`).classList.toggle("stale", health.stale === true);
      put(`${key}-age`, number(health.age_seconds) === null ? "" : `${duration(health.age_seconds)} old`);
      $(`health-${key}`).title = health.details?.error || "Recorded state; no live process probe";
    }
    put("retry-count", count(snapshot.retries?.additional_attempts));
    put("allocation-count", count(entries(snapshot.allocations?.counts).reduce((sum, [, value]) => sum + (number(value) ?? 0), 0)));
    $("allocation-count").title = snapshot.allocations?.note || "Saved submission intents, not a live Slurm queue";
    const diag = snapshot.diagnostics || {}, content = $("diagnostic-content"), nodes = [];
    nodes.push(element("p", "", `Snapshot: ${snapshot.generated_at || "time not recorded"}. ${count(snapshot.totals?.observed_tasks)} task records observed; ${count(snapshot.totals?.unscanned_tasks)} unscanned.`));
    const notes = list(diag.notes).slice(0, 20);
    if (notes.length) { const ul = element("ul"); notes.forEach((note) => ul.append(element("li", "", note))); nodes.push(ul); }
    for (const [key, value] of entries(diag.truncation)) nodes.push(element("p", "", `${key.replaceAll("_", " ")}: ${typeof value === "object" ? JSON.stringify(value) : String(value)}`));
    const errors = list(diag.errors).slice(0, 12);
    if (errors.length) {
      const ul = element("ul"); errors.forEach((error) => ul.append(element("li", "", typeof error === "string" ? error : JSON.stringify(error)))); nodes.push(element("p", "", "Record errors"), ul);
    }
    content.replaceChildren(...nodes); $("diagnostics").hidden = false;
  }
  function render(data) {
    if (!data || data.schema !== "hhq-dashboard/1" || !data.totals || !Array.isArray(data.campaigns)) throw new Error("Unsupported dashboard snapshot");
    snapshot = data;
    const totals = data.totals, counts = totals.counts || {}, attention = attentionOf(totals), unfinished = unfinishedOf(totals);
    put("stat-completed", count(totals.completed)); put("detail-completed", `${count(totals.succeeded)} successful · ${count(totals.tasks)} total tasks`);
    put("stat-running", count(stateCount(counts, "running") + unfinished));
    put("detail-running", `${count(stateCount(counts, "running"))} recorded running · ${count(unfinished)} unfinished · not a live probe`);
    put("stat-waiting", count(sumStates(counts, ["ready", "submitted", "retry_wait"])));
    put("stat-attention", count(attention)); $("stat-attention").closest("article").classList.toggle("is-attention", attention > 0);
    put("detail-attention", `${count(stateCount(counts, "failed"))} failed · ${count(attention - stateCount(counts, "failed"))} blocked or uncertain`);
    document.querySelectorAll(".submission-link").forEach((node) => { node.hidden = saved || data.capabilities?.submission_enabled !== true; });
    renderCampaigns(); renderMetrics(); renderActivity(); renderHealth(); renderFlow(); freshness(); notices();
  }
  function schedule() {
    clearTimeout(timer);
    if (!saved && !document.hidden && navigator.onLine !== false) timer = setTimeout(refresh, 10000);
  }
  async function refresh() {
    if (saved || pending || document.hidden || navigator.onLine === false) return;
    clearTimeout(timer);
    const controller = new AbortController(); pending = controller; $("refresh").disabled = true;
    const timeout = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch("/api/dashboard", { credentials: "same-origin", cache: "no-store", signal: controller.signal, headers: { Accept: "application/json" } });
      if (!response.ok) throw Object.assign(new Error(`Snapshot response ${response.status}`), { status: response.status });
      const data = await response.json(); lastError = null; render(data);
    } catch (error) {
      lastError = error; statusBadge("Connection lost", "stale"); freshness(); notices();
      if (!snapshot) { $("campaign-empty").querySelector("h3").textContent = "Workspace unavailable"; $("campaign-empty").querySelector("p").textContent = "The saved records could not be read. Refresh to try again."; }
    } finally {
      clearTimeout(timeout); pending = null; $("refresh").disabled = false; schedule();
    }
  }
  $("campaign-search").addEventListener("input", () => { visibleCampaigns = 15; renderCampaigns(); });
  $("campaign-filter").addEventListener("change", () => { visibleCampaigns = 15; renderCampaigns(); });
  $("show-more").addEventListener("click", () => { visibleCampaigns += 25; renderCampaigns(); });
  $("flow-motion")?.addEventListener("click", () => { motionEnabled = !motionEnabled; syncFlowMotion(); });
  reducedMotion.addEventListener("change", syncFlowMotion);
  // Freeze promptly even before an outstanding request times out. Reconnection
  // still requires a successful fresh response before any current can resume.
  if (!saved) {
    window.addEventListener("offline", () => { lastError = new Error("Browser offline"); clearTimeout(timer); pending?.abort(); freshness(); notices(); });
    window.addEventListener("online", () => { freshness(); refresh(); });
  }
  syncFlowMotion();
  if (saved) {
    document.body.classList.add("is-snapshot"); $("refresh").hidden = true;
    put("refresh-policy", "Saved snapshot · does not refresh"); put("workspace-note", "A point-in-time export. Live details are available in the running dashboard.");
    put("footer-note", "Saved execution records · no live connections");
    document.querySelectorAll('a[href^="/"]').forEach(disableLink);
    try { render(window.HHQ_SNAPSHOT); } catch (error) { lastError = error; put("notice", "This saved snapshot is not supported or is incomplete."); $("notice").hidden = false; statusBadge("Invalid snapshot", "stale"); }
  } else {
    statusBadge("Connecting", "loading"); $("refresh").addEventListener("click", refresh);
    document.addEventListener("visibilitychange", () => { clearTimeout(timer); freshness(); if (!document.hidden) refresh(); });
    refresh();
  }
}
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeDashboard, { once: true });
} else {
  initializeDashboard();
}

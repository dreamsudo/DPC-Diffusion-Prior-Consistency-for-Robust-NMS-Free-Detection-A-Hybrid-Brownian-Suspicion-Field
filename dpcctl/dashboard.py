"""Live dashboard for DPC-YOLO26.

Serves a single HTML page that polls `<live_dir>/state.json` for the
active phase and then polls `<live_dir>/<phase>_state.json` for that
phase's current payload.

Generic event bus: phases publish state files with a `kind` discriminator
(`training`, `diagnostic`, `eval`, `aggregate`, `progress`). The
dashboard JS renders one of a small set of per-kind layouts; unknown
kinds fall through to a generic key/value renderer. Adding a phase
later requires no dashboard changes as long as its payload carries one
of the known kinds.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dpcctl.config import OrchestratorConfig


DASHBOARD_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>DPC-YOLO26 dashboard</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; margin: 1.5rem;
         background: #0e1117; color: #d0d7de; }
  h1 { font-size: 1.2rem; margin: 0 0 0.5rem 0; }
  .meta { color: #8b949e; font-size: 0.85rem; margin-bottom: 1rem; }
  .tabs { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
  .tab { padding: 0.4rem 0.8rem; border: 1px solid #30363d; border-radius: 6px;
         cursor: pointer; background: #161b22; color: #8b949e; font-size: 0.9rem; }
  .tab.active { background: #1f6feb; color: white; border-color: #1f6feb; }
  .tab.running { box-shadow: 0 0 0 2px #2ea043 inset; color: #2ea043; }
  .tab.failed  { box-shadow: 0 0 0 2px #f85149 inset; color: #f85149; }
  .panel { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
           padding: 1rem; }
  .row { display: flex; gap: 1rem; margin-bottom: 0.5rem; flex-wrap: wrap; }
  .stat { background: #0d1117; border: 1px solid #30363d; padding: 0.5rem 0.8rem;
          border-radius: 4px; min-width: 140px; }
  .stat .k { color: #8b949e; font-size: 0.75rem; text-transform: uppercase; }
  .stat .v { color: #d0d7de; font-family: ui-monospace, monospace; font-size: 1.1rem; }
  pre { background: #0d1117; border: 1px solid #30363d; padding: 0.75rem;
        border-radius: 4px; overflow-x: auto; font-size: 0.8rem; }
  .empty { color: #8b949e; font-style: italic; padding: 1rem; }
</style>
</head><body>
<h1>DPC-YOLO26 dashboard</h1>
<div class="meta" id="meta">connecting...</div>
<div class="tabs" id="tabs"></div>
<div class="panel" id="panel"><div class="empty">waiting for phase data...</div></div>

<script>
let state = { active_phase: null, run: null, status: null };
let phases = {};
let manualTab = null;

async function fetchJson(url) {
  try {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) return null;
    return await r.json();
  } catch (e) { return null; }
}

async function poll() {
  // 1. global state
  const g = await fetchJson("/state.json");
  if (g) {
    const phaseChanged = g.active_phase !== state.active_phase;
    state = g;
    document.getElementById("meta").innerText =
      `run: ${g.run || "?"}  |  status: ${g.status}  |  active: ${g.active_phase}`;
    if (phaseChanged && manualTab === null && g.active_phase) {
      // auto-switch tab on phase transition unless user pinned one
      currentTab = g.active_phase;
    }
  }

  // 2. discover all phase state files
  const list = await fetchJson("/list");
  if (list && list.phases) {
    for (const p of list.phases) {
      const data = await fetchJson(`/${p}_state.json`);
      if (data) phases[p] = data;
    }
  }

  renderTabs();
  renderPanel();
  setTimeout(poll, 1500);
}

let currentTab = null;
function renderTabs() {
  const tabs = document.getElementById("tabs");
  const phaseNames = Object.keys(phases).sort();
  tabs.innerHTML = "";
  for (const p of phaseNames) {
    const t = document.createElement("div");
    t.className = "tab";
    if (p === currentTab) t.classList.add("active");
    if (p === state.active_phase && state.status === "running") t.classList.add("running");
    if (state.status === "failed" && p === state.active_phase) t.classList.add("failed");
    t.innerText = p;
    t.onclick = () => { manualTab = p; currentTab = p; renderTabs(); renderPanel(); };
    tabs.appendChild(t);
  }
  if (currentTab === null && phaseNames.length) currentTab = phaseNames[0];
}

function renderPanel() {
  const panel = document.getElementById("panel");
  if (!currentTab || !phases[currentTab]) {
    panel.innerHTML = '<div class="empty">no data for this phase yet</div>';
    return;
  }
  const data = phases[currentTab];
  const kind = data.kind || "generic";
  if (kind === "training") renderTraining(panel, data);
  else if (kind === "eval") renderEval(panel, data);
  else if (kind === "diagnostic") renderDiagnostic(panel, data);
  else if (kind === "aggregate") renderAggregate(panel, data);
  else if (kind === "progress") renderProgress(panel, data);
  else renderGeneric(panel, data);
}

function stat(k, v) {
  return `<div class="stat"><div class="k">${k}</div><div class="v">${v ?? "—"}</div></div>`;
}

function renderTraining(panel, d) {
  const cur = d.current_loss != null ? d.current_loss.toFixed(4) : "—";
  let html = `<div class="row">
    ${stat("seed", d.seed)}
    ${stat("global step", d.global_step)}
    ${stat("epochs", d.epochs_total)}
    ${stat("loss", cur)}
    ${stat("rows", d.n_rows)}
  </div>`;
  if (d.loss_recent && d.loss_recent.length) {
    html += `<pre>loss_recent (last ${d.loss_recent.length}): ${d.loss_recent.map(x => x.toFixed(3)).join(", ")}</pre>`;
  }
  panel.innerHTML = html;
}

function renderEval(panel, d) {
  let html = `<div class="row">
    ${stat("seed", d.seed)}
    ${stat("status", d.status)}
    ${stat("alpha", d.alpha)}
    ${stat("alphas total", d.alphas_total)}
    ${stat("alphas done", d.alphas_complete)}
    ${stat("elapsed (min)", d.elapsed_min)}
  </div>`;
  if (d.aggregate) {
    html += `<pre>${JSON.stringify(d.aggregate, null, 2)}</pre>`;
  }
  panel.innerHTML = html;
}

function renderDiagnostic(panel, d) {
  let html = `<div class="row">
    ${stat("seed", d.seed)}
    ${stat("elapsed (min)", d.elapsed_min)}
  </div>`;
  if (d.aggregate) html += `<pre>${JSON.stringify(d.aggregate, null, 2)}</pre>`;
  panel.innerHTML = html;
}

function renderAggregate(panel, d) {
  let html = `<div class="row">
    ${stat("seeds", JSON.stringify(d.seeds))}
    ${stat("n alphas", d.n_alphas)}
    ${stat("n seeds w/ data", d.n_seeds_with_data)}
  </div>`;
  if (d.summary_path) html += `<pre>summary written to: ${d.summary_path}</pre>`;
  panel.innerHTML = html;
}

function renderProgress(panel, d) {
  panel.innerHTML = `<div class="row">
    ${stat("message", d.message)}
    ${stat("elapsed (sec)", d.elapsed_sec != null ? d.elapsed_sec.toFixed(1) : "—")}
  </div>`;
}

function renderGeneric(panel, d) {
  panel.innerHTML = `<pre>${JSON.stringify(d, null, 2)}</pre>`;
}

poll();
</script>
</body></html>
"""


class _DashboardHandler(BaseHTTPRequestHandler):
    """Static + JSON file handler scoped to the live/ directory."""

    live_dir: Path = None

    def log_message(self, *args, **kwargs):
        # Silence the default request-per-line stderr noise.
        pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_html(DASHBOARD_HTML)
            return
        if path == "/list":
            phases = sorted(
                f.stem.replace("_state", "")
                for f in self.live_dir.glob("*_state.json")
            )
            self._send_json({"phases": phases})
            return
        if path.endswith(".json"):
            name = path.lstrip("/")
            file_path = self.live_dir / name
            if file_path.is_file():
                try:
                    self._send_json(json.loads(file_path.read_text()))
                    return
                except Exception:
                    pass
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def _send_html(self, body: str):
        b = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _send_json(self, obj: dict):
        b = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server that swallows benign client disconnects."""

    def handle_error(self, request, client_address):
        import sys as _sys
        exc_type = _sys.exc_info()[0]
        if exc_type and issubclass(exc_type, (
            ConnectionResetError,
            BrokenPipeError,
            ConnectionAbortedError,
        )):
            return
        super().handle_error(request, client_address)


def serve_dashboard(cfg: OrchestratorConfig, port: int = 8080) -> None:
    """Block-serve the dashboard for a given run config."""
    live = cfg.run_dir / "live"
    live.mkdir(parents=True, exist_ok=True)

    handler_cls = type(
        "Handler",
        (_DashboardHandler,),
        {"live_dir": live},
    )
    server = _QuietThreadingHTTPServer(("0.0.0.0", port), handler_cls)
    print(f"dashboard serving on http://localhost:{port}/  (run: {cfg.name})")
    print(f"  live dir: {live}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping dashboard")
        server.server_close()

"""
Python Q&A Assistant — FastAPI entry point with Integrated Dashboard
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import uvicorn
from app.routers import qa, health
from app.core.rag import rag_pipeline

# Store your HTML dashboard code in a clean multiline string
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Python Q&A Assistant — Dashboard</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.19.0/dist/tabler-icons.min.css" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg-primary:    #ffffff;
      --bg-secondary:  #f4f4f0;
      --bg-tertiary:   #eeeee8;
      --bg-info:       #e6f1fb;
      --text-primary:  #1a1a18;
      --text-secondary:#5f5e5a;
      --text-tertiary: #888780;
      --text-info:     #185fa5;
      --text-danger:   #a32d2d;
      --text-success:  #3b6d11;
      --border:        rgba(0,0,0,0.12);
      --border-hover:  rgba(0,0,0,0.22);
      --border-info:   rgba(24,95,165,0.3);
      --radius-md:     8px;
      --radius-lg:     12px;
      --font-sans:     -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --font-mono:     "SF Mono", "Fira Code", "Cascadia Code", monospace;
    }

    @media (prefers-color-scheme: dark) {
      :root {
        --bg-primary:    #1e1e1c;
        --bg-secondary:  #2a2a27;
        --bg-tertiary:   #161614;
        --bg-info:       #0c2a44;
        --text-primary:  #f0efe8;
        --text-secondary:#b4b2a9;
        --text-tertiary: #888780;
        --text-info:     #85b7eb;
        --text-danger:   #f09595;
        --text-success:  #97c459;
        --border:        rgba(255,255,255,0.1);
        --border-hover:  rgba(255,255,255,0.2);
        --border-info:   rgba(133,183,235,0.3);
      }
    }

    html { font-size: 16px; }
    body {
      font-family: var(--font-sans);
      background: var(--bg-tertiary);
      color: var(--text-primary);
      min-height: 100vh;
      padding: 24px 16px;
    }

    .container {
      max-width: 900px;
      margin: 0 auto;
    }

    /* ── Top bar ─────────────────────────────────── */
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1.5rem;
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-icon {
      width: 40px; height: 40px;
      background: var(--bg-info);
      border-radius: var(--radius-md);
      display: flex; align-items: center; justify-content: center;
      color: var(--text-info);
      font-size: 22px;
    }
    .brand-name  { font-size: 17px; font-weight: 500; color: var(--text-primary); }
    .brand-sub   { font-size: 12px; color: var(--text-secondary); margin-top: 2px; }

    .topbar-right { display: flex; align-items: center; gap: 8px; }

    .status-pill {
      display: inline-flex; align-items: center; gap: 6px;
      font-size: 12px; padding: 5px 12px;
      border-radius: 20px;
      border: 0.5px solid var(--border);
      background: var(--bg-secondary);
      color: var(--text-secondary);
    }
    .dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--text-tertiary);
      flex-shrink: 0;
    }
    .dot.ok      { background: #639922; }
    .dot.err     { background: #e24b4a; }
    .dot.loading { background: #ef9f27; animation: pulse 1s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }

    /* ── Metric cards ────────────────────────────── */
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin-bottom: 14px;
    }
    @media(max-width:600px){
      .metrics { grid-template-columns: repeat(2,1fr); }
    }
    .metric {
      background: var(--bg-secondary);
      border-radius: var(--radius-md);
      padding: 12px 14px;
    }
    .metric-label { font-size: 12px; color: var(--text-secondary); margin-bottom: 4px; }
    .metric-val   { font-size: 22px; font-weight: 500; color: var(--text-primary); }
    .metric-sub   { font-size: 11px; color: var(--text-tertiary); margin-top: 3px; }

    /* ── Cards ───────────────────────────────────── */
    .card {
      background: var(--bg-primary);
      border: 0.5px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 14px 16px;
    }
    .card-title {
      font-size: 14px; font-weight: 500;
      color: var(--text-primary);
      margin-bottom: 12px;
      display: flex; align-items: center; gap: 7px;
    }
    .card-title i { font-size: 16px; color: var(--text-secondary); }

    .card-full { margin-bottom: 14px; }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }
    @media(max-width:600px){ .grid2 { grid-template-columns: 1fr; } }

    /* ── Ask panel ───────────────────────────────── */
    textarea {
      width: 100%; resize: vertical; min-height: 84px;
      border: 0.5px solid var(--border-hover);
      border-radius: var(--radius-md);
      padding: 10px 12px;
      font-size: 13px; font-family: var(--font-sans);
      color: var(--text-primary);
      background: var(--bg-secondary);
      margin-bottom: 10px;
      line-height: 1.5;
    }
    textarea:focus { outline: none; border-color: #378add; }

    .btn-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }

    button {
      font-size: 13px; padding: 7px 14px;
      border-radius: var(--radius-md);
      border: 0.5px solid var(--border-hover);
      background: var(--bg-primary);
      color: var(--text-primary);
      cursor: pointer;
      display: inline-flex; align-items: center; gap: 6px;
      font-family: var(--font-sans);
      transition: background .15s;
    }
    button:hover { background: var(--bg-secondary); }
    button:active { transform: scale(0.98); }
    button:disabled { opacity: .5; cursor: not-allowed; }

    .btn-primary {
      background: var(--bg-info);
      color: var(--text-info);
      border-color: var(--border-info);
    }
    .btn-primary:hover { opacity: .85; background: var(--bg-info); }

    .latency-badge { font-size: 11px; color: var(--text-tertiary); margin-left: auto; }

    /* ── Answer ──────────────────────────────────── */
    #answerSection { margin-top: 14px; display: none; }
    .answer-box {
      background: var(--bg-secondary);
      border-radius: var(--radius-md);
      padding: 12px 14px;
      font-size: 13px; line-height: 1.65;
      color: var(--text-primary);
      white-space: pre-wrap; word-break: break-word;
    }
    .sources { margin-top: 10px; display: flex; flex-direction: column; gap: 6px; }
    .source-chip {
      display: flex; align-items: center; justify-content: space-between;
      font-size: 12px; padding: 6px 10px;
      background: var(--bg-primary);
      border: 0.5px solid var(--border);
      border-radius: var(--radius-md);
      gap: 8px;
    }
    .source-title {
      color: var(--text-secondary); flex: 1; min-width: 0;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .source-rel {
      font-size: 11px; padding: 2px 8px; border-radius: 20px;
      background: var(--bg-info); color: var(--text-info); white-space: nowrap;
    }

    /* ── Status messages ─────────────────────────── */
    .loading-msg {
      display: flex; align-items: center; gap: 7px;
      font-size: 12px; color: var(--text-secondary);
      margin-top: 8px;
    }
    .err-msg {
      font-size: 12px; color: var(--text-danger);
      margin-top: 8px;
      display: flex; align-items: center; gap: 6px;
    }

    /* ── History ─────────────────────────────────── */
    .history-row {
      display: flex; align-items: flex-start; gap: 8px;
      padding: 8px 0;
      border-bottom: 0.5px solid var(--border);
      font-size: 12px;
    }
    .history-row:last-child { border-bottom: none; }
    .hq { flex: 1; min-width: 0; color: var(--text-secondary); line-height: 1.4; }
    .hq strong { color: var(--text-primary); font-weight: 500; }
    .hms { white-space: nowrap; color: var(--text-tertiary); font-size: 11px; padding-left: 8px; margin-left: auto; }
    .empty-note { font-size: 12px; color: var(--text-tertiary); font-style: italic; }

    /* ── Endpoints ───────────────────────────────── */
    .endpoint-row {
      display: flex; align-items: center; gap: 8px;
      padding: 7px 0; border-bottom: 0.5px solid var(--border);
      font-size: 13px;
    }
    .endpoint-row:last-child { border-bottom: none; }
    .method {
      font-size: 11px; font-weight: 500;
      padding: 2px 7px; border-radius: 4px;
      font-family: var(--font-mono);
    }
    .get  { background: #e1f5ee; color: #0f6e56; }
    .post { background: #e6f1fb; color: #185fa5; }
    .path { font-family: var(--font-mono); font-size: 12px; color: var(--text-primary); }
    .endpoint-desc { font-size: 12px; color: var(--text-secondary); margin-left: auto; }

    /* ── Latency bars ────────────────────────────── */
    .lat-item { margin-bottom: 10px; }
    .lat-header { display: flex; justify-content: space-between; font-size: 12px; color: var(--text-secondary); margin-bottom: 3px; }
    .lat-track { height: 5px; border-radius: 3px; background: var(--bg-secondary); overflow: hidden; }
    .lat-fill { height: 100%; border-radius: 3px; background: #378add; transition: width .6s ease; }

    /* ── Sample buttons ──────────────────────────── */
    .sample-btn {
      font-size: 12px; padding: 5px 10px;
      text-align: left; width: 100%;
      margin-bottom: 5px; display: block;
    }
    .sample-btn:last-child { margin-bottom: 0; }

    /* ── Footer ──────────────────────────────────── */
    footer {
      text-align: center;
      font-size: 11px;
      color: var(--text-tertiary);
      margin-top: 1.5rem;
      padding-top: 1rem;
      border-top: 0.5px solid var(--border);
    }
    footer a { color: var(--text-info); text-decoration: none; }
    footer a:hover { text-decoration: underline; }
  </style>
</head>
<body>
<div class="container">

  <div class="topbar">
    <div class="brand">
      <div class="brand-icon"><i class="ti ti-brand-python" aria-hidden="true"></i></div>
      <div>
        <div class="brand-name">Python Q&A Assistant</div>
        <div class="brand-sub">v1.0.0 · Stack Overflow RAG · Groq LLM</div>
      </div>
    </div>
    <div class="topbar-right">
      <div class="status-pill">
        <div class="dot loading" id="healthDot"></div>
        <span id="healthLabel">Checking…</span>
      </div>
      <button onclick="checkHealth()" title="Refresh health"><i class="ti ti-refresh" aria-hidden="true"></i></button>
      <a href="http://localhost:8000/docs" target="_blank" rel="noopener">
        <button title="Open API docs"><i class="ti ti-external-link" aria-hidden="true"></i> API docs</button>
      </a>
    </div>
  </div>

  <div class="metrics">
    <div class="metric">
      <div class="metric-label">Status</div>
      <div class="metric-val" id="mStatus">—</div>
      <div class="metric-sub">API health</div>
    </div>
    <div class="metric">
      <div class="metric-label">Documents</div>
      <div class="metric-val" id="mDocs">—</div>
      <div class="metric-sub">vector store</div>
    </div>
    <div class="metric">
      <div class="metric-label">LLM model</div>
      <div class="metric-val" id="mModel" style="font-size:13px;padding-top:5px">—</div>
      <div class="metric-sub">generation</div>
    </div>
    <div class="metric">
      <div class="metric-label">Embedding</div>
      <div class="metric-val" id="mEmbed" style="font-size:11px;padding-top:7px">—</div>
      <div class="metric-sub">retrieval</div>
    </div>
  </div>

  <div class="card card-full">
    <div class="card-title"><i class="ti ti-message-question" aria-hidden="true"></i>Ask a Python question</div>
    <textarea
      id="questionInput"
      placeholder="e.g. How do I reverse a list in Python?&#10;e.g. What is the difference between __str__ and __repr__?&#10;&#10;Tip: Ctrl+Enter to submit"
    ></textarea>
    <div class="btn-row">
      <button class="btn-primary" id="askBtn" onclick="askQuestion()">
        <i class="ti ti-send" aria-hidden="true"></i>Ask
      </button>
      <button onclick="clearAnswer()"><i class="ti ti-x" aria-hidden="true"></i>Clear</button>
      <span class="latency-badge" id="latencyBadge"></span>
    </div>
    <div id="askStatus"></div>
    <div id="answerSection">
      <div class="answer-box" id="answerBox"></div>
      <div class="sources" id="sourcesBox"></div>
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <div class="card-title"><i class="ti ti-clock-history" aria-hidden="true"></i>Session history</div>
      <div id="historyList"><div class="empty-note">No questions asked yet.</div></div>
    </div>
    <div class="card">
      <div class="card-title"><i class="ti ti-api" aria-hidden="true"></i>Endpoints</div>
      <div class="endpoint-row"><span class="method get">GET</span><span class="path">/health</span><span class="endpoint-desc">API health &amp; stats</span></div>
      <div class="endpoint-row"><span class="method post">POST</span><span class="path">/ask</span><span class="endpoint-desc">Q&amp;A query</span></div>
      <div class="endpoint-row"><span class="method get">GET</span><span class="path">/docs</span><span class="endpoint-desc">OpenAPI UI</span></div>
      <div class="endpoint-row"><span class="method get">GET</span><span class="path">/</span><span class="endpoint-desc">Root info</span></div>
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <div class="card-title"><i class="ti ti-chart-bar" aria-hidden="true"></i>Response latency</div>
      <div id="latencyPanel"><div class="empty-note">Ask a question to see latency.</div></div>
    </div>
    <div class="card">
      <div class="card-title"><i class="ti ti-bulb" aria-hidden="true"></i>Sample questions</div>
      <div id="samplesPanel"></div>
    </div>
  </div>

  <footer>
    Python Q&A Assistant &mdash; powered by
    <a href="https://groq.com" target="_blank" rel="noopener">Groq</a> ·
    <a href="http://localhost:8000/docs" target="_blank" rel="noopener">API docs</a> ·
    <a href="https://www.kaggle.com/datasets/stackoverflow/pythonquestions" target="_blank" rel="noopener">Dataset</a>
  </footer>

</div>

<script>
  const BASE = window.location.origin; // Dynamically sets base url matching backend location

  const sessionHistory = [];
  const sessionLatencies = [];

  const SAMPLES = [
    'How do I reverse a list in Python?',
    'What is the difference between __str__ and __repr__?',
    'How do I read a CSV file with pandas?',
    'How do I merge two dictionaries in Python 3?',
    'How do I use @property decorator in Python?',
    'How do I profile Python code to find bottlenecks?',
  ];

  function $(id) { return document.getElementById(id); }

  // ── Health check ──────────────────────────────────────────────────────────

  async function checkHealth() {
    const dot = $('healthDot');
    const lbl = $('healthLabel');
    dot.className = 'dot loading';
    lbl.textContent = 'Checking…';
    try {
      const res = await fetch(BASE + '/health');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const d = await res.json();
      dot.className = 'dot ok';
      lbl.textContent = d.status || 'ok';
      $('mStatus').textContent = d.status || 'ok';
      $('mDocs').textContent = d.documents != null ? Number(d.documents).toLocaleString() : '—';
      $('mModel').textContent = d.model || '—';
      $('mEmbed').textContent = d.embedding || '—';
    } catch (err) {
      dot.className = 'dot err';
      lbl.textContent = 'Offline';
      $('mStatus').textContent = 'offline';
    }
  }

  // ── Ask ───────────────────────────────────────────────────────────────────

  async function askQuestion() {
    const q = $('questionInput').value.trim();
    if (!q) return;

    const btn = $('askBtn');
    btn.disabled = true;
    btn.innerHTML = '<i class="ti ti-loader" aria-hidden="true"></i> Thinking…';
    $('askStatus').innerHTML = '<div class="loading-msg"><div class="dot loading"></div>Querying RAG pipeline…</div>';
    $('answerSection').style.display = 'none';
    $('latencyBadge').textContent = '';

    try {
      const res = await fetch(BASE + '/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Request failed' }));
        throw new Error(err.detail || 'HTTP ' + res.status);
      }

      const d = await res.json();
      const ms = d.latency_ms || 0;

      $('latencyBadge').textContent = ms + ' ms';
      $('answerBox').textContent = d.answer || '—';

      const sb = $('sourcesBox');
      sb.innerHTML = '';
      if (d.sources && d.sources.length) {
        d.sources.forEach(s => {
          const chip = document.createElement('div');
          chip.className = 'source-chip';
          const rel = s.relevance != null ? Math.round(s.relevance * 100) + '%' : '';
          chip.innerHTML =
            '<span class="source-title" title="' + escHtml(s.title || '') + '">' + escHtml(s.title || 'Source') + '</span>' +
            (rel ? '<span class="source-rel">' + rel + '</span>' : '');
          sb.appendChild(chip);
        });
      }

      $('answerSection').style.display = 'block';
      $('askStatus').innerHTML = '';

      addHistory(q, ms);
      addLatency(ms, q);

    } catch (err) {
      $('askStatus').innerHTML =
        '<div class="err-msg"><i class="ti ti-alert-triangle" aria-hidden="true"></i> ' + escHtml(err.message) + '</div>';
    } finally {
      btn.disabled = false;
      btn.innerHTML = '<i class="ti ti-send" aria-hidden="true"></i> Ask';
    }
  }

  function clearAnswer() {
    $('questionInput').value = '';
    $('answerSection').style.display = 'none';
    $('askStatus').innerHTML = '';
    $('latencyBadge').textContent = '';
  }

  // ── History ───────────────────────────────────────────────────────────────

  function addHistory(q, ms) {
    sessionHistory.unshift({ q, ms, t: new Date().toLocaleTimeString() });
    if (sessionHistory.length > 10) sessionHistory.pop();
    const el = $('historyList');
    el.innerHTML = sessionHistory.map(h =>
      '<div class="history-row">' +
        '<div class="hq"><strong>' + escHtml(h.q.slice(0, 65)) + (h.q.length > 65 ? '…' : '') + '</strong></div>' +
        '<span class="hms">' + h.ms + 'ms · ' + h.t + '</span>' +
      '</div>'
    ).join('');
  }

  // ── Latency bars ──────────────────────────────────────────────────────────

  function addLatency(ms, q) {
    sessionLatencies.unshift({ ms, label: q.slice(0, 30) + (q.length > 30 ? '…' : '') });
    if (sessionLatencies.length > 6) sessionLatencies.pop();
    const max = Math.max(...sessionLatencies.map(l => l.ms), 500);
    const el = $('latencyPanel');
    el.innerHTML = sessionLatencies.map((l, i) =>
      '<div class="lat-item">' +
        '<div class="lat-header"><span>Query ' + (sessionLatencies.length - i) + '</span><span>' + l.ms + ' ms</span></div>' +
        '<div class="lat-track"><div class="lat-fill" style="width:' + Math.round(l.ms / max * 100) + '%"></div></div>' +
      '</div>'
    ).join('');
  }

  // ── Sample questions ──────────────────────────────────────────────────────

  function buildSamples() {
    const el = $('samplesPanel');
    el.innerHTML = SAMPLES.map(s =>
      '<button class="sample-btn" onclick="useSample(this)">' + escHtml(s) + '</button>'
    ).join('');
  }

  function useSample(btn) {
    $('questionInput').value = btn.textContent.trim();
    $('questionInput').focus();
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  function escHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // ── Keyboard shortcut ─────────────────────────────────────────────────────

  $('questionInput').addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) askQuestion();
  });

  // ── Init ──────────────────────────────────────────────────────────────────

  checkHealth();
  buildSamples();
</script>
</body>
</html>"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    await rag_pipeline.initialize()
    yield


app = FastAPI(
    title="Python Q&A Assistant",
    description="AI-powered Q&A system grounded in Stack Overflow Python data",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(qa.router, tags=["Q&A"])


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def serve_dashboard():
    """Serves the interactive single-page dashboard directly on the root endpoint."""
    return HTMLResponse(content=DASHBOARD_HTML, status_code=200)


@app.get("/api-info", tags=["Root"])
async def root():
    return {
        "service": "Python Q&A Assistant",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "GET /health",
        "ask":     "POST /ask",
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=5001, reload=True)
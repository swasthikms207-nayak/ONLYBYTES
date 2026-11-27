import { useState, useRef, useCallback } from "react";
import EvaluationTab from "./components/EvaluationTab.jsx";

/* ── Sample data ─────────────────────────────────────────────────── */
const SAMPLES = {
  text: [
    "Where is my order ORD_5501?",
    "Sab band kar do abhi",
    "Cancel it before dispatch",
    "Block my card",
    "Help please",
  ].join("\n"),
  csv: `text\n"Where is my order ORD_5501?"\n"Sab band kar do abhi"\n"Cancel it before dispatch"\n"Close it now"`,
  context:
    "Customer last ordered item ORD_5501 on Nov 20. Order is pending dispatch.",
};

/* ── Type badge colours ──────────────────────────────────────────── */
const TYPE_COLORS = {
  lexical: "#818cf8",
  syntactic: "#fb923c",
  referential: "#34d399",
  scope: "#f472b6",
  pragmatic: "#facc15",
  underspecified: "#94a3b8",
  none: "#4ade80",
};

/* ── SVG Donut chart ─────────────────────────────────────────────── */
function DonutChart({ data }) {
  const total = data.reduce((s, d) => s + d.count, 0);
  if (!total) return null;
  const R = 60, cx = 80, cy = 80;
  let angle = -Math.PI / 2;
  const slices = data.map((d) => {
    const slice = (d.count / total) * 2 * Math.PI;
    const x1 = cx + R * Math.cos(angle);
    const y1 = cy + R * Math.sin(angle);
    angle += slice;
    const x2 = cx + R * Math.cos(angle);
    const y2 = cy + R * Math.sin(angle);
    const large = slice > Math.PI ? 1 : 0;
    return { ...d, x1, y1, x2, y2, large, end: angle };
  });

  return (
    <svg viewBox="0 0 160 160" className="donut-svg">
      {slices.map((s, i) => (
        <path
          key={i}
          d={`M${cx},${cy} L${s.x1},${s.y1} A${R},${R} 0 ${s.large},1 ${s.x2},${s.y2} Z`}
          fill={TYPE_COLORS[s.type] || "#52525b"}
          opacity={0.85}
        >
          <title>{s.type}: {s.count}</title>
        </path>
      ))}
      <circle cx={cx} cy={cy} r={36} fill="#18181b" />
      <text x={cx} y={cy + 5} textAnchor="middle" fill="#f4f4f5" fontSize="14" fontWeight="700">
        {total}
      </text>
      <text x={cx} y={cy + 20} textAnchor="middle" fill="#a1a1aa" fontSize="9">
        total
      </text>
    </svg>
  );
}

/* ── Confidence gauge ────────────────────────────────────────────── */
function ConfGauge({ value }) {
  const pct = Math.round(value * 100);
  const color = value > 0.8 ? "#22c55e" : value > 0.5 ? "#eab308" : "#ef4444";
  return (
    <div className="conf-gauge">
      <div className="gauge-track">
        <div className="gauge-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="gauge-label" style={{ color }}>{pct}%</span>
    </div>
  );
}

/* ── Status chip ─────────────────────────────────────────────────── */
function StatusChip({ status }) {
  return <span className={`status-badge status-${status}`}>{status.replace(/_/g, " ")}</span>;
}

/* ── Highlighted text ────────────────────────────────────────────── */
function HighlightedText({ text, type }) {
  if (!type || type === "none") return <span>{text}</span>;
  // Bold the whole sentence and tag with type colour
  return (
    <span style={{ fontWeight: 600, color: TYPE_COLORS[type] || "#f4f4f5" }}>
      {text}
    </span>
  );
}

/* ── Probability bar ─────────────────────────────────────────────── */
function ProbBar({ reading, probability, isTop, conflict }) {
  const pct = Math.round(probability * 100);
  const color = conflict && isTop ? "#ef4444" : isTop ? "#818cf8" : "#3f3f46";
  return (
    <div className="prob-row">
      <span className="prob-label">{reading}</span>
      <div className="prob-track">
        <div className="prob-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="prob-pct">{pct}%</span>
    </div>
  );
}

/* ── Single result card (center+right) ──────────────────────────── */
function ResultCard({ r, idx }) {
  const [open, setOpen] = useState(idx === 0);
  return (
    <div className={`result-card ${open ? "open" : ""}`}>
      <button className="result-header" onClick={() => setOpen((v) => !v)}>
        <div className="result-header-left">
          <span className="result-index">#{idx + 1}</span>
          <span className="result-text-preview">{r.text}</span>
        </div>
        <div className="result-header-right">
          <span
            className="type-badge"
            style={{ background: (TYPE_COLORS[r.ambiguity_type] || "#52525b") + "22",
                     color: TYPE_COLORS[r.ambiguity_type] || "#a1a1aa",
                     borderColor: TYPE_COLORS[r.ambiguity_type] || "#52525b" }}
          >
            {r.ambiguity_type}
          </span>
          {r.conflict && <span className="conflict-flag">⚡ conflict</span>}
          <StatusChip status={r.status} />
          <span className="chevron">{open ? "▲" : "▼"}</span>
        </div>
      </button>

      {open && (
        <div className="result-body">
          {/* Centre column */}
          <div className="result-col">
            <p className="col-title">Text</p>
            <p className="highlighted-text">
              <HighlightedText text={r.text} type={r.ambiguity_type} />
            </p>

            <p className="col-title">Top reading</p>
            <p className="top-reading">{r.top_reading}</p>

            <p className="col-title">Confidence</p>
            <ConfGauge value={r.confidence} />

            {r.context_used && (
              <>
                <p className="col-title">Evidence</p>
                <p className="evidence-text">{r.context_used}</p>
              </>
            )}
          </div>

          {/* Right column */}
          <div className="result-col">
            <p className="col-title">Readings &amp; probabilities</p>
            {r.alternatives.length === 0 ? (
              <p className="muted-sm">No ambiguity detected.</p>
            ) : (
              <div className="prob-list">
                {/* Top reading bar */}
                <ProbBar
                  reading={r.top_reading}
                  probability={1 - r.alternatives.reduce((s, a) => s + a.probability, 0)}
                  isTop
                  conflict={r.conflict}
                />
                {r.alternatives.map((alt, i) => (
                  <ProbBar key={i} reading={alt.reading} probability={alt.probability} />
                ))}
              </div>
            )}

            {r.conflict && (
              <div className="conflict-box">
                <p className="col-title" style={{ color: "#ef4444" }}>Conflict reason</p>
                <p className="conflict-reason">{r.conflict_reason || "Sentence-alone and context readings disagree."}</p>
                <p className="disagree-label">Disagreement: {Math.round((r.disagreement || 0) * 100)}%</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Summary cards ───────────────────────────────────────────────── */
function SummaryCards({ results }) {
  const total = results.length;
  const auto = results.filter((r) => r.status === "AUTO_ACCEPT").length;
  const alt = results.filter((r) => r.status === "ACCEPT_WITH_ALTERNATIVES").length;
  const review = results.filter((r) => r.status === "REVIEW").length;
  const unambig = results.filter((r) => r.ambiguity_type === "none").length;

  const cards = [
    { label: "Total", value: total, color: "#a1a1aa" },
    { label: "Auto-accepted", value: auto, color: "#22c55e" },
    { label: "With alternatives", value: alt, color: "#eab308" },
    { label: "Needs review", value: review, color: "#ef4444" },
    { label: "Unambiguous", value: unambig, color: "#4ade80" },
  ];

  // Type distribution for donut
  const typeCounts = {};
  results.forEach((r) => {
    typeCounts[r.ambiguity_type] = (typeCounts[r.ambiguity_type] || 0) + 1;
  });
  const donutData = Object.entries(typeCounts).map(([type, count]) => ({ type, count }));

  return (
    <div className="summary-row">
      {cards.map((c) => (
        <div className="summary-card" key={c.label}>
          <span className="summary-value" style={{ color: c.color }}>{c.value}</span>
          <span className="summary-label">{c.label}</span>
        </div>
      ))}
      <div className="summary-card donut-card">
        <DonutChart data={donutData} />
        <div className="donut-legend">
          {donutData.map((d) => (
            <span key={d.type} className="legend-item">
              <span className="legend-dot" style={{ background: TYPE_COLORS[d.type] || "#52525b" }} />
              {d.type} ({d.count})
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ── Export helpers ──────────────────────────────────────────────── */
function exportJSON(data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "ambiguity_results.json";
  a.click();
}
function exportCSV(data) {
  const headers = ["text", "ambiguity_type", "top_reading", "confidence", "disagreement", "conflict", "status"];
  const rows = data.map((r) => headers.map((h) => JSON.stringify(r[h] ?? "")).join(","));
  const blob = new Blob([[headers.join(","), ...rows].join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "ambiguity_results.csv";
  a.click();
}

/* ── Main App ────────────────────────────────────────────────────── */
export default function App() {
  const [activeTab, setActiveTab] = useState("resolver");
  const [format, setFormat] = useState("text");
  const [rawText, setRawText] = useState(SAMPLES.text);
  const [context, setContext] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [filterType, setFilterType] = useState("all");
  const [filterStatus, setFilterStatus] = useState("all");
  const fileRef = useRef(null);

  function loadSample() {
    setRawText(format === "csv" ? SAMPLES.csv : SAMPLES.text);
    setContext(SAMPLES.context);
    setError("");
  }

  async function handleFileUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    setFormat("csv");
    setRawText(text);
    setError("");
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!rawText.trim() || loading) return;
    setLoading(true);
    setError("");
    try {
      const body = { raw_text: rawText, format };
      if (context.trim()) body.context = context.trim();
      const res = await fetch("/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to process");
      setResults(data.results || []);
      setFilterType("all");
      setFilterStatus("all");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  /* Filtered results */
  const filtered = results.filter((r) => {
    const typeOk = filterType === "all" || r.ambiguity_type === filterType;
    const statOk = filterStatus === "all" || r.status === filterStatus;
    return typeOk && statOk;
  });

  const allTypes = [...new Set(results.map((r) => r.ambiguity_type))];
  const allStatuses = [...new Set(results.map((r) => r.status))];

  return (
    <main className="app-root">
      {/* ── Top bar ── */}
      <header className="topbar">
        <div>
          <h1 className="app-title">Ambiguity Resolver</h1>
          <p className="app-sub">Detect, analyse, and route ambiguous text with LLM self-consistency.</p>
        </div>
        <nav className="tab-nav">
          {["resolver", "evaluation"].map((t) => (
            <button
              key={t}
              className={`tab-btn ${activeTab === t ? "active" : ""}`}
              onClick={() => setActiveTab(t)}
            >
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </nav>
      </header>

      {activeTab === "evaluation" ? (
        <EvaluationTab />
      ) : (
        <div className="resolver-layout">
          {/* ════ LEFT PANEL ════ */}
          <aside className="left-panel card">
            <p className="panel-title">Input</p>

            {/* Format toggle */}
            <div className="toggle-group">
              {["text", "csv"].map((f) => (
                <button
                  key={f}
                  className={`toggle-btn ${format === f ? "active" : ""}`}
                  onClick={() => { setFormat(f); setError(""); }}
                >
                  {f === "text" ? "Pasted Text" : "CSV"}
                </button>
              ))}
            </div>

            {/* Text input */}
            <form onSubmit={handleSubmit} className="input-form">
              <label className="field-label">
                {format === "csv" ? "CSV records" : "Text records (one per line)"}
              </label>
              <textarea
                rows={6}
                placeholder={format === "csv" ? 'CSV with a "text" column…' : "One record per line…"}
                value={rawText}
                onChange={(e) => setRawText(e.target.value)}
              />

              {/* Context box */}
              <label className="field-label">
                Context <span className="field-hint">(pasted text, chat history, doc metadata…)</span>
              </label>
              <textarea
                rows={3}
                placeholder="Optional: paste previous messages, document excerpt, or metadata…"
                value={context}
                onChange={(e) => setContext(e.target.value)}
              />

              {error && <span className="error-text">{error}</span>}

              <div className="btn-row">
                <button type="button" className="ghost-btn" onClick={loadSample}>
                  Load sample
                </button>
                <label className="ghost-btn" style={{ cursor: "pointer" }}>
                  Upload CSV
                  <input ref={fileRef} type="file" accept=".csv,text/csv" onChange={handleFileUpload} hidden />
                </label>
                <button type="submit" className="primary-btn" disabled={loading || !rawText.trim()}>
                  {loading ? (
                    <span className="spinner-row"><span className="spinner" /> Processing…</span>
                  ) : (
                    "Process"
                  )}
                </button>
              </div>
            </form>

            {/* Empty state */}
            {!rawText.trim() && (
              <div className="empty-state">
                <span className="empty-icon">📋</span>
                <p>Paste text or upload a CSV to begin.</p>
              </div>
            )}
          </aside>

          {/* ════ MAIN AREA ════ */}
          <div className="main-area">
            {results.length === 0 ? (
              <div className="empty-full card">
                <span className="empty-icon">🔍</span>
                <p>Results will appear here after processing.</p>
              </div>
            ) : (
              <>
                {/* Summary cards + donut */}
                <SummaryCards results={results} />

                {/* Filter bar + export */}
                <div className="filter-bar card">
                  <div className="filter-group">
                    <select
                      className="filter-select"
                      value={filterType}
                      onChange={(e) => setFilterType(e.target.value)}
                    >
                      <option value="all">All types</option>
                      {allTypes.map((t) => <option key={t} value={t}>{t}</option>)}
                    </select>
                    <select
                      className="filter-select"
                      value={filterStatus}
                      onChange={(e) => setFilterStatus(e.target.value)}
                    >
                      <option value="all">All statuses</option>
                      {allStatuses.map((s) => <option key={s} value={s}>{s}</option>)}
                    </select>
                    <span className="filter-count">{filtered.length} / {results.length} records</span>
                  </div>
                  <div className="export-group">
                    <button className="ghost-btn" onClick={() => exportJSON(results)}>Export JSON</button>
                    <button className="ghost-btn" onClick={() => exportCSV(results)}>Export CSV</button>
                  </div>
                </div>

                {/* Result cards */}
                <div className="results-list">
                  {filtered.length === 0 ? (
                    <div className="empty-full card">
                      <p>No records match the current filters.</p>
                    </div>
                  ) : (
                    filtered.map((r, i) => <ResultCard key={i} r={r} idx={i} />)
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </main>
  );
}

import { useEffect, useState } from "react";

const FINAL_SYSTEM = {
  key: "uncertainty_resolver",
  label: "Uncertainty Resolver (spaCy + LLM)",
  color: "#1D2D44",
};

const TYPES = [
  "lexical", "syntactic", "referential", "scope", "semantic",
  "pragmatic", "underspecified", "ellipsis", "temporal_numeric", "cultural_domain", "none",
];

const TYPE_COLORS = {
  lexical: "#4338ca", syntactic: "#c2410c", referential: "#15803d",
  scope: "#be185d", semantic: "#7e22ce", pragmatic: "#b45309",
  underspecified: "#475569", ellipsis: "#0f766e", temporal_numeric: "#be123c",
  cultural_domain: "#1d4ed8", none: "#15803d",
};

function fmt(v) { return v == null ? "—" : v.toFixed(3); }

function Bar({ value, color = "#818cf8", label }) {
  const w = Math.round((value || 0) * 100);
  return (
    <div style={{ display: "grid", gridTemplateColumns: "90px 1fr 48px", gap: "0.5rem", alignItems: "center", fontSize: "0.82rem" }}>
      <span style={{ color: "var(--muted)" }}>{label}</span>
      <div style={{ height: 8, background: "var(--bg3)", borderRadius: 999, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${w}%`, background: color, borderRadius: 999, transition: "width 0.4s" }} />
      </div>
      <span style={{ color: "var(--text)", fontWeight: 600, textAlign: "right" }}>{fmt(value)}</span>
    </div>
  );
}

function LineChart({ title, data, color, xKey, yKey, xLabel, yLabel, W = 460, H = 220, PAD = 40 }) {
  const toX = (v) => PAD + v * (W - 2 * PAD);
  const toY = (v) => H - PAD - v * (H - 2 * PAD);
  const ticks = [0, 0.25, 0.5, 0.75, 1.0];
  const pts = (data || []).map((p) => `${toX(p[xKey])},${toY(p[yKey])}`).join(" ");
  return (
    <div>
      {title && <p className="col-title" style={{ marginBottom: "0.5rem" }}>{title}</p>}
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 10 }}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD} y1={toY(t)} x2={W - PAD} y2={toY(t)} stroke="#D5C8A8" strokeDasharray="4 4" />
            <text x={PAD - 4} y={toY(t) + 4} fill="#1D2D44" fontSize={9} textAnchor="end">{(t * 100).toFixed(0)}</text>
            <line x1={toX(t)} y1={PAD} x2={toX(t)} y2={H - PAD} stroke="#D5C8A8" strokeDasharray="4 4" />
            <text x={toX(t)} y={H - PAD + 14} fill="#1D2D44" fontSize={9} textAnchor="middle">{(t * 100).toFixed(0)}</text>
          </g>
        ))}
        <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#567C8D" />
        <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#567C8D" />
        {xLabel && <text x={W / 2} y={H - 4} fill="#000000" fontSize={10} textAnchor="middle">{xLabel}</text>}
        {yLabel && <text x={10} y={H / 2} fill="#000000" fontSize={10} textAnchor="middle" transform={`rotate(-90 10 ${H / 2})`}>{yLabel}</text>}
        {data && data.length > 0 && (
          <g>
            <polyline fill="none" stroke={color} strokeWidth={2.5} strokeLinejoin="round" points={pts} />
            {data.map((p, i) => (
              <circle key={i} cx={toX(p[xKey])} cy={toY(p[yKey])} r={3.5} fill={color}>
                <title>{`${xKey}=${p[xKey].toFixed(2)}, ${yKey}=${p[yKey].toFixed(2)}`}</title>
              </circle>
            ))}
          </g>
        )}
      </svg>
    </div>
  );
}

function ReliabilityDiagram({ bins, color, W = 460, H = 220, PAD = 40 }) {
  const toX = (v) => PAD + v * (W - 2 * PAD);
  const toY = (v) => H - PAD - v * (H - 2 * PAD);
  const binW = (W - 2 * PAD) / 10 - 4;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 10 }}>
      <line x1={toX(0)} y1={toY(0)} x2={toX(1)} y2={toY(1)} stroke="#567C8D" strokeDasharray="6 4" />
      <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#567C8D" />
      <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#567C8D" />
      <text x={W / 2} y={H - 4} fill="#000000" fontSize={10} textAnchor="middle">Mean Confidence</text>
      <text x={10} y={H / 2} fill="#000000" fontSize={10} textAnchor="middle" transform={`rotate(-90 10 ${H / 2})`}>Accuracy</text>
      {(bins || []).map((b, bi) => {
        if (b.count === 0) return null;
        const x = toX(b.lo) + 2;
        const yTop = toY(b.accuracy);
        const yBot = toY(0);
        return (
          <rect key={bi} x={x} y={yTop} width={Math.max(binW, 4)} height={yBot - yTop} fill={color} opacity={0.75} rx={2}>
            <title>{`conf=[${b.lo}-${b.hi}] acc=${(b.accuracy * 100).toFixed(1)}% n=${b.count}`}</title>
          </rect>
        );
      })}
    </svg>
  );
}

const TH = {
  textAlign: "left", padding: "0.55rem 0.75rem",
  fontSize: "0.72rem", textTransform: "uppercase", letterSpacing: "0.06em",
  color: "var(--muted)", fontWeight: 700, borderBottom: "1px solid var(--bg3)",
};
const TD = { padding: "0.55rem 0.75rem", verticalAlign: "middle" };

function FailureCases({ failures }) {
  const [open, setOpen] = useState(false);
  if (!failures || failures.length === 0) {
    return <p style={{ color: "var(--muted)", fontSize: "0.88rem" }}>No failure cases.</p>;
  }
  return (
    <div>
      <button className="ghost-btn" onClick={() => setOpen((v) => !v)} style={{ marginBottom: "0.75rem" }}>
        {open ? "Hide" : "Show"} {failures.length} failure case{failures.length !== 1 ? "s" : ""}
      </button>
      {open && (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.8rem" }}>
            <thead>
              <tr>
                {["ID", "Category", "Text", "Gold type", "Pred type", "Gold reading", "Pred reading", "Conf", "Reason"].map((h) => (
                  <th key={h} style={TH}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {failures.map((f, i) => (
                <tr key={i} style={{ borderBottom: "1px solid var(--bg3)" }}>
                  <td style={TD}>{f.id}</td>
                  <td style={{ ...TD, color: f.category === "context_conflicting" ? "#eab308" : "var(--muted)" }}>{f.category}</td>
                  <td style={{ ...TD, maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{f.text}</td>
                  <td style={{ ...TD, color: TYPE_COLORS[f.gold_type] || "var(--muted)" }}>{f.gold_type}</td>
                  <td style={{ ...TD, color: TYPE_COLORS[f.pred_type] || "var(--red)" }}>{f.pred_type}</td>
                  <td style={{ ...TD, color: "var(--muted)", maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{f.gold_reading}</td>
                  <td style={{ ...TD, color: "var(--red)", maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{f.pred_reading}</td>
                  <td style={TD}>{(f.confidence * 100).toFixed(0)}%</td>
                  <td style={{ ...TD, color: "var(--muted)" }}>{f.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

const EVAL_CACHE_KEY = "ambiguity_resolver_eval_cache_v5";
let memoryEvalCache = null;

function clearPreviousEvalCache() {
  memoryEvalCache = null;
  try {
    for (let i = sessionStorage.length - 1; i >= 0; i--) {
      const k = sessionStorage.key(i);
      if (k && k.startsWith("ambiguity_resolver_eval_cache")) {
        sessionStorage.removeItem(k);
      }
    }
  } catch {}
}

function UploadedEvaluationView({ results, sourceName }) {
  const [downloading, setDownloading] = useState(false);
  const total = results.length;
  const conflictCount = results.filter((r) => Boolean(r.conflict)).length;
  const ambCount = results.filter((r) => r.ambiguity_type !== "none" && !r.conflict).length;
  const unambCount = results.filter((r) => r.ambiguity_type === "none" && !r.conflict).length;
  const autoCount = results.filter((r) => r.status === "AUTO_ACCEPT").length;
  const altCount = results.filter((r) => r.status === "ACCEPT_WITH_ALTERNATIVES").length;
  const reviewCount = results.filter((r) => r.status === "REVIEW").length;
  const clarifiedCount = results.filter((r) => Boolean(r.clarified_by_user)).length;
  const avgConf = total ? results.reduce((s, r) => s + (Number(r.confidence) || 0), 0) / total : 0;

  const summaryCards = [
    { label: "Total Sentences", value: String(total), color: FINAL_SYSTEM.color },
    { label: "Ambiguous", value: String(ambCount), color: "#c2410c" },
    { label: "Context-Conflict", value: String(conflictCount), color: "#b91c1c" },
    { label: "Unambiguous", value: String(unambCount), color: "#15803d" },
    { label: "Avg Confidence", value: fmt(avgConf), color: "#4338ca" },
    { label: "Auto-Accept", value: `${Math.round((autoCount / (total || 1)) * 100)}%`, color: "#15803d" },
  ];

  const typeStats = TYPES.map((t) => {
    const items = results.filter((r) => r.ambiguity_type === t);
    const count = items.length;
    const avg = count ? items.reduce((s, r) => s + (Number(r.confidence) || 0), 0) / count : null;
    const conflicts = items.filter((r) => r.conflict).length;
    return { type: t, count, share: total ? count / total : 0, avg, conflicts };
  }).filter((row) => row.count > 0);

  async function downloadReport() {
    if (downloading) return;
    setDownloading(true);
    try {
      const res = await fetch("/export/resolved.pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ results }),
      });
      if (!res.ok) return;
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "evaluation_report.pdf";
      a.click();
      URL.revokeObjectURL(a.href);
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.1rem" }}>
      <div className="card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.75rem" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: "1.1rem", fontWeight: 800 }}>
            Final Evaluation Result — {total} sentence{total === 1 ? "" : "s"} ({sourceName})
          </h2>
          <p style={{ margin: "0.2rem 0 0", color: "var(--muted)", fontSize: "0.88rem" }}>
            {FINAL_SYSTEM.label} · Ambiguous: {ambCount} · Context-conflicting: {conflictCount} · Unambiguous: {unambCount}
          </p>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", flexShrink: 0 }}>
          <button
            type="button"
            className="ghost-btn"
            onClick={downloadReport}
            disabled={downloading}
          >
            {downloading ? "Generating PDF…" : "Download Report"}
          </button>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(145px, 1fr))", gap: "0.75rem" }}>
        {summaryCards.map((c) => (
          <div className="summary-card" key={c.label}>
            <span className="summary-value" style={{ color: c.color }}>{c.value}</span>
            <span className="summary-label">{c.label}</span>
          </div>
        ))}
      </div>

      <div className="card">
        <p className="col-title" style={{ marginBottom: "0.75rem", color: FINAL_SYSTEM.color }}>
          Final Metrics Breakdown
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
          <Bar label="Auto-Accept" value={autoCount / (total || 1)} color="#15803d" />
          <Bar label="Alternatives" value={altCount / (total || 1)} color="#b45309" />
          <Bar label="Review" value={reviewCount / (total || 1)} color="#b91c1c" />
          <Bar label="Conflicts" value={conflictCount / (total || 1)} color="#b91c1c" />
          <Bar label="Clarified" value={clarifiedCount / (total || 1)} color="#0369a1" />
          <Bar label="Avg Conf" value={avgConf} color={FINAL_SYSTEM.color} />
        </div>
      </div>

      <div className="card">
        <p className="col-title" style={{ marginBottom: "0.75rem" }}>Per-type breakdown</p>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.84rem" }}>
            <thead>
              <tr>
                <th style={TH}>Type</th>
                <th style={TH}>Count</th>
                <th style={TH}>Share</th>
                <th style={TH}>Conflicts</th>
                <th style={{ ...TH, color: FINAL_SYSTEM.color }}>Avg Conf</th>
              </tr>
            </thead>
            <tbody>
              {typeStats.map((row) => (
                <tr key={row.type} style={{ borderBottom: "1px solid var(--bg3)" }}>
                  <td style={TD}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.45rem" }}>
                      <span style={{ width: 8, height: 8, borderRadius: "50%", background: TYPE_COLORS[row.type], display: "inline-block", flexShrink: 0 }} />
                      <span>{row.type}</span>
                    </div>
                  </td>
                  <td style={TD}>{row.count}</td>
                  <td style={TD}>{`${(row.share * 100).toFixed(1)}%`}</td>
                  <td style={TD}>{row.conflicts}</td>
                  <td style={{ ...TD, fontWeight: 700, color: FINAL_SYSTEM.color }}>{fmt(row.avg)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <p className="col-title" style={{ marginBottom: "0.75rem", color: FINAL_SYSTEM.color }}>
          Sentence Evaluation Results ({total})
        </p>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82rem" }}>
            <thead>
              <tr>
                {["#", "Input Text", "Category", "Type", "Top Reading", "Conf", "Status"].map((h) => (
                  <th key={h} style={TH}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {results.map((r, i) => {
                const cat = r.conflict ? "context_conflicting" : r.ambiguity_type === "none" ? "unambiguous" : "ambiguous";
                return (
                  <tr key={r.id || i} style={{ borderBottom: "1px solid var(--bg3)" }}>
                    <td style={TD}>{i + 1}</td>
                    <td style={{ ...TD, maxWidth: 220 }}>{r.text}</td>
                    <td style={{ ...TD, color: cat === "context_conflicting" ? "#b45309" : "var(--muted)" }}>{cat}</td>
                    <td style={{ ...TD, color: TYPE_COLORS[r.ambiguity_type] || "var(--muted)", fontWeight: 600 }}>{r.ambiguity_type}</td>
                    <td style={{ ...TD, maxWidth: 240 }}>{r.clarified_by_user ? (r.resolved_text || r.top_reading) : r.top_reading}</td>
                    <td style={TD}>{`${Math.round((Number(r.confidence) || 0) * 100)}%`}</td>
                    <td style={TD}>{r.status}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default function EvaluationTab({ results = [], sourceName = "Uploaded Dataset", clarificationScore = null }) {
  const hasCurrent = Array.isArray(results) && results.length > 0;
  const [data, setData] = useState(() => memoryEvalCache);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  function runEval(forceRefresh = true) {
    if (!forceRefresh && memoryEvalCache) {
      setData(memoryEvalCache);
      return;
    }
    clearPreviousEvalCache();
    setData(null);
    setLoading(true);
    setError("");
    fetch("/evaluate?refresh=true")
      .then((r) => { if (!r.ok) throw new Error("Evaluation failed"); return r.json(); })
      .then((d) => {
        memoryEvalCache = d;
        try { sessionStorage.setItem(EVAL_CACHE_KEY, JSON.stringify(d)); } catch {}
        setData(d);
        setLoading(false);
      })
      .catch((e) => { setError(e.message); setLoading(false); });
  }

  useEffect(() => {
    if (hasCurrent) {
      clearPreviousEvalCache();
      return;
    }
    if (!memoryEvalCache) {
      runEval(true);
    }
  }, [hasCurrent]);

  if (hasCurrent) {
    return <UploadedEvaluationView results={results} sourceName={sourceName} />;
  }

  if (loading) return (
    <div className="card" style={{ display: "flex", alignItems: "center", gap: "0.75rem", padding: "2rem" }}>
      <span className="spinner" />
      <span style={{ color: "var(--muted)" }}>Running evaluation on held-out test set…</span>
    </div>
  );

  if (error) return <div className="card"><span className="error-text">{error}</span></div>;
  if (!data) return null;

  const cats = data.dataset?.categories || {};
  const m = data.systems?.[FINAL_SYSTEM.key];
  if (!m) return <div className="card"><span className="error-text">Final evaluation result unavailable.</span></div>;

  const summaryCards = [
    { label: "Detection F1", value: fmt(m.detection_f1), color: FINAL_SYSTEM.color },
    { label: "Precision", value: fmt(m.detection_precision), color: "#4338ca" },
    { label: "Recall", value: fmt(m.detection_recall), color: "#15803d" },
    { label: "Resolution Acc", value: fmt(m.resolution_accuracy), color: FINAL_SYSTEM.color },
    { label: "FPR (Unambig) ↓", value: fmt(m.false_positive_rate), color: "#b45309" },
    { label: "ECE ↓", value: fmt(m.ece), color: "#0369a1" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.1rem" }}>
      {/* ── Header ── */}
      <div className="card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.75rem" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: "1.1rem", fontWeight: 800 }}>
            Final Evaluation Result — {data.dataset?.total} held-out sentences
          </h2>
          <p style={{ margin: "0.2rem 0 0", color: "var(--muted)", fontSize: "0.88rem" }}>
            {FINAL_SYSTEM.label} · Ambiguous: {cats.ambiguous} · Context-conflicting: {cats.context_conflicting} · Unambiguous: {cats.unambiguous}
          </p>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", flexShrink: 0 }}>
          <a
            href="/export/evaluation.pdf"
            download="evaluation_results.pdf"
            className="ghost-btn"
            style={{ textDecoration: "none", display: "inline-flex", alignItems: "center" }}
          >
            Download PDF
          </a>
          <button className="ghost-btn" onClick={() => runEval(true)}>Re-run</button>
        </div>
      </div>

      {/* ── Single Final Result Summary Cards ── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(145px, 1fr))", gap: "0.75rem" }}>
        {summaryCards.map((c) => (
          <div className="summary-card" key={c.label}>
            <span className="summary-value" style={{ color: c.color }}>{c.value}</span>
            <span className="summary-label">{c.label}</span>
          </div>
        ))}
      </div>

      {/* ── Interactive MCQ Clarification Score (if answered) ── */}
      {clarificationScore && (
        <div className="card" style={{ borderColor: "#22c55e" }}>
          <p className="col-title" style={{ color: "#22c55e", marginBottom: "0.65rem" }}>
            Interactive MCQ Clarification Score ({clarificationScore.answered_count} Questions)
          </p>
          <div style={{ display: "flex", gap: "1.5rem", flexWrap: "wrap", fontSize: "0.88rem" }}>
            <span><strong>Pre-Clarification Acc:</strong> {fmt(clarificationScore.pre_clarification_accuracy)}</span>
            <span><strong>Post-Clarification Acc:</strong> <span style={{ color: "#22c55e", fontWeight: 700 }}>{fmt(clarificationScore.post_clarification_accuracy)}</span></span>
            <span><strong>Accuracy Gain Δ:</strong> <span style={{ color: "#22c55e", fontWeight: 700 }}>+{fmt(clarificationScore.accuracy_gain)}</span></span>
            <span><strong>Confidence:</strong> {fmt(clarificationScore.pre_confidence_avg)} → {fmt(clarificationScore.post_confidence_avg)}</span>
            <span><strong>Conflicts Resolved:</strong> {clarificationScore.resolved_conflicts}</span>
          </div>
        </div>
      )}

      {/* ── Final Metrics Breakdown ── */}
      <div className="card">
        <p className="col-title" style={{ marginBottom: "0.75rem", color: FINAL_SYSTEM.color }}>
          Final Metrics Breakdown
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
          <Bar label="Detection F1" value={m.detection_f1} color={FINAL_SYSTEM.color} />
          <Bar label="Precision" value={m.detection_precision} color="#4338ca" />
          <Bar label="Recall" value={m.detection_recall} color="#15803d" />
          <Bar label="Accuracy" value={m.resolution_accuracy} color={FINAL_SYSTEM.color} />
          <Bar label="Normal Acc" value={m.resolution_accuracy_normal} color="#0369a1" />
          <Bar label="Conflict Acc" value={m.resolution_accuracy_conflict} color="#b45309" />
          <Bar label="FPR ↓" value={m.false_positive_rate} color="#b91c1c" />
          <Bar label="ECE ↓" value={m.ece} color="#b91c1c" />
        </div>
      </div>

      {/* ── Simulated User Clarification Evaluation ── */}
      {data.simulated_user_clarification && (
        <div className="card">
          <p className="col-title" style={{ marginBottom: "0.75rem" }}>
            Simulated User Clarification (Never-Ask vs. Selective-Ask vs. Always-Ask)
          </p>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82rem" }}>
              <thead>
                <tr>
                  <th style={TH}>Simulated User</th>
                  <th style={TH}>Policy</th>
                  <th style={TH}>Acc Before</th>
                  <th style={{ ...TH, color: FINAL_SYSTEM.color }}>Acc After</th>
                  <th style={TH}>Questions / 100</th>
                  <th style={TH}>Share Needed</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { uKey: "always_correct", uLabel: "Always correct" },
                  { uKey: "ten_percent_wrong", uLabel: "10% wrong" },
                ].flatMap(({ uKey, uLabel }) =>
                  [
                    { pKey: "never_ask", pLabel: "Never ask" },
                    { pKey: "selective_ask", pLabel: "Selective ask (Ours)" },
                    { pKey: "always_ask", pLabel: "Always ask" },
                  ].map(({ pKey, pLabel }) => {
                    const row = data.simulated_user_clarification?.[uKey]?.[pKey];
                    if (!row) return null;
                    const isOurs = pKey === "selective_ask";
                    return (
                      <tr key={`${uKey}-${pKey}`} style={{ borderBottom: "1px solid var(--bg3)", fontWeight: isOurs ? 700 : 400 }}>
                        <td style={TD}>{uLabel}</td>
                        <td style={{ ...TD, color: isOurs ? FINAL_SYSTEM.color : "var(--text)" }}>{pLabel}</td>
                        <td style={TD}>{fmt(row.accuracy_before)}</td>
                        <td style={{ ...TD, color: FINAL_SYSTEM.color }}>{fmt(row.accuracy_after)}</td>
                        <td style={TD}>{row.questions_per_100.toFixed(1)}</td>
                        <td style={TD}>{`${(row.share_questions_needed * 100).toFixed(1)}%`}</td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Per-type P/R/F1 ── */}
      <div className="card">
        <p className="col-title" style={{ marginBottom: "0.75rem" }}>Per-type precision / recall / F1</p>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.84rem" }}>
            <thead>
              <tr>
                <th style={TH}>Type</th>
                <th style={TH}>Precision</th>
                <th style={TH}>Recall</th>
                <th style={{ ...TH, color: FINAL_SYSTEM.color }}>F1</th>
              </tr>
            </thead>
            <tbody>
              {TYPES.map((t) => {
                const pt = m.per_type?.[t] || {};
                return (
                  <tr key={t} style={{ borderBottom: "1px solid var(--bg3)" }}>
                    <td style={TD}>
                      <div style={{ display: "flex", alignItems: "center", gap: "0.45rem" }}>
                        <span style={{ width: 8, height: 8, borderRadius: "50%", background: TYPE_COLORS[t], display: "inline-block", flexShrink: 0 }} />
                        <span>{t}</span>
                      </div>
                    </td>
                    <td style={TD}>{fmt(pt.precision)}</td>
                    <td style={TD}>{fmt(pt.recall)}</td>
                    <td style={{ ...TD, fontWeight: 700, color: FINAL_SYSTEM.color }}>{fmt(pt.f1)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Charts row ── */}
      <div className="charts-grid">
        <div className="card">
          <LineChart
            title="Accuracy vs Coverage"
            xKey="coverage" yKey="accuracy"
            xLabel="Coverage →" yLabel="Accuracy"
            data={m.accuracy_vs_coverage || []}
            color={FINAL_SYSTEM.color}
          />
        </div>

        <div className="card">
          <p className="col-title" style={{ marginBottom: "0.5rem" }}>
            Reliability diagram (ECE = {fmt(m.ece)})
          </p>
          <ReliabilityDiagram bins={m.reliability_diagram || []} color={FINAL_SYSTEM.color} />
        </div>
      </div>

      {/* ── Single Failure Cases Card ── */}
      <div className="card">
        <p className="col-title" style={{ marginBottom: "0.5rem", color: FINAL_SYSTEM.color }}>
          Failure cases ({m.failure_count ?? (m.failure_cases || []).length})
        </p>
        <FailureCases failures={m.failure_cases || []} />
      </div>

      {/* ── Steering Check Card ── */}
      {data.steering_check && (
        <div className="card">
          <p className="col-title" style={{ marginBottom: "0.75rem", color: FINAL_SYSTEM.color }}>
            Steering Check (Controllability, Quality Preserved, Faithfulness & Simulated Reviewer)
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.75rem", marginBottom: "0.9rem" }}>
            <div style={{ background: "var(--bg)", padding: "0.75rem", borderRadius: 10, border: "1px solid var(--bg3)" }}>
              <div style={{ fontSize: "0.76rem", color: "var(--muted)", fontWeight: 700 }}>Controllability (α: 0 → 1)</div>
              <div style={{ fontSize: "0.85rem", marginTop: "0.25rem" }}>
                Sim Δ: <strong>+{fmt(data.steering_check.controllability?.similarity_delta)}</strong> · r = <strong>{fmt(data.steering_check.controllability?.similarity_correlation)}</strong>
              </div>
              <div style={{ fontSize: "0.85rem" }}>
                Prob Δ: <strong>+{fmt(data.steering_check.controllability?.probability_delta)}</strong> · r = <strong>{fmt(data.steering_check.controllability?.probability_correlation)}</strong>
              </div>
              <div style={{ fontSize: "0.8rem", color: "var(--muted)", marginTop: "0.2rem" }}>
                Avg Flip α: <strong>{data.steering_check.controllability?.avg_flip_alpha == null ? "—" : data.steering_check.controllability.avg_flip_alpha.toFixed(2)}</strong>
              </div>
            </div>

            <div style={{ background: "var(--bg)", padding: "0.75rem", borderRadius: 10, border: "1px solid var(--bg3)" }}>
              <div style={{ fontSize: "0.76rem", color: "var(--muted)", fontWeight: 700 }}>Quality Preserved (Frozen Encoder)</div>
              <div style={{ fontSize: "0.85rem", marginTop: "0.25rem" }}>
                Encoder Frozen: <strong>{data.steering_check.quality_preserved?.encoder_frozen ? "Yes" : "No"}</strong>
              </div>
              <div style={{ fontSize: "0.85rem" }}>
                Pair Ranking ρ: <strong>{fmt(data.steering_check.quality_preserved?.pair_ranking_spearman)}</strong>
              </div>
            </div>

            <div style={{ background: "var(--bg)", padding: "0.75rem", borderRadius: 10, border: "1px solid var(--bg3)" }}>
              <div style={{ fontSize: "0.76rem", color: "var(--muted)", fontWeight: 700 }}>2D Projection Faithfulness</div>
              <div style={{ fontSize: "0.85rem", marginTop: "0.25rem" }}>
                Trustworthiness: <strong>{fmt(data.steering_check.faithfulness?.trustworthiness)}</strong>
              </div>
              <div style={{ fontSize: "0.85rem" }}>
                k-NN Overlap (k={data.steering_check.faithfulness?.k}): <strong>{fmt(data.steering_check.faithfulness?.knn_overlap)}</strong>
              </div>
            </div>
          </div>

          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82rem" }}>
              <thead>
                <tr>
                  <th style={TH}>Simulated Reviewer</th>
                  <th style={TH}>Alpha</th>
                  <th style={{ ...TH, color: FINAL_SYSTEM.color }}>Resolution Acc</th>
                  <th style={TH}>ECE ↓</th>
                  <th style={TH}>Status Changed</th>
                </tr>
              </thead>
              <tbody>
                {(data.steering_check.simulated_reviewer || []).map((row) => (
                  <tr key={row.alpha} style={{ borderBottom: "1px solid var(--bg3)" }}>
                    <td style={TD}>{row.alpha === 0 ? "No steering (baseline)" : "Steer toward gold"}</td>
                    <td style={TD}>{row.alpha.toFixed(1)}</td>
                    <td style={{ ...TD, fontWeight: 700, color: FINAL_SYSTEM.color }}>{fmt(row.resolution_accuracy)}</td>
                    <td style={TD}>{fmt(row.ece)}</td>
                    <td style={TD}>{row.status_changed_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

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

const EVAL_CACHE_KEY = "ambiguity_resolver_eval_cache_v2";
let memoryEvalCache = null;

export default function EvaluationTab({ clarificationScore = null }) {
  const [data, setData] = useState(() => {
    if (memoryEvalCache) return memoryEvalCache;
    try {
      const saved = sessionStorage.getItem(EVAL_CACHE_KEY);
      if (saved) {
        memoryEvalCache = JSON.parse(saved);
        return memoryEvalCache;
      }
    } catch {}
    return null;
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  function runEval(forceRefresh = false) {
    if (!forceRefresh && memoryEvalCache) {
      setData(memoryEvalCache);
      return;
    }
    setLoading(true);
    setError("");
    fetch(forceRefresh ? "/evaluate?refresh=true" : "/evaluate")
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
    if (!data) runEval(false);
  }, []);

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
    </div>
  );
}

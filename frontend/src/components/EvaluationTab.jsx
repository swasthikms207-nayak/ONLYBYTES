import { Fragment, useEffect, useState } from "react";

const SYSTEMS = [
  { key: "most_frequent_sense", label: "Most Frequent Sense", color: "#71717a" },
  { key: "plain_llm", label: "Plain LLM (no uncertainty)", color: "#eab308" },
  { key: "llm_only", label: "LLM Only (no spaCy)", color: "#38bdf8" },
  { key: "uncertainty_resolver", label: "spaCy + LLM (Uncertainty)", color: "#22c55e" },
];

const TYPES = [
  "lexical", "syntactic", "referential", "scope", "semantic",
  "pragmatic", "underspecified", "ellipsis", "temporal_numeric", "cultural_domain", "none",
];

const TYPE_COLORS = {
  lexical: "#818cf8", syntactic: "#fb923c", referential: "#34d399",
  scope: "#f472b6", semantic: "#c084fc", pragmatic: "#facc15",
  underspecified: "#94a3b8", ellipsis: "#2dd4bf", temporal_numeric: "#fb7185",
  cultural_domain: "#60a5fa", none: "#4ade80",
};

/* ── Helpers ─────────────────────────────────────────────────────── */
function pct(v) { return v == null ? "—" : `${(v * 100).toFixed(1)}%`; }
function fmt(v) { return v == null ? "—" : v.toFixed(3); }

function Bar({ value, color = "#818cf8", label }) {
  const w = Math.round((value || 0) * 100);
  return (
    <div style={{ display: "grid", gridTemplateColumns: "56px 1fr 42px", gap: "0.5rem", alignItems: "center", fontSize: "0.8rem" }}>
      <span style={{ color: "var(--muted)" }}>{label}</span>
      <div style={{ height: 7, background: "var(--bg3)", borderRadius: 999, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${w}%`, background: color, borderRadius: 999, transition: "width 0.4s" }} />
      </div>
      <span style={{ color: "var(--muted)", textAlign: "right" }}>{fmt(value)}</span>
    </div>
  );
}

/* ── SVG Line chart ──────────────────────────────────────────────── */
function LineChart({ title, series, xKey, yKey, xLabel, yLabel, W = 460, H = 220, PAD = 40 }) {
  const toX = (v) => PAD + v * (W - 2 * PAD);
  const toY = (v) => H - PAD - v * (H - 2 * PAD);
  const ticks = [0, 0.25, 0.5, 0.75, 1.0];
  return (
    <div>
      {title && <p style={{ fontSize: "0.78rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.07em", color: "var(--muted)", marginBottom: "0.5rem" }}>{title}</p>}
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", background: "var(--bg)", border: "1px solid var(--bg3)", borderRadius: 10 }}>
        {/* Grid lines */}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD} y1={toY(t)} x2={W - PAD} y2={toY(t)} stroke="#27272a" strokeDasharray="4 4" />
            <text x={PAD - 4} y={toY(t) + 4} fill="#71717a" fontSize={9} textAnchor="end">{(t * 100).toFixed(0)}</text>
            <line x1={toX(t)} y1={PAD} x2={toX(t)} y2={H - PAD} stroke="#27272a" strokeDasharray="4 4" />
            <text x={toX(t)} y={H - PAD + 14} fill="#71717a" fontSize={9} textAnchor="middle">{(t * 100).toFixed(0)}</text>
          </g>
        ))}
        {/* Axes */}
        <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#3f3f46" />
        <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#3f3f46" />
        {xLabel && <text x={W / 2} y={H - 4} fill="#a1a1aa" fontSize={10} textAnchor="middle">{xLabel}</text>}
        {yLabel && <text x={10} y={H / 2} fill="#a1a1aa" fontSize={10} textAnchor="middle" transform={`rotate(-90 10 ${H / 2})`}>{yLabel}</text>}
        {/* Series */}
        {series.map(({ key, color, data }) => {
          if (!data || data.length === 0) return null;
          const pts = data.map((p) => `${toX(p[xKey])},${toY(p[yKey])}`).join(" ");
          return (
            <g key={key}>
              <polyline fill="none" stroke={color} strokeWidth={2.5} strokeLinejoin="round" points={pts} />
              {data.map((p, i) => (
                <circle key={i} cx={toX(p[xKey])} cy={toY(p[yKey])} r={3.5} fill={color}>
                  <title>{`${xKey}=${p[xKey].toFixed(2)}, ${yKey}=${p[yKey].toFixed(2)}`}</title>
                </circle>
              ))}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/* ── Reliability diagram (ECE) ───────────────────────────────────── */
function ReliabilityDiagram({ systems, W = 460, H = 240, PAD = 40 }) {
  const toX = (v) => PAD + v * (W - 2 * PAD);
  const toY = (v) => H - PAD - v * (H - 2 * PAD);
  const binW = (W - 2 * PAD) / 10 - 2;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", background: "var(--bg)", border: "1px solid var(--bg3)", borderRadius: 10 }}>
      {/* Perfect calibration diagonal */}
      <line x1={toX(0)} y1={toY(0)} x2={toX(1)} y2={toY(1)} stroke="#3f3f46" strokeDasharray="6 4" />
      {/* Axes */}
      <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#3f3f46" />
      <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#3f3f46" />
      <text x={W / 2} y={H - 4} fill="#a1a1aa" fontSize={10} textAnchor="middle">Mean Confidence</text>
      <text x={10} y={H / 2} fill="#a1a1aa" fontSize={10} textAnchor="middle" transform={`rotate(-90 10 ${H / 2})`}>Accuracy</text>
      {/* Bins per system */}
      {systems.map(({ key, color }, si) => {
        const bins = systems[si].bins || [];
        return bins.map((b, bi) => {
          if (b.count === 0) return null;
          const x = toX(b.lo) + si * 3;
          const yTop = toY(b.accuracy);
          const yBot = toY(0);
          return (
            <rect key={`${key}-${bi}`} x={x} y={yTop} width={Math.max(binW / SYSTEMS.length, 2)}
              height={yBot - yTop} fill={color} opacity={0.65} rx={1}>
              <title>{`${key} conf=[${b.lo}-${b.hi}] acc=${(b.accuracy * 100).toFixed(1)}% n=${b.count}`}</title>
            </rect>
          );
        });
      })}
    </svg>
  );
}

/* ── Per-type table ──────────────────────────────────────────────── */
function PerTypeTable({ systems }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82rem" }}>
        <thead>
          <tr>
            <th style={TH}>Type</th>
            {SYSTEMS.map((s) => (
              <th key={s.key} colSpan={3} style={{ ...TH, color: s.color }}>{s.label}</th>
            ))}
          </tr>
          <tr>
            <th style={TH} />
            {SYSTEMS.map((s) => (
              <Fragment key={s.key}>
                <th style={TH}>P</th>
                <th style={TH}>R</th>
                <th style={{ ...TH, color: s.color }}>F1</th>
              </Fragment>
            ))}
          </tr>
        </thead>
        <tbody>
          {TYPES.map((t) => (
            <tr key={t} style={{ borderBottom: "1px solid var(--bg3)" }}>
              <td style={{ ...TD, display: "flex", alignItems: "center", gap: "0.4rem", padding: "0.6rem 0.75rem" }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: TYPE_COLORS[t], display: "inline-block", flexShrink: 0 }} />
                {t}
              </td>
              {SYSTEMS.map((s) => {
                const pt = systems[s.key]?.per_type?.[t] || {};
                return (
                  <>
                    <td key={`${s.key}-p`} style={TD}>{fmt(pt.precision)}</td>
                    <td key={`${s.key}-r`} style={TD}>{fmt(pt.recall)}</td>
                    <td key={`${s.key}-f`} style={{ ...TD, fontWeight: 700, color: s.color }}>{fmt(pt.f1)}</td>
                  </>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const TH = {
  textAlign: "left", padding: "0.55rem 0.75rem",
  fontSize: "0.72rem", textTransform: "uppercase", letterSpacing: "0.06em",
  color: "var(--muted)", fontWeight: 700, borderBottom: "1px solid var(--bg3)",
};
const TD = { padding: "0.55rem 0.75rem", verticalAlign: "middle" };

/* ── Failure cases table ─────────────────────────────────────────── */
function FailureCases({ failures }) {
  const [open, setOpen] = useState(false);
  if (!failures || failures.length === 0) return <p style={{ color: "var(--muted)", fontSize: "0.88rem" }}>No failure cases.</p>;
  return (
    <div>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{ background: "transparent", border: "1px solid var(--border)", borderRadius: 8, color: "var(--text)", padding: "0.4rem 0.9rem", fontSize: "0.84rem", fontWeight: 600, cursor: "pointer", marginBottom: "0.75rem" }}
      >
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

/* ── Summary metrics table ───────────────────────────────────────── */
function SummaryTable({ systems }) {
  const rows = [
    { label: "Detection Precision", key: "detection_precision" },
    { label: "Detection Recall", key: "detection_recall" },
    { label: "Detection F1", key: "detection_f1" },
    { label: "FPR (Unambiguous)", key: "false_positive_rate" },
    { label: "Resolution Accuracy", key: "resolution_accuracy" },
    { label: "Resolution Acc (normal)", key: "resolution_accuracy_normal" },
    { label: "Resolution Acc (conflict)", key: "resolution_accuracy_conflict" },
    { label: "ECE ↓", key: "ece" },
  ];
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
        <thead>
          <tr>
            <th style={TH}>Metric</th>
            {SYSTEMS.map((s) => <th key={s.key} style={{ ...TH, color: s.color }}>{s.label}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ label, key }) => (
            <tr key={key} style={{ borderBottom: "1px solid var(--bg3)" }}>
              <td style={{ ...TD, fontWeight: 600 }}>{label}</td>
              {SYSTEMS.map((s) => {
                const v = systems[s.key]?.[key];
                const isBest = key !== "false_positive_rate" && key !== "ece"
                  ? SYSTEMS.every((os) => (systems[os.key]?.[key] ?? 0) <= (v ?? 0))
                  : SYSTEMS.every((os) => (systems[os.key]?.[key] ?? 1) >= (v ?? 1));
                return (
                  <td key={s.key} style={{ ...TD, fontWeight: isBest ? 700 : 400, color: isBest ? s.color : "var(--text)" }}>
                    {fmt(v)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ── Main EvaluationTab ──────────────────────────────────────────── */
export default function EvaluationTab() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeSystem, setActiveSystem] = useState("uncertainty_resolver");

  function runEval() {
    setLoading(true);
    setError("");
    fetch("/evaluate")
      .then((r) => { if (!r.ok) throw new Error("Evaluation failed"); return r.json(); })
      .then((d) => { setData(d); setLoading(false); })
      .catch((e) => { setError(e.message); setLoading(false); });
  }

  useEffect(() => { runEval(); }, []);

  if (loading) return (
    <div className="card" style={{ display: "flex", alignItems: "center", gap: "0.75rem", padding: "2rem" }}>
      <span className="spinner" />
      <span style={{ color: "var(--muted)" }}>Running evaluation on {data?.dataset?.total ?? 40} test sentences…</span>
    </div>
  );

  if (error) return (
    <div className="card"><span className="error-text">{error}</span></div>
  );

  if (!data) return null;

  const sys = data.systems || {};
  const cats = data.dataset?.categories || {};
  const typeDist = data.dataset?.type_distribution || {};

  // Prepare reliability diagram data (inject system key into each)
  const relSeries = SYSTEMS.map((s) => ({
    ...s,
    bins: sys[s.key]?.reliability_diagram || [],
  }));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.1rem" }}>

      {/* ── Header ── */}
      <div className="card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.75rem" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: "1.1rem", fontWeight: 800 }}>
            Evaluation — {data.dataset?.total} held-out sentences
          </h2>
          <p style={{ margin: "0.2rem 0 0", color: "var(--muted)", fontSize: "0.88rem" }}>
            Ambiguous: {cats.ambiguous} · Context-conflicting: {cats.context_conflicting} · Unambiguous: {cats.unambiguous}
          </p>
        </div>
        <button className="ghost-btn" onClick={runEval} style={{ flexShrink: 0 }}>Re-run</button>
      </div>

      {/* ── Summary metrics table ── */}
      <div className="card">
        <p className="col-title" style={{ marginBottom: "0.75rem" }}>Summary metrics</p>
        <SummaryTable systems={sys} />
      </div>

      {/* ── Ablation: spaCy + LLM vs LLM Only ── */}
      {data.ablation_spacy_contribution && (
        <div className="card">
          <p className="col-title" style={{ marginBottom: "0.65rem" }}>
            Ablation: spaCy + LLM vs. LLM Only (spaCy Contribution)
          </p>
          <div style={{ display: "flex", gap: "1.5rem", flexWrap: "wrap", fontSize: "0.88rem" }}>
            <span>
              <strong>Detection F1 Δ:</strong>{" "}
              <span style={{ color: "#22c55e", fontWeight: 700 }}>
                {data.ablation_spacy_contribution.delta_f1 >= 0 ? "+" : ""}
                {fmt(data.ablation_spacy_contribution.delta_f1)}
              </span>
            </span>
            <span>
              <strong>Resolution Accuracy Δ:</strong>{" "}
              <span style={{ color: "#22c55e", fontWeight: 700 }}>
                {data.ablation_spacy_contribution.delta_accuracy >= 0 ? "+" : ""}
                {fmt(data.ablation_spacy_contribution.delta_accuracy)}
              </span>
            </span>
            <span>
              <strong>FPR Δ:</strong>{" "}
              <span style={{ color: "#38bdf8", fontWeight: 700 }}>
                {data.ablation_spacy_contribution.delta_fpr >= 0 ? "+" : ""}
                {fmt(data.ablation_spacy_contribution.delta_fpr)}
              </span>
            </span>
          </div>
        </div>
      )}

      {/* ── Per-type P/R/F1 ── */}
      <div className="card">
        <p className="col-title" style={{ marginBottom: "0.75rem" }}>Per-type precision / recall / F1</p>
        <PerTypeTable systems={sys} />
      </div>

      {/* ── Charts row ── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.1rem" }}>

        {/* Accuracy vs Coverage */}
        <div className="card">
          <LineChart
            title="Accuracy vs Coverage"
            xKey="coverage" yKey="accuracy"
            xLabel="Coverage →" yLabel="Accuracy"
            series={SYSTEMS.map((s) => ({ ...s, data: sys[s.key]?.accuracy_vs_coverage || [] }))}
          />
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem", marginTop: "0.75rem", fontSize: "0.78rem" }}>
            {SYSTEMS.map((s) => (
              <span key={s.key} style={{ display: "flex", alignItems: "center", gap: "0.35rem", color: "var(--muted)" }}>
                <span style={{ width: 9, height: 9, borderRadius: "50%", background: s.color, display: "inline-block" }} />
                {s.label}
              </span>
            ))}
          </div>
        </div>

        {/* Reliability diagram */}
        <div className="card">
          <p className="col-title" style={{ marginBottom: "0.5rem" }}>Reliability diagram (ECE)</p>
          <ReliabilityDiagram systems={relSeries} />
          <div style={{ display: "flex", gap: "1rem", marginTop: "0.65rem", fontSize: "0.78rem", flexWrap: "wrap" }}>
            {SYSTEMS.map((s) => (
              <span key={s.key} style={{ color: s.color, fontWeight: 700 }}>
                {s.label}: ECE={fmt(sys[s.key]?.ece)}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* ── Metric bars per system ── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1.1rem" }}>
        {SYSTEMS.map((s) => {
          const m = sys[s.key];
          if (!m) return null;
          return (
            <div key={s.key} className="card">
              <p style={{ fontWeight: 700, fontSize: "0.88rem", color: s.color, marginBottom: "0.75rem" }}>{s.label}</p>
              <div style={{ display: "flex", flexDirection: "column", gap: "0.55rem" }}>
                <Bar label="F1" value={m.detection_f1} color={s.color} />
                <Bar label="Accuracy" value={m.resolution_accuracy} color={s.color} />
                <Bar label="FPR ↓" value={m.false_positive_rate} color="#ef4444" />
                <Bar label="ECE ↓" value={m.ece} color="#ef4444" />
                <Bar label="Conflict" value={m.resolution_accuracy_conflict} color="#eab308" />
              </div>
            </div>
          );
        })}
      </div>

      {/* ── Failure cases ── */}
      {SYSTEMS.map((s) => {
        const fails = sys[s.key]?.failure_cases || [];
        return (
          <div key={s.key} className="card">
            <p className="col-title" style={{ marginBottom: "0.5rem", color: s.color }}>
              {s.label} — Failure cases
            </p>
            <FailureCases failures={fails} />
          </div>
        );
      })}
    </div>
  );
}

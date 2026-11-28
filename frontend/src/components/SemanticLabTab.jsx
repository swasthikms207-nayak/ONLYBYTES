import { useEffect, useMemo, useRef, useState } from "react";

const TYPE_COLORS = {
  lexical: "#818cf8",
  syntactic: "#fb923c",
  referential: "#34d399",
  scope: "#f472b6",
  semantic: "#c084fc",
  pragmatic: "#facc15",
  underspecified: "#94a3b8",
  ellipsis: "#2dd4bf",
  temporal_numeric: "#fb7185",
  cultural_domain: "#60a5fa",
  none: "#4ade80",
};

const STATUS_STYLE = {
  AUTO_ACCEPT: { color: "#4ade80", border: "#22c55e", bg: "rgba(34,197,94,0.14)" },
  ACCEPT_WITH_ALTERNATIVES: { color: "#facc15", border: "#eab308", bg: "rgba(234,179,8,0.14)" },
  REVIEW: { color: "#f87171", border: "#ef4444", bg: "rgba(239,68,68,0.14)" },
};

const CARD_DARK = {
  background: "#131b2e",
  border: "1px solid #26344d",
  borderRadius: 14,
  padding: "1rem",
  color: "#e2e8f0",
};

const INPUT_DARK = {
  background: "#0d1321",
  border: "1px solid #334155",
  borderRadius: 8,
  color: "#f8fafc",
  padding: "0.45rem 0.7rem",
  fontSize: "0.86rem",
};

function HighlightSpan({ text, span }) {
  if (!span) return <span>{text}</span>;
  const idx = text.toLowerCase().indexOf(span.toLowerCase());
  if (idx === -1) return <span>{text}</span>;
  const before = text.slice(0, idx);
  const match = text.slice(idx, idx + span.length);
  const after = text.slice(idx + span.length);
  return (
    <span>
      {before}
      <mark style={{ background: "rgba(250, 204, 21, 0.25)", color: "#fde047", padding: "0.08rem 0.3rem", borderRadius: 5, fontWeight: 700 }}>
        {match}
      </mark>
      {after}
    </span>
  );
}

function MeterBar({ label, value, color = "#60a5fa" }) {
  const pct = Math.max(0, Math.min(100, Math.round((value || 0) * 100)));
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 120px 44px", gap: "0.5rem", alignItems: "center", fontSize: "0.8rem" }}>
      <span style={{ color: "#cbd5e1", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={label}>
        {label}
      </span>
      <div style={{ height: 8, background: "#0d1321", borderRadius: 999, overflow: "hidden", border: "1px solid #26344d" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 999, transition: "width 0.18s ease" }} />
      </div>
      <span style={{ textAlign: "right", fontWeight: 700, color: "#f8fafc" }}>{pct}%</span>
    </div>
  );
}

export default function SemanticLabTab({ results = [] }) {
  const defaultIdx = Math.max(0, results.findIndex((r) => r.ambiguity_type !== "none"));
  const [selectedRecordIdx, setSelectedRecordIdx] = useState(defaultIdx);
  const [initData, setInitData] = useState(null);
  const [steerData, setSteerData] = useState(null);
  const [targetReading, setTargetReading] = useState("");
  const [alpha, setAlpha] = useState(0.5);
  const [loadingInit, setLoadingInit] = useState(false);
  const [error, setError] = useState("");
  const [sweeping, setSweeping] = useState(false);
  const sweepRef = useRef(null);

  async function runAnalyze(targetText) {
    if (!targetText || !targetText.trim()) return;
    setLoadingInit(true);
    setError("");
    setSteerData(null);
    try {
      const res = await fetch("/lab/init", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentence: targetText.trim(), context: null }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to initialize Semantic Lab");
      setInitData(data);
      const firstReading = data.readings?.[0]?.reading || "";
      setTargetReading(firstReading);
      setAlpha(data.auto_alpha ?? 0.5);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingInit(false);
    }
  }

  // Automatically load and analyze from uploaded/entered results
  useEffect(() => {
    if (!results || results.length === 0) return;
    const idx = Math.max(0, results.findIndex((r) => r.ambiguity_type !== "none"));
    const rec = results[idx];
    setSelectedRecordIdx(idx);
    if (rec?.text) {
      runAnalyze(rec.text);
    }
  }, [results]);

  const isUnambiguous = initData?.ambiguity_type === "none";

  // Debounced /lab/steer on alpha or targetReading changes (~100 ms) for ambiguous prompts
  useEffect(() => {
    if (!initData?.sentence_id || !targetReading || isUnambiguous) return;
    const timer = setTimeout(async () => {
      try {
        const res = await fetch("/lab/steer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sentence_id: initData.sentence_id,
            target_reading: targetReading,
            alpha,
          }),
        });
        const data = await res.json();
        if (res.ok) setSteerData(data);
      } catch {
        // Keep last valid state on transient error
      }
    }, 100);
    return () => clearTimeout(timer);
  }, [initData?.sentence_id, targetReading, alpha, isUnambiguous]);

  // Cleanup sweep timer on unmount
  useEffect(() => () => { if (sweepRef.current) clearInterval(sweepRef.current); }, []);

  function startSweep() {
    if (sweeping) {
      clearInterval(sweepRef.current);
      setSweeping(false);
      return;
    }
    setSweeping(true);
    let cur = 0.0;
    setAlpha(0.0);
    sweepRef.current = setInterval(() => {
      cur = Math.round((cur + 0.05) * 100) / 100;
      if (cur >= 1.0) {
        setAlpha(1.0);
        clearInterval(sweepRef.current);
        setSweeping(false);
      } else {
        setAlpha(cur);
      }
    }, 120);
  }

  // Compute SVG bounding box over all 2D points
  const svgGeom = useMemo(() => {
    if (!initData) return null;
    const W = 480;
    const H = 360;
    const PAD = 38;
    const pts = [
      initData.sentence_point,
      ...(initData.readings || []).map((r) => r.point_2d),
      ...(initData.nearby_training_examples || []).map((n) => n.point_2d),
      ...(steerData?.steered_point_2d ? [steerData.steered_point_2d] : []),
    ].filter(Boolean);

    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const spanX = Math.max(0.05, maxX - minX);
    const spanY = Math.max(0.05, maxY - minY);

    const toX = (x) => PAD + ((x - minX) / spanX) * (W - 2 * PAD);
    const toY = (y) => H - PAD - ((y - minY) / spanY) * (H - 2 * PAD);
    return { W, H, toX, toY };
  }, [initData, steerData]);

  const activeSims = steerData?.similarities_to_readings ||
    (initData?.readings || []).map((r) => ({ reading: r.reading, similarity: r.embedding_similarity }));
  const activeProbs = steerData?.fused_reading_probabilities ||
    (initData?.readings || []).map((r) => ({ reading: r.reading, probability: r.probability }));
  const activeConf = steerData?.calibrated_confidence ?? (initData ? 1.0 - initData.auto_alpha : 0);
  const activeStatus = steerData?.status || (activeConf > 0.8 ? "AUTO_ACCEPT" : activeConf >= 0.5 ? "ACCEPT_WITH_ALTERNATIVES" : "REVIEW");
  const stStyle = STATUS_STYLE[activeStatus] || STATUS_STYLE.ACCEPT_WITH_ALTERNATIVES;
  const steeredPt = steerData?.steered_point_2d || initData?.sentence_point;

  return (
    <div style={{ background: "#0d1321", color: "#e2e8f0", borderRadius: 16, padding: "1.1rem", border: "1px solid #1e293b", display: "flex", flexDirection: "column", gap: "1rem" }}>
      {/* ── Top: Uploaded/Entered Record Selector + Highlighted Sentence ── */}
      <div style={{ ...CARD_DARK, display: "flex", flexDirection: "column", gap: "0.7rem" }}>
        {results.length > 0 && (
          <div style={{ display: "flex", gap: "0.6rem", alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ fontSize: "0.76rem", fontWeight: 700, textTransform: "uppercase", color: "#94a3b8" }}>
              From Uploaded / Entered Input ({results.length}):
            </span>
            <select
              style={{ ...INPUT_DARK, flex: 1, minWidth: 260 }}
              value={selectedRecordIdx}
              disabled={loadingInit}
              onChange={(e) => {
                const idx = Number(e.target.value);
                setSelectedRecordIdx(idx);
                const rec = results[idx];
                if (rec?.text) {
                  runAnalyze(rec.text);
                }
              }}
            >
              {results.map((r, i) => (
                <option key={r.id || i} value={i}>
                  #{i + 1} [{r.ambiguity_type}] {r.text}
                </option>
              ))}
            </select>
          </div>
        )}

        {initData && (
          <div style={{ background: "#0d1321", border: "1px solid #26344d", borderRadius: 8, padding: "0.55rem 0.75rem", fontSize: "0.9rem", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.5rem" }}>
            <div>
              <span style={{ color: "#94a3b8", fontSize: "0.75rem", textTransform: "uppercase", marginRight: "0.5rem" }}>
                [{initData.ambiguity_type}]
              </span>
              <HighlightSpan text={initData.sentence} span={initData.ambiguous_span} />
            </div>
            {isUnambiguous && (
              <span style={{ color: "#4ade80", fontWeight: 700, fontSize: "0.82rem" }}>
                No ambiguity found
              </span>
            )}
          </div>
        )}
        {loadingInit && <span style={{ color: "#94a3b8", fontSize: "0.84rem" }}>Analyzing sentence…</span>}
        {error && <span style={{ color: "#f87171", fontSize: "0.84rem" }}>{error}</span>}
      </div>

      {!initData ? (
        <div style={{ ...CARD_DARK, textAlign: "center", padding: "2.5rem", color: "#94a3b8" }}>
          {loadingInit ? "Loading Semantic Lab…" : "Upload a dataset or enter a prompt first to explore in Semantic Lab."}
        </div>
      ) : (
        <>
          {/* ── Middle 2-Column Layout: SVG Map (left) + Live Panels (right) ── */}
          <div style={{ display: "grid", gridTemplateColumns: "1.1fr 1fr", gap: "1rem", alignItems: "start" }}>
            {/* Left: SVG Semantic Map */}
            <div style={CARD_DARK}>
              <p style={{ margin: "0 0 0.6rem", fontSize: "0.78rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "#f8fafc" }}>
                2D Semantic Map (PCA · Frozen Sentence-Transformer)
              </p>
              {svgGeom && (
                <svg viewBox={`0 0 ${svgGeom.W} ${svgGeom.H}`} style={{ width: "100%", background: "#070b14", border: "1px solid #26344d", borderRadius: 10 }}>
                  <defs>
                    <marker id="steer-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                      <path d="M0,0 L0,6 L7,3 z" fill="#22c55e" />
                    </marker>
                  </defs>

                  {/* Nearby training examples colored by ambiguity type */}
                  {(initData.nearby_training_examples || []).map((ex) => {
                    const cx = svgGeom.toX(ex.point_2d[0]);
                    const cy = svgGeom.toY(ex.point_2d[1]);
                    const col = TYPE_COLORS[ex.ambiguity_type] || "#94a3b8";
                    return (
                      <circle key={ex.id} cx={cx} cy={cy} r={5} fill={col} opacity={0.85}>
                        <title>{`[Train #${ex.id} · ${ex.ambiguity_type}] ${ex.text} (sim=${ex.similarity})`}</title>
                      </circle>
                    );
                  })}

                  {/* Labeled reading points */}
                  {(initData.readings || []).map((r, i) => {
                    const rx = svgGeom.toX(r.point_2d[0]);
                    const ry = svgGeom.toY(r.point_2d[1]);
                    const isTarget = r.reading === targetReading;
                    return (
                      <g key={i}>
                        <rect
                          x={rx - 5}
                          y={ry - 5}
                          width={10}
                          height={10}
                          fill={isTarget ? "#38bdf8" : "#a78bfa"}
                          stroke="#ffffff"
                          strokeWidth={isTarget ? 1.5 : 0.7}
                          transform={`rotate(45 ${rx} ${ry})`}
                        >
                          <title>{r.reading}</title>
                        </rect>
                        <text x={rx + 8} y={ry + 3} fill="#cbd5e1" fontSize={9.5}>
                          {`R${i + 1}: ${r.reading.slice(0, 22)}${r.reading.length > 22 ? "…" : ""}`}
                        </text>
                      </g>
                    );
                  })}

                  {/* Arrow from original sentence to steered sentence */}
                  {steeredPt && (
                    <line
                      x1={svgGeom.toX(initData.sentence_point[0])}
                      y1={svgGeom.toY(initData.sentence_point[1])}
                      x2={svgGeom.toX(steeredPt[0])}
                      y2={svgGeom.toY(steeredPt[1])}
                      stroke="#22c55e"
                      strokeWidth={2}
                      markerEnd="url(#steer-arrow)"
                      style={{ transition: "all 0.16s ease-out" }}
                    />
                  )}

                  {/* Original sentence point */}
                  <circle
                    cx={svgGeom.toX(initData.sentence_point[0])}
                    cy={svgGeom.toY(initData.sentence_point[1])}
                    r={6.5}
                    fill="#ffffff"
                    stroke="#3b82f6"
                    strokeWidth={2}
                  >
                    <title>{`Original: ${initData.sentence}`}</title>
                  </circle>

                  {/* Steered sentence point (smoothly animated) */}
                  {steeredPt && (
                    <circle
                      cx={svgGeom.toX(steeredPt[0])}
                      cy={svgGeom.toY(steeredPt[1])}
                      r={7}
                      fill="#22c55e"
                      stroke="#fde047"
                      strokeWidth={2}
                      style={{ transition: "cx 0.16s ease-out, cy 0.16s ease-out" }}
                    >
                      <title>{`Steered (alpha=${alpha.toFixed(2)})`}</title>
                    </circle>
                  )}
                </svg>
              )}

              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.6rem", marginTop: "0.6rem", fontSize: "0.75rem", color: "#94a3b8" }}>
                <span>● White = Original</span>
                <span style={{ color: "#4ade80" }}>● Green = Steered</span>
                <span style={{ color: "#38bdf8" }}>◆ = Readings</span>
                <span>• Colored dots = Nearby train examples</span>
              </div>
            </div>

            {/* Right: Live Panels */}
            <div style={{ display: "flex", flexDirection: "column", gap: "0.85rem" }}>
              <div style={CARD_DARK}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.55rem" }}>
                  <span style={{ fontSize: "0.76rem", fontWeight: 700, textTransform: "uppercase", color: "#94a3b8" }}>
                    Confidence & Status
                  </span>
                  <span style={{ fontSize: "0.74rem", fontWeight: 700, padding: "0.18rem 0.55rem", borderRadius: 999, color: stStyle.color, border: `1px solid ${stStyle.border}`, background: stStyle.bg }}>
                    {activeStatus.replace(/_/g, " ")}
                  </span>
                </div>
                <MeterBar label="Calibrated Confidence" value={activeConf} color="#3b82f6" />
              </div>

              <div style={CARD_DARK}>
                <p style={{ margin: "0 0 0.5rem", fontSize: "0.76rem", fontWeight: 700, textTransform: "uppercase", color: "#94a3b8" }}>
                  Embedding Similarity to Readings
                </p>
                <div style={{ display: "flex", flexDirection: "column", gap: "0.45rem" }}>
                  {activeSims.map((s, i) => (
                    <MeterBar key={i} label={s.reading} value={s.similarity} color="#38bdf8" />
                  ))}
                </div>
              </div>

              <div style={CARD_DARK}>
                <p style={{ margin: "0 0 0.5rem", fontSize: "0.76rem", fontWeight: 700, textTransform: "uppercase", color: "#94a3b8" }}>
                  Fused Reading Probabilities
                </p>
                <div style={{ display: "flex", flexDirection: "column", gap: "0.45rem" }}>
                  {activeProbs.map((p, i) => (
                    <MeterBar key={i} label={p.reading} value={p.probability} color="#22c55e" />
                  ))}
                </div>
              </div>

              <div style={CARD_DARK}>
                <p style={{ margin: "0 0 0.45rem", fontSize: "0.76rem", fontWeight: 700, textTransform: "uppercase", color: "#94a3b8" }}>
                  Training Neighbor Votes
                </p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
                  {(steerData?.training_votes || []).map((v) => (
                    <span key={v.ambiguity_type} style={{ background: "#0d1321", border: `1px solid ${TYPE_COLORS[v.ambiguity_type] || "#475569"}`, color: TYPE_COLORS[v.ambiguity_type] || "#e2e8f0", borderRadius: 999, padding: "0.18rem 0.6rem", fontSize: "0.76rem", fontWeight: 600 }}>
                      {v.ambiguity_type}: {v.weight.toFixed(2)}
                    </span>
                  ))}
                  {!steerData?.training_votes?.length && <span style={{ color: "#94a3b8", fontSize: "0.8rem" }}>Adjust alpha or target to see neighbor votes.</span>}
                </div>
              </div>

              <div style={CARD_DARK}>
                <p style={{ margin: "0 0 0.4rem", fontSize: "0.76rem", fontWeight: 700, textTransform: "uppercase", color: "#94a3b8" }}>
                  Rewritten Text (Winning Reading)
                </p>
                <p style={{ margin: 0, fontSize: "0.9rem", fontWeight: 600, color: "#f8fafc" }}>
                  {steerData?.rewrite?.resolved_text || initData.readings?.[0]?.reading}
                </p>
              </div>
            </div>
          </div>

          {/* ── Bottom Controls: Target Chips, Alpha Slider with Auto Marker, Reset, Sweep ── */}
          <div style={{ ...CARD_DARK, display: "flex", flexDirection: "column", gap: "0.85rem" }}>
            <div>
              <span style={{ fontSize: "0.76rem", fontWeight: 700, textTransform: "uppercase", color: "#94a3b8", display: "block", marginBottom: "0.45rem" }}>
                Target Reading
              </span>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "0.45rem" }}>
                {(initData.readings || []).map((r, i) => {
                  const selected = r.reading === targetReading;
                  return (
                    <button
                      key={i}
                      type="button"
                      disabled={isUnambiguous}
                      onClick={() => setTargetReading(r.reading)}
                      style={{
                        background: selected ? "#3b82f6" : "#0d1321",
                        color: selected ? "#ffffff" : "#cbd5e1",
                        border: `1px solid ${selected ? "#3b82f6" : "#334155"}`,
                        borderRadius: 999,
                        padding: "0.35rem 0.75rem",
                        fontSize: "0.8rem",
                        fontWeight: 600,
                        cursor: isUnambiguous ? "not-allowed" : "pointer",
                        opacity: isUnambiguous ? 0.5 : 1,
                      }}
                    >
                      {r.reading}
                    </button>
                  );
                })}
              </div>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "1rem", flexWrap: "wrap" }}>
              <div style={{ flex: 1, minWidth: 240 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", marginBottom: "0.3rem" }}>
                  <span>Steering Strength (α = <strong>{alpha.toFixed(2)}</strong>){isUnambiguous ? " — Disabled (no ambiguity)" : ""}</span>
                  <span style={{ color: "#facc15" }}>Auto α = {initData.auto_alpha?.toFixed(2)}</span>
                </div>
                <div style={{ position: "relative", paddingTop: "0.35rem" }}>
                  <div
                    title={`Auto alpha = ${initData.auto_alpha}`}
                    style={{
                      position: "absolute",
                      left: `${(initData.auto_alpha ?? 0) * 100}%`,
                      top: 0,
                      width: 2,
                      height: 18,
                      background: "#facc15",
                      pointerEvents: "none",
                    }}
                  />
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={alpha}
                    disabled={isUnambiguous}
                    onChange={(e) => setAlpha(Number(e.target.value))}
                    style={{ width: "100%", opacity: isUnambiguous ? 0.45 : 1 }}
                  />
                </div>
              </div>

              <button
                type="button"
                disabled={isUnambiguous}
                onClick={() => setAlpha(initData.auto_alpha ?? 0.5)}
                style={{ background: "transparent", color: "#cbd5e1", border: "1px solid #334155", borderRadius: 8, padding: "0.45rem 0.85rem", fontSize: "0.82rem", fontWeight: 600, cursor: isUnambiguous ? "not-allowed" : "pointer", opacity: isUnambiguous ? 0.45 : 1 }}
              >
                Reset to auto
              </button>
              <button
                type="button"
                disabled={isUnambiguous}
                onClick={startSweep}
                style={{ background: sweeping ? "#ef4444" : "#22c55e", color: "#070b14", border: "none", borderRadius: 8, padding: "0.45rem 0.95rem", fontSize: "0.82rem", fontWeight: 700, cursor: isUnambiguous ? "not-allowed" : "pointer", opacity: isUnambiguous ? 0.45 : 1 }}
              >
                {sweeping ? "Stop Sweep" : "Sweep 0 → 1"}
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

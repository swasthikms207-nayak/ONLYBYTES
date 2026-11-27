import { useEffect, useMemo, useState } from "react";

const ALL_INTENTS = [
  "payment_reversal",
  "cancel_order",
  "refund_status",
  "account_closure",
  "change_address",
  "card_block",
  "order_status",
  "transaction_dispute",
];

const SVG_W = 520;
const SVG_H = 260;
const PAD = 24;

export default function SemanticMap({ currentQuery, candidates }) {
  const [points, setPoints] = useState([]);
  const [queryPoint, setQueryPoint] = useState(null);
  const [sourceIntent, setSourceIntent] = useState("account_closure");
  const [targetIntent, setTargetIntent] = useState("card_block");
  const [alpha, setAlpha] = useState(0);
  const [steerResult, setSteerResult] = useState(null);
  const [saveMessage, setSaveMessage] = useState("");
  const [saving, setSaving] = useState(false);

  // Load static 2D clusters once
  useEffect(() => {
    fetch("/map")
      .then((res) => res.json())
      .then((data) => setPoints(data.points || []))
      .catch(() => {});
  }, []);

  // Update highlighted query position & default source/target when currentQuery changes
  useEffect(() => {
    setSteerResult(null);
    setAlpha(0);
    setSaveMessage("");

    if (!currentQuery) {
      setQueryPoint(null);
      return;
    }

    if (candidates?.length >= 1) {
      setSourceIntent(candidates[0].intent);
      if (candidates.length >= 2) {
        setTargetIntent(candidates[1].intent);
      }
    }

    fetch(`/map?query=${encodeURIComponent(currentQuery)}`)
      .then((res) => res.json())
      .then((data) => setQueryPoint(data.query_before || null))
      .catch(() => {});
  }, [currentQuery, candidates]);

  const bounds = useMemo(() => {
    const allX = points.map((p) => p.x);
    const allY = points.map((p) => p.y);
    if (queryPoint) {
      allX.push(queryPoint.x);
      allY.push(queryPoint.y);
    }
    if (steerResult?.new_point) {
      allX.push(steerResult.new_point.x);
      allY.push(steerResult.new_point.y);
    }
    if (allX.length === 0) {
      return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
    }
    const minX = Math.min(...allX);
    const maxX = Math.max(...allX);
    const minY = Math.min(...allY);
    const maxY = Math.max(...allY);
    return {
      minX,
      maxX: maxX === minX ? minX + 1 : maxX,
      minY,
      maxY: maxY === minY ? minY + 1 : maxY,
    };
  }, [points, queryPoint, steerResult]);

  function toSvg(pt) {
    const sx =
      PAD + ((pt.x - bounds.minX) / (bounds.maxX - bounds.minX)) * (SVG_W - 2 * PAD);
    const sy =
      SVG_H - PAD - ((pt.y - bounds.minY) / (bounds.maxY - bounds.minY)) * (SVG_H - 2 * PAD);
    return { x: sx, y: sy };
  }

  async function handleAlphaChange(nextAlpha) {
    setAlpha(nextAlpha);
    setSaveMessage("");
    if (!currentQuery) return;

    try {
      const res = await fetch("/steer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: currentQuery,
          source_intent: sourceIntent,
          target_intent: targetIntent,
          alpha: nextAlpha,
        }),
      });
      if (!res.ok) return;
      const data = await res.json();
      setSteerResult(data);
    } catch {
      // ignore transient errors while dragging
    }
  }

  async function handleSaveCorrection() {
    if (!currentQuery || saving) return;
    setSaving(true);
    setSaveMessage("");

    try {
      const res = await fetch("/corrections", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: currentQuery,
          source_intent: sourceIntent,
          target_intent: targetIntent,
          alpha,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        setSaveMessage(`Saved (#${data.count})`);
      }
    } finally {
      setSaving(false);
    }
  }

  const oldPt = steerResult?.old_point || queryPoint;
  const newPt = steerResult?.new_point || null;
  const oldSvg = oldPt ? toSvg(oldPt) : null;
  const newSvg = newPt ? toSvg(newPt) : null;

  return (
    <section className="panel map-panel">
      <div className="panel-header map-header">
        <h2>Semantic Map</h2>
        {currentQuery && <span className="query-pill">{currentQuery}</span>}
      </div>

      <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="map-svg">
        <defs>
          <marker
            id="steer-arrow"
            markerWidth="8"
            markerHeight="8"
            refX="6"
            refY="3"
            orient="auto"
          >
            <path d="M0,0 L0,6 L7,3 z" fill="#f4f4f5" />
          </marker>
        </defs>

        {points.map((pt, idx) => {
          const pos = toSvg(pt);
          return (
            <circle
              key={idx}
              cx={pos.x}
              cy={pos.y}
              r={3.5}
              fill={pt.color}
              opacity={0.7}
            >
              <title>{`${pt.intent}: ${pt.text}`}</title>
            </circle>
          );
        })}

        {oldSvg && newSvg && (
          <line
            x1={oldSvg.x}
            y1={oldSvg.y}
            x2={newSvg.x}
            y2={newSvg.y}
            stroke="#f4f4f5"
            strokeWidth="2"
            markerEnd="url(#steer-arrow)"
          />
        )}

        {oldSvg && (
          <circle
            cx={oldSvg.x}
            cy={oldSvg.y}
            r={6}
            fill="none"
            stroke="#f4f4f5"
            strokeWidth="2.5"
          >
            <title>Current query (before)</title>
          </circle>
        )}

        {newSvg && (
          <circle cx={newSvg.x} cy={newSvg.y} r={5.5} fill="#f4f4f5">
            <title>Steered query (after)</title>
          </circle>
        )}
      </svg>

      <div className="steer-controls">
        <div className="steer-dropdowns">
          <label>
            Source
            <select
              value={sourceIntent}
              onChange={(e) => setSourceIntent(e.target.value)}
            >
              {ALL_INTENTS.map((intent) => (
                <option key={intent} value={intent}>
                  {intent}
                </option>
              ))}
            </select>
          </label>
          <label>
            Target
            <select
              value={targetIntent}
              onChange={(e) => setTargetIntent(e.target.value)}
            >
              {ALL_INTENTS.map((intent) => (
                <option key={intent} value={intent}>
                  {intent}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="alpha-row">
          <label htmlFor="alpha-slider">Alpha: {alpha.toFixed(2)}</label>
          <input
            id="alpha-slider"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={alpha}
            disabled={!currentQuery}
            onChange={(e) => handleAlphaChange(parseFloat(e.target.value))}
          />
          <button
            type="button"
            className="preset-btn"
            disabled={!currentQuery || saving}
            onClick={handleSaveCorrection}
          >
            Save correction
          </button>
          {saveMessage && <span className="save-status">{saveMessage}</span>}
        </div>
      </div>

      {steerResult?.probabilities && (
        <div className="steer-probs">
          <span className="metric-label">Updated Intent Probabilities</span>
          <div className="probs-grid">
            {steerResult.probabilities.slice(0, 4).map((item) => (
              <div key={item.intent} className="prob-chip">
                <span>{item.intent}</span>
                <strong>{Math.round(item.probability * 100)}%</strong>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

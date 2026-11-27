import { useEffect, useMemo, useState } from "react";

const SVG_W = 520;
const SVG_H = 260;
const PAD = 24;

export default function SemanticMap({ currentQuery, readings }) {
  const [points, setPoints] = useState([]);
  const [queryPoint, setQueryPoint] = useState(null);
  const [targetReading, setTargetReading] = useState("");
  const [alpha, setAlpha] = useState(0);
  const [steerResult, setSteerResult] = useState(null);
  const [saveMessage, setSaveMessage] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetch("/map")
      .then((res) => res.json())
      .then((data) => setPoints(data.points || []))
      .catch(() => {});
  }, []);

  useEffect(() => {
    setSteerResult(null);
    setAlpha(0);
    setSaveMessage("");

    if (!currentQuery) {
      setQueryPoint(null);
      setTargetReading("");
      return;
    }

    if (readings?.length > 0) {
      setTargetReading(readings[0].paraphrase);
    } else {
      setTargetReading(currentQuery);
    }

    fetch(`/map?query=${encodeURIComponent(currentQuery)}`)
      .then((res) => res.json())
      .then((data) => setQueryPoint(data.query_before || null))
      .catch(() => {});
  }, [currentQuery, readings]);

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
    if (!currentQuery || !targetReading) return;

    try {
      const res = await fetch("/steer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: currentQuery,
          target_reading: targetReading,
          alpha: nextAlpha,
          readings: readings || [],
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
    if (!currentQuery || !targetReading || saving) return;
    setSaving(true);
    setSaveMessage("");

    const matched = (readings || []).find((r) => r.paraphrase === targetReading);
    try {
      const res = await fetch("/corrections", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: currentQuery,
          target_intent: matched?.intent || targetReading,
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
            Target Reading
            <select
              value={targetReading}
              disabled={!currentQuery || !readings?.length}
              onChange={(e) => setTargetReading(e.target.value)}
            >
              {(readings || []).map((r, i) => (
                <option key={i} value={r.paraphrase}>
                  {r.paraphrase}
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
            disabled={!currentQuery || !targetReading}
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

      {steerResult?.similarities && (
        <div className="steer-probs">
          <span className="metric-label">Steered Reading Similarities</span>
          <div className="probs-grid">
            {steerResult.similarities.map((item, idx) => (
              <div key={idx} className="prob-chip">
                <span>{item.intent || item.paraphrase}</span>
                <strong>{item.similarity.toFixed(3)}</strong>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

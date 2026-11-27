import { useEffect, useState } from "react";

const SYSTEMS = [
  { key: "most_frequent_sense", label: "Most Frequent Sense", color: "#71717a" },
  { key: "plain_llm", label: "Plain LLM (no uncertainty)", color: "#eab308" },
  { key: "uncertainty_resolver", label: "Our Uncertainty Resolver", color: "#22c55e" },
];

export default function EvaluationTab() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/evaluate")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load evaluation results");
        return res.json();
      })
      .then((json) => setData(json))
      .catch((err) => setError(err.message));
  }, []);

  if (error) {
    return (
      <section className="panel">
        <span className="error-text">{error}</span>
      </section>
    );
  }

  if (!data) {
    return (
      <section className="panel">
        <p className="muted">Running evaluation across 40 test sentences...</p>
      </section>
    );
  }

  const systems = data.systems || {};
  const cats = data.dataset?.categories || {};

  const W = 480;
  const H = 240;
  const PAD = 36;
  const toX = (cov) => PAD + cov * (W - 2 * PAD);
  const toY = (acc) => H - PAD - acc * (H - 2 * PAD);

  return (
    <div className="eval-stack">
      <section className="panel">
        <div className="results-header">
          <div>
            <h2>Benchmark Dataset ({data.dataset?.total} Sentences)</h2>
            <p className="subtitle">
              Ambiguous: {cats.ambiguous} · Context-conflicting: {cats.context_conflicting} ·
              Unambiguous controls: {cats.unambiguous}
            </p>
          </div>
        </div>

        <div className="table-wrap">
          <table className="results-table">
            <thead>
              <tr>
                <th>System</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>FPR (Unambiguous)</th>
                <th>Resolution Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {SYSTEMS.map(({ key, label }) => {
                const m = systems[key];
                if (!m) return null;
                return (
                  <tr key={key}>
                    <td className="record-cell">{label}</td>
                    <td>{m.detection_precision.toFixed(3)}</td>
                    <td>{m.detection_recall.toFixed(3)}</td>
                    <td>{m.detection_f1.toFixed(3)}</td>
                    <td>{m.false_positive_rate.toFixed(3)}</td>
                    <td>{m.resolution_accuracy.toFixed(3)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <div className="charts-grid">
        <section className="panel">
          <h2>Metrics Comparison</h2>
          <div className="bars-list">
            {SYSTEMS.map(({ key, label, color }) => {
              const m = systems[key];
              if (!m) return null;
              return (
                <div key={key} className="bar-group">
                  <div className="bar-title">{label}</div>
                  <div className="bar-row">
                    <span>F1</span>
                    <div className="bar-track">
                      <div
                        className="bar-fill"
                        style={{ width: `${m.detection_f1 * 100}%`, background: color }}
                      />
                    </div>
                    <span>{m.detection_f1.toFixed(2)}</span>
                  </div>
                  <div className="bar-row">
                    <span>Accuracy</span>
                    <div className="bar-track">
                      <div
                        className="bar-fill"
                        style={{ width: `${m.resolution_accuracy * 100}%`, background: color }}
                      />
                    </div>
                    <span>{m.resolution_accuracy.toFixed(2)}</span>
                  </div>
                  <div className="bar-row">
                    <span>FPR</span>
                    <div className="bar-track">
                      <div
                        className="bar-fill"
                        style={{ width: `${m.false_positive_rate * 100}%`, background: "#ef4444" }}
                      />
                    </div>
                    <span>{m.false_positive_rate.toFixed(2)}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        <section className="panel">
          <h2>Accuracy vs. Coverage Curve</h2>
          <svg viewBox={`0 0 ${W} ${H}`} className="curve-svg">
            <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#3f3f46" />
            <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#3f3f46" />
            <text x={W / 2} y={H - 8} fill="#a1a1aa" fontSize="11" textAnchor="middle">
              Coverage (0 → 1)
            </text>
            <text
              x={12}
              y={H / 2}
              fill="#a1a1aa"
              fontSize="11"
              textAnchor="middle"
              transform={`rotate(-90 12 ${H / 2})`}
            >
              Accuracy
            </text>

            {SYSTEMS.map(({ key, color }) => {
              const pts = systems[key]?.accuracy_vs_coverage || [];
              if (pts.length === 0) return null;
              const poly = pts.map((p) => `${toX(p.coverage)},${toY(p.accuracy)}`).join(" ");
              return (
                <g key={key}>
                  <polyline fill="none" stroke={color} strokeWidth="2.5" points={poly} />
                  {pts.map((p, idx) => (
                    <circle
                      key={idx}
                      cx={toX(p.coverage)}
                      cy={toY(p.accuracy)}
                      r={3.5}
                      fill={color}
                    >
                      <title>{`cov=${p.coverage}, acc=${p.accuracy}`}</title>
                    </circle>
                  ))}
                </g>
              );
            })}
          </svg>
          <div className="legend-row">
            {SYSTEMS.map(({ key, label, color }) => (
              <span key={key} className="legend-item">
                <span className="legend-dot" style={{ background: color }} />
                {label}
              </span>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}

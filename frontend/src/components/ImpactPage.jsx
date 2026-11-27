import { useEffect, useState } from "react";

const SYSTEM_ORDER = [
  { key: "always_act", label: "Always ACT" },
  { key: "always_ask", label: "Always ASK" },
  { key: "confidence_only", label: "Confidence-only (0.80)" },
  { key: "risk_aware_margin", label: "Risk-aware + Margin" },
];

export default function ImpactPage() {
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/results")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load results.json");
        return res.json();
      })
      .then((data) => setResults(data))
      .catch((err) => setError(err.message));
  }, []);

  if (error) {
    return (
      <section className="panel">
        <div className="error-banner">{error}</div>
      </section>
    );
  }

  if (!results) {
    return (
      <section className="panel">
        <p className="empty-state">Loading evaluation results...</p>
      </section>
    );
  }

  const headline = results.headline || {};
  const systems = results.systems || {};
  const ablation = results.ablation || {};
  const checks = results.checks || {};
  const failureCases = results.failure_cases || [];

  return (
    <div className="impact-stack">
      {/* Headline Numbers */}
      <section className="headline-grid">
        <div className="panel headline-card">
          <span className="metric-label">Wrong Actions Prevented</span>
          <strong className="headline-num risk-low-text">
            {headline.wrong_actions_prevented ?? 0}
          </strong>
          <span className="subtitle">vs. Always ACT baseline</span>
        </div>
        <div className="panel headline-card">
          <span className="metric-label">Questions Asked</span>
          <strong className="headline-num">{headline.questions_asked ?? 0}</strong>
          <span className="subtitle">
            out of {results.dataset?.num_queries ?? 0} test queries
          </span>
        </div>
        <div className="panel headline-card">
          <span className="metric-label">Necessary Questions</span>
          <strong className="headline-num risk-medium-text">
            {headline.necessary_questions ?? 0}
          </strong>
          <span className="subtitle">
            {headline.questions_asked
              ? `${Math.round(
                  ((headline.necessary_questions || 0) / headline.questions_asked) * 100
                )}% of questions asked`
              : "0%"}
          </span>
        </div>
      </section>

      {/* Baseline Comparison Table */}
      <section className="panel">
        <div className="panel-header">
          <h2>Baseline Comparison</h2>
        </div>
        <div className="table-wrap">
          <table className="policy-table">
            <thead>
              <tr>
                <th>System</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>Intent Acc</th>
                <th>UQR</th>
                <th>Cost Error</th>
                <th>ECE</th>
              </tr>
            </thead>
            <tbody>
              {SYSTEM_ORDER.map(({ key, label }) => {
                const m = systems[key];
                if (!m) return null;
                return (
                  <tr key={key}>
                    <td className="intent-cell">{label}</td>
                    <td>{m.detection_precision.toFixed(3)}</td>
                    <td>{m.detection_recall.toFixed(3)}</td>
                    <td>{m.detection_f1.toFixed(3)}</td>
                    <td>{m.intent_accuracy.toFixed(3)}</td>
                    <td>{m.unnecessary_question_rate.toFixed(3)}</td>
                    <td>{m.cost_weighted_error.toFixed(3)}</td>
                    <td>{m.expected_calibration_error.toFixed(3)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* Ablation + Calibration Plot */}
      <div className="live-grid">
        <section className="panel">
          <div className="panel-header">
            <h2>Ablation: LLM-only vs. LLM + Margin + Steering</h2>
          </div>
          <div className="table-wrap">
            <table className="policy-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>LLM-only</th>
                  <th>+ Margin + Steering</th>
                  <th>Delta</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(ablation.difference || {}).map(([metric, delta]) => (
                  <tr key={metric}>
                    <td className="intent-cell">{metric}</td>
                    <td>{ablation.llm_only?.[metric]?.toFixed(4)}</td>
                    <td>{ablation.llm_plus_margin_plus_steering?.[metric]?.toFixed(4)}</td>
                    <td>{delta >= 0 ? `+${delta.toFixed(4)}` : delta.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>Calibration Plot</h2>
          </div>
          <img
            src="/calibration-plot"
            alt="Calibration Plot"
            className="calibration-img"
          />
        </section>
      </div>

      {/* Three Theme 5 Checks */}
      <section className="headline-grid">
        <div className="panel">
          <span className="metric-label">1. Steering Control</span>
          <div className="check-stat">
            Gain: <strong>{checks.steering_control?.similarity_gain}</strong>
          </div>
          <div className="check-stat">
            Correlation: <strong>{checks.steering_control?.correlation}</strong>
          </div>
        </div>
        <div className="panel">
          <span className="metric-label">2. Quality Preserved</span>
          <div className="check-stat">
            Acc Before: <strong>{checks.quality_preserved?.accuracy_before}</strong>
          </div>
          <div className="check-stat">
            Acc After: <strong>{checks.quality_preserved?.accuracy_after}</strong>
          </div>
        </div>
        <div className="panel">
          <span className="metric-label">3. Visualizer Faithfulness</span>
          <div className="check-stat">
            Trustworthiness:{" "}
            <strong>{checks.visualizer_faithfulness?.trustworthiness}</strong>
          </div>
          <div className="check-stat">
            Points: <strong>{checks.visualizer_faithfulness?.num_points}</strong>
          </div>
        </div>
      </section>

      {/* Failure Cases */}
      <section className="panel">
        <div className="panel-header">
          <h2>Failure Cases</h2>
        </div>
        {failureCases.length === 0 ? (
          <p className="empty-state">No failure cases recorded.</p>
        ) : (
          <div className="table-wrap">
            <table className="policy-table">
              <thead>
                <tr>
                  <th>Query</th>
                  <th>Correct Intent</th>
                  <th>Predicted Intent</th>
                  <th>Decision</th>
                  <th>Ambiguity Type</th>
                </tr>
              </thead>
              <tbody>
                {failureCases.map((fc, idx) => (
                  <tr key={idx}>
                    <td>{fc.query}</td>
                    <td>{fc.correct_intent}</td>
                    <td>{fc.predicted_intent}</td>
                    <td>{fc.decision}</td>
                    <td>{fc.ambiguity_type}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

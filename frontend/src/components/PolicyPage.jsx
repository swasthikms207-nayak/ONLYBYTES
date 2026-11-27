import { useEffect, useState } from "react";

const RISK_TIERS = ["low", "medium", "high"];
const COST_WEIGHTS = [1, 3, 10];

export default function PolicyPage({ onPolicySaved }) {
  const [policies, setPolicies] = useState({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/policy")
      .then((res) => res.json())
      .then((data) => setPolicies(data.policies || {}))
      .catch(() => setError("Failed to load policy configuration."))
      .finally(() => setLoading(false));
  }, []);

  function updateField(intent, field, value) {
    setStatus("");
    setError("");
    setPolicies((prev) => ({
      ...prev,
      [intent]: {
        ...prev[intent],
        [field]: value,
      },
    }));
  }

  async function handleSave() {
    setSaving(true);
    setStatus("");
    setError("");
    try {
      const res = await fetch("/api/policy", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ policies }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Failed to save policy");
      }
      setPolicies(data.policies || policies);
      setStatus("Policy saved. Live tab will use updated values immediately.");
      if (onPolicySaved) onPolicySaved(data.policies);
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  }

  const intentNames = Object.keys(policies);

  return (
    <section className="panel policy-panel">
      <div className="panel-header policy-header">
        <div>
          <h2>Intent Policy Table</h2>
          <p className="subtitle">
            Edit risk tier, cost weight, and confidence threshold per intent.
          </p>
        </div>
        <div className="policy-actions">
          {status && <span className="save-status">{status}</span>}
          {error && <span className="error-text">{error}</span>}
          <button
            type="button"
            className="send-btn"
            disabled={loading || saving || intentNames.length === 0}
            onClick={handleSave}
          >
            {saving ? "Saving..." : "Save Policy"}
          </button>
        </div>
      </div>

      {loading ? (
        <p className="empty-state">Loading policy...</p>
      ) : (
        <div className="table-wrap">
          <table className="policy-table">
            <thead>
              <tr>
                <th>Intent</th>
                <th>Risk Tier</th>
                <th>Cost Weight</th>
                <th>Confidence Threshold</th>
              </tr>
            </thead>
            <tbody>
              {intentNames.map((intent) => {
                const row = policies[intent];
                return (
                  <tr key={intent}>
                    <td className="intent-cell">{intent}</td>
                    <td>
                      <select
                        className={`risk-select risk-${row.risk_tier}`}
                        value={row.risk_tier}
                        onChange={(e) => updateField(intent, "risk_tier", e.target.value)}
                      >
                        {RISK_TIERS.map((tier) => (
                          <option key={tier} value={tier}>
                            {tier}
                          </option>
                        ))}
                      </select>
                    </td>
                    <td>
                      <select
                        value={row.cost_weight}
                        onChange={(e) =>
                          updateField(intent, "cost_weight", Number(e.target.value))
                        }
                      >
                        {COST_WEIGHTS.map((w) => (
                          <option key={w} value={w}>
                            {w}
                          </option>
                        ))}
                      </select>
                    </td>
                    <td>
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.05"
                        value={row.confidence_threshold}
                        onChange={(e) =>
                          updateField(
                            intent,
                            "confidence_threshold",
                            Math.min(1, Math.max(0, parseFloat(e.target.value) || 0))
                          )
                        }
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

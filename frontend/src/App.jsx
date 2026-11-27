import { useEffect, useState } from "react";
import ImpactPage from "./components/ImpactPage.jsx";
import PolicyPage from "./components/PolicyPage.jsx";
import SemanticMap from "./components/SemanticMap.jsx";

const PRESETS = [
  {
    id: "clear",
    label: "Clear",
    customerId: "CUST_1001",
    query: "Where is my order?",
  },
  {
    id: "ambiguous",
    label: "Ambiguous",
    customerId: "CUST_1003",
    query: "Sab band kar do abhi",
  },
  {
    id: "context",
    label: "Context-resolved",
    customerId: "CUST_1001",
    query: "Please reverse that duplicate charge",
  },
  {
    id: "high-risk",
    label: "High-risk",
    customerId: "CUST_1003",
    query: "Block my debit card immediately, unknown transaction!",
  },
];

const RISK_CLASS = {
  low: "risk-low",
  medium: "risk-medium",
  high: "risk-high",
};

export default function App() {
  const [activeTab, setActiveTab] = useState("live");
  const [customers, setCustomers] = useState([]);
  const [selectedCustomerId, setSelectedCustomerId] = useState("CUST_1001");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [lastQuery, setLastQuery] = useState("");
  const [decisionData, setDecisionData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/customers")
      .then((res) => res.json())
      .then((data) => {
        const list = data.customers || [];
        setCustomers(list);
        if (list.length > 0) {
          const first = list[0];
          setSelectedCustomerId(first.customer_id);
          setMessages(first.chat_history || []);
        }
      })
      .catch(() => {});
  }, []);

  function handleCustomerChange(customerId) {
    setSelectedCustomerId(customerId);
    const found = customers.find((c) => c.customer_id === customerId);
    setMessages(found?.chat_history || []);
    setDecisionData(null);
    setError("");
  }

  async function sendDecisionRequest({ customerId, query, clarificationChoice = null }) {
    if (!query.trim()) return;
    setLoading(true);
    setError("");

    const userText = clarificationChoice ? `${query} → [${clarificationChoice}]` : query;
    setMessages((prev) => [...prev, { role: "user", message: userText }]);

    try {
      const res = await fetch("/decide", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          customer_id: customerId,
          query,
          clarification_choice: clarificationChoice,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Request failed");
      }

      setDecisionData(data);
      setLastQuery(query);

      let botReply = data.reason;
      if (data.decision === "ASK" && data.clarification?.question) {
        botReply = data.clarification.question;
      } else if (data.decision === "ACT" && data.action_result) {
        botReply = data.action_result;
      }

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          message: botReply,
          clarification: data.decision === "ASK" ? data.clarification : null,
        },
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    sendDecisionRequest({ customerId: selectedCustomerId, query: q });
  }

  function handlePreset(preset) {
    if (loading) return;
    if (preset.customerId !== selectedCustomerId) {
      handleCustomerChange(preset.customerId);
    }
    setInput("");
    sendDecisionRequest({ customerId: preset.customerId, query: preset.query });
  }

  function handleClarificationOption(option) {
    if (loading || !lastQuery) return;
    sendDecisionRequest({
      customerId: selectedCustomerId,
      query: lastQuery,
      clarificationChoice: option.label,
    });
  }

  const riskClass = decisionData ? RISK_CLASS[decisionData.risk_tier] || "risk-low" : "";

  return (
    <div className="page">
      <header className="topbar">
        <div>
          <h1>ONLYBYTES</h1>
          <p className="subtitle">Bilingual Intent Ambiguity &amp; Risk-Aware Decision Engine</p>
        </div>

        <nav className="tabs">
          <button
            type="button"
            className={`tab-btn ${activeTab === "live" ? "active" : ""}`}
            onClick={() => setActiveTab("live")}
          >
            Live
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === "policy" ? "active" : ""}`}
            onClick={() => setActiveTab("policy")}
          >
            Policy
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === "impact" ? "active" : ""}`}
            onClick={() => setActiveTab("impact")}
          >
            Impact
          </button>
        </nav>

        {activeTab === "live" && (
          <div className="customer-picker">
            <label htmlFor="customer-select">Customer</label>
            <select
              id="customer-select"
              value={selectedCustomerId}
              onChange={(e) => handleCustomerChange(e.target.value)}
            >
              {customers.map((c) => (
                <option key={c.customer_id} value={c.customer_id}>
                  {c.name} ({c.customer_id})
                </option>
              ))}
            </select>
          </div>
        )}
      </header>

      {activeTab === "policy" ? (
        <PolicyPage />
      ) : activeTab === "impact" ? (
        <ImpactPage />
      ) : (
        <>

      <div className="live-grid">
        {/* Left Column: Customer Chat */}
        <section className="panel chat-panel">
          <div className="panel-header">
            <h2>Customer Chat</h2>
          </div>

          <div className="presets-row">
            {PRESETS.map((preset) => (
              <button
                key={preset.id}
                type="button"
                className="preset-btn"
                disabled={loading}
                onClick={() => handlePreset(preset)}
              >
                {preset.label}
              </button>
            ))}
          </div>

          <div className="chat-thread">
            {messages.length === 0 && (
              <p className="empty-state">Send a message or pick a preset above to begin.</p>
            )}
            {messages.map((msg, idx) => (
              <div key={idx} className={`bubble ${msg.role}`}>
                <div className="bubble-role">{msg.role === "user" ? "Customer" : "Assistant"}</div>
                <div>{msg.message}</div>
                {msg.clarification?.options?.length > 0 && (
                  <div className="clarification-options">
                    {msg.clarification.options.map((opt) => (
                      <button
                        key={opt.intent}
                        type="button"
                        className="option-btn"
                        disabled={loading}
                        onClick={() => handleClarificationOption(opt)}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>

          {error && <div className="error-banner">{error}</div>}

          <form className="chat-input-form" onSubmit={handleSubmit}>
            <input
              type="text"
              placeholder="Type a message in English or Hinglish..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <button type="submit" className="send-btn" disabled={loading || !input.trim()}>
              {loading ? "Evaluating..." : "Send"}
            </button>
          </form>
        </section>

        {/* Right Column: Decision Card */}
        <section className="panel decision-panel">
          <div className="panel-header">
            <h2>Decision Card</h2>
          </div>

          {!decisionData ? (
            <div className="empty-state">
              No decision yet. Send a query or click a preset example to inspect the policy decision.
            </div>
          ) : (
            <div className="decision-body">
              <div className="metrics-grid">
                <div className="metric-box">
                  <span className="metric-label">Ambiguity</span>
                  <span className="metric-value">
                    {decisionData.is_ambiguous ? "Ambiguous" : "Clear"}
                  </span>
                </div>
                <div className="metric-box">
                  <span className="metric-label">Ambiguity Type</span>
                  <span className="metric-value">{decisionData.type}</span>
                </div>
                <div className="metric-box">
                  <span className="metric-label">Ambiguity Score</span>
                  <span className="metric-value">
                    {decisionData.score?.toFixed(4)} (E:{decisionData.entropy?.toFixed(2)} · D:
                    {decisionData.divergence?.toFixed(2)})
                  </span>
                </div>
                <div className="metric-box">
                  <span className="metric-label">Risk Tier</span>
                  <span className={`risk-badge ${riskClass}`}>{decisionData.risk_tier}</span>
                </div>
                <div className="metric-box">
                  <span className="metric-label">Threshold</span>
                  <span className="metric-value">{decisionData.threshold}</span>
                </div>
                <div className="metric-box">
                  <span className="metric-label">Decision</span>
                  <span className={`decision-badge ${riskClass}`}>{decisionData.decision}</span>
                </div>
              </div>

              <div className="reason-box">
                <span className="metric-label">Reason</span>
                <p>{decisionData.reason}</p>
              </div>

              <div className="candidates-box">
                <span className="metric-label">Candidate Intents</span>
                <div className="candidates-list">
                  {(decisionData.candidates || []).map((c) => {
                    const pct = Math.round(c.confidence * 100);
                    return (
                      <div key={c.intent} className="candidate-row">
                        <div className="candidate-meta">
                          <span>{c.intent}</span>
                          <span>{pct}%</span>
                        </div>
                        <div className="bar-track">
                          <div
                            className={`bar-fill ${riskClass}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </section>
      </div>

      <SemanticMap
        currentQuery={lastQuery}
        candidates={decisionData?.candidates || []}
      />
        </>
      )}
    </div>
  );
}

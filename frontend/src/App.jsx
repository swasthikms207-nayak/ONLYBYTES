import { useState } from "react";
import EvaluationTab from "./components/EvaluationTab.jsx";

const SAMPLE_TEXT = [
  "Where is my order ORD_5501?",
  "Sab band kar do abhi",
  "Cancel it before dispatch",
  "Block my card",
  "Help please",
].join("\n");

const SAMPLE_CSV = `text
"Where is my order ORD_5501?"
"Sab band kar do abhi"
"Cancel it before dispatch"
"Close it now"`;

export default function App() {
  const [activeTab, setActiveTab] = useState("resolver");
  const [format, setFormat] = useState("text");
  const [rawText, setRawText] = useState(SAMPLE_TEXT);
  const [results, setResults] = useState([]);
  const [showJson, setShowJson] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function handleFormatChange(nextFormat) {
    setFormat(nextFormat);
    setError("");
    if (!rawText.trim()) {
      setRawText(nextFormat === "csv" ? SAMPLE_CSV : SAMPLE_TEXT);
    }
  }

  async function handleFileUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    setFormat("csv");
    setRawText(text);
    setError("");
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!rawText.trim() || loading) return;
    setLoading(true);
    setError("");

    try {
      const res = await fetch("/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ raw_text: rawText, format }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Failed to process batch");
      }
      setResults(data.results || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const autoAcceptCount = results.filter((r) => r.status === "AUTO_ACCEPT").length;
  const acceptAltCount = results.filter((r) => r.status === "ACCEPT_WITH_ALTERNATIVES").length;
  const reviewCount = results.filter((r) => r.status === "REVIEW").length;

  return (
    <main className="container">
      <header className="header topbar">
        <div>
          <h1>Ambiguity Resolver</h1>
          <p className="subtitle">
            Batch analyze pasted text or CSV records into structured ambiguity readings.
          </p>
        </div>
        <nav className="tabs">
          <button
            type="button"
            className={`tab-btn ${activeTab === "resolver" ? "active" : ""}`}
            onClick={() => setActiveTab("resolver")}
          >
            Resolver
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === "evaluation" ? "active" : ""}`}
            onClick={() => setActiveTab("evaluation")}
          >
            Evaluation
          </button>
        </nav>
      </header>

      {activeTab === "evaluation" ? (
        <EvaluationTab />
      ) : (
        <>

      <section className="panel">
        <div className="toolbar">
          <div className="toggle-group">
            <button
              type="button"
              className={`toggle-btn ${format === "text" ? "active" : ""}`}
              onClick={() => handleFormatChange("text")}
            >
              Pasted Text
            </button>
            <button
              type="button"
              className={`toggle-btn ${format === "csv" ? "active" : ""}`}
              onClick={() => handleFormatChange("csv")}
            >
              CSV
            </button>
          </div>

          <div className="toolbar-actions">
            <label className="file-btn">
              Upload CSV
              <input type="file" accept=".csv,text/csv" onChange={handleFileUpload} hidden />
            </label>
            <button
              type="button"
              className="secondary-btn"
              onClick={() => setRawText(format === "csv" ? SAMPLE_CSV : SAMPLE_TEXT)}
            >
              Load Sample
            </button>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="input-form">
          <textarea
            rows={6}
            placeholder={
              format === "csv"
                ? 'Paste CSV with a "text" column...'
                : "Paste one text record per line..."
            }
            value={rawText}
            onChange={(e) => setRawText(e.target.value)}
          />
          <div className="submit-row">
            {error && <span className="error-text">{error}</span>}
            <button type="submit" className="primary-btn" disabled={loading || !rawText.trim()}>
              {loading ? "Processing..." : "Process Batch"}
            </button>
          </div>
        </form>
      </section>

      {results.length > 0 && (
        <section className="panel">
          <div className="results-header">
            <div className="summary-pills">
              <span className="pill">Total: {results.length}</span>
              <span className="pill status-AUTO_ACCEPT">AUTO_ACCEPT: {autoAcceptCount}</span>
              <span className="pill status-ACCEPT_WITH_ALTERNATIVES">
                ACCEPT_WITH_ALTERNATIVES: {acceptAltCount}
              </span>
              <span className="pill status-REVIEW">REVIEW: {reviewCount}</span>
            </div>
            <button
              type="button"
              className="secondary-btn"
              onClick={() => setShowJson((v) => !v)}
            >
              {showJson ? "Table View" : "View JSON"}
            </button>
          </div>

          {showJson ? (
            <pre className="json-view">{JSON.stringify(results, null, 2)}</pre>
          ) : (
            <div className="table-wrap">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Record</th>
                    <th>Ambiguity Type</th>
                    <th>Top Reading</th>
                    <th>Alternatives</th>
                    <th>Confidence</th>
                    <th>Disagreement / Conflict</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((row, idx) => (
                    <tr key={idx}>
                      <td className="record-cell">{row.text}</td>
                      <td>
                        <code>{row.ambiguity_type}</code>
                      </td>
                      <td>{row.top_reading}</td>
                      <td>
                        {row.alternatives.length === 0 ? (
                          <span className="muted">None</span>
                        ) : (
                          <ul className="alt-list">
                            {row.alternatives.map((alt, i) => (
                              <li key={i}>
                                {alt.reading}{" "}
                                <span className="muted">({Math.round(alt.probability * 100)}%)</span>
                              </li>
                            ))}
                          </ul>
                        )}
                      </td>
                      <td>{Math.round(row.confidence * 100)}%</td>
                      <td>
                        {Math.round((row.disagreement || 0) * 100)}%
                        {row.conflict ? " · conflict" : ""}
                      </td>
                      <td>
                        <span className={`status-badge status-${row.status}`}>{row.status}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}
        </>
      )}
    </main>
  );
}

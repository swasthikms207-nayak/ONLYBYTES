import { useEffect, useState } from "react";

export default function App() {
  const [health, setHealth] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [reply, setReply] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/health")
      .then((res) => res.json())
      .then((data) => setHealth(data))
      .catch(() => setHealth({ status: "unreachable" }));
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!prompt.trim()) return;

    setLoading(true);
    setError("");
    setReply("");

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Request failed");
      }
      setReply(data.reply);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <header className="header">
        <h1>ONLYBYTES</h1>
        <p className="subtitle">FastAPI + React (Vite) Starter</p>
      </header>

      <section className="status-card">
        <span>
          Backend:{" "}
          <strong>{health ? health.status : "checking..."}</strong>
        </span>
        {health?.status === "ok" && (
          <span>
            LLM Key:{" "}
            <strong>
              {health.llm_configured ? `Configured (${health.model})` : "Not configured (.env)"}
            </strong>
          </span>
        )}
      </section>

      <form className="chat-form" onSubmit={handleSubmit}>
        <textarea
          rows={4}
          placeholder="Ask something..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <button type="submit" disabled={loading || !prompt.trim()}>
          {loading ? "Sending..." : "Send Prompt"}
        </button>
      </form>

      {error && <div className="output error">{error}</div>}
      {reply && <div className="output">{reply}</div>}
    </main>
  );
}

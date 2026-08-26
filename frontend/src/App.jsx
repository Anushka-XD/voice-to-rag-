import { useRef, useState } from "react";

const API = import.meta.env.VITE_API_URL || "";

export default function App() {
  const [recording, setRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const recRef = useRef(null);
  const chunksRef = useRef([]);

  async function submitQuery(q) {
    setLoading(true);
    setError("");
    try {
      const r = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail || r.statusText);
      setResult(body);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  async function submitAudio(blob) {
    setLoading(true);
    setError("");
    try {
      const fd = new FormData();
      fd.append("audio", blob, "clip.webm");
      const r = await fetch(`${API}/voice-query`, { method: "POST", body: fd });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail || r.statusText);
      setResult(body);
      if (body.transcript) setText(body.transcript);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  async function toggleRec() {
    if (recording) {
      recRef.current?.stop();
      setRecording(false);
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const rec = new MediaRecorder(stream);
    chunksRef.current = [];
    rec.ondataavailable = (ev) => {
      if (ev.data.size) chunksRef.current.push(ev.data);
    };
    rec.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunksRef.current, { type: rec.mimeType || "audio/webm" });
      submitAudio(blob);
    };
    recRef.current = rec;
    rec.start();
    setRecording(true);
  }

  const lat = result?.latency || {};

  return (
    <main className="page">
      <h1>VaaniX</h1>
      <p className="sub">Voice or text → grounded answer from MSMARCO-XI. Refuses when evidence is weak.</p>

      <label>
        Text query
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} />
      </label>
      <div className="row">
        <button disabled={loading} onClick={() => submitQuery(text)}>
          Ask
        </button>
        <button disabled={loading} className={recording ? "rec" : ""} onClick={toggleRec}>
          {recording ? "Stop recording" : "Record"}
        </button>
      </div>

      {loading && <p className="status">Working…</p>}
      {error && <p className="err">{error}</p>}

      {result && (
        <section className="card">
          <p>
            <strong>Status:</strong> {result.status} {result.route ? `· ${result.route}` : ""}
          </p>
          {result.transcript && (
            <p>
              <strong>Transcript:</strong> {result.transcript}
            </p>
          )}
          <p>
            <strong>Answer:</strong> {result.answer}
          </p>
          <p>
            <strong>Latency (ms)</strong> — STT {lat.stt_ms ?? "—"} · retrieval {lat.retrieval_ms ?? "—"} ·
            rerank {lat.rerank_ms ?? "—"} · LLM {lat.generation_ms ?? "—"} · total {lat.total_ms ?? "—"}
          </p>
          <h3>Sources</h3>
          <ul>
            {(result.sources || []).length === 0 && <li>None</li>}
            {(result.sources || []).map((s) => (
              <li key={s.chunk_id}>
                {s.chunk_id} ({s.language}) score {s.score}
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}

import React, { useEffect, useRef, useState } from "react";
import { apiFetch } from "../../lib/api.js";

const createSessionId = () =>
  window.crypto?.randomUUID?.()?.replace(/-/g, "") || `${Date.now()}${Math.floor(Math.random() * 10000)}`;

const formatTimestamp = (value) => {
  if (!value) return "";
  const d = new Date(value.replace(" ", "T") + (value.endsWith("Z") ? "" : "Z"));
  return Number.isNaN(d.getTime()) ? value : d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
};

const makeLabel = () => {
  const now = new Date();
  const day = now.getDate();
  const mon = now.toLocaleString("default", { month: "short" });
  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  return `${day} ${mon} · ${hh}:${mm}`;
};


/** Minimal markdown → JSX: bold, italic, inline code, bullet lines */
function renderMarkdown(text) {
  if (!text) return null;
  const lines = text.split("\n");
  return lines.map((line, li) => {
    // Parse inline: **bold**, *italic*, `code`
    const parts = [];
    const regex = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)/g;
    let last = 0, m;
    while ((m = regex.exec(line)) !== null) {
      if (m.index > last) parts.push(line.slice(last, m.index));
      if (m[2]) parts.push(<strong key={m.index}>{m[2]}</strong>);
      else if (m[3]) parts.push(<em key={m.index}>{m[3]}</em>);
      else if (m[4]) parts.push(<code key={m.index} className="inline-code">{m[4]}</code>);
      last = m.index + m[0].length;
    }
    if (last < line.length) parts.push(line.slice(last));

    // Bullet lines
    if (line.trimStart().startsWith("- ") || line.trimStart().startsWith("• ")) {
      const content = line.replace(/^[\s\-•]+/, "");
      return <div key={li} className="md-bullet">• {content.length ? renderInline(content) : parts}</div>;
    }
    // Heading lines
    if (line.startsWith("### ")) return <div key={li} className="md-h3">{line.slice(4)}</div>;
    if (line.startsWith("## "))  return <div key={li} className="md-h2">{line.slice(3)}</div>;
    if (line.startsWith("# "))   return <div key={li} className="md-h1">{line.slice(2)}</div>;
    // Empty line = spacer
    if (line.trim() === "") return <div key={li} style={{ height: "0.5em" }} />;
    return <div key={li}>{parts.length ? parts : line}</div>;
  });
}

function renderInline(text) {
  const parts = [];
  const regex = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)/g;
  let last = 0, m;
  while ((m = regex.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    if (m[2]) parts.push(<strong key={m.index}>{m[2]}</strong>);
    else if (m[3]) parts.push(<em key={m.index}>{m[3]}</em>);
    else if (m[4]) parts.push(<code key={m.index} className="inline-code">{m[4]}</code>);
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}



export default function ChatPanel({ fieldId, cropId, userId }) {
  const sessionListKey = `nava_ag_sessions_${userId}_${cropId}`;
  const activeSessionKey = `nava_ag_active_${userId}_${cropId}`;

  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState("");
  const [summary, setSummary] = useState("");
  const [history, setHistory] = useState([]);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [editName, setEditName] = useState("");
  const [showSummary, setShowSummary] = useState(false);
  
  const [animating, setAnimating] = useState(false);
  const [revealedText, setRevealedText] = useState("");
  const [pendingWords, setPendingWords] = useState([]);
  
  const messagesEnd = useRef(null);
  const inputRef = useRef(null);

  const loadPersistedSessions = () => {
    try { return JSON.parse(localStorage.getItem(sessionListKey)) || []; } catch { return []; }
  };

  const persistSessions = (list) => {
    localStorage.setItem(sessionListKey, JSON.stringify(list));
    setSessions(list);
  };

  const spawnSession = (existingList) => {
    const fresh = { id: createSessionId(), label: makeLabel(), createdAt: new Date().toISOString() };
    const next = [fresh, ...existingList];
    persistSessions(next);
    localStorage.setItem(activeSessionKey, fresh.id);
    setActiveSession(fresh.id);
    setHistory([]);
    setSummary("");
    return fresh.id;
  };

  const ensureSession = () => {
    const existing = loadPersistedSessions();
    const stored = localStorage.getItem(activeSessionKey);
    if (stored && existing.some((s) => s.id === stored)) {
      setSessions(existing);
      setActiveSession(stored);
      return stored;
    }
    return spawnSession(existing);
  };

  const refreshSummary = async (sid) => {
    try {
      const data = await apiFetch("/api/chat/summary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sid }),
      });
      setSummary(data.summary || "");
    } catch { setSummary(""); }
  };

  const refreshHistory = async (sid) => {
    try {
      const data = await apiFetch("/api/chat/history", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sid }),
      });
      setHistory(data.messages || []);
    } catch { setHistory([]); }
  };

  useEffect(() => {
    const sid = ensureSession();
    refreshSummary(sid);
    refreshHistory(sid);
  }, [cropId]);

  useEffect(() => {
    if (!animating) return;
    if (pendingWords.length === 0) {
      setAnimating(false);
      setHistory(prev => {
        const newHist = [...prev];
        const last = newHist[newHist.length - 1];
        if (last && last.isAnimating) {
          last.isAnimating = false;
          last.content = last.fullContent;
        }
        return newHist;
      });
      return;
    }

    const timer = setTimeout(() => {
      setRevealedText(prev => prev + (prev ? " " : "") + pendingWords[0]);
      setPendingWords(prev => prev.slice(1));
    }, 40);

    return () => clearTimeout(timer);
  }, [animating, pendingWords]);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, revealedText]);

  const switchSession = (sid) => {
    setActiveSession(sid);
    localStorage.setItem(activeSessionKey, sid);
    refreshSummary(sid);
    refreshHistory(sid);
    setEditingId(null);
  };

  const handleSend = async () => {
    if (!message.trim() || busy) return;
    const sid = activeSession || ensureSession();
    const userMsg = message.trim();
    setMessage("");

    if (animating) {
      setAnimating(false);
      setPendingWords([]);
      setHistory(prev => {
        const newHist = [...prev];
        const last = newHist[newHist.length - 1];
        if (last && last.isAnimating) {
          last.isAnimating = false;
          last.content = last.fullContent;
        }
        return newHist;
      });
    }

    setHistory((prev) => [...prev, { role: "user", content: userMsg, created_at: new Date().toISOString() }]);
    setBusy(true);
    try {
      const data = await apiFetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg, session_id: sid, field_id: Number(fieldId), crop_id: Number(cropId) }),
      });
      if (data.reply) {
        const wordsArr = data.reply.split(" ");
        setPendingWords(wordsArr);
        setRevealedText("");
        setAnimating(true);
        setHistory((prev) => [...prev, { role: "assistant", content: "", fullContent: data.reply, isAnimating: true, created_at: new Date().toISOString() }]);
      }
      refreshSummary(sid);
    } catch (err) {
      setHistory((prev) => [...prev, { role: "assistant", content: `Error: ${err.message}`, created_at: new Date().toISOString() }]);
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const clearCurrentSession = async () => {
    if (!activeSession || !confirm("Clear all messages in this chat?")) return;
    await apiFetch("/api/chat/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: activeSession }),
    });
    setSummary("");
    setHistory([]);
  };

  const deleteSession = async (sid, e) => {
    e?.stopPropagation();
    if (!confirm("Delete this chat session permanently?")) return;
    try {
      await apiFetch("/api/chat/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sid }),
      });
    } catch { /* backend error is ok, still delete locally */ }

    const current = loadPersistedSessions();
    const next = current.filter((s) => s.id !== sid);

    if (next.length === 0) {
      // Spawn exactly ONE new session from an empty list
      spawnSession([]);
    } else {
      persistSessions(next);
      if (sid === activeSession) {
        switchSession(next[0].id);
      }
    }
    setEditingId(null);
  };

  const startRename = (s, e) => {
    e.stopPropagation();
    setEditingId(s.id);
    setEditName(s.label);
  };

  const commitRename = (sid) => {
    if (!editName.trim()) { setEditingId(null); return; }
    const current = loadPersistedSessions();
    const next = current.map((s) => s.id === sid ? { ...s, label: editName.trim() } : s);
    persistSessions(next);
    setEditingId(null);
  };

  const newSession = () => {
    const current = loadPersistedSessions();
    spawnSession(current);
  };

  return (
    <div className="chat-wrapper">
      {/* Session rail */}
      <div className="session-rail">
        <div className="session-rail-list">
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`session-item ${s.id === activeSession ? "active" : ""}`}
              onClick={() => switchSession(s.id)}
            >
              {editingId === s.id ? (
                <input
                  className="session-rename-input"
                  value={editName}
                  autoFocus
                  onChange={(e) => setEditName(e.target.value)}
                  onBlur={() => commitRename(s.id)}
                  onKeyDown={(e) => { if (e.key === "Enter") commitRename(s.id); if (e.key === "Escape") setEditingId(null); }}
                  onClick={(e) => e.stopPropagation()}
                />
              ) : (
                <span className="session-item-label">{s.label}</span>
              )}
              <div className="session-item-actions">
                <button className="session-action-btn" onClick={(e) => startRename(s, e)} title="Rename">✏️</button>
                <button className="session-action-btn danger" onClick={(e) => deleteSession(s.id, e)} title="Delete">×</button>
              </div>
            </div>
          ))}
        </div>
        <button className="new-session-btn" onClick={newSession}>+ New Chat</button>
      </div>

      {/* Chat area */}
      <div className="chat-panel">
        {summary && (
          <div className="summary-panel" style={{ padding: "8px 16px", margin: "12px 16px 0 16px", flexShrink: 0 }}>
            <div 
              onClick={() => setShowSummary(s => !s)} 
              style={{ cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center" }}
            >
              <span className="text-xs" style={{ color: "var(--green-400)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600 }}>
                🧠 Chat Memory
              </span>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                {showSummary ? "Hide" : "Show"}
              </span>
            </div>
            {showSummary && (
              <div style={{ marginTop: 8, paddingTop: 8, borderTop: "1px solid var(--border-default)" }}>
                <p style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", whiteSpace: "pre-wrap", margin: 0 }}>{summary}</p>
              </div>
            )}
          </div>
        )}

        <div className="chat-messages">
          {history.length === 0 && (
            <div className="empty-state" style={{ padding: "40px 24px" }}>
              <div className="icon">💬</div>
              <h3>Ask NAVA anything</h3>
              <p>Crop diseases, management tips, irrigation advice — NAVA has full context of this crop's history.</p>
            </div>
          )}
          {history.map((item, i) => (
            <div key={`${item.created_at}-${i}`} className={`chat-bubble ${item.role}`}>
              <div className="bubble-content">
                {item.isAnimating ? (
                  <>{revealedText}<span style={{ animation: "blink 1.2s ease infinite", marginLeft: "2px" }}>▍</span></>
                ) : (
                  item.role === "assistant" ? renderMarkdown(item.content) : item.content
                )}
              </div>
              <div className="bubble-time">{formatTimestamp(item.created_at)}</div>
            </div>
          ))}

          {busy && (
            <div className="chat-bubble assistant typing">
              <span /><span /><span />
            </div>
          )}
          <div ref={messagesEnd} />
        </div>

        <div className="chat-footer">
          <button className="chat-clear-btn" onClick={clearCurrentSession} title="Clear chat">🗑</button>
          <input
            ref={inputRef}
            className="chat-input"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about this crop…"
            disabled={busy}
          />
          <button className="chat-send-btn" onClick={handleSend} disabled={busy || !message.trim()}>
            {busy ? <span className="spinner-sm" /> : "↑"}
          </button>
        </div>
      </div>
    </div>
  );
}

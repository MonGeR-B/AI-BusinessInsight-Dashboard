import React, { useState } from "react";
import axios from "axios";

const ChatPanel = ({ sessionId }) => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);

  const backendUrl = "http://127.0.0.1:8000";

  const sendMessage = async () => {
    if (!input.trim()) return;
    const formData = new FormData();
    formData.append("message", input);

    try {
      const res = await axios.post(
        `${backendUrl}/ai/chat?session_id=${sessionId}`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      setMessages((prev) => [
        ...prev,
        { role: "user", text: input },
        { role: "assistant", text: res.data.reply },
      ]);
      setInput("");
    } catch (err) {
      console.error(err);
      alert("Chat request failed. Check backend connection.");
    }
  };

  return (
    <div style={{ marginTop: "30px", textAlign: "left" }}>
      <h2>💬 Chat with AI</h2>
      <div
        style={{
          border: "1px solid #ccc",
          borderRadius: "8px",
          padding: "10px",
          maxHeight: "250px",
          overflowY: "auto",
          backgroundColor: "#f9f9f9",
        }}
      >
        {messages.map((m, i) => (
          <p key={i}>
            <strong>{m.role === "user" ? "You" : "AI"}:</strong> {m.text}
          </p>
        ))}
      </div>
      <div style={{ marginTop: "10px" }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your question..."
          style={{ width: "80%", padding: "6px" }}
        />
        <button onClick={sendMessage} style={{ marginLeft: "10px" }}>
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatPanel;

import { useState } from "react";
import { API } from "../api/api";

const Chatbot = () => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [botTyping, setBotTyping] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const text = input;
    setInput("");

    // Add user msg instantly
    setMessages((prev) => [...prev, { sender: "user", text }]);

    setLoading(true);
    setBotTyping(true);

    try {
      const res = await API.post("/chat", { message: text });

      const botMsg = {
        sender: "bot",
        text: formatBotMessage(res.data.reply),
      };

      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "⚠️ Unable to get a response right now.",
        },
      ]);
    }

    setLoading(false);
    setBotTyping(false);
  };

  const handleKey = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  /** Format response */
  const formatBotMessage = (text) => {
    return text
      .replace(/\*/g, "")
      .replace(/\n/g, "<br />")
      .replace(
        /(\bDosage\b|\bUse\b|\bSymptoms\b|\bFever\b|\bHerb\b)/gi,
        "<strong>$1</strong>"
      );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-100 to-green-300 flex justify-center p-6">
      <div className="w-full max-w-3xl bg-white rounded-2xl shadow-xl flex flex-col overflow-hidden">

        {/* Header */}
        <div className="bg-green-600 text-white p-5 text-xl font-bold">
          🌱 Ayurveda Chatbot
        </div>

        {/* Chat Window */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4 bg-green-50">

          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`p-4 rounded-xl max-w-[80%] leading-relaxed text-sm animate-fade ${
                msg.sender === "user"
                  ? "ml-auto bg-green-600 text-white"
                  : "mr-auto bg-white shadow border border-green-200"
              }`}
              dangerouslySetInnerHTML={{ __html: msg.text }}
            />
          ))}

          {/* Bot typing indicator */}
          {botTyping && (
            <div className="mr-auto p-3 bg-white border border-green-200 rounded-xl shadow text-gray-600 flex items-center gap-2 text-sm animate-pulse">
              <span className="w-2 h-2 bg-green-600 rounded-full animate-bounce"></span>
              <span className="w-2 h-2 bg-green-600 rounded-full animate-bounce delay-150"></span>
              <span className="w-2 h-2 bg-green-600 rounded-full animate-bounce delay-300"></span>
              <span>AyurBot is typing…</span>
            </div>
          )}
        </div>

        {/* Input Box */}
        <div className="flex items-center gap-3 p-4 border-t bg-white">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask about your health..."
            className="flex-1 p-3 rounded-xl border focus:outline-green-600"
          />
          <button
            onClick={sendMessage}
            className="bg-green-600 text-white px-6 py-2 rounded-xl hover:bg-green-700 transition-all flex items-center gap-2"
          >
            {loading ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            ) : (
              "Send"
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;

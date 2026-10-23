// Quản lý đa phiên hội thoại (Multi-Session Chat History) với localStorage
const STORAGE_KEY = "samsung_agent_sessions_v1";

let sessions = [];
let currentSessionId = null;
let isGenerating = false;
let selectedEmbeddingModel = "finetuned";

const messagesBox = document.getElementById("messages-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const toolStatusBar = document.getElementById("tool-status-bar");
const toolStatusText = document.getElementById("tool-status-text");
const sessionListContainer = document.getElementById("session-list");
const sessionCountSpan = document.getElementById("session-count");
const modelSelect = document.getElementById("model-select");

function handleModelChange(val) {
  selectedEmbeddingModel = val;
  const labels = {
    "finetuned": "🔥 Model đã Fine-tune (MultipleNegativesRankingLoss)",
    "bge-m3": "🌐 Pretrained BGE-M3 (1024D)",
    "vietnamese-bi-encoder": "🇻🇳 Pretrained BKAI Bi-Encoder (Gốc)"
  };
  showToolStatus(`⚙️ Đã chuyển sang mô hình: ${labels[val] || val}`);
  setTimeout(hideToolStatus, 2500);
}

// ==================== QUẢN LÝ SESSION & LOCALSTORAGE ====================

function loadSessionsFromStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      sessions = JSON.parse(raw);
    } else {
      sessions = [];
    }
  } catch (e) {
    console.error("Lỗi đọc sessions từ localStorage:", e);
    sessions = [];
  }

  if (sessions.length === 0) {
    createNewSession(false);
  } else {
    currentSessionId = sessions[0].id;
    renderSessionList();
    renderCurrentSessionMessages();
  }
}

function saveSessionsToStorage() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    renderSessionList();
  } catch (e) {
    console.error("Lỗi lưu sessions:", e);
  }
}

function getCurrentSession() {
  return sessions.find(s => s.id === currentSessionId);
}

function createNewSession(autoFocus = true) {
  const newId = "session_" + Date.now();
  const newSession = {
    id: newId,
    title: "Đoạn chat mới",
    createdAt: new Date().toISOString(),
    messages: [] // mảng lưu các tin nhắn {role, content, thought, sources}
  };

  sessions.unshift(newSession);
  currentSessionId = newId;
  saveSessionsToStorage();
  renderCurrentSessionMessages();

  if (autoFocus && userInput) {
    userInput.focus();
  }
}

function switchSession(sessionId) {
  if (isGenerating) {
    alert("Vui lòng đợi câu trả lời hiện tại hoàn tất trước khi chuyển cuộc trò chuyện!");
    return;
  }
  currentSessionId = sessionId;
  renderSessionList();
  renderCurrentSessionMessages();
}

function deleteSession(sessionId, event) {
  if (event) event.stopPropagation();
  
  if (sessions.length <= 1) {
    clearCurrentSession();
    return;
  }

  sessions = sessions.filter(s => s.id !== sessionId);
  if (currentSessionId === sessionId) {
    currentSessionId = sessions[0].id;
  }
  saveSessionsToStorage();
  renderCurrentSessionMessages();
}

function clearCurrentSession() {
  const session = getCurrentSession();
  if (session) {
    session.messages = [];
    session.title = "Đoạn chat mới";
    saveSessionsToStorage();
    renderCurrentSessionMessages();
  }
}

function renderSessionList() {
  if (!sessionListContainer) return;
  sessionListContainer.innerHTML = "";
  sessionCountSpan.textContent = sessions.length;

  if (sessions.length === 0) {
    sessionListContainer.innerHTML = `<div class="empty-history">Chưa có lịch sử</div>`;
    return;
  }

  sessions.forEach(sess => {
    const item = document.createElement("div");
    item.className = `session-item ${sess.id === currentSessionId ? "active" : ""}`;
    item.onclick = () => switchSession(sess.id);

    const titleSpan = document.createElement("span");
    titleSpan.className = "session-title";
    titleSpan.textContent = sess.title || "Đoạn chat mới";
    titleSpan.title = sess.title;

    const delBtn = document.createElement("button");
    delBtn.className = "session-delete-btn";
    delBtn.title = "Xoá đoạn chat này";
    delBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
        <line x1="18" y1="6" x2="6" y2="18"/>
        <line x1="6" y1="6" x2="18" y2="18"/>
      </svg>
    `;
    delBtn.onclick = (e) => deleteSession(sess.id, e);

    item.appendChild(titleSpan);
    item.appendChild(delBtn);
    sessionListContainer.appendChild(item);
  });
}

// ==================== HIỂN THỊ TIN NHẮN ====================

function renderCurrentSessionMessages() {
  messagesBox.innerHTML = "";
  const session = getCurrentSession();

  if (!session || session.messages.length === 0) {
    // Hiển thị tin nhắn chào mừng mặc định
    messagesBox.innerHTML = `
      <div class="message-row bot-row">
        <div class="avatar bot-avatar">🤖</div>
        <div class="message-bubble bot-bubble">
          <p>Xin chào! Tôi là <strong>AI Agent chuyên trách hướng dẫn sử dụng máy giặt Samsung WA12CG5886 / WA14CG5886</strong>.</p>
          <p>Tôi có thể hỗ trợ bạn:</p>
          <ul>
            <li>Tra cứu tức thì các <strong>mã lỗi</strong> (4C, 1C, 5C, Ub, dC...) và các bước xử lý chuẩn xác.</li>
            <li>Hướng dẫn <strong>lắp đặt</strong> nguồn nước, ống xả, cân bằng máy.</li>
            <li>Quy trình <strong>vệ sinh lồng giặt</strong>, bộ lọc xơ vải Magic, chất tẩy rửa an toàn.</li>
            <li>Tra cứu <strong>thông số kỹ thuật</strong> chi tiết của từng model.</li>
          </ul>
          <p><em>Hãy đặt câu hỏi bên dưới hoặc chọn gợi ý nhanh để bắt đầu!</em></p>
        </div>
      </div>
    `;
    return;
  }

  session.messages.forEach(msg => {
    if (msg.role === "user") {
      appendUserMessageDOM(msg.content);
    } else if (msg.role === "assistant") {
      appendBotMessageDOM(msg.content, msg.thought, msg.sources);
    }
  });

  autoScroll();
}

function appendUserMessageDOM(text) {
  const row = document.createElement("div");
  row.className = "message-row user-row";
  row.innerHTML = `
    <div class="avatar user-avatar">👤</div>
    <div class="message-bubble user-bubble">
      <p>${escapeHtml(text)}</p>
    </div>
  `;
  messagesBox.appendChild(row);
}

function appendBotMessageDOM(content, thought, sources) {
  const row = document.createElement("div");
  row.className = "message-row bot-row";

  let thoughtHtml = "";
  if (thought) {
    thoughtHtml = `<div class="thought-card">${escapeHtml(thought)}</div>`;
  }

  let sourcesHtml = "";
  if (sources && sources.length > 0) {
    sourcesHtml = `
      <div class="sources-box">
        <div class="source-title">📄 Tài liệu tham chiếu:</div>
        <div class="source-badges">
    `;
    sources.forEach(s => {
      sourcesHtml += `<span class="source-badge">Trang ${s.page}${s.section ? ' - ' + s.section : ''}</span>`;
    });
    sourcesHtml += `</div></div>`;
  }

  row.innerHTML = `
    <div class="avatar bot-avatar">🤖</div>
    <div class="message-bubble bot-bubble">
      <div class="thought-container">${thoughtHtml}</div>
      <div class="content-container">${marked.parse(content || "")}</div>
      <div class="sources-container">${sourcesHtml}</div>
    </div>
  `;
  messagesBox.appendChild(row);
}

// ==================== TƯƠNG TÁC GỬI NHẬN TIN NHẮN ====================

function showToolStatus(text) {
  toolStatusText.textContent = text;
  toolStatusBar.classList.remove("hidden");
}

function hideToolStatus() {
  toolStatusBar.classList.add("hidden");
}

function autoScroll() {
  messagesBox.scrollTop = messagesBox.scrollHeight;
}

function handleKeyDown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleFormSubmit(e);
  }
}

function sendQuickPrompt(promptText) {
  userInput.value = promptText;
  handleFormSubmit(new Event("submit"));
}

function escapeHtml(string) {
  const entityMap = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  };
  return String(string).replace(/[&<>"']/g, s => entityMap[s]);
}

function createLiveBotBubble() {
  const row = document.createElement("div");
  row.className = "message-row bot-row";
  row.innerHTML = `
    <div class="avatar bot-avatar">🤖</div>
    <div class="message-bubble bot-bubble" id="current-bot-bubble">
      <div class="thought-container"></div>
      <div class="content-container"><em>Đang suy nghĩ...</em></div>
      <div class="sources-container"></div>
    </div>
  `;
  messagesBox.appendChild(row);
  autoScroll();
  return row.querySelector("#current-bot-bubble");
}

async function handleFormSubmit(e) {
  if (e) e.preventDefault();
  const text = userInput.value.trim();
  if (!text || isGenerating) return;

  const session = getCurrentSession();
  if (!session) return;

  // Đặt tiêu đề cuộc trò chuyện bằng câu hỏi đầu tiên
  if (session.messages.length === 0) {
    session.title = text.length > 28 ? text.substring(0, 28) + "..." : text;
  }

  userInput.value = "";
  userInput.style.height = "auto";
  isGenerating = true;
  sendBtn.disabled = true;

  // Lưu tin nhắn User vào session và cập nhật sidebar ngay lập tức
  session.messages.push({ role: "user", content: text });
  saveSessionsToStorage();
  appendUserMessageDOM(text);
  autoScroll();

  const botBubble = createLiveBotBubble();
  const thoughtContainer = botBubble.querySelector(".thought-container");
  const contentContainer = botBubble.querySelector(".content-container");
  const sourcesContainer = botBubble.querySelector(".sources-container");

  let fullResponseText = "";
  let currentThought = "";
  let currentSources = [];

  showToolStatus("🤖 AI Agent đang lập luận...");

  // Chuẩn bị history cho backend (chỉ lấy {role, content})
  const historyForBackend = session.messages.slice(0, -1).map(m => ({
    role: m.role,
    content: m.content
  }));

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        history: historyForBackend,
        model: selectedEmbeddingModel
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP Error: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop();

      for (const block of lines) {
        if (!block.startsWith("data: ")) continue;
        const jsonStr = block.replace("data: ", "").trim();
        if (jsonStr === "[DONE]") break;

        try {
          const event = JSON.parse(jsonStr);

          if (event.type === "thought") {
            currentThought = event.message;
            const card = document.createElement("div");
            card.className = "thought-card";
            card.textContent = event.message;
            thoughtContainer.appendChild(card);
          } else if (event.type === "tool_start") {
            const toolLabels = {
              "lookup_error_code": `🔍 Đang tra cứu mã lỗi '${event.arg}' O(1)...`,
              "get_specifications": `📐 Đang trích xuất thông số kỹ thuật model ${event.arg}...`,
              "search_manual_rag": `📚 Đang tìm kiếm tài liệu 2-Stage Neural Retrieval...`
            };
            showToolStatus(toolLabels[event.tool] || `Đang kích hoạt ${event.tool}...`);
          } else if (event.type === "tool_end") {
            showToolStatus("✍️ Đang tổng hợp câu trả lời theo tài liệu...");
          } else if (event.type === "sources" && event.sources.length > 0) {
            currentSources = event.sources;
            let sourcesHtml = `
              <div class="sources-box">
                <div class="source-title">📄 Tài liệu tham chiếu:</div>
                <div class="source-badges">
            `;
            event.sources.forEach(s => {
              sourcesHtml += `<span class="source-badge">Trang ${s.page}${s.section ? ' - ' + s.section : ''}</span>`;
            });
            sourcesHtml += `</div></div>`;
            sourcesContainer.innerHTML = sourcesHtml;
          } else if (event.type === "token") {
            if (fullResponseText === "") {
              contentContainer.innerHTML = "";
            }
            fullResponseText += event.content;
            contentContainer.innerHTML = marked.parse(fullResponseText);
            autoScroll();
          } else if (event.type === "error") {
            contentContainer.innerHTML = `<p style="color:#ef4444">⚠️ ${event.message}</p>`;
          }
        } catch (err) {
          console.error("Lỗi parse SSE event:", err, jsonStr);
        }
      }
    }

    // Lưu phản hồi bot vào session
    session.messages.push({
      role: "assistant",
      content: fullResponseText,
      thought: currentThought,
      sources: currentSources
    });
    saveSessionsToStorage();

  } catch (err) {
    console.error("Lỗi kết nối:", err);
    contentContainer.innerHTML = `<p style="color:#ef4444">⚠️ Không thể kết nối với Backend hoặc Ollama. Vui lòng đảm bảo server đang chạy.</p>`;
  } finally {
    hideToolStatus();
    isGenerating = false;
    sendBtn.disabled = false;
    botBubble.removeAttribute("id");
    userInput.focus();
    autoScroll();
  }
}

// Khởi chạy khi load trang
document.addEventListener("DOMContentLoaded", () => {
  loadSessionsFromStorage();
});

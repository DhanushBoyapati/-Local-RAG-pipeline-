from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import ollama
import chromadb
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── CONFIG ────────────────────────────────────────────────────────────────────
PDF_PATH    = "data/Final_Research_Paper.pdf"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL  = "llama3"
DB_PATH     = "./chroma_db"
COLLECTION  = "pdf_collection"

app = FastAPI(title="Local RAG Chatbot", description="PDF-powered local AI assistant")

# ── VECTOR DB ─────────────────────────────────────────────────────────────────
client     = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(COLLECTION)


# ── INGESTION ─────────────────────────────────────────────────────────────────
def ingest():
    if collection.count() > 0:
        print(f"✅ DB already has {collection.count()} chunks — skipping ingestion")
        return

    print("📄 Running ingestion…")
    text = "".join(
        page.extract_text() or ""
        for page in PdfReader(PDF_PATH).pages
    )

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    ).split_text(text)

    for i, chunk in enumerate(chunks):
        emb = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)["embedding"]
        collection.add(ids=[str(i)], embeddings=[emb], documents=[chunk])

    print(f"✅ Ingestion complete — {len(chunks)} chunks stored")


ingest()


# ── REQUEST MODEL ─────────────────────────────────────────────────────────────
class Query(BaseModel):
    question: str


# ── RAG CORE ──────────────────────────────────────────────────────────────────
def ask(question: str) -> str:
    q_emb = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]

    results = collection.query(query_embeddings=[q_emb], n_results=3)
    docs    = results["documents"][0]

    if not docs:
        return "I couldn't find any relevant information in the document to answer your question."

    context = "\n\n".join(docs)

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"You are a helpful assistant. Answer the question using ONLY the context below.\n"
                f"If the answer is not in the context, say so.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}"
            )
        }]
    )

    return response["message"]["content"]


# ── HOME ──────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Lumina · RAG Assistant</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=DM+Mono:wght@300;400&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --ink: #0d0d0d;
    --paper: #f7f4ef;
    --accent: #c8a96e;
    --muted: #9a9089;
    --border: #e0dbd2;
  }
  html, body {
    height: 100%;
    background: var(--paper);
    color: var(--ink);
    font-family: 'DM Mono', monospace;
  }
  body {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    padding: 2rem;
    background-image: radial-gradient(circle at 20% 20%, rgba(200,169,110,.07) 0%, transparent 60%),
                      radial-gradient(circle at 80% 80%, rgba(200,169,110,.05) 0%, transparent 60%);
  }
  .container {
    max-width: 560px;
    width: 100%;
    text-align: center;
  }
  .logo {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.5rem;
    font-weight: 300;
    letter-spacing: .12em;
    color: var(--ink);
    line-height: 1;
  }
  .logo span { color: var(--accent); font-style: italic; }
  .tagline {
    font-size: .7rem;
    letter-spacing: .25em;
    text-transform: uppercase;
    color: var(--muted);
    margin: .75rem 0 2.5rem;
  }
  .divider {
    width: 40px; height: 1px;
    background: var(--accent);
    margin: 0 auto 2.5rem;
  }
  .links {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
  }
  .btn {
    display: inline-block;
    padding: .7rem 1.8rem;
    font-family: 'DM Mono', monospace;
    font-size: .72rem;
    letter-spacing: .15em;
    text-transform: uppercase;
    text-decoration: none;
    border: 1px solid var(--ink);
    color: var(--ink);
    background: transparent;
    transition: background .2s, color .2s;
    cursor: pointer;
  }
  .btn:hover { background: var(--ink); color: var(--paper); }
  .btn.primary {
    background: var(--ink);
    color: var(--paper);
  }
  .btn.primary:hover { background: var(--accent); border-color: var(--accent); color: var(--ink); }
  footer {
    position: fixed; bottom: 1.5rem; left: 50%; transform: translateX(-50%);
    font-size: .62rem; letter-spacing: .12em; color: var(--muted); text-transform: uppercase;
  }
</style>
</head>
<body>
<div class="container">
  <div class="logo">Lumi<span>na</span></div>
  <p class="tagline">Local RAG · Document Intelligence</p>
  <div class="divider"></div>
  <div class="links">
    <a class="btn primary" href="/chat-ui">Open Chat</a>
    <a class="btn" href="/docs">API Docs</a>
  </div>
</div>
<footer>Powered by Ollama · ChromaDB · FastAPI</footer>
</body>
</html>"""


# ── CHAT API ──────────────────────────────────────────────────────────────────
@app.post("/chat")
def chat(query: Query):
    return {"answer": ask(query.question)}


# ── CHAT UI ───────────────────────────────────────────────────────────────────
@app.get("/chat-ui", response_class=HTMLResponse)
def chat_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Lumina · Chat</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=DM+Mono:wght@300;400&display=swap" rel="stylesheet">
<style>
/* ── RESET ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --ink: #0d0d0d;
  --paper: #f7f4ef;
  --paper2: #f0ece4;
  --accent: #c8a96e;
  --accent2: #a07840;
  --muted: #9a9089;
  --border: #e0dbd2;
  --user-bg: #0d0d0d;
  --user-fg: #f7f4ef;
  --ai-bg: #ffffff;
  --ai-fg: #0d0d0d;
  --shadow: 0 2px 16px rgba(13,13,13,.07);
  --radius: 2px;
}

html, body {
  height: 100%;
  background: var(--paper);
  color: var(--ink);
  font-family: 'DM Mono', monospace;
  font-size: 14px;
  line-height: 1.7;
}

/* ── LAYOUT ── */
.app {
  display: grid;
  grid-template-rows: auto 1fr auto;
  height: 100vh;
  max-width: 860px;
  margin: 0 auto;
}

/* ── HEADER ── */
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.1rem 1.8rem;
  border-bottom: 1px solid var(--border);
  background: var(--paper);
  position: sticky; top: 0; z-index: 10;
}
.logo {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.6rem;
  font-weight: 300;
  letter-spacing: .1em;
}
.logo em { font-style: italic; color: var(--accent); }
.status {
  display: flex;
  align-items: center;
  gap: .5rem;
  font-size: .62rem;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--muted);
}
.dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: #4caf82;
  animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: .35; }
}

/* ── MESSAGES ── */
.messages {
  overflow-y: auto;
  padding: 2rem 1.8rem;
  display: flex;
  flex-direction: column;
  gap: 1.6rem;
  scroll-behavior: smooth;
}

.message {
  display: flex;
  gap: 1rem;
  align-items: flex-start;
  animation: fadeUp .35s ease both;
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
.message.user  { flex-direction: row-reverse; }

.avatar {
  flex-shrink: 0;
  width: 32px; height: 32px;
  border-radius: var(--radius);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: .65rem;
  letter-spacing: .1em;
  font-weight: 400;
  text-transform: uppercase;
}
.message.user  .avatar { background: var(--ink); color: var(--paper); }
.message.ai    .avatar { background: var(--border); color: var(--muted); border: 1px solid var(--border); }

.bubble {
  max-width: 72%;
  padding: .9rem 1.2rem;
  border-radius: var(--radius);
  line-height: 1.8;
  font-size: .82rem;
  white-space: pre-wrap;
  word-break: break-word;
}
.message.user .bubble {
  background: var(--user-bg);
  color: var(--user-fg);
}
.message.ai .bubble {
  background: var(--ai-bg);
  color: var(--ai-fg);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
}

/* typing dots */
.typing-bubble {
  display: flex;
  align-items: center;
  gap: .35rem;
  padding: .9rem 1.2rem;
  background: var(--ai-bg);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  border-radius: var(--radius);
}
.typing-bubble span {
  display: block;
  width: 5px; height: 5px;
  border-radius: 50%;
  background: var(--muted);
  animation: bounce 1.1s ease infinite;
}
.typing-bubble span:nth-child(2) { animation-delay: .18s; }
.typing-bubble span:nth-child(3) { animation-delay: .36s; }
@keyframes bounce {
  0%, 80%, 100% { transform: translateY(0); }
  40% { transform: translateY(-7px); }
}

/* welcome card */
.welcome {
  margin: auto;
  text-align: center;
  padding: 3rem 1rem;
  animation: fadeUp .5s ease both;
}
.welcome-icon {
  font-size: 2.2rem;
  margin-bottom: 1rem;
  opacity: .8;
}
.welcome h2 {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.8rem;
  font-weight: 300;
  letter-spacing: .06em;
  margin-bottom: .5rem;
}
.welcome p {
  font-size: .72rem;
  color: var(--muted);
  letter-spacing: .06em;
  max-width: 340px;
  margin: 0 auto 1.8rem;
}
.suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: .6rem;
  justify-content: center;
}
.suggestion {
  padding: .45rem 1rem;
  font-size: .68rem;
  letter-spacing: .08em;
  border: 1px solid var(--border);
  background: var(--paper2);
  cursor: pointer;
  color: var(--ink);
  font-family: 'DM Mono', monospace;
  transition: all .18s;
  border-radius: var(--radius);
}
.suggestion:hover {
  background: var(--ink);
  color: var(--paper);
  border-color: var(--ink);
}

/* ── INPUT BAR ── */
.input-bar {
  border-top: 1px solid var(--border);
  padding: 1rem 1.8rem 1.4rem;
  background: var(--paper);
}
.input-row {
  display: flex;
  gap: .7rem;
  align-items: flex-end;
}
textarea {
  flex: 1;
  resize: none;
  border: 1px solid var(--border);
  background: var(--paper2);
  color: var(--ink);
  font-family: 'DM Mono', monospace;
  font-size: .8rem;
  line-height: 1.6;
  padding: .75rem 1rem;
  border-radius: var(--radius);
  outline: none;
  min-height: 44px;
  max-height: 160px;
  transition: border-color .18s;
  overflow-y: auto;
}
textarea:focus { border-color: var(--accent); }
textarea::placeholder { color: var(--muted); }

.send-btn {
  flex-shrink: 0;
  width: 44px; height: 44px;
  background: var(--ink);
  color: var(--paper);
  border: none;
  border-radius: var(--radius);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background .18s, transform .1s;
}
.send-btn:hover  { background: var(--accent); }
.send-btn:active { transform: scale(.95); }
.send-btn:disabled { background: var(--border); cursor: not-allowed; }
.send-btn svg { width: 16px; height: 16px; fill: currentColor; }

.hint {
  margin-top: .5rem;
  font-size: .6rem;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--muted);
  text-align: center;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
</head>
<body>
<div class="app">

  <!-- HEADER -->
  <header>
    <div class="logo">Lumi<em>na</em></div>
    <div class="status">
      <div class="dot"></div>
      <span>llama3 · local</span>
    </div>
  </header>

  <!-- MESSAGES -->
  <div class="messages" id="messages">
    <div class="welcome" id="welcome">
      <div class="welcome-icon">📄</div>
      <h2>Ask the Document</h2>
      <p>I have read your research paper. Ask me anything about its contents.</p>
      <div class="suggestions">
        <div class="suggestion" onclick="useSuggestion(this)">What is the main topic?</div>
        <div class="suggestion" onclick="useSuggestion(this)">Summarize the key findings</div>
        <div class="suggestion" onclick="useSuggestion(this)">What methodology was used?</div>
        <div class="suggestion" onclick="useSuggestion(this)">What are the conclusions?</div>
      </div>
    </div>
  </div>

  <!-- INPUT -->
  <div class="input-bar">
    <div class="input-row">
      <textarea
        id="q"
        placeholder="Ask a question about your document…"
        rows="1"
        onkeydown="handleKey(event)"
        oninput="autoResize(this)"
      ></textarea>
      <button class="send-btn" id="sendBtn" onclick="send()" title="Send">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
        </svg>
      </button>
    </div>
    <p class="hint">Enter to send · Shift+Enter for new line</p>
  </div>

</div>

<script>
const messagesEl = document.getElementById('messages');
const inputEl    = document.getElementById('q');
const sendBtn    = document.getElementById('sendBtn');
const welcomeEl  = document.getElementById('welcome');

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
}

function useSuggestion(el) {
  inputEl.value = el.textContent;
  autoResize(inputEl);
  send();
}

function addMessage(role, text) {
  // Remove welcome on first message
  if (welcomeEl && welcomeEl.parentNode) {
    welcomeEl.remove();
  }

  const msg = document.createElement('div');
  msg.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? 'You' : 'AI';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  messagesEl.appendChild(msg);
  scrollBottom();
  return bubble;
}

function showTyping() {
  if (welcomeEl && welcomeEl.parentNode) welcomeEl.remove();

  const msg = document.createElement('div');
  msg.className = 'message ai';
  msg.id = 'typing-indicator';

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = 'AI';

  const bubble = document.createElement('div');
  bubble.className = 'typing-bubble';
  bubble.innerHTML = '<span></span><span></span><span></span>';

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  messagesEl.appendChild(msg);
  scrollBottom();
}

function removeTyping() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

function scrollBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function send() {
  const q = inputEl.value.trim();
  if (!q) return;

  addMessage('user', q);
  inputEl.value = '';
  inputEl.style.height = 'auto';
  sendBtn.disabled = true;

  showTyping();

  try {
    const res  = await fetch('/chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ question: q })
    });
    const data = await res.json();
    removeTyping();
    addMessage('ai', data.answer || 'No answer returned.');
  } catch (err) {
    removeTyping();
    addMessage('ai', '⚠ An error occurred. Please check that the server is running.');
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}
</script>
</body>
</html>"""
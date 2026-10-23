"""
FastAPI Server cho Samsung AI Agent:
- REST API /api/chat với Server-Sent Events (SSE) streaming
- Serve Frontend HTML/CSS/JS
- Healthcheck endpoint
"""

import json
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from .agent import SamsungAgent

app = FastAPI(title="Samsung Washing Machine AI Agent API", version="2.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []
    model: Optional[str] = "finetuned"

# Khởi tạo singleton agent
agent_instance = None

def get_agent():
    global agent_instance
    if agent_instance is None:
        agent_instance = SamsungAgent(embedding_model="finetuned")
    return agent_instance

@app.on_event("startup")
async def startup_event():
    get_agent()

@app.get("/api/health")
def health():
    agent = get_agent()
    return {"status": "ok", "agent": "SamsungAgent", "llm": "Qwen2.5-7B", "current_embedding": agent.current_model}

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    agent = get_agent()
    if request.model:
        agent.set_embedding_model(request.model)

    def event_generator():
        for event in agent.run_agent(request.message, request.history):
            data = json.dumps(event, ensure_ascii=False)
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Mount thư mục frontend static files
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Frontend chưa được triển khai</h1>"

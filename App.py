import uuid
import asyncio
from typing import Dict, List
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from ChatIA import rag_chain_stream
from ChatIA import session_manager


# --- Configuración de la API con FastAPI ---
app = FastAPI()

# Modelo de datos para el endpoint /chat
class ChatRequest(BaseModel):
    session_id: str = None  # Opcional, se genera uno nuevo si no se envía
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str

class HistoryResponse(BaseModel):
    session_id: str
    history: List[Dict[str, str]]

# Endpoint para generar una nueva sesión (usado por el HTML)
@app.get("/new_session")
async def new_session():
    new_session_id = str(uuid.uuid4())
    return {"session_id": new_session_id}

# Endpoint para procesar los mensajes del chat
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    # Si no se envía session_id, se genera uno nuevo
    session_id = chat_request.session_id or str(uuid.uuid4())
    return StreamingResponse(rag_chain_stream(chat_request.message, session_id), media_type="text/event-stream")

@app.get("/history", response_model=HistoryResponse)
async def get_history(session_id: str):
    #Devuelve el historial de la sesión si existe.
    history = session_manager.get_history(session_id)
    
    # Convertir la lista de mensajes en diccionarios legibles por JSON
    formatted_history = [
        {"role": "user" if isinstance(msg, HumanMessage) else "bot", "message": msg.content}
        for msg in history
    ]
    
    return {"session_id": session_id, "history": formatted_history}

# Monta la carpeta "static" para servir archivos estáticos (incluyendo index.html)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
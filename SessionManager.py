from typing import Dict, List
import threading
from langchain_core.messages import HumanMessage, AIMessage

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, List] = {}
        self.lock = threading.Lock()  # Bloqueo síncrono

    def get_history(self, session_id: str) -> List:
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            return self.sessions[session_id].copy()

    def update_history(self, session_id: str, human_msg: HumanMessage, ai_msg: AIMessage):
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            self.sessions[session_id].extend([human_msg, ai_msg])

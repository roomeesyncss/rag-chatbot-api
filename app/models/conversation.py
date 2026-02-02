from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ConversationRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    top_k: Optional[int] = 3

class ConversationResponse(BaseModel):
    answer: str
    sources: List[dict]
    conversation_id: str
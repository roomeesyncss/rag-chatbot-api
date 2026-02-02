from pydantic import BaseModel
from typing import List, Optional

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    message: str

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

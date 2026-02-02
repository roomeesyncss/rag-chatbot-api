from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.models.document import DocumentUploadResponse, QueryRequest, QueryResponse
from app.models.user import User
from app.services.auth import get_current_user
from app.services.rag import rag_service
from app.utils.pdf import extract_text_from_pdf

router = APIRouter(prefix="/rag", tags=["RAG"])

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload a PDF document"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Extract text from PDF
        text = extract_text_from_pdf(file.file)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF contains no extractable text")
        
        # Add to RAG system
        result = rag_service.add_document(text, file.filename)
        
        return DocumentUploadResponse(
            document_id=result["document_id"],
            filename=result["filename"],
            chunks_created=result["chunks_created"],
            message="Document uploaded successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.post("/query", response_model=QueryResponse)
async def query_documents(
    query: QueryRequest,
    current_user: User = Depends(get_current_user)
):
    """Query the RAG system"""
    try:
        result = rag_service.query(query.question, query.top_k)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

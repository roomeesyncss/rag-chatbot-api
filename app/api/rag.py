from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.models.document import DocumentUploadResponse, QueryRequest, QueryResponse
from app.models.conversation import ConversationRequest, ConversationResponse
from app.models.user import User
from app.services.auth import get_current_user
from app.services.rag import rag_service
from app.utils.pdf import extract_text_from_pdf
from app.database import get_db
import uuid
import json

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
        text = extract_text_from_pdf(file.file)

        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF contains no extractable text")

        result = rag_service.add_document(text, file.filename, current_user.email)

        return DocumentUploadResponse(
            document_id=result["document_id"],
            filename=result["filename"],
            chunks_created=result["chunks_created"],
            message="Document uploaded successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.get("/documents")
async def list_documents(current_user: User = Depends(get_current_user)):
    """List all user documents"""
    try:
        docs = rag_service.get_user_documents(current_user.email)
        return {"documents": docs, "count": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching documents: {str(e)}")


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, current_user: User = Depends(get_current_user)):
    """Delete a document"""
    try:
        success = rag_service.delete_document(doc_id, current_user.email)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"message": "Document deleted successfully", "doc_id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query_documents(
        query: QueryRequest,
        current_user: User = Depends(get_current_user)
):
    """Query without conversation history"""
    try:
        result = rag_service.query(query.question, query.top_k)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/chat", response_model=ConversationResponse)
async def chat_with_documents(
        request: ConversationRequest,
        current_user: User = Depends(get_current_user)
):
    """Chat with conversation memory"""
    try:
        # Create or get conversation
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            # Create conversation in DB
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO conversations (id, user_email) VALUES (?, ?)",
                    (conversation_id, current_user.email)
                )
                conn.commit()

        result = rag_service.query_with_conversation(
            request.question,
            conversation_id,
            current_user.email,
            request.top_k
        )

        return ConversationResponse(
            answer=result["answer"],
            sources=result["sources"],
            conversation_id=conversation_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@router.get("/conversations")
async def list_conversations(current_user: User = Depends(get_current_user)):
    """List all user conversations"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT c.id, c.created_at, 
                   (SELECT content FROM messages WHERE conversation_id = c.id AND role = 'user' ORDER BY created_at ASC LIMIT 1) as first_message
                   FROM conversations c 
                   WHERE c.user_email = ? 
                   ORDER BY c.created_at DESC""",
                (current_user.email,)
            )
            rows = cursor.fetchall()
            conversations = [
                {
                    "id": row["id"],
                    "created_at": row["created_at"],
                    "preview": row["first_message"][:100] + "..." if row["first_message"] else "Empty conversation"
                }
                for row in rows
            ]
            return {"conversations": conversations, "count": len(conversations)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching conversations: {str(e)}")


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
        conversation_id: str,
        current_user: User = Depends(get_current_user)
):
    """Get all messages in a conversation"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            # Verify ownership
            cursor.execute(
                "SELECT user_email FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            row = cursor.fetchone()
            if not row or row["user_email"] != current_user.email:
                raise HTTPException(status_code=404, detail="Conversation not found")

            # Get messages
            cursor.execute(
                "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,)
            )
            messages = [
                {
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": row["created_at"]
                }
                for row in cursor.fetchall()
            ]
            return {"messages": messages, "count": len(messages)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching messages: {str(e)}")
import faiss
import numpy as np
import pickle
import os
import cohere
from groq import Groq
from typing import List, Dict
import uuid
from app.core.config import settings as app_settings
from app.database import get_db


class RAGService:
    def __init__(self):
        self.dimension = 1024
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadatas = []

        self.index_path = "./faiss_index.bin"
        self.meta_path = "./faiss_meta.pkl"
        self._load_index()

        self.cohere_client = cohere.Client(api_key=app_settings.COHERE_API_KEY)
        self.groq_client = Groq(api_key=app_settings.GROQ_API_KEY)

    def _load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadatas = data['metadatas']

    def _save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadatas': self.metadatas
            }, f)

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
        return chunks

    def add_document(self, text: str, filename: str, user_email: str) -> Dict:
        chunks = self.chunk_text(text)
        doc_id = str(uuid.uuid4())

        response = self.cohere_client.embed(
            texts=chunks,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        embeddings = np.array(response.embeddings).astype('float32')

        self.index.add(embeddings)

        for i, chunk in enumerate(chunks):
            self.documents.append(chunk)
            self.metadatas.append({
                "filename": filename,
                "doc_id": doc_id,
                "chunk_id": i,
                "user_email": user_email
            })

        self._save_index()

        # Save to database
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO documents (doc_id, user_email, filename, chunks_count) VALUES (?, ?, ?, ?)",
                (doc_id, user_email, filename, len(chunks))
            )
            conn.commit()

        return {
            "document_id": doc_id,
            "filename": filename,
            "chunks_created": len(chunks)
        }

    def get_user_documents(self, user_email: str) -> List[Dict]:
        """Get all documents for a user"""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT doc_id, filename, chunks_count, uploaded_at FROM documents WHERE user_email = ?",
                (user_email,)
            )
            rows = cursor.fetchall()
            return [
                {
                    "doc_id": row["doc_id"],
                    "filename": row["filename"],
                    "chunks_count": row["chunks_count"],
                    "uploaded_at": row["uploaded_at"]
                }
                for row in rows
            ]

    def delete_document(self, doc_id: str, user_email: str) -> bool:
        """Delete a document and its embeddings"""
        # Find indices to remove
        indices_to_remove = []
        for i, meta in enumerate(self.metadatas):
            if meta["doc_id"] == doc_id and meta["user_email"] == user_email:
                indices_to_remove.append(i)

        if not indices_to_remove:
            return False

        # Remove from lists (reverse order to maintain indices)
        for idx in sorted(indices_to_remove, reverse=True):
            del self.documents[idx]
            del self.metadatas[idx]

        # Rebuild FAISS index
        if len(self.documents) > 0:
            # Re-embed all remaining documents
            response = self.cohere_client.embed(
                texts=self.documents,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            embeddings = np.array(response.embeddings).astype('float32')
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

        self._save_index()

        # Remove from database
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM documents WHERE doc_id = ? AND user_email = ?",
                (doc_id, user_email)
            )
            conn.commit()

        return True

    def query_with_conversation(self, question: str, conversation_id: str, user_email: str, top_k: int = 3) -> Dict:
        """Query with conversation history"""
        # Get conversation history
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC LIMIT 10",
                (conversation_id,)
            )
            history = [{"role": row["role"], "content": row["content"]} for row in cursor.fetchall()]

        # Get relevant documents
        response = self.cohere_client.embed(
            texts=[question],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        query_embedding = np.array(response.embeddings).astype('float32')

        distances, indices = self.index.search(query_embedding, top_k)

        contexts = []
        metadatas = []
        for idx in indices[0]:
            if idx < len(self.documents):
                contexts.append(self.documents[idx])
                metadatas.append(self.metadatas[idx])

        # Build prompt with history
        context_text = "\n\n".join([f"Context {i + 1}: {ctx}" for i, ctx in enumerate(contexts)])

        # Build conversation messages
        messages = [
            {"role": "system",
             "content": "You are a helpful assistant that answers questions based on the provided context and conversation history. Be concise and accurate."}
        ]

        # Add history (last 5 messages)
        messages.extend(history[-5:])

        # Add current query with context
        user_message = f"""Based on the following context, answer the question:

Context:
{context_text}

Question: {question}"""

        messages.append({"role": "user", "content": user_message})

        # Get answer
        chat_completion = self.groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=500
        )

        answer = chat_completion.choices[0].message.content

        # Save messages to database
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                (conversation_id, "user", question)
            )
            cursor.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                (conversation_id, "assistant", answer)
            )
            conn.commit()

        sources = [
            {
                "filename": meta["filename"],
                "chunk_id": meta["chunk_id"],
                "text": ctx[:200] + "..."
            }
            for meta, ctx in zip(metadatas, contexts)
        ]

        return {"answer": answer, "sources": sources}

    def query(self, question: str, top_k: int = 3) -> Dict:
        """Simple query without conversation history"""
        response = self.cohere_client.embed(
            texts=[question],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        query_embedding = np.array(response.embeddings).astype('float32')

        distances, indices = self.index.search(query_embedding, top_k)

        contexts = []
        metadatas = []
        for idx in indices[0]:
            if idx < len(self.documents):
                contexts.append(self.documents[idx])
                metadatas.append(self.metadatas[idx])

        context_text = "\n\n".join([f"Context {i + 1}: {ctx}" for i, ctx in enumerate(contexts)])
        prompt = f"""Based on the following context, answer the question. If you cannot answer based on the context, say so.

Context:
{context_text}

Question: {question}

Answer:"""

        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",

            temperature=0.3,
            max_tokens=500
        )

        answer = chat_completion.choices[0].message.content

        sources = [
            {
                "filename": meta["filename"],
                "chunk_id": meta["chunk_id"],
                "text": ctx[:200] + "..."
            }
            for meta, ctx in zip(metadatas, contexts)
        ]

        return {"answer": answer, "sources": sources}


rag_service = RAGService()
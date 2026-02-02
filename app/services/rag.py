import faiss
import numpy as np
import pickle
import os
import cohere
from groq import Groq
from typing import List, Dict
import uuid
from app.core.config import settings as app_settings


class RAGService:
    def __init__(self):
        # Initialize FAISS
        self.dimension = 1024  # Cohere embed-english-v3.0 dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadatas = []

        # Load existing index if available
        self.index_path = "./faiss_index.bin"
        self.meta_path = "./faiss_meta.pkl"
        self._load_index()

        # Initialize Cohere client (FREE!)
        self.cohere_client = cohere.Client(api_key=app_settings.COHERE_API_KEY)

        # Initialize Groq client
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
            chunks.append(chunk)
        return chunks

    def add_document(self, text: str, filename: str) -> Dict:
        chunks = self.chunk_text(text)
        doc_id = str(uuid.uuid4())

        # Get embeddings from Cohere
        response = self.cohere_client.embed(
            texts=chunks,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        embeddings = np.array(response.embeddings).astype('float32')

        # Add to FAISS
        self.index.add(embeddings)

        for i, chunk in enumerate(chunks):
            self.documents.append(chunk)
            self.metadatas.append({
                "filename": filename,
                "doc_id": doc_id,
                "chunk_id": i
            })

        self._save_index()

        return {
            "document_id": doc_id,
            "filename": filename,
            "chunks_created": len(chunks)
        }

    def query(self, question: str, top_k: int = 3) -> Dict:
        # Get query embedding
        response = self.cohere_client.embed(
            texts=[question],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        query_embedding = np.array(response.embeddings).astype('float32')

        # Search
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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import auth, rag

app = FastAPI(
    title="RAG Chatbot API",
    description="Production-ready RAG system with JWT authentication",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(rag.router)

@app.get("/")
async def root():
    return {
        "message": "RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "auth": "/auth",
            "rag": "/rag",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

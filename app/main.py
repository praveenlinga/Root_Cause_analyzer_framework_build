from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import get_settings
from app.models import (
    DocumentAdd, DocumentBatch, QueryRequest, QueryResponse,
    DocumentResponse, HealthResponse
)
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.llm import LLMService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
embedding_service = None
vector_store_service = None
llm_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize all services
    global embedding_service, vector_store_service, llm_service
    settings = get_settings()
    
    logger.info("🚀 Initializing RAG System...")
    logger.info(f"📁 ChromaDB Path: {settings.chroma_persist_dir}")
    logger.info(f"🧠 Embedding Model: {settings.embedding_model}")
    logger.info(f"🤖 LLM Provider: Groq")
    
    try:
        embedding_service = EmbeddingService(settings.embedding_model)
        vector_store_service = VectorStoreService(
            settings.chroma_persist_dir,
            settings.collection_name
        )
        llm_service = LLMService(settings.groq_api_key)
        logger.info("✅ All services initialized successfully!")
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down RAG System...")

# Create FastAPI app
app = FastAPI(
    title="Local RAG System",
    description="Cloud-agnostic RAG system with FastAPI + Groq + ChromaDB",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info"""
    return {
        "message": "🚀 RAG FastAPI System is running!",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint"""
    doc_count = vector_store_service.count()
    return HealthResponse(
        status="healthy",
        embedding_model=get_settings().embedding_model,
        vector_db=f"chromadb ({doc_count} documents)",
        llm_provider="groq (llama-3.1-8b-instant)"
    )

@app.post("/documents/add", status_code=status.HTTP_201_CREATED, tags=["Documents"])
async def add_document(doc: DocumentAdd):
    """Add single document to vector store"""
    try:
        logger.info(f"📝 Adding document: {doc.id}")
        
        # Create embedding
        embedding = embedding_service.create_embedding(doc.content)
        
        # Store in vector DB
        vector_store_service.add_document(
            doc_id=doc.id,
            content=doc.content,
            embedding=embedding,
            metadata=doc.metadata
        )
        
        total_docs = vector_store_service.count()
        logger.info(f"✅ Document added. Total documents: {total_docs}")
        
        return {
            "message": "Document added successfully",
            "id": doc.id,
            "total_documents": total_docs
        }
    except Exception as e:
        logger.error(f"❌ Error adding document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/add-batch", status_code=status.HTTP_201_CREATED, tags=["Documents"])
async def add_documents_batch(batch: DocumentBatch):
    """Add multiple documents to vector store"""
    try:
        logger.info(f"📚 Adding batch of {len(batch.documents)} documents")
        
        doc_ids = [doc.id for doc in batch.documents]
        contents = [doc.content for doc in batch.documents]
        metadatas = [doc.metadata for doc in batch.documents]
        
        # Create embeddings for all documents
        embeddings = embedding_service.create_embeddings(contents)
        
        # Store in vector DB
        vector_store_service.add_documents(
            doc_ids=doc_ids,
            contents=contents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        total_docs = vector_store_service.count()
        logger.info(f"✅ Batch added. Total documents: {total_docs}")
        
        return {
            "message": f"{len(batch.documents)} documents added successfully",
            "total_documents": total_docs
        }
    except Exception as e:
        logger.error(f"❌ Error adding batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(request: QueryRequest):
    """Query RAG system with semantic search + LLM generation"""
    try:
        logger.info(f"🔍 Query: {request.query[:50]}...")
        
        # Create query embedding
        query_embedding = embedding_service.create_embedding(request.query)
        
        # Search similar documents
        results = vector_store_service.query(
            query_embedding=query_embedding,
            top_k=request.top_k
        )
        
        # Filter by similarity threshold
        relevant_docs = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]
                similarity = 1 - distance
                
                if similarity >= request.similarity_threshold:
                    relevant_docs.append({
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': distance,
                        'similarity': similarity
                    })
        
        logger.info(f"📊 Found {len(relevant_docs)} relevant documents")
        
        # Generate answer using LLM
        answer = llm_service.generate_answer(request.query, relevant_docs)
        
        # Format sources
        sources = [
            {
                'id': doc['id'],
                'content': doc['document'][:200] + "..." if len(doc['document']) > 200 else doc['document'],
                'similarity': f"{doc['similarity']*100:.1f}%",
                'metadata': doc['metadata']
            }
            for doc in relevant_docs
        ]
        
        logger.info(f"✅ Answer generated with {len(sources)} sources")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            num_sources=len(sources)
        )
    
    except Exception as e:
        logger.error(f"❌ Error querying RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", tags=["Documents"])
async def list_documents():
    """List all documents in vector store"""
    try:
        results = vector_store_service.get_all_documents()
        
        documents = []
        if results['ids']:
            for i in range(len(results['ids'])):
                documents.append({
                    'id': results['ids'][i],
                    'content': results['documents'][i][:200] + "..." if len(results['documents'][i]) > 200 else results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
        
        total = vector_store_service.count()
        logger.info(f"📋 Listed {total} documents")
        
        return {
            "total": total,
            "documents": documents
        }
    except Exception as e:
        logger.error(f"❌ Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Delete document by ID"""
    try:
        vector_store_service.delete_document(doc_id)
        total_docs = vector_store_service.count()
        
        logger.info(f"🗑️ Document {doc_id} deleted. Remaining: {total_docs}")
        
        return {
            "message": f"Document {doc_id} deleted successfully",
            "total_documents": total_docs
        }
    except Exception as e:
        logger.error(f"❌ Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents", tags=["Documents"])
async def reset_database():
    """Reset entire database (delete all documents)"""
    try:
        vector_store_service.reset()
        logger.info("🔄 Database reset successfully")
        
        return {
            "message": "Database reset successfully",
            "total_documents": 0
        }
    except Exception as e:
        logger.error(f"❌ Error resetting database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Run with uvicorn
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False
    )



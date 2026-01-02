from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentAdd(BaseModel):
    id: str = Field(..., description="Unique document ID")
    content: str = Field(..., description="Document content")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata")

class DocumentBatch(BaseModel):
    documents: List[DocumentAdd]

class QueryRequest(BaseModel):
    query: str = Field(..., description="Query text")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of similar documents to retrieve")
    similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum similarity score")

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    num_sources: int

class DocumentResponse(BaseModel):
    id: str
    content: str
    metadata: dict

class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    vector_db: str
    llm_provider: str


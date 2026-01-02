from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    groq_api_key: str
    chroma_persist_dir: str = "./data/chroma_db"
    embedding_model: str = "intfloat/e5-large-v2"
    collection_name: str = "rag_documents"
    host: str = "0.0.0.0"
    port: int = 8000
    
    class Config:
        case_sensitive = False

@lru_cache()
def get_settings():
    return Settings()

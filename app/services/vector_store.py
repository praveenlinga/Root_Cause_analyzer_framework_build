import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional
import os

class VectorStoreService:
    def __init__(self, persist_dir: str, collection_name: str):
        os.makedirs(persist_dir, exist_ok=True)
        
        print(f"Initializing ChromaDB at: {persist_dir}")
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Collection '{collection_name}' ready. Documents: {self.collection.count()}")
    
    def add_document(self, doc_id: str, content: str, embedding: List[float], metadata: dict):
        self.collection.add(
            ids=[doc_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata]
        )
    
    def add_documents(self, doc_ids: List[str], contents: List[str], 
                     embeddings: List[List[float]], metadatas: List[dict]):
        self.collection.add(
            ids=doc_ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def query(self, query_embedding: List[float], top_k: int = 3) -> Dict:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return results
    
    def get_all_documents(self) -> Dict:
        results = self.collection.get(
            include=["documents", "metadatas"]
        )
        return results
    
    def delete_document(self, doc_id: str):
        self.collection.delete(ids=[doc_id])
    
    def count(self) -> int:
        return self.collection.count()
    
    def reset(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )

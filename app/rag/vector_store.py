"""
Vector Store Module using ChromaDB for RAG
Handles document embedding and similarity search
"""

import os
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Any
import json


class VectorStore:
    """Manages ChromaDB vector store for document retrieval"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB vector store
        
        Args:
            persist_directory: Path to persist the vector database
        """
        self.persist_directory = persist_directory
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize Chroma client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="samar_resume",
            metadata={"hnsw:space": "cosine"}
        )
        
    def add_documents(self, chunks: List[Dict[str, str]]) -> None:
        """
        Add documents to the vector store
        
        Args:
            chunks: List of document chunks with 'id', 'category', and 'content' keys
        """
        documents = []
        ids = []
        metadatas = []
        
        for chunk in chunks:
            documents.append(chunk["content"])
            ids.append(chunk["id"])
            metadatas.append({
                "category": chunk["category"],
                "source": "resume"
            })
        
        # Add documents to collection
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        
        print(f"✓ Added {len(chunks)} document chunks to vector store")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant document chunks with scores
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["documents"] or not results["documents"][0]:
            return []
        
        # Format results
        formatted_results = []
        for i, doc in enumerate(results["documents"][0]):
            formatted_results.append({
                "content": doc,
                "category": results["metadatas"][0][i]["category"],
                "relevance_score": 1 - results["distances"][0][i],  # Convert distance to similarity
                "source": results["metadatas"][0][i]["source"]
            })
        
        return formatted_results
    
    def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """
        Get context string for LLM from relevant documents
        
        Args:
            query: Search query
            max_tokens: Maximum tokens for context (approximate)
            
        Returns:
            Formatted context string
        """
        results = self.search(query, n_results=5)
        
        context_parts = []
        current_tokens = 0
        
        for result in results:
            # Rough token estimation (1 token ≈ 4 characters)
            tokens = len(result["content"]) // 4
            
            if current_tokens + tokens > max_tokens:
                break
            
            context_parts.append(f"[{result['category']}]\n{result['content']}")
            current_tokens += tokens
        
        return "\n\n".join(context_parts)
    
    def reset(self) -> None:
        """Reset the vector store"""
        self.client.delete_collection(name="samar_resume")
        self.collection = self.client.get_or_create_collection(
            name="samar_resume",
            metadata={"hnsw:space": "cosine"}
        )
        print("✓ Vector store reset")


def initialize_vector_store(chunks: List[Dict[str, str]]) -> VectorStore:
    """
    Initialize and populate vector store
    
    Args:
        chunks: Resume chunks to add to store
        
    Returns:
        Initialized VectorStore instance
    """
    vector_store = VectorStore()
    
    # Check if already populated
    if vector_store.collection.count() == 0:
        vector_store.add_documents(chunks)
    else:
        print(f"✓ Vector store already populated with {vector_store.collection.count()} documents")
    
    return vector_store

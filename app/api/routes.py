"""
FastAPI routes for the RAG-based chatbot
Handles chat endpoints and thread management
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime

from app.agent.samar_agent import SamarAgent
from app.rag.vector_store import VectorStore


# Pydantic models for request/response
class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    success: bool
    message: str
    thread_id: str
    timestamp: str


class HistoryResponse(BaseModel):
    """Response model for conversation history"""
    thread_id: str
    messages: List[dict]


class ThreadResponse(BaseModel):
    """Response model for thread creation"""
    thread_id: str
    created_at: str


def create_router(app) -> APIRouter:
    """
    Create the API router with all endpoints
    
    Args:
        app: FastAPI app instance
        
    Returns:
        APIRouter with configured routes
    """
    router = APIRouter(prefix="/api/chat", tags=["chat"])
    
    @router.post("/thread", response_model=ThreadResponse)
    async def create_thread():
        """
        Create a new conversation thread
        
        Returns:
            New thread ID and creation timestamp
        """
        thread_id = str(uuid.uuid4())
        return {
            "thread_id": thread_id,
            "created_at": datetime.now().isoformat()
        }
    
    @router.post("/message", response_model=ChatResponse)
    async def send_message(request: ChatRequest):
        """
        Send a message and get a response from the agent
        
        Args:
            request: Chat request with message and optional thread_id
            
        Returns:
            Agent response with thread_id and timestamp
        """
        # Validate message
        if not request.message or not request.message.strip():
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )
        
        # Create thread if not provided
        thread_id = request.thread_id or str(uuid.uuid4())
        
        # Get agent from app state
        agent = app.state.agent
        
        # Process message through agent
        response = agent.chat(request.message, thread_id)
        
        return response
    
    @router.get("/history/{thread_id}", response_model=HistoryResponse)
    async def get_history(thread_id: str):
        """
        Get conversation history for a thread
        
        Args:
            thread_id: Thread ID to retrieve history for
            
        Returns:
            List of messages in the conversation
        """
        # Get agent from app state
        agent = app.state.agent
        
        messages = agent.get_conversation_history(thread_id)
        
        return {
            "thread_id": thread_id,
            "messages": messages
        }
    
    @router.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "samar-chatbot",
            "timestamp": datetime.now().isoformat()
        }
    
    return router

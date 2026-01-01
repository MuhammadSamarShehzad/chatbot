"""
Main FastAPI Application
RAG-based Chatbot representing Muhammad Samar Shehzad
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager

from data.resume_chunks import RESUME_CHUNKS
from app.rag.vector_store import initialize_vector_store
from app.agent.samar_agent import SamarAgent
from app.api.routes import create_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    try:
        logger.info("Initializing vector store...")
        vector_store = initialize_vector_store(RESUME_CHUNKS)
        
        logger.info("Initializing agent...")
        agent = SamarAgent(vector_store)
        
        # Store in app state for access in routes
        app.state.agent = agent
        app.state.vector_store = vector_store
        
        logger.info("✓ Chatbot initialized successfully!")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down chatbot...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Samar Shehzad - AI Chatbot API",
    description="RAG-based conversational AI representing Muhammad Samar Shehzad",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (configure as needed for security)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include chat routes after app creation
app.include_router(create_router(app))


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Samar Shehzad's AI Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "create_thread": "POST /api/chat/thread",
            "send_message": "POST /api/chat/message",
            "get_history": "GET /api/chat/history/{thread_id}",
            "health_check": "GET /api/chat/health"
        }
    }


@app.get("/docs")
async def docs():
    """API documentation"""
    return {
        "info": "Samar Shehzad - AI Chatbot API",
        "base_url": "/api/chat",
        "examples": {
            "create_thread": {
                "method": "POST",
                "path": "/api/chat/thread",
                "description": "Create a new conversation thread"
            },
            "send_message": {
                "method": "POST",
                "path": "/api/chat/message",
                "body": {
                    "message": "What are your skills?",
                    "thread_id": "uuid (optional)"
                }
            },
            "get_history": {
                "method": "GET",
                "path": "/api/chat/history/{thread_id}",
                "description": "Retrieve conversation history"
            }
        }
    }


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

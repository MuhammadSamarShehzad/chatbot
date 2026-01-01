"""
Setup script for initializing the chatbot
Run this after installing dependencies
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def setup():
    """Initialize the chatbot"""
    
    print("\n" + "=" * 60)
    print("SAMAR SHEHZAD - CHATBOT SETUP")
    print("=" * 60)
    
    # Check environment
    print("\n1. Checking Environment...")
    try:
        import fastapi
        print("   ✓ FastAPI installed")
    except ImportError:
        print("   ✗ FastAPI not found. Run: pip install -r requirements.txt")
        return False
    
    try:
        import langchain
        print("   ✓ LangChain installed")
    except ImportError:
        print("   ✗ LangChain not found. Run: pip install -r requirements.txt")
        return False
    
    try:
        import chromadb
        print("   ✓ ChromaDB installed")
    except ImportError:
        print("   ✗ ChromaDB not found. Run: pip install -r requirements.txt")
        return False
    
    try:
        import langgraph
        print("   ✓ LangGraph installed")
    except ImportError:
        print("   ✗ LangGraph not found. Run: pip install -r requirements.txt")
        return False
    
    # Initialize vector store
    print("\n2. Initializing Vector Store...")
    try:
        from data.resume_chunks import RESUME_CHUNKS
        from app.rag.vector_store import initialize_vector_store
        
        vector_store = initialize_vector_store(RESUME_CHUNKS)
        print(f"   ✓ Vector store ready with {vector_store.collection.count()} documents")
    except Exception as e:
        print(f"   ✗ Error initializing vector store: {e}")
        return False
    
    # Test agent initialization
    print("\n3. Testing Agent Initialization...")
    try:
        from app.agent.samar_agent import SamarAgent
        
        agent = SamarAgent(vector_store)
        print("   ✓ Agent initialized successfully")
    except Exception as e:
        print(f"   ✗ Error initializing agent: {e}")
        return False
    
    # Create necessary directories
    print("\n4. Creating Directories...")
    try:
        os.makedirs("chroma_db", exist_ok=True)
        print("   ✓ Vector store directory ready")
    except Exception as e:
        print(f"   ✗ Error creating directories: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE ✓")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run tests: python test_chatbot.py")
    print("2. Start server: python main.py")
    print("3. Visit: http://localhost:8000/docs")
    print("\n" + "=" * 60 + "\n")
    
    return True


if __name__ == "__main__":
    success = setup()
    sys.exit(0 if success else 1)

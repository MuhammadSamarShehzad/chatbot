# Samar Shehzad - RAG-Based AI Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) chatbot representing Muhammad Samar Shehzad, a Machine Learning Engineer and LLM/Agentic AI Developer.

## Overview

This chatbot leverages:

- **LangGraph**: For advanced multi-node agent workflows and conversation management
- **ChromaDB**: Vector database for semantic search over resume/background data
- **FastAPI**: High-performance backend API
- **Mistral AI**: LLM for intelligent response generation
- **RAG Pattern**: Retrieves relevant context from knowledge base before responding

## Architecture

```
Frontend (Browser)
    ↓ (Session Storage for thread IDs)
    ↓
FastAPI Backend
    ↓
LangGraph Agent (SamarAgent)
    ├── Retrieve Context Node
    │   └── ChromaDB Vector Store
    │       └── Resume Chunks
    └── Generate Response Node
        └── Mistral LLM
```

## Project Structure

```
ChatBot/
├── main.py                    # Main FastAPI application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data/
│   └── resume_chunks.py       # Pre-processed resume data chunks
├── app/
│   ├── rag/
│   │   ├── __init__.py
│   │   └── vector_store.py    # ChromaDB vector store management
│   ├── agent/
│   │   ├── __init__.py
│   │   └── samar_agent.py     # LangGraph agent implementation
│   └── api/
│       ├── __init__.py
│       └── routes.py          # FastAPI route definitions
└── chroma_db/                 # Vector database (auto-created)
```

## Setup & Installation

### 1. Prerequisites

- Python 3.10+
- pip or poetry

### 2. Create Virtual Environment

```bash
cd ChatBot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the ChatBot directory:

```bash
# Mistral API Configuration
MISTRAL_API_KEY=TuVMj9BVPejhPOYCZwIc5VGIkT0wH9DA
MISTRAL_API_BASE=https://api.mistral.ai/v1

# FastAPI Configuration
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
```

### 5. Run the Application

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Create a Conversation Thread

**POST** `/api/chat/thread`

Creates a new conversation thread and returns a unique thread ID.

**Response:**

```json
{
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2024-01-01T12:00:00.000000"
}
```

### 2. Send a Message

**POST** `/api/chat/message`

Send a user message and receive a response from Samar's AI agent.

**Request:**

```json
{
  "message": "What are your main skills?",
  "thread_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response:**

```json
{
  "success": true,
  "message": "I specialize in Machine Learning, Deep Learning, NLP, and LLM-based applications...",
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-01T12:00:30.123456"
}
```

### 3. Get Conversation History

**GET** `/api/chat/history/{thread_id}`

Retrieve the complete conversation history for a thread.

**Response:**

```json
{
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "messages": [
    {
      "role": "user",
      "content": "What are your skills?"
    },
    {
      "role": "assistant",
      "content": "I specialize in..."
    }
  ]
}
```

### 4. Health Check

**GET** `/api/chat/health`

Check if the service is running.

**Response:**

```json
{
  "status": "healthy",
  "service": "samar-chatbot",
  "timestamp": "2024-01-01T12:00:00.123456"
}
```

## Frontend Integration

### Example: React/TypeScript Integration

```typescript
// Store thread_id in session storage
const chatManager = {
  async getThreadId(): Promise<string> {
    let threadId = sessionStorage.getItem("chatThreadId");

    if (!threadId) {
      const response = await fetch("http://localhost:8000/api/chat/thread", {
        method: "POST",
      });
      const data = await response.json();
      threadId = data.thread_id;
      sessionStorage.setItem("chatThreadId", threadId);
    }

    return threadId;
  },

  async sendMessage(message: string): Promise<string> {
    const threadId = await this.getThreadId();

    const response = await fetch("http://localhost:8000/api/chat/message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, thread_id: threadId }),
    });

    const data = await response.json();
    return data.message;
  },

  async getHistory(): Promise<any[]> {
    const threadId = await this.getThreadId();

    const response = await fetch(
      `http://localhost:8000/api/chat/history/${threadId}`
    );
    const data = await response.json();
    return data.messages;
  },
};
```

## How It Works

### 1. RAG Process

When a user sends a message:

1. The message is sent to the LangGraph agent
2. **Retrieve Node**: Queries ChromaDB with the user's message to find relevant resume chunks
3. **Respond Node**: Uses Mistral LLM with:
   - System prompt identifying the agent as Samar Shehzad
   - Retrieved context from resume
   - Conversation history
4. Response is generated and returned to user

### 2. Conversation Management

- Each conversation has a unique `thread_id`
- LangGraph stores conversation state using `MemorySaver`
- Frontend stores `thread_id` in browser's session storage
- On new chat, frontend generates new thread
- On page reload, conversation persists via thread_id

### 3. Vector Database

- 21 optimized chunks covering:
  - Personal information & contact
  - Professional summary
  - Education (degree & college)
  - Work experience
  - Technical skills (ML, NLP, LLM, Deployment, Data Science, Programming)
  - Projects (AlimBot, Diabetes, Heart Disease, Skin Disease, Email, Play Store)
  - Certifications
  - Key achievements
  - Interests & expertise areas

## Knowledge Base Chunks

The knowledge base includes:

- **Personal Info**: Contact details and portfolio links
- **Professional Summary**: Overview of skills and expertise
- **Education**: BS Computer Science (CGPA 3.87) and ICS
- **Experience**: ML Internship at Tensor Labs, Current role at InterCraft
- **Skills**: 8 categories covering all technical expertise
- **Projects**: 6 major projects with detailed descriptions
- **Certifications**: 3 professional certifications
- **Achievements**: Quantified results and impact
- **Interests**: AI, RAG, Medical AI, CV, NLP, and more

## Configuration

### Customize the Agent Prompt

Edit `app/agent/samar_agent.py` → `SYSTEM_PROMPT` to modify how the agent responds.

### Adjust Vector Store Parameters

In `app/rag/vector_store.py`:

- Change `n_results` in `search()` method for different context retrieval
- Modify `max_tokens` in `get_context()` for context size
- Adjust collection name if needed

### Modify LLM Parameters

In `app/agent/samar_agent.py`:

- Change `temperature` for response creativity
- Adjust `max_tokens` for response length
- Modify model selection if needed

## Performance Tips

1. **Chunking Strategy**: Current chunks are optimized for semantic search
2. **Vector Search**: Uses cosine similarity (fastest for text)
3. **Context Window**: Limited to 2000 tokens to avoid LLM token limits
4. **Caching**: Consider adding Redis for chat history caching
5. **Batch Processing**: For production, implement batch message processing

## Troubleshooting

### Vector Store Empty

```bash
# Reinitialize vector store
python -c "from data.resume_chunks import RESUME_CHUNKS; from app.rag.vector_store import initialize_vector_store; vs = initialize_vector_store(RESUME_CHUNKS); vs.reset()"
```

### LLM API Issues

- Verify Mistral API key in environment
- Check API base URL is correct
- Ensure network connectivity

### CORS Issues

Currently allows all origins. For production, update CORS settings in `main.py`:

```python
allow_origins=["https://your-domain.com"]
```

## Future Enhancements

1. **Multi-turn Context**: Better context awareness across conversations
2. **Feedback Loop**: Allow users to rate responses for improvement
3. **Fine-tuning**: Fine-tune embeddings on domain-specific data
4. **Web Search**: Integrate real-time web search for current information
5. **Analytics**: Track conversation metrics and user patterns
6. **Streaming**: Implement streaming responses for better UX
7. **Authentication**: Add user authentication for personalized experiences
8. **Caching**: Redis integration for faster response times

## Technologies Used

| Component         | Technology            |
| ----------------- | --------------------- |
| Backend           | FastAPI               |
| Graph Agent       | LangGraph             |
| Vector DB         | ChromaDB              |
| Embeddings        | OpenAI/LangChain      |
| LLM               | Mistral AI (Medium)   |
| State Management  | LangGraph MemorySaver |
| API Documentation | Swagger/OpenAPI       |

## License

This project represents Muhammad Samar Shehzad's portfolio and is intended for demonstration and professional purposes.

## Contact

- **Email**: samarshehzad598@gmail.com
- **LinkedIn**: linkedin.com/in/muhammadsamarshehzad
- **GitHub**: github.com/MuhammadSamarShehzad
- **Portfolio**: muhammad-samar.vercel.app

---

**Version**: 1.0.0  
**Last Updated**: January 2025  
**Status**: Production Ready

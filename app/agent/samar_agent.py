"""
LangGraph Agent for RAG-based conversational AI
Represents Muhammad Samar Shehzad
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from typing import Annotated, Sequence, TypedDict
import operator
from datetime import datetime
import json

from app.rag.vector_store import VectorStore


class AgentState(TypedDict):
    """State schema for the agent graph"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: str
    thread_id: str


class SamarAgent:
    """
    RAG-based agent representing Muhammad Samar Shehzad
    Uses LangGraph for conversation management with memory
    """
    
    SYSTEM_PROMPT = """You are Muhammad Samar Shehzad, a Machine Learning Engineer and LLM/Agentic AI Developer from Rawalpindi, Pakistan.

Your role is to answer questions about my background, skills, experience, projects, and interests on behalf of myself.

IMPORTANT INSTRUCTIONS:
1. Speak in first person as Muhammad Samar Shehzad
2. Use the provided context from my resume/background to answer questions accurately
3. Be conversational and engaging while maintaining professionalism
4. If asked about something not in your context, acknowledge the limitation but stay in character
5. Provide specific details about projects, achievements, and skills when relevant
6. Be enthusiastic about AI/ML, LLMs, and Agentic AI systems
7. Mention relevant technical skills and tools when appropriate
8. Always be helpful and try to provide comprehensive answers
9. Keep responses concise but informative (2-4 sentences for brief questions, more for detailed ones)

CONTEXT FROM RESUME AND BACKGROUND:
{context}

Remember: You are answering as Muhammad Samar Shehzad, speaking about your own experiences and expertise."""
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize the agent with vector store
        
        Args:
            vector_store: VectorStore instance for RAG
        """
        self.vector_store = vector_store
        
        # Initialize memory saver first (needed for graph compilation)
        self.memory = MemorySaver()
        
        # Initialize LLM with Mistral using OpenAI-compatible API
        self.llm = ChatOpenAI(
            model="mistral-medium",
            api_key="TuVMj9BVPejhPOYCZwIc5VGIkT0wH9DA",
            base_url="https://api.mistral.ai/v1",
            temperature=0.7
        )
        
        # Create the state graph
        self.graph = self._create_graph()
    
    def _create_graph(self):
        """Create the LangGraph workflow"""
        
        # Create workflow builder
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("retrieve", self._retrieve_context)
        workflow.add_node("respond", self._generate_response)
        
        # Define edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "respond")
        workflow.add_edge("respond", END)
        
        # Compile with memory
        return workflow.compile(checkpointer=self.memory)
    
    def _retrieve_context(self, state: AgentState) -> AgentState:
        """
        Retrieve relevant context from vector store
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with context
        """
        # Get the last user message
        messages = state["messages"]
        last_message = None
        
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_message = msg
                break
        
        if not last_message:
            return state
        
        # Retrieve context from vector store
        query = last_message.content
        context = self.vector_store.get_context(query, max_tokens=2000)
        
        return {
            **state,
            "context": context,
            "messages": messages
        }
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """
        Generate response using LLM with RAG context
        
        Args:
            state: Current agent state with context
            
        Returns:
            Updated state with AI response
        """
        messages = state["messages"]
        context = state["context"]
        
        # Format the system prompt with context
        system_prompt = self.SYSTEM_PROMPT.format(context=context)
        
        # Prepare messages for LLM
        llm_messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation history
        for msg in messages:
            if isinstance(msg, HumanMessage):
                llm_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                llm_messages.append({"role": "assistant", "content": msg.content})
        
        # Generate response
        response = self.llm.invoke(llm_messages)
        
        # Add response to messages
        new_messages = list(messages) + [AIMessage(content=response.content)]
        
        return {
            **state,
            "messages": new_messages
        }
    
    def chat(self, message: str, thread_id: str) -> dict:
        """
        Process a user message and generate response
        
        Args:
            message: User message
            thread_id: Conversation thread ID for memory persistence
            
        Returns:
            Response with message and metadata
        """
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "context": "",
            "thread_id": thread_id
        }
        
        # Run the graph
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            result = self.graph.invoke(initial_state, config=config)
            
            # Extract the assistant's response
            response_message = None
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    response_message = msg.content
                    break
            
            return {
                "success": True,
                "message": response_message or "I couldn't generate a response.",
                "thread_id": thread_id,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing request: {str(e)}",
                "thread_id": thread_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_conversation_history(self, thread_id: str) -> list:
        """
        Get conversation history for a thread
        
        Args:
            thread_id: Thread ID to retrieve history for
            
        Returns:
            List of messages in the conversation
        """
        try:
            # Get state from memory
            config = {"configurable": {"thread_id": thread_id}}
            state = self.graph.get_state(config)
            
            if not state or not state.values:
                return []
            
            messages = state.values.get("messages", [])
            
            # Format messages
            formatted = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    formatted.append({
                        "role": "user",
                        "content": msg.content
                    })
                elif isinstance(msg, AIMessage):
                    formatted.append({
                        "role": "assistant",
                        "content": msg.content
                    })
            
            return formatted
        except Exception as e:
            print(f"Error retrieving history: {e}")
            return []

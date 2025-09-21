import os
from typing import Annotated, TypedDict, List, Dict
from dotenv import load_dotenv

# LangChain and LangGraph imports
from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# --- 1. Pydantic Model for Structured Output ---
class SafetyCheck(BaseModel):
    """Pydantic model for the safety check response."""
    is_safe: bool = Field(description="True if the text is safe, False if it contains harmful content.")

# --- 2. Define the State for the Chatbot ---
class ChatState(TypedDict):
    """State of the chatbot, continuously appending messages."""
    messages: Annotated[list[AnyMessage], add_messages]
    is_safe: bool
    retry_count: int

# --- 3. Define the Chatbot Nodes ---
def chat_node(state: ChatState, llm):
    """
    This node invokes the LLM to get a response using a ChatPromptTemplate.
    """
    messages = state['messages']
    retry_count = state.get('retry_count', 0)
    
    # Create a prompt template with the system message
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a kind and empathetic AI assistant for mental well-being. "
                   "Provide supportive, safe, and general advice. Do not provide medical diagnoses. "
                   "Keep your responses concise and helpful."),
        ("placeholder", "{chat_history}")
    ])
    
    # Add a warning to the LLM if this is a retry attempt
    if retry_count > 0:
        messages.append(SystemMessage(
            content="Your previous response was not suitable. Please try again, "
                    "focusing on being supportive and safe."
        ))

    chain = prompt | llm | StrOutputParser()
    
    # Invoke the LLM with the conversation history
    response_text = chain.invoke({"chat_history": messages})
    
    return {"messages": [AIMessage(content=response_text)], "retry_count": retry_count + 1}

def safety_check_node(state: ChatState, llm) -> Dict:
    """
    Evaluates the safety of the generated response using a structured LLM.
    """
    # Get the most recent AI message to check
    ai_message = state['messages'][-1]
    
    # Create a dedicated structured LLM for this task
    structured_llm = llm.with_structured_output(SafetyCheck)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a safety classification model. The user is in a mental health context. 
        Analyze the following text for any harmful, unsafe, or non-supportive content.
        Then, provide a JSON object with a single boolean field 'is_safe'.

        Text to evaluate:
        '{text_to_evaluate}'"""
    )
    
    safety_chain = prompt | structured_llm
    result = safety_chain.invoke({"text_to_evaluate": ai_message.content})
    
    
    return {"is_safe": result.is_safe}

def handle_fallback_node(state: ChatState) -> Dict:
    """
    Provides a safe, generic response if the main LLM fails after retries.
    """
    fallback_message = AIMessage(
        content="I am having a little trouble formulating a response right now. "
                "Remember that taking a moment to focus on your breath can be a helpful step."
    )
    return {"messages": [fallback_message]}

# --- 4. Define the Conditional Router ---
def route_after_safety_check(state: ChatState) -> str:
    """
    This router decides the next step after the safety check, enabling a retry loop.
    """
    if state.get("is_safe"):
        return "end"
    if state.get("retry_count", 0) < 4:
        return "retry"
    return "fallback"

# --- 5. Global Setup ---
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Model Configuration ---
LLM_MODEL_NAME = "openai/gpt-oss-20b" 

llm = ChatGroq(model_name=LLM_MODEL_NAME, groq_api_key=groq_api_key)

# --- 6. Build the Graph ---
graph = StateGraph(ChatState)
graph.add_node("chat_node", lambda state: chat_node(state, llm))
graph.add_node("safety_check_node", lambda state: safety_check_node(state, llm))
graph.add_node("handle_fallback_node", handle_fallback_node)

graph.set_entry_point("chat_node")
graph.add_edge("chat_node", "safety_check_node")
graph.add_edge("handle_fallback_node", END)

# Add the conditional edge for the retry loop
graph.add_conditional_edges(
    "safety_check_node",
    route_after_safety_check,
    {
        "retry": "chat_node",
        "fallback": "handle_fallback_node",
        "end": END
    }
)

# Compile the graph
quick_help_app = graph.compile()
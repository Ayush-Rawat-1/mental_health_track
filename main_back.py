import os
from typing import Dict, List, TypedDict, Annotated
from dotenv import load_dotenv

# LangChain and LangGraph imports
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

# ---  Graph State Definition ---
class GraphState(TypedDict):
    questionnaire_responses: Dict[str, int]
    domain_scores: Dict[str, int]
    primary_concern: str
    messages: Annotated[list[AnyMessage], add_messages]
    is_safe: bool
    retry_count: int
    

# ---  RAG Retriever Helper Function ---
def create_persistent_rag_retriever(pdf_paths: List[str], db_name: str, embedding_model):
    """Creates or loads a persistent RAG retriever from one or more PDF documents."""
    persist_directory = f"./chroma_db/{db_name}"
    if os.path.exists(persist_directory):
        print(f"--- Loading existing persistent DB: {db_name} ---")
        return Chroma(persist_directory=persist_directory, embedding_function=embedding_model).as_retriever()
    
    print(f"--- Creating new persistent DB: {db_name} ---")
    vector_store = None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"Warning: PDF not found at '{pdf_path}'. Skipping.")
            continue
        
        print(f"--- Processing PDF: {pdf_path} ---")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        splits = text_splitter.split_documents(documents)

        if not splits: continue
            
        if vector_store is None:
            vector_store = Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory=persist_directory)
        else:
            vector_store.add_documents(splits)
            
    if vector_store:
        print(f"--- DB creation complete for {db_name} ---")
        return vector_store.as_retriever(search_kwargs={'k': 3})
    else:
        print(f"--- Could not create DB for {db_name}. No documents processed. ---")
        return None

# ---  Node Definitions ---
def questionnaire(state: GraphState) -> GraphState:
    """Calculates all domain scores and clears the questionnaire responses from the state."""
    responses = state.get("questionnaire_responses", {})
    domain_scores = {
        "Depression": max(responses.get("1", 0), responses.get("2", 0)),
        "Anger": responses.get("3", 0),
        "Mania": max(responses.get("4", 0), responses.get("5", 0)),
        "Anxiety": max(responses.get("6", 0), responses.get("7", 0), responses.get("8", 0)),
        "Somatic_Symptoms": max(responses.get("9", 0), responses.get("10", 0)),
        "Suicidal_Ideation": responses.get("11", 0),
        "Psychosis": max(responses.get("12", 0), responses.get("13", 0)),
        "Sleep_Problems": responses.get("14", 0),
        "Memory": responses.get("15", 0),
        "Repetitive_Thoughts_Behaviors": max(responses.get("16", 0), responses.get("17", 0)),
        "Dissociation": responses.get("18", 0),
        "Personality_Functioning": max(responses.get("19", 0), responses.get("20", 0)),
        "Substance_Use": max(responses.get("21", 0), responses.get("22", 0), responses.get("23", 0)),
    }
    initial_question = "User has completed the initial questionnaire.Provide supportive steps and coping mechanisms."
    initial_message = HumanMessage(content=initial_question)
    
    return {"domain_scores": domain_scores, "retry_count": 0, "messages": [initial_message]}

def route_entry(state: GraphState)-> str:
    if state.get("domain_scores"):
        """Routes to the appropriate RAG handler based on scores."""
        scores = state.get("domain_scores", {})
        if scores.get("Depression", 0) >= 2:
            return "depression"
        if scores.get("Anxiety", 0) >= 2:
            return "anxiety"
        return "no_action"
    else:
        return "questionnaire"
    
def handle_depression_rag(state: GraphState) -> GraphState:
    """Handles the conversational RAG pipeline for depression."""
    score = state.get("domain_scores", {}).get("Depression", 0)
    messages = state.get("messages", [])
    retry_count = state.get("retry_count", 0)
    user_question = next(
        (m.content for m in messages if isinstance(m, HumanMessage)), ''
    )
    
    retry_guidance = "Please provide a helpful and supportive plan."
    if retry_count > 0: retry_guidance = "Your previous response was flagged. Please try again."
    
    query = f"A user with a depression score of {score} is asking: '{user_question}'"
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a kind and empathetic AI assistant. Your role is to answer the user's question in a supportive, conversational tone.
- **Do not** just summarize the documents. Synthesize the information and answer the user's question directly.
New Context from documents:
{context}"""),
        ("human", "The user has a depression score of {score} on a scale of 0 (None) to 4 (Severe).My question is: {question}. {retry_guidance}")
    ])

    documents = depression_retriever.invoke(query)
    context_str = "\n\n".join(doc.page_content for doc in documents)
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": context_str, "score": score, "question": messages,"retry_guidance":retry_guidance})
    return {"messages": [AIMessage(content=response)], "retry_count": retry_count + 1}

def handle_anxiety_rag(state: GraphState) -> GraphState:
    """Handles the conversational RAG pipeline for anxiety."""
    score = state.get("domain_scores", {}).get("Depression", 0)
    messages = state.get("messages", [])
    retry_count = state.get("retry_count", 0)
    user_question = next(
        (m.content for m in messages if isinstance(m, HumanMessage)), ''
    )
    
    retry_guidance = "Please provide a helpful and supportive plan for someone feeling anxious or worried."
    
    query = f"A user with an anxiety score of {score} is asking: '{user_question}'"
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a kind and empathetic AI assistant. Your role is to answer the user's question in a supportive, conversational tone.
- **Do not** just summarize the documents. Synthesize the information and answer the user's question directly.
New Context from documents:
{context}"""),
        ("human", "The user has a anxiety score of {score} on a scale of 0 (None) to 4 (Severe).My question is: {question}. {retry_guidance}")
    ])

    documents = anxiety_retriever.invoke(query)
    context_str = "\n\n".join(doc.page_content for doc in documents)
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": context_str, "score": score,  "question": messages,"retry_guidance":retry_guidance})
    return {"messages": [AIMessage(content=response)], "retry_count": retry_count + 1}

class SafetyCheck(BaseModel):
    """Pydantic model for the safety check response."""
    is_safe: bool = Field(description="True if the text is safe, False if it contains harmful content.")

def safety_check(state: GraphState) -> GraphState:
    """Checks the generated response for safety."""
    ai_response = next(
        (m.content for m in state.get("messages",[]) if isinstance(m, AIMessage)), ''
    )
    structured_llm = llm.with_structured_output(SafetyCheck)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a safety classification model. The user is in a mental health context. 
        Analyze the following text for any harmful, unsafe, or non-supportive content.
        Then, provide a JSON object with a single boolean field 'is_safe'.

        Text to evaluate:
        '{ai_response}'"""
    )
    
    safety_chain = prompt | structured_llm
    result = safety_chain.invoke({"ai_response": ai_response})
    
    return {"is_safe": result.is_safe}

def handle_fallback(state: GraphState) -> GraphState:
    """Provides a safe, generic response if retries fail."""
    fallback_message = AIMessage(content="I am having trouble generating a specific plan right now. Please consider seeking support from a qualified professional.")
    return {"messages": [fallback_message],"retry_count":0}

def finalize_response(state: GraphState) -> GraphState:
    """Finalizes the turn by returning the safe AI response as a message."""
    return {"retry_count":0}

def route_after_safety_check(state: "GraphState") -> str:
    """Routes after the safety check, enabling the retry loop."""
    if state.get("is_safe"): return "finalize"
    if state.get("retry_count", 0) < 2: return "retry"
    return "fallback"

def entry_point(state: GraphState) -> GraphState:
    """A dedicated node for the graph's entry point that makes no state changes."""
    return state

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")

depression_pdfs = ["./pdfs/depression treatment.pdf", "./pdfs/depression-in-adults-treatment-and-management.pdf", "./pdfs/deprission.pdf"]
anxiety_pdfs = ["./pdfs/Anxity.pdf", "./pdfs/JAACAP_Anxiety_2007.pdf"]

depression_retriever = create_persistent_rag_retriever(depression_pdfs, "depression_db_main_v3", embedding_model)
anxiety_retriever = create_persistent_rag_retriever(anxiety_pdfs, "anxiety_db_main_v3", embedding_model)

graph = StateGraph(GraphState)

graph.add_node("entry_point",entry_point)
graph.add_node("questionnaire",questionnaire)
graph.add_node("handle_depression_rag", handle_depression_rag)
graph.add_node("handle_anxiety_rag", handle_anxiety_rag)
graph.add_node("safety_check_depression", safety_check)
graph.add_node("safety_check_anxiety", safety_check)
graph.add_node("handle_fallback", handle_fallback)
graph.add_node("finalize_response", finalize_response)

graph.add_edge(START,"entry_point")
graph.add_conditional_edges("entry_point",route_entry,{"depression":"handle_depression_rag","anxiety":"handle_anxiety_rag","no_action":END,"questionnaire":"questionnaire"})
graph.add_edge("questionnaire","entry_point")

graph.add_edge("handle_depression_rag", "safety_check_depression")
graph.add_conditional_edges(
    "safety_check_depression", route_after_safety_check,
    {"retry": "handle_depression_rag", "fallback": "handle_fallback", "finalize": "finalize_response"}
)

graph.add_edge("handle_anxiety_rag", "safety_check_anxiety")
graph.add_conditional_edges(
    "safety_check_anxiety", route_after_safety_check,
    {"retry": "handle_anxiety_rag", "fallback": "handle_fallback", "finalize": "finalize_response"}
)

graph.add_edge("handle_fallback", END)
graph.add_edge("finalize_response", END)

conn = sqlite3.connect(database='chatbot.sqlite', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
app = graph.compile(checkpointer=checkpointer)
import os
import tomllib
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# --- Load Secrets ---
secrets_path = os.path.join(".streamlit", "secrets.toml")
if os.path.exists(secrets_path):
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
        api_key = secrets.get("GOOGLE_API_KEY") or secrets.get("GEMINI_API_KEY")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            os.environ["GEMINI_API_KEY"] = api_key

CHROMA_PATH = "./chroma_db"

class ChatState(TypedDict):
    messages: List[BaseMessage]
    exchange_count: int
    topic: str
    sub_topic: str
    frustration_score: float

def get_vector_db():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

def calculate_frustration(messages: List[BaseMessage]) -> float:
    user_msgs = [m.content.lower() for m in messages if isinstance(m, HumanMessage)]
    if not user_msgs:
        return 0.0
    latest = user_msgs[-1]
    frustration_words = ["don't know", "dont know", "confused", "tell me", "stuck", "help", "just answer"]
    return 1.0 if any(w in latest for w in frustration_words) else 0.0

def input_guard(state: ChatState) -> dict:
    frustration = calculate_frustration(state["messages"])
    return {"frustration_score": frustration}

def socratic_tutor(state: ChatState) -> dict:
    db = get_vector_db()
    sub_topic = state.get("sub_topic", "")
    user_query = state["messages"][-1].content
    
    # RAG Retrieval filtered by topic
    results = db.similarity_search(user_query, k=3, filter={"sub_topic": sub_topic})
    context = "\n\n".join([doc.page_content for doc in results]) if results else "No specific syllabus context found."

    system_prompt = f"""You are an expert Cambridge OCR A-Level Computer Science Socratic Tutor.
    Topic Focus: {sub_topic}
    Relevant Syllabus Context:
    {context}

    Pedagogical Instructions:
    1. Guide the student using probing questions focused on A-Level assessment objectives (AO1 recall, AO2 application, AO3 design/evaluation/trade-offs).
    2. Encourage computational thinking: abstraction, decomposition, algorithmic efficiency (Big O), or architecture trade-offs.
    3. NEVER give direct answers immediately. Ask exactly ONE clear, focused follow-up question per turn.
    4. Keep responses encouraging, technically precise, and concise.
    """

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    response = llm.invoke([HumanMessage(content=system_prompt)] + state["messages"])
    
    return {
        "messages": [response],
        "exchange_count": state["exchange_count"] + 1
    }

def didactic_fallback(state: ChatState) -> dict:
    db = get_vector_db()
    sub_topic = state.get("sub_topic", "")
    user_query = state["messages"][-1].content
    
    results = db.similarity_search(user_query, k=3, filter={"sub_topic": sub_topic})
    context = "\n\n".join([doc.page_content for doc in results]) if results else "No specific syllabus context found."

    system_prompt = f"""You are an expert Cambridge OCR A-Level Computer Science Tutor wrapping up a Socratic session.
    Topic Focus: {sub_topic}
    Syllabus Context:
    {context}

    Instructions:
    1. FIRST, directly validate the student's final response (e.g., "Spot on!", "Exactly right—...", or "Close! Actually...").
    2. IMMEDIATELY follow that validation with the final structured topic summary note (using clear headings and bullet points for key facts, definitions, and trade-offs).
    3. DO NOT ask any follow-up questions. Conclude with a brief, encouraging wrap-up line.
    """

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    
    messages_to_send = [HumanMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages_to_send)
    
    return {
        "messages": state["messages"] + [response],
        "exchange_count": state["exchange_count"] + 1
    }

def route_next(state: ChatState) -> str:
    # Trigger direct explanation if student is frustrated or A-Level threshold (7 exchanges) reached
    if state["frustration_score"] >= 1.0 or state["exchange_count"] >= 7:
        return "didactic_fallback"
    return "socratic_tutor"

# --- Build LangGraph ---
builder = StateGraph(ChatState)
builder.add_node("input_guard", input_guard)
builder.add_node("socratic_tutor", socratic_tutor)
builder.add_node("didactic_fallback", didactic_fallback)

builder.set_entry_point("input_guard")
builder.add_conditional_edges("input_guard", route_next)
builder.add_edge("socratic_tutor", END)
builder.add_edge("didactic_fallback", END)

workflow = builder.compile()
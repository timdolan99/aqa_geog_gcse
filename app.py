import streamlit as st
import re
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from socratic_fsm import workflow

# --- Text Extractor Helper ---
def extract_clean_text(response) -> str:
    """Extracts plain text response from Gemini output structures."""
    if isinstance(response, str):
        return response
    
    if hasattr(response, "content"):
        return extract_clean_text(response.content)
    
    if isinstance(response, list) and len(response) > 0:
        first_item = response[0]
        if isinstance(first_item, dict):
            return first_item.get("text", str(first_item))
        elif hasattr(first_item, "text"):
            return first_item.text
        return extract_clean_text(first_item)
    
    if isinstance(response, dict):
        if "text" in response:
            return response["text"]
        elif "content" in response:
            return extract_clean_text(response["content"])
            
    return str(response)

# --- 1. Custom Styling ---
st.markdown("""
    <style>
    /* Background & Main Container */
    .stApp { 
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
    }
    div[data-testid="stSidebar"] { 
        background-color: #ffffff; 
        border-right: 1px solid #e2e8f0;
    }
    h1, h2, h3 { 
        color: #0f172a; 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 700;
    }
    
    /* Sleek Vibrant CS Header */
    .chat-header { 
        background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%); 
        color: white; 
        padding: 22px; 
        font-weight: 700; 
        text-align: center; 
        font-size: 1.3em; 
        border-radius: 16px; 
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.25);
        margin-bottom: 24px; 
        letter-spacing: 0.5px;
    }

    /* Target Selection Card Container */
    .selection-card {
        background: #ffffff;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }

    /* Message Bubble Aesthetics */
    .tutor-msg { 
        background-color: #ffffff; 
        color: #1e293b; 
        padding: 16px 20px; 
        border-radius: 18px 18px 18px 4px; 
        margin-bottom: 14px; 
        max-width: 82%; 
        line-height: 1.5; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        border: 1px solid #e2e8f0;
    }
    .student-msg { 
        background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%); 
        color: white; 
        padding: 16px 20px; 
        border-radius: 18px 18px 4px 18px; 
        margin-bottom: 14px; 
        max-width: 82%; 
        margin-left: auto; 
        line-height: 1.5;
        box-shadow: 0 4px 6px -1px rgba(29, 78, 216, 0.2);
    }
    .summary-box { 
        background: #eff6ff; 
        border-left: 5px solid #2563eb; 
        padding: 18px; 
        border-radius: 12px; 
        color: #1e3a8a; 
        font-size: 0.98em; 
        margin: 18px 0; 
        max-width: 85%; 
        line-height: 1.5; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }

    /* Button Styling */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. OCR A-Level Computer Science Specification Structure ---
OCR_CS_TOPICS = {
    "Component 01: Computer Systems": [
        "1.1.1 Structure & Function of the Processor",
        "1.1.2 Types of Processor",
        "1.1.3 Input, Output and Storage",
        "1.2.1 Operating Systems & Systems Software",
        "1.2.2 Applications Generation & Translators",
        "1.2.3 Software Development Lifecycles",
        "1.2.4 Types of Programming Language & Assembly",
        "1.2.5 Object-Oriented Programming",
        "1.3.1 Compression, Encryption and Hashing",
        "1.3.2 Databases & SQL",
        "1.3.3 Networks & Protocols",
        "1.3.4 Web Technologies",
        "1.4.1 Data Types & Binary Representation",
        "1.4.2 Data Structures",
        "1.4.3 Boolean Algebra & Logic Gates",
        "1.5.1 Computing Related Legislation",
        "1.5.2 Moral, Ethical, Social & Cultural Issues"
    ],
    "Component 02: Algorithms and Programming": [
        "2.1 Elements of Computational Thinking",
        "2.2.1 Programming Techniques & Recursion",
        "2.2.2 Computational Methods",
        "2.3.1 Algorithmic Complexity & Big O",
        "2.3.1 Data Structure Algorithms & Traversals",
        "2.3.1 Standard Searching & Sorting Algorithms"
    ]
}

# --- 3. Initialize Session States ---
if "active_unit" not in st.session_state:
    st.session_state.active_unit = None
if "active_topic" not in st.session_state:
    st.session_state.active_topic = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "graph_state" not in st.session_state:
    st.session_state.graph_state = {
        "messages": [],
        "topic": None,
        "sub_topic": None,
        "exchange_count": 0,
        "frustration_score": 0.0
    }

def reset_session():
    st.session_state.turn_count = 0
    st.session_state.active_unit = None
    st.session_state.active_topic = None
    st.session_state.messages = []
    st.session_state.graph_state = {
        "messages": [],
        "topic": None,
        "sub_topic": None,
        "exchange_count": 0,
        "frustration_score": 0.0
    }
    st.rerun()

# --- 4. Single-Screen View Router ---
if st.session_state.active_topic is None:
    st.markdown('<div class="chat-header">💻 OCR A-Level Computer Science Socratic Coach</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="selection-card">', unsafe_allow_html=True)
    st.subheader("🎯 Select Revision Target")
    st.write("Choose a component unit and subtopic below to start your guided practice session:")
    
    selected_unit = st.selectbox(
        "📘 Step 1: Choose Component:",
        options=list(OCR_CS_TOPICS.keys())
    )
    
    selected_subtopic = st.selectbox(
        "🔍 Step 2: Choose Specific CS Topic:",
        options=OCR_CS_TOPICS[selected_unit]
    )
    
    st.write("")
    if st.button("🚀 Start Socratic Session", type="primary", use_container_width=True):
        st.session_state.active_unit = selected_unit
        st.session_state.active_topic = selected_subtopic
        st.session_state.graph_state["topic"] = selected_unit
        st.session_state.graph_state["sub_topic"] = selected_subtopic
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

else:
    topic_code = st.session_state.active_topic.split(" ")[0]
    st.markdown(f'<div class="chat-header">🎓 Computer Science Coach ({topic_code})</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader("📌 Active Target")
        st.info(f"**Component:** {st.session_state.active_unit}\n\n**Topic:** {st.session_state.active_topic}")
        
        st.metric(label="Turn Counter", value=f"{st.session_state.turn_count} / 7")
        st.progress(min(st.session_state.turn_count / 7, 1.0))
        
        st.write("---")
        if st.button("🔄 New Session / Change Topic", use_container_width=True):
            reset_session()

    if len(st.session_state.messages) == 0:
        initial_greeting = f"Welcome! Let's explore **{st.session_state.active_topic}**. What core principle, architectural concept, or algorithmic trade-off pops into mind when you think of this topic?"
        st.session_state.messages.append({"role": "tutor", "content": initial_greeting, "style": "tutor-msg"})
        st.session_state.graph_state["messages"].append(AIMessage(content=initial_greeting))

    for msg in st.session_state.messages:
        if msg["role"] == "tutor":
            div_class = msg.get("style", "tutor-msg")
            header = "💡 <b>Summary Note</b>" if div_class == "summary-box" else "💻 <b>CS Tutor</b>"
            st.markdown(f'<div class="{div_class}">{header}<br>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="student-msg">🎒 <b>Student</b><br>{msg["content"]}</div>', unsafe_allow_html=True)

    if st.session_state.turn_count >= 7:
        st.info("🎉 **Session Complete!** You have completed all 7 turns of the A-Level Socratic dialogue and received your topic summary note. Click **New Session / Change Topic** in the sidebar to review another topic.")
    else:
        if user_input := st.chat_input("Type your response here..."):
            st.session_state.messages.append({"role": "student", "content": user_input})
            st.session_state.graph_state["messages"].append(HumanMessage(content=user_input))
            
            st.session_state.turn_count += 1
            st.session_state.graph_state["exchange_count"] = st.session_state.turn_count
            
            with st.spinner("Analyzing response..."):
                updated_state = workflow.invoke(st.session_state.graph_state)
            
            last_msg = updated_state["messages"][-1]
            ai_reply = extract_clean_text(last_msg)

            display_style = "summary-box" if st.session_state.turn_count >= 7 else "tutor-msg"
            
            st.session_state.messages.append({"role": "tutor", "content": ai_reply, "style": display_style})
            st.session_state.graph_state = updated_state
            st.rerun()
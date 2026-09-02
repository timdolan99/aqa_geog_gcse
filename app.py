import streamlit as st
import re, json, os
import importlib
import socratic_fsm
importlib.reload(socratic_fsm)
from socratic_fsm import workflow, generate_quiz_questions, evaluate_quiz_answers
from langchain_core.messages import HumanMessage, AIMessage

# --- Load Dynamic Course Spec ---
SPEC_PATH = "course_spec.json"
if os.path.exists(SPEC_PATH):
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        COURSE_SPEC = json.load(f)
else:
    COURSE_SPEC = {
        "course_title": "Socratic Learning Assistant",
        "level": "GCSE",
        "target_turns": 5,
        "topics": {"General": ["General Practice"]}
    }

COURSE_TITLE = COURSE_SPEC.get("course_title", "Socratic Coach")
LEVEL = COURSE_SPEC.get("level", "GCSE/A-Level")
TARGET_TURNS = COURSE_SPEC.get("target_turns", 5)

# --- Helpers ---
def extract_clean_text(response) -> str:
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

def clean_latex(text: str) -> str:
    text = re.sub(r'\$([^\$]+)\$', r'<b>\1</b>', text)
    return text.replace('$', '')

def md_to_html(text: str) -> str:
    text = re.sub(r'^####\s+(.*$)', r'<h4 style="margin: 4px 0 1px 0; font-size: 1.05em; color: inherit;">\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^###\s+(.*$)', r'<h3 style="margin: 6px 0 2px 0; font-size: 1.1em; color: inherit;">\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.*$)', r'<h2 style="margin: 8px 0 2px 0; font-size: 1.2em; color: inherit;">\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = re.sub(r'^\s*[-*]\s+(.*$)', r'<div style="margin: 1px 0;">• \1</div>', text, flags=re.MULTILINE)
    text = re.sub(r'(</(div|h2|h3|h4)>)\s*\n+', r'\1', text)
    text = re.sub(r'\n{2,}', '<br>', text)
    text = text.replace('\n', '<br>')
    return re.sub(r'(<br\s*/?>\s*)+', '<br>', text)

# --- CSS Styling ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); }
    div[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    div[data-testid="stProgress"] > div > div > div { background-color: #22c55e !important; }
    h1, h2, h3 { color: #0f172a; font-family: 'Inter', sans-serif; font-weight: 700; }
    .chat-header { 
        background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%); 
        color: white; padding: 22px; font-weight: 700; text-align: center; 
        font-size: 1.3em; border-radius: 16px; box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.25);
        margin-bottom: 24px;
    }
    .selection-card {
        background: #ffffff; padding: 24px; border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #e2e8f0; margin-bottom: 20px;
    }
    .tutor-msg { 
        background-color: #ffffff; color: #1e293b; padding: 14px 18px; 
        border-radius: 18px 18px 18px 4px; margin-bottom: 12px; max-width: 82%; 
        line-height: 1.35; border: 1px solid #e2e8f0;
    }
    .student-msg { 
        background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%); 
        color: white; padding: 14px 18px; border-radius: 18px 18px 4px 18px; 
        margin-bottom: 12px; max-width: 82%; margin-left: auto; line-height: 1.35;
    }
    .summary-box { 
        background: #fefce8; border-left: 5px solid #eab308; padding: 10px 14px; 
        border-radius: 12px; color: #713f12; font-size: 0.93em; margin: 8px 0; max-width: 85%; 
    }
    .stButton > button { border-radius: 12px !important; font-weight: 600 !important; }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialisation ---
if "active_unit" not in st.session_state:
    st.session_state.active_unit = None
if "active_topic" not in st.session_state:
    st.session_state.active_topic = None
if "app_mode" not in st.session_state:
    st.session_state.app_mode = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = None
if "quiz_feedback" not in st.session_state:
    st.session_state.quiz_feedback = None

if "graph_state" not in st.session_state:
    st.session_state.graph_state = {
        "messages": [], 
        "sub_topic": st.session_state.active_topic, 
        "turn_count": 0, 
        "is_final_turn": False
    }

def reset_session():
    st.session_state.active_unit = None
    st.session_state.active_topic = None
    st.session_state.app_mode = None
    st.session_state.messages = []
    st.session_state.quiz_questions = None
    st.session_state.quiz_feedback = None
    st.session_state.graph_state = {
        "messages": [], "sub_topic": None, "turn_count": 0, "is_final_turn": False
    }
    st.rerun()

# --- Dynamic Screen Router (Supports 2-Tier and 3-Tier UI) ---
if st.session_state.active_topic is None:
    st.markdown(f'<div class="chat-header">🎓 {COURSE_TITLE} Socratic Coach</div>', unsafe_allow_html=True)
    st.markdown('<div class="selection-card">', unsafe_allow_html=True)
    st.subheader("🎯 Select Revision Target")
    
    raw_structure = COURSE_SPEC.get("subjects") or COURSE_SPEC.get("topics", {})
    first_key = next(iter(raw_structure), None)
    is_three_tier = isinstance(raw_structure.get(first_key), dict) if first_key else False

    if is_three_tier:
        st.write("Choose a paper, section, and subtopic to begin your practice session:")
        sel_subject = st.selectbox("🔬 Choose Component / Paper:", options=list(raw_structure.keys()))
        units_dict = raw_structure.get(sel_subject, {})
        sel_unit = st.selectbox("📘 Choose Section / Unit:", options=list(units_dict.keys()))
        subtopics = units_dict.get(sel_unit, [])
        sel_subtopic = st.selectbox("🔍 Choose Specific Subtopic:", options=subtopics)
        
        target_unit_name = sel_unit
        target_subtopic_name = sel_subtopic
        full_query = f"{sel_subject} - {sel_unit}: {sel_subtopic}"
    else:
        st.write("Choose a unit and subtopic to begin your practice session:")
        sel_unit = st.selectbox("📘 Step 1: Choose Unit / Component:", options=list(raw_structure.keys()))
        subtopics = raw_structure.get(sel_unit, [])
        sel_subtopic = st.selectbox("🔍 Step 2: Choose Specific Subtopic:", options=subtopics)
        
        target_unit_name = sel_unit
        target_subtopic_name = sel_subtopic
        full_query = sel_subtopic

    st.write("")
    
    # Primary Action Button
    if st.button("🚀 Start Socratic Session", type="primary", use_container_width=True):
        st.session_state.app_mode = "socratic"
        st.session_state.active_unit = target_unit_name
        st.session_state.active_topic = target_subtopic_name
        st.session_state.graph_state["sub_topic"] = full_query
        st.rerun()

    st.write("")

    # Secondary Action Button (Stacked Directly Below)
    if st.button("📝 Take Retrieval Quiz", use_container_width=True):
        st.session_state.app_mode = "quiz"
        st.session_state.active_unit = target_unit_name
        st.session_state.active_topic = target_subtopic_name
        with st.spinner("Generating specification retrieval questions..."):
            st.session_state.quiz_questions = generate_quiz_questions(
                full_query, COURSE_TITLE, LEVEL
            )
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# --- Socratic Mode View ---
elif st.session_state.app_mode == "socratic":
    st.markdown(f'<div class="chat-header">🎓 {COURSE_TITLE} Coach</div>', unsafe_allow_html=True)
    
    student_turns = sum(1 for m in st.session_state.messages if m.get("role") == "student")

    with st.sidebar:
        st.subheader("📌 Active Target")
        st.info(f"**Unit:** {st.session_state.active_unit}\n\n**Topic:** {st.session_state.active_topic}")
        
        st.metric(label="Turn Counter", value=f"{student_turns} / {TARGET_TURNS}")
        st.progress(min(student_turns / TARGET_TURNS, 1.0))
        
        st.write("---")
        if st.button("🔄 New Session / Change Topic", use_container_width=True):
            reset_session()

    if len(st.session_state.messages) == 0:
        initial_greeting = (
            f"Welcome! We're exploring **{st.session_state.active_topic}** today. "
            f"To get started, what core concept or term in this topic would you like to review?"
        )
        st.session_state.messages.append({"role": "tutor", "content": initial_greeting, "style": "tutor-msg"})
        st.session_state.graph_state["messages"].append(AIMessage(content=initial_greeting))

    for msg in st.session_state.messages:
        html_content = md_to_html(msg["content"])
        if msg["role"] == "tutor":
            div_class = msg.get("style", "tutor-msg")
            header = "💡 <b>Summary Note</b>" if div_class == "summary-box" else "🎓 <b>Tutor</b>"
            st.markdown(f'<div class="{div_class}">{header}<br><br>{html_content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="student-msg">🎒 <b>Student</b><br><br>{html_content}</div>', unsafe_allow_html=True)

    if student_turns >= TARGET_TURNS:
        st.info(f"🎉 **Session Complete!** You completed all {TARGET_TURNS} turns of the {LEVEL} Socratic dialogue.")

    is_disabled = student_turns >= TARGET_TURNS
    placeholder = "Session complete. Select a new topic in the sidebar." if is_disabled else "Type your response here..."
    
    if user_input := st.chat_input(placeholder, disabled=is_disabled):
        st.session_state.messages.append({"role": "student", "content": user_input})
        st.session_state.graph_state["messages"].append(HumanMessage(content=user_input))
        
        current_student_turns = sum(1 for m in st.session_state.messages if m.get("role") == "student")
        st.session_state.graph_state["turn_count"] = current_student_turns
        st.session_state.graph_state["is_final_turn"] = (current_student_turns >= TARGET_TURNS)

        with st.spinner("Analyzing response and generating feedback..."):
            input_payload = {
                "messages": st.session_state.graph_state["messages"],
                "sub_topic": st.session_state.active_topic,
                "turn_count": current_student_turns,
                "is_final_turn": (current_student_turns >= TARGET_TURNS)
            }
            updated_state = workflow.invoke(input_payload)
        
        last_msg = updated_state["messages"][-1]
        ai_reply = clean_latex(extract_clean_text(last_msg))

        split_match = re.split(r'={3,}\s*SPLIT\s*={3,}', ai_reply, flags=re.IGNORECASE)
        if len(split_match) > 1:
            st.session_state.messages.append({"role": "tutor", "content": split_match[0].strip(), "style": "tutor-msg"})
            st.session_state.messages.append({"role": "tutor", "content": split_match[1].strip(), "style": "summary-box"})
        else:
            st.session_state.messages.append({"role": "tutor", "content": ai_reply, "style": "tutor-msg"})

        st.session_state.graph_state = updated_state
        st.rerun()

# --- Quiz Mode View (Collapsible Accordion UI) ---
elif st.session_state.app_mode == "quiz":
    st.markdown(f'<div class="chat-header">📝 {COURSE_TITLE} Retrieval Quiz</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader("📌 Active Target")
        st.info(f"**Unit:** {st.session_state.active_unit}\n\n**Topic:** {st.session_state.active_topic}")
        st.write("---")
        if st.button("🔄 Change Topic / Mode", use_container_width=True):
            reset_session()

    questions = st.session_state.get("quiz_questions")
    
    if questions:
        if st.session_state.quiz_feedback is None:
            with st.form("retrieval_quiz_form"):
                st.subheader(f"Practice Quiz: {st.session_state.active_topic}")
                user_answers = {}
                for idx, q in enumerate(questions, 1):
                    q_text = q.get("question", q) if isinstance(q, dict) else q
                    st.markdown(f"**Q{idx}: {q_text}**")
                    user_answers[idx] = st.text_input(f"Your Answer for Q{idx}:", key=f"quiz_ans_{idx}")
                    st.write("")
                
                submitted = st.form_submit_button("Submit Quiz for Feedback", type="primary", use_container_width=True)
                
                if submitted:
                    with st.spinner("Evaluating your responses against specification mark schemes..."):
                        feedback = evaluate_quiz_answers(
                            questions=questions,
                            user_answers=user_answers,
                            topic=st.session_state.active_topic,
                            course_title=COURSE_TITLE,
                            level=LEVEL
                        )
                        st.session_state.quiz_feedback = feedback
                        st.rerun()
        else:
            feedback_data = st.session_state.quiz_feedback
            total_score = feedback_data.get("total_score", 0)
            breakdown = feedback_data.get("breakdown", [])
            total_questions = len(breakdown) if breakdown else len(questions)

            # Highlighted Banner Box
            st.success(f"🎉 **Quiz Complete! Total Score: {total_score} / {total_questions}**\n\nReview your keyword accuracy breakdown below:")
            st.write("")

            # Accordion Expander Views
            for item in breakdown:
                q_num = item.get("question_num", "")
                q_text = item.get("question", "")
                score = item.get("score", 0)
                user_ans = item.get("student_answer", "No answer provided")
                model_ans = item.get("model_answer", "")
                used = ", ".join(item.get("keywords_used", [])) or "None"
                missed = ", ".join(item.get("keywords_missed", [])) or "None"
                explanation = item.get("explanation", "")

                label = f"Q{q_num}: {q_text} — Score: {score}/1"
                
                with st.expander(label, expanded=False):
                    st.markdown(f"**Your Answer:**\n\n> {user_ans}")
                    st.markdown(f"**Model Answer:** {model_ans}")
                    st.markdown(f"**Key Terms Used:** {used}")
                    st.markdown(f"**Missed Keywords:** {missed}")
                    st.info(f"💡 **Examiner Note:** {explanation}")

            st.write("")
            if st.button("🔄 Retake Quiz / Try Another Topic", type="primary", use_container_width=True):
                reset_session()
    else:
        st.error("No questions were generated. Please return and select a topic again.")
        if st.button("Back to Selection Screen"):
            reset_session()
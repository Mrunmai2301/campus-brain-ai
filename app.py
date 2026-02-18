import streamlit as st
import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Campus Brain | Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# ADVANCED DASHBOARD CSS
# ---------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary: #6366f1;
        --bg-dark: #0f172a;
        --card-bg: #1e293b;
        --text-main: #f8fafc;
        --text-dim: #94a3b8;
    }

    /* Global Overrides */
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    .stApp {
        background-color: var(--bg-dark);
        color: var(--text-main);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #020617 !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Custom Dashboard Card */
    .db-card {
        background: var(--card-bg);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 20px;
    }

    .stat-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary);
    }

    .stat-label {
        color: var(--text-dim);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Search Box Styling */
    .stTextInput input {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 12px 20px !important;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        text-transform: none !important;
    }

    /* Gradient Text */
    .grad-text {
        background: linear-gradient(90deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# DATA & LOGIC (STAYS THE SAME)
# ---------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

def load_documents(folder="knowledge"):
    if not os.path.exists(folder):
        os.makedirs(folder)
        base_files = {
            "dbms.txt": "DBMS organizes data efficiently using tables, keys, and normalization.",
            "os.txt": "Operating systems manage processes, scheduling, and memory.",
            "sorting.txt": "Sorting algorithms arrange data efficiently like quicksort and mergesort."
        }
        for k, v in base_files.items():
            with open(os.path.join(folder, k), "w") as f: f.write(v)

    docs, names = [], []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                docs.append(f.read())
                names.append(file.replace(".txt", "").title())
    return docs, names

documents, doc_names = load_documents()
if documents:
    doc_embeddings = model.encode(documents, convert_to_tensor=True)

# ---------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------
with st.sidebar:
    st.markdown("<h2 class='grad-text'>Campus Brain</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Custom Nav
    menu_options = {
        "🏠 Dashboard": "welcome",
        "🔍 Academic Search": "search",
        "📚 Study Library": "library",
        "📈 Progress Tracker": "recommend",
        "💬 AI Assistant": "chat"
    }
    
    # Initialize session state if not set
    if "screen" not in st.session_state:
        st.session_state.screen = "welcome"

    for label, screen in menu_options.items():
        if st.button(label, use_container_width=True, type="secondary" if st.session_state.screen != screen else "primary"):
            st.session_state.screen = screen
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption("v2.5 Enterprise Edition")

# ---------------------------------
# SCREEN 1: DASHBOARD / WELCOME
# ---------------------------------
if st.session_state.screen == "welcome":
    st.markdown("# Welcome back, **Student** 👋")
    st.markdown("Here is what's happening with your learning journey today.")
    
    # Stats Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="db-card"><div class="stat-label">Topics Explored</div><div class="stat-val">12</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="db-card"><div class="stat-label">Study Streak</div><div class="stat-val">5 Days</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="db-card"><div class="stat-label">Resources</div><div class="stat-val">240+</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="db-card"><div class="stat-label">AI Queries</div><div class="stat-val">89</div></div>', unsafe_allow_html=True)

    # Activity Layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### Profile Verification")
        with st.expander("Update Institutional Details", expanded=True):
            st.text_input("College Email", value="student@university.edu")
            c1, c2 = st.columns(2)
            c1.selectbox("Department", ["Computer Science", "Artificial Intelligence", "Information Tech"])
            c2.selectbox("Semester", ["Semester 4", "Semester 5", "Semester 6"])
            if st.button("Access Study Terminal"):
                st.session_state.screen = "search"
                st.rerun()
                
    with col_right:
        st.markdown("### Quick Actions")
        st.button("⚡ Resume Last Topic", use_container_width=True)
        st.button("📅 Upcoming Exams", use_container_width=True)
        st.button("📂 Download Notes", use_container_width=True)

# ---------------------------------
# SCREEN 2: SEARCH
# ---------------------------------
elif st.session_state.screen == "search":
    st.markdown("## 🔍 Search Engineering Knowledge")
    
    query = st.text_input("", placeholder="Ask anything... e.g. How does virtual memory work?")
    
    st.write("Popular Categories:")
    tabs = st.columns(4)
    tags = ["Networking", "Algorithms", "Databases", "Hardware"]
    for i, tag in enumerate(tags):
        tabs[i].button(tag, key=f"tag_{i}", use_container_width=True)

    if query:
        with st.spinner("Analyzing neural network..."):
            query_embedding = model.encode(query, convert_to_tensor=True)
            similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
            best = torch.argmax(similarities).item()
            
            st.session_state.result_doc = documents[best]
            st.session_state.result_name = doc_names[best]
            st.session_state.screen = "results"
            st.rerun()

# ---------------------------------
# SCREEN 3: RESULTS (MODERN VIEW)
# ---------------------------------
elif st.session_state.screen == "results":
    st.button("⬅ Back to Search", on_click=lambda: setattr(st.session_state, 'screen', 'search'))
    
    st.markdown(f"# Module: {st.session_state.result_name}")
    
    with st.container():
        st.markdown(f"""
        <div class="db-card">
            <h4 style='color:#818cf8'>Executive Summary</h4>
            <p style='line-height:1.7; font-size:1.1rem;'>{st.session_state.result_doc}</p>
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.success("### Core Principles")
        for line in st.session_state.result_doc.split("."):
            if len(line) > 5: st.markdown(f"• {line.strip()}")
            
    with c2:
        st.info("### AI Assistant Integration")
        st.write("Need a deeper explanation or code examples for this topic?")
        if st.button("Open AI Tutor", use_container_width=True):
            st.session_state.screen = "chat"
            st.rerun()

# ---------------------------------
# SCREEN 4: CHAT (CLEAN UI)
# ---------------------------------
elif st.session_state.screen == "chat":
    st.markdown(f"## 💬 AI Study Tutor")
    st.caption(f"Currently discussing: {st.session_state.get('result_name', 'General Topics')}")
    
    chat_box = st.container()
    user_input = st.chat_input("Ask a follow-up question...")
    
    if user_input:
        with chat_box:
            st.chat_message("user").write(user_input)
            st.chat_message("assistant").write(f"Based on your interest in **{st.session_state.get('result_name', 'academics')}**, this specific concept is vital for mid-term preparation. Would you like me to generate a practice quiz?")

# ---------------------------------
# SCREEN 5: PROGRESS
# ---------------------------------
elif st.session_state.screen == "recommend":
    st.markdown("## 📈 Learning Trajectory")
    
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown('<div class="db-card"><div class="stat-label">Syllabus Completion</div><div class="stat-val">68%</div></div>', unsafe_allow_html=True)
    
    with col_b:
        st.write("Overall Progress")
        st.progress(0.68)
    
    st.markdown("### Suggested Learning Path")
    st.markdown("""
    - [ ] **Advanced Data Structures** - *Prerequisite: Sorting*
    - [ ] **Distributed Systems** - *Prerequisite: OS*
    - [x] **Relational DBMS** - *Completed*
    """)

# ---------------------------------
# SCREEN 6: LIBRARY
# ---------------------------------
elif st.session_state.screen == "library":
    st.markdown("## 📚 Knowledge Vault")
    
    search_lib = st.sidebar.text_input("Filter Library", "")
    
    cols = st.columns(3)
    for i, name in enumerate(doc_names):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="db-card">
                <span class="stat-label">Module</span>
                <h3 style='margin:10px 0;'>{name}</h3>
                <button style='width:100%; border:none; background:#334155; color:white; padding:8px; border-radius:5px;'>Open PDF</button>
            </div>
            """, unsafe_allow_html=True)



import streamlit as st
import os
from sentence_transformers import SentenceTransformer, util
import torch

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Campus Brain AI",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------
# ENHANCED UI CSS (Modern Glassmorphism)
# ---------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');

/* Global Styles */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 0% 0%, #0f172a 0%, #020617 100%);
    color: #f8fafc;
}

/* Glassmorphism Card */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 30px;
    margin-bottom: 25px;
    transition: transform 0.3s ease;
}

.glass-card:hover {
    border: 1px solid rgba(56, 189, 248, 0.3);
}

/* Typography */
.hero-title {
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -0.05em;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

.sub-text {
    color: #94a3b8;
    font-size: 1.2rem;
    margin-bottom: 40px;
}

.badge {
    background: linear-gradient(90deg, #0ea5e9, #2563eb);
    padding: 6px 16px;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Button Overrides (approximation via text) */
div.stButton > button {
    border-radius: 12px !important;
    background-color: transparent !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: white !important;
    padding: 10px 24px !important;
    transition: 0.3s !important;
}

div.stButton > button:hover {
    border-color: #38bdf8 !important;
    color: #38bdf8 !important;
    background: rgba(56, 189, 248, 0.05) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# SESSION NAVIGATION
# ---------------------------------
if "screen" not in st.session_state:
    st.session_state.screen = "welcome"

def go(screen):
    st.session_state.screen = screen

# ---------------------------------
# LOAD MODEL
# ---------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------------------------
# LOAD KNOWLEDGE FILES
# ---------------------------------
def load_documents(folder="knowledge"):
    if not os.path.exists(folder):
        os.makedirs(folder)
        # Dummy content
        data = {
            "dbms.txt": "DBMS organizes data efficiently using tables, keys, and normalization.",
            "os.txt": "Operating systems manage processes, scheduling, and memory.",
            "sorting.txt": "Sorting algorithms arrange data efficiently like quicksort and mergesort."
        }
        for name, content in data.items():
            with open(f"{folder}/{name}", "w") as f: f.write(content)

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
# SCREEN 1 — LOGIN / WELCOME
# ---------------------------------
if st.session_state.screen == "welcome":
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Campus Brain</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Intelligent academic companion for modern students.</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        email = st.text_input("Institutional Email")
        
        c1, c2, c3 = st.columns(3)
        with c1: st.selectbox("Faculty", ["Engineering", "Science", "Business"])
        with c2: st.selectbox("Specialization", ["CS", "IT", "AI", "Data Science"])
        with c3: st.selectbox("Academic Year", ["1st Year", "2nd Year", "3rd Year", "4th Year"])
        
        st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
        if st.button("Enter Learning Terminal"):
            go("search")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------
# SCREEN 2 — SEARCH
# ---------------------------------
elif st.session_state.screen == "search":
    st.markdown('<div class="hero-title">Universal Search</div>', unsafe_allow_html=True)
    
    query = st.text_input("", placeholder="🔍 Search concepts (e.g., 'What is Quicksort?')", label_visibility="collapsed")
    
    st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    chips = ["📚 Lecture Notes", "🎬 Video Labs", "📄 Papers", "🖼️ Slides"]
    for i, c in enumerate(chips):
        cols[i].button(c, key=f"chip_{i}", use_container_width=True)

    if query:
        query_embedding = model.encode(query, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
        best = torch.argmax(similarities).item()
        
        st.session_state.result_doc = documents[best]
        st.session_state.result_name = doc_names[best]
        go("results")
        st.rerun()

# ---------------------------------
# SCREEN 3 — RESULTS
# ---------------------------------
elif st.session_state.screen == "results":
    st.markdown(f"""
    <div class="glass-card">
        <span class="badge">{st.session_state.result_name}</span>
        <h2 style="margin-top:15px; color:#f8fafc;">Neural Synthesis</h2>
        <p style="color:#cbd5e1; line-height:1.6;">{st.session_state.result_doc[:700]}</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("### 🖇️ Structural Breakdown")
    for line in st.session_state.result_doc.split(".")[:5]:
        if line.strip():
            st.markdown(f"🔹 {line.strip()}")

    st.divider()
    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("Brief Summary"): st.info("Simplified view enabled.")
    if c2.button("Technical Specs"): st.success("Detailed view enabled.")
    if c3.button("Initialize AI Dialogue"):
        go("chat")
        st.rerun()

# ---------------------------------
# SCREEN 4 — CHAT
# ---------------------------------
elif st.session_state.screen == "chat":
    st.markdown('<div class="hero-title">AI Assistant</div>', unsafe_allow_html=True)
    
    user_q = st.chat_input("Ask a follow-up question...")
    
    if user_q:
        with st.container():
            st.markdown(f"""
            <div class="glass-card">
                <b>User:</b> {user_q}<br><br>
                <b style="color:#38bdf8;">Campus Brain:</b> This concept directly correlates with <b>{st.session_state.get('result_name', 'your topic')}</b>. 
                In a broader context, it ensures system stability and efficient resource allocation.
            </div>
            """, unsafe_allow_html=True)

    if st.button("Generate Recommendations"):
        go("recommend")
        st.rerun()

# ---------------------------------
# SCREEN 5 — RECOMMENDATIONS
# ---------------------------------
elif st.session_state.screen == "recommend":
    st.markdown('<div class="hero-title">Growth Path</div>', unsafe_allow_html=True)
    
    st.write("Current Mastery Level:")
    st.progress(0.65)

    st.markdown("""
    <div class="glass-card">
    ✅ <b>Core Foundations</b> — Completed <br>
    ⚡ <b>Contextual Application</b> — In Progress <br>
    🔒 <b>Advanced Optimization</b> — Locked
    </div>
    """, unsafe_allow_html=True)

    if st.button("Return to Module Library"):
        go("library")
        st.rerun()

# ---------------------------------
# SCREEN 6 — LIBRARY
# ---------------------------------
elif st.session_state.screen == "library":
    st.markdown('<div class="hero-title">Academic Vault</div>', unsafe_allow_html=True)
    
    t1, t2, t3, t4 = st.tabs(["📂 Documents", "📹 Media", "🔬 Research", "⭐ Bookmarks"])

    with t1:
        for name in doc_names:
            st.markdown(f'<div class="glass-card" style="padding:15px; margin-bottom:10px;">📄 {name}</div>', unsafe_allow_html=True)

    with t2: st.info("No video sequences found in this repository.")
    with t3: st.info("Peer-reviewed journals will appear here.")
    with t4: st.info("Your pinned modules will be stored here.")

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True)
st.divider()
st.caption("Campus Brain AI Framework v2.0 — Powered by Neural Search")




import streamlit as st
import os
import torch
from sentence_transformers import SentenceTransformer, util

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Campus Brain AI",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------
# MINIMALIST PREMIUM CSS
# ---------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #8b5cf6;
    --bg-dark: #0b1120;
    --card-bg: rgba(15, 23, 42, 0.8);
    --text-main: #f8fafc;
}

* { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp {
    background: radial-gradient(circle at top, #1e293b, #0b1120 70%);
}

[data-testid="stHeader"], [data-testid="stFooter"] { visibility: hidden; }

/* Ultra-Clean Auth Card */
.auth-container {
    max-width: 420px;
    margin: 80px auto;
    padding: 50px 40px;
    background: var(--card-bg);
    border-radius: 28px;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(20px);
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
}

/* Subtle Glow Effect */
.auth-container::before {
    content: "";
    position: absolute;
    top: -2px; left: -2px; right: -2px; bottom: -2px;
    background: linear-gradient(45deg, #8b5cf6, transparent, #6366f1);
    z-index: -1;
    border-radius: 30px;
    opacity: 0.15;
}

/* Input Customization */
div[data-baseweb="input"] > div {
    background-color: rgba(2, 6, 23, 0.5) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: white !important;
}

/* Button Styling */
.stButton > button {
    border-radius: 12px !important;
    transition: 0.3s all ease !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
    border: none !important;
    padding: 0.6rem 1rem !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(139, 92, 246, 0.3);
}

.grad-text {
    background: linear-gradient(135deg, #ddd6fe, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

.db-card {
    background: rgba(30,41,59,0.5);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.05);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# INITIALIZE STATE
# ---------------------------------
for key, val in {"authenticated": False, "user_name": "", "screen": "welcome", "auth_mode": "login"}.items():
    if key not in st.session_state: st.session_state[key] = val

# ---------------------------------
# LOGIN / REGISTRATION SCREEN
# ---------------------------------
if not st.session_state.authenticated:
    
    # Elegant Centered Header
    st.markdown("<div style='text-align:center; margin-top:50px;'><h1 class='grad-text' style='font-size:3rem; margin-bottom:0;'>Campus Brain</h1><p style='color:#94a3b8; font-size:1.1rem;'>The future of student intelligence.</p></div>", unsafe_allow_html=True)
    
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    
    if st.session_state.auth_mode == "login":
        st.markdown("<h3 style='margin-bottom:25px;'>Sign In</h3>", unsafe_allow_html=True)
        email = st.text_input("Email", placeholder="name@university.edu")
        password = st.text_input("Password", type="password")
        
        if st.button("Access Dashboard", use_container_width=True, type="primary"):
            if email and password:
                st.session_state.user_name = email.split("@")[0].capitalize()
                st.session_state.authenticated = True
                st.rerun()
        
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.button("Create new account", on_click=lambda: setattr(st.session_state, 'auth_mode', 'register'), use_container_width=True)
    
    else:
        st.markdown("<h3 style='margin-bottom:25px;'>Join Campus Brain</h3>", unsafe_allow_html=True)
        name = st.text_input("Full Name")
        email = st.text_input("Student Email")
        password = st.text_input("Password", type="password")
        
        if st.button("Verify & Create", use_container_width=True, type="primary"):
            if name and email and password:
                st.session_state.user_name = name
                st.session_state.authenticated = True
                st.rerun()
                
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.button("Back to login", on_click=lambda: setattr(st.session_state, 'auth_mode', 'login'), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------
# MAIN DASHBOARD
# ---------------------------------
else:
    # Model & Data Logic (Kept from your context)
    @st.cache_resource
    def load_model(): return SentenceTransformer("all-MiniLM-L6-v2")
    model = load_model()

    def load_documents(folder="knowledge"):
        if not os.path.exists(folder):
            os.makedirs(folder)
            data = {"dbms.txt": "DBMS organizes data...", "os.txt": "Operating systems manage...", "sorting.txt": "Sorting algorithms..."}
            for k,v in data.items():
                with open(os.path.join(folder,k), "w") as f: f.write(v)
        docs, names = [], []
        for file in os.listdir(folder):
            if file.endswith(".txt"):
                with open(os.path.join(folder,file),"r", encoding="utf-8") as f:
                    docs.append(f.read()); names.append(file.replace(".txt","").title())
        return docs, names

    documents, doc_names = load_documents()
    doc_embeddings = model.encode(documents, convert_to_tensor=True) if documents else None

    # Sleek Sidebar
    with st.sidebar:
        st.markdown("<h2 class='grad-text'>Campus Brain</h2>", unsafe_allow_html=True)
        st.caption(f"Welcome back, **{st.session_state.user_name}**")
        st.markdown("---")
        
        menus = {"🏠 Dashboard": "welcome", "🔍 Search": "search", "📚 Library": "library", "💬 AI Chat": "chat"}
        for lbl, scr in menus.items():
            if st.button(lbl, use_container_width=True, type="primary" if st.session_state.screen == scr else "secondary"):
                st.session_state.screen = scr
                st.rerun()
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

    # Dashboard Views
    if st.session_state.screen == "welcome":
        st.title(f"Hello, {st.session_state.user_name} 👋")
        
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown('<div class="db-card"><b>Topics Mastered</b><h2>14</h2></div>', unsafe_allow_html=True)
        with c2: st.markdown('<div class="db-card"><b>Study Hours</b><h2>42.5</h2></div>', unsafe_allow_html=True)
        with c3: st.markdown('<div class="db-card"><b>Learning Rank</b><h2>#8</h2></div>', unsafe_allow_html=True)

        st.markdown("### Progress Overview")
        st.progress(0.7)
        st.caption("You are ahead of 85% of your classmates.")

    elif st.session_state.screen == "search":
        st.markdown("## 🔍 Neural Topic Search")
        query = st.text_input("", placeholder="Explain database normalization...")
        
        if query and doc_embeddings is not None:
            query_embedding = model.encode(query, convert_to_tensor=True)
            sims = util.cos_sim(query_embedding, doc_embeddings)[0]
            best_idx = torch.argmax(sims).item()
            
            st.markdown(f"### 📄 Source: {doc_names[best_idx]}")
            st.markdown(f"<div class='db-card'>{documents[best_idx]}</div>", unsafe_allow_html=True)

    elif st.session_state.screen == "chat":
        st.markdown("## 💬 AI Study Assistant")
        u_input = st.chat_input("Ask about your syllabus...")
        if u_input:
            with st.chat_message("user"): st.write(u_input)
            with st.chat_message("assistant"): st.write("That's an insightful question. In the context of your current module, this concept handles resource optimization by...")


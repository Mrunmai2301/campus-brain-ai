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
# BORDERLESS PREMIUM CSS
# ---------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #8b5cf6;
    --bg-dark: #0b1120;
    --text-main: #f8fafc;
}

* { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp {
    background: radial-gradient(circle at top, #1e293b, #0b1120 80%);
}

/* Hide Streamlit elements */
[data-testid="stHeader"], [data-testid="stFooter"] { visibility: hidden; }

/* Centering and spacing for the floating inputs */
.floating-auth {
    max-width: 380px;
    margin: 40px auto;
}

/* Custom Input Styling to make them pop without a box */
div[data-baseweb="input"] > div {
    background-color: rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    padding: 2px 4px !important;
    transition: 0.3s all ease;
}

div[data-baseweb="input"] > div:focus-within {
    border-color: #8b5cf6 !important;
    background-color: rgba(255, 255, 255, 0.08) !important;
}

/* Button Styling */
.stButton > button {
    border-radius: 12px !important;
    height: 3rem;
    font-weight: 600 !important;
    transition: 0.3s all ease !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
    border: none !important;
    margin-top: 10px;
}

.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: #94a3b8 !important;
}

.grad-text {
    background: linear-gradient(135deg, #ffffff, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

.db-card {
    background: rgba(255,255,255,0.03);
    padding: 24px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.05);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# INITIALIZE STATE
# ---------------------------------
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if "user_name" not in st.session_state: st.session_state.user_name = ""
if "screen" not in st.session_state: st.session_state.screen = "welcome"
if "auth_mode" not in st.session_state: st.session_state.auth_mode = "login"

# ---------------------------------
# FLOATING LOGIN / REGISTRATION
# ---------------------------------
if not st.session_state.authenticated:
    
    # Title Section
    st.markdown("<div style='text-align:center; margin-top:100px;'><h1 class='grad-text' style='font-size:3.5rem; margin-bottom:0;'>Campus Brain</h1><p style='color:#94a3b8; font-size:1.2rem; margin-bottom:40px;'>The future of student intelligence.</p></div>", unsafe_allow_html=True)
    
    # Grid for alignment
    _, col, _ = st.columns([1, 1, 1])
    
    with col:
        if st.session_state.auth_mode == "login":
            email = st.text_input("Institutional Email", placeholder="alex@university.edu", label_visibility="collapsed")
            password = st.text_input("Password", type="password", placeholder="••••••••", label_visibility="collapsed")
            
            if st.button("Enter Dashboard", use_container_width=True, type="primary"):
                if email:
                    st.session_state.user_name = email.split("@")[0].capitalize()
                    st.session_state.authenticated = True
                    st.rerun()
            
            st.button("Create Account", on_click=lambda: setattr(st.session_state, 'auth_mode', 'register'), use_container_width=True, type="secondary")
        
        else:
            name = st.text_input("Name", placeholder="Full Name", label_visibility="collapsed")
            email = st.text_input("Email", placeholder="University Email", label_visibility="collapsed")
            password = st.text_input("Password", type="password", placeholder="Password", label_visibility="collapsed")
            
            if st.button("Register Now", use_container_width=True, type="primary"):
                if name:
                    st.session_state.user_name = name
                    st.session_state.authenticated = True
                    st.rerun()
                    
            st.button("Back to Login", on_click=lambda: setattr(st.session_state, 'auth_mode', 'login'), use_container_width=True, type="secondary")

# ---------------------------------
# MAIN DASHBOARD (Same Logic)
# ---------------------------------
else:
    @st.cache_resource
    def load_model(): return SentenceTransformer("all-MiniLM-L6-v2")
    model = load_model()

    def load_documents(folder="knowledge"):
        if not os.path.exists(folder):
            os.makedirs(folder); data = {"dbms.txt": "DBMS organizes data...", "os.txt": "Operating systems manage..."}
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

    with st.sidebar:
        st.markdown("<h2 class='grad-text'>Campus Brain</h2>", unsafe_allow_html=True)
        st.caption(f"Active: **{st.session_state.user_name}**")
        st.markdown("---")
        menus = {"Home": "welcome", "Search": "search", "Chat": "chat"}
        for lbl, scr in menus.items():
            if st.button(lbl, use_container_width=True, type="primary" if st.session_state.screen == scr else "secondary"):
                st.session_state.screen = scr; st.rerun()
        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False; st.rerun()

    if st.session_state.screen == "welcome":
        st.title(f"Hello, {st.session_state.user_name} 👋")
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="db-card"><b>Efficiency</b><br><h2>94%</h2></div>', unsafe_allow_html=True)
        c2.markdown('<div class="db-card"><b>Resources</b><br><h2>1.2k</h2></div>', unsafe_allow_html=True)
        c3.markdown('<div class="db-card"><b>Status</b><br><h2>Pro</h2></div>', unsafe_allow_html=True)

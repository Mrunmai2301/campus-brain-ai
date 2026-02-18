import streamlit as st
import os
from sentence_transformers import SentenceTransformer, util
import torch

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Campus Brain AI",
    page_icon="🎓",
    layout="centered"
)

# ---------------------------------
# PREMIUM UI CSS
# ---------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #8b5cf6;
    --bg-dark: #0b1120;
    --card-bg: rgba(30, 41, 59, 0.65);
    --text-main: #f8fafc;
}

* { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp {
    background: radial-gradient(circle at top, #111827, #0b1120 60%);
    color: var(--text-main);
}

header, footer { visibility: hidden; }

.auth-container {
    max-width: 480px;
    margin: 40px auto;
    padding: 45px 35px;
    background: var(--card-bg);
    border-radius: 24px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(18px);
}

div[data-baseweb="input"] > div {
    background: #1e293b !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(90deg,#8b5cf6,#6366f1);
    border: none;
    border-radius: 14px;
    padding: 12px;
    font-weight: 600;
}

.stButton > button {
    border-radius: 14px;
}

.grad-text {
    background: linear-gradient(90deg,#a78bfa,#c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

.db-card {
    background: rgba(30,41,59,0.7);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# SESSION STATE INIT
# ---------------------------------
defaults = {
    "authenticated": False,
    "user_name": "",
    "screen": "welcome",
    "auth_mode": "login",
    "open_doc": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------------------------
# AUTH FUNCTIONS
# ---------------------------------
def handle_login():
    st.session_state.authenticated = True
    st.rerun()

def toggle_auth():
    st.session_state.auth_mode = (
        "register" if st.session_state.auth_mode == "login" else "login"
    )

# ---------------------------------
# AUTH SCREEN
# ---------------------------------
if not st.session_state.authenticated:

    # HERO BOX
    st.markdown("""
    <div style="
        max-width:750px;
        margin:60px auto 40px auto;
        padding:50px 30px;
        text-align:center;
        border-radius:28px;
        background:rgba(30,41,59,0.6);
        backdrop-filter:blur(20px);
        border:1px solid rgba(255,255,255,0.08);
        box-shadow:0 0 60px rgba(139,92,246,0.25);
    ">
        <h1 style="
            font-size:64px;
            font-weight:800;
            margin-bottom:15px;
            background:linear-gradient(90deg,#a78bfa,#c084fc);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
        ">
            🎓 Campus Brain
        </h1>

        <p style="
            color:#94a3b8;
            font-size:20px;
            font-weight:500;
        ">
            Your AI-Powered Academic Companion
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.container():

        st.markdown('<div class="auth-container">', unsafe_allow_html=True)

        title = "Create Account" if st.session_state.auth_mode == "register" else "Welcome Back"
        st.markdown(f"<h2 class='grad-text' style='text-align:center;'>{title}</h2>", unsafe_allow_html=True)

        if st.session_state.auth_mode == "register":

            name = st.text_input("Full Name")
            email = st.text_input("University Email")
            password = st.text_input("Password", type="password")
            confirm_pw = st.text_input("Confirm Password", type="password")

            college = st.selectbox(
                "Institution",
                ["Engineering College", "Tech Institute", "Science University"]
            )

            if st.button("Complete Registration", use_container_width=True, type="primary"):
                if name and email and password:
                    st.session_state.user_name = name
                    handle_login()
                else:
                    st.error("Please fill all fields")

            st.markdown("<p style='text-align:center;color:#94a3b8;'>Already have an account?</p>", unsafe_allow_html=True)
            st.button("Back to Login", on_click=toggle_auth, use_container_width=True)

        else:

            email = st.text_input("Email")
            password = st.text_input("Password", type="password")

            if st.button("Sign In to Campus Brain", use_container_width=True, type="primary"):
                if email and password:
                    st.session_state.user_name = email.split("@")[0].capitalize()
                    handle_login()

            st.markdown("<p style='text-align:center;color:#94a3b8;'>New to the platform?</p>", unsafe_allow_html=True)
            st.button("Create Student Profile", on_click=toggle_auth, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------
# MAIN DASHBOARD
# ---------------------------------
else:

    @st.cache_resource
    def load_model():
        return SentenceTransformer("all-MiniLM-L6-v2")

    model = load_model()

    def load_documents(folder="knowledge"):
        if not os.path.exists(folder):
            os.makedirs(folder)
            sample_data = {
                "dbms.txt": "DBMS organizes data efficiently using tables and schemas.",
                "os.txt": "Operating systems manage memory, processes, and hardware.",
                "sorting.txt": "Sorting algorithms arrange data in ascending or descending order."
            }
            for file, content in sample_data.items():
                with open(os.path.join(folder, file), "w") as f:
                    f.write(content)

        docs, names = [], []
        for file in os.listdir(folder):
            if file.endswith(".txt"):
                with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                    docs.append(f.read())
                    names.append(file.replace(".txt", "").title())
        return docs, names

    documents, doc_names = load_documents()
    doc_embeddings = model.encode(documents, convert_to_tensor=True) if documents else None

    # SIDEBAR
    with st.sidebar:
        st.markdown("## 🎓 Campus Brain")
        st.write(f"Logged in as: **{st.session_state.user_name}**")
        st.markdown("---")

        menus = {
            "🏠 Dashboard": "welcome",
            "🔍 Search": "search",
            "📚 Library": "library",
            "📈 Progress": "recommend",
            "💬 AI Chat": "chat"
        }

        for label, screen in menus.items():
            if st.button(label, use_container_width=True):
                st.session_state.screen = screen
                st.rerun()

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

    # SCREENS
    if st.session_state.screen == "welcome":

        st.title(f"Hello, {st.session_state.user_name} 👋")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="db-card"><b>Topics Mastered</b><h2>14</h2></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="db-card"><b>Study Hours</b><h2>42.5</h2></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="db-card"><b>Learning Rank</b><h2>#8</h2></div>', unsafe_allow_html=True)

        st.progress(0.7)
        st.caption("70% of Semester Completed")

    elif st.session_state.screen == "search":

        st.markdown("## 🔍 Smart Academic Search")
        query = st.text_input("Ask anything about your syllabus...")

        if query and doc_embeddings is not None:
            query_embedding = model.encode(query, convert_to_tensor=True)
            sims = util.cos_sim(query_embedding, doc_embeddings)[0]
            best_idx = torch.argmax(sims).item()

            st.session_state.result_doc = documents[best_idx]
            st.session_state.result_name = doc_names[best_idx]
            st.session_state.screen = "results"
            st.rerun()

    elif st.session_state.screen == "results":

        if st.button("← Back"):
            st.session_state.screen = "search"
            st.rerun()

        st.markdown(f"<div class='db-card'><h3>{st.session_state.result_name}</h3><p>{st.session_state.result_doc}</p></div>", unsafe_allow_html=True)

    elif st.session_state.screen == "chat":

        st.markdown("## 💬 AI Study Assistant")
        user_input = st.chat_input("Ask a question...")

        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                st.write("That is a great question regarding your study material!")

    elif st.session_state.screen == "library":

        st.markdown("## 📚 Study Library")
        cols = st.columns(3)
        for i, name in enumerate(doc_names):
            with cols[i % 3]:
                st.button(f"📄 {name}", use_container_width=True)

    elif st.session_state.screen == "recommend":
        st.markdown("## 📈 Performance Tracking")
        st.info("Analytics coming soon.")


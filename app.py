import streamlit as st
import os
import torch
from sentence_transformers import SentenceTransformer, util

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Campus Brain AI",
    page_icon="🎓",
    layout="wide"
)

# ----------------------------
# PREMIUM CSS
# ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
* { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp {
    background: radial-gradient(circle at top, #1e293b, #0b1120 80%);
}

[data-testid="stHeader"], [data-testid="stFooter"] {
    visibility: hidden;
}

.grad-text {
    background: linear-gradient(135deg, #ffffff, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

.db-card {
    background: rgba(255,255,255,0.04);
    padding: 24px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# SESSION STATE DEFAULTS
# ----------------------------
defaults = {
    "authenticated": False,
    "user_name": "",
    "screen": "welcome",
    "auth_mode": "login",
    "result_doc": None,
    "result_name": None,
    "chat_history": []
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
# ----------------------------
# USER DATABASE FUNCTIONS
# ----------------------------

USER_FILE = "users.txt"

def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    
    users = {}
    with open(USER_FILE, "r") as f:
        for line in f:
            name, email, password = line.strip().split("|")
            users[email] = {"name": name, "password": password}
    return users

def save_user(name, email, password):
    with open(USER_FILE, "a") as f:
        f.write(f"{name}|{email}|{password}\n")
# ----------------------------
# AUTH SCREEN
# ----------------------------
if not st.session_state.authenticated:

    users = load_users()

    st.markdown("""
    <div style='text-align:center; margin-top:120px;'>
        <h1 class='grad-text' style='font-size:3.5rem;'>Campus Brain</h1>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1,1,1])

    with col:

        if st.session_state.auth_mode == "login":

            st.subheader("Login")

            email = st.text_input("Email")
            password = st.text_input("Password", type="password")

            if st.button("Login", use_container_width=True):

                if email in users and users[email]["password"] == password:
                    st.session_state.user_name = users[email]["name"]
                    st.session_state.authenticated = True
                    st.success("Login Successful!")
                    st.rerun()
                else:
                    st.error("Invalid Email or Password")

            st.button(
                "Create Account",
                on_click=lambda: setattr(st.session_state, "auth_mode", "register"),
                use_container_width=True
            )

        else:

            st.subheader("Create Account")

            name = st.text_input("Full Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")

            if st.button("Register", use_container_width=True):

                if email in users:
                    st.error("Email already exists")
                elif not name or not email or not password:
                    st.warning("Please fill all fields")
                else:
                    save_user(name, email, password)
                    st.success("Account Created! Please Login")
                    st.session_state.auth_mode = "login"
                    st.rerun()

            st.button(
                "Back to Login",
                on_click=lambda: setattr(st.session_state, "auth_mode", "login"),
                use_container_width=True
            )

# ----------------------------
# MAIN DASHBOARD
# ----------------------------
else:

    # -------- Load Model --------
    @st.cache_resource
    def load_model():
        return SentenceTransformer("all-MiniLM-L6-v2")

    model = load_model()

    # -------- Load Documents --------
    def load_documents(folder="knowledge"):

        if not os.path.exists(folder):
            os.makedirs(folder)

            sample_data = {
                "dbms.txt": "DBMS organizes data efficiently using tables and schemas.",
                "os.txt": "Operating systems manage memory, processes, and hardware.",
                "sorting.txt": "Sorting algorithms arrange data in ascending or descending order like Bubble Sort and Quick Sort."
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
    doc_embeddings = model.encode(documents, convert_to_tensor=True)

    # -------- Sidebar --------
    with st.sidebar:

        st.markdown("<h2 class='grad-text'>Campus Brain</h2>", unsafe_allow_html=True)
        st.caption(f"Active: **{st.session_state.user_name}**")
        st.markdown("---")

        menus = {
            "🏠 Home": "welcome",
            "🔍 Search": "search",
            "📚 Library": "library",
            "💬 AI Chat": "chat"
        }

        for label, screen in menus.items():
            if st.button(label, use_container_width=True):
                st.session_state.screen = screen
                st.rerun()

        st.markdown("---")

        if st.button("Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ----------------------------
    # SCREENS (PROPERLY INDENTED)
    # ----------------------------

    if st.session_state.screen == "welcome":

        st.markdown(f"""
            <h1 style='margin-top:30px;'>
                Hello, {st.session_state.user_name} 👋
            </h1>
            """, unsafe_allow_html=True)

        st.markdown("### What would you like to do?")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔍 Search", use_container_width=True):
                st.session_state.screen = "search"
                st.rerun()

        with col2:
            if st.button("📚 Library", use_container_width=True):
                st.session_state.screen = "library"
                st.rerun()

        with col3:
            if st.button("💬 AI Chat", use_container_width=True):
                st.session_state.screen = "chat"
                st.rerun()

    elif st.session_state.screen == "search":

        st.markdown("## 🔍 Smart Academic Search")

        query = st.text_input("Ask anything from your syllabus...")

        if query:
            query_embedding = model.encode(query, convert_to_tensor=True)
            sims = util.cos_sim(query_embedding, doc_embeddings)[0]
            best_idx = torch.argmax(sims).item()

            st.session_state.result_doc = documents[best_idx]
            st.session_state.result_name = doc_names[best_idx]
            st.session_state.screen = "results"
            st.rerun()

    elif st.session_state.screen == "results":

        if st.button("← Back to Search"):
            st.session_state.screen = "search"
            st.rerun()

        st.markdown(
            f"<div class='db-card'><h3>{st.session_state.result_name}</h3><p>{st.session_state.result_doc}</p></div>",
            unsafe_allow_html=True
        )

    elif st.session_state.screen == "library":

        st.markdown("## 📚 Study Library")

        if st.button("← Back to Home"):
            st.session_state.screen = "welcome"
            st.rerun()

        for i, name in enumerate(doc_names):
            if st.button(f"📄 {name}", use_container_width=True):
                st.markdown(
                    f"<div class='db-card'><h3>{name}</h3><p>{documents[i]}</p></div>",
                    unsafe_allow_html=True
                )

    elif st.session_state.screen == "chat":

        st.markdown("## 💬 AI Study Assistant")

        if st.button("← Back to Home"):
            st.session_state.screen = "welcome"
            st.rerun()

        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(message)

        user_input = st.chat_input("Ask something about your syllabus...")

        if user_input:

            st.session_state.chat_history.append(("user", user_input))

            query_embedding = model.encode(user_input, convert_to_tensor=True)
            sims = util.cos_sim(query_embedding, doc_embeddings)[0]
            best_idx = torch.argmax(sims).item()

            topic_name = doc_names[best_idx]
            context = documents[best_idx]

            answer = f"""
### 📘 Topic: {topic_name}

{context}

### 📝 Simple Explanation:
This topic explains how {topic_name.lower()} works in computer science.

### 💡 Example:
Understanding {topic_name} helps you apply it in real-world systems.
"""

            st.session_state.chat_history.append(("assistant", answer))

            with st.chat_message("assistant"):
                st.markdown(answer)





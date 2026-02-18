from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
# PREMIUM FLOATING CSS
# ---------------------------------
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

# ---------------------------------
# SESSION STATE INIT
# ---------------------------------
defaults = {
    "authenticated": False,
    "user_name": "",
    "screen": "welcome",
    "auth_mode": "login",
    "open_doc": None,
    "result_doc": None,
    "result_name": None,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------
# AUTH SCREEN
# ---------------------------------
if not st.session_state.authenticated:

    st.markdown("""
    <div style='text-align:center; margin-top:120px;'>
        <h1 class='grad-text' style='font-size:3.5rem;'>Campus Brain</h1>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1,1,1])

    with col:

        if st.session_state.auth_mode == "login":

            email = st.text_input("Email", placeholder="alex@college.edu")
            password = st.text_input("Password", type="password")

            if st.button("Enter Dashboard", use_container_width=True):
                if email and password:
                    st.session_state.user_name = email.split("@")[0].capitalize()
                    st.session_state.authenticated = True
                    st.rerun()

            st.button("Create Account",
                      on_click=lambda: setattr(st.session_state,"auth_mode","register"),
                      use_container_width=True)

        else:

            name = st.text_input("Full Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")

            if st.button("Register Now", use_container_width=True):
                if name and email and password:
                    st.session_state.user_name = name
                    st.session_state.authenticated = True
                    st.rerun()

            st.button("Back to Login",
                      on_click=lambda: setattr(st.session_state,"auth_mode","login"),
                      use_container_width=True)

# ---------------------------------
# MAIN DASHBOARD
# ---------------------------------
else:

    # -------- LOAD MODEL --------
    @st.cache_resource
    def load_model():
        return SentenceTransformer("all-MiniLM-L6-v2")

    model = load_model()

    # -------- LOAD DOCUMENTS --------
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
                    names.append(file.replace(".txt","").title())

        return docs, names

    documents, doc_names = load_documents()
    doc_embeddings = model.encode(documents, convert_to_tensor=True)

    # -------- SIDEBAR --------
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
            st.session_state.authenticated = False
            st.rerun()

    # -------- SCREENS --------

    # HOME
    if st.session_state.screen == "welcome":

        st.markdown(f"""
        <h1 style='margin-top:30px;'>
            Hello, {st.session_state.user_name} 👋
        </h1>
        """, unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)

        c1.markdown('<div class="db-card"><b>Topics</b><h2>14</h2></div>', unsafe_allow_html=True)
        c2.markdown('<div class="db-card"><b>Hours</b><h2>42</h2></div>', unsafe_allow_html=True)
        c3.markdown('<div class="db-card"><b>Status</b><h2>Pro</h2></div>', unsafe_allow_html=True)

    # SEARCH
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

    # RESULTS
    elif st.session_state.screen == "results":

        if st.button("← Back"):
            st.session_state.screen = "search"
            st.rerun()

        st.markdown(
            f"<div class='db-card'><h3>{st.session_state.result_name}</h3><p>{st.session_state.result_doc}</p></div>",
            unsafe_allow_html=True
        )

        # LIBRARY
    elif st.session_state.screen == "library":

        st.markdown("## 📚 Study Library")

        cols = st.columns(3)

        for i, name in enumerate(doc_names):
            with cols[i % 3]:
                if st.button(f"📄 {name}", use_container_width=True):
                    st.markdown(
                        f"<div class='db-card'><h3>{name}</h3><p>{documents[i]}</p></div>",
                        unsafe_allow_html=True
                    )

    # CHAT
    elif st.session_state.screen == "chat":

        st.markdown("## 💬 AI Study Assistant")

        user_input = st.chat_input("Ask something about your syllabus...")

        if user_input:

            with st.chat_message("user"):
                st.write(user_input)

            # Retrieve most relevant document
            query_embedding = model.encode(user_input, convert_to_tensor=True)
            sims = util.cos_sim(query_embedding, doc_embeddings)[0]
            best_idx = torch.argmax(sims).item()
            context = documents[best_idx]

            # Send to OpenAI
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI study assistant that explains academic topics clearly in simple student-friendly language."
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Context from syllabus:
                        {context}

                        Student Question:
                        {user_input}

                        Explain clearly with examples and simple language.
                        """
                    }
                ]
            )

            ai_answer = response.choices[0].message.content

            with st.chat_message("assistant"):
                st.write(ai_answer)

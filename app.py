import streamlit as st
import os
from sentence_transformers import SentenceTransformer, util
import torch

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Campus Brain AI",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------
# CUSTOM CSS — PREMIUM LOOK
# ---------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#0f172a,#020617);
    color: white;
}

.card {
    background: #111827;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}

.topic-card {
    background: #1f2937;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-weight: bold;
}

h1, h2, h3 {
    color: #e5e7eb;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------------------
# LOAD DOCUMENTS
# ---------------------------
def load_documents(folder="knowledge"):
    docs = []
    names = []

    if not os.path.exists(folder):
        return docs, names

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                docs.append(f.read())
                names.append(file)

    return docs, names

documents, doc_names = load_documents()

if documents:
    doc_embeddings = model.encode(documents, convert_to_tensor=True)

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("🎓 Campus Brain")
page = st.sidebar.radio(
    "Navigate",
    ["🔍 Search", "📚 Topics", "ℹ About"]
)

# =====================================================
# SEARCH PAGE
# =====================================================
if page == "🔍 Search":

    st.title("🔍 AI Academic Search")

    query = st.text_input("Ask anything academic…")

    if query and documents:

        with st.spinner("Thinking..."):

            query_embedding = model.encode(query, convert_to_tensor=True)
            similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
            best_idx = torch.argmax(similarities).item()

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader(f"📘 Best Match — {doc_names[best_idx]}")
        st.write(documents[best_idx][:900] + "...")

        st.markdown('</div>', unsafe_allow_html=True)

        # Recommendation logic
        st.subheader("💡 Recommended Next Step")

        name = doc_names[best_idx].lower()

        if "dbms" in name:
            st.success("👉 Learn SQL & database indexing")
        elif "sorting" in name:
            st.success("👉 Explore recursion & search algorithms")
        elif "os" in name:
            st.success("👉 Study CPU scheduling")
        else:
            st.success("👉 Continue related fundamentals")

# =====================================================
# TOPICS PAGE
# =====================================================
elif page == "📚 Topics":

    st.title("📚 Knowledge Library")

    cols = st.columns(3)

    for i, name in enumerate(doc_names):
        with cols[i % 3]:
            st.markdown(
                f'<div class="topic-card">📘 {name}</div>',
                unsafe_allow_html=True
            )

# =====================================================
# ABOUT PAGE
# =====================================================
else:

    st.title("ℹ About Campus Brain")

    st.markdown("""
<div class="card">

Campus Brain AI is a semantic academic search system built for students.

✨ Features:

• AI semantic retrieval  
• Knowledge navigation  
• Smart topic recommendation  
• Fast learning discovery  

Built as a hackathon prototype.

</div>
""", unsafe_allow_html=True)

# ---------------------------
st.divider()
st.caption("🚀 Campus Brain AI — Hackathon Edition")



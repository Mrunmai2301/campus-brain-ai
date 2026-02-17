import streamlit as st
import os
from sentence_transformers import SentenceTransformer, util
import torch

# --------------------------------
# Page Config — Premium Layout
# --------------------------------
st.set_page_config(
    page_title="Campus Brain AI",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------
# Custom Styling (Premium Look)
# --------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: bold;
    color: #1f4ed8;
}
.subtitle {
    font-size: 18px;
    color: gray;
}
.card {
    padding: 20px;
    border-radius: 12px;
    background-color: #f7f9fc;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# Sidebar Navigation
# --------------------------------
st.sidebar.title("🎓 Campus Brain")
st.sidebar.markdown("AI Academic Navigator")

page = st.sidebar.radio(
    "Navigate",
    ["🔍 Search", "📚 Topics", "ℹ About"]
)

# --------------------------------
# Load AI Model
# --------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --------------------------------
# Load Knowledge Files
# --------------------------------
def load_documents(folder="knowledge"):
    docs = []
    names = []

    if not os.path.exists(folder):
        return [], []

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                docs.append(f.read())
                names.append(file)

    return docs, names

documents, doc_names = load_documents()

if documents:
    doc_embeddings = model.encode(documents, convert_to_tensor=True)

# =================================
# 🔍 SEARCH PAGE
# =================================
if page == "🔍 Search":

    st.markdown('<div class="main-title">🎓 Campus Brain AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Smart Academic Semantic Search Assistant</div>', unsafe_allow_html=True)
    st.divider()

    query = st.text_input("🔍 Ask your academic question:")

    if query and documents:

        with st.spinner("Analyzing knowledge…"):

            query_embedding = model.encode(query, convert_to_tensor=True)
            similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
            best_match_idx = torch.argmax(similarities).item()

        st.success("Best academic match found!")

        # Result Card
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("📘 Best Match Source")
        st.write(f"**{doc_names[best_match_idx]}**")

        preview = documents[best_match_idx][:800]

        with st.expander("📖 View Explanation"):
            st.write(preview + "...")

        st.markdown('</div>', unsafe_allow_html=True)

        # Recommendation Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💡 Recommended Next Learning")

        name = doc_names[best_match_idx].lower()

        if "dbms" in name:
            st.info("👉 Explore SQL basics next.")
        elif "sorting" in name:
            st.info("👉 Learn recursion and searching algorithms.")
        elif "os" in name:
            st.info("👉 Study process scheduling concepts.")
        else:
            st.info("👉 Continue exploring related fundamentals.")

        st.markdown('</div>', unsafe_allow_html=True)

# =================================
# 📚 TOPICS PAGE
# =================================
elif page == "📚 Topics":

    st.header("📚 Available Knowledge Topics")

    if doc_names:
        cols = st.columns(3)

        for i, name in enumerate(doc_names):
            with cols[i % 3]:
                st.markdown(
                    f"""
                    <div class="card">
                    📘 <b>{name}</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.warning("No knowledge files found.")

# =================================
# ℹ ABOUT PAGE
# =================================
else:

    st.header("ℹ About Campus Brain")

    st.write("""
Campus Brain is an AI-powered semantic academic navigator designed
to help students quickly understand complex topics.

### Features
✅ Semantic search  
✅ Smart recommendations  
✅ Academic knowledge assistant  

Built as a hackathon prototype demonstrating AI-assisted learning.
""")

st.divider()
st.caption("🚀 Hackathon Prototype — Campus Brain AI")


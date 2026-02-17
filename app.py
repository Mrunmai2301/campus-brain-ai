import streamlit as st
import os
from search import semantic_search
from sentence_transformers import SentenceTransformer, util
import torch

# ---------------------------
# Load model (cached)
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------------------
# Load knowledge files
# ---------------------------
def load_documents(folder="knowledge"):
    docs = []
    names = []

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                docs.append(f.read())
                names.append(file)

    return docs, names

documents, doc_names = load_documents()

# Precompute embeddings
doc_embeddings = model.encode(documents, convert_to_tensor=True)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎓 Campus Brain — AI Academic Navigator")

st.write("Ask any academic topic and get smart explanations.")

query = st.text_input("Enter your question:")

if query:

    query_embedding = model.encode(query, convert_to_tensor=True)

    similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
    best_match_idx = torch.argmax(similarities).item()

    st.subheader("📘 Best Match")
    st.write(f"**Source:** {doc_names[best_match_idx]}")

    # Show summarized preview
    preview = documents[best_match_idx][:800]
    st.write(preview + "...")

    # Simple recommendation logic
    st.subheader("💡 Recommended Next Topic")

    if "dbms" in doc_names[best_match_idx].lower():
        st.write("👉 Explore SQL basics next.")
    elif "sorting" in doc_names[best_match_idx].lower():
        st.write("👉 Learn recursion and searching algorithms.")
    elif "os" in doc_names[best_match_idx].lower():
        st.write("👉 Study process scheduling concepts.")
    else:
        st.write("👉 Continue exploring related fundamentals.")

st.divider()
st.caption("Hackathon prototype — Semantic academic search demo.")

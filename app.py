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
# SESSION STATE
# ----------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ----------------------------
# LOAD DOCUMENTS
# ----------------------------
def load_documents(folder="knowledge"):

    if not os.path.exists(folder):
        os.makedirs(folder)

        sample_data = {
            "dbms.txt": "DBMS is a system that stores and manages data in tables.",
            "os.txt": "Operating System manages memory, CPU scheduling, and processes.",
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

# ----------------------------
# HOME PAGE
# ----------------------------
if st.session_state.page == "home":

    st.title("🎓 Campus Brain - Basic AI Study Assistant")

    st.write("Welcome! Click below to start chatting with AI.")

    if st.button("💬 Go to AI Chat"):
        st.session_state.page = "chat"
        st.rerun()

# ----------------------------
# CHAT PAGE
# ----------------------------
elif st.session_state.page == "chat":

    # Back Button
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("## 💬 AI Study Assistant")

    user_input = st.text_input("Ask something about your syllabus...")

    if user_input:

        query_embedding = model.encode(user_input, convert_to_tensor=True)
        sims = util.cos_sim(query_embedding, doc_embeddings)[0]
        best_idx = torch.argmax(sims).item()

        context = documents[best_idx]
        topic_name = doc_names[best_idx]

        explanation = f"""
### 📘 Topic: {topic_name}

Let me explain this clearly:

{context}

### 📝 In Simple Words:
This topic is about understanding how {topic_name.lower()} works in computer science.

### 💡 Example:
If you study {topic_name}, you will understand its practical usage in real-world systems.
"""

        st.markdown(explanation)

    st.markdown(explanation)


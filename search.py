import os
import torch
from sentence_transformers import SentenceTransformer, util

# ---------------------------
# Load embedding model
# ---------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# Load knowledge documents
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

# Precompute embeddings
if documents:
    doc_embeddings = model.encode(documents, convert_to_tensor=True)
else:
    doc_embeddings = None


# ---------------------------
# Semantic search function
# ---------------------------
def semantic_search(query):

    if doc_embeddings is None:
        return "No knowledge files found.", "", ""

    query_embedding = model.encode(query, convert_to_tensor=True)

    similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
    best_match_idx = torch.argmax(similarities).item()

    best_doc = documents[best_match_idx]
    source = doc_names[best_match_idx]

    preview = best_doc[:800] + "..."

    recommendation = get_recommendation(source)

    return preview, source, recommendation


# ---------------------------
# Simple recommendation logic
# ---------------------------
def get_recommendation(source):

    name = source.lower()

    if "dbms" in name:
        return "Explore SQL basics next."
    elif "sorting" in name:
        return "Learn recursion and searching algorithms."
    elif "os" in name:
        return "Study process scheduling concepts."
    else:
        return "Continue exploring related fundamentals."

import streamlit as st
import os
from sentence_transformers import SentenceTransformer, util
import torch

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="Campus Brain AI",
    page_icon="🎓",
    layout="wide"
)

# --------------------------
# PREMIUM EDUCATIONAL CSS
# --------------------------
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main App Background */
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a, #020617);
        color: #f8fafc;
    }

    /* Glassmorphism Card Style */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 25px;
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(56, 189, 248, 0.5);
    }

    /* Topic Grid Items */
    .topic-item {
        background: linear-gradient(135deg, #334155 0%, #1e293b 100%);
        padding: 30px 15px;
        border-radius: 16px;
        text-align: center;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.05);
        cursor: pointer;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Custom Headers */
    .main-title {
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        margin-bottom: 10px;
    }

    /* Highlight badge */
    .badge {
        background: #0ea5e9;
        color: white;
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: bold;
        text-transform: uppercase;
    }

    /* Success indicator for recommendations */
    .learning-path {
        border-left: 4px solid #10b981;
        padding-left: 15px;
        margin: 10px 0;
    }

</style>
""", unsafe_allow_html=True)

# --------------------------
# LOAD MODEL & DOCS
# --------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

def load_documents(folder="knowledge"):
    # Ensure the folder exists for demo purposes
    if not os.path.exists(folder):
        os.makedirs(folder)
        # Placeholder for demo if folder is empty
        with open(f"{folder}/dbms_intro.txt", "w") as f: f.write("Database Management Systems allow structured data storage...")
        with open(f"{folder}/sorting_algo.txt", "w") as f: f.write("Sorting algorithms like Quicksort O(n log n) are fundamental...")

    docs, names = [], []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                docs.append(f.read())
                names.append(file.replace(".txt", "").replace("_", " ").title())
    return docs, names

documents, doc_names = load_documents()

if documents:
    doc_embeddings = model.encode(documents, convert_to_tensor=True)

# --------------------------
# NAVIGATION (SIDEBAR)
# --------------------------
with st.sidebar:
    st.markdown("<h2 style='color: #38bdf8;'>🧠 Academy OS</h2>", unsafe_allow_html=True)
    page = st.radio("Learning Hub", ["🔍 Intelligent Search", "📚 Knowledge Map", "📊 Study Stats"])
    st.divider()
    st.info("Level Up: You've completed 2 topics this week! 🏆")

# --------------------------
# SEARCH PAGE
# --------------------------
if page == "🔍 Intelligent Search":
    st.markdown('<h1 class="main-title">Semantic Search</h1>', unsafe_allow_html=True)
    st.write("Unlock knowledge across your entire academic library using AI.")
    
    query = st.text_input("", placeholder="Ex: How does CPU scheduling work?")

    if query and documents:
        with st.spinner("Analyzing neural patterns..."):
            query_embedding = model.encode(query, convert_to_tensor=True)
            similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
            best_idx = torch.argmax(similarities).item()

        # Result Display
        st.markdown(f'''
            <div class="glass-card">
                <span class="badge">Best Match: {doc_names[best_idx]}</span>
                <h3 style="margin-top:15px;">Key Insight</h3>
                <p style="color: #cbd5e1; line-height: 1.6;">{documents[best_idx][:1000]}</p>
            </div>
        ''', unsafe_allow_html=True)

        # Smart Recommendation
        st.markdown("### 💡 Recommended Learning Path")
        name = doc_names[best_idx].lower()
        
        recs = {
            "dbms": "Learn SQL normalization & B+ Tree indexing",
            "sorting": "Explore Big O notation & Binary Search",
            "os": "Study Deadlocks & Virtual Memory Management"
        }
        
        found_rec = next((v for k, v in recs.items() if k in name), "Continue exploring related academic papers")
        
        st.markdown(f'''
            <div class="learning-path">
                <p style="color:#10b981; font-weight:600;">Next Step: {found_rec}</p>
            </div>
        ''', unsafe_allow_html=True)

# --------------------------
# KNOWLEDGE MAP
# --------------------------
elif page == "📚 Knowledge Map":
    st.markdown('<h1 class="main-title">Knowledge Map</h1>', unsafe_allow_html=True)
    st.write("Explore your digital brain's connections.")
    
    cols = st.columns(3)
    for i, name in enumerate(doc_names):
        with cols[i % 3]:
            st.markdown(f'''
                <div class="glass-card topic-item">
                    <div style="font-size: 2rem;">📘</div>
                    <div style="margin-top:10px;">{name}</div>
                </div>
            ''', unsafe_allow_html=True)

# --------------------------
# STATS (PLACEHOLDER)
# --------------------------
else:
    st.markdown('<h1 class="main-title">Study Insights</h1>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Recall Accuracy", "92%", "+4%")
    c2.metric("Library Size", f"{len(doc_names)} Files", "Active")
    c3.metric("Study Streak", "5 Days", "🔥")
    
    st.markdown('<div class="glass-card"><h3>Activity Log</h3><p>Your AI has indexed 5 new academic papers in the last 24 hours.</p></div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><hr><center style='opacity:0.5;'>Campus Brain AI Engine v2.0 | Secured Academic Environment</center>", unsafe_allow_html=True)



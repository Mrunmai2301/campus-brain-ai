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
    layout="wide"
)

# ---------------------------------
# PREMIUM UI CSS
# ---------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at top right, #1e293b, #0f172a, #020617);
    color: white;
}

.card {
    background: rgba(30,41,59,0.75);
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}

.big-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(to right, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.badge {
    background: #0ea5e9;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# SESSION NAVIGATION
# ---------------------------------
if "screen" not in st.session_state:
    st.session_state.screen = "welcome"

def go(screen):
    st.session_state.screen = screen

# ---------------------------------
# LOAD MODEL
# ---------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------------------------
# LOAD KNOWLEDGE FILES
# ---------------------------------
def load_documents(folder="knowledge"):
    if not os.path.exists(folder):
        os.makedirs(folder)

        with open(f"{folder}/dbms.txt","w") as f:
            f.write("DBMS organizes data efficiently using tables, keys, and normalization.")

        with open(f"{folder}/os.txt","w") as f:
            f.write("Operating systems manage processes, scheduling, and memory.")

        with open(f"{folder}/sorting.txt","w") as f:
            f.write("Sorting algorithms arrange data efficiently like quicksort and mergesort.")

    docs, names = [], []

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder,file),"r",encoding="utf-8") as f:
                docs.append(f.read())
                names.append(file.replace(".txt","").title())

    return docs, names

documents, doc_names = load_documents()

if documents:
    doc_embeddings = model.encode(documents, convert_to_tensor=True)

# ---------------------------------
# SCREEN 1 — LOGIN / WELCOME
# ---------------------------------
if st.session_state.screen == "welcome":

    st.markdown('<div class="big-title">Campus Brain AI</div>', unsafe_allow_html=True)
    st.write("### AI That Understands How Students Learn")

    email = st.text_input("College Email")

    col1, col2, col3 = st.columns(3)
    col1.selectbox("College", ["Engineering College", "Science College"])
    col2.selectbox("Department", ["Computer", "IT", "AI"])
    col3.selectbox("Semester", ["Sem 1", "Sem 2", "Sem 3"])

    if st.button("Enter Learning Hub"):
        go("search")
        st.rerun()

# ---------------------------------
# SCREEN 2 — SEARCH
# ---------------------------------
elif st.session_state.screen == "search":

    st.markdown('<div class="big-title">Smart Academic Search</div>', unsafe_allow_html=True)

    query = st.text_input(
        "",
        placeholder="Ask anything… CPU scheduling? DBMS keys? Sorting complexity?"
    )

    st.write("Quick Topics:")
    cols = st.columns(4)
    chips = ["Notes", "Videos", "Research", "Presentations"]

    for i,c in enumerate(chips):
        cols[i].button(c)

    if query:

        query_embedding = model.encode(query, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
        best = torch.argmax(similarities).item()

        st.session_state.result_doc = documents[best]
        st.session_state.result_name = doc_names[best]

        go("results")
        st.rerun()

# ---------------------------------
# SCREEN 3 — RESULTS
# ---------------------------------
elif st.session_state.screen == "results":

    st.markdown(f"""
    <div class="card">
        <span class="badge">{st.session_state.result_name}</span>
        <h3>Auto Summary</h3>
        <p>{st.session_state.result_doc[:700]}...</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("### Key Points")

    for line in st.session_state.result_doc.split(".")[:5]:
        st.write("•", line)

    c1,c2 = st.columns(2)

    if c1.button("Explain Simply"):
        st.info("Simplified explanation helps beginners understand fundamentals.")

    if c2.button("Deep Dive"):
        st.success("Advanced explanation explores inner system mechanics.")

    if st.button("Ask AI About This"):
        go("chat")
        st.rerun()

# ---------------------------------
# SCREEN 4 — CHAT
# ---------------------------------
elif st.session_state.screen == "chat":

    st.markdown('<div class="big-title">AI Study Chat</div>', unsafe_allow_html=True)

    user_q = st.text_input("Ask follow-up question")

    if user_q:
        st.markdown(f"""
        <div class="card">
        <b>You:</b> {user_q}<br><br>
        <b>AI:</b> This topic connects to your selected material and expands your understanding.
        </div>
        """, unsafe_allow_html=True)

    if st.button("See Recommendations"):
        go("recommend")
        st.rerun()

# ---------------------------------
# SCREEN 5 — RECOMMENDATIONS
# ---------------------------------
elif st.session_state.screen == "recommend":

    st.markdown('<div class="big-title">Recommended For You</div>', unsafe_allow_html=True)

    st.write("Learning Path:")

    st.progress(0.6)

    st.markdown("""
    ✔ Basics Complete  
    ➜ Intermediate Concepts  
    ⬜ Advanced Topics
    """)

    if st.button("Open Library"):
        go("library")
        st.rerun()

# ---------------------------------
# SCREEN 6 — LIBRARY
# ---------------------------------
elif st.session_state.screen == "library":

    tab1,tab2,tab3,tab4 = st.tabs(["Notes","Videos","Research","Saved"])

    with tab1:
        for name in doc_names:
            st.markdown(f"📘 {name}")

    with tab2:
        st.write("Video resources coming soon.")

    with tab3:
        st.write("Research papers index.")

    with tab4:
        st.write("Saved materials.")

# ---------------------------------
# FOOTER
# ---------------------------------
st.divider()
st.caption("Campus Brain AI — Hackathon Premium Prototype")




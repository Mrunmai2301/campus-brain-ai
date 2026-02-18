# 🎓 Campus Brain AI

Campus Brain AI is an AI-powered academic assistant built using Streamlit and Sentence Transformers.  
It helps students search syllabus topics, explore study material, and get simple AI explanations.

---

## Features

- User Authentication (Login & Register)
- Smart Semantic Search
- Study Library
- AI Chat Assistant (Syllabus-based)
- Premium UI with modern design
- Fast & Lightweight

---

## Tech Stack

- Python
- Streamlit
- Sentence Transformers
- PyTorch
- Local File Storage (users.txt)

---

## Project Structure
campus-brain-ai/
│
├── app.py
├── users.txt
├── knowledge/
│ ├── dbms.txt
│ ├── os.txt
│ └── sorting.txt
└── README.md
---

## ▶️ How to Run Locally

1️⃣ Clone the repository

git clone https://github.com/your-username/campus-brain-ai.git  
cd campus-brain-ai  

2️⃣ Install dependencies

pip install -r requirements.txt  

3️⃣ Run the application

streamlit run app.py  

---

## 🧠 How It Works

- The app loads study material from the `knowledge` folder.
- It converts text into embeddings using `all-MiniLM-L6-v2`.
- When a user asks a question, semantic similarity is calculated.
- The most relevant topic is selected and displayed.
- The chat assistant provides simplified explanations.

---

## 📌 Future Improvements

- Secure password hashing
- SQLite / Firebase database integration
- PDF upload support
- Multi-subject support
- Deployment with custom domain
- Admin dashboard

---

## 🌐 Deployment

This app can be deployed easily using:

- Streamlit Cloud
- Render
- Railway
- AWS / Azure

---

## Author

Mrunmai Magade  
BTech IT Student  
Passionate about Web Development & AI  

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!




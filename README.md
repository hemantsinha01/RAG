# 📄 Retrieval-Augmented Generation (RAG) Streamlit Apps 🚀

This repository contains three RAG (Retrieval-Augmented Generation) applications built using **LangChain**, **Streamlit**, **FAISS**, and **Groq API (LLaMa3-8b-8192)**.

These apps allow users to interact with documents (PDFs or Web content) using intelligent question-answering powered by vector search and LLMs.

---

## 🔗 Live Demo Links

### 1. **PDF RAG App (Pre-uploaded PDF)**
- ✅ This app allows users to ask questions based on a **pre-loaded PDF document**.
- 🔥 PDF is already embedded in the app.
- 📎 [Try it here](https://aa9hnzmr7maqrnbnzmsrzb.streamlit.app/)

---

### 2. **Web RAG App (User URL input)**
- 🌐 Enter any **website URL**.
- 🤖 Ask questions based on the web page content.
- 📎 [Try it here](https://nm7qfybun8zscv3duzztax.streamlit.app/)

---

### 3. **PDF RAG App (User-uploaded PDF)**
- 📤 Upload a **PDF of your choice**.
- 🧠 Get answers based on your uploaded document.
- 📎 [Try it here](https://x9x4xjhxph2nmzzwttelmw.streamlit.app/)

---

## 🛠️ Tech Stack
- **LangChain** for RAG chains
- **FAISS** for vector storage
- **Groq LLaMa3-8b-8192** as LLM
- **HuggingFace Embeddings** (all-MiniLM-L6-v2)
- **Streamlit** for UI
- **Python 3.10**

---

## 📚 How it works
1. Documents are loaded (PDF/Web).
2. Text is split and embedded using HuggingFace models.
3. Vector similarity search retrieves relevant context.
4. Groq LLM generates final answer using RAG approach.

---

## 🚀 Upcoming Features
- ✅ Multi-file PDF RAG
- ✅ History chat feature
- ✅ Better UI/UX polish
- ✅ Evaluation metrics for answers

---

## 🧑‍💻 Author
Made with ❤️ by Hemant Kuamr 


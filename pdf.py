import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import time
import tempfile

# Load environment variables
load_dotenv()

# Set Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Setup Groq LLM
llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

# Prompt template (can be used in advanced chains)
prompt = ChatPromptTemplate.from_template(
    """Answer the question based on the context only.
       Please provide the most accurate response based on the question.
       <context>
       {context}
       <context>
       Question: {input}
    """
)

# Initialize session state for vectorstore
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Function to create vector embedding from uploaded file
def create_vector_embedding(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create vectorstore
    vectorstore = FAISS.from_documents(final_documents, embeddings)
    return vectorstore

# Streamlit UI
st.title("📄 RAG Document Q&A with Groq API & HuggingFace Embeddings")
st.write("Upload a PDF and ask questions about its content using Retrieval-Augmented Generation (RAG).")

# File uploader for user PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Button to build the vectorstore from uploaded file
if uploaded_file and st.button("Create Vector Database"):
    st.session_state.vectorstore = create_vector_embedding(uploaded_file)
    if st.session_state.vectorstore:
        st.success("Vector database created successfully! You can now ask questions.")
    else:
        st.error("Failed to create vectorstore.")

# Input box for user queries
user_prompt = st.text_input("Enter your query:")

# RAG QA chain function
def run_qa_chain(user_prompt, vectorstore):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    start = time.process_time()
    response = qa_chain.run({"query": user_prompt})
    end = time.process_time()

    st.write(f"⏱️ Response time: {end - start:.2f} seconds")
    st.write("### 📥 Answer:")
    st.write(response)

# If user enters query, process it
if user_prompt:
    if st.session_state.vectorstore is None:
        st.warning("Please upload a PDF and create the vector database first.")
    else:
        run_qa_chain(user_prompt, st.session_state.vectorstore)

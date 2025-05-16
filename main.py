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
import asyncio


# Load environment variables
load_dotenv()

# Set Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Setup Groq LLM
llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

# Prompt template
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

# Function to create vector embedding
def create_vector_embedding():
    loader = PyPDFLoader('LLM.pdf')  # Make sure this file exists
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
st.title("RAG Document Q&A with Groq API & HuggiggFace Embeddings")
st.write("This app allows you to ask questions about the content of a PDF document using a Retrieval-Augmented Generation (RAG) approach.")
user_prompt = st.text_input("Enter your query:")

# Button to build the vectorstore
if st.button("Create Vector Database"):
    st.session_state.vectorstore = create_vector_embedding()
    if st.session_state.vectorstore:
        st.success("Vector database is ready!")
    else:
        st.error("Failed to create vectorstore.")

# RAG QA chain function
def run_qa_chain(user_prompt, vectorstore):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    start = time.process_time()
    response = qa_chain.run({"query":user_prompt})
    end = time.process_time()

    st.write(f"⏱️ Response time: {end - start:.2f} seconds")
    st.write("### 📥 Answer:")
    st.write(response)
    
# Evaluation function
    

# If user enters query, process it
if user_prompt:
    if st.session_state.vectorstore is None:
        st.warning("Please create the vector database first by clicking the button above.")
    else:
        run_qa_chain(user_prompt, st.session_state.vectorstore)

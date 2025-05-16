import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Setup Groq LLM
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", groq_api_key=groq_api_key)

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """Answer the question based on the context only.
       Please provide the most accurate response based on the question and after carefull analysis of the context. PLease try to answer in yes or no
       <context>
       {context}
       <context>
       Question: {input}
    """
)

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

st.title("🌐 Webpage RAG Q&A with Groq API & HuggingFace Embeddings")
st.write("This app allows you to ask questions about the content of a webpage using a Retrieval-Augmented Generation (RAG) approach.")
st.write("Enter a URL to fetch the content and create a vector database.")
web_url = st.text_input("Enter the URL of a website:")
user_prompt = st.text_input("Ask a question based on the web content:")

# Function to create vectorstore from a webpage
def create_vectorstore_from_web(url):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Failed to load content from URL: {e}")
        return None

# Create vectorstore from website
if st.button("Fetch & Create Vector DB from Web URL"):
    if not web_url:
        st.warning("Please enter a valid URL.")
    else:
        st.write("Fetching content and creating vector database...")
        st.session_state.vectorstore = create_vectorstore_from_web(web_url)
        if st.session_state.vectorstore:
            st.success("Vector database created from webpage content!")

# QA function
def run_qa_chain(user_prompt, vectorstore):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    start = time.process_time()
    response = qa_chain.run({"query": user_prompt})
    end = time.process_time()
    st.write(f"⏱️ Response time: {end - start:.2f} seconds")
    st.write("### 📥 Answer:")
    st.write(response)

# Run QA if prompt is given
if user_prompt:
    if st.session_state.vectorstore:
        run_qa_chain(user_prompt, st.session_state.vectorstore)
    else:
        st.warning("Please fetch and create the vector DB first.")

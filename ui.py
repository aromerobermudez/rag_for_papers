import os, pathlib, configparser
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

config = configparser.ConfigParser()
config.read('app_config.cfg')

# configuration: open AI and FAISS
os.environ["OPENAI_API_KEY"] = config.get('openai','OPENAI_API_KEY')
section = 'paths'
FAISS_INDEX_PATH = config.get(section,'FAISS_INDEX_PATH')
UPLOAD_FOLDER = config.get(section,'UPLOAD_FOLDER')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
index_file = pathlib.Path(FAISS_INDEX_PATH) / "index.faiss"

st.title("My bibliography")

# Section to Upload PDFs
uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
if uploaded_files:
    st.write("Processing PDFs...")
    docs = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

    # Splitting into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.getint('pdf_processing', 'chunk_overlap', fallback=500), 
        chunk_overlap=config.getint('pdf_processing', 'chunk_overlap', fallback=50))
    split_docs = splitter.split_documents(docs)

    # Embed pdfs and save to FAISS index
    embedding_model = OpenAIEmbeddings(
        model=config.get('models','EMBEDDING_MODEL')
        )
    
    if index_file.exists():
        # Load existing FAISS index
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
        vectorstore.add_documents(split_docs)
    else:
        # Create FAISS index from scratch
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        vectorstore = FAISS.from_documents(split_docs, embedding_model)
    vectorstore.save_local(FAISS_INDEX_PATH)
    st.success("PDFs have been embedded and FAISS index updated!")

# Chatbot ui
st.subheader("💬 Ask questions about your papers")
query = st.text_input("My question is:")
if query:
    import requests
    response = requests.post("http://localhost:8000/ask", json={"query": query})
    if response.status_code == 200:
        answer = response.json()
        st.write("**Answer:**", answer["answer"])
        with st.expander("Sources"):
            st.write(answer["sources"])
    else:
        st.error("Error contacting API. Make sure FastAPI backend is running.")

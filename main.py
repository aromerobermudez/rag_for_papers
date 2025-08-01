import os, configparser
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

config = configparser.ConfigParser()
config.read('app_config.cfg')

# API Key
os.environ["OPENAI_API_KEY"] = config.get('openai','OPENAI_API_KEY')

# Paths
section = 'paths'
FAISS_INDEX_PATH = config.get(section,'FAISS_INDEX_PATH')

# Embedding model
embedding_model = OpenAIEmbeddings(model=config.get('models','EMBEDDING_MODEL'))
if not os.path.exists(FAISS_INDEX_PATH):
        raise Exception("No FAISS index found. Upload pdfs")
embedded_database = FAISS.load_local(FAISS_INDEX_PATH, 
                                     embedding_model,
                                     allow_dangerous_deserialization=True)

# Initialize RAG pipeline
retriever = embedded_database.as_retriever(
    search_kwargs={"k": config.getint('retriever','RETRIEVER_NUMBER_DOCS') }
    )
llm = ChatOpenAI(model_name=config.get('models','LLM_MODEL'), 
                 temperature=config.getint('models','LLM_temperature',fallback=1)
                 )
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)


# application
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    result = qa_chain({"query": request.query})
    sources = [doc.metadata for doc in result["source_documents"]]
    return {"answer": result["result"], "sources": sources}

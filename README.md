Minimal RAG
-------------------
RAG application to find information from bibliography (collection of papers in pdf)

Requirements 
 * OpenAI LLM (o4-mini)
 * OpenAI Embeddings (text-embedding-3-small)
 * FAISS (for vector search with persistent storage)
 * FastAPI backend (API access)
 * Streamlit frontend (interactive chat and PDF uploads)


rag_app/
├── app_config.cfg      # Configuration file
├── main.py             # FastAPI backend
├── ui.py               # Streamlit UI
├── faiss_index/        # faiss index file (see cfg)
└── papers/             # upload folder (see cfg)
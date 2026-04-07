# LangChain + Ollama Notebook (searchContextInContent/ollama.ipynb)


This notebook demonstrates a complete RAG-style workflow:
- Load web content (LangChain `WebBaseLoader`)
- Split into chunks (`RecursiveCharacterTextSplitter`)
- Create embeddings (`OllamaEmbeddings`)
- Store with FAISS vector index
- Build retriever and document chain
- Query via `ChatOllama` model (`gemma:2b`)
- Return answer + context

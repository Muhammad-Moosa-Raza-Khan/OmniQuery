import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

os.makedirs("data", exist_ok=True)

def build_retriever():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    
    if not documents:
        from langchain_core.documents import Document
        documents = [Document(page_content="The RAM Net Ultra architecture utilizes MoE and frequency-guided attention for breast cancer detection.")]
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store.as_retriever(search_kwargs={"k": 3})

retriever = build_retriever()
import streamlit as st
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "course_materials"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
load_dotenv()

# Page config
st.set_page_config(
    page_title="Course Material QA Bot",
    layout="wide"
)

# Initialize embedding model
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

#Initialize Chroma Vector Store
@st.cache_resource
def get_vector_store():
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

def load_document(file_path: str, file_type: str):
    """Load document based on file type"""
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    else:  # txt document
        loader = TextLoader(file_path, encoding='utf-8')
    return loader.load()

# Split into chunks
def process_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)

# Add chunks to vector store
def add_to_vector_store(chunks, filename):
    vector_store = get_vector_store()
    
    # Add source filename to metadata
    for chunk in chunks:
        chunk.metadata["source_file"] = filename
    
    vector_store.add_documents(chunks)
    return len(chunks)

# Get list of documents in Vector Store
def get_stored_documents():
    vector_store = get_vector_store()
    try:
        collection = vector_store._collection
        if collection.count() == 0:
            return [], 0
        
        results = collection.get(include=["metadatas"])
        sources = set()
        for meta in results.get("metadatas", []):
            if meta and "source_file" in meta:
                sources.add(meta["source_file"])
        return list(sources), collection.count()
    except:
        return [], 0

# Clear Vector Store
def clear_vector_store():
    vector_store = get_vector_store()
    vector_store._collection.delete(where={})

#----------------------------------- UI -----------------------------------------#

st.title(" Course Material QA Bot")
st.markdown("Upload your course materials and get ready to learn!")

# Two-column layout
col1, col2 = st.columns([1, 1])

# -----------------------------------------------------------------------------
# LEFT COLUMN: Document Upload Section
# -----------------------------------------------------------------------------
with col1:
    st.header("Document Upload")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload course materials (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="You can upload multiple files at once"
    )
    
    # Processing options
    with st.expander("Processing Options", expanded=False):
        chunk_size = st.slider("Chunk Size", 500, 2000, CHUNK_SIZE, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, CHUNK_OVERLAP, 50)
    
    # Process button
    if uploaded_files:
        if st.button("Process & Store Documents", type="primary", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            
            total_chunks = 0
            
            for idx, file in enumerate(uploaded_files):
                status.text(f"Processing: {file.name}")
                
                # Save to temp file
                suffix = ".pdf" if file.type == "application/pdf" else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    # Load document
                    file_type = "pdf" if suffix == ".pdf" else "txt"
                    docs = load_document(tmp_path, file_type)
                    
                    # Chunk documents
                    chunks = process_documents(docs)
                    
                    # Add to vector store
                    num_chunks = add_to_vector_store(chunks, file.name)
                    total_chunks += num_chunks
                    
                finally:
                    os.unlink(tmp_path)  # Clean up temp file
                
                progress.progress((idx + 1) / len(uploaded_files))
            
            status.empty()
            progress.empty()
            st.success(f"Processed {len(uploaded_files)} file(s) → {total_chunks} chunks stored!")
            st.rerun()
    
    # Current vector store status
    st.divider()
    st.subheader("Vector Store Status")
    
    stored_docs, total_chunks = get_stored_documents()
    
    col_a, col_b = st.columns(2)
    col_a.metric("Documents", len(stored_docs))
    col_b.metric("Total Chunks", total_chunks)
    
    if stored_docs:
        with st.expander("📁 Stored Documents", expanded=True):
            for doc in stored_docs:
                st.text(f"• {doc}")
        
        if st.button("Clear All Documents", type="secondary"):
            clear_vector_store()
            st.success("Vector store cleared!")
            st.rerun()

# -----------------------------------------------------------------------------
# RIGHT COLUMN: Chat Section (Placeholder)
# -----------------------------------------------------------------------------
with col2:
    st.header("Chat")
    
    # Placeholder for future chat implementation
    st.info("Coming soon!")
    
    # Disabled chat input as preview
    st.chat_input("Ask a question about your course materials...", disabled=True)

# -----------------------------------------------------------------------------
# Sidebar: Info & Instructions
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app lets you:
    1. **Upload** course PDFs/TXTs
    2. **Process** them into chunks
    3. **Store** embeddings in ChromaDB
    4. **Query** using natural language
    """)
    
    st.divider()
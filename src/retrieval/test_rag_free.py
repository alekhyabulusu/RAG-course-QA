from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Use same free embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load vector store
vectorstore = Chroma(
    persist_directory="data/chroma",
    embedding_function=embeddings,
    collection_name="course_materials"
)

# Test retrieval
query = "What are transformers in LLMs?"
docs = vectorstore.similarity_search(query, k=3)

print(f"Found {len(docs)} relevant chunks:")
for i, doc in enumerate(docs, 1):
    print(f"\n{i}. From: {doc.metadata.get('source', 'Unknown')}")
    print(f"   Content: {doc.page_content[:200]}...")

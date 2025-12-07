import json
import sys
from pathlib import Path
from typing import List, Dict
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document  # Updated import
import chromadb
from tqdm import tqdm

def create_embeddings():
    """Create embeddings and store in ChromaDB"""
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please set it in your .env file")
        return
    
    # Load processed documents
    processed_file = Path("data/processed/processed_documents.json")
    if not processed_file.exists():
        print(f"❌ Processed file not found: {processed_file}")
        print("Please run process_documents.py first")
        return
    
    print("📚 Loading processed documents...")
    with open(processed_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} chunks")
    
    # Convert to Document objects
    documents = []
    for item in data:
        doc = Document(
            page_content=item['content'],
            metadata=item['metadata']
        )
        documents.append(doc)
    
    # Initialize embeddings
    print("🔧 Initializing OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        model="text-embedding-ada-002"
    )
    
    # Create ChromaDB directory
    chroma_path = Path("data/chroma")
    chroma_path.mkdir(parents=True, exist_ok=True)
    
    # Create vector store
    print("🗂️ Creating ChromaDB vector store...")
    print("⏳ This may take a few minutes depending on the number of chunks...")
    
    try:
        # Process in batches to avoid rate limits
        batch_size = 50  # Smaller batch size to avoid rate limits
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="course_materials",
            persist_directory=str(chroma_path)
        )
        
        print(f"\n✅ Successfully created embeddings for {len(documents)} chunks")
        print(f"📁 Embeddings stored in: {chroma_path}")
        
        # Test the vector store
        print("\n🧪 Testing vector store with a sample query...")
        test_query = "What is a language model?"
        results = vectorstore.similarity_search(test_query, k=2)
        print(f"Found {len(results)} similar chunks for test query")
        
    except Exception as e:
        print(f"❌ Error creating embeddings: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check your OpenAI API key is valid")
        print("2. Check you have API credits remaining")
        print("3. Try reducing batch_size if rate limited")

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    create_embeddings()

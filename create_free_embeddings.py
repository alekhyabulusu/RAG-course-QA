import json
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

print("📚 Loading processed documents...")
with open("data/processed/processed_documents.json", 'r') as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} chunks")

# Extract texts and metadatas
texts = [item['content'] for item in data]
metadatas = [item['metadata'] for item in data]

print("\n🤖 Loading FREE HuggingFace model (first time will download ~90MB)...")
# This uses a FREE model that runs locally - no API needed!
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("\n📊 Creating embeddings (this will take 2-5 minutes)...")
print("Processing locally - no API costs!")

# Create vector store with progress bar
Path("data/chroma").mkdir(parents=True, exist_ok=True)

# Process in batches for better progress tracking
batch_size = 100
for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
    batch_texts = texts[i:i+batch_size]
    batch_metadata = metadatas[i:i+batch_size]
    
    if i == 0:
        # First batch - create new store
        vectorstore = Chroma.from_texts(
            texts=batch_texts,
            metadatas=batch_metadata,
            embedding=embeddings,
            collection_name="course_materials",
            persist_directory="data/chroma"
        )
    else:
        # Add to existing store
        vectorstore.add_texts(
            texts=batch_texts,
            metadatas=batch_metadata
        )

print(f"\n✅ SUCCESS! Created FREE embeddings for {len(texts)} chunks")
print("📁 Stored in: data/chroma")
print("💰 Cost: $0.00")
print("\n🎯 You can now test your RAG system!")

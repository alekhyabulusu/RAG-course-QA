from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "course_materials"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PATH
    )


def print_separator():
    print("\n" + "=" * 70 + "\n")

# Stats about the vector store
def test_collection_stats(vector_store):
    collection = vector_store._collection
    count = collection.count()
    
    print("VECTOR STORE STATISTICS")
    print(f"   Total chunks stored: {count}")
    
    if count > 0:
        results = collection.get(include=["metadatas"])
        sources = set()
        for meta in results.get("metadatas", []):
            if meta and "source_file" in meta:
                sources.add(meta["source_file"])
        
        print(f"   Documents indexed: {len(sources)}")
        for src in sources:
            print(f"     • {src}")
    
    return count > 0


# Run Similarity Search
def test_similarity_search(vector_store, query: str, k: int = 3):
    print(f" QUERY: \"{query}\"")
    print(f"   Retrieving top {k} chunks...\n")
    
    # Method 1: Basic similarity search
    results = vector_store.similarity_search(query, k=k)
    
    for i, doc in enumerate(results, 1):
        print(f"Result {i}")
        print(f"   Source: {doc.metadata.get('source_file', 'Unknown')}")
        if 'page' in doc.metadata:
            print(f"   Page: {doc.metadata['page']}")
        print(f"   Content Preview:")
        # Show first 300 chars of content
        preview = doc.page_content[:300].replace('\n', ' ')
        print(f"   \"{preview}...\"")
        print()

# Similarity Search with relevance scores
def test_similarity_search_with_scores(vector_store, query: str, k: int = 3):
    print(f" QUERY WITH SCORES: \"{query}\"")
    print(f"   Retrieving top {k} chunks...\n")
    
    # Method 2: Similarity search with scores
    results = vector_store.similarity_search_with_score(query, k=k)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"Result {i} | Similarity Score: {score:.4f}")
        print(f"   Source: {doc.metadata.get('source_file', 'Unknown')}")
        if 'page' in doc.metadata:
            print(f"   Page: {doc.metadata['page']}")
        print(f"   Content Preview:")
        preview = doc.page_content[:300].replace('\n', ' ')
        print(f"   \"{preview}...\"")
        print()

# MMR Search
def test_mmr_search(vector_store, query: str, k: int = 3):

    print(f"MMR SEARCH: \"{query}\"")
    print(f"Retrieving top {k} diverse chunks...\n")
    
    # Method 3: Maximum Marginal Relevance for diversity
    results = vector_store.max_marginal_relevance_search(
        query, 
        k=k,
        fetch_k=10,
        lambda_mult=0.5  
    )
    
    for i, doc in enumerate(results, 1):
        print(f"Result {i}")
        print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
        preview = doc.page_content[:300].replace('\n', ' ')
        print(f"   \"{preview}...\"")
        print()

# Test Retriever
def test_retriever(vector_store, query: str):
    print(f"RETRIEVER TEST: \"{query}\"")
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    docs = retriever.invoke(query)
    print(f"   Retrieved {len(docs)} documents\n")
    
    for i, doc in enumerate(docs, 1):
        print(f"   [{i}] {doc.metadata.get('source_file', 'Unknown')}: {doc.page_content[:100]}...")


# Write queries to get results from the vector stores
def interactive_mode(vector_store):
    print("\nINTERACTIVE MODE")
    print("Type your queries to test retrieval.\n")
    
    while True:
        query = input("Enter query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        print()
        test_similarity_search_with_scores(vector_store, query, k=3)
        print_separator()



if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("VECTOR STORE TEST SUITE")
    print("=" * 70 + "\n")
    
    # Load vector store
    print("Loading vector store...")
    vector_store = get_vector_store()
    
    # Check if there's data
    if not test_collection_stats(vector_store):
        print("\nNo documents found!")
        exit(1)
    
    print_separator()
    

    test_queries = [
        "What is a V3 architecture?",
        "Explain MTP",
        "How DeepSeekMoE work?",
    ]
    
    # Run tests for each query
    for query in test_queries:
        test_similarity_search_with_scores(vector_store, query, k=3)
        print_separator()
    
    # Test MMR for comparison
    if test_queries:
        test_mmr_search(vector_store, test_queries[0], k=3)
        print_separator()
    
    # Test retriever interface
    if test_queries:
        test_retriever(vector_store, test_queries[0])
        print_separator()
    
    # Interactive mode
    response = input("Enter interactive mode? (y/n): ").strip().lower()
    if response == 'y':
        interactive_mode(vector_store)
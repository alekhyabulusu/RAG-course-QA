"""RAG System for Course Q&A with Multiple Model Comparison"""

"""Simplified RAG System that works with current LangChain"""

import os
from typing import Dict, List
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

load_dotenv()

class SimpleRAG:
    def __init__(self):
        """Initialize the simple RAG system"""
        print("🚀 Initializing Simple RAG System...")
        
        # Use free embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Load vector store
        self.vectorstore = Chroma(
            persist_directory="data/chroma",
            embedding_function=self.embeddings,
            collection_name="course_materials"
        )
        
        print("✅ RAG System ready!")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant chunks"""
        
        print(f"\n🔍 Searching for: '{query}'")
        
        # Search for similar documents
        docs = self.vectorstore.similarity_search(query, k=k)
        
        results = []
        for i, doc in enumerate(docs, 1):
            results.append({
                "chunk_num": i,
                "content": doc.page_content,
                "source": doc.metadata.get('source', 'Unknown'),
                "chunk_index": doc.metadata.get('chunk_index', 0)
            })
        
        return results
    
    def answer_with_context(self, question: str, use_llm: bool = False) -> Dict:
        """Get answer with retrieved context"""
        
        # Get relevant chunks
        relevant_chunks = self.search(question, k=5)
        
        if not relevant_chunks:
            return {
                "question": question,
                "context": "No relevant content found",
                "answer": "No relevant information found in the course materials."
            }
        
        # Combine context
        context = "\n\n".join([
            f"[From {chunk['source']}]:\n{chunk['content'][:300]}"
            for chunk in relevant_chunks[:3]
        ])
        
        result = {
            "question": question,
            "context": context,
            "sources": [chunk['source'] for chunk in relevant_chunks],
            "num_chunks": len(relevant_chunks)
        }
        
        if use_llm:
            try:
                # Try to use OpenAI
                llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.7,
                    openai_api_key=os.getenv("OPENAI_API_KEY")
                )
                
                prompt = f"""Based on the following context from course materials, answer the question.
                
Context:
{context}

Question: {question}

Answer:"""
                
                answer = llm.predict(prompt)
                result["answer"] = answer
                result["llm_used"] = True
                
            except Exception as e:
                result["answer"] = f"LLM Error: {str(e)[:100]}"
                result["llm_used"] = False
        else:
            result["answer"] = "Context retrieved successfully (LLM not used)"
            result["llm_used"] = False
        
        return result

def main():
    """Test the simple RAG system"""
    
    rag = SimpleRAG()
    
    print("\n" + "="*60)
    print("SIMPLE RAG SYSTEM - TESTING")
    print("="*60)
    
    # Test questions
    test_questions = [
        "What are transformers?",
        "What is attention mechanism?",
        "Explain fine-tuning"
    ]
    
    print("\n📝 Testing retrieval:")
    print("-"*40)
    
    for question in test_questions[:2]:
        results = rag.search(question, k=3)
        print(f"\n❓ Question: {question}")
        print(f"📚 Found {len(results)} relevant chunks:")
        
        for r in results[:2]:  # Show first 2
            print(f"\n  Chunk {r['chunk_num']} from {r['source']}:")
            print(f"  {r['content'][:150]}...")
    
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("Type your questions or 'quit' to exit")
    print("="*60)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        # Just do retrieval (free)
        results = rag.search(question, k=5)
        
        print(f"\n📚 Found {len(results)} relevant chunks:")
        for r in results[:3]:  # Show top 3
            print(f"\n  From {r['source']}:")
            print(f"  {r['content'][:200]}...")
        
        # Show sources
        sources = list(set([r['source'] for r in results]))
        print(f"\n📄 Sources: {', '.join(sources[:3])}")

if __name__ == "__main__":
    main()

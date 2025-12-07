"""FastAPI backend for RAG system"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

app = FastAPI(title="RAG Course Q&A API", version="1.0")

# Load RAG system
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="data/chroma",
    embedding_function=embeddings,
    collection_name="course_materials"
)

class Query(BaseModel):
    question: str
    k: int = 5

class Response(BaseModel):
    question: str
    answer: str
    sources: List[str]
    chunks_used: int

@app.get("/")
def read_root():
    return {"message": "RAG Course Q&A API", "status": "active"}

@app.post("/query", response_model=Response)
def query_rag(query: Query):
    try:
        docs = vectorstore.similarity_search(query.question, k=query.k)
        
        if not docs:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        answer = "\n".join([doc.page_content[:200] for doc in docs[:3]])
        sources = list(set([doc.metadata.get('source', 'Unknown') for doc in docs]))
        
        return Response(
            question=query.question,
            answer=answer,
            sources=sources,
            chunks_used=len(docs)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

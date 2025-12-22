# Course Companion: RAG-Powered Course Material Q$A System

A retrieval-augmented generation (RAG) system that enables question answering over uploaded course materials. Built with LangChain, ChromaDB, and Meta's Llama 3.1.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)

---

**Course Companion** solves the problem of finding specific information across multiple course documents—lecture slides, readings, syllabi by:
1. Ingesting your course materials into a vector database
2. Retrieving only the relevant chunks for each question
3. Generating grounded answers with source citations

---

## Features

- **Document ingestion**: Upload PDFs of lecture notes, slides, and readings
- **Semantic search**: ChromaDB vector store with MiniLM embeddings
- **Smart chunking**: Semantic text splitting for better context preservation
- **Query classification**: Routes questions to appropriate retrieval strategies
- **8 tutor personas**: Different teaching styles via prompt engineering (Socratic, concise, detailed, etc.)
- **Source citations**: Every answer includes references to source documents
- **Streamlit UI**: Simple web interface for uploading and querying

---

## Installation

### Prerequisites
- Python 3.9+
- Hugging Face account (for Llama 3.1 API access)

---

Type your question in the input box. Optionally select a tutor persona:

| Persona | Description |
|---------|-------------|
| **Default** | Balanced, informative responses |
| **Socratic** | Guides you with questions |
| **Concise** | Brief, direct answers |
| **Detailed** | Comprehensive explanations |
| **ELI5** | Simple language, analogies |
| **Technical** | Precise, formal terminology |
| **Encouraging** | Supportive, positive framing |
| **Critical** | Challenges assumptions |

---


## Tech Stack

- **Framework**: LangChain
- **Vector Store**: ChromaDB
- **Embeddings**: all-MiniLM-L6-v2 (Sentence Transformers)
- **LLM**: Meta Llama 3.1 (via Hugging Face Inference API)
- **PDF Processing**: PyPDF
- **UI**: Streamlit
- **Evaluation**: RAGAS


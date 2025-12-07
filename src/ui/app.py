"""Streamlit UI for RAG Course Q&A"""
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Course Q&A", page_icon="📚")
st.title("📚 LLM Course Q&A System")
st.markdown("Ask questions about your course materials with adaptive complexity!")

# Initialize
@st.cache_resource
def load_system():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="data/chroma")
    collection = client.get_collection("course_materials")
    return model, collection

model, collection = load_system()

# Function to adapt response based on difficulty
def adapt_response(content, difficulty):
    """Adapt the response based on difficulty level"""
    
    if difficulty == "Beginner":
        # Simplify the response
        intro = "🟢 **Simple Explanation:**\n\n"
        # Take first 200 chars and add simple context
        simplified = content[:200]
        response = intro + simplified + "\n\n💡 **In simple terms:** This is a fundamental concept in machine learning that helps the model understand and process information."
        
    elif difficulty == "Intermediate":
        # Standard response
        intro = "🟡 **Detailed Explanation:**\n\n"
        # Show more content
        response = intro + content[:400] + "\n\n📚 **Key Points:** This concept is important for understanding how modern LLMs work."
        
    else:  # Advanced
        # Technical response
        intro = "🔴 **Technical Deep-Dive:**\n\n"
        # Show full content with technical details
        response = intro + content[:600] + "\n\n🔬 **Advanced Considerations:** Consider the mathematical foundations, optimization techniques, and implementation details when working with this concept."
    
    return response

# Main interface
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    question = st.text_input(
        "Ask a question:",
        placeholder="e.g., What is the attention mechanism?"
    )

with col2:
    difficulty = st.selectbox(
        "Difficulty:",
        ["Beginner", "Intermediate", "Advanced"],
        index=1
    )

with col3:
    num_results = st.selectbox("Sources:", [3, 5, 10], index=0)

# Show difficulty explanation
if difficulty == "Beginner":
    st.info("🟢 **Beginner Mode**: Simple explanations with basic concepts")
elif difficulty == "Intermediate":
    st.info("🟡 **Intermediate Mode**: Detailed explanations with context")
else:
    st.info("🔴 **Advanced Mode**: Technical details and in-depth analysis")

if st.button("🔍 Get Answer", type="primary"):
    if question:
        with st.spinner(f"Searching for {difficulty.lower()}-level explanation..."):
            # Create embedding for the question
            query_embedding = model.encode([question])
            
            # Search in ChromaDB
            results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=num_results
            )
            
            # Display results
            if results['documents'] and results['documents'][0]:
                st.success(f"Found {len(results['documents'][0])} relevant chunks!")
                
                # Combine top results
                combined_content = " ".join(results['documents'][0][:2])
                
                # Adapt response based on difficulty
                adapted_response = adapt_response(combined_content, difficulty)
                
                # Show main answer
                st.subheader("📝 Answer:")
                st.markdown(adapted_response)
                
                # Show sources differently based on difficulty
                st.subheader("📚 Sources:")
                
                if difficulty == "Beginner":
                    # Just show source names
                    sources = list(set([m.get('source', 'Unknown') for m in results['metadatas'][0]]))
                    for source in sources[:2]:
                        st.write(f"• {source}")
                
                elif difficulty == "Intermediate":
                    # Show sources with snippets
                    for i, (doc, metadata) in enumerate(zip(results['documents'][0][:2], results['metadatas'][0][:2]), 1):
                        with st.expander(f"Source {i}: {metadata.get('source', 'Unknown')}"):
                            st.write(doc[:200] + "...")
                
                else:  # Advanced
                    # Show full sources with metadata
                    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                        with st.expander(f"Source {i}: {metadata.get('source', 'Unknown')} (Chunk {metadata.get('chunk_index', 'N/A')})"):
                            st.write(doc)
                            st.json(metadata)
                
                # Add follow-up suggestions based on difficulty
                st.subheader("🤔 Follow-up Questions:")
                if difficulty == "Beginner":
                    st.write("• What are the basic components?")
                    st.write("• Can you give a simple example?")
                elif difficulty == "Intermediate":
                    st.write("• How does this compare to other approaches?")
                    st.write("• What are the practical applications?")
                else:
                    st.write("• What are the mathematical foundations?")
                    st.write("• How is this implemented in production?")
                    
            else:
                st.warning("No relevant content found.")
    else:
        st.error("Please enter a question!")

# Sidebar
st.sidebar.title("ℹ️ About Difficulty Levels")
st.sidebar.markdown("""
**🟢 Beginner:**
- Simple explanations
- Basic concepts only
- Minimal technical terms

**🟡 Intermediate:**
- Detailed explanations
- Some technical details
- Practical examples

**🔴 Advanced:**
- Full technical depth
- Mathematical details
- Implementation specifics
""")

# Metrics based on difficulty
st.sidebar.title("📊 Current Settings")
st.sidebar.metric("Mode", difficulty)
st.sidebar.metric("Complexity", 
    "Low" if difficulty == "Beginner" else 
    "Medium" if difficulty == "Intermediate" else 
    "High")
st.sidebar.metric("Detail Level",
    "Basic" if difficulty == "Beginner" else
    "Standard" if difficulty == "Intermediate" else
    "Complete")

# Sample questions with difficulty indicators
st.sidebar.title("💡 Sample Questions")
questions_by_level = {
    "Beginner": [
        "What is machine learning?",
        "What are neural networks?",
    ],
    "Intermediate": [
        "How do transformers work?",
        "What is fine-tuning?",
    ],
    "Advanced": [
        "Explain multi-head attention mathematics",
        "Describe RLHF implementation details",
    ]
}

current_questions = questions_by_level.get(difficulty, [])
for q in current_questions:
    if st.sidebar.button(q, key=q):
        st.session_state.sample_q = q
        st.rerun()
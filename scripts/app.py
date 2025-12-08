import streamlit as st
import os
import tempfile
import re
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Configuration
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "course_materials"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
load_dotenv()

# =============================================================================
# TUTOR PERSONALITIES
# =============================================================================
TUTOR_PERSONALITIES = {
    "professor": {
        "name": "Dr. Watson",
        "emoji": "👨‍🏫",
        "tagline": "The Classic Professor",
        "description": "Formal, thorough, academically rigorous",
        "system_prompt": """You are Dr. Watson, a distinguished professor with decades of teaching experience.

PERSONALITY:
- Formal but approachable academic tone
- Thorough and well-structured explanations
- Uses proper terminology (but explains it)
- Builds concepts from fundamentals

STYLE:
- Start with clear definitions
- Use phrases like "Let us consider...", "It's important to note that..."
- Summarize key takeaways at the end"""
    },
    "socratic": {
        "name": "Socrates",
        "emoji": "🏛️",
        "tagline": "The Questioner",
        "description": "Guides through questions, never gives direct answers easily",
        "system_prompt": """You are Socrates, a wise philosopher who teaches through questions.

PERSONALITY:
- NEVER give direct answers immediately
- Guide students to discover knowledge themselves
- Ask thought-provoking questions
- Be patient and wise

STYLE:
- Respond to questions with guiding questions
- "What do you think would happen if...?"
- Only reveal answers after exploration"""
    },
    "coach": {
        "name": "Coach Maya",
        "emoji": "🎯",
        "tagline": "Exam Prep Expert",
        "description": "Focused on what matters for exams, no fluff",
        "system_prompt": """You are Coach Maya, an exam preparation specialist.

PERSONALITY:
- Direct and efficient - no fluff
- Highlights what's likely to be tested
- Points out common mistakes
- Creates mnemonics and memory tricks

STYLE:
- Lead with "KEY POINTS:"
- Flag common exam traps
- Provide quick recall techniques"""
    },
    "buddy": {
        "name": "Alex",
        "emoji": "😊",
        "tagline": "Your Study Buddy",
        "description": "Casual, friendly, uses lots of analogies",
        "system_prompt": """You are Alex, an enthusiastic study buddy who makes learning fun.

PERSONALITY:
- Super friendly and casual
- Gets excited about cool concepts
- Uses tons of real-world analogies
- Encouraging and supportive

STYLE:
- Talk like explaining to a friend
- "Okay so basically...", "Here's the cool part..."
- Use everyday analogies"""
    },
    "pirate": {
        "name": "Captain Blackboard",
        "emoji": "🏴‍☠️",
        "tagline": "The Pirate Professor",
        "description": "Arr! Learns ye the ways of knowledge!",
        "system_prompt": """You are Captain Blackboard, a pirate who teaches like treasure hunting.

PERSONALITY:
- Speak in pirate dialect (arr, matey, ye)
- Treat concepts as "treasure" to discover
- Call difficult topics "treacherous waters"

STYLE:
- "Ahoy! Let me chart these waters..."
- "X marks the spot!" for key insights
- Make learning an adventure"""
    },
    "anime_sensei": {
        "name": "Sensei Tanaka",
        "emoji": "⚔️",
        "tagline": "The Anime Master",
        "description": "Teaches like a wise anime mentor",
        "system_prompt": """You are Sensei Tanaka, a wise anime-style mentor.

PERSONALITY:
- Speak with dramatic anime flair
- Believe in student's hidden potential
- Use training/power-up metaphors

STYLE:
- "I see great potential in you..."
- "This technique requires focus..."
- "Your training continues!" """
    },
    "gamer": {
        "name": "xX_Prof_Xx",
        "emoji": "🎮",
        "tagline": "The Gamer Guide",
        "description": "Treats studying like a video game",
        "system_prompt": """You are a pro gamer who teaches like explaining game mechanics.

PERSONALITY:
- Everything is a game to master
- Concepts are like skills/abilities
- Learning is grinding XP

STYLE:
- "Let's break down this meta..."
- Call hard topics "boss fights"
- "GG!" for correct understanding"""
    },
    "grandma": {
        "name": "Grandma Rose",
        "emoji": "👵",
        "tagline": "The Wise Grandma",
        "description": "Warm, patient, explains with life wisdom",
        "system_prompt": """You are Grandma Rose, a warm grandmother who makes everything understandable.

PERSONALITY:
- Infinitely patient and kind
- Uses life stories and simple analogies
- Encouraging and nurturing

STYLE:
- "Now dear, let me explain..."
- Use homey analogies (baking, gardening)
- "You're doing wonderful, sweetie" """
    }
}

# =============================================================================
# QUERY TYPE DETECTION
# =============================================================================
AGGREGATE_PATTERNS = [
    r'\b(summarize|summary|summarise|overview|outline)\b',
    r'\b(explain|describe|what is|what are).*\b(entire|whole|full|all|document|paper|chapter)\b',
    r'\b(main|key|important|major)\s+(points?|concepts?|ideas?|topics?|themes?)\b',
    r'\b(cover|covers|about|contains?)\b.*\b(document|paper|material|reading)\b',
    r'\b(list|give me|what are)\s+(the\s+)?(topics?|concepts?|sections?)\b',
    r'\btl;?dr\b',
    r'\bgist\b'
]

def is_aggregate_query(query: str) -> bool:
    """Detect if query needs broad coverage vs specific retrieval"""
    query_lower = query.lower()
    for pattern in AGGREGATE_PATTERNS:
        if re.search(pattern, query_lower):
            return True
    return False

def get_specific_doc_name(query: str, available_docs: list) -> str | None:
    """Check if query mentions a specific document"""
    query_lower = query.lower()
    for doc in available_docs:
        doc_name_lower = doc.lower()
        # Check for filename match (with or without extension)
        base_name = os.path.splitext(doc_name_lower)[0]
        if base_name in query_lower or doc_name_lower in query_lower:
            return doc
    return None

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

st.set_page_config(page_title="Course Companion", layout="wide")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def get_llm():
    return HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation"
    )

@st.cache_resource
def get_chat_model():
    return ChatHuggingFace(llm=get_llm())

@st.cache_resource
def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PATH
    )

def load_document(file_path: str, file_type: str):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')
    return loader.load()

def process_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)

def add_to_vector_store(chunks, filename):
    vector_store = get_vector_store()
    for i, chunk in enumerate(chunks):
        chunk.metadata["source_file"] = filename
        chunk.metadata["chunk_index"] = i  # Track position in document
        chunk.metadata["total_chunks"] = len(chunks)
    vector_store.add_documents(chunks)
    return len(chunks)

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

def clear_vector_store():
    vector_store = get_vector_store()
    vector_store._collection.delete(where={})

# =============================================================================
# SMART RETRIEVAL - Different strategies for different query types
# =============================================================================

def retrieve_for_summary(target_doc: str = None, sample_size: int = 8) -> str:
    """Retrieve diverse chunks across document(s) for summarization"""
    vector_store = get_vector_store()
    collection = vector_store._collection
    
    # Build filter if specific doc requested
    where_filter = {"source_file": target_doc} if target_doc else None
    
    # Get all chunks (or from specific doc)
    results = collection.get(
        include=["documents", "metadatas"],
        where=where_filter
    )
    
    if not results["documents"]:
        return ""
    
    docs = results["documents"]
    metas = results["metadatas"]
    
    # Strategy: Sample from beginning, middle, and end of each document
    # Group by source file
    doc_chunks = {}
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        source = meta.get("source_file", "unknown")
        chunk_idx = meta.get("chunk_index", i)
        if source not in doc_chunks:
            doc_chunks[source] = []
        doc_chunks[source].append((chunk_idx, doc, meta))
    
    sampled_chunks = []
    
    for source, chunks in doc_chunks.items():
        # Sort by chunk index
        chunks.sort(key=lambda x: x[0])
        total = len(chunks)
        
        if total <= 3:
            # Small doc - take all
            indices = list(range(total))
        else:
            # Sample: first, 25%, 50%, 75%, last
            indices = [
                0,  # Beginning
                total // 4,  # Early
                total // 2,  # Middle
                3 * total // 4,  # Late
                total - 1  # End
            ]
            # Remove duplicates and limit
            indices = sorted(set(indices))[:sample_size // len(doc_chunks) or 2]
        
        for idx in indices:
            if idx < total:
                chunk_idx, content, meta = chunks[idx]
                sampled_chunks.append(f"[{source} - Section {chunk_idx + 1}/{total}]\n{content}")
    
    return "\n\n---\n\n".join(sampled_chunks[:sample_size])

def retrieve_similarity(query: str, k: int = 4) -> str:
    """Standard similarity search for specific questions"""
    vector_store = get_vector_store()
    
    # Use MMR for some diversity even in specific queries
    docs = vector_store.max_marginal_relevance_search(
        query, 
        k=k, 
        fetch_k=k * 3,
        lambda_mult=0.7  # Balance relevance and diversity
    )
    
    if not docs:
        return ""
    
    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source_file", "Unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        context_parts.append(f"[{source} - Section {chunk_idx}]\n{doc.page_content}")
    
    return "\n\n---\n\n".join(context_parts)

def smart_retrieve(query: str, stored_docs: list) -> tuple[str, str]:
    """Choose retrieval strategy based on query type"""
    
    # Check if asking about specific document
    specific_doc = get_specific_doc_name(query, stored_docs)
    
    if is_aggregate_query(query):
        # Summary/overview query - use diverse sampling
        context = retrieve_for_summary(target_doc=specific_doc, sample_size=8)
        strategy = "diverse_sampling"
    else:
        # Specific question - use similarity with MMR
        context = retrieve_similarity(query, k=4)
        strategy = "similarity_mmr"
    
    return context, strategy

# =============================================================================
# OVERVIEW AND RESPONSE GENERATION
# =============================================================================

def generate_overview(personality: dict) -> str:
    """Generate overview using diverse sampling"""
    context = retrieve_for_summary(sample_size=10)
    
    if not context:
        return None
    
    stored_docs, _ = get_stored_documents()
    
    model = get_chat_model()
    
    overview_prompt = f"""You have access to these course materials: {', '.join(stored_docs)}

Here are representative excerpts from throughout the materials:

{context}

Provide a welcome message including:
1. A greeting in character
2. What these materials cover (2-3 sentences)
3. List of key topics you identified
4. A suggested first question

Be concise and stay in character!"""

    messages = [
        SystemMessage(content=personality["system_prompt"]),
        HumanMessage(content=overview_prompt)
    ]
    
    response = model.invoke(messages)
    return response.content

def get_rag_response(query: str, chat_history: list, personality: dict, stored_docs: list) -> str:
    """Generate response with smart retrieval"""
    
    # Smart retrieval based on query type
    context, strategy = smart_retrieve(query, stored_docs)
    
    model = get_chat_model()
    
    base_instruction = """Answer based on course materials. Be helpful and accurate while staying in character.
If context seems insufficient, acknowledge it but provide what help you can."""

    full_system = f"{personality['system_prompt']}\n\n{base_instruction}"

    # Adjust prompt based on retrieval strategy
    if strategy == "diverse_sampling":
        rag_prompt = f"""COURSE MATERIAL EXCERPTS (sampled from throughout the document):
{context}

STUDENT REQUEST: {query}

Provide a comprehensive response covering the main themes from these materials. Stay in character!"""
    else:
        rag_prompt = f"""RELEVANT COURSE MATERIAL:
{context}

STUDENT QUESTION: {query}

Answer based on the materials above. Stay in character!"""

    messages = [SystemMessage(content=full_system)]
    
    # Add recent history (limit to reduce latency)
    for msg in chat_history[-4:]:  # Reduced from 8 to 4
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    
    messages.append(HumanMessage(content=rag_prompt))
    
    response = model.invoke(messages)
    return response.content

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "overview_generated" not in st.session_state:
    st.session_state.overview_generated = False
if "selected_tutor" not in st.session_state:
    st.session_state.selected_tutor = "professor"
if "pending_overview" not in st.session_state:
    st.session_state.pending_overview = False

# =============================================================================
# UI
# =============================================================================

st.title("Course Material QA Bot")
st.markdown("Upload your course materials and learn with your favorite tutor!")

col1, col2 = st.columns([1, 1])

# -----------------------------------------------------------------------------
# LEFT COLUMN: Document Upload
# -----------------------------------------------------------------------------
with col1:
    st.header("Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload course materials (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    with st.expander("Processing Options", expanded=False):
        chunk_size = st.slider("Chunk Size", 500, 2000, CHUNK_SIZE, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, CHUNK_OVERLAP, 50)
    
    if uploaded_files:
        if st.button("Process & Store Documents", type="primary", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            total_chunks = 0
            
            for idx, file in enumerate(uploaded_files):
                status.text(f"Processing: {file.name}")
                suffix = ".pdf" if file.type == "application/pdf" else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    file_type = "pdf" if suffix == ".pdf" else "txt"
                    docs = load_document(tmp_path, file_type)
                    chunks = process_documents(docs)
                    num_chunks = add_to_vector_store(chunks, file.name)
                    total_chunks += num_chunks
                finally:
                    os.unlink(tmp_path)
                
                progress.progress((idx + 1) / len(uploaded_files))
            
            status.empty()
            progress.empty()
            st.success(f"Processed {len(uploaded_files)} file(s) → {total_chunks} chunks!")
            st.session_state.overview_generated = False
            st.session_state.pending_overview = True
            st.rerun()
    
    st.divider()
    st.subheader("Vector Store Status")
    
    stored_docs, total_chunks = get_stored_documents()
    
    col_a, col_b = st.columns(2)
    col_a.metric("Documents", len(stored_docs))
    col_b.metric("Total Chunks", total_chunks)
    
    if stored_docs:
        with st.expander("Stored Documents", expanded=True):
            for doc in stored_docs:
                st.text(f"• {doc}")
        
        if st.button("Clear All Documents", type="secondary"):
            clear_vector_store()
            st.session_state.messages = []
            st.session_state.overview_generated = False
            st.session_state.pending_overview = False
            st.success("Vector store cleared!")
            st.rerun()

# -----------------------------------------------------------------------------
# RIGHT COLUMN: Chat
# -----------------------------------------------------------------------------
with col2:
    st.header("Chat")
    
    current_tutor = TUTOR_PERSONALITIES[st.session_state.selected_tutor]
    
    # Tutor Selection
    with st.expander(f"{current_tutor['emoji']} Tutor: {current_tutor['name']}", expanded=False):
        st.caption(current_tutor['description'])
        st.markdown("**Switch Tutor:**")
        
        tutor_keys = list(TUTOR_PERSONALITIES.keys())
        cols = st.columns(len(tutor_keys))
        
        for i, key in enumerate(tutor_keys):
            tutor = TUTOR_PERSONALITIES[key]
            with cols[i]:
                is_selected = key == st.session_state.selected_tutor
                if st.button(
                    tutor['emoji'],
                    key=f"tutor_{key}",
                    help=f"{tutor['name']}: {tutor['tagline']}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary"
                ):
                    if key != st.session_state.selected_tutor:
                        st.session_state.selected_tutor = key
                        st.session_state.messages = []
                        st.session_state.overview_generated = False
                        st.session_state.pending_overview = True  # Trigger overview
                        st.rerun()
    
    # Generate overview if pending
    if stored_docs and st.session_state.pending_overview and not st.session_state.messages:
        with st.spinner(f"{current_tutor['emoji']} {current_tutor['name']} is reviewing materials..."):
            try:
                overview = generate_overview(current_tutor)
                if overview:
                    st.session_state.messages.append({"role": "assistant", "content": overview})
            except Exception as e:
                fallback = f"{current_tutor['emoji']} Materials ready! {len(stored_docs)} doc(s) loaded. Ask me anything!"
                st.session_state.messages.append({"role": "assistant", "content": fallback})
            
            st.session_state.overview_generated = True
            st.session_state.pending_overview = False
            st.rerun()
    
    if not stored_docs:
        st.info("Upload course materials to get started!")
    
    # Chat Display
    chat_container = st.container(height=350)
    with chat_container:
        for message in st.session_state.messages:
            avatar = current_tutor['emoji'] if message["role"] == "assistant" else "🧑‍🎓"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    
    # Chat Input
    if stored_docs:
        if prompt := st.chat_input(f"Ask {current_tutor['name']}..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner(f"{current_tutor['emoji']} Thinking..."):
                try:
                    response = get_rag_response(
                        prompt, 
                        st.session_state.messages[:-1], 
                        current_tutor,
                        stored_docs
                    )
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
            
            st.rerun()
    else:
        st.chat_input("Upload documents first...", disabled=True)
    
    if st.session_state.messages:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.overview_generated = False
            st.session_state.pending_overview = True  # Regenerate overview
            st.rerun()

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown("""
    **How to use:**
    1. Upload course PDFs/TXTs
    2. Choose your tutor
    3. Ask questions!
    """)
    st.divider()
    st.subheader("Tutors")
    for key, tutor in TUTOR_PERSONALITIES.items():
        st.markdown(f"{tutor['emoji']} **{tutor['name']}** - {tutor['tagline']}")

    #FEATURES to ADD
    # Streaming
    # Sources

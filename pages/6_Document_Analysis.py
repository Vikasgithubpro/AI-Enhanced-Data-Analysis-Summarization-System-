import streamlit as st
import os
import fitz
import pandas as pd
import re
import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import faiss
import tempfile
from utils import GroqChatWrapper

st.set_page_config(page_title="📄 Document Analysis", layout="wide")
st.title("📄 AI Document Analysis & Chatbot")

# Initialize session state
if "document_data" not in st.session_state:
    st.session_state.document_data = ""
if "embedded_chunks" not in st.session_state:
    st.session_state.embedded_chunks = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "faiss_metadata" not in st.session_state:
    st.session_state.faiss_metadata = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# Document Processing Functions
# -------------------------------
def extract_text_from_pdf(file_path):
    """Extract text from PDF files."""
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        st.error(f"❌ Failed to extract PDF: {e}")
    return text

def extract_text_from_excel(file_path):
    """Extract text from Excel files."""
    text = ""
    try:
        dfs = pd.read_excel(file_path, sheet_name=None)
        for sheet_name, df in dfs.items():
            text += f"\n\n--- Sheet: {sheet_name} ---\n"
            text += df.astype(str).to_string(index=False)
    except Exception as e:
        st.error(f"❌ Failed to extract Excel: {e}")
    return text

def extract_text_from_csv(file_path):
    """Extract text from CSV files."""
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="ISO-8859-1")
    return df.astype(str).to_string(index=False)

def clean_text(text):
    """Clean and normalize text."""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"Page \d+|Slide \d+", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text, source_name, chunk_size=500, overlap=100):
    """Split text into chunks for processing."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append({"source": source_name, "chunk": chunk})
    return chunks

# -------------------------------
# AI Embedding and Search
# -------------------------------
@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(_model, chunks):
    """Create embeddings for text chunks."""
    embedded_data = []
    for chunk in chunks:
        emb = _model.encode(chunk["chunk"]).tolist()
        embedded_data.append({
            "embedding": emb,
            "text": chunk["chunk"],
            "source": chunk["source"]
        })
    return embedded_data

def build_faiss_index(embedded_chunks):
    """Build FAISS index for semantic search."""
    if not embedded_chunks:
        return None, []
    
    embedding_dim = len(embedded_chunks[0]["embedding"])
    index = faiss.IndexFlatL2(embedding_dim)
    vectors = np.array([chunk["embedding"] for chunk in embedded_chunks], dtype="float32")
    index.add(vectors)
    metadata = [{"text": chunk["text"], "source": chunk["source"]} for chunk in embedded_chunks]
    return index, metadata

def search_documents(query, _model, _index, _metadata, top_k=3):
    """Search documents using semantic search."""
    if _index is None:
        return []
    
    query_vector = np.array(_model.encode([query]), dtype="float32")
    D, I = _index.search(query_vector, top_k)
    return [_metadata[idx] for idx in I[0]]

# -------------------------------
# Document Analysis Interface
# -------------------------------
st.markdown("## 📁 Upload Documents for Analysis")

# File upload section
uploaded_files = st.file_uploader(
    "Choose documents to analyze",
    type=["pdf", "txt", "csv", "xlsx", "xls"],
    accept_multiple_files=True,
    help="Upload PDFs, Excel files, CSVs, or text documents"
)

col1, col2 = st.columns([2, 1])
with col1:
    process_docs = st.button("🚀 Process Documents", use_container_width=True)
with col2:
    clear_docs = st.button("🗑️ Clear Documents", use_container_width=True)

if clear_docs:
    st.session_state.document_data = ""
    st.session_state.embedded_chunks = []
    st.session_state.faiss_index = None
    st.session_state.faiss_metadata = []
    st.session_state.chat_history = []
    st.rerun()

if process_docs and uploaded_files:
    with st.spinner("🤖 Processing documents..."):
        all_chunks = []
        total_files = len(uploaded_files)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{total_files})")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Extract text based on file type
                file_ext = uploaded_file.name.lower().split('.')[-1]
                text_content = ""
                
                if file_ext == "pdf":
                    text_content = extract_text_from_pdf(tmp_path)
                elif file_ext in ["xlsx", "xls"]:
                    text_content = extract_text_from_excel(tmp_path)
                elif file_ext == "csv":
                    text_content = extract_text_from_csv(tmp_path)
                elif file_ext == "txt":
                    text_content = uploaded_file.getvalue().decode("utf-8")
                
                # Clean and chunk text
                cleaned_text = clean_text(text_content)
                if cleaned_text:
                    chunks = chunk_text(cleaned_text, uploaded_file.name)
                    all_chunks.extend(chunks)
                
                # Clean up temp file
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
            
            progress_bar.progress((i + 1) / total_files)
        
        if all_chunks:
            # Create embeddings and build index
            model = load_embedding_model()
            embedded_data = embed_chunks(model, all_chunks)
            index, metadata = build_faiss_index(embedded_data)
            
            # Store in session state
            st.session_state.embedded_chunks = embedded_data
            st.session_state.faiss_index = index
            st.session_state.faiss_metadata = metadata
            
            st.success(f"✅ Processed {len(uploaded_files)} files, created {len(embedded_data)} chunks")
        else:
            st.error("❌ No text could be extracted from the uploaded files")

# Display document statistics
if st.session_state.embedded_chunks:
    st.markdown("## 📊 Document Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents", len(uploaded_files) if uploaded_files else 0)
    with col2:
        st.metric("Text Chunks", len(st.session_state.embedded_chunks))
    with col3:
        sources = len(set(chunk['source'] for chunk in st.session_state.embedded_chunks))
        st.metric("Unique Sources", sources)
    with col4:
        total_words = sum(len(chunk['text'].split()) for chunk in st.session_state.embedded_chunks)
        st.metric("Total Words", f"{total_words:,}")
    
    # Show chunk preview
    with st.expander("🔍 Preview Document Chunks"):
        for i, chunk in enumerate(st.session_state.embedded_chunks[:5]):
            st.write(f"**Chunk {i+1}** (from: {chunk['source']})")
            st.text(chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'])
            st.markdown("---")

# -------------------------------
# Document Chatbot Interface
# -------------------------------
st.markdown("## 💬 Document Chatbot")

if not st.session_state.embedded_chunks:
    st.info("📝 Please upload and process documents first to enable the chatbot.")
else:
    # Chat interface
    st.markdown("### Ask questions about your documents")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Searching documents..."):
                try:
                    # Search for relevant content
                    model = load_embedding_model()
                    results = search_documents(
                        prompt, model, 
                        st.session_state.faiss_index, 
                        st.session_state.faiss_metadata, 
                        top_k=3
                    )
                    
                    # Combine context
                    combined_context = "\n\n".join([res["text"] for res in results])
                    
                    # Generate answer using Groq
                    if "groq_key" in st.session_state:
                        groq_model = GroqChatWrapper(
                            model="llama-3.3-70b-versatile", 
                            api_key=st.session_state.groq_key
                        )
                        
                        system_prompt = """You are a helpful document analysis assistant. 
                        Answer the user's question based ONLY on the provided document context.
                        If the context doesn't contain relevant information, say so clearly.
                        Be precise and cite sources when possible."""
                        
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Context:\n{combined_context}\n\nQuestion: {prompt}"}
                        ]
                        
                        response = groq_model.run(messages)
                        
                        # Add sources
                        sources = list(set([res["source"] for res in results]))
                        response += f"\n\n📚 **Sources**: {', '.join(sources)}"
                        
                    else:
                        response = "⚠️ Groq API key not found. Please set your API key in the Upload page."
                    
                    st.write(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"❌ Error generating response: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# -------------------------------
# Document Analysis Features
# -------------------------------
if st.session_state.embedded_chunks:
    st.markdown("## 🔍 Advanced Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📈 Summary Analysis", "🔎 Semantic Search", "📋 Content Explorer"])
    
    with tab1:
        st.markdown("### Generate Document Summary")
        if st.button("📊 Generate Summary", key="summary_btn"):
            if "groq_key" in st.session_state:
                with st.spinner("🤖 Generating comprehensive summary..."):
                    try:
                        # Sample chunks for summary
                        sample_chunks = st.session_state.embedded_chunks[:10]
                        combined_content = "\n\n".join([chunk["text"] for chunk in sample_chunks])
                        
                        groq_model = GroqChatWrapper(
                            model="llama-3.3-70b-versatile", 
                            api_key=st.session_state.groq_key
                        )
                        
                        summary_prompt = f"""
                        Analyze the following document content and provide a comprehensive summary:
                        
                        {combined_content[:3000]}  # Limit content length
                        
                        Please provide:
                        1. Main topics and themes
                        2. Key findings or insights
                        3. Important data points
                        4. Overall document purpose
                        
                        Format the response in clear sections.
                        """
                        
                        summary = groq_model.run([{"role": "user", "content": summary_prompt}])
                        st.success("### 📋 Document Summary")
                        st.write(summary)
                        
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
            else:
                st.error("Groq API key required for summary generation")
    
    with tab2:
        st.markdown("### Semantic Search")
        search_query = st.text_input("Enter search query:", placeholder="Search for specific information...")
        if st.button("🔍 Search", key="search_btn") and search_query:
            with st.spinner("Searching documents..."):
                model = load_embedding_model()
                results = search_documents(
                    search_query, model, 
                    st.session_state.faiss_index, 
                    st.session_state.faiss_metadata, 
                    top_k=5
                )
                
                if results:
                    st.success(f"🔍 Found {len(results)} relevant results:")
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i} - From: {result['source']}"):
                            st.write(result['text'])
                            st.caption(f"Source: {result['source']}")
                else:
                    st.info("No relevant results found.")
    
    with tab3:
        st.markdown("### Document Content Explorer")
        if st.session_state.embedded_chunks:
            # Show documents by source
            sources = set(chunk['source'] for chunk in st.session_state.embedded_chunks)
            selected_source = st.selectbox("Select document:", list(sources))
            
            if selected_source:
                source_chunks = [chunk for chunk in st.session_state.embedded_chunks 
                               if chunk['source'] == selected_source]
                
                st.write(f"**Document:** {selected_source}")
                st.write(f"**Chunks:** {len(source_chunks)}")
                
                # Show chunks from this document
                for i, chunk in enumerate(source_chunks[:10]):
                    with st.expander(f"Chunk {i+1}"):
                        st.text(chunk['text'])

# -------------------------------
# Requirements Check
# -------------------------------
st.markdown("---")
with st.expander("🔧 Installation Requirements"):
    st.markdown("""
    **Required packages for document analysis:**
    ```bash
    pip install pymupdf pandas sentence-transformers faiss-cpu requests
    ```
    
    **Features included:**
    - 📄 Multi-format document support (PDF, Excel, CSV, TXT)
    - 🤖 Semantic search with FAISS
    - 💬 AI-powered document chatbot
    - 📊 Document summarization
    - 🔍 Content exploration
    """)
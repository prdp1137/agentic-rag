"""
Streamlit UI for Agentic RAG System.

Run with: streamlit run app.py
"""

import asyncio
import os
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from src.graph.builder import compile_graph_with_persistence
from src.retrieval.qdrant_handler import QdrantHandler
from src.state.agent_state import create_initial_state

load_dotenv()

st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 2rem; color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; }
    .doc-card {
        background: white; padding: 1rem; border-radius: 8px;
        border-left: 4px solid #667eea; margin-bottom: 0.75rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .doc-card .source { font-size: 0.75rem; color: #888; text-transform: uppercase; }
    .doc-card .content { margin-top: 0.5rem; color: #444; line-height: 1.5; }
    .doc-card .score {
        display: inline-block; background: #667eea; color: white;
        padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.75rem; margin-top: 0.5rem;
    }
    .answer-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem; border-radius: 12px; color: white; margin: 1rem 0;
    }
    .answer-box h4 { margin: 0 0 1rem 0; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "thread_id": str(uuid.uuid4()),
        "chat_history": [],
        "app": None,
        "qdrant_handler": None,
        # Settings with defaults
        "llm_model": os.getenv("LLM_MODEL", "gpt-4.1-mini"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "temperature": 0.0,
        "top_k": 5,
        "score_threshold": 0.25,
        "rrf_k": 60,
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "collection_name": os.getenv("QDRANT_COLLECTION_NAME", "enterprise_rag"),
        # Grading strategy
        "grading_strategy": os.getenv("GRADING_STRATEGY", "parallel"),
        # Track if settings changed
        "settings_hash": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


def get_settings_hash():
    """Get hash of current settings to detect changes."""
    return hash((
        st.session_state.qdrant_url,
        st.session_state.collection_name,
        st.session_state.llm_model,
        st.session_state.embedding_model,
    ))


def invalidate_handlers():
    """Invalidate cached handlers when settings change."""
    st.session_state.qdrant_handler = None
    st.session_state.app = None


def run_async(coro):
    """Run async coroutine in sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def get_qdrant_handler_async(force_refresh: bool = False):
    """Get or create Qdrant handler with current settings."""
    if st.session_state.qdrant_handler is None or force_refresh:
        handler = QdrantHandler(
            collection_name=st.session_state.collection_name,
            qdrant_url=st.session_state.qdrant_url,
            embedding_model=st.session_state.embedding_model,
        )
        await handler.initialize()
        st.session_state.qdrant_handler = handler
    return st.session_state.qdrant_handler


async def get_collection_info():
    """Get collection information from Qdrant."""
    try:
        handler = await get_qdrant_handler_async()
        if handler.client is None:
            return None
        
        collections = await handler.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if st.session_state.collection_name in collection_names:
            info = await handler.client.get_collection(st.session_state.collection_name)
            return {
                "name": st.session_state.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "config": info.config,
            }
        return None
    except Exception as e:
        return {"error": str(e)}


async def list_documents(limit: int = 100):
    """List documents from Qdrant collection."""
    try:
        handler = await get_qdrant_handler_async()
        if handler.client is None:
            return []
        
        result = await handler.client.scroll(
            collection_name=st.session_state.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        
        documents = []
        for point in result[0]:
            documents.append({
                "id": str(point.id),
                "content": point.payload.get("content", "")[:500],
                "source": point.payload.get("source", "unknown"),
                "metadata": point.payload.get("metadata", {}),
            })
        return documents
    except Exception as e:
        return [{"error": str(e)}]


async def delete_document(doc_id: str):
    """Delete a document from Qdrant."""
    try:
        handler = await get_qdrant_handler_async()
        if handler.client is None:
            return False
        
        await handler.client.delete(
            collection_name=st.session_state.collection_name,
            points_selector=[doc_id],
        )
        return True
    except Exception:
        return False


async def run_query_async(query: str, thread_id: str):
    """Run a query through the RAG pipeline."""
    # Set grading strategy from session state
    os.environ["GRADING_STRATEGY"] = st.session_state.grading_strategy
    
    if st.session_state.app is None:
        st.session_state.app = compile_graph_with_persistence()
    
    initial_state = create_initial_state(query)
    config = {"configurable": {"thread_id": thread_id}}
    
    result = await st.session_state.app.ainvoke(initial_state, config)
    return result


def render_sidebar():
    """Render the sidebar with settings and info."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        st.markdown("### 🧵 Session")
        col1, col2 = st.columns([3, 1])
        with col1:
            new_thread_id = st.text_input(
                "Thread ID",
                value=st.session_state.thread_id,
                help="UUID for conversation persistence",
            )
            if new_thread_id != st.session_state.thread_id:
                st.session_state.thread_id = new_thread_id
        with col2:
            if st.button("🔄", help="Generate new thread ID"):
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.chat_history = []
                st.rerun()
        
        st.divider()
        
        st.markdown("### 🤖 Model Configuration")
        
        llm_model = st.selectbox(
            "LLM Model",
            options=["gpt-4.1-mini", "gpt-4.1", "gpt-4o", "gpt-4o-mini"],
            index=["gpt-4.1-mini", "gpt-4.1", "gpt-4o", "gpt-4o-mini"].index(st.session_state.llm_model) if st.session_state.llm_model in ["gpt-4.1-mini", "gpt-4.1", "gpt-4o", "gpt-4o-mini"] else 0,
            help="Select the LLM model for generation",
            key="llm_model_select",
        )
        if llm_model != st.session_state.llm_model:
            st.session_state.llm_model = llm_model
            invalidate_handlers()
        
        embedding_model = st.selectbox(
            "Embedding Model",
            options=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
            index=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"].index(st.session_state.embedding_model) if st.session_state.embedding_model in ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"] else 0,
            help="Select the embedding model",
            key="embedding_model_select",
        )
        if embedding_model != st.session_state.embedding_model:
            st.session_state.embedding_model = embedding_model
            invalidate_handlers()
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="LLM temperature for generation",
            key="temperature_slider",
        )
        if temperature != st.session_state.temperature:
            st.session_state.temperature = temperature
        
        st.divider()
        
        st.markdown("### 🔍 Retrieval Settings")
        
        top_k = st.slider(
            "Top K Documents",
            min_value=1,
            max_value=20,
            value=st.session_state.top_k,
            help="Number of documents to retrieve",
            key="top_k_slider",
        )
        if top_k != st.session_state.top_k:
            st.session_state.top_k = top_k
        
        score_threshold = st.slider(
            "Score Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.score_threshold,
            step=0.05,
            help="Minimum relevance score",
            key="score_threshold_slider",
        )
        if score_threshold != st.session_state.score_threshold:
            st.session_state.score_threshold = score_threshold
        
        rrf_k = st.slider(
            "RRF K Constant",
            min_value=1,
            max_value=100,
            value=st.session_state.rrf_k,
            help="Reciprocal Rank Fusion constant",
            key="rrf_k_slider",
        )
        if rrf_k != st.session_state.rrf_k:
            st.session_state.rrf_k = rrf_k
        
        st.divider()
        
        st.markdown("### 📊 Grading Strategy")
        grading_options = {
            "parallel": "⚡ Parallel (fast, 1 call per doc)",
            "batch": "💰 Batch (cheap, 1 call total)",
            "skip": "🚀 Skip (free, score-only)",
        }
        grading_strategy = st.selectbox(
            "Document Grading",
            options=list(grading_options.keys()),
            format_func=lambda x: grading_options[x],
            index=list(grading_options.keys()).index(st.session_state.grading_strategy),
            help="How to grade retrieved documents for relevance",
            key="grading_strategy_select",
        )
        if grading_strategy != st.session_state.grading_strategy:
            st.session_state.grading_strategy = grading_strategy
            os.environ["GRADING_STRATEGY"] = grading_strategy
        
        if grading_strategy == "skip":
            st.info("💡 Skip mode uses retrieval scores only - no LLM cost!")
        elif grading_strategy == "batch":
            st.info("💡 Batch mode grades all docs in 1 LLM call - most cost-effective!")
        
        st.divider()
        
        st.markdown("### 🗄️ Qdrant Connection")
        
        qdrant_url = st.text_input(
            "Qdrant URL",
            value=st.session_state.qdrant_url,
            help="Qdrant server URL",
            key="qdrant_url_input",
        )
        if qdrant_url != st.session_state.qdrant_url:
            st.session_state.qdrant_url = qdrant_url
            invalidate_handlers()
        
        collection_name = st.text_input(
            "Collection Name",
            value=st.session_state.collection_name,
            help="Qdrant collection name",
            key="collection_name_input",
        )
        if collection_name != st.session_state.collection_name:
            st.session_state.collection_name = collection_name
            invalidate_handlers()
        
        # Apply & Refresh button
        if st.button("🔄 Apply & Refresh", use_container_width=True, type="primary"):
            invalidate_handlers()
            st.rerun()
        
        collection_info = run_async(get_collection_info())
        if collection_info and "error" not in collection_info:
            st.success(f"✅ Connected | {collection_info.get('points_count', 0)} docs")
        elif collection_info and "error" in collection_info:
            st.error(f"❌ Error: {collection_info['error'][:50]}...")
        else:
            st.warning("⚠️ Collection not found")
        
        st.divider()
        
        st.markdown("### 🔧 Actions")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            invalidate_handlers()
            st.rerun()


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Agentic RAG System</h1>
        <p>Multi-Agent Retrieval-Augmented Generation with Corrective RAG</p>
    </div>
    """, unsafe_allow_html=True)


def render_metrics():
    """Render metrics cards."""
    collection_info = run_async(get_collection_info())
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📚 Documents",
            value=collection_info.get("points_count", 0) if collection_info and "error" not in collection_info else "N/A",
        )
    
    with col2:
        st.metric(
            label="💬 Messages",
            value=len(st.session_state.chat_history),
        )
    
    with col3:
        st.metric(
            label="🧵 Thread",
            value=st.session_state.thread_id[:8] + "...",
        )
    
    with col4:
        status = "🟢 Online" if collection_info and "error" not in collection_info else "🔴 Offline"
        st.metric(
            label="📡 Status",
            value=status,
        )


def render_chat_interface():
    """Render the chat interface."""
    st.markdown("### 💬 Query Interface")
    
    query = st.text_area(
        "Enter your question",
        placeholder="Ask anything about your documents...",
        height=100,
        key="query_input",
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        submit = st.button("🚀 Submit", type="primary", use_container_width=True)
    with col2:
        clear = st.button("🗑️ Clear", use_container_width=True)
    
    if clear:
        st.session_state.chat_history = []
        st.rerun()
    
    if submit and query.strip():
        with st.spinner("🔄 Processing query..."):
            try:
                result = run_async(run_query_async(query, st.session_state.thread_id))
                
                st.session_state.chat_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "result": result,
                })
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    if st.session_state.chat_history:
        st.markdown("---")
        
        for i, item in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"**🙋 Query:** {item['query']}")
                
                result = item["result"]
                
                st.markdown(f"""
                <div class="answer-box">
                    <h4>📤 Answer</h4>
                    <p>{result.get('generation', 'No answer generated')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📊 Execution Details"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Documents Used", len(result.get("documents", [])))
                    with col2:
                        st.metric("Web Search", result.get("web_search", "No"))
                    with col3:
                        st.metric("Grading Status", result.get("grading_status", "N/A"))
                    with col4:
                        st.metric("Query Rewrites", result.get("query_rewrite_count", 0))
                    
                    if result.get("documents"):
                        st.markdown("#### Retrieved Documents")
                        for j, doc in enumerate(result["documents"]):
                            source = doc.metadata.get("source", "unknown")
                            score = doc.metadata.get("score", "N/A")
                            if isinstance(score, float):
                                score = f"{score:.4f}"
                            
                            st.markdown(f"""
                            <div class="doc-card">
                                <span class="source">📄 {source}</span>
                                <span class="score">Score: {score}</span>
                                <div class="content">{doc.page_content[:300]}...</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("---")


def render_documents_tab():
    """Render the documents management tab."""
    st.markdown("### 📚 Document Management")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_docs"):
            invalidate_handlers()
            st.rerun()
    
    documents = run_async(list_documents(limit=50))
    
    if documents and (len(documents) == 0 or "error" not in documents[0]):
        st.markdown(f"**Total Documents:** {len(documents)}")
        
        search = st.text_input("🔍 Filter documents", placeholder="Search by content or source...", key="doc_search")
        
        for doc in documents:
            if "error" in doc:
                st.error(doc["error"])
                continue
            
            if search:
                if search.lower() not in doc["content"].lower() and search.lower() not in doc["source"].lower():
                    continue
            
            with st.expander(f"📄 {doc['source']} - {doc['id'][:8]}..."):
                st.markdown(f"**ID:** `{doc['id']}`")
                st.markdown(f"**Source:** {doc['source']}")
                st.markdown(f"**Content:**")
                st.text(doc["content"])
                
                if doc.get("metadata"):
                    st.markdown("**Metadata:**")
                    st.json(doc["metadata"])
                
                if st.button(f"🗑️ Delete", key=f"del_{doc['id']}"):
                    if run_async(delete_document(doc["id"])):
                        st.success("Deleted!")
                        invalidate_handlers()
                        st.rerun()
                    else:
                        st.error("Failed to delete")
    else:
        if documents and "error" in documents[0]:
            st.error(documents[0]["error"])
        else:
            st.info("No documents found in the collection.")
    
    st.divider()
    
    st.markdown("### ➕ Add Document")
    with st.form("add_document"):
        doc_content = st.text_area("Document Content", height=150)
        doc_source = st.text_input("Source", value="manual_upload")
        
        if st.form_submit_button("Add Document"):
            if doc_content.strip():
                from langchain_core.documents import Document
                
                async def add_doc():
                    handler = await get_qdrant_handler_async(force_refresh=True)
                    await handler.create_collection_with_binary_quantization()
                    doc = Document(page_content=doc_content, metadata={"source": doc_source})
                    await handler.add_documents([doc])
                
                try:
                    run_async(add_doc())
                    st.success("Document added!")
                    invalidate_handlers()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter document content")


def render_collection_tab():
    """Render the collection info tab."""
    st.markdown("### 🗄️ Collection Information")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_collection"):
            invalidate_handlers()
            st.rerun()
    
    collection_info = run_async(get_collection_info())
    
    if collection_info and "error" not in collection_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### General Info")
            st.json({
                "name": collection_info["name"],
                "vectors_count": collection_info["vectors_count"],
                "points_count": collection_info["points_count"],
                "status": str(collection_info["status"]),
            })
        
        with col2:
            st.markdown("#### Current Settings")
            st.json({
                "qdrant_url": st.session_state.qdrant_url,
                "collection_name": st.session_state.collection_name,
                "llm_model": st.session_state.llm_model,
                "embedding_model": st.session_state.embedding_model,
                "top_k": st.session_state.top_k,
                "rrf_k": st.session_state.rrf_k,
            })
        
        st.markdown("---")
        st.markdown("#### ⚠️ Danger Zone")
        
        with st.expander("Delete Collection", expanded=False):
            st.warning("This will permanently delete the collection and all its data!")
            confirm = st.text_input("Type collection name to confirm:", key="delete_confirm")
            
            if st.button("🗑️ Delete Collection", type="secondary"):
                if confirm == st.session_state.collection_name:
                    async def delete_collection():
                        handler = await get_qdrant_handler_async()
                        await handler.client.delete_collection(st.session_state.collection_name)
                    
                    try:
                        run_async(delete_collection())
                        st.success("Collection deleted!")
                        invalidate_handlers()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.error("Collection name doesn't match")
    
    elif collection_info and "error" in collection_info:
        st.error(f"Error connecting to Qdrant: {collection_info['error']}")
    else:
        st.info("Collection not found. It will be created when you add documents.")
        
        if st.button("➕ Create Collection"):
            async def create_collection():
                handler = await get_qdrant_handler_async(force_refresh=True)
                await handler.create_collection_with_binary_quantization()
            
            try:
                run_async(create_collection())
                st.success("Collection created!")
                invalidate_handlers()
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")


def main():
    """Main application."""
    render_sidebar()
    render_header()
    render_metrics()
    
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📚 Documents", "🗄️ Collection"])
    
    with tab1:
        render_chat_interface()
    
    with tab2:
        render_documents_tab()
    
    with tab3:
        render_collection_tab()


if __name__ == "__main__":
    main()

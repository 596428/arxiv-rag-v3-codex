"""
arXiv RAG v1 - Streamlit Search Demo

Interactive search interface for arXiv LLM papers.
"""

import streamlit as st
import time
from typing import Optional

# Page config must be first Streamlit command
st.set_page_config(
    page_title="arXiv LLM Paper Search",
    page_icon="📚",
    layout="wide",
)


# Lazy imports to avoid slow startup
@st.cache_resource
def get_retriever():
    """Load Qdrant retriever (cached)."""
    from src.rag.qdrant_retriever import QdrantHybridRetriever
    return QdrantHybridRetriever()


@st.cache_resource
def get_reranker():
    """Load reranker (cached)."""
    from src.rag.reranker import BGEReranker
    return BGEReranker()


@st.cache_resource
def get_cached_db_client():
    """Load database client (cached)."""
    from src.storage import get_db_client as build_db_client
    return build_db_client()


def format_score(score: Optional[float]) -> str:
    """Format score for display."""
    if score is None:
        return "-"
    return f"{score:.3f}"


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def main():
    """Main Streamlit app."""

    # Header
    st.title("📚 arXiv LLM Paper Search")
    st.markdown(
        "Hybrid search over 2025 LLM papers using **BGE-M3** embeddings "
        "with **RRF fusion** and cross-encoder reranking."
    )

    # Sidebar - Settings
    with st.sidebar:
        st.header("⚙️ Settings")

        search_mode = st.selectbox(
            "Search Mode",
            options=["adaptive", "qdrant_hybrid", "dense", "sparse"],
            index=0,
            help="Adaptive uses query-aware Qdrant retrieval; hybrid combines dense and sparse search"
        )

        top_k = st.slider(
            "Initial Results",
            min_value=5,
            max_value=50,
            value=20,
            help="Number of results before reranking"
        )

        use_reranker = st.checkbox(
            "Use Reranker",
            value=True,
            help="Apply cross-encoder reranking for better precision"
        )

        if use_reranker:
            rerank_top_k = st.slider(
                "Final Results",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of results after reranking"
            )
        else:
            rerank_top_k = top_k

        st.divider()

        # Database stats
        st.header("📊 Database Stats")
        try:
            client = get_cached_db_client()
            stats = client.get_collection_stats()
            chunk_count = client.get_chunk_count()

            col1, col2 = st.columns(2)
            col1.metric("Papers", stats.get("total", 0))
            col2.metric("Embedded", stats.get("embedded", 0))
            st.metric("Chunks", f"{chunk_count:,}")
        except Exception as e:
            st.error(f"Failed to load stats: {e}")

    # Main content - Search
    query = st.text_input(
        "🔍 Search Query",
        placeholder="e.g., What is RLHF and how does it work?",
        help="Enter your search query about LLM research"
    )

    # Example queries
    with st.expander("💡 Example Queries"):
        examples = [
            "What is reinforcement learning from human feedback?",
            "How does chain-of-thought prompting improve reasoning?",
            "What are the scaling laws for large language models?",
            "Explain instruction tuning and its benefits",
            "What is retrieval-augmented generation (RAG)?",
        ]
        for example in examples:
            if st.button(example, key=f"ex_{example[:20]}"):
                st.session_state["query"] = example
                st.rerun()

    # Handle example query selection
    if "query" in st.session_state and st.session_state["query"]:
        query = st.session_state["query"]
        st.session_state["query"] = ""  # Reset

    # Search button
    if st.button("🔎 Search", type="primary") or query:
        if not query:
            st.warning("Please enter a search query.")
            return

        with st.spinner("Searching..."):
            start_time = time.time()

            try:
                # Get retriever
                retriever = get_retriever()

                # Perform search
                if search_mode == "dense":
                    response = retriever.search_dense_only(query, top_k=top_k)
                elif search_mode == "sparse":
                    response = retriever.search_sparse_only(query, top_k=top_k)
                elif search_mode == "adaptive":
                    response = retriever.search_adaptive(query, top_k=top_k, use_reranker=False)
                else:
                    response = retriever.search(query, top_k=top_k, use_reranker=False)

                results = response.results

                # Apply reranking
                if use_reranker and len(results) > rerank_top_k:
                    reranker = get_reranker()
                    results = reranker.rerank(query, results, top_k=rerank_top_k)

                elapsed = time.time() - start_time

                # Display results
                st.success(
                    f"Found {len(results)} results in {elapsed:.2f}s "
                    f"({response.search_time_ms:.0f}ms search + reranking)"
                )

                if not results:
                    st.info("No results found. Try a different query.")
                    return

                # Results display
                for i, result in enumerate(results):
                    with st.container():
                        # Header with paper ID and scores
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

                        with col1:
                            arxiv_link = f"https://arxiv.org/abs/{result.paper_id}"
                            st.markdown(f"**{i+1}. [{result.paper_id}]({arxiv_link})**")

                        with col2:
                            st.caption(f"Score: {format_score(result.score)}")

                        with col3:
                            if result.dense_score:
                                st.caption(f"Dense: {format_score(result.dense_score)}")

                        with col4:
                            reranker_score = result.metadata.get("reranker_score")
                            if reranker_score:
                                st.caption(f"Rerank: {format_score(reranker_score)}")

                        # Section title
                        if result.section_title:
                            st.markdown(f"*Section: {result.section_title}*")

                        # Content
                        st.markdown(truncate_text(result.content, 600))

                        # Expand for full content
                        with st.expander("Show full content"):
                            st.text(result.content)

                        st.divider()

            except Exception as e:
                st.error(f"Search failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "arXiv RAG v1 | Hybrid Search Demo | 2025 LLM Papers"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

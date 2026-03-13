"""
arXiv RAG v1 - Chat API Routes

RAG-powered chat endpoint using Qdrant + Gemini/OpenAI.
"""

import os
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...embedding.bge_embedder import BGEEmbedder
from ...embedding.openai_embedder import OpenAIEmbedder
from ...embedding.models import EmbeddingConfig
from ...storage import get_db_client
from ...storage.qdrant_client import get_qdrant_client
from ...utils.config import settings
from ...utils.logging import get_logger

logger = get_logger("api.chat")

router = APIRouter()


class ChatMessage(BaseModel):
    """Chat message."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    """Chat request payload."""
    query: str = Field(..., min_length=1, max_length=4000, description="User question")
    search_mode: str = Field(default="hybrid", description="Search mode for retrieval")
    embedding_model: str = Field(default="openai", description="Embedding model (bge/openai)")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of sources to retrieve")
    history: list[ChatMessage] = Field(default=[], description="Conversation history")
    stream: bool = Field(default=False, description="Stream response")


class ChatSource(BaseModel):
    """Source document for chat response."""
    paper_id: str
    title: str
    content: str
    section_title: Optional[str] = None
    similarity: float


class ChatResponse(BaseModel):
    """Chat response."""
    answer: str
    sources: list[ChatSource]
    metrics: dict


# Lazy-loaded embedders
_bge_embedder: Optional[BGEEmbedder] = None
_openai_embedder: Optional[OpenAIEmbedder] = None


def get_bge_embedder() -> BGEEmbedder:
    """Get or create BGE embedder."""
    global _bge_embedder
    if _bge_embedder is None:
        _bge_embedder = BGEEmbedder(EmbeddingConfig(use_openai=False))
    return _bge_embedder


def get_openai_embedder() -> OpenAIEmbedder:
    """Get or create OpenAI embedder."""
    global _openai_embedder
    if _openai_embedder is None:
        _openai_embedder = OpenAIEmbedder(EmbeddingConfig(use_openai=True))
    return _openai_embedder


def build_context(sources: list[dict], max_tokens: int = 4000) -> str:
    """Build context string from sources."""
    context_parts = []
    estimated_tokens = 0

    for i, source in enumerate(sources):
        content = source.get("content", "")
        paper_id = source.get("paper_id", "unknown")
        section = source.get("section_title", "")

        # Estimate tokens (rough: 1 token ~= 4 chars)
        part_tokens = len(content) // 4

        if estimated_tokens + part_tokens > max_tokens:
            break

        header = f"[{i+1}] Paper: {paper_id}"
        if section:
            header += f" | Section: {section}"

        context_parts.append(f"{header}\n{content}\n")
        estimated_tokens += part_tokens

    return "\n---\n".join(context_parts)


def build_prompt(query: str, context: str, history: list[ChatMessage]) -> str:
    """Build the chat prompt."""
    system_prompt = """You are an AI research assistant specialized in Large Language Models (LLMs) and deep learning.
You answer questions based on the provided research paper excerpts.

Guidelines:
- Cite sources using [1], [2], etc. notation
- Be concise but comprehensive
- If the context doesn't contain relevant information, say so
- Focus on technical accuracy
- Highlight key findings and methodologies"""

    # Format history
    history_text = ""
    if history:
        history_text = "\n\nPrevious conversation:\n"
        for msg in history[-6:]:  # Last 6 messages
            role = "User" if msg.role == "user" else "Assistant"
            history_text += f"{role}: {msg.content}\n"

    prompt = f"""{system_prompt}

Context from research papers:
{context}
{history_text}
User Question: {query}

Please provide a comprehensive answer based on the context above:"""

    return prompt


async def generate_with_gemini(prompt: str, stream: bool = False):
    """Generate response using Gemini."""
    try:
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(settings.gemini_model)

        if stream:
            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        else:
            response = model.generate_content(prompt)
            yield response.text

    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        raise


async def generate_with_openai(prompt: str, stream: bool = False):
    """Generate response using OpenAI."""
    try:
        from openai import OpenAI

        client = OpenAI()

        messages = [{"role": "user", "content": prompt}]

        if stream:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                stream=True,
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            yield response.choices[0].message.content

    except Exception as e:
        logger.error(f"OpenAI generation failed: {e}")
        raise


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    RAG-powered chat endpoint.

    Retrieves relevant paper chunks and generates an answer using LLM.

    Args:
        request: Chat request with query and options

    Returns:
        Generated answer with sources
    """
    start_time = time.time()
    metrics = {
        "retrieval_time_ms": 0,
        "generation_time_ms": 0,
        "total_time_ms": 0,
        "chunks_found": 0,
    }

    try:
        # Step 1: Retrieve relevant chunks
        retrieval_start = time.time()

        qdrant = get_qdrant_client()
        supabase = get_db_client()

        # Get embeddings based on model choice
        if request.embedding_model == "openai":
            embedder = get_openai_embedder()
            dense_vec = embedder.embed_single(request.query)
            vector_name = "dense_3large"
            sparse_indices, sparse_values = None, None
        else:
            embedder = get_bge_embedder()
            dense_vec, sparse_vec, _ = embedder.embed_single(request.query)
            vector_name = "dense_bge"
            sparse_indices = sparse_vec.indices if sparse_vec else None
            sparse_values = sparse_vec.values if sparse_vec else None

        # Perform search (fetch more to allow deduplication by paper)
        search_top_k = request.top_k * 3  # Fetch 3x to ensure enough unique papers
        if request.search_mode == "hybrid" and sparse_indices:
            raw_results = qdrant.search_hybrid(
                dense_vector=dense_vec,
                sparse_indices=sparse_indices,
                sparse_values=sparse_values,
                top_k=search_top_k,
            )
        else:
            raw_results = qdrant.search_dense(
                query_vector=dense_vec,
                vector_name=vector_name,
                top_k=search_top_k,
            )

        metrics["retrieval_time_ms"] = round((time.time() - retrieval_start) * 1000, 1)
        metrics["chunks_found"] = len(raw_results)

        # Step 2: Enrich with paper titles (deduplicate by paper_id)
        sources = []
        deduped_results = []  # For context building with correct indices
        paper_cache = {}
        seen_papers = set()

        for result in raw_results:
            paper_id = result["paper_id"]

            # Skip duplicate papers (keep only first/best chunk per paper)
            if paper_id in seen_papers:
                continue
            seen_papers.add(paper_id)

            # Get paper title (cached)
            if paper_id not in paper_cache:
                paper = supabase.get_paper(paper_id)
                paper_cache[paper_id] = paper.get("title", paper_id) if paper else paper_id

            sources.append(ChatSource(
                paper_id=paper_id,
                title=paper_cache[paper_id],
                content=result["content"][:500],  # Truncate for response
                section_title=result.get("section_title"),
                similarity=round(result.get("similarity", result.get("score", 0)), 3),
            ))
            deduped_results.append(result)  # Keep for context

            # Stop after collecting enough unique papers
            if len(sources) >= request.top_k:
                break

        # Step 3: Build context and prompt (use deduped results for correct [1], [2] indices)
        context = build_context(deduped_results)
        prompt = build_prompt(request.query, context, request.history)

        # Step 4: Generate response
        generation_start = time.time()

        # Try Gemini first, fall back to OpenAI
        answer = ""
        try:
            async for chunk in generate_with_gemini(prompt, stream=False):
                answer += chunk
        except Exception as e:
            logger.warning(f"Gemini failed, trying OpenAI: {e}")
            async for chunk in generate_with_openai(prompt, stream=False):
                answer += chunk

        metrics["generation_time_ms"] = round((time.time() - generation_start) * 1000, 1)
        metrics["total_time_ms"] = round((time.time() - start_time) * 1000, 1)

        return ChatResponse(
            answer=answer,
            sources=sources,
            metrics=metrics,
        )

    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming RAG chat endpoint.

    Same as /chat but streams the response.
    """
    try:
        # Retrieve chunks (same as non-streaming)
        qdrant = get_qdrant_client()

        if request.embedding_model == "openai":
            embedder = get_openai_embedder()
            dense_vec = embedder.embed_single(request.query)
            vector_name = "dense_3large"
            sparse_indices, sparse_values = None, None
        else:
            embedder = get_bge_embedder()
            dense_vec, sparse_vec, _ = embedder.embed_single(request.query)
            vector_name = "dense_bge"
            sparse_indices = sparse_vec.indices if sparse_vec else None
            sparse_values = sparse_vec.values if sparse_vec else None

        if request.search_mode == "hybrid" and sparse_indices:
            raw_results = qdrant.search_hybrid(
                dense_vector=dense_vec,
                sparse_indices=sparse_indices,
                sparse_values=sparse_values,
                top_k=request.top_k,
            )
        else:
            raw_results = qdrant.search_dense(
                query_vector=dense_vec,
                vector_name=vector_name,
                top_k=request.top_k,
            )

        # Build context and prompt
        context = build_context(raw_results)
        prompt = build_prompt(request.query, context, request.history)

        # Stream generator
        async def generate():
            try:
                async for chunk in generate_with_gemini(prompt, stream=True):
                    yield f"data: {chunk}\n\n"
            except Exception:
                async for chunk in generate_with_openai(prompt, stream=True):
                    yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )

    except Exception as e:
        logger.error(f"Stream chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

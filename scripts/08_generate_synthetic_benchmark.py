#!/usr/bin/env python3
"""
arXiv RAG v2 - Enhanced Benchmark Query Generation

Generates diverse search queries with 4 styles, hard negatives, and difficulty labels.

Key improvements over v1:
- 4 query styles: keyword, natural_short, natural_long, conceptual
- Hard negative mining via Qdrant embeddings
- Optional intro/conclusion section support
- Difficulty heuristics (easy/medium/hard)

Usage:
    python scripts/08_generate_synthetic_benchmark.py
    python scripts/08_generate_synthetic_benchmark.py --limit 100 --use-sections --hard-negatives 3
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.storage import get_db_client
from src.storage.qdrant_client import get_qdrant_client
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger("benchmark_v2")


# =============================================================================
# Query Generation Prompt (v2)
# =============================================================================

QUERY_GENERATION_PROMPT_V2 = """Generate diverse search queries for an academic paper retrieval benchmark.

Paper:
{paper_context}

Generate exactly 4 queries with these REQUIRED styles:

1. KEYWORD: Search engine style with 4-7 technical terms from the paper.
   - MUST include the main method/model name if mentioned (e.g., "DeepSeek-R1", "RLHF", "GPT-4")
   - Use actual technical terminology from the paper
   - Example: "DeepSeek-R1 GRPO reinforcement learning reasoning LLM"

2. NATURAL_SHORT: Simple question, 6-12 words.
   - Can reference the main topic directly
   - Example: "How does DeepSeek-R1 improve reasoning with RL?"

3. NATURAL_LONG: Detailed research question, 15-25 words.
   - Should capture the paper's main contribution
   - Can use some terminology from the paper
   - Example: "What techniques does DeepSeek-R1 use to develop reasoning capabilities through reinforcement learning without human demonstrations?"

4. CONCEPTUAL: Fully paraphrased using synonyms and different terminology.
   - Do NOT use any specific names, acronyms, or technical terms from the paper
   - Describe the concept in general terms
   - Example: "Training language models to think step-by-step using reward signals instead of human examples"

Output JSON (no markdown):
{{"queries": [
  {{"text": "...", "style": "keyword"}},
  {{"text": "...", "style": "natural_short"}},
  {{"text": "...", "style": "natural_long"}},
  {{"text": "...", "style": "conceptual"}}
]}}"""


VALID_STYLES = {"keyword", "natural_short", "natural_long", "conceptual"}


# =============================================================================
# Batch Query Generation Prompt (2 papers per request)
# =============================================================================

BATCH_QUERY_GENERATION_PROMPT = """Generate diverse search queries for an academic paper retrieval benchmark.

You must generate queries for EACH paper separately. Do not mix queries between papers.

=== PAPER 1 ===
Title: {title_1}
Abstract: {abstract_1}

=== PAPER 2 ===
Title: {title_2}
Abstract: {abstract_2}

For EACH paper, generate exactly 4 queries with these styles:

1. KEYWORD: Search engine style with 4-7 technical terms from the paper.
   - MUST include the main method/model name if mentioned
   - Use actual technical terminology from the paper

2. NATURAL_SHORT: Simple question, 6-12 words.
   - Can reference the main topic directly

3. NATURAL_LONG: Detailed research question, 15-25 words.
   - Should capture the paper's main contribution

4. CONCEPTUAL: Fully paraphrased using synonyms and different terminology.
   - Do NOT use any specific names, acronyms, or technical terms from the paper
   - Describe the concept in general terms

Output JSON (no markdown):
{{
  "paper_1": {{
    "queries": [
      {{"text": "...", "style": "keyword"}},
      {{"text": "...", "style": "natural_short"}},
      {{"text": "...", "style": "natural_long"}},
      {{"text": "...", "style": "conceptual"}}
    ]
  }},
  "paper_2": {{
    "queries": [
      {{"text": "...", "style": "keyword"}},
      {{"text": "...", "style": "natural_short"}},
      {{"text": "...", "style": "natural_long"}},
      {{"text": "...", "style": "conceptual"}}
    ]
  }}
}}"""


# =============================================================================
# Section Extraction
# =============================================================================

def get_section_excerpt(parsed_data: dict, section_name: str, max_chars: int = 500) -> Optional[str]:
    """
    Extract excerpt from a section (introduction/conclusion).

    Args:
        parsed_data: Parsed paper JSON
        section_name: Section to find (case-insensitive)
        max_chars: Maximum characters to extract

    Returns:
        Section excerpt or None if not found
    """
    sections = parsed_data.get("sections", [])

    for section in sections:
        title = section.get("title", "").lower()
        if section_name.lower() in title:
            # Combine paragraph contents
            paragraphs = section.get("paragraphs", [])
            content_parts = []
            total_chars = 0

            for para in paragraphs:
                text = para.get("content", "")
                if total_chars + len(text) > max_chars:
                    # Truncate at word boundary
                    remaining = max_chars - total_chars
                    if remaining > 50:  # Only add if meaningful
                        truncated = text[:remaining].rsplit(" ", 1)[0] + "..."
                        content_parts.append(truncated)
                    break
                content_parts.append(text)
                total_chars += len(text) + 1  # +1 for space

            if content_parts:
                return " ".join(content_parts)

    return None


def load_parsed_json(parsed_path: str) -> Optional[dict]:
    """Load parsed JSON file for a paper."""
    try:
        path = Path(parsed_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.debug(f"Could not load parsed JSON {parsed_path}: {e}")
    return None


def get_paper_context(paper: dict, use_sections: bool = False, parsed_dir: Path = None) -> str:
    """
    Build paper context for query generation.

    Args:
        paper: Paper metadata dict
        use_sections: Whether to include intro/conclusion
        parsed_dir: Directory containing parsed JSON files

    Returns:
        Formatted context string
    """
    context = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"

    if use_sections and parsed_dir:
        arxiv_id = paper.get("arxiv_id", "")
        parsed_path = parsed_dir / f"{arxiv_id}.json"

        parsed_data = load_parsed_json(str(parsed_path))
        if parsed_data:
            intro = get_section_excerpt(parsed_data, "introduction", max_chars=500)
            conclusion = get_section_excerpt(parsed_data, "conclusion", max_chars=500)

            if intro:
                context += f"\n\nIntroduction excerpt: {intro}"
            if conclusion:
                context += f"\n\nConclusion excerpt: {conclusion}"

    return context


# =============================================================================
# Hard Negative Mining & Similar Paper Discovery
# =============================================================================

def get_hard_negatives_from_qdrant(
    paper_id: str,
    abstract: str,
    qdrant_client,
    embedder,
    top_k: int = 3,
) -> list[str]:
    """
    Find hard negatives using Qdrant dense search.

    Searches for papers with similar abstracts but different paper_id.

    Args:
        paper_id: Source paper ID to exclude
        abstract: Paper abstract to use as query
        qdrant_client: Qdrant client instance
        embedder: BGE embedder for query encoding
        top_k: Number of hard negatives to return

    Returns:
        List of paper IDs that are semantically similar
    """
    try:
        # Embed the abstract
        dense_vec, _, _ = embedder.embed_single(abstract[:1000])

        # Search for similar chunks
        results = qdrant_client.search_dense(
            query_vector=dense_vec,
            vector_name="dense_bge",
            top_k=top_k * 5,  # Get more to filter
        )

        # Extract unique paper IDs (excluding source)
        seen_papers = set()
        hard_negatives = []

        for result in results:
            result_paper_id = result.get("paper_id", "")
            if result_paper_id and result_paper_id != paper_id and result_paper_id not in seen_papers:
                seen_papers.add(result_paper_id)
                hard_negatives.append(result_paper_id)
                if len(hard_negatives) >= top_k:
                    break

        return hard_negatives

    except Exception as e:
        logger.warning(f"Hard negative mining failed for {paper_id}: {e}")
        return []


def get_similar_papers_for_ground_truth(
    paper_id: str,
    abstract: str,
    qdrant_client,
    embedder,
    similarity_threshold: float = 0.85,
    max_papers: int = 3,
) -> list[str]:
    """
    Find highly similar papers to include in multi-paper ground truth.

    Only includes papers with very high similarity (>threshold) that could
    reasonably answer the same query.

    Args:
        paper_id: Source paper ID to exclude
        abstract: Paper abstract to use as query
        qdrant_client: Qdrant client instance
        embedder: BGE embedder for query encoding
        similarity_threshold: Minimum similarity score (0-1) to include
        max_papers: Maximum number of similar papers to return

    Returns:
        List of highly similar paper IDs
    """
    try:
        # Embed the abstract
        dense_vec, _, _ = embedder.embed_single(abstract[:1000])

        # Search for similar chunks with scores
        results = qdrant_client.search_dense(
            query_vector=dense_vec,
            vector_name="dense_bge",
            top_k=max_papers * 10,  # Get more to filter by similarity
            with_payload=True,
            score_threshold=similarity_threshold,
        )

        # Extract unique paper IDs (excluding source) above threshold
        seen_papers = set()
        similar_papers = []

        for result in results:
            result_paper_id = result.get("paper_id", "")
            score = result.get("score", 0)

            if result_paper_id and result_paper_id != paper_id and result_paper_id not in seen_papers:
                if score >= similarity_threshold:
                    seen_papers.add(result_paper_id)
                    similar_papers.append(result_paper_id)
                    if len(similar_papers) >= max_papers:
                        break

        return similar_papers

    except Exception as e:
        logger.debug(f"Similar paper discovery failed for {paper_id}: {e}")
        return []


# =============================================================================
# Difficulty Estimation
# =============================================================================

def estimate_difficulty(query: str, style: str, paper: dict) -> str:
    """
    Estimate query difficulty using simple heuristics (word overlap).

    Args:
        query: Generated query text
        style: Query style (keyword, natural_short, etc.)
        paper: Source paper dict

    Returns:
        Difficulty level: easy, medium, or hard
    """
    query_words = set(query.lower().split())
    title_words = set(paper.get("title", "").lower().split())

    # Get first 100 words of abstract
    abstract = paper.get("abstract", "")
    abstract_words = set(abstract.lower().split()[:100])

    # Calculate word overlap ratio
    combined_paper_words = title_words | abstract_words
    if not query_words:
        return "medium"

    overlap = len(query_words & combined_paper_words) / len(query_words)

    # Difficulty rules
    if style == "keyword" or overlap > 0.5:
        return "easy"
    elif style == "conceptual" or overlap < 0.2:
        return "hard"
    else:
        return "medium"


def estimate_difficulty_v2(
    query: str,
    style: str,
    paper: dict,
    embedder=None,
    query_embedding: list[float] = None,
    abstract_embedding: list[float] = None,
) -> tuple[str, float]:
    """
    Estimate query difficulty using embedding similarity (v2).

    Uses cosine similarity between query and abstract embeddings
    instead of word overlap for more accurate difficulty estimation.

    Args:
        query: Generated query text
        style: Query style (keyword, natural_short, etc.)
        paper: Source paper dict
        embedder: BGE embedder instance (optional)
        query_embedding: Pre-computed query embedding (optional)
        abstract_embedding: Pre-computed abstract embedding (optional)

    Returns:
        Tuple of (difficulty level, similarity score)
    """
    import numpy as np

    # If embedder available, compute embeddings
    similarity = None

    if embedder and query_embedding is None:
        try:
            query_vec, _, _ = embedder.embed_single(query)
            abstract = paper.get("abstract", "")[:1000]
            abstract_vec, _, _ = embedder.embed_single(abstract)

            # Cosine similarity
            query_np = np.array(query_vec)
            abstract_np = np.array(abstract_vec)
            similarity = float(np.dot(query_np, abstract_np) / (
                np.linalg.norm(query_np) * np.linalg.norm(abstract_np)
            ))
        except Exception:
            pass

    elif query_embedding is not None and abstract_embedding is not None:
        try:
            query_np = np.array(query_embedding)
            abstract_np = np.array(abstract_embedding)
            similarity = float(np.dot(query_np, abstract_np) / (
                np.linalg.norm(query_np) * np.linalg.norm(abstract_np)
            ))
        except Exception:
            pass

    # If embedding similarity available, use it
    if similarity is not None:
        if similarity > 0.7:
            return "easy", similarity
        elif similarity > 0.4:
            return "medium", similarity
        else:
            return "hard", similarity

    # Fallback to word-overlap heuristic
    difficulty = estimate_difficulty(query, style, paper)
    return difficulty, -1.0  # -1 indicates no embedding similarity computed


# =============================================================================
# Query Generator
# =============================================================================

class BenchmarkGeneratorV2:
    """Enhanced benchmark generator with 4 query styles and hard negatives."""

    def __init__(
        self,
        model_name: str = None,
        use_sections: bool = False,
        hard_negatives_k: int = 3,
        parsed_dir: Path = None,
        multi_paper_gt: bool = False,
        similar_paper_threshold: float = 0.85,
        max_similar_papers: int = 3,
        use_embedding_difficulty: bool = False,
    ):
        # Gemini setup
        self.api_key = settings.gemini_api_key
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")

        genai.configure(api_key=self.api_key)
        self.model_name = model_name or settings.gemini_model
        self.model = genai.GenerativeModel(self.model_name)

        # Options
        self.use_sections = use_sections
        self.hard_negatives_k = hard_negatives_k
        self.parsed_dir = parsed_dir or Path("data/parsed")

        # Multi-paper ground truth options (Phase 4.1)
        self.multi_paper_gt = multi_paper_gt
        self.similar_paper_threshold = similar_paper_threshold
        self.max_similar_papers = max_similar_papers

        # Embedding-based difficulty (Phase 4.2)
        self.use_embedding_difficulty = use_embedding_difficulty

        # Qdrant and embedder (lazy init)
        self._qdrant_client = None
        self._embedder = None

        logger.info(f"Initialized BenchmarkGeneratorV2: model={self.model_name}, "
                   f"sections={use_sections}, hard_negatives={hard_negatives_k}, "
                   f"multi_paper_gt={multi_paper_gt}, embedding_difficulty={use_embedding_difficulty}")

    @property
    def qdrant_client(self):
        """Lazy init Qdrant client."""
        if self._qdrant_client is None and self.hard_negatives_k > 0:
            try:
                self._qdrant_client = get_qdrant_client()
                if not self._qdrant_client.health_check():
                    logger.warning("Qdrant not available - disabling hard negatives")
                    self._qdrant_client = None
            except Exception as e:
                logger.warning(f"Could not connect to Qdrant: {e}")
                self._qdrant_client = None
        return self._qdrant_client

    @property
    def embedder(self):
        """Lazy init BGE embedder."""
        if self._embedder is None and self.hard_negatives_k > 0:
            try:
                from src.embedding.bge_embedder import BGEEmbedder
                self._embedder = BGEEmbedder()
                logger.info("BGE embedder loaded for hard negative mining")
            except Exception as e:
                logger.warning(f"Could not load BGE embedder: {e}")
                self._embedder = None
        return self._embedder

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate_queries(self, paper: dict) -> list[dict]:
        """
        Generate 4 diverse queries for a paper.

        Args:
            paper: Paper dict with title, abstract, etc.

        Returns:
            List of query dicts with text and style
        """
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        if not abstract or len(abstract.strip()) < 50:
            logger.warning(f"Abstract too short for: {title[:50]}...")
            return []

        # Build context
        context = get_paper_context(
            paper,
            use_sections=self.use_sections,
            parsed_dir=self.parsed_dir,
        )

        prompt = QUERY_GENERATION_PROMPT_V2.format(paper_context=context[:4000])

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()

            # Clean up response - remove markdown code blocks
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            # Parse JSON
            data = json.loads(text)
            queries = data.get("queries", [])

            # Validate queries
            valid_queries = []
            for q in queries:
                if not isinstance(q, dict):
                    continue

                query_text = q.get("text", "")
                style = q.get("style", "").lower()

                # Validate
                if not query_text or len(query_text) < 5:
                    continue
                if style not in VALID_STYLES:
                    continue
                if len(query_text) > 300:
                    query_text = query_text[:300]

                valid_queries.append({
                    "text": query_text.strip(),
                    "style": style,
                })

            return valid_queries

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error for {title[:30]}: {e}")
            return []
        except Exception as e:
            logger.error(f"Generation failed for {title[:30]}: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate_queries_batch(self, paper1: dict, paper2: dict) -> dict[str, list[dict]]:
        """
        Generate queries for 2 papers in a single API call.

        Args:
            paper1: First paper dict
            paper2: Second paper dict

        Returns:
            Dict mapping "paper_1" and "paper_2" to their query lists
        """
        title1 = paper1.get("title", "")
        abstract1 = paper1.get("abstract", "")[:2000]  # Truncate for batch
        title2 = paper2.get("title", "")
        abstract2 = paper2.get("abstract", "")[:2000]

        # Validate both papers have content
        if len(abstract1.strip()) < 50 or len(abstract2.strip()) < 50:
            logger.warning("One or both abstracts too short for batch processing")
            return {"paper_1": [], "paper_2": []}

        prompt = BATCH_QUERY_GENERATION_PROMPT.format(
            title_1=title1,
            abstract_1=abstract1,
            title_2=title2,
            abstract_2=abstract2,
        )

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()

            # Clean up response - remove markdown code blocks
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            # Parse JSON
            data = json.loads(text)

            result = {"paper_1": [], "paper_2": []}

            # Process each paper's queries
            for paper_key in ["paper_1", "paper_2"]:
                paper_data = data.get(paper_key, {})
                queries = paper_data.get("queries", [])

                valid_queries = []
                for q in queries:
                    if not isinstance(q, dict):
                        continue

                    query_text = q.get("text", "")
                    style = q.get("style", "").lower()

                    if not query_text or len(query_text) < 5:
                        continue
                    if style not in VALID_STYLES:
                        continue
                    if len(query_text) > 300:
                        query_text = query_text[:300]

                    valid_queries.append({
                        "text": query_text.strip(),
                        "style": style,
                    })

                result[paper_key] = valid_queries

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error in batch: {e}")
            return {"paper_1": [], "paper_2": []}
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            raise

    def process_paper(self, paper: dict) -> list[dict]:
        """
        Process a single paper: generate queries + hard negatives + difficulty.

        Args:
            paper: Paper dict

        Returns:
            List of benchmark query dicts
        """
        arxiv_id = paper.get("arxiv_id", "")
        categories = paper.get("categories", [])
        category = categories[0] if categories else "unknown"

        # Generate queries
        queries = self.generate_queries(paper)
        if not queries:
            return []

        # Get hard negatives (once per paper)
        hard_negatives = []
        if self.hard_negatives_k > 0 and self.qdrant_client and self.embedder:
            hard_negatives = get_hard_negatives_from_qdrant(
                paper_id=arxiv_id,
                abstract=paper.get("abstract", ""),
                qdrant_client=self.qdrant_client,
                embedder=self.embedder,
                top_k=self.hard_negatives_k,
            )

        # Get similar papers for multi-paper ground truth (Phase 4.1)
        similar_papers = []
        if self.multi_paper_gt and self.qdrant_client and self.embedder:
            similar_papers = get_similar_papers_for_ground_truth(
                paper_id=arxiv_id,
                abstract=paper.get("abstract", ""),
                qdrant_client=self.qdrant_client,
                embedder=self.embedder,
                similarity_threshold=self.similar_paper_threshold,
                max_papers=self.max_similar_papers,
            )

        # Build relevant papers list (source + similar)
        relevant_papers = [arxiv_id] + similar_papers

        # Pre-compute abstract embedding for difficulty estimation (if using v2)
        abstract_embedding = None
        if self.use_embedding_difficulty and self.embedder:
            try:
                abstract_embedding, _, _ = self.embedder.embed_single(
                    paper.get("abstract", "")[:1000]
                )
            except Exception:
                pass

        # Build benchmark entries
        benchmark_queries = []
        for q in queries:
            # Use embedding-based difficulty if enabled (Phase 4.2)
            if self.use_embedding_difficulty and self.embedder:
                difficulty, similarity_score = estimate_difficulty_v2(
                    q["text"], q["style"], paper,
                    embedder=self.embedder,
                    abstract_embedding=abstract_embedding,
                )
                metadata = {
                    "source_paper_title": paper.get("title", "")[:100],
                    "word_count": len(q["text"].split()),
                    "similarity_score": similarity_score,
                    "similar_papers_count": len(similar_papers),
                }
            else:
                difficulty = estimate_difficulty(q["text"], q["style"], paper)
                metadata = {
                    "source_paper_title": paper.get("title", "")[:100],
                    "word_count": len(q["text"].split()),
                }

            entry = {
                "query": q["text"],
                "style": q["style"],
                "relevant_papers": relevant_papers,
                "hard_negatives": hard_negatives,
                "category": category,
                "difficulty": difficulty,
                "metadata": metadata,
            }
            benchmark_queries.append(entry)

        return benchmark_queries

    def process_papers(
        self,
        papers: list[dict],
        batch_size: int = 10,
        delay: float = 0.5,
    ) -> list[dict]:
        """
        Process multiple papers with rate limiting.

        Args:
            papers: List of paper dicts
            batch_size: Papers per batch for progress logging
            delay: Delay between batches in seconds

        Returns:
            List of all benchmark query dicts
        """
        all_queries = []
        total = len(papers)

        for i, paper in enumerate(papers):
            arxiv_id = paper.get("arxiv_id", "")

            try:
                queries = self.process_paper(paper)
                all_queries.extend(queries)

                if queries:
                    logger.info(f"[{i+1}/{total}] {arxiv_id}: {len(queries)} queries")
                else:
                    logger.debug(f"[{i+1}/{total}] {arxiv_id}: no queries generated")

            except Exception as e:
                logger.error(f"[{i+1}/{total}] {arxiv_id} failed: {e}")
                continue

            # Rate limiting
            if (i + 1) % batch_size == 0:
                logger.info(f"Progress: {i+1}/{total} papers, {len(all_queries)} queries")
                time.sleep(delay)

        return all_queries

    def _build_benchmark_entries(
        self,
        queries: list[dict],
        paper: dict,
        hard_negatives: list[str],
    ) -> list[dict]:
        """Build benchmark entries from queries with metadata."""
        arxiv_id = paper.get("arxiv_id", "")
        categories = paper.get("categories", [])
        category = categories[0] if categories else "unknown"

        benchmark_queries = []
        for q in queries:
            difficulty = estimate_difficulty(q["text"], q["style"], paper)
            entry = {
                "query": q["text"],
                "style": q["style"],
                "relevant_papers": [arxiv_id],
                "hard_negatives": hard_negatives,
                "category": category,
                "difficulty": difficulty,
                "metadata": {
                    "source_paper_title": paper.get("title", "")[:100],
                    "word_count": len(q["text"].split()),
                },
            }
            benchmark_queries.append(entry)
        return benchmark_queries

    def process_papers_batch(
        self,
        papers: list[dict],
        batch_size: int = 10,
        delay: float = 0.5,
    ) -> list[dict]:
        """
        Process papers in pairs (2 per API call) for 2x throughput.

        Args:
            papers: List of paper dicts
            batch_size: Papers per batch for progress logging
            delay: Delay between batches in seconds

        Returns:
            List of all benchmark query dicts
        """
        all_queries = []
        total = len(papers)

        # Process in pairs
        for i in range(0, total, 2):
            paper1 = papers[i]
            paper2 = papers[i + 1] if i + 1 < total else None

            arxiv_id1 = paper1.get("arxiv_id", "")

            try:
                if paper2 is None:
                    # Odd paper at the end - process single
                    queries = self.process_paper(paper1)
                    all_queries.extend(queries)
                    if queries:
                        logger.info(f"[{i+1}/{total}] {arxiv_id1}: {len(queries)} queries (single)")
                else:
                    arxiv_id2 = paper2.get("arxiv_id", "")

                    # Generate queries for both papers
                    batch_result = self.generate_queries_batch(paper1, paper2)

                    # Get hard negatives for each paper
                    hard_neg1 = []
                    hard_neg2 = []
                    if self.hard_negatives_k > 0 and self.qdrant_client and self.embedder:
                        hard_neg1 = get_hard_negatives_from_qdrant(
                            arxiv_id1, paper1.get("abstract", ""),
                            self.qdrant_client, self.embedder, self.hard_negatives_k
                        )
                        hard_neg2 = get_hard_negatives_from_qdrant(
                            arxiv_id2, paper2.get("abstract", ""),
                            self.qdrant_client, self.embedder, self.hard_negatives_k
                        )

                    # Build entries for paper 1
                    queries1 = batch_result.get("paper_1", [])
                    if queries1:
                        entries1 = self._build_benchmark_entries(queries1, paper1, hard_neg1)
                        all_queries.extend(entries1)
                        logger.info(f"[{i+1}/{total}] {arxiv_id1}: {len(entries1)} queries")

                    # Build entries for paper 2
                    queries2 = batch_result.get("paper_2", [])
                    if queries2:
                        entries2 = self._build_benchmark_entries(queries2, paper2, hard_neg2)
                        all_queries.extend(entries2)
                        logger.info(f"[{i+2}/{total}] {arxiv_id2}: {len(entries2)} queries")

            except Exception as e:
                logger.error(f"[{i+1}/{total}] Batch failed: {e}")
                # Fallback to single processing
                try:
                    queries = self.process_paper(paper1)
                    all_queries.extend(queries)
                    if paper2:
                        queries = self.process_paper(paper2)
                        all_queries.extend(queries)
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")
                continue

            # Rate limiting (every batch_size papers)
            papers_processed = min(i + 2, total)
            if papers_processed % batch_size == 0:
                logger.info(f"Progress: {papers_processed}/{total} papers, {len(all_queries)} queries")
                time.sleep(delay)

        return all_queries

    def cleanup(self):
        """Cleanup resources."""
        if self._embedder:
            self._embedder.unload()
        if self._qdrant_client:
            self._qdrant_client.close()


# =============================================================================
# Data Loading
# =============================================================================

def get_papers_with_abstracts(client, limit: int = 1000) -> list[dict]:
    """Fetch papers with title and abstract from the configured metadata DB."""
    try:
        return client.get_papers(
            fields=["arxiv_id", "title", "abstract", "categories"],
            limit=limit,
            order_by="citation_count",
            require_abstract=True,
        )
    except Exception as e:
        logger.error(f"Failed to fetch papers: {e}")
        return []


# =============================================================================
# Statistics & Validation
# =============================================================================

def compute_statistics(queries: list[dict]) -> dict:
    """Compute benchmark statistics."""
    stats = {
        "total_queries": len(queries),
        "by_style": {},
        "by_difficulty": {},
        "by_category": {},
        "with_hard_negatives": 0,
        "avg_word_count": 0,
    }

    word_counts = []

    for q in queries:
        style = q.get("style", "unknown")
        difficulty = q.get("difficulty", "unknown")
        category = q.get("category", "unknown")

        stats["by_style"][style] = stats["by_style"].get(style, 0) + 1
        stats["by_difficulty"][difficulty] = stats["by_difficulty"].get(difficulty, 0) + 1
        stats["by_category"][category] = stats["by_category"].get(category, 0) + 1

        if q.get("hard_negatives"):
            stats["with_hard_negatives"] += 1

        wc = q.get("metadata", {}).get("word_count", 0)
        if wc:
            word_counts.append(wc)

    if word_counts:
        stats["avg_word_count"] = sum(word_counts) / len(word_counts)

    return stats


def validate_benchmark(queries: list[dict]) -> tuple[bool, list[str]]:
    """
    Validate benchmark queries.

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    if not queries:
        issues.append("No queries generated")
        return False, issues

    # Check style distribution
    style_counts = {}
    for q in queries:
        style = q.get("style", "unknown")
        style_counts[style] = style_counts.get(style, 0) + 1

    total = len(queries)
    for style in VALID_STYLES:
        count = style_counts.get(style, 0)
        pct = (count / total) * 100 if total > 0 else 0
        if pct < 20:  # Less than 20% for any style
            issues.append(f"Low {style} queries: {count} ({pct:.1f}%)")

    # Check for missing hard negatives (if expected)
    has_hn = sum(1 for q in queries if q.get("hard_negatives"))
    hn_pct = (has_hn / total) * 100 if total > 0 else 0
    if hn_pct < 50:
        issues.append(f"Low hard negative coverage: {has_hn}/{total} ({hn_pct:.1f}%)")

    # Check for required fields
    required_fields = ["query", "style", "relevant_papers", "category", "difficulty"]
    for i, q in enumerate(queries[:10]):  # Sample check
        for field in required_fields:
            if field not in q:
                issues.append(f"Query {i} missing field: {field}")

    return len(issues) == 0, issues


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Enhanced Benchmark Queries (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic generation
  python scripts/08_generate_synthetic_benchmark.py --limit 100

  # Full v2 features
  python scripts/08_generate_synthetic_benchmark.py \\
    --limit 2400 \\
    --use-sections \\
    --hard-negatives 3 \\
    --output data/eval/v2_benchmark_queries.json
        """,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of papers to process (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/eval/v2_benchmark_queries.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Gemini model name (default: from config)",
    )
    parser.add_argument(
        "--use-sections",
        action="store_true",
        help="Include intro/conclusion excerpts in context",
    )
    parser.add_argument(
        "--hard-negatives",
        type=int,
        default=3,
        help="Number of hard negatives per query (0 to disable, default: 3)",
    )
    parser.add_argument(
        "--parsed-dir",
        type=str,
        default="data/parsed",
        help="Directory containing parsed paper JSONs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for rate limiting (default: 10)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between batches in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation checks",
    )
    parser.add_argument(
        "--batch-papers",
        type=int,
        default=1,
        choices=[1, 2],
        help="Papers per API request (1=single, 2=batch mode for 2x speed). Default: 1",
    )
    parser.add_argument(
        "--multi-paper-gt",
        action="store_true",
        help="Include similar papers in ground truth (Phase 4.1)",
    )
    parser.add_argument(
        "--similar-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for multi-paper ground truth (default: 0.85)",
    )
    parser.add_argument(
        "--max-similar",
        type=int,
        default=3,
        help="Maximum similar papers to include in ground truth (default: 3)",
    )
    parser.add_argument(
        "--embedding-difficulty",
        action="store_true",
        help="Use embedding-based difficulty estimation (Phase 4.2)",
    )

    args = parser.parse_args()

    # Initialize
    logger.info("Initializing v2 benchmark generator...")
    client = get_db_client()

    generator = BenchmarkGeneratorV2(
        model_name=args.model,
        use_sections=args.use_sections,
        hard_negatives_k=args.hard_negatives,
        parsed_dir=Path(args.parsed_dir),
        multi_paper_gt=args.multi_paper_gt,
        similar_paper_threshold=args.similar_threshold,
        max_similar_papers=args.max_similar,
        use_embedding_difficulty=args.embedding_difficulty,
    )

    # Fetch papers
    logger.info(f"Fetching papers (limit: {args.limit})...")
    papers = get_papers_with_abstracts(client, limit=args.limit)
    logger.info(f"Found {len(papers)} papers with abstracts")

    if not papers:
        print("No papers found with abstracts")
        return

    # Generate queries
    print(f"\n{'='*60}")
    print("Enhanced Benchmark Generation (v2)")
    print(f"{'='*60}")
    print(f"Papers: {len(papers)}")
    print(f"Model: {generator.model_name}")
    print(f"Use sections: {args.use_sections}")
    print(f"Hard negatives: {args.hard_negatives}")
    print(f"Batch papers: {args.batch_papers} papers/request")
    print(f"Progress batch size: {args.batch_size}")
    print(f"Multi-paper GT: {args.multi_paper_gt} (threshold={args.similar_threshold}, max={args.max_similar})")
    print(f"Embedding difficulty: {args.embedding_difficulty}")
    print()

    start_time = time.time()

    if args.batch_papers == 2:
        logger.info("Using batch mode: 2 papers per API request")
        eval_queries = generator.process_papers_batch(
            papers,
            batch_size=args.batch_size,
            delay=args.delay,
        )
    else:
        eval_queries = generator.process_papers(
            papers,
            batch_size=args.batch_size,
            delay=args.delay,
        )

    elapsed = time.time() - start_time

    # Cleanup
    generator.cleanup()

    # Validate
    if not args.skip_validation:
        is_valid, issues = validate_benchmark(eval_queries)
        if not is_valid:
            print(f"\n⚠️  Validation warnings:")
            for issue in issues:
                print(f"  - {issue}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_queries, f, indent=2, ensure_ascii=False)

    # Compute and display statistics
    stats = compute_statistics(eval_queries)

    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Papers processed: {len(papers)}")
    print(f"Queries generated: {stats['total_queries']}")
    print(f"Avg queries/paper: {stats['total_queries']/len(papers):.2f}")
    print(f"Queries with hard negatives: {stats['with_hard_negatives']}")
    print(f"Avg word count: {stats['avg_word_count']:.1f}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Output: {output_path}")

    print(f"\nStyle distribution:")
    for style, count in sorted(stats["by_style"].items()):
        pct = (count / stats["total_queries"]) * 100 if stats["total_queries"] > 0 else 0
        print(f"  {style}: {count} ({pct:.1f}%)")

    print(f"\nDifficulty distribution:")
    for diff, count in sorted(stats["by_difficulty"].items()):
        pct = (count / stats["total_queries"]) * 100 if stats["total_queries"] > 0 else 0
        print(f"  {diff}: {count} ({pct:.1f}%)")

    print(f"\nTop categories:")
    sorted_cats = sorted(stats["by_category"].items(), key=lambda x: -x[1])[:10]
    for cat, count in sorted_cats:
        print(f"  {cat}: {count}")

    return eval_queries


if __name__ == "__main__":
    main()

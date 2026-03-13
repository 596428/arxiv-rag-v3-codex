"""
arXiv RAG v1 - HyDE (Hypothetical Document Embeddings)

Expands conceptual/abstract queries into hypothetical paper abstracts
for improved semantic retrieval.

HyDE transforms a query into a hypothetical document that would answer
the query, then uses that document's embedding for retrieval. This helps
bridge the lexical gap between conceptual queries and technical papers.

Reference:
    Gao et al. "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    https://arxiv.org/abs/2212.10496

Usage:
    from src.rag.hyde import HyDEExpander, expand_query

    expander = HyDEExpander()
    expanded = expander.expand("Teaching machines to think step by step")
    # Returns hypothetical abstract with technical terminology
"""

from dataclasses import dataclass
from typing import Optional
import time

from ..utils.config import settings
from ..utils.logging import get_logger

logger = get_logger("hyde")


# =============================================================================
# HyDE Prompts
# =============================================================================

HYDE_PROMPT_ABSTRACT = """You are a research paper abstract generator. Given a search query, generate a hypothetical abstract (2-3 sentences) for a paper that would perfectly answer the query.

The abstract should:
1. Use technical terminology appropriate for academic ML/AI papers
2. Mention specific methods, architectures, or techniques
3. Include concrete claims about results or contributions
4. Be written in academic style

Query: {query}

Hypothetical Abstract:"""


HYDE_PROMPT_PASSAGE = """Given a search query about machine learning research, write a short passage (2-3 sentences) that would appear in a relevant paper. Use technical terminology and be specific.

Query: {query}

Relevant passage:"""


HYDE_PROMPT_KEYWORDS = """Given a conceptual search query, extract or generate the technical keywords that would appear in relevant academic papers.

Query: {query}

Output only the keywords separated by commas (no explanation):"""


@dataclass
class HyDEResult:
    """Result from HyDE expansion."""
    original_query: str
    expanded_text: str
    expansion_type: str  # "abstract", "passage", "keywords"
    latency_ms: float
    success: bool
    error: Optional[str] = None


class HyDEExpander:
    """
    HyDE (Hypothetical Document Embeddings) query expander.

    Transforms conceptual queries into hypothetical documents that
    contain relevant technical terminology for better retrieval.
    """

    def __init__(
        self,
        model_name: str = None,
        expansion_type: str = "abstract",
        cache_enabled: bool = True,
    ):
        """
        Initialize HyDE expander.

        Args:
            model_name: Gemini model name (default from config)
            expansion_type: Type of expansion ("abstract", "passage", "keywords")
            cache_enabled: Whether to cache expansions
        """
        self.model_name = model_name or settings.gemini_model
        self.expansion_type = expansion_type
        self.cache_enabled = cache_enabled
        self._cache: dict[str, str] = {}
        self._model = None

        # Select prompt template
        self.prompts = {
            "abstract": HYDE_PROMPT_ABSTRACT,
            "passage": HYDE_PROMPT_PASSAGE,
            "keywords": HYDE_PROMPT_KEYWORDS,
        }

        if expansion_type not in self.prompts:
            raise ValueError(f"Unknown expansion_type: {expansion_type}")

        logger.info(f"Initialized HyDEExpander: model={self.model_name}, type={expansion_type}")

    @property
    def model(self):
        """Lazy load Gemini model."""
        if self._model is None:
            import google.generativeai as genai

            api_key = settings.gemini_api_key
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")

            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(self.model_name)
            logger.info(f"Loaded Gemini model: {self.model_name}")

        return self._model

    def expand(
        self,
        query: str,
        expansion_type: str = None,
        bypass_cache: bool = False,
    ) -> str:
        """
        Expand query using HyDE.

        Args:
            query: Original search query
            expansion_type: Override default expansion type
            bypass_cache: Skip cache lookup

        Returns:
            Expanded query text (hypothetical document)
        """
        result = self.expand_detailed(query, expansion_type, bypass_cache)
        return result.expanded_text if result.success else query

    def expand_detailed(
        self,
        query: str,
        expansion_type: str = None,
        bypass_cache: bool = False,
    ) -> HyDEResult:
        """
        Expand query with detailed result.

        Args:
            query: Original search query
            expansion_type: Override default expansion type
            bypass_cache: Skip cache lookup

        Returns:
            HyDEResult with expanded text and metadata
        """
        exp_type = expansion_type or self.expansion_type
        cache_key = f"{exp_type}:{query}"

        # Check cache
        if self.cache_enabled and not bypass_cache and cache_key in self._cache:
            logger.debug(f"HyDE cache hit for: {query[:50]}...")
            return HyDEResult(
                original_query=query,
                expanded_text=self._cache[cache_key],
                expansion_type=exp_type,
                latency_ms=0.0,
                success=True,
            )

        start = time.time()

        try:
            # Build prompt
            prompt_template = self.prompts[exp_type]
            prompt = prompt_template.format(query=query)

            # Generate
            response = self.model.generate_content(prompt)
            expanded = response.text.strip()

            # Clean up response
            expanded = self._clean_response(expanded, exp_type)

            latency_ms = (time.time() - start) * 1000

            # Cache result
            if self.cache_enabled:
                self._cache[cache_key] = expanded

            logger.debug(f"HyDE expanded [{exp_type}] in {latency_ms:.0f}ms: {query[:30]}... -> {expanded[:50]}...")

            return HyDEResult(
                original_query=query,
                expanded_text=expanded,
                expansion_type=exp_type,
                latency_ms=latency_ms,
                success=True,
            )

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            logger.warning(f"HyDE expansion failed: {e}")

            return HyDEResult(
                original_query=query,
                expanded_text=query,  # Fallback to original
                expansion_type=exp_type,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )

    def _clean_response(self, text: str, expansion_type: str) -> str:
        """Clean up model response."""
        # Remove common prefixes
        prefixes = [
            "Here is a hypothetical abstract:",
            "Hypothetical Abstract:",
            "Abstract:",
            "Here is a relevant passage:",
            "Relevant passage:",
        ]

        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Remove quotes if wrapped
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]

        return text.strip()

    def expand_for_search(
        self,
        query: str,
        query_type: str = None,
    ) -> str:
        """
        Expand query for search, adapting to query type.

        Only expands conceptual queries; returns original for others.

        Args:
            query: Search query
            query_type: Query type from classifier ("keyword", "natural", "conceptual")

        Returns:
            Original or expanded query
        """
        # Only expand conceptual queries
        if query_type and query_type != "conceptual":
            logger.debug(f"Skipping HyDE for {query_type} query")
            return query

        # For conceptual queries, use abstract expansion
        return self.expand(query, expansion_type="abstract")

    def clear_cache(self) -> int:
        """Clear expansion cache. Returns number of entries cleared."""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared HyDE cache: {count} entries")
        return count


# =============================================================================
# Convenience Functions
# =============================================================================

_default_expander: Optional[HyDEExpander] = None


def get_expander() -> HyDEExpander:
    """Get singleton expander instance."""
    global _default_expander
    if _default_expander is None:
        _default_expander = HyDEExpander()
    return _default_expander


def expand_query(query: str, query_type: str = None) -> str:
    """
    Expand query using HyDE if beneficial.

    Args:
        query: Search query
        query_type: Optional query type for adaptive expansion

    Returns:
        Expanded query (or original if expansion not beneficial)
    """
    return get_expander().expand_for_search(query, query_type)


def expand_conceptual_query(query: str) -> str:
    """
    Expand a conceptual query to hypothetical abstract.

    Args:
        query: Conceptual search query

    Returns:
        Hypothetical abstract text
    """
    return get_expander().expand(query, expansion_type="abstract")


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    import sys

    # Test queries
    test_queries = [
        "Teaching machines to think step by step using reward signals",
        "Making language models better at following instructions without human examples",
        "Understanding how neural networks learn from text",
    ]

    print("HyDE Expansion Tests")
    print("=" * 70)

    try:
        expander = HyDEExpander()

        for query in test_queries:
            print(f"\nQuery: {query}")
            result = expander.expand_detailed(query)

            if result.success:
                print(f"Expanded ({result.latency_ms:.0f}ms):")
                print(f"  {result.expanded_text[:200]}...")
            else:
                print(f"Failed: {result.error}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

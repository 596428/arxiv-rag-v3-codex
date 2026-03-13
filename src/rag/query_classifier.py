"""
arXiv RAG v1 - Query Type Classifier

Classifies search queries into types for adaptive retrieval strategy selection.

Query Types:
- keyword: Short queries with technical terms (e.g., "BERT attention mechanism")
- natural: Question-form queries with some technical terms
- conceptual: Abstract/paraphrased queries with minimal technical overlap

Usage:
    from src.rag.query_classifier import QueryClassifier, classify_query

    classifier = QueryClassifier()
    query_type = classifier.classify("How does attention work in transformers?")
    # Returns: "natural"
"""

import re
from dataclasses import dataclass
from typing import Optional

from ..utils.logging import get_logger

logger = get_logger("query_classifier")


# =============================================================================
# Technical Term Patterns
# =============================================================================

# Common ML/AI technical terms and acronyms
TECHNICAL_TERMS = {
    # Models and architectures
    "transformer", "bert", "gpt", "llm", "lstm", "cnn", "rnn", "gan",
    "vae", "autoencoder", "diffusion", "vit", "clip", "dalle",
    "deepseek", "llama", "mistral", "gemini", "claude", "chatgpt",
    "t5", "bart", "roberta", "xlnet", "albert", "electra",
    # Techniques
    "attention", "self-attention", "multi-head", "embedding", "tokenization",
    "fine-tuning", "pretraining", "transfer learning", "few-shot", "zero-shot",
    "in-context", "prompt", "chain-of-thought", "cot", "rag", "retrieval",
    "rlhf", "dpo", "ppo", "grpo", "reinforcement learning",
    "quantization", "pruning", "distillation", "lora", "qlora", "peft",
    # Concepts
    "multimodal", "vision-language", "text-to-image", "image-to-text",
    "encoder", "decoder", "sequence-to-sequence", "seq2seq",
    "cross-entropy", "softmax", "layer normalization", "dropout",
    "positional encoding", "sinusoidal", "rotary", "alibi",
    # Benchmarks
    "glue", "superglue", "squad", "mmlu", "hellaswag", "arc",
    "humaneval", "mbpp", "gsm8k", "math", "big-bench",
    # Tasks
    "classification", "regression", "generation", "summarization",
    "translation", "qa", "question answering", "ner", "entity recognition",
    "sentiment analysis", "semantic similarity", "entailment",
}

# Question words that indicate natural language queries
QUESTION_WORDS = {"how", "what", "why", "when", "where", "which", "who", "can", "does", "is", "are"}

# Abstract/conceptual language patterns
CONCEPTUAL_PATTERNS = [
    r"\bmaking\s+\w+\s+better\b",
    r"\bimproving\s+\w+\b",
    r"\bteaching\s+(machines?|computers?|models?)\b",
    r"\blearning\s+from\s+\w+\b",
    r"\bunderstanding\s+\w+\b",
    r"\bpredicting\s+\w+\b",
    r"\bstep.by.step\b",
    r"\bthink(ing)?\s+(through|step)\b",
    r"\breward\s+signal\b",
    r"\bhuman\s+(feedback|examples?|demonstrations?)\b",
    r"\bwithout\s+(human|examples?|labels?)\b",
    # Additional conceptual patterns (v2)
    r"\bhow\s+does\s+\w+\s+enable\b",
    r"\bhow\s+do\s+\w+\s+handle\b",
    r"\bhow\s+can\s+\w+\s+(improve|enhance|achieve)\b",
    r"\bwhat\s+are\s+the\s+(limitations?|challenges?|benefits?)\b",
    r"\bwhy\s+do\s+\w+\s+(fail|struggle|succeed)\b",
    r"\bwhat\s+makes\s+\w+\s+(work|effective|possible)\b",
    r"\benabling\s+\w+\b",
    r"\bachieving\s+\w+\b",
    r"\bovercoming\s+\w+\b",
    r"\baddressing\s+\w+\b",
]


@dataclass
class ClassificationResult:
    """Query classification result with confidence scores."""
    query_type: str  # "keyword", "natural", or "conceptual"
    confidence: float  # 0.0 to 1.0
    features: dict  # Feature values used for classification
    recommended_preset: str  # RRF preset recommendation


class QueryClassifier:
    """
    Classifies queries into types for adaptive retrieval.

    Features used:
    - Word count
    - Technical term ratio
    - Question word presence
    - Conceptual pattern matches
    """

    def __init__(self, technical_terms: set[str] = None):
        """
        Initialize classifier.

        Args:
            technical_terms: Custom set of technical terms (optional)
        """
        self.technical_terms = technical_terms or TECHNICAL_TERMS
        self._compiled_conceptual = [re.compile(p, re.IGNORECASE) for p in CONCEPTUAL_PATTERNS]

    def classify(self, query: str) -> str:
        """
        Classify query into type.

        Args:
            query: Search query text

        Returns:
            Query type: "keyword", "natural", or "conceptual"
        """
        result = self.classify_detailed(query)
        return result.query_type

    def classify_detailed(self, query: str) -> ClassificationResult:
        """
        Classify query with detailed feature analysis.

        Args:
            query: Search query text

        Returns:
            ClassificationResult with type, confidence, and features
        """
        query_lower = query.lower().strip()
        words = query_lower.split()
        word_count = len(words)

        # Feature extraction
        features = {
            "word_count": word_count,
            "has_question_word": any(w in QUESTION_WORDS for w in words[:3]),
            "ends_with_question": query.strip().endswith("?"),
            "technical_term_count": 0,
            "technical_term_ratio": 0.0,
            "conceptual_pattern_count": 0,
        }

        # Count technical terms
        tech_count = 0
        for term in self.technical_terms:
            if term in query_lower:
                tech_count += 1
        features["technical_term_count"] = tech_count
        features["technical_term_ratio"] = tech_count / max(word_count, 1)

        # Count conceptual pattern matches
        conceptual_count = 0
        for pattern in self._compiled_conceptual:
            if pattern.search(query_lower):
                conceptual_count += 1
        features["conceptual_pattern_count"] = conceptual_count

        # Classification logic
        query_type, confidence = self._classify_from_features(features)

        # Map to RRF preset
        preset_map = {
            "keyword": "keyword",
            "natural": "default",
            "conceptual": "conceptual",
        }

        return ClassificationResult(
            query_type=query_type,
            confidence=confidence,
            features=features,
            recommended_preset=preset_map[query_type],
        )

    def _classify_from_features(self, features: dict) -> tuple[str, float]:
        """
        Classify based on extracted features.

        Returns:
            Tuple of (query_type, confidence)
        """
        word_count = features["word_count"]
        has_question = features["has_question_word"] or features["ends_with_question"]
        tech_ratio = features["technical_term_ratio"]
        tech_count = features["technical_term_count"]
        conceptual_count = features["conceptual_pattern_count"]

        # Rule-based classification with confidence scoring

        # KEYWORD: Short queries with technical terms, no question words
        if word_count <= 8 and not has_question and tech_count >= 1:
            confidence = min(0.9, 0.5 + tech_ratio * 0.5 + (0.1 if word_count <= 5 else 0))
            return "keyword", confidence

        # CONCEPTUAL: Conceptual patterns detected (prioritize over technical terms)
        if conceptual_count >= 1:
            # Conceptual patterns are strong indicators, even with some technical terms
            if tech_count <= 2:
                confidence = min(0.9, 0.6 + conceptual_count * 0.15)
                return "conceptual", confidence
            # Many technical terms + conceptual pattern → lower confidence conceptual
            confidence = 0.55
            return "conceptual", confidence

        if tech_ratio < 0.1 and word_count >= 10:
            # Long query with almost no technical terms
            confidence = 0.7 + (0.1 if conceptual_count > 0 else 0)
            return "conceptual", confidence

        if tech_count <= 1 and word_count >= 10 and not has_question:
            # Very few technical terms, long query, not a question
            confidence = 0.6
            return "conceptual", confidence

        if tech_count == 0 and word_count >= 6:
            # No technical terms and moderate length (relaxed from 8 to 6)
            confidence = 0.6
            return "conceptual", confidence

        # NATURAL: Everything else (question form, mixed technical/natural)
        if has_question:
            if tech_count >= 2:
                confidence = 0.8
            else:
                confidence = 0.7
            return "natural", confidence

        # Default to natural for ambiguous cases
        confidence = 0.5
        return "natural", confidence

    def get_rrf_preset(self, query: str) -> str:
        """
        Get recommended RRF preset for query.

        Args:
            query: Search query text

        Returns:
            RRF preset name ("keyword", "default", "conceptual")
        """
        result = self.classify_detailed(query)
        return result.recommended_preset


# =============================================================================
# Convenience Functions
# =============================================================================

_default_classifier: Optional[QueryClassifier] = None


def get_classifier() -> QueryClassifier:
    """Get singleton classifier instance."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = QueryClassifier()
    return _default_classifier


def classify_query(query: str) -> str:
    """
    Classify a query into type.

    Args:
        query: Search query text

    Returns:
        Query type: "keyword", "natural", or "conceptual"
    """
    return get_classifier().classify(query)


def classify_query_detailed(query: str) -> ClassificationResult:
    """
    Classify query with detailed feature analysis.

    Args:
        query: Search query text

    Returns:
        ClassificationResult with type, confidence, and features
    """
    return get_classifier().classify_detailed(query)


def get_recommended_preset(query: str) -> str:
    """
    Get recommended RRF weight preset for query.

    Args:
        query: Search query text

    Returns:
        Preset name for use with QdrantHybridRetriever.set_weights()
    """
    return get_classifier().get_rrf_preset(query)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test cases
    test_queries = [
        # Keyword style
        ("BERT attention mechanism transformer", "keyword"),
        ("LLM RLHF fine-tuning GPT", "keyword"),
        ("RAG retrieval augmented generation", "keyword"),

        # Natural style
        ("How does attention work in transformer models?", "natural"),
        ("What techniques improve LLM reasoning capabilities?", "natural"),
        ("Can you explain how BERT handles masked tokens?", "natural"),

        # Conceptual style
        ("Teaching machines to think step by step using reward signals", "conceptual"),
        ("Making language models better at following instructions without human examples", "conceptual"),
        ("Understanding text by learning from large amounts of writing", "conceptual"),
    ]

    classifier = QueryClassifier()

    print("Query Classification Tests")
    print("=" * 70)

    correct = 0
    for query, expected in test_queries:
        result = classifier.classify_detailed(query)
        status = "OK" if result.query_type == expected else "FAIL"
        if result.query_type == expected:
            correct += 1

        print(f"\n[{status}] Query: {query[:60]}...")
        print(f"  Expected: {expected}, Got: {result.query_type} (conf: {result.confidence:.2f})")
        print(f"  Features: tech={result.features['technical_term_count']}, "
              f"conceptual={result.features['conceptual_pattern_count']}, "
              f"words={result.features['word_count']}")
        print(f"  Preset: {result.recommended_preset}")

    print(f"\n{'=' * 70}")
    print(f"Accuracy: {correct}/{len(test_queries)} ({100*correct/len(test_queries):.0f}%)")

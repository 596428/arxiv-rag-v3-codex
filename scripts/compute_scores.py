#!/usr/bin/env python3
"""
arXiv RAG v1 - Multi-Score Paper Ranking

Computes final paper scores using 4 metrics:
1. Citation Score (40%) - Normalized citation count / months
2. Recency Boost (30%) - Exponential decay with 6-month half-life
3. Semantic Impact (20%) - Cosine similarity with anchor query
4. Stratified Bonus (10%) - Topic diversity bonus

Usage:
    python scripts/compute_scores.py --input data/collection/semantic_filtered.json
    python scripts/compute_scores.py --output data/collection/scored_papers.json
    python scripts/compute_scores.py --fetch-citations  # Fetch latest from DB
"""

import argparse
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.supabase_client import get_supabase_client
from src.utils.logging import get_logger

logger = get_logger("compute_scores")

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "collection"

# Scoring weights
WEIGHTS = {
    "citation": 0.40,
    "recency": 0.30,
    "semantic": 0.20,
    "stratified": 0.10,
}

# Recency decay parameter (months)
RECENCY_TAU = 6.0

# Topic categories for stratification
TOPIC_KEYWORDS = {
    "llm_pretraining": [
        "pretraining", "pre-training", "pre-trained", "foundation model",
        "scaling", "large language model", "llm training",
    ],
    "reasoning_alignment": [
        "reasoning", "chain-of-thought", "cot", "alignment", "rlhf",
        "reinforcement learning from human feedback", "safety",
    ],
    "multimodal": [
        "multimodal", "vision-language", "vlm", "image-text",
        "visual question answering", "vqa",
    ],
    "retrieval_rag": [
        "retrieval", "rag", "retrieval-augmented", "knowledge base",
        "information retrieval", "dense retrieval",
    ],
    "architecture": [
        "transformer", "attention", "architecture", "moe", "mixture of experts",
        "sparse", "efficient",
    ],
    "optimization": [
        "quantization", "pruning", "distillation", "compression",
        "efficient", "lora", "peft", "fine-tuning",
    ],
    "evaluation": [
        "benchmark", "evaluation", "metric", "leaderboard",
        "assessment", "capability",
    ],
}


def fetch_citation_counts() -> dict[str, int]:
    """Fetch citation counts from database."""
    supabase = get_supabase_client()
    citation_map = {}
    offset = 0
    batch_size = 1000

    while True:
        result = (
            supabase.client.table("papers")
            .select("arxiv_id, citation_count")
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        batch = result.data or []
        if not batch:
            break

        for row in batch:
            citation_map[row["arxiv_id"]] = row.get("citation_count") or 0

        offset += batch_size

    logger.info(f"Fetched citation counts for {len(citation_map)} papers from DB")
    return citation_map


def compute_citation_score(citation_count: int, months_since_pub: float) -> float:
    """
    Compute normalized citation score.

    Score = citations / months * scaling_factor
    Newer papers get higher scores for same citation count.
    """
    if months_since_pub <= 0:
        months_since_pub = 1

    # Citations per month, with diminishing returns
    raw_score = citation_count / months_since_pub
    # Log scaling to prevent outliers dominating
    return math.log1p(raw_score)


def compute_recency_boost(published_date: date, reference_date: date = None) -> float:
    """
    Compute recency boost with exponential decay.

    Score = exp(-age_months / tau)
    """
    reference_date = reference_date or date.today()

    if not published_date:
        return 0.5  # Default for missing dates

    months_age = (reference_date - published_date).days / 30.0

    # Exponential decay
    return math.exp(-months_age / RECENCY_TAU)


def compute_semantic_score(similarity: float) -> float:
    """
    Normalize semantic similarity score.

    Input is already 0-1 from cosine similarity.
    """
    return max(0, min(1, similarity))


def classify_topic(title: str, abstract: str) -> str:
    """
    Classify paper into a topic category.

    Returns the best matching topic or 'other'.
    """
    text = f"{title} {abstract}".lower()

    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[topic] = score

    if scores:
        return max(scores, key=scores.get)
    return "other"


def compute_stratified_bonus(
    topic: str,
    topic_counts: dict[str, int],
    target_distribution: dict[str, float] = None,
) -> float:
    """
    Compute stratification bonus to encourage topic diversity.

    Papers in underrepresented topics get higher bonus.
    """
    if not target_distribution:
        # Equal distribution target
        n_topics = len(TOPIC_KEYWORDS) + 1  # +1 for 'other'
        target_distribution = {t: 1.0 / n_topics for t in list(TOPIC_KEYWORDS.keys()) + ["other"]}

    total_papers = sum(topic_counts.values()) or 1
    current_ratio = topic_counts.get(topic, 0) / total_papers
    target_ratio = target_distribution.get(topic, 0.1)

    # Bonus for underrepresented topics
    if current_ratio < target_ratio:
        return 1.0 + (target_ratio - current_ratio) * 2
    return 1.0


def compute_final_score(
    paper: dict,
    reference_date: date,
    topic_counts: dict[str, int],
) -> tuple[float, dict]:
    """
    Compute final weighted score for a paper.

    Returns:
        Tuple of (final_score, component_scores)
    """
    # Parse date
    pub_date = None
    if paper.get("published_date"):
        try:
            if isinstance(paper["published_date"], str):
                pub_date = datetime.fromisoformat(paper["published_date"].replace("Z", "+00:00")).date()
            else:
                pub_date = paper["published_date"]
        except Exception:
            pass

    months_since_pub = (reference_date - pub_date).days / 30.0 if pub_date else 12

    # Component scores
    citation_score = compute_citation_score(
        paper.get("citation_count", 0),
        months_since_pub,
    )

    recency_score = compute_recency_boost(pub_date, reference_date)

    semantic_score = compute_semantic_score(
        paper.get("semantic_similarity", 0.5)
    )

    topic = classify_topic(
        paper.get("title", ""),
        paper.get("abstract", ""),
    )

    stratified_score = compute_stratified_bonus(topic, topic_counts)

    # Normalize citation score (log scale, typical range 0-5)
    citation_normalized = min(1.0, citation_score / 5.0)

    # Weighted final score
    final_score = (
        WEIGHTS["citation"] * citation_normalized +
        WEIGHTS["recency"] * recency_score +
        WEIGHTS["semantic"] * semantic_score +
        WEIGHTS["stratified"] * (stratified_score - 1.0) * 0.5  # Bonus only
    )

    components = {
        "citation": round(citation_normalized, 4),
        "recency": round(recency_score, 4),
        "semantic": round(semantic_score, 4),
        "stratified": round(stratified_score, 4),
        "topic": topic,
    }

    return final_score, components


def main():
    parser = argparse.ArgumentParser(
        description="Compute multi-score paper ranking"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(OUTPUT_DIR / "semantic_filtered.json"),
        help="Input JSON file with papers"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(OUTPUT_DIR / "scored_papers.json"),
        help="Output JSON file"
    )
    parser.add_argument(
        "--reference-date",
        type=str,
        help="Reference date for recency (default: today)"
    )
    parser.add_argument(
        "--fetch-citations",
        action="store_true",
        help="Fetch latest citation counts from database"
    )

    args = parser.parse_args()

    # Load papers
    input_file = Path(args.input)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    with open(input_file, 'r') as f:
        papers = json.load(f)

    logger.info(f"Loaded {len(papers)} papers from {input_file}")

    # Fetch and merge citation counts from database
    if args.fetch_citations:
        logger.info("Fetching citation counts from database...")
        citation_map = fetch_citation_counts()

        updated = 0
        for paper in papers:
            arxiv_id = paper.get("arxiv_id")
            if arxiv_id and arxiv_id in citation_map:
                old_count = paper.get("citation_count", 0)
                new_count = citation_map[arxiv_id]
                if new_count != old_count:
                    paper["citation_count"] = new_count
                    updated += 1

        logger.info(f"Updated citation counts for {updated} papers")

        # Show citation stats
        with_citations = sum(1 for p in papers if p.get("citation_count", 0) > 0)
        total_citations = sum(p.get("citation_count", 0) for p in papers)
        logger.info(f"Papers with citations: {with_citations}/{len(papers)}")
        logger.info(f"Total citations: {total_citations}")

    # Reference date
    if args.reference_date:
        reference_date = datetime.fromisoformat(args.reference_date).date()
    else:
        reference_date = date.today()

    logger.info(f"Reference date: {reference_date}")

    # First pass: classify topics for stratification
    topic_counts = defaultdict(int)
    for paper in papers:
        topic = classify_topic(paper.get("title", ""), paper.get("abstract", ""))
        topic_counts[topic] += 1

    logger.info(f"Topic distribution: {dict(topic_counts)}")

    # Second pass: compute scores
    scored_papers = []
    for paper in papers:
        final_score, components = compute_final_score(paper, reference_date, topic_counts)

        paper["final_score"] = round(final_score, 4)
        paper["score_components"] = components

        scored_papers.append(paper)

    # Sort by final score
    scored_papers.sort(key=lambda x: x["final_score"], reverse=True)

    # Add rank
    for i, paper in enumerate(scored_papers):
        paper["rank"] = i + 1

    # Save results
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(scored_papers, f, indent=2)

    logger.info(f"Saved {len(scored_papers)} scored papers to {output_file}")

    # Summary
    scores = [p["final_score"] for p in scored_papers]
    print("\n" + "=" * 50)
    print("SCORING SUMMARY")
    print("=" * 50)
    print(f"Total papers:        {len(scored_papers)}")
    print(f"Score range:         {min(scores):.3f} - {max(scores):.3f}")
    print(f"Mean score:          {sum(scores)/len(scores):.3f}")
    print(f"\nTopic distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        pct = count / len(papers) * 100
        print(f"  {topic:20s}: {count:4d} ({pct:.1f}%)")
    print(f"\nTop 5 papers:")
    for p in scored_papers[:5]:
        print(f"  {p['rank']:3d}. [{p['final_score']:.3f}] {p.get('title', '')[:50]}...")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    exit(main())

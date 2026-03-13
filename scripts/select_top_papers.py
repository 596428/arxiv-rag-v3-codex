#!/usr/bin/env python3
"""
arXiv RAG v1 - Constraint-Aware Paper Selection

Selects top papers using "minimum guarantee + free competition" strategy:
1. Guarantee minimum papers per topic
2. Fill remaining slots by final_score (free competition)

Usage:
    python scripts/select_top_papers.py --top-k 2500 --min-per-topic 50
    python scripts/select_top_papers.py --top-k 1000 --min-per-topic 30
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import get_logger

logger = get_logger("select_top_papers")

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "collection"


def select_with_min_guarantee(
    papers: list[dict],
    top_k: int,
    min_per_topic: int,
) -> list[dict]:
    """
    Select papers with minimum guarantee per topic + free competition.

    Strategy:
    1. For each topic, guarantee min_per_topic papers (sorted by score)
    2. Fill remaining slots from all papers by score

    Args:
        papers: List of papers with 'final_score' and 'score_components.topic'
        top_k: Total number of papers to select
        min_per_topic: Minimum papers guaranteed per topic

    Returns:
        Selected papers sorted by final_score
    """
    # Group papers by topic
    by_topic = defaultdict(list)
    for paper in papers:
        topic = paper.get("score_components", {}).get("topic", "other")
        by_topic[topic].append(paper)

    # Sort each topic by score
    for topic in by_topic:
        by_topic[topic].sort(key=lambda x: x.get("final_score", 0), reverse=True)

    selected = set()  # arxiv_ids
    selected_papers = []

    # Phase 1: Guarantee minimum per topic
    logger.info(f"Phase 1: Guaranteeing {min_per_topic} papers per topic...")
    for topic, topic_papers in by_topic.items():
        count = 0
        for paper in topic_papers:
            if count >= min_per_topic:
                break
            arxiv_id = paper.get("arxiv_id")
            if arxiv_id not in selected:
                selected.add(arxiv_id)
                selected_papers.append(paper)
                count += 1
        logger.info(f"  {topic}: guaranteed {count} papers")

    logger.info(f"After Phase 1: {len(selected_papers)} papers selected")

    # Phase 2: Free competition for remaining slots
    remaining_slots = top_k - len(selected_papers)
    logger.info(f"Phase 2: Free competition for {remaining_slots} remaining slots...")

    # Get all unselected papers, sorted by score
    unselected = [p for p in papers if p.get("arxiv_id") not in selected]
    unselected.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    # Fill remaining slots
    for paper in unselected[:remaining_slots]:
        selected_papers.append(paper)

    # Final sort by score
    selected_papers.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    # Re-assign ranks
    for i, paper in enumerate(selected_papers):
        paper["rank"] = i + 1

    return selected_papers


def validate_selection(papers: list[dict], min_per_topic: int) -> dict:
    """Validate selection meets constraints."""
    # Count by topic
    topic_counts = defaultdict(int)
    for paper in papers:
        topic = paper.get("score_components", {}).get("topic", "other")
        topic_counts[topic] += 1

    # Check constraints
    violations = []
    for topic, count in topic_counts.items():
        if count < min_per_topic:
            violations.append(f"{topic}: {count} < {min_per_topic}")

    # Citation stats
    with_citations = sum(1 for p in papers if p.get("citation_count", 0) > 0)
    total_citations = sum(p.get("citation_count", 0) for p in papers)

    return {
        "topic_counts": dict(topic_counts),
        "violations": violations,
        "with_citations": with_citations,
        "total_citations": total_citations,
        "valid": len(violations) == 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Select top papers with constraint-aware strategy"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(OUTPUT_DIR / "scored_papers.json"),
        help="Input JSON file with scored papers"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(OUTPUT_DIR / "final_papers.json"),
        help="Output JSON file"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=2500,
        help="Number of papers to select (default: 2500)"
    )
    parser.add_argument(
        "--min-per-topic", "-m",
        type=int,
        default=50,
        help="Minimum papers per topic (default: 50)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show selection stats without saving"
    )

    args = parser.parse_args()

    # Load papers
    input_file = Path(args.input)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    with open(input_file, 'r') as f:
        papers = json.load(f)

    logger.info(f"Loaded {len(papers)} scored papers from {input_file}")

    # Validate inputs
    n_topics = len(set(
        p.get("score_components", {}).get("topic", "other")
        for p in papers
    ))
    min_required = n_topics * args.min_per_topic

    if args.top_k < min_required:
        logger.warning(
            f"top_k ({args.top_k}) < min_required ({min_required} = "
            f"{n_topics} topics * {args.min_per_topic})"
        )

    if args.top_k > len(papers):
        logger.warning(f"top_k ({args.top_k}) > available papers ({len(papers)})")
        args.top_k = len(papers)

    # Select papers
    selected = select_with_min_guarantee(papers, args.top_k, args.min_per_topic)

    logger.info(f"Selected {len(selected)} papers")

    # Validate
    validation = validate_selection(selected, args.min_per_topic)

    # Summary
    print("\n" + "=" * 60)
    print("SELECTION SUMMARY")
    print("=" * 60)
    print(f"Strategy:          Min {args.min_per_topic}/topic + free competition")
    print(f"Total selected:    {len(selected)}")
    print(f"Papers with cites: {validation['with_citations']}")
    print(f"Total citations:   {validation['total_citations']}")
    print(f"\nTopic distribution:")
    for topic, count in sorted(validation['topic_counts'].items(), key=lambda x: -x[1]):
        pct = count / len(selected) * 100
        marker = "" if count >= args.min_per_topic else " [!]"
        print(f"  {topic:20s}: {count:4d} ({pct:5.1f}%){marker}")

    if validation['violations']:
        print(f"\n[WARNING] Constraint violations:")
        for v in validation['violations']:
            print(f"  - {v}")
    else:
        print(f"\n[OK] All constraints satisfied")

    # Score stats
    scores = [p.get("final_score", 0) for p in selected]
    print(f"\nScore range:       {min(scores):.3f} - {max(scores):.3f}")
    print(f"Mean score:        {sum(scores)/len(scores):.3f}")

    # Top 10 papers
    print(f"\nTop 10 papers:")
    for p in selected[:10]:
        cites = p.get("citation_count", 0)
        topic = p.get("score_components", {}).get("topic", "?")
        print(f"  {p['rank']:3d}. [{p['final_score']:.3f}] [{cites:4d} cites] [{topic:15s}] {p.get('title', '')[:40]}...")

    print("=" * 60)

    if args.dry_run:
        logger.info("Dry run - not saving")
        return 0

    # Save results
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(selected, f, indent=2)

    logger.info(f"Saved {len(selected)} papers to {output_file}")

    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Test text-embedding-3-large (3072 dims) embedding on a few chunks.

1. Add new vector field 'dense_3large' (3072d) to Qdrant collection
2. Embed a few chunks with 3-large
3. Verify storage and retrieval
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client.http.models import VectorParams, Distance
from src.storage.qdrant_client import get_qdrant_client, COLLECTION_NAME
from src.embedding import OpenAIEmbedder, EmbeddingConfig
from src.utils.logging import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger("test_3large")


TEST_COLLECTION = "test_3large"


def create_test_collection():
    """Create a test collection with 3072d vector support."""
    client = get_qdrant_client()

    try:
        # Check if exists
        collections = client.client.get_collections().collections
        exists = any(c.name == TEST_COLLECTION for c in collections)

        if exists:
            logger.info(f"Collection '{TEST_COLLECTION}' already exists, recreating...")
            client.client.delete_collection(TEST_COLLECTION)

        # Create with 3072d vector
        client.client.create_collection(
            collection_name=TEST_COLLECTION,
            vectors_config={
                "dense_3large": VectorParams(
                    size=3072,
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
            },
        )
        logger.info(f"Created collection '{TEST_COLLECTION}' with dense_3large (3072d)")
        return True

    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        return False


def test_3large_embedding():
    """Test 3-large embedding on sample texts."""

    # Configure for 3072d (no MRL reduction)
    config = EmbeddingConfig(
        use_openai=True,
        openai_model="text-embedding-3-large",
        openai_dimensions=3072,  # Full dimensions
    )

    embedder = OpenAIEmbedder(config)

    # Test texts
    test_texts = [
        "Large language models have revolutionized natural language processing through transformer architectures.",
        "Retrieval-augmented generation combines the power of retrieval systems with generative models.",
        "Chain-of-thought prompting enables complex reasoning in language models.",
    ]

    logger.info(f"Testing {config.openai_model} with {config.openai_dimensions} dims")

    try:
        embeddings = embedder.embed_texts(test_texts)

        logger.info(f"Generated {len(embeddings)} embeddings")
        logger.info(f"Embedding dimension: {len(embeddings[0])}")

        # Verify dimensions
        for i, emb in enumerate(embeddings):
            if len(emb) != 3072:
                logger.error(f"Text {i}: expected 3072, got {len(emb)}")
                return None
            logger.info(f"Text {i}: {len(emb)} dims ✓")

        return embeddings

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return None


def test_qdrant_upsert(embeddings: list[list[float]]):
    """Test upserting 3072d vectors to Qdrant."""
    from qdrant_client.http.models import PointStruct

    client = get_qdrant_client()

    test_chunks = [
        {"id": "test_3large_0", "paper_id": "test", "content": "LLM test 1"},
        {"id": "test_3large_1", "paper_id": "test", "content": "RAG test 2"},
        {"id": "test_3large_2", "paper_id": "test", "content": "CoT test 3"},
    ]

    points = []
    for i, (chunk, emb) in enumerate(zip(test_chunks, embeddings)):
        point_id = abs(hash(chunk["id"])) % (2**63)
        points.append(PointStruct(
            id=point_id,
            vector={"dense_3large": emb},
            payload=chunk,
        ))

    try:
        client.client.upsert(
            collection_name=TEST_COLLECTION,
            points=points,
            wait=True,
        )
        logger.info(f"Upserted {len(points)} points with dense_3large vectors")
        return True

    except Exception as e:
        logger.error(f"Upsert failed: {e}")
        return False


def test_search():
    """Test search with 3072d vector."""
    config = EmbeddingConfig(
        openai_model="text-embedding-3-large",
        openai_dimensions=3072,
    )
    embedder = OpenAIEmbedder(config)
    client = get_qdrant_client()

    query = "transformer language model"
    query_emb = embedder.embed_single(query)

    logger.info(f"Query embedding: {len(query_emb)} dims")

    try:
        results = client.client.query_points(
            collection_name=TEST_COLLECTION,
            query=query_emb,
            using="dense_3large",
            limit=3,
            with_payload=True,
        )

        logger.info(f"Search results ({len(results.points)} hits):")
        for r in results.points:
            logger.info(f"  - {r.payload.get('content', 'N/A')[:50]}... (score: {r.score:.4f})")

        return True

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return False


def cleanup():
    """Delete test collection."""
    client = get_qdrant_client()

    try:
        client.client.delete_collection(TEST_COLLECTION)
        logger.info(f"Deleted test collection '{TEST_COLLECTION}'")
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")


def main():
    logger.info("=" * 60)
    logger.info("Testing text-embedding-3-large (3072 dims)")
    logger.info("=" * 60)

    # Step 1: Create test collection
    logger.info("\n[1/4] Creating test collection with dense_3large (3072d)...")
    if not create_test_collection():
        return 1

    # Step 2: Generate embeddings
    logger.info("\n[2/4] Generating 3072d embeddings...")
    embeddings = test_3large_embedding()
    if not embeddings:
        return 1

    # Step 3: Upsert to Qdrant
    logger.info("\n[3/4] Upserting to Qdrant...")
    if not test_qdrant_upsert(embeddings):
        return 1

    # Step 4: Test search
    logger.info("\n[4/4] Testing search...")
    if not test_search():
        return 1

    # Cleanup
    logger.info("\n[Cleanup] Removing test points...")
    cleanup()

    logger.info("\n" + "=" * 60)
    logger.info("✓ text-embedding-3-large (3072d) test PASSED")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())

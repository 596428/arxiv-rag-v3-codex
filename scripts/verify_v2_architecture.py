#!/usr/bin/env python3
"""Lightweight v3 architecture verification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_metadata_db() -> bool:
    print("\n[1/5] Checking metadata DB client...")
    try:
        from src.storage import get_db_client
        client = get_db_client()
        count = client.get_paper_count()
        print(f"  Client: {type(client).__name__}")
        print(f"  Paper count: {count}")
        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def check_qdrant_connection() -> bool:
    print("\n[2/5] Checking Qdrant connection...")
    try:
        from src.storage.qdrant_client import get_qdrant_client
        client = get_qdrant_client()
        healthy = client.health_check()
        if not healthy:
            print("  [WARN] Qdrant not healthy")
            return False
        info = client.get_collection_info()
        print(f"  Collection: {info.get('name', 'N/A')}")
        print(f"  Points: {info.get('points_count', 0)}")
        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [WARN] {e}")
        return False


def check_embedding_models() -> bool:
    print("\n[3/5] Checking embedding model conversions...")
    try:
        from src.embedding.models import Chunk, ChunkType, EmbeddedChunk, SparseVector
        chunk = Chunk(
            chunk_id="test_paper_chunk_0",
            paper_id="test_paper",
            content="This is a test chunk about transformers.",
            section_title="Introduction",
            chunk_type=ChunkType.TEXT,
            chunk_index=0,
            token_count=10,
        )
        embedded = EmbeddedChunk(
            chunk=chunk,
            embedding_dense=[0.1] * 1024,
            embedding_sparse=SparseVector(indices=[1, 2, 3], values=[0.5, 0.3, 0.2]),
            embedding_openai=[0.2] * 3072,
        )
        qdrant_dict = embedded.to_qdrant_dict()
        supabase_dict = embedded.to_supabase_dict()
        assert "dense_3large" in qdrant_dict, "Missing dense_3large"
        assert "chunk_index" in supabase_dict, "Missing top-level chunk_index"
        print(f"  to_qdrant_dict() keys: {list(qdrant_dict.keys())}")
        print(f"  to_supabase_dict() keys: {list(supabase_dict.keys())}")
        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def check_qdrant_retriever() -> bool:
    print("\n[4/5] Checking QdrantHybridRetriever...")
    try:
        from src.rag.qdrant_retriever import QdrantHybridRetriever, qdrant_hybrid_search
        assert hasattr(QdrantHybridRetriever, "search")
        assert hasattr(QdrantHybridRetriever, "search_dense_only")
        assert hasattr(QdrantHybridRetriever, "search_sparse_only")
        print("  QdrantHybridRetriever interface looks correct")
        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def check_reprocess_script() -> bool:
    print("\n[5/5] Checking Phase 2 reprocess script...")
    try:
        path = Path(__file__).parent.parent / "scripts" / "11_reprocess_pipeline.py"
        assert path.exists(), "scripts/11_reprocess_pipeline.py not found"
        print(f"  Found: {path}")
        print("  [PASS]")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify v3 architecture setup")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.parse_args()

    print("=" * 60)
    print("arXiv RAG v3 - Architecture Verification")
    print("=" * 60)

    results = [
        ("Metadata DB", check_metadata_db()),
        ("Qdrant", check_qdrant_connection()),
        ("Embedding models", check_embedding_models()),
        ("Qdrant retriever", check_qdrant_retriever()),
        ("Reprocess script", check_reprocess_script()),
    ]

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\nPassed: {passed}/{len(results)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

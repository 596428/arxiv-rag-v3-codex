#!/usr/bin/env python3
"""Phase 2 reprocess pipeline for parsing and embedding refresh."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.collection.models import PaperStatus
from src.embedding.models import ChunkingConfig, EmbeddingConfig
from src.parsing.models import ParseMethod, ParsedDocument
from src.storage import get_db_client, get_qdrant_client
from src.utils.config import get_settings
from src.utils.logging import get_logger, setup_logging

logger = get_logger("reprocess")


def load_script_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parse_module = load_script_module("parse_pipeline_module", ROOT / "scripts" / "02_parse.py")
embed_module = load_script_module("embed_pipeline_module", ROOT / "scripts" / "03_embed.py")


def backup_parsed_outputs(parsed_dir: Path, arxiv_ids: list[str]) -> Path | None:
    existing = [parsed_dir / f"{arxiv_id.replace('/', '_')}.json" for arxiv_id in arxiv_ids]
    existing = [path for path in existing if path.exists()]
    if not existing:
        return None

    backup_dir = parsed_dir.parent / f"parsed_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for file_path in existing:
        shutil.copy2(file_path, backup_dir / file_path.name)
    return backup_dir


def load_paper_ids_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def get_target_papers(
    db_client,
    settings,
    arxiv_id: str | None,
    limit: int | None,
    offset: int,
    status_filter: str | None,
    paper_ids_file: Path | None,
) -> list[dict]:
    if arxiv_id:
        paper = db_client.get_paper(arxiv_id)
        if paper:
            return [paper]
        return [{
            "arxiv_id": arxiv_id,
            "pdf_path": str(settings.pdf_dir / f"{arxiv_id.replace('/', '_')}.pdf"),
            "latex_path": str(settings.latex_dir / f"{arxiv_id.replace('/', '_')}.tar.gz"),
        }]

    if paper_ids_file:
        ordered_ids = load_paper_ids_file(paper_ids_file)
        rows = []
        for paper_id in ordered_ids:
            paper = db_client.get_paper(paper_id)
            if paper:
                rows.append(paper)
        if offset:
            rows = rows[offset:]
        if limit is not None:
            rows = rows[:limit]
        return rows

    fetch_limit = limit + offset if limit is not None else None
    papers = db_client.get_papers(
        fields=["arxiv_id", "pdf_path", "latex_path", "citation_count", "parse_status", "parse_method"],
        limit=fetch_limit,
        order_by="citation_count",
        status=status_filter,
    )
    papers = [paper for paper in papers if paper.get("latex_path")]
    if offset:
        papers = papers[offset:]
    if limit is not None:
        papers = papers[:limit]
    return papers


def reparse_papers(
    papers: list[dict],
    settings,
    latex_only: bool,
    marker_only: bool,
    with_equations: bool,
    max_equations: int,
    dry_run: bool,
) -> tuple[list[str], dict[str, ParseMethod], list[str]]:
    pipeline = parse_module.ParsingPipeline(
        settings=settings,
        latex_only=latex_only,
        marker_only=marker_only,
        with_equations=with_equations,
        max_equations_per_paper=max_equations,
    )

    parsed_ids: list[str] = []
    methods: dict[str, ParseMethod] = {}
    failed_ids: list[str] = []

    for idx, paper in enumerate(papers, start=1):
        arxiv_id = paper["arxiv_id"]
        pdf_path = Path(paper["pdf_path"]) if paper.get("pdf_path") else None
        latex_path = Path(paper["latex_path"]) if paper.get("latex_path") else None

        logger.info("[%s/%s] Parsing %s", idx, len(papers), arxiv_id)
        doc, error = pipeline.parse_paper(arxiv_id, latex_path=latex_path, pdf_path=pdf_path)
        if not doc:
            logger.warning("Failed to parse %s: %s", arxiv_id, error)
            failed_ids.append(arxiv_id)
            continue

        if not dry_run:
            output_path = pipeline.save_parsed_document(doc)
            logger.info("Saved parsed document: %s", output_path)
        parsed_ids.append(arxiv_id)
        methods[arxiv_id] = doc.parse_method

    return parsed_ids, methods, failed_ids


def reset_existing_embeddings(db_client, qdrant_client, arxiv_ids: list[str], recreate_qdrant: bool) -> None:
    if recreate_qdrant:
        qdrant_client.ensure_collection(recreate=True)
    else:
        qdrant_client.ensure_collection(recreate=False)
        for arxiv_id in arxiv_ids:
            qdrant_client.delete_by_paper_id(arxiv_id)

    for arxiv_id in arxiv_ids:
        deleted = db_client.delete_chunks_by_paper(arxiv_id)
        logger.info("Deleted %s local chunks for %s", deleted, arxiv_id)


def mark_parsed(db_client, parsed_ids: list[str], methods: dict[str, ParseMethod]) -> None:
    for arxiv_id in parsed_ids:
        db_client.update_paper_status(arxiv_id, PaperStatus.PARSED, parse_method=methods.get(arxiv_id))


def mark_failed(db_client, failed_ids: list[str]) -> None:
    for arxiv_id in failed_ids:
        db_client.update_paper_status(arxiv_id, PaperStatus.FAILED)


def load_selected_parsed_documents(parsed_dir: Path, arxiv_ids: list[str]) -> list[ParsedDocument]:
    documents: list[ParsedDocument] = []
    for arxiv_id in arxiv_ids:
        json_path = parsed_dir / f"{arxiv_id.replace('/', '_')}.json"
        if not json_path.exists():
            logger.warning("Parsed JSON missing for %s: %s", arxiv_id, json_path)
            continue
        documents.append(ParsedDocument.from_json_file(str(json_path)))
    logger.info("Loaded %s parsed documents for embedding", len(documents))
    return documents


def embed_parsed_docs(
    arxiv_ids: list[str],
    parsed_dir: Path,
    device: str,
    with_openai: bool,
    dry_run: bool,
    bge_batch_size: int,
) -> tuple[object, object]:
    chunking_config = ChunkingConfig(
        max_tokens=512,
        overlap_tokens=50,
        include_abstract=True,
        add_paper_context=False,
    )
    embedding_config = EmbeddingConfig(
        use_bge=True,
        bge_batch_size=bge_batch_size,
        sparse_top_k=128,
        device=device,
        use_openai=with_openai,
    )
    documents = load_selected_parsed_documents(parsed_dir, arxiv_ids)
    return embed_module.run_embedding_pipeline(
        documents=documents,
        chunking_config=chunking_config,
        embedding_config=embedding_config,
        with_openai=with_openai,
        dry_run=dry_run,
        save_to_db=not dry_run,
        metadata_only=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Reprocess parse + embedding pipeline")
    parser.add_argument("--arxiv-id", type=str, help="Reprocess a single paper")
    parser.add_argument("--paper-ids-file", type=Path, help="Optional text file with arXiv IDs, one per line")
    parser.add_argument("--limit", type=int, help="Limit papers for this batch")
    parser.add_argument("--offset", type=int, default=0, help="Skip N selected papers before processing")
    parser.add_argument("--status-filter", type=str, help="Optional parse_status filter when selecting from DB")
    parser.add_argument("--with-openai", action="store_true", help="Include OpenAI 3-large embeddings")
    parser.add_argument("--with-equations", action="store_true", help="Generate equation descriptions during parse")
    parser.add_argument("--max-equations", type=int, default=20, help="Max equations per paper")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu", help="Embedding device")
    parser.add_argument("--bge-batch-size", type=int, default=None, help="Override BGE batch size")
    parser.add_argument("--recreate-qdrant", action="store_true", help="Recreate Qdrant collection before embedding")
    parser.add_argument("--skip-backup", action="store_true", help="Skip parsed JSON backup")
    parser.add_argument("--skip-embed", action="store_true", help="Only reparse and update parse status")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, do not mutate files or DB")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(level="DEBUG" if args.verbose else "INFO")
    settings = get_settings()
    settings.ensure_directories()

    db_client = get_db_client()
    qdrant_client = get_qdrant_client()
    papers = get_target_papers(
        db_client,
        settings,
        arxiv_id=args.arxiv_id,
        limit=args.limit,
        offset=args.offset,
        status_filter=args.status_filter,
        paper_ids_file=args.paper_ids_file,
    )
    if not papers:
        logger.info("No papers selected for reprocessing")
        return 0

    arxiv_ids = [paper["arxiv_id"] for paper in papers]
    logger.info("Selected %s papers for reprocessing", len(arxiv_ids))
    logger.info("Selection preview: %s", ", ".join(arxiv_ids[:5]))

    if not args.skip_backup and not args.dry_run:
        backup_dir = backup_parsed_outputs(settings.parsed_dir, arxiv_ids)
        if backup_dir:
            logger.info("Backed up parsed JSON to %s", backup_dir)

    parsed_ids, methods, failed_ids = reparse_papers(
        papers,
        settings=settings,
        latex_only=False,
        marker_only=False,
        with_equations=args.with_equations,
        max_equations=args.max_equations,
        dry_run=args.dry_run,
    )
    if not parsed_ids:
        logger.error("No papers were successfully parsed")
        return 1

    logger.info("Parse summary: success=%s failed=%s", len(parsed_ids), len(failed_ids))

    if args.dry_run:
        logger.info("Dry run complete. Parsed papers: %s", ", ".join(parsed_ids))
        return 0

    mark_parsed(db_client, parsed_ids, methods)
    if failed_ids:
        mark_failed(db_client, failed_ids)

    if args.skip_embed:
        logger.info("Skipping embedding stage by request")
        return 0

    reset_existing_embeddings(db_client, qdrant_client, parsed_ids, recreate_qdrant=args.recreate_qdrant)

    if args.bge_batch_size is not None:
        bge_batch_size = args.bge_batch_size
    else:
        bge_batch_size = 8 if args.device == "cpu" else 32

    chunking_stats, embedding_stats = embed_parsed_docs(
        arxiv_ids=parsed_ids,
        parsed_dir=settings.parsed_dir,
        device=args.device,
        with_openai=args.with_openai,
        dry_run=False,
        bge_batch_size=bge_batch_size,
    )

    logger.info("%s", chunking_stats.summary())
    logger.info("%s", embedding_stats.summary())
    logger.info("Reprocess complete: parsed=%s embedded=%s failed=%s", len(parsed_ids), embedding_stats.bge_embedded, len(failed_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

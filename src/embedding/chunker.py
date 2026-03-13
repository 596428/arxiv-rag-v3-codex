"""
arXiv RAG v1 - Hybrid Chunker

Section-based chunking with paragraph-level splitting for long sections.
"""

from typing import Generator

import tiktoken

from ..parsing.models import ParsedDocument, Section, Paragraph, Equation
from ..utils.logging import get_logger
from .models import Chunk, ChunkType, ChunkingConfig, ChunkingStats

logger = get_logger("chunker")


class HybridChunker:
    """
    Hybrid chunking strategy for academic papers.

    Strategy:
    1. Abstract as separate chunk
    2. Section-based primary chunking
    3. Paragraph-based splitting for oversized sections
    4. Token-based overlap for context continuity
    """

    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self._tokenizer = None
        self.stats = ChunkingStats()

    @property
    def tokenizer(self):
        """Lazy load tiktoken tokenizer."""
        if self._tokenizer is None:
            try:
                self._tokenizer = tiktoken.get_encoding(self.config.tokenizer_name)
            except Exception:
                # Fallback to cl100k_base
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        # allowed_special="all" to handle <|endoftext|> and similar tokens in papers
        return len(self.tokenizer.encode(text, allowed_special="all"))

    def _build_paper_context(self, doc: ParsedDocument) -> str:
        """
        Build paper context prefix for chunks.

        Args:
            doc: Parsed document

        Returns:
            Context string to prepend to chunks
        """
        if not self.config.add_paper_context:
            return ""

        # Build context: Paper title + truncated abstract
        context_parts = []

        if doc.title:
            context_parts.append(f"Paper: {doc.title}")

        if doc.abstract:
            # Truncate abstract to fit within token budget
            abstract_budget = self.config.paper_context_tokens - 20  # Reserve for "Paper:" prefix
            abstract_tokens = self.tokenizer.encode(doc.abstract, allowed_special="all")

            if len(abstract_tokens) > abstract_budget:
                truncated = self.tokenizer.decode(abstract_tokens[:abstract_budget])
                # Clean truncation at word boundary
                truncated = truncated.rsplit(" ", 1)[0] + "..."
                context_parts.append(f"Topic: {truncated}")
            else:
                # Use first sentence or two as topic
                sentences = doc.abstract.split(". ")[:2]
                topic = ". ".join(sentences)
                if not topic.endswith("."):
                    topic += "."
                context_parts.append(f"Topic: {topic}")

        return "\n".join(context_parts) + "\n\n" if context_parts else ""

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        """
        Chunk a parsed document into retrieval-ready pieces.

        Args:
            doc: Parsed document to chunk

        Returns:
            List of chunks
        """
        chunks = []
        chunk_index = 0

        # Build paper context prefix (if enabled)
        paper_context = self._build_paper_context(doc)

        # 1. Abstract chunk (if enabled and present)
        if self.config.include_abstract and doc.abstract:
            abstract_chunk = self._create_abstract_chunk(doc, chunk_index)
            if abstract_chunk:
                chunks.append(abstract_chunk)
                chunk_index += 1
                self.stats.abstract_chunks += 1

        # 2. Section-based chunking
        for section in doc.sections:
            section_chunks = list(self._chunk_section(doc.arxiv_id, section, chunk_index, paper_context))
            for chunk in section_chunks:
                chunks.append(chunk)
                chunk_index += 1

        # 3. Equation chunks (if enabled)
        if self.config.include_equations and doc.equations:
            for chunk in self._create_equation_chunks(doc, chunk_index, paper_context):
                chunks.append(chunk)
                chunk_index += 1
                self.stats.equation_chunks += 1

        # 4. Add overlap markers
        self._mark_overlaps(chunks)

        # 5. Update stats
        self.stats.total_papers += 1
        self.stats.total_chunks += len(chunks)
        self.stats.total_tokens += sum(c.token_count for c in chunks)

        if chunks:
            tokens = [c.token_count for c in chunks]
            self.stats.min_tokens = min(self.stats.min_tokens, min(tokens)) if self.stats.min_tokens else min(tokens)
            self.stats.max_tokens = max(self.stats.max_tokens, max(tokens))

        logger.info(f"Chunked {doc.arxiv_id}: {len(chunks)} chunks, {sum(c.token_count for c in chunks)} tokens")
        return chunks

    def _create_abstract_chunk(self, doc: ParsedDocument, chunk_index: int) -> Chunk | None:
        """Create chunk from abstract."""
        if not doc.abstract or not doc.abstract.strip():
            return None

        content = doc.abstract.strip()
        token_count = self.count_tokens(content)

        # Skip if too short
        if token_count < self.config.min_chunk_tokens:
            return None

        return Chunk(
            chunk_id=f"{doc.arxiv_id}_chunk_{chunk_index}",
            paper_id=doc.arxiv_id,
            content=content,
            section_id=None,
            section_title="Abstract",
            paragraph_ids=[],
            chunk_type=ChunkType.ABSTRACT,
            chunk_index=chunk_index,
            token_count=token_count,
            char_count=len(content),
            metadata={"is_abstract": True, "paper_title": doc.title},
        )

    def _create_equation_chunks(
        self, doc: ParsedDocument, start_index: int, paper_context: str = ""
    ) -> Generator[Chunk, None, None]:
        """
        Create chunks from equation text descriptions.

        Only equations with text_description (Gemini-generated) are chunked.
        Includes LaTeX and context in metadata for reference.

        Args:
            doc: Parsed document with equations
            start_index: Starting chunk index
            paper_context: Paper context prefix to prepend

        Yields:
            Chunks for each equation with description
        """
        chunk_index = start_index

        for eq in doc.equations:
            # Skip equations without text description
            if not eq.text_description or not eq.text_description.strip():
                continue

            # Build rich content: description + context
            content_parts = []

            # Add context before if available
            if eq.context_before:
                content_parts.append(f"Context: {eq.context_before.strip()}")

            # Main description
            content_parts.append(eq.text_description.strip())

            # Add context after if available
            if eq.context_after:
                content_parts.append(f"Following context: {eq.context_after.strip()}")

            content = "\n\n".join(content_parts)

            # Prepend paper context if provided
            if paper_context:
                content = paper_context + content

            token_count = self.count_tokens(content)

            # For equations, use a much lower minimum threshold (10 tokens)
            # since equation descriptions are naturally shorter but valuable for retrieval
            min_equation_tokens = 10
            if token_count < min_equation_tokens:
                continue

            yield Chunk(
                chunk_id=f"{doc.arxiv_id}_chunk_{chunk_index}",
                paper_id=doc.arxiv_id,
                content=content,
                section_id=eq.section_id,
                section_title="Equation",
                paragraph_ids=[],
                chunk_type=ChunkType.EQUATION,
                chunk_index=chunk_index,
                token_count=token_count,
                char_count=len(content),
                metadata={
                    "equation_id": eq.equation_id,
                    "latex": eq.latex,
                    "label": eq.label,
                    "is_inline": eq.is_inline,
                },
            )
            chunk_index += 1

    def _chunk_section(
        self, arxiv_id: str, section: Section, start_index: int, paper_context: str = ""
    ) -> Generator[Chunk, None, None]:
        """
        Chunk a section, splitting if too long.

        Yields chunks from this section and its subsections.
        """
        chunk_index = start_index

        # Get all paragraphs from section
        paragraphs = section.paragraphs
        if not paragraphs:
            # Process subsections directly
            for subsection in section.subsections:
                for chunk in self._chunk_section(arxiv_id, subsection, chunk_index, paper_context):
                    yield chunk
                    chunk_index += 1
            return

        # Calculate total tokens for section
        section_text = section.full_text
        total_tokens = self.count_tokens(section_text)

        # Account for paper context in token budget
        context_tokens = self.count_tokens(paper_context) if paper_context else 0
        effective_max = self.config.max_tokens - context_tokens

        # If section fits in one chunk
        if total_tokens <= effective_max:
            if total_tokens >= self.config.min_chunk_tokens:
                # Prepend paper context
                content = paper_context + section_text if paper_context else section_text
                chunk = Chunk(
                    chunk_id=f"{arxiv_id}_chunk_{chunk_index}",
                    paper_id=arxiv_id,
                    content=content,
                    section_id=section.section_id,
                    section_title=section.title,
                    paragraph_ids=[p.paragraph_id for p in paragraphs],
                    chunk_type=ChunkType.TEXT,
                    chunk_index=chunk_index,
                    token_count=self.count_tokens(content),
                    char_count=len(content),
                    metadata={"section_level": section.level, "has_paper_context": bool(paper_context)},
                )
                yield chunk
                self.stats.text_chunks += 1
                chunk_index += 1
        else:
            # Split by paragraphs
            for chunk in self._split_paragraphs(
                arxiv_id, section.section_id, section.title, paragraphs, chunk_index, paper_context
            ):
                yield chunk
                self.stats.text_chunks += 1
                chunk_index += 1

        # Process subsections
        for subsection in section.subsections:
            for chunk in self._chunk_section(arxiv_id, subsection, chunk_index, paper_context):
                yield chunk
                chunk_index += 1

    def _split_paragraphs(
        self,
        arxiv_id: str,
        section_id: str,
        section_title: str,
        paragraphs: list[Paragraph],
        start_index: int,
        paper_context: str = "",
    ) -> Generator[Chunk, None, None]:
        """
        Split paragraphs into chunks respecting token limits.

        Uses greedy accumulation with overlap.
        """
        chunk_index = start_index
        current_paragraphs: list[Paragraph] = []
        current_tokens = 0
        current_text_parts: list[str] = []

        # Account for paper context in token budget
        context_tokens = self.count_tokens(paper_context) if paper_context else 0
        effective_max = self.config.max_tokens - context_tokens

        for para in paragraphs:
            para_tokens = self.count_tokens(para.content)

            # If single paragraph exceeds max, split it further
            if para_tokens > effective_max:
                # Flush current buffer first
                if current_paragraphs:
                    chunk = self._create_chunk_from_paragraphs(
                        arxiv_id, section_id, section_title,
                        current_paragraphs, current_text_parts, chunk_index, paper_context
                    )
                    yield chunk
                    chunk_index += 1
                    current_paragraphs = []
                    current_text_parts = []
                    current_tokens = 0

                # Split large paragraph
                for chunk in self._split_large_paragraph(
                    arxiv_id, section_id, section_title, para, chunk_index, paper_context
                ):
                    yield chunk
                    chunk_index += 1
                continue

            # Check if adding this paragraph exceeds limit
            if current_tokens + para_tokens > effective_max:
                # Create chunk from current buffer
                if current_paragraphs:
                    chunk = self._create_chunk_from_paragraphs(
                        arxiv_id, section_id, section_title,
                        current_paragraphs, current_text_parts, chunk_index, paper_context
                    )
                    yield chunk
                    chunk_index += 1

                    # Start new buffer with overlap from last paragraph
                    if self.config.overlap_tokens > 0 and current_text_parts:
                        overlap_text = self._get_overlap_text(current_text_parts[-1])
                        if overlap_text:
                            current_text_parts = [overlap_text]
                            current_tokens = self.count_tokens(overlap_text)
                        else:
                            current_text_parts = []
                            current_tokens = 0
                    else:
                        current_text_parts = []
                        current_tokens = 0

                    current_paragraphs = []

            # Add paragraph to buffer
            current_paragraphs.append(para)
            current_text_parts.append(para.content)
            current_tokens += para_tokens

        # Flush remaining buffer
        if current_paragraphs and current_tokens >= self.config.min_chunk_tokens:
            chunk = self._create_chunk_from_paragraphs(
                arxiv_id, section_id, section_title,
                current_paragraphs, current_text_parts, chunk_index, paper_context
            )
            yield chunk

    def _create_chunk_from_paragraphs(
        self,
        arxiv_id: str,
        section_id: str,
        section_title: str,
        paragraphs: list[Paragraph],
        text_parts: list[str],
        chunk_index: int,
        paper_context: str = "",
    ) -> Chunk:
        """Create a chunk from accumulated paragraphs."""
        content = "\n\n".join(text_parts)

        # Prepend paper context if provided
        if paper_context:
            content = paper_context + content

        token_count = self.count_tokens(content)

        return Chunk(
            chunk_id=f"{arxiv_id}_chunk_{chunk_index}",
            paper_id=arxiv_id,
            content=content,
            section_id=section_id,
            section_title=section_title,
            paragraph_ids=[p.paragraph_id for p in paragraphs],
            chunk_type=ChunkType.TEXT,
            chunk_index=chunk_index,
            token_count=token_count,
            char_count=len(content),
            metadata={"has_paper_context": bool(paper_context)},
        )

    def _split_large_paragraph(
        self,
        arxiv_id: str,
        section_id: str,
        section_title: str,
        para: Paragraph,
        start_index: int,
        paper_context: str = "",
    ) -> Generator[Chunk, None, None]:
        """
        Split a large paragraph by sentences.

        Uses sentence boundaries where possible.
        """
        chunk_index = start_index
        text = para.content

        # Split by sentences (simple heuristic)
        sentences = self._split_sentences(text)

        current_sentences: list[str] = []
        current_tokens = 0

        # Account for paper context in token budget
        context_tokens = self.count_tokens(paper_context) if paper_context else 0
        effective_max = self.config.max_tokens - context_tokens

        for sentence in sentences:
            sent_tokens = self.count_tokens(sentence)

            # If single sentence exceeds limit, truncate
            if sent_tokens > effective_max:
                # Flush current
                if current_sentences:
                    content = " ".join(current_sentences)
                    if paper_context:
                        content = paper_context + content
                    yield Chunk(
                        chunk_id=f"{arxiv_id}_chunk_{chunk_index}",
                        paper_id=arxiv_id,
                        content=content,
                        section_id=section_id,
                        section_title=section_title,
                        paragraph_ids=[para.paragraph_id],
                        chunk_type=ChunkType.TEXT,
                        chunk_index=chunk_index,
                        token_count=self.count_tokens(content),
                        char_count=len(content),
                        metadata={"split_from_large_paragraph": True, "has_paper_context": bool(paper_context)},
                    )
                    chunk_index += 1
                    current_sentences = []
                    current_tokens = 0

                # Truncate long sentence
                truncated = self._truncate_to_tokens(sentence, effective_max)
                if paper_context:
                    truncated = paper_context + truncated
                yield Chunk(
                    chunk_id=f"{arxiv_id}_chunk_{chunk_index}",
                    paper_id=arxiv_id,
                    content=truncated,
                    section_id=section_id,
                    section_title=section_title,
                    paragraph_ids=[para.paragraph_id],
                    chunk_type=ChunkType.TEXT,
                    chunk_index=chunk_index,
                    token_count=self.count_tokens(truncated),
                    char_count=len(truncated),
                    metadata={"truncated": True, "has_paper_context": bool(paper_context)},
                )
                chunk_index += 1
                continue

            if current_tokens + sent_tokens > effective_max:
                # Create chunk
                if current_sentences:
                    content = " ".join(current_sentences)
                    if paper_context:
                        content = paper_context + content
                    yield Chunk(
                        chunk_id=f"{arxiv_id}_chunk_{chunk_index}",
                        paper_id=arxiv_id,
                        content=content,
                        section_id=section_id,
                        section_title=section_title,
                        paragraph_ids=[para.paragraph_id],
                        chunk_type=ChunkType.TEXT,
                        chunk_index=chunk_index,
                        token_count=self.count_tokens(content),
                        char_count=len(content),
                        metadata={"split_from_large_paragraph": True, "has_paper_context": bool(paper_context)},
                    )
                    chunk_index += 1

                # Overlap
                if self.config.overlap_tokens > 0 and current_sentences:
                    overlap = self._get_overlap_sentences(current_sentences)
                    current_sentences = overlap
                    current_tokens = sum(self.count_tokens(s) for s in overlap)
                else:
                    current_sentences = []
                    current_tokens = 0

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        # Flush remaining
        if current_sentences and current_tokens >= self.config.min_chunk_tokens:
            content = " ".join(current_sentences)
            if paper_context:
                content = paper_context + content
            yield Chunk(
                chunk_id=f"{arxiv_id}_chunk_{chunk_index}",
                paper_id=arxiv_id,
                content=content,
                section_id=section_id,
                section_title=section_title,
                paragraph_ids=[para.paragraph_id],
                chunk_type=ChunkType.TEXT,
                chunk_index=chunk_index,
                token_count=self.count_tokens(content),
                char_count=len(content),
                metadata={"split_from_large_paragraph": True, "has_paper_context": bool(paper_context)},
            )

    def _split_sentences(self, text: str) -> list[str]:
        """Simple sentence splitting."""
        import re
        # Split on sentence-ending punctuation followed by space
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from end of previous chunk."""
        tokens = self.tokenizer.encode(text, allowed_special="all")
        if len(tokens) <= self.config.overlap_tokens:
            return text
        overlap_tokens = tokens[-self.config.overlap_tokens:]
        return self.tokenizer.decode(overlap_tokens)

    def _get_overlap_sentences(self, sentences: list[str]) -> list[str]:
        """Get sentences for overlap."""
        if not sentences:
            return []

        # Take last sentence(s) up to overlap token limit
        overlap: list[str] = []
        total_tokens = 0

        for sent in reversed(sentences):
            sent_tokens = self.count_tokens(sent)
            if total_tokens + sent_tokens > self.config.overlap_tokens:
                break
            overlap.insert(0, sent)
            total_tokens += sent_tokens

        return overlap

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to max tokens."""
        tokens = self.tokenizer.encode(text, allowed_special="all")
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)

    def _mark_overlaps(self, chunks: list[Chunk]) -> None:
        """Mark chunks that have overlap with neighbors."""
        for i, chunk in enumerate(chunks):
            if i > 0 and chunks[i - 1].paper_id == chunk.paper_id:
                # Check for actual overlap
                prev_content = chunks[i - 1].content
                curr_content = chunk.content

                # Simple overlap detection
                if self.config.overlap_tokens > 0:
                    prev_end = prev_content[-200:] if len(prev_content) > 200 else prev_content
                    curr_start = curr_content[:200] if len(curr_content) > 200 else curr_content

                    # Check for common substring
                    if len(set(prev_end.split()) & set(curr_start.split())) > 5:
                        chunk.has_overlap_before = True
                        chunks[i - 1].has_overlap_after = True
                        chunk.overlap_tokens = self.config.overlap_tokens
                        self.stats.chunks_with_overlap += 1


def chunk_papers(
    documents: list[ParsedDocument],
    config: ChunkingConfig = None
) -> tuple[list[Chunk], ChunkingStats]:
    """
    Chunk multiple documents.

    Args:
        documents: List of parsed documents
        config: Chunking configuration

    Returns:
        Tuple of (all chunks, statistics)
    """
    chunker = HybridChunker(config)
    all_chunks = []

    for doc in documents:
        try:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to chunk {doc.arxiv_id}: {e}")

    return all_chunks, chunker.stats

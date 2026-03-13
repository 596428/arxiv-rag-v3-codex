-- arXiv RAG v3 - Local PostgreSQL schema

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS papers (
    id SERIAL PRIMARY KEY,
    arxiv_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    authors TEXT[] DEFAULT '{}',
    abstract TEXT,
    categories TEXT[] DEFAULT '{}',
    published_date DATE,
    updated_date DATE,
    citation_count INTEGER DEFAULT 0,
    download_count INTEGER DEFAULT 0,
    pdf_url TEXT,
    latex_url TEXT,
    pdf_path TEXT,
    latex_path TEXT,
    parse_status TEXT DEFAULT 'pending',
    parse_method TEXT,
    is_llm_relevant BOOLEAN,
    relevance_reason TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    paper_id TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    section_title TEXT,
    chunk_type TEXT DEFAULT 'text',
    chunk_index INTEGER,
    token_count INTEGER,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS equations (
    id SERIAL PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    equation_id TEXT UNIQUE NOT NULL,
    latex TEXT NOT NULL,
    text_description TEXT,
    is_inline BOOLEAN DEFAULT FALSE,
    label TEXT,
    section_id TEXT,
    context_before TEXT,
    context_after TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS figures (
    id SERIAL PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    figure_id TEXT UNIQUE NOT NULL,
    image_path TEXT,
    caption TEXT,
    label TEXT,
    section_id TEXT,
    figure_number INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tables (
    id SERIAL PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    table_id TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    caption TEXT,
    label TEXT,
    section_id TEXT,
    table_number INTEGER,
    headers JSONB DEFAULT '[]'::jsonb,
    row_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS citation_edges (
    id SERIAL PRIMARY KEY,
    source_arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    target_arxiv_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (source_arxiv_id, target_arxiv_id)
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    paper_id TEXT REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    chunk_id TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(parse_status);
CREATE INDEX IF NOT EXISTS idx_papers_citation ON papers(citation_count DESC);
CREATE INDEX IF NOT EXISTS idx_papers_published_date ON papers(published_date DESC);
CREATE INDEX IF NOT EXISTS idx_chunks_paper_chunk_index ON chunks(paper_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_content_trgm ON chunks USING gin (content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_equations_paper ON equations(paper_id);
CREATE INDEX IF NOT EXISTS idx_figures_paper ON figures(paper_id);
CREATE INDEX IF NOT EXISTS idx_tables_paper ON tables(paper_id);
CREATE INDEX IF NOT EXISTS idx_citation_edges_source ON citation_edges(source_arxiv_id);
CREATE INDEX IF NOT EXISTS idx_citation_edges_target ON citation_edges(target_arxiv_id);
CREATE INDEX IF NOT EXISTS idx_entities_paper ON entities(paper_id);
CREATE INDEX IF NOT EXISTS idx_entities_chunk ON entities(chunk_id);

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_papers_updated_at ON papers;
CREATE TRIGGER update_papers_updated_at
    BEFORE UPDATE ON papers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_chunks_updated_at ON chunks;
CREATE TRIGGER update_chunks_updated_at
    BEFORE UPDATE ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_equations_updated_at ON equations;
CREATE TRIGGER update_equations_updated_at
    BEFORE UPDATE ON equations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_figures_updated_at ON figures;
CREATE TRIGGER update_figures_updated_at
    BEFORE UPDATE ON figures
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_tables_updated_at ON tables;
CREATE TRIGGER update_tables_updated_at
    BEFORE UPDATE ON tables
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

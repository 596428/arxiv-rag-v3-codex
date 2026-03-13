from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------------------------
    # Supabase
    # -------------------------------------------
    db_backend: str = Field(default="local", description="Database backend: local or supabase")
    supabase_url: str = Field(default="", description="Supabase project URL")
    supabase_key: str = Field(default="", description="Supabase anon key")
    supabase_service_key: str = Field(default="", description="Supabase service role key")

    # -------------------------------------------
    # Local PostgreSQL
    # -------------------------------------------
    pg_host: str = Field(default="localhost", description="PostgreSQL host")
    pg_port: int = Field(default=5432, description="PostgreSQL port")
    pg_database: str = Field(default="arxiv_rag", description="PostgreSQL database name")
    pg_user: str = Field(default="", description="PostgreSQL user")
    pg_password: str = Field(default="", description="PostgreSQL password")

    # -------------------------------------------
    # Vector DB
    # -------------------------------------------
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant base URL")

    # -------------------------------------------
    # AI APIs
    # -------------------------------------------
    gemini_api_key: str = Field(default="", description="Gemini API key")
    gemini_model: str = Field(default="gemini-3-flash-preview", description="Gemini model name")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    semantic_scholar_api_key: str = Field(default="", description="Semantic Scholar API key")

    # -------------------------------------------
    # Paths
    # -------------------------------------------
    data_dir: Path = Field(default=Path("./data"), description="Root data directory")
    pdf_dir: Path = Field(default=Path("./data/pdfs"), description="PDF storage")
    latex_dir: Path = Field(default=Path("./data/latex"), description="LaTeX source storage")
    parsed_dir: Path = Field(default=Path("./data/parsed"), description="Parsed JSON storage")
    figures_dir: Path = Field(default=Path("./data/figures"), description="Extracted figures")
    cache_dir: Path = Field(default=Path("./data/cache"), description="Cache directory")

    # -------------------------------------------
    # Processing Parameters
    # -------------------------------------------
    batch_size: int = Field(default=10, description="Batch size for processing")
    max_concurrent_downloads: int = Field(default=5, description="Max concurrent downloads")
    chunk_max_tokens: int = Field(default=512, description="Maximum tokens per chunk")
    chunk_overlap_tokens: int = Field(default=50, description="Overlap tokens between chunks")

    # -------------------------------------------
    # API Rate Limits
    # -------------------------------------------
    arxiv_request_interval: float = Field(default=3.0, description="Seconds between arXiv requests")
    semantic_scholar_requests_per_minute: int = Field(default=20, description="S2 rate limit")

    # -------------------------------------------
    # UI
    # -------------------------------------------
    streamlit_port: int = Field(default=8501, description="Streamlit port")
    fastapi_port: int = Field(default=8000, description="FastAPI port")

    def ensure_directories(self) -> None:
        """Create all data directories if they don't exist."""
        for path in [
            self.data_dir,
            self.pdf_dir,
            self.latex_dir,
            self.parsed_dir,
            self.figures_dir,
            self.cache_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def has_supabase(self) -> bool:
        """Check if Supabase is configured."""
        return bool(self.supabase_url and self.supabase_key)

    @property
    def has_postgres(self) -> bool:
        """Check if local PostgreSQL is configured."""
        return bool(self.pg_host and self.pg_database and self.pg_user and self.pg_password)

    @property
    def qdrant_host(self) -> str:
        """Extract Qdrant host from URL if present."""
        parsed = urlparse(self.qdrant_url)
        return parsed.hostname or "localhost"

    @property
    def qdrant_port(self) -> int:
        """Extract Qdrant port from URL if present."""
        parsed = urlparse(self.qdrant_url)
        return parsed.port or 6333

    @property
    def has_gemini(self) -> bool:
        """Check if Gemini API is configured."""
        return bool(self.gemini_api_key)

    @property
    def has_openai(self) -> bool:
        """Check if OpenAI API is configured."""
        return bool(self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience alias
settings = get_settings()

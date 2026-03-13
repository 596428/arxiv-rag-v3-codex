"""
arXiv RAG v1 - FastAPI Main Application

REST API for hybrid vector search using Qdrant + Supabase.

Endpoints:
- POST /api/v1/search      - Hybrid vector search
- POST /api/v1/chat        - RAG chat with LLM
- GET  /api/v1/papers      - List papers
- GET  /api/v1/papers/{id} - Get paper details
- GET  /api/v1/health      - Health check
"""

import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..utils.logging import get_logger
from ..storage.qdrant_client import get_qdrant_client

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting arXiv RAG API...")

    # Initialize Qdrant client
    qdrant = get_qdrant_client()
    if qdrant.health_check():
        logger.info("Qdrant connection healthy")
    else:
        logger.warning("Qdrant connection failed - some features may be unavailable")

    yield

    # Shutdown
    logger.info("Shutting down arXiv RAG API...")
    qdrant.close()


# Create FastAPI app
app = FastAPI(
    title="arXiv RAG API",
    description="Hybrid vector search API for LLM research papers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS configuration
# Allow GitHub Pages and local development
ALLOWED_ORIGINS = [
    "https://ajh428.github.io",  # GitHub Pages (old)
    "https://596428.github.io",  # GitHub Pages (current)
    "https://acacia.chat",       # Custom domain
    "https://api.acacia.chat",   # API domain
    "http://localhost:3000",     # Local dev
    "http://localhost:8080",     # Local dev
    "http://localhost:9090",     # Local dev
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:9090",
]

# In development, allow all origins
if os.getenv("ENV", "production") == "development":
    ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time", "X-RateLimit-Remaining"],
)


# ==========================================
# RATE LIMITING (In-memory, IP-based)
# ==========================================

# Configuration
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))  # requests per window
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # window in seconds

# In-memory store: {ip: [(timestamp, count)]}
rate_limit_store: dict[str, list[float]] = defaultdict(list)


def get_client_ip(request: Request) -> str:
    """Extract client IP from request, considering proxies."""
    # Check X-Forwarded-For header (Cloudflare, proxies)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    # Check CF-Connecting-IP (Cloudflare specific)
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip
    # Fallback to direct client
    return request.client.host if request.client else "unknown"


def is_rate_limited(ip: str) -> tuple[bool, int]:
    """Check if IP is rate limited. Returns (is_limited, remaining_requests)."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    # Clean old entries
    rate_limit_store[ip] = [ts for ts in rate_limit_store[ip] if ts > window_start]

    # Check limit
    current_count = len(rate_limit_store[ip])
    remaining = max(0, RATE_LIMIT_REQUESTS - current_count)

    if current_count >= RATE_LIMIT_REQUESTS:
        return True, 0

    # Record this request
    rate_limit_store[ip].append(now)
    return False, remaining - 1


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limit requests by IP address."""
    # Skip rate limiting for health checks and docs
    if request.url.path in ["/", "/docs", "/redoc", "/openapi.json", "/api/v1/health"]:
        return await call_next(request)

    # Skip in development mode
    if os.getenv("ENV") == "development":
        return await call_next(request)

    client_ip = get_client_ip(request)
    is_limited, remaining = is_rate_limited(client_ip)

    if is_limited:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "detail": f"Maximum {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds",
                "retry_after": RATE_LIMIT_WINDOW,
            },
            headers={
                "Retry-After": str(RATE_LIMIT_WINDOW),
                "X-RateLimit-Remaining": "0",
            },
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    return response


# Request timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add response timing header."""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Response-Time"] = f"{process_time:.0f}ms"
    return response


# Health check endpoint
@app.get("/api/v1/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Service health status
    """
    qdrant = get_qdrant_client()
    qdrant_healthy = qdrant.health_check()

    status = "healthy" if qdrant_healthy else "degraded"

    return {
        "status": status,
        "version": "1.0.0",
        "services": {
            "qdrant": "healthy" if qdrant_healthy else "unhealthy",
        }
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "arXiv RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# Import and include routers
from .routes import search, papers, chat

app.include_router(search.router, prefix="/api/v1", tags=["Search"])
app.include_router(papers.router, prefix="/api/v1", tags=["Papers"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("ENV") == "development" else None,
        }
    )

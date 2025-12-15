"""
Ingestion pipeline for real-time legal document processing.

This module implements the production-grade ingestion architecture described
in the legal RAG framework, including:
- Webhook listeners with HMAC signature verification
- Asynchronous queue-based processing
- Distributed rate limiting
- Worker pool management
"""

from src.ingestion.webhook_listener import WebhookListener
from src.ingestion.queue_manager import QueueManager, QueueBackend
from src.ingestion.rate_limiter import DistributedRateLimiter

__all__ = [
    "WebhookListener",
    "QueueManager",
    "QueueBackend",
    "DistributedRateLimiter",
]

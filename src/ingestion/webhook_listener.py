"""
CourtListener Webhook Listener with HMAC Signature Verification.

Implements the secure webhook architecture described in Section 3.2 of the
legal RAG framework. Key features:
- HMAC-SHA256 signature verification (constant-time comparison)
- Idempotency checking to prevent duplicate processing
- Instant 200 OK response (< 10ms target)
- Queue-based decoupling of receipt and processing

Security Model:
- X-Signature header contains HMAC-SHA256(payload, secret)
- Constant-time comparison prevents timing attacks
- Invalid signatures are rejected immediately
- Replay protection via idempotency database

References:
- CourtListener Webhooks: https://www.courtlistener.com/help/api/webhooks/
- Section 3.2.1: Establishing the Webhook Handshake
- Section 7.2.2: Security - HMAC Validation
"""

import hmac
import hashlib
import json
import time
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
import redis


class WebhookListener:
    """
    Secure webhook listener for CourtListener events.

    Validates HMAC signatures, checks idempotency, and enqueues payloads
    for asynchronous processing.
    """

    def __init__(
        self,
        secret_key: str,
        queue_manager: Any,  # QueueManager instance
        redis_client: Optional[redis.Redis] = None,
        idempotency_ttl: int = 86400,  # 24 hours
    ):
        """
        Initialize webhook listener.

        Args:
            secret_key: Shared secret for HMAC verification (from CourtListener)
            queue_manager: Queue manager instance for enqueueing payloads
            redis_client: Redis client for idempotency checking (None = in-memory)
            idempotency_ttl: Time-to-live for idempotency records (seconds)
        """
        self.secret_key = secret_key.encode()
        self.queue_manager = queue_manager
        self.redis_client = redis_client
        self.idempotency_ttl = idempotency_ttl

        # In-memory fallback for idempotency (if Redis unavailable)
        self._idempotency_cache: Dict[str, float] = {}

        # Metrics
        self.metrics = {
            "total_webhooks": 0,
            "valid_signatures": 0,
            "invalid_signatures": 0,
            "duplicate_webhooks": 0,
            "enqueued_webhooks": 0,
            "errors": 0,
        }

        # Initialize FastAPI app
        self.app = FastAPI(title="CourtListener Webhook Listener")
        self._register_routes()

    def _register_routes(self):
        """Register FastAPI routes."""

        @self.app.post("/webhook/courtlistener")
        async def handle_webhook(request: Request):
            """
            Handle incoming CourtListener webhook.

            Expected workflow (< 10ms total):
            1. Read raw body
            2. Verify HMAC signature (< 1ms)
            3. Check idempotency (< 1ms)
            4. Enqueue payload (< 5ms)
            5. Return 200 OK

            Returns:
                JSONResponse with status
            """
            start_time = time.time()
            self.metrics["total_webhooks"] += 1

            try:
                # Step 1: Read raw body
                raw_body = await request.body()

                # Step 2: Verify HMAC signature
                signature = request.headers.get("X-Signature", "")
                if not self._verify_signature(raw_body, signature):
                    self.metrics["invalid_signatures"] += 1
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid HMAC signature",
                    )

                self.metrics["valid_signatures"] += 1

                # Step 3: Parse payload
                try:
                    payload = json.loads(raw_body)
                except json.JSONDecodeError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid JSON payload",
                    )

                # Step 4: Check idempotency
                event_id = self._extract_event_id(payload)
                if self._is_duplicate(event_id):
                    self.metrics["duplicate_webhooks"] += 1
                    # Still return 200 OK (acknowledge receipt)
                    elapsed_ms = (time.time() - start_time) * 1000
                    return JSONResponse(
                        status_code=status.HTTP_200_OK,
                        content={
                            "status": "duplicate",
                            "event_id": event_id,
                            "processing_time_ms": round(elapsed_ms, 2),
                        },
                    )

                # Step 5: Mark as seen
                self._mark_seen(event_id)

                # Step 6: Enqueue for processing
                self.queue_manager.enqueue(payload)
                self.metrics["enqueued_webhooks"] += 1

                # Step 7: Return 200 OK
                elapsed_ms = (time.time() - start_time) * 1000
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content={
                        "status": "enqueued",
                        "event_id": event_id,
                        "processing_time_ms": round(elapsed_ms, 2),
                    },
                )

            except HTTPException:
                raise
            except Exception as e:
                self.metrics["errors"] += 1
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Internal error: {str(e)}",
                )

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "queue_backend": self.queue_manager.backend.value,
                "redis_connected": self.redis_client is not None,
            }

        @self.app.get("/metrics")
        async def get_metrics():
            """Metrics endpoint for monitoring."""
            return self.metrics

    def _verify_signature(self, payload: bytes, header_signature: str) -> bool:
        """
        Verify HMAC-SHA256 signature using constant-time comparison.

        Implements Section 7.2.2: HMAC Validation
        Uses hmac.compare_digest() to prevent timing attacks.

        Args:
            payload: Raw webhook payload (bytes)
            header_signature: Signature from X-Signature header

        Returns:
            True if signature is valid, False otherwise
        """
        if not header_signature:
            return False

        # Calculate expected signature
        expected_signature = hmac.new(
            self.secret_key,
            payload,
            hashlib.sha256,
        ).hexdigest()

        # Constant-time comparison (prevents timing attacks)
        return hmac.compare_digest(expected_signature, header_signature)

    def _extract_event_id(self, payload: Dict) -> str:
        """
        Extract unique event ID from payload.

        CourtListener webhooks include:
        - id: Document/docket ID
        - date_modified: Timestamp

        We combine these for idempotency key.

        Args:
            payload: Webhook payload

        Returns:
            Unique event ID
        """
        doc_id = payload.get("id", "unknown")
        date_modified = payload.get("date_modified", "unknown")
        return f"{doc_id}:{date_modified}"

    def _is_duplicate(self, event_id: str) -> bool:
        """
        Check if event has been seen before (idempotency check).

        Uses Redis if available, otherwise falls back to in-memory cache
        with TTL-based cleanup.

        Args:
            event_id: Unique event identifier

        Returns:
            True if duplicate, False if new
        """
        if self.redis_client:
            # Redis-based idempotency (production)
            key = f"webhook:seen:{event_id}"
            return self.redis_client.exists(key) > 0
        else:
            # In-memory fallback (development)
            # Clean expired entries
            now = time.time()
            self._idempotency_cache = {
                k: v
                for k, v in self._idempotency_cache.items()
                if now - v < self.idempotency_ttl
            }
            return event_id in self._idempotency_cache

    def _mark_seen(self, event_id: str):
        """
        Mark event as seen (for idempotency).

        Args:
            event_id: Unique event identifier
        """
        if self.redis_client:
            # Redis-based (production)
            key = f"webhook:seen:{event_id}"
            self.redis_client.setex(key, self.idempotency_ttl, "1")
        else:
            # In-memory fallback (development)
            self._idempotency_cache[event_id] = time.time()

    def get_app(self) -> FastAPI:
        """
        Get FastAPI application instance.

        Returns:
            FastAPI app for deployment (uvicorn, gunicorn, etc.)
        """
        return self.app


# Example usage
if __name__ == "__main__":
    import uvicorn
    from src.ingestion.queue_manager import QueueManager, QueueBackend

    # Initialize components
    queue_manager = QueueManager(backend=QueueBackend.REDIS)
    listener = WebhookListener(
        secret_key="your-secret-key-here",
        queue_manager=queue_manager,
        redis_client=redis.Redis(host="localhost", port=6379, db=0),
    )

    # Run server
    uvicorn.run(listener.get_app(), host="0.0.0.0", port=8000)

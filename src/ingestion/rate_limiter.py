"""
Distributed Rate Limiter for CourtListener API.

Implements the token bucket algorithm with Redis-based coordination
described in Section 3.4 and 7.3 of the legal RAG framework.

Key Features:
- Token bucket algorithm for smooth rate limiting
- Distributed coordination via Redis (multi-worker safe)
- Header-based dynamic adaptation (X-RateLimit headers)
- Proactive backoff to prevent 429 errors
- Circuit breaker for API downtime

Architecture:
```
Worker 1 ──┐
Worker 2 ──┼──> Redis Counter ──> CourtListener API
Worker N ──┘     (Atomic INCR)      (5000 req/hour)
```

The token bucket metaphor:
- Bucket holds N tokens (e.g., 5000)
- Each API request costs 1 token
- Bucket refills every hour
- If bucket empty, wait until refill

References:
- Section 3.4: Rate Limit Management and Resilience
- Section 3.4.1: The Token Bucket Algorithm
- Section 3.4.2: Header-Based Dynamic Backoff
- Section 7.1.2: Handling API Downtime and Retries
- Section 7.3: Managing Rate Limits
"""

import time
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import redis


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistributedRateLimiter:
    """
    Distributed rate limiter using Redis token bucket.

    Coordinates rate limiting across multiple workers to respect
    CourtListener API limits (5000 requests/hour).
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        rate_limit: int = 5000,  # CourtListener default
        window_seconds: int = 3600,  # 1 hour
        safety_buffer: int = 50,  # Stop at 4950/5000
        key_prefix: str = "courtlistener:ratelimit",
    ):
        """
        Initialize distributed rate limiter.

        Args:
            redis_client: Redis client for distributed coordination
            rate_limit: Maximum requests per window
            window_seconds: Time window in seconds (default: 1 hour)
            safety_buffer: Stop N requests before limit (prevents hard 429)
            key_prefix: Redis key prefix
        """
        self.redis = redis_client
        self.rate_limit = rate_limit
        self.window_seconds = window_seconds
        self.safety_buffer = safety_buffer
        self.key_prefix = key_prefix

        # Dynamic state (updated from API response headers)
        self.last_remaining: Optional[int] = None
        self.last_reset_time: Optional[float] = None

        # Circuit breaker state
        self.circuit_open = False
        self.circuit_open_until: Optional[float] = None
        self.consecutive_failures = 0
        self.failure_threshold = 5  # Trip circuit after 5 consecutive failures

        logger.info(
            f"Initialized rate limiter: {rate_limit} req/{window_seconds}s, "
            f"buffer={safety_buffer}"
        )

    def wait_if_needed(self):
        """
        Check rate limit and wait if necessary.

        This method should be called BEFORE every API request.
        It will block if the rate limit is exceeded or circuit is open.
        """
        # Check circuit breaker
        if self.circuit_open:
            if time.time() < self.circuit_open_until:
                wait_seconds = self.circuit_open_until - time.time()
                logger.warning(
                    f"Circuit breaker open, waiting {wait_seconds:.1f}s..."
                )
                time.sleep(wait_seconds)
            else:
                # Reset circuit breaker
                logger.info("Circuit breaker reset, resuming API calls")
                self.circuit_open = False
                self.consecutive_failures = 0

        # Get current count
        current_count = self._get_current_count()

        # Check if we're approaching the limit
        effective_limit = self.rate_limit - self.safety_buffer

        if current_count >= effective_limit:
            # Wait until window resets
            wait_until = self._get_window_reset_time()
            wait_seconds = wait_until - time.time()

            if wait_seconds > 0:
                logger.warning(
                    f"Rate limit approaching ({current_count}/{self.rate_limit}), "
                    f"waiting {wait_seconds:.1f}s until reset..."
                )
                time.sleep(wait_seconds)

        # Increment counter atomically
        self._increment_count()

    def _get_current_count(self) -> int:
        """
        Get current request count for this window.

        Returns:
            Current request count
        """
        key = self._get_redis_key()
        count = self.redis.get(key)
        return int(count) if count else 0

    def _increment_count(self):
        """
        Atomically increment request count.

        Uses Redis INCR for atomic increment across workers.
        Sets TTL on first request of window.
        """
        key = self._get_redis_key()

        # Atomic increment
        new_count = self.redis.incr(key)

        # Set TTL if this is the first request
        if new_count == 1:
            self.redis.expire(key, self.window_seconds)

        logger.debug(f"Request count: {new_count}/{self.rate_limit}")

    def _get_redis_key(self) -> str:
        """
        Get Redis key for current time window.

        Uses time-based key to automatically rotate windows.
        Format: "courtlistener:ratelimit:2025-12-10:14" (hour bucket)

        Returns:
            Redis key for current window
        """
        # Round timestamp to window boundary
        now = int(time.time())
        window_id = now // self.window_seconds
        return f"{self.key_prefix}:{window_id}"

    def _get_window_reset_time(self) -> float:
        """
        Get Unix timestamp when current window resets.

        Returns:
            Reset time (Unix timestamp)
        """
        now = int(time.time())
        window_id = now // self.window_seconds
        reset_time = (window_id + 1) * self.window_seconds
        return float(reset_time)

    def update_from_headers(self, response_headers: Dict[str, str]):
        """
        Update rate limiter state from API response headers.

        CourtListener provides:
        - X-RateLimit-Limit: Total limit
        - X-RateLimit-Remaining: Requests remaining
        - X-RateLimit-Reset: Unix timestamp when limit resets

        This allows dynamic adaptation to actual API state.

        Args:
            response_headers: HTTP response headers from CourtListener
        """
        remaining = response_headers.get("X-RateLimit-Remaining")
        reset_time = response_headers.get("X-RateLimit-Reset")
        limit = response_headers.get("X-RateLimit-Limit")

        if remaining:
            self.last_remaining = int(remaining)
            logger.debug(f"API rate limit remaining: {self.last_remaining}")

        if reset_time:
            self.last_reset_time = float(reset_time)

        if limit:
            reported_limit = int(limit)
            if reported_limit != self.rate_limit:
                logger.warning(
                    f"API rate limit changed: {self.rate_limit} -> {reported_limit}"
                )
                self.rate_limit = reported_limit

        # Proactive backoff: If remaining is low, slow down preemptively
        if self.last_remaining and self.last_remaining < self.safety_buffer:
            logger.warning(
                f"Rate limit critically low ({self.last_remaining} remaining), "
                f"entering proactive backoff..."
            )
            # Wait until reset time
            if self.last_reset_time:
                wait_seconds = self.last_reset_time - time.time()
                if wait_seconds > 0:
                    time.sleep(wait_seconds)

    def record_success(self):
        """
        Record successful API call.

        Resets circuit breaker failure counter.
        """
        if self.consecutive_failures > 0:
            self.consecutive_failures = 0
            logger.debug("Reset circuit breaker failure counter")

    def record_failure(self, status_code: int):
        """
        Record failed API call.

        Implements circuit breaker pattern for 5xx errors.
        - 429 (Rate Limit): Immediate backoff until reset
        - 5xx (Server Error): Trip circuit after N failures

        Args:
            status_code: HTTP status code from failed request
        """
        if status_code == 429:
            # Rate limit exceeded (should not happen with proper limiting)
            logger.error(
                "Received 429 Too Many Requests! Rate limiter failed to prevent this."
            )
            # Emergency backoff: 5 minutes
            self._trip_circuit(backoff_seconds=300)

        elif 500 <= status_code < 600:
            # Server error
            self.consecutive_failures += 1
            logger.warning(
                f"API server error ({status_code}), "
                f"failures: {self.consecutive_failures}/{self.failure_threshold}"
            )

            if self.consecutive_failures >= self.failure_threshold:
                # Trip circuit breaker
                # Exponential backoff: 2^failures seconds (capped at 30 min)
                backoff_seconds = min(2 ** self.consecutive_failures, 1800)
                self._trip_circuit(backoff_seconds)

    def _trip_circuit(self, backoff_seconds: int):
        """
        Trip circuit breaker (stop all requests).

        Args:
            backoff_seconds: How long to keep circuit open
        """
        self.circuit_open = True
        self.circuit_open_until = time.time() + backoff_seconds

        logger.error(
            f"Circuit breaker TRIPPED! Stopping all API requests for "
            f"{backoff_seconds}s (until {datetime.fromtimestamp(self.circuit_open_until)})"
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics.

        Returns:
            Dictionary with stats
        """
        current_count = self._get_current_count()
        reset_time = self._get_window_reset_time()

        return {
            "current_count": current_count,
            "rate_limit": self.rate_limit,
            "remaining": self.rate_limit - current_count,
            "utilization_pct": (current_count / self.rate_limit) * 100,
            "reset_time": datetime.fromtimestamp(reset_time).isoformat(),
            "seconds_until_reset": max(0, reset_time - time.time()),
            "circuit_open": self.circuit_open,
            "consecutive_failures": self.consecutive_failures,
            "last_remaining": self.last_remaining,
        }


class LocalRateLimiter:
    """
    Local (single-process) rate limiter fallback.

    Uses in-memory token bucket when Redis is not available.
    NOT SAFE for distributed deployments (multiple workers).
    """

    def __init__(
        self,
        rate_limit: int = 5000,
        window_seconds: int = 3600,
        safety_buffer: int = 50,
    ):
        """
        Initialize local rate limiter.

        Args:
            rate_limit: Maximum requests per window
            window_seconds: Time window in seconds
            safety_buffer: Stop N requests before limit
        """
        self.rate_limit = rate_limit
        self.window_seconds = window_seconds
        self.safety_buffer = safety_buffer

        # Token bucket state
        self.tokens = rate_limit
        self.last_refill = time.time()

        logger.warning(
            "Using LOCAL rate limiter (not safe for multiple workers). "
            "Use Redis-based DistributedRateLimiter for production."
        )

    def wait_if_needed(self):
        """Wait if token bucket is empty."""
        self._refill_tokens()

        if self.tokens <= self.safety_buffer:
            # Wait for refill
            time_since_refill = time.time() - self.last_refill
            wait_seconds = self.window_seconds - time_since_refill

            if wait_seconds > 0:
                logger.warning(
                    f"Rate limit approaching ({self.tokens}/{self.rate_limit}), "
                    f"waiting {wait_seconds:.1f}s..."
                )
                time.sleep(wait_seconds)
                self._refill_tokens()

        # Consume token
        self.tokens -= 1

    def _refill_tokens(self):
        """Refill token bucket if window has elapsed."""
        now = time.time()
        time_since_refill = now - self.last_refill

        if time_since_refill >= self.window_seconds:
            self.tokens = self.rate_limit
            self.last_refill = now
            logger.info("Token bucket refilled")

    def update_from_headers(self, response_headers: Dict[str, str]):
        """Update from response headers (no-op for local limiter)."""
        pass

    def record_success(self):
        """Record success (no-op for local limiter)."""
        pass

    def record_failure(self, status_code: int):
        """Record failure (no-op for local limiter)."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get stats."""
        return {
            "current_count": self.rate_limit - self.tokens,
            "rate_limit": self.rate_limit,
            "remaining": self.tokens,
            "utilization_pct": ((self.rate_limit - self.tokens) / self.rate_limit) * 100,
        }


# Example usage
if __name__ == "__main__":
    import redis

    # Initialize Redis-based rate limiter
    redis_client = redis.Redis(host="localhost", port=6379, db=0)
    limiter = DistributedRateLimiter(redis_client=redis_client)

    # Simulate API calls
    for i in range(10):
        limiter.wait_if_needed()
        print(f"Request {i+1} allowed")
        limiter.record_success()

        # Simulate response headers
        limiter.update_from_headers({
            "X-RateLimit-Remaining": str(5000 - i - 1),
            "X-RateLimit-Limit": "5000",
            "X-RateLimit-Reset": str(int(time.time()) + 3600),
        })

    # Print stats
    print("\nRate Limiter Stats:")
    print(limiter.get_stats())

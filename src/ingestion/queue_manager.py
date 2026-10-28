"""
Queue Manager for Asynchronous Document Processing.

Implements the decoupled queue architecture described in Section 3.2.2 and 7.2.1
of the legal RAG framework. Supports multiple queue backends:
- Redis (lightweight, good for moderate scale)
- RabbitMQ (robust, message persistence, complex routing)
- AWS SQS (managed service, serverless)

Key Features:
- Abstract interface for backend-agnostic code
- Retry logic with exponential backoff
- Dead letter queue for failed messages
- Poison pill detection (messages that repeatedly fail)

Architecture Pattern:
```
Webhook → Queue → Worker Pool → Vector DB
   |         ↑
   ↓         |
Validate  Retry/DLQ
```

References:
- Section 3.2.2: The Asynchronous Processing Queue
- Section 7.2.1: The Decoupled Pattern
"""

import json
import time
import logging
from enum import Enum
from typing import Dict, Optional, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict

import redis


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueueBackend(Enum):
    """Supported queue backends."""

    REDIS = "redis"
    RABBITMQ = "rabbitmq"
    SQS = "sqs"


@dataclass
class QueueMessage:
    """
    Standard queue message format.

    All messages follow this structure regardless of backend.
    """

    payload: Dict[str, Any]  # Webhook payload
    enqueued_at: float  # Unix timestamp
    retry_count: int = 0  # Number of retry attempts
    max_retries: int = 3  # Maximum retry attempts
    message_id: Optional[str] = None  # Backend-specific message ID


class BaseQueueBackend(ABC):
    """Abstract base class for queue backends."""

    @abstractmethod
    def enqueue(self, message: QueueMessage) -> bool:
        """
        Enqueue a message for processing.

        Args:
            message: Message to enqueue

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def dequeue(self, timeout: int = 10) -> Optional[QueueMessage]:
        """
        Dequeue a message for processing (blocking).

        Args:
            timeout: Timeout in seconds

        Returns:
            Message if available, None if timeout
        """
        pass

    @abstractmethod
    def ack(self, message: QueueMessage):
        """
        Acknowledge successful processing of message.

        Args:
            message: Message to acknowledge
        """
        pass

    @abstractmethod
    def nack(self, message: QueueMessage, requeue: bool = True):
        """
        Negative acknowledge (processing failed).

        Args:
            message: Message that failed
            requeue: Whether to requeue for retry
        """
        pass

    @abstractmethod
    def dead_letter(self, message: QueueMessage, reason: str):
        """
        Move message to dead letter queue.

        Args:
            message: Message that repeatedly failed
            reason: Reason for failure
        """
        pass

    @abstractmethod
    def get_queue_depth(self) -> int:
        """
        Get current queue depth.

        Returns:
            Number of messages in queue
        """
        pass


class RedisQueueBackend(BaseQueueBackend):
    """
    Redis-based queue backend using LIST operations.

    Uses Redis LIST for main queue and HASH for DLQ.
    Lightweight and fast, good for moderate scale (< 1M messages/day).
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        queue_name: str = "courtlistener:ingestion",
        dlq_name: str = "courtlistener:dlq",
    ):
        """
        Initialize Redis queue backend.

        Args:
            redis_client: Redis client instance
            queue_name: Name of main queue
            dlq_name: Name of dead letter queue
        """
        self.redis = redis_client
        self.queue_name = queue_name
        self.dlq_name = dlq_name

    def enqueue(self, message: QueueMessage) -> bool:
        """Enqueue message to Redis LIST."""
        try:
            serialized = json.dumps(asdict(message))
            self.redis.rpush(self.queue_name, serialized)
            logger.debug(f"Enqueued message to {self.queue_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to enqueue message: {e}")
            return False

    def dequeue(self, timeout: int = 10) -> Optional[QueueMessage]:
        """Dequeue message from Redis LIST (blocking)."""
        try:
            # BLPOP: Blocking left pop with timeout
            result = self.redis.blpop(self.queue_name, timeout=timeout)
            if result is None:
                return None

            _, serialized = result
            data = json.loads(serialized)
            message = QueueMessage(**data)
            logger.debug(f"Dequeued message from {self.queue_name}")
            return message

        except Exception as e:
            logger.error(f"Failed to dequeue message: {e}")
            return None

    def ack(self, message: QueueMessage):
        """
        Acknowledge message (no-op for Redis LIST).

        Redis LIST doesn't have built-in ACK mechanism.
        Message is already removed by BLPOP.
        """
        pass

    def nack(self, message: QueueMessage, requeue: bool = True):
        """
        Negative acknowledge message.

        Increments retry count and requeues if under max_retries.
        """
        message.retry_count += 1

        if requeue and message.retry_count < message.max_retries:
            logger.warning(
                f"Requeuing message (attempt {message.retry_count}/{message.max_retries})"
            )
            self.enqueue(message)
        else:
            logger.error(
                f"Message exceeded max retries ({message.max_retries}), moving to DLQ"
            )
            self.dead_letter(message, f"Max retries exceeded: {message.max_retries}")

    def dead_letter(self, message: QueueMessage, reason: str):
        """Move message to dead letter queue (Redis HASH)."""
        try:
            message_id = message.message_id or str(time.time())
            dlq_entry = {
                "message": json.dumps(asdict(message)),
                "reason": reason,
                "timestamp": time.time(),
            }
            self.redis.hset(self.dlq_name, message_id, json.dumps(dlq_entry))
            logger.info(f"Moved message to DLQ: {message_id}")
        except Exception as e:
            logger.error(f"Failed to move message to DLQ: {e}")

    def get_queue_depth(self) -> int:
        """Get current queue depth."""
        return self.redis.llen(self.queue_name)


class RabbitMQBackend(BaseQueueBackend):
    """
    RabbitMQ-based queue backend.

    Robust message broker with persistence, complex routing, and built-in DLQ.
    Good for high-scale production (> 1M messages/day).

    Requires: pika library
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        queue_name: str = "courtlistener.ingestion",
        dlq_name: str = "courtlistener.dlq",
    ):
        """
        Initialize RabbitMQ queue backend.

        Args:
            host: RabbitMQ host
            port: RabbitMQ port
            queue_name: Name of main queue
            dlq_name: Name of dead letter queue
        """
        try:
            import pika
        except ImportError:
            raise ImportError(
                "RabbitMQ backend requires 'pika' library. "
                "Install with: poetry add pika"
            )

        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host, port=port)
        )
        self.channel = self.connection.channel()
        self.queue_name = queue_name
        self.dlq_name = dlq_name

        # Declare main queue
        self.channel.queue_declare(
            queue=queue_name,
            durable=True,  # Persist across broker restarts
            arguments={"x-dead-letter-exchange": dlq_name},
        )

        # Declare DLQ
        self.channel.queue_declare(queue=dlq_name, durable=True)

    def enqueue(self, message: QueueMessage) -> bool:
        """Enqueue message to RabbitMQ."""
        try:
            import pika

            serialized = json.dumps(asdict(message))
            self.channel.basic_publish(
                exchange="",
                routing_key=self.queue_name,
                body=serialized,
                properties=pika.BasicProperties(delivery_mode=2),  # Persistent
            )
            logger.debug(f"Enqueued message to RabbitMQ: {self.queue_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to enqueue message to RabbitMQ: {e}")
            return False

    def dequeue(self, timeout: int = 10) -> Optional[QueueMessage]:
        """Dequeue message from RabbitMQ."""
        try:
            method, properties, body = self.channel.basic_get(
                queue=self.queue_name, auto_ack=False
            )
            if method is None:
                return None

            data = json.loads(body)
            message = QueueMessage(**data)
            message.message_id = str(method.delivery_tag)
            logger.debug(f"Dequeued message from RabbitMQ: {self.queue_name}")
            return message

        except Exception as e:
            logger.error(f"Failed to dequeue message from RabbitMQ: {e}")
            return None

    def ack(self, message: QueueMessage):
        """Acknowledge message in RabbitMQ."""
        if message.message_id:
            self.channel.basic_ack(delivery_tag=int(message.message_id))

    def nack(self, message: QueueMessage, requeue: bool = True):
        """Negative acknowledge message in RabbitMQ."""
        if message.message_id:
            self.channel.basic_nack(
                delivery_tag=int(message.message_id), requeue=requeue
            )

    def dead_letter(self, message: QueueMessage, reason: str):
        """Move message to DLQ (automatically handled by RabbitMQ)."""
        # RabbitMQ handles DLQ automatically via x-dead-letter-exchange
        logger.info(f"Message moved to DLQ by RabbitMQ: {reason}")

    def get_queue_depth(self) -> int:
        """Get current queue depth."""
        queue = self.channel.queue_declare(queue=self.queue_name, passive=True)
        return queue.method.message_count


class SQSBackend(BaseQueueBackend):
    """
    AWS SQS-based queue backend.

    Managed service, serverless, scales automatically.
    Good for cloud-native deployments.

    Requires: boto3 library
    """

    def __init__(
        self,
        queue_url: str,
        dlq_url: Optional[str] = None,
        region_name: str = "us-east-1",
    ):
        """
        Initialize SQS queue backend.

        Args:
            queue_url: SQS queue URL
            dlq_url: DLQ URL (optional)
            region_name: AWS region
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "SQS backend requires 'boto3' library. " "Install with: poetry add boto3"
            )

        self.sqs = boto3.client("sqs", region_name=region_name)
        self.queue_url = queue_url
        self.dlq_url = dlq_url

    def enqueue(self, message: QueueMessage) -> bool:
        """Enqueue message to SQS."""
        try:
            serialized = json.dumps(asdict(message))
            response = self.sqs.send_message(
                QueueUrl=self.queue_url, MessageBody=serialized
            )
            logger.debug(f"Enqueued message to SQS: {response['MessageId']}")
            return True
        except Exception as e:
            logger.error(f"Failed to enqueue message to SQS: {e}")
            return False

    def dequeue(self, timeout: int = 10) -> Optional[QueueMessage]:
        """Dequeue message from SQS."""
        try:
            response = self.sqs.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=timeout,  # Long polling
            )

            messages = response.get("Messages", [])
            if not messages:
                return None

            sqs_message = messages[0]
            data = json.loads(sqs_message["Body"])
            message = QueueMessage(**data)
            message.message_id = sqs_message["ReceiptHandle"]
            logger.debug(f"Dequeued message from SQS")
            return message

        except Exception as e:
            logger.error(f"Failed to dequeue message from SQS: {e}")
            return None

    def ack(self, message: QueueMessage):
        """Acknowledge message in SQS."""
        if message.message_id:
            self.sqs.delete_message(
                QueueUrl=self.queue_url, ReceiptHandle=message.message_id
            )

    def nack(self, message: QueueMessage, requeue: bool = True):
        """Negative acknowledge message in SQS."""
        if message.message_id and not requeue:
            # Change visibility timeout to 0 (makes it immediately available)
            self.sqs.change_message_visibility(
                QueueUrl=self.queue_url, ReceiptHandle=message.message_id, VisibilityTimeout=0
            )

    def dead_letter(self, message: QueueMessage, reason: str):
        """Move message to DLQ."""
        if self.dlq_url:
            # Send to DLQ
            dlq_entry = {"message": asdict(message), "reason": reason, "timestamp": time.time()}
            self.sqs.send_message(QueueUrl=self.dlq_url, MessageBody=json.dumps(dlq_entry))

    def get_queue_depth(self) -> int:
        """Get current queue depth."""
        response = self.sqs.get_queue_attributes(
            QueueUrl=self.queue_url, AttributeNames=["ApproximateNumberOfMessages"]
        )
        return int(response["Attributes"]["ApproximateNumberOfMessages"])


class QueueManager:
    """
    High-level queue manager with backend abstraction.

    Provides a simple interface for enqueueing/dequeueing messages
    regardless of the underlying queue backend.
    """

    def __init__(
        self,
        backend: QueueBackend = QueueBackend.REDIS,
        **backend_kwargs,
    ):
        """
        Initialize queue manager.

        Args:
            backend: Queue backend type
            **backend_kwargs: Backend-specific configuration
        """
        self.backend = backend

        # Initialize backend
        if backend == QueueBackend.REDIS:
            redis_client = backend_kwargs.get(
                "redis_client", redis.Redis(host="localhost", port=6379, db=0)
            )
            self.backend_impl = RedisQueueBackend(redis_client=redis_client)

        elif backend == QueueBackend.RABBITMQ:
            self.backend_impl = RabbitMQBackend(**backend_kwargs)

        elif backend == QueueBackend.SQS:
            self.backend_impl = SQSBackend(**backend_kwargs)

        else:
            raise ValueError(f"Unsupported queue backend: {backend}")

        logger.info(f"Initialized queue manager with backend: {backend.value}")

    def enqueue(self, payload: Dict[str, Any]) -> bool:
        """
        Enqueue a payload for processing.

        Args:
            payload: Webhook payload or document data

        Returns:
            True if successful, False otherwise
        """
        message = QueueMessage(payload=payload, enqueued_at=time.time())
        return self.backend_impl.enqueue(message)

    def dequeue(self, timeout: int = 10) -> Optional[QueueMessage]:
        """
        Dequeue a message for processing.

        Args:
            timeout: Timeout in seconds

        Returns:
            Message if available, None if timeout
        """
        return self.backend_impl.dequeue(timeout=timeout)

    def ack(self, message: QueueMessage):
        """Acknowledge successful processing."""
        self.backend_impl.ack(message)

    def nack(self, message: QueueMessage, requeue: bool = True):
        """Negative acknowledge (processing failed)."""
        self.backend_impl.nack(message, requeue=requeue)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics.

        Returns:
            Dictionary with queue stats
        """
        return {
            "backend": self.backend.value,
            "queue_depth": self.backend_impl.get_queue_depth(),
        }

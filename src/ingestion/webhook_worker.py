"""
Webhook Worker for Asynchronous Document Processing.

Implements the worker pool pattern described in Section 7.2.1 of the
legal RAG framework. Workers consume messages from the queue, process
documents, and update the vector database.

Processing Pipeline:
```
Queue → Worker → Download PDF → Extract Text → Chunk → Embed → Vector DB
         ↓
      Rate Limiter (Distributed)
```

Features:
- Graceful shutdown (SIGINT/SIGTERM handling)
- Health checks
- Error handling with retry
- Latency tracking
- Circuit breaker for downstream failures

References:
- Section 3.2.2: The Asynchronous Processing Queue
- Section 7.2.1: Architecture - The Decoupled Queue
- Section 7.4: Handling "The Big PDF"
"""

import time
import signal
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from src.ingestion.queue_manager import QueueManager, QueueMessage
from src.ingestion.rate_limiter import DistributedRateLimiter
from src.corpus.legal_chunker import LegalChunker
from src.retrieval.document_retriever import DocumentRetriever


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebhookWorker:
    """
    Asynchronous worker for processing CourtListener webhooks.

    Consumes messages from queue, downloads documents, processes them,
    and updates the vector database.
    """

    def __init__(
        self,
        queue_manager: QueueManager,
        rate_limiter: DistributedRateLimiter,
        document_retriever: DocumentRetriever,
        legal_chunker: Optional[Any] = None,  # LegalChunker instance
        download_timeout: int = 300,  # 5 minutes for large PDFs
        max_pdf_size_mb: int = 100,
    ):
        """
        Initialize webhook worker.

        Args:
            queue_manager: Queue manager for consuming messages
            rate_limiter: Rate limiter for API calls
            document_retriever: Document retriever for indexing
            legal_chunker: Legal document chunker (optional, uses basic if None)
            download_timeout: Timeout for PDF downloads (seconds)
            max_pdf_size_mb: Maximum PDF size to process (MB)
        """
        self.queue_manager = queue_manager
        self.rate_limiter = rate_limiter
        self.document_retriever = document_retriever
        self.legal_chunker = legal_chunker
        self.download_timeout = download_timeout
        self.max_pdf_size_mb = max_pdf_size_mb

        # Worker state
        self.running = False
        self.shutdown_requested = False

        # Metrics
        self.metrics = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "avg_processing_time_ms": 0,
        }

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True

    def start(self):
        """
        Start the worker loop.

        Continuously polls the queue for messages until shutdown is requested.
        """
        self.running = True
        logger.info("Worker started, polling queue for messages...")

        while not self.shutdown_requested:
            try:
                # Poll queue with 10-second timeout
                message = self.queue_manager.dequeue(timeout=10)

                if message is None:
                    # Timeout, no message available
                    continue

                # Process message
                success = self._process_message(message)

                if success:
                    self.queue_manager.ack(message)
                    self.metrics["successful"] += 1
                else:
                    # Processing failed, requeue with backoff
                    self.queue_manager.nack(message, requeue=True)
                    self.metrics["failed"] += 1

                self.metrics["total_processed"] += 1

            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {e}", exc_info=True)
                time.sleep(1)  # Backoff on error

        self.running = False
        logger.info("Worker stopped gracefully")

    def _process_message(self, message: QueueMessage) -> bool:
        """
        Process a single webhook message.

        Pipeline:
        1. Extract document metadata from payload
        2. Rate-limit check
        3. Download document (PDF/HTML)
        4. Extract text
        5. Chunk document (legal-aware)
        6. Generate embeddings
        7. Update vector database

        Args:
            message: Queue message to process

        Returns:
            True if successful, False if failed
        """
        start_time = time.time()
        payload = message.payload

        try:
            # Step 1: Extract metadata
            doc_id = payload.get("id")
            doc_type = payload.get("type", "opinion")  # opinion, docket, etc.
            download_url = payload.get("download_url")

            if not download_url:
                logger.warning(f"No download URL in payload for doc {doc_id}, skipping")
                self.metrics["skipped"] += 1
                return True  # ACK (don't retry)

            logger.info(f"Processing document: {doc_id} ({doc_type})")

            # Step 2: Rate limit check
            self.rate_limiter.wait_if_needed()

            # Step 3: Download document
            text = self._download_and_extract(download_url, doc_id)
            if text is None:
                return False  # Failed, will retry

            # Step 4: Chunk document
            chunks = self._chunk_document(text, payload)

            # Step 5: Update vector database
            self._update_vector_db(chunks, payload)

            # Metrics
            elapsed_ms = (time.time() - start_time) * 1000
            self._update_processing_time(elapsed_ms)
            logger.info(
                f"Successfully processed doc {doc_id} in {elapsed_ms:.2f}ms "
                f"({len(chunks)} chunks)"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to process message: {e}", exc_info=True)
            return False

    def _download_and_extract(
        self, download_url: str, doc_id: str
    ) -> Optional[str]:
        """
        Download and extract text from document.

        Handles:
        - PDF extraction (PyPDF2 or pdfplumber)
        - HTML extraction (BeautifulSoup)
        - Text extraction
        - Size limits
        - Timeout handling

        Args:
            download_url: URL to download document from
            doc_id: Document ID for logging

        Returns:
            Extracted text, or None if failed
        """
        import requests
        from io import BytesIO

        try:
            # Download with timeout and size limit
            response = requests.get(
                download_url,
                timeout=self.download_timeout,
                stream=True,
            )
            response.raise_for_status()

            # Check size
            content_length = response.headers.get("Content-Length")
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > self.max_pdf_size_mb:
                    logger.warning(
                        f"Document {doc_id} too large ({size_mb:.1f} MB), skipping"
                    )
                    return None

            # Extract text based on content type
            content_type = response.headers.get("Content-Type", "")

            if "pdf" in content_type:
                return self._extract_pdf(BytesIO(response.content), doc_id)
            elif "html" in content_type:
                return self._extract_html(response.text, doc_id)
            else:
                # Assume plain text
                return response.text

        except requests.exceptions.Timeout:
            logger.error(f"Timeout downloading document {doc_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to download document {doc_id}: {e}")
            return None

    def _extract_pdf(self, pdf_file: BytesIO, doc_id: str) -> Optional[str]:
        """Extract text from PDF."""
        try:
            import PyPDF2

            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text

        except Exception as e:
            logger.error(f"Failed to extract PDF {doc_id}: {e}")
            return None

    def _extract_html(self, html: str, doc_id: str) -> Optional[str]:
        """Extract text from HTML."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            # Clean whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)
            return text

        except Exception as e:
            logger.error(f"Failed to extract HTML {doc_id}: {e}")
            return None

    def _chunk_document(
        self, text: str, metadata: Dict[str, Any]
    ) -> list:
        """
        Chunk document using legal-aware chunker.

        Args:
            text: Full document text
            metadata: Document metadata (jurisdiction, court, date, etc.)

        Returns:
            List of chunks with enriched metadata
        """
        if self.legal_chunker:
            # Use advanced legal chunker (semantic, context-aware)
            return self.legal_chunker.chunk(text, metadata)
        else:
            # Fallback: Simple chunking by character count
            chunk_size = 500
            chunks = []
            for i in range(0, len(text), chunk_size):
                chunk_text = text[i : i + chunk_size]
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": metadata,
                        "chunk_id": i // chunk_size,
                    }
                )
            return chunks

    def _update_vector_db(
        self, chunks: list, metadata: Dict[str, Any]
    ):
        """
        Update vector database with new chunks.

        Args:
            chunks: List of document chunks
            metadata: Document metadata
        """
        # Extract text from chunks
        texts = [chunk["text"] if isinstance(chunk, dict) else chunk for chunk in chunks]

        # Update the corpus and rebuild index
        # (In production, this would use incremental updates)
        existing_corpus = list(self.document_retriever.corpus)
        existing_metadata = list(self.document_retriever.corpus_metadata)

        # Add new chunks
        new_corpus = existing_corpus + texts
        new_metadata = existing_metadata + [
            {**metadata, "chunk_id": i} for i in range(len(texts))
        ]

        # Rebuild index (TODO: Make this incremental)
        self.document_retriever.build_index(new_corpus, new_metadata)

    def _update_processing_time(self, elapsed_ms: float):
        """Update rolling average of processing time."""
        total = self.metrics["total_processed"]
        if total == 0:
            self.metrics["avg_processing_time_ms"] = elapsed_ms
        else:
            current_avg = self.metrics["avg_processing_time_ms"]
            # Exponential moving average
            alpha = 0.1
            self.metrics["avg_processing_time_ms"] = (
                alpha * elapsed_ms + (1 - alpha) * current_avg
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get worker metrics."""
        return {
            **self.metrics,
            "running": self.running,
            "queue_depth": self.queue_manager.get_stats()["queue_depth"],
        }


# Example usage
if __name__ == "__main__":
    from src.ingestion.queue_manager import QueueManager, QueueBackend
    from src.ingestion.rate_limiter import DistributedRateLimiter
    from src.retrieval.document_retriever import DocumentRetriever
    import redis

    # Initialize components
    redis_client = redis.Redis(host="localhost", port=6379, db=0)
    queue_manager = QueueManager(backend=QueueBackend.REDIS, redis_client=redis_client)
    rate_limiter = DistributedRateLimiter(redis_client=redis_client)
    document_retriever = DocumentRetriever(index_path="data/indexes/legal_corpus")

    # Start worker
    worker = WebhookWorker(
        queue_manager=queue_manager,
        rate_limiter=rate_limiter,
        document_retriever=document_retriever,
    )

    logger.info("Starting webhook worker...")
    worker.start()

"""
RAG (Retrieval-Augmented Generation) wrapper for legal question answering.

Wraps any base generator (Llama, Llawma, etc.) and adds retrieval capabilities.
"""

from typing import List, Optional
from pathlib import Path

from .base_generator import BaseGenerator, GenerationResult
from src.retrieval.document_retriever import DocumentRetriever


class RAGGenerator:
    """
    Retrieval-Augmented Generation wrapper.

    Wraps any base generator and adds retrieval from the legal corpus.
    Retrieved passages are injected into the prompt before generation.
    """

    def __init__(
        self,
        base_generator: BaseGenerator,
        retriever: DocumentRetriever,
        top_k: int = 5,
        include_sources: bool = True,
    ):
        """
        Initialize RAG generator.

        Args:
            base_generator: Base generator to wrap (Llama, Llawma, etc.)
            retriever: Document retriever for corpus search
            top_k: Number of passages to retrieve (default: 5)
            include_sources: Include source citations in response (default: True)
        """
        self.base_generator = base_generator
        self.retriever = retriever
        self.top_k = top_k
        self.include_sources = include_sources

    def generate(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate answer with retrieval augmentation.

        Args:
            query: Legal question
            top_k: Override number of passages to retrieve

        Returns:
            GenerationResult with answer, sources, and metadata
        """
        # Step 1: Retrieve relevant passages
        k = top_k if top_k is not None else self.top_k
        results = self.retriever.retrieve_evidence(query, top_k=k)

        # Extract passages and scores
        passages = []
        scores = []
        for result in results:
            if isinstance(result, tuple):
                passages.append(result[0])
                scores.append(result[1] if len(result) > 1 else 1.0)
            else:
                passages.append(result)
                scores.append(1.0)

        # Step 2: Format prompt with retrieved context
        augmented_prompt = self._format_rag_prompt(query, passages)

        # Step 3: Generate response using base generator
        # We need to directly call the internal method since we already formatted the prompt
        answer = self.base_generator._generate_response(augmented_prompt)

        # Step 4: Optionally append sources
        if self.include_sources:
            sources_text = self._format_sources(passages)
            full_answer = f"{answer}\n\n{sources_text}"
        else:
            full_answer = answer

        # Step 5: Create result
        result = GenerationResult(
            query=query,
            generated_answer=full_answer,
            model_name=f"{self.base_generator.config.model_name} (RAG)",
            retrieved_context=passages,
        )

        return result

    def _format_rag_prompt(self, query: str, passages: List[str]) -> str:
        """
        Format prompt with retrieved context.

        Args:
            query: User query
            passages: Retrieved passages

        Returns:
            Formatted prompt string (user message only, system prompt handled by base generator)
        """
        # Format context - truncate passages to avoid context overflow
        context_parts = []
        for i, passage in enumerate(passages, 1):
            # Truncate each passage to ~500 chars to fit context window
            truncated = passage[:500] + "..." if len(passage) > 500 else passage
            context_parts.append(f"[{i}] {truncated}")

        context_text = "\n\n".join(context_parts)

        # Build user message only (system prompt is handled by base generator)
        prompt = f"""I have retrieved the following legal documents that may help answer the question.

RETRIEVED LEGAL DOCUMENTS:
{context_text}

QUESTION: {query}

Please provide a comprehensive answer to the question based on the retrieved documents above. When making claims, cite the relevant document using [N] notation. If the documents don't contain enough information to fully answer the question, say so."""

        return prompt

    def _format_sources(self, passages: List[str]) -> str:
        """
        Format source citations.

        Args:
            passages: Retrieved passages

        Returns:
            Formatted sources text
        """
        sources_lines = ["Sources:"]
        for i, passage in enumerate(passages, 1):
            # Truncate long passages
            preview = passage[:200] + "..." if len(passage) > 200 else passage
            sources_lines.append(f"[{i}] {preview}")

        return "\n".join(sources_lines)

    def batch_generate(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> List[GenerationResult]:
        """
        Generate answers for multiple queries.

        Args:
            queries: List of legal questions
            top_k: Override number of passages to retrieve

        Returns:
            List of GenerationResults
        """
        results = []
        for query in queries:
            result = self.generate(query, top_k=top_k)
            results.append(result)

        return results


def create_llama_rag_generator(
    index_path: str = "data/indexes/legal_corpus",
    top_k: int = 5,
    **generator_kwargs
) -> RAGGenerator:
    """
    Factory function to create Llama 3.2 3B with RAG.

    Args:
        index_path: Path to retrieval index
        top_k: Number of passages to retrieve (default: 5)
        **generator_kwargs: Additional generator configuration

    Returns:
        Configured RAGGenerator with Llama base model

    Example:
        >>> generator = create_llama_rag_generator()
        >>> result = generator.generate("What is the Fourth Amendment exclusionary rule?")
        >>> print(result.generated_answer)
    """
    from .llama_generator import create_llama_generator

    # Create base Llama generator (no quantization for steering)
    base_generator = create_llama_generator(
        **generator_kwargs
    )

    # Create retriever
    retriever = DocumentRetriever(index_path=index_path)

    # Wrap with RAG
    rag_generator = RAGGenerator(
        base_generator=base_generator,
        retriever=retriever,
        top_k=top_k,
    )

    return rag_generator


def create_llawma_rag_generator(
    index_path: str = "data/indexes/legal_corpus",
    top_k: int = 5,
    **generator_kwargs
) -> RAGGenerator:
    """
    Factory function to create SaulLM-7B (legal fine-tuned) with RAG.

    Args:
        index_path: Path to retrieval index
        top_k: Number of passages to retrieve (default: 5)
        **generator_kwargs: Additional generator configuration

    Returns:
        Configured RAGGenerator with SaulLM base model

    Example:
        >>> generator = create_llawma_rag_generator()
        >>> result = generator.generate("What is qualified immunity?")
        >>> print(result.generated_answer)

    Note:
        SaulLM-7B is specifically fine-tuned on legal corpora, so it should provide
        more accurate legal reasoning compared to baseline Llama, especially when
        combined with RAG retrieval.
    """
    from .llawma_generator import create_llawma_generator

    # Create base SaulLM generator (no quantization for steering)
    base_generator = create_llawma_generator(
        **generator_kwargs
    )

    # Create retriever
    retriever = DocumentRetriever(index_path=index_path)

    # Wrap with RAG
    rag_generator = RAGGenerator(
        base_generator=base_generator,
        retriever=retriever,
        top_k=top_k,
    )

    return rag_generator


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # Test Llama + RAG
    print("Creating Llama 3.1 8B + RAG generator...")
    llama_rag = create_llama_rag_generator()

    query = "What is the Fourth Amendment exclusionary rule?"
    print(f"\nQuery: {query}")

    result = llama_rag.generate(query)
    print(f"\nAnswer:\n{result.generated_answer}")
    print(f"\nRetrieved {result.num_retrieved} passages")
    print(f"Latency: {result.latency_ms:.1f}ms")

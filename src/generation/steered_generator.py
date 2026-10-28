"""
Steering-augmented generation for Llama3.1-8B.

Uses Contrastive Activation Addition (CAA) to steer model activations
toward reduced hallucination behavior during inference.
"""

import torch
from pathlib import Path
from typing import Optional, Union

from steering_vectors import SteeringVector

from .base_generator import BaseGenerator, GenerationConfig, GenerationResult, set_seed
from .llama_generator import LlamaGenerator


class SteeredLlamaGenerator:
    """
    Llama3.1-8B generator with activation steering.

    Wraps the base LlamaGenerator and applies steering vectors
    during generation to reduce hallucinations.
    """

    def __init__(
        self,
        steering_vector_path: Union[str, Path],
        multiplier: float = 1.0,
        config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize steered generator.

        Args:
            steering_vector_path: Path to pre-trained steering vector (.pt file)
            multiplier: Steering strength (1.0 = full strength, 0 = no steering, -1 = opposite)
            config: Generator configuration (uses defaults if None)
        """
        self.steering_vector_path = Path(steering_vector_path)
        self.multiplier = multiplier

        # Initialize base generator (float16, no quantization for steering)
        self.config = config or GenerationConfig(model_name="meta-llama/Llama-3.1-8B-Instruct")
        print("Initializing SteeredLlamaGenerator...")
        print(f"Steering vector: {self.steering_vector_path}")
        print(f"Multiplier: {self.multiplier}")
        self.base_generator = LlamaGenerator(self.config)

        # Load steering vector
        print(f"Loading steering vector from {self.steering_vector_path}...")
        self.steering_vector: SteeringVector = torch.load(
            self.steering_vector_path,
            weights_only=False,
            map_location=self.base_generator.model.device,
        )

        # Move steering vector to same device as model
        for layer, tensor in self.steering_vector.layer_activations.items():
            self.steering_vector.layer_activations[layer] = tensor.to(
                self.base_generator.model.device
            )

        print(f"Steering vector loaded: layers={list(self.steering_vector.layer_activations.keys())}")
        print(f"Steering multiplier: {self.multiplier}")
        print("[OK] SteeredLlamaGenerator ready")

    def generate(self, query: str) -> GenerationResult:
        """
        Generate answer with steering applied.

        Args:
            query: Legal question

        Returns:
            GenerationResult with steered answer
        """
        # Apply steering during generation
        with self.steering_vector.apply(
            self.base_generator.model,
            multiplier=self.multiplier,
            min_token_index=0,
        ):
            # Generate using base generator's internal method
            answer = self.base_generator._generate_response(query)

        # Create result
        result = GenerationResult(
            query=query,
            generated_answer=answer,
            model_name=f"{self.config.model_name} (Steered x{self.multiplier})",
            retrieved_context=None,
        )

        return result

    def _generate_response(self, prompt: str) -> str:
        """
        Generate response with steering (for RAG wrapper compatibility).

        Args:
            prompt: Formatted prompt

        Returns:
            Generated answer text
        """
        with self.steering_vector.apply(
            self.base_generator.model,
            multiplier=self.multiplier,
            min_token_index=0,
        ):
            return self.base_generator._generate_response(prompt)


def create_steered_llama_generator(
    steering_vector_path: str = "steering_materials/Vectors/llama3.1-8b_hallu_steering_layer_15.pt",
    multiplier: float = 1.0,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    **kwargs
) -> SteeredLlamaGenerator:
    """
    Factory function to create steered Llama3.1-8B generator.

    Args:
        steering_vector_path: Path to steering vector file
        multiplier: Steering strength (default: 1.0)
        model_name: HuggingFace model name (default: Llama-3.1-8B-Instruct)
        **kwargs: Additional config options (e.g., seed=42)

    Returns:
        Configured SteeredLlamaGenerator

    Example:
        >>> generator = create_steered_llama_generator()
        >>> result = generator.generate("What is qualified immunity?")
        >>> print(result.generated_answer)
    """
    config = GenerationConfig(model_name=model_name, **kwargs)
    return SteeredLlamaGenerator(
        steering_vector_path=steering_vector_path,
        multiplier=multiplier,
        config=config,
    )


class SteeredRAGGenerator:
    """
    RAG generator with activation steering.

    Combines retrieval-augmented generation with steering vectors
    for the RAG + Steering experimental condition.
    """

    def __init__(
        self,
        steering_vector_path: Union[str, Path],
        index_path: str = "data/indexes/oracle_corpus",
        multiplier: float = 1.0,
        top_k: int = 3,
        config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize steered RAG generator.

        Args:
            steering_vector_path: Path to steering vector file
            index_path: Path to retrieval index
            multiplier: Steering strength
            top_k: Number of passages to retrieve
            config: Generator configuration
        """
        self.multiplier = multiplier
        self.top_k = top_k

        # Initialize base steered generator
        self.steered_generator = SteeredLlamaGenerator(
            steering_vector_path=steering_vector_path,
            multiplier=multiplier,
            config=config,
        )

        # Initialize retriever
        from src.retrieval.document_retriever import DocumentRetriever
        print(f"Loading retrieval index from {index_path}...")
        self.retriever = DocumentRetriever(index_path=index_path)

        print("[OK] SteeredRAGGenerator ready")

    def generate(self, query: str) -> GenerationResult:
        """
        Generate answer with RAG + steering.

        Args:
            query: Legal question

        Returns:
            GenerationResult with RAG-augmented steered answer
        """
        # Step 1: Retrieve relevant passages
        results = self.retriever.retrieve_evidence(query, top_k=self.top_k)

        # Extract passages
        passages = []
        for result in results:
            if isinstance(result, tuple):
                passages.append(result[0])
            else:
                passages.append(result)

        # Step 2: Format RAG prompt
        context_parts = []
        for i, passage in enumerate(passages, 1):
            truncated = passage[:500] + "..." if len(passage) > 500 else passage
            context_parts.append(f"[{i}] {truncated}")

        context_text = "\n\n".join(context_parts)

        augmented_prompt = f"""I have retrieved the following legal documents that may help answer the question.

RETRIEVED LEGAL DOCUMENTS:
{context_text}

QUESTION: {query}

Please provide a comprehensive answer to the question based on the retrieved documents above. When making claims, cite the relevant document using [N] notation. If the documents don't contain enough information to fully answer the question, say so."""

        # Step 3: Generate with steering applied
        answer = self.steered_generator._generate_response(augmented_prompt)

        # Step 4: Add sources
        sources_lines = ["Sources:"]
        for i, passage in enumerate(passages, 1):
            preview = passage[:200] + "..." if len(passage) > 200 else passage
            sources_lines.append(f"[{i}] {preview}")

        full_answer = f"{answer}\n\n" + "\n".join(sources_lines)

        # Create result
        result = GenerationResult(
            query=query,
            generated_answer=full_answer,
            model_name=f"{self.steered_generator.config.model_name} (RAG + Steered x{self.multiplier})",
            retrieved_context=passages,
        )

        return result


def create_steered_rag_generator(
    steering_vector_path: str = "steering_materials/Vectors/llama3.1-8b_hallu_steering_layer_15.pt",
    index_path: str = "data/indexes/oracle_corpus",
    multiplier: float = 1.0,
    top_k: int = 3,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    **kwargs
) -> SteeredRAGGenerator:
    """
    Factory function to create steered RAG generator (RAG + Steering variant).

    Args:
        steering_vector_path: Path to steering vector file
        index_path: Path to retrieval index
        multiplier: Steering strength
        top_k: Number of passages to retrieve
        model_name: HuggingFace model name (default: Llama-3.1-8B-Instruct)
        **kwargs: Additional config options (e.g., seed=42)

    Returns:
        Configured SteeredRAGGenerator

    Example:
        >>> generator = create_steered_rag_generator()
        >>> result = generator.generate("What is the exclusionary rule?")
        >>> print(result.generated_answer)
    """
    config = GenerationConfig(model_name=model_name, **kwargs)
    return SteeredRAGGenerator(
        steering_vector_path=steering_vector_path,
        index_path=index_path,
        multiplier=multiplier,
        top_k=top_k,
        config=config,
    )

"""
Base generator class and configuration for legal question answering.

Provides abstract interface for all model variants.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For full determinism (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Seed] Random state set to {seed} for reproducibility")


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    # Model settings
    model_name: str
    device: str = "cuda"  # "cuda" or "cpu"
    torch_dtype: str = "float16"  # "float16" or "bfloat16" - NO quantization for steering

    # Reproducibility
    seed: int = 42  # Random seed for reproducibility

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1

    # Legal-specific settings
    system_prompt: str = (
        "You are a legal expert assistant. Provide accurate, well-reasoned "
        "answers to legal questions. If you're unsure, say so. Always cite "
        "relevant legal principles or precedents when applicable."
    )

    # RAG settings (used by RAG variants)
    use_rag: bool = False
    retrieval_top_k: int = 5
    retrieval_index_path: str = "data/indexes/legal_corpus"


@dataclass
class GenerationResult:
    """Result from text generation."""

    query: str
    generated_answer: str
    model_name: str

    # Context (for RAG variants)
    retrieved_context: Optional[List[str]] = None

    # Metadata
    generation_time_ms: float = 0.0
    num_tokens_generated: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseGenerator(ABC):
    """
    Abstract base class for all response generators.

    Subclasses must implement:
    - _load_model(): Load the model
    - _generate_response(): Generate text from prompt
    """

    def __init__(self, config: GenerationConfig):
        """
        Initialize generator with config.

        Args:
            config: Generation configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None

        print(f"Initializing {self.__class__.__name__}...")
        print(f"Model: {config.model_name}")
        print(f"Device: {config.device}")
        print(f"Dtype: {config.torch_dtype}")

        # Set random seed for reproducibility BEFORE loading model
        set_seed(config.seed)

        self._load_model()

        print(f"[OK] {self.__class__.__name__} ready")

    @abstractmethod
    def _load_model(self):
        """Load the model and tokenizer. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _generate_response(self, prompt: str) -> str:
        """
        Generate response from prompt. Must be implemented by subclasses.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        pass

    def generate(self, query: str) -> GenerationResult:
        """
        Generate answer to legal question.

        Args:
            query: Legal question

        Returns:
            GenerationResult with answer and metadata
        """
        import time

        # Format prompt with system message
        prompt = self._format_prompt(query)

        # Generate response
        start_time = time.time()
        answer = self._generate_response(prompt)
        generation_time_ms = (time.time() - start_time) * 1000

        # Count tokens (approximate)
        num_tokens = len(answer.split())

        return GenerationResult(
            query=query,
            generated_answer=answer,
            model_name=self.config.model_name,
            generation_time_ms=generation_time_ms,
            num_tokens_generated=num_tokens,
        )

    def _format_prompt(self, query: str, context: Optional[List[str]] = None) -> str:
        """
        Format prompt with system message and optional context.

        Args:
            query: User question
            context: Optional retrieved context passages

        Returns:
            Formatted prompt
        """
        # Base prompt with system message
        if context:
            # RAG format: include retrieved context
            context_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(context)])
            prompt = f"""<|system|>
{self.config.system_prompt}

You have access to the following legal references:

{context_str}

Use these references to provide an accurate answer. Cite specific references when relevant.
<|end|>
<|user|>
{query}
<|end|>
<|assistant|>
"""
        else:
            # Baseline format: no context
            prompt = f"""<|system|>
{self.config.system_prompt}
<|end|>
<|user|>
{query}
<|end|>
<|assistant|>
"""

        return prompt

    def batch_generate(self, queries: List[str]) -> List[GenerationResult]:
        """
        Generate answers for multiple queries.

        Args:
            queries: List of legal questions

        Returns:
            List of generation results
        """
        results = []
        for i, query in enumerate(queries, 1):
            print(f"Generating {i}/{len(queries)}: {query[:50]}...")
            result = self.generate(query)
            results.append(result)

        return results

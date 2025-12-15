"""
Response generation module for legal question answering.

Supports 4 experimental variants (all Llama3.1-8B):
1. Baseline: Pure generation (no RAG, no steering)
2. RAG: Retrieval-augmented generation
3. Steering: Activation steering via CAA
4. RAG + Steering: Both interventions combined
"""

from .base_generator import BaseGenerator, GenerationConfig, GenerationResult
from .llama_generator import LlamaGenerator, create_llama_generator
from .llawma_generator import LlawmaGenerator, create_llawma_generator
from .rag_generator import (
    RAGGenerator,
    create_llama_rag_generator,
    create_llawma_rag_generator,
)
from .steered_generator import (
    SteeredLlamaGenerator,
    SteeredRAGGenerator,
    create_steered_llama_generator,
    create_steered_rag_generator,
)

__all__ = [
    # Base classes
    "BaseGenerator",
    "GenerationConfig",
    "GenerationResult",
    # Variant 1: Baseline (Llama 3.1 8B)
    "LlamaGenerator",
    "create_llama_generator",
    # Variant 2: RAG
    "RAGGenerator",
    "create_llama_rag_generator",
    # Variant 3: Steering
    "SteeredLlamaGenerator",
    "create_steered_llama_generator",
    # Variant 4: RAG + Steering
    "SteeredRAGGenerator",
    "create_steered_rag_generator",
    # Legacy (Mistral-based)
    "LlawmaGenerator",
    "create_llawma_generator",
    "create_llawma_rag_generator",
]

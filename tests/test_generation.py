"""
Test response generation with all 4 model variants.

Variants:
1. Llama 3.1 8B (baseline)
2. Llawma (legal fine-tuned)
3. Llama 3.1 8B + RAG
4. Llawma + RAG

Tests memory usage and generation quality.
"""

from pathlib import Path
import sys
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation import (
    create_llama_generator,
    create_llawma_generator,
    create_llama_rag_generator,
    create_llawma_rag_generator,
)


def check_gpu_memory():
    """Check available GPU memory."""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated_memory = torch.cuda.memory_allocated(0) / 1e9
        cached_memory = torch.cuda.memory_reserved(0) / 1e9

        print("\nGPU Memory Status:")
        print(f"  Total: {total_memory:.2f} GB")
        print(f"  Allocated: {allocated_memory:.2f} GB")
        print(f"  Cached: {cached_memory:.2f} GB")
        print(f"  Free: {total_memory - cached_memory:.2f} GB")
    else:
        print("\n⚠ No GPU available - models will run on CPU (very slow)")


def test_llama_baseline():
    """Test Qwen2.5-3B baseline."""
    print("\n" + "=" * 60)
    print("Test 1: Qwen2.5-3B (Baseline)")
    print("=" * 60)

    print("\nLoading model...")
    generator = create_llama_generator()

    check_gpu_memory()

    # Test query
    query = "What is the Fourth Amendment exclusionary rule?"
    print(f"\nQuery: {query}")

    print("\nGenerating response...")
    result = generator.generate(query)

    print(f"\nAnswer:\n{result.generated_answer}")
    print(f"\nGeneration time: {result.generation_time_ms:.1f}ms")
    print(f"Tokens generated: ~{len(result.generated_answer.split())}")

    check_gpu_memory()

    print("\n[OK] Llama baseline test complete")
    return generator


def test_llawma_baseline():
    """Test Mistral 7B baseline."""
    print("\n" + "=" * 60)
    print("Test 2: Mistral 7B (Instruct)")
    print("=" * 60)

    print("\nLoading model...")
    generator = create_llawma_generator()

    check_gpu_memory()

    # Test query
    query = "What is qualified immunity?"
    print(f"\nQuery: {query}")

    print("\nGenerating response...")
    result = generator.generate(query)

    print(f"\nAnswer:\n{result.generated_answer}")
    print(f"\nGeneration time: {result.generation_time_ms:.1f}ms")
    print(f"Tokens generated: ~{len(result.generated_answer.split())}")

    check_gpu_memory()

    print("\n[OK] Llawma baseline test complete")
    return generator


def test_llama_rag():
    """Test Qwen2.5-3B + RAG."""
    print("\n" + "=" * 60)
    print("Test 3: Qwen2.5-3B + RAG")
    print("=" * 60)

    # Check if index exists
    index_path = Path("data/indexes/legal_corpus")
    if not index_path.exists():
        print(f"\n⚠ SKIPPED: Retrieval index not found at {index_path}")
        print("Run: python scripts/build_retrieval_index.py")
        return None

    print("\nLoading model and retrieval index...")
    generator = create_llama_rag_generator(
        index_path=str(index_path),
        top_k=3,
    )

    check_gpu_memory()

    # Test query
    query = "What is the Fourth Amendment exclusionary rule?"
    print(f"\nQuery: {query}")

    print("\nRetrieving evidence and generating response...")
    result = generator.generate(query)

    print(f"\nAnswer:\n{result.generated_answer}")
    if hasattr(result, 'retrieved_context') and result.retrieved_context:
        print(f"\nRetrieved passages: {len(result.retrieved_context)}")
    print(f"Generation time: {result.generation_time_ms:.1f}ms")

    check_gpu_memory()

    print("\n[OK] Llama + RAG test complete")
    return generator


def test_llawma_rag():
    """Test Mistral 7B + RAG."""
    print("\n" + "=" * 60)
    print("Test 4: Mistral 7B + RAG")
    print("=" * 60)

    # Check if index exists
    index_path = Path("data/indexes/legal_corpus")
    if not index_path.exists():
        print(f"\n⚠ SKIPPED: Retrieval index not found at {index_path}")
        print("Run: python scripts/build_retrieval_index.py")
        return None

    print("\nLoading model and retrieval index...")
    generator = create_llawma_rag_generator(
        index_path=str(index_path),
        top_k=3,
    )

    check_gpu_memory()

    # Test query
    query = "What is qualified immunity?"
    print(f"\nQuery: {query}")

    print("\nRetrieving evidence and generating response...")
    result = generator.generate(query)

    print(f"\nAnswer:\n{result.generated_answer}")
    if hasattr(result, 'retrieved_context') and result.retrieved_context:
        print(f"\nRetrieved passages: {len(result.retrieved_context)}")
    print(f"Generation time: {result.generation_time_ms:.1f}ms")

    check_gpu_memory()

    print("\n[OK] Llawma + RAG test complete")
    return generator


def main():
    """Run all generation tests."""
    print("=" * 60)
    print("Response Generation Test Suite")
    print("=" * 60)
    print("\nTesting all 4 model variants:")
    print("  1. Qwen2.5-3B (baseline)")
    print("  2. Mistral 7B (instruct)")
    print("  3. Qwen2.5-3B + RAG")
    print("  4. Mistral 7B + RAG")

    print("\nHardware requirement: RTX 4080 with 16GB VRAM")
    print("Memory strategy: float16 (no quantization for steering compatibility)")

    check_gpu_memory()

    generators = {}

    # Test 1: Llama baseline
    try:
        generators['llama'] = test_llama_baseline()
    except Exception as e:
        print(f"\n[ERROR] Test 1 failed: {e}")
        import traceback
        traceback.print_exc()

    # Clear GPU memory between tests
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Test 2: Llawma baseline
    try:
        generators['llawma'] = test_llawma_baseline()
    except Exception as e:
        print(f"\n[ERROR] Test 2 failed: {e}")
        import traceback
        traceback.print_exc()

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Test 3: Llama + RAG
    try:
        generators['llama_rag'] = test_llama_rag()
    except Exception as e:
        print(f"\n[ERROR] Test 3 failed: {e}")
        import traceback
        traceback.print_exc()

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Test 4: Llawma + RAG
    try:
        generators['llawma_rag'] = test_llawma_rag()
    except Exception as e:
        print(f"\n[ERROR] Test 4 failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    successful_tests = [name for name, gen in generators.items() if gen is not None]
    print(f"\nSuccessful tests: {len(successful_tests)}/4")
    for name in successful_tests:
        print(f"  [OK] {name}")

    check_gpu_memory()

    print("\n[OK] Generation test suite complete")


if __name__ == "__main__":
    main()

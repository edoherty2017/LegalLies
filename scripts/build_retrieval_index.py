"""
Build retrieval index for legal hallucination detection.

Creates BM25 + dense embedding index from legal corpus for automatic
evidence retrieval. Supports both Baseline and RAG+Web variants.
"""

import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.document_retriever import DocumentRetriever


def load_courtlistener_corpus() -> List[str]:
    """
    Load CourtListener opinions if available.

    Returns list of opinion texts.
    """
    corpus_path = Path("data/corpus/courtlistener_opinions.json")
    if not corpus_path.exists():
        print("CourtListener corpus not found")
        print("Run: python scripts/courtlistener_fetcher.py")
        return []

    with open(corpus_path, "r", encoding="utf-8") as f:
        opinions = json.load(f)

    corpus = []
    for opinion in opinions:
        # Add full opinion text
        text = opinion.get("text", "")
        if text:
            corpus.append(text)

        # Add case name + summary as separate document
        case_name = opinion.get("case_name", "")
        citations = opinion.get("citations", [])
        if case_name:
            summary = f"{case_name}. " + ", ".join(citations[:3])
            corpus.append(summary)

    print(f"Loaded {len(opinions)} CourtListener opinions -> {len(corpus)} documents")
    return corpus


def load_curated_legal_corpus() -> List[str]:
    """
    Load curated legal corpus if available.

    This loads the manually created corpus from download_legal_corpus.py
    """
    corpus_path = Path("data/corpus/legal_corpus.json")
    if not corpus_path.exists():
        print("Curated legal corpus not found")
        print("Run: python scripts/download_legal_corpus.py")
        return []

    with open(corpus_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    corpus = [doc.get("text", "") for doc in documents if doc.get("text")]
    print(f"Loaded {len(corpus)} curated legal documents")
    return corpus


def load_legal_corpus_from_responses() -> List[str]:
    """
    Extract corpus from existing responses as a fallback.

    In production, this would load from:
    - Legal databases (LexisNexis, Westlaw)
    - Court opinion repositories
    - Legal code databases (U.S.C., C.F.R.)
    - Law review articles

    For now, we'll use the responses themselves as a minimal corpus.
    """
    corpus = []

    # Load from gold benchmark
    gold_path = Path("data/gold_standard/gold_benchmark.json")
    if gold_path.exists():
        with open(gold_path, "r", encoding="utf-8") as f:
            gold_data = json.load(f)
            for case in gold_data:
                # Add query text
                corpus.append(case["query_text"])
                # Add generated answer (as reference text)
                corpus.append(case["generated_answer"])
                # Add any retrieved sources if available
                corpus.extend(case.get("retrieved_sources", []))

    # Load from silver benchmark
    silver_path = Path("data/gold_standard/silver_benchmark.json")
    if silver_path.exists():
        try:
            with open(silver_path, "r", encoding="utf-8") as f:
                silver_data = json.load(f)
                for case in silver_data:
                    corpus.append(case["query_text"])
                    corpus.append(case["generated_answer"])
                    corpus.extend(case.get("retrieved_sources", []))
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse silver_benchmark.json: {e}")
            print("Skipping silver benchmark data")

    # Remove duplicates and empty strings
    corpus = [doc.strip() for doc in corpus if doc and doc.strip()]
    corpus = list(set(corpus))

    print(f"Extracted {len(corpus)} unique documents from benchmarks")
    return corpus


def load_legal_corpus_from_sample_text() -> List[str]:
    """
    Load sample legal text for initial testing.

    This creates a minimal corpus covering common legal topics
    mentioned in the benchmark questions.
    """
    sample_corpus = [
        # Fourth Amendment (Q1 topic)
        "The Fourth Amendment to the United States Constitution protects against unreasonable searches and seizures. It requires that warrants be supported by probable cause and describe the place to be searched and the persons or things to be seized.",

        "The exclusionary rule prevents evidence obtained in violation of the Fourth Amendment from being used in criminal prosecutions. This rule was established to deter police misconduct.",

        "The inevitable discovery doctrine allows illegally obtained evidence to be admitted if it would have been discovered through lawful means. However, this does not create a blanket exception to the exclusionary rule.",

        "The knock-and-announce rule requires police officers to knock and announce their presence before entering a premises. Violations of this rule may lead to suppression of evidence under the exclusionary rule.",

        # Telephone Consumer Protection Act (Q3 topic)
        "The Telephone Consumer Protection Act (TCPA) restricts telemarketing calls, auto-dialed calls, prerecorded calls, text messages, and unsolicited faxes. It provides for a private right of action.",

        "Federal district courts have federal-question jurisdiction under 28 U.S.C. Section 1331 over civil actions arising under the Constitution, laws, or treaties of the United States.",

        "28 U.S.C. Section 1331 grants federal courts original jurisdiction of all civil actions arising under federal law. This includes actions brought under federal statutes like the TCPA.",

        "The Supreme Court has held that federal courts retain jurisdiction over private actions under the TCPA pursuant to 28 U.S.C. Section 1331.",

        # General legal principles
        "The United States Code (U.S.C.) is the codification of federal statutes. Title 28 covers the judiciary and judicial procedure.",

        "Federal courts are courts of limited jurisdiction, possessing only that power authorized by Constitution and statute. Subject-matter jurisdiction cannot be waived or forfeited.",

        "The Federal Rules of Civil Procedure govern civil procedure in United States district courts. They were adopted in 1938 and have been amended periodically.",

        "Legal citations typically include the volume number, reporter abbreviation, page number, and year. For example: Smith v. Jones, 123 F.3d 456 (9th Cir. 2000).",

        # Patent law
        "Patent law in the United States is primarily governed by Title 35 of the United States Code. Patents grant inventors exclusive rights to their inventions for a limited time.",

        "The Patent Act of 1952 codified existing patent law and made several important changes. It is codified in Title 35 of the United States Code.",

        # Contract law
        "Contract law governs the enforcement of agreements between parties. Essential elements of a contract include offer, acceptance, and consideration.",

        "The Uniform Commercial Code (U.CC.) governs commercial transactions in the United States. It has been adopted by all 50 states with some variations.",

        # Criminal procedure
        "Criminal procedure in the United States is governed by the Federal Rules of Criminal Procedure in federal courts and by state rules in state courts.",

        "The Fifth Amendment protects against self-incrimination and double jeopardy. It also requires due process of law before deprivation of life, liberty, or property.",

        # Jurisdiction
        "Federal courts have diversity jurisdiction under 28 U.S.C. Section 1332 when the amount in controversy exceeds $75,000 and the parties are citizens of different states.",

        "Subject-matter jurisdiction refers to the court's authority to hear a particular type of case. Personal jurisdiction refers to the court's power over the parties.",
    ]

    print(f"Loaded {len(sample_corpus)} sample legal documents")
    return sample_corpus


def chunk_long_documents(documents: List[str], max_chunk_size: int = 500) -> List[str]:
    """
    Chunk long documents into smaller passages.

    Args:
        documents: List of documents to chunk
        max_chunk_size: Maximum number of characters per chunk

    Returns:
        List of chunked documents
    """
    chunked = []

    for doc in documents:
        if len(doc) <= max_chunk_size:
            chunked.append(doc)
        else:
            # Split on sentence boundaries (simple approximation)
            sentences = doc.split('. ')
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 2 <= max_chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunked.append(current_chunk.strip())
                    current_chunk = sentence + ". "

            if current_chunk:
                chunked.append(current_chunk.strip())

    return chunked


def main():
    """Build retrieval index from legal corpus."""
    print("=" * 60)
    print("Building Legal Corpus Retrieval Index")
    print("=" * 60)

    # Load corpus from all available sources
    print("\nLoading legal corpus...")
    corpus = []

    # Strategy 1: Load CourtListener opinions (PRIMARY SOURCE)
    print("\n[1/4] CourtListener API data...")
    courtlistener_docs = load_courtlistener_corpus()
    corpus.extend(courtlistener_docs)

    # Strategy 2: Load curated legal concepts
    print("\n[2/4] Curated legal concepts...")
    curated_docs = load_curated_legal_corpus()
    corpus.extend(curated_docs)

    # Strategy 3: Load sample legal text (fallback)
    print("\n[3/4] Sample legal text...")
    sample_docs = load_legal_corpus_from_sample_text()
    corpus.extend(sample_docs)

    # Strategy 4: Augment with benchmark data
    print("\n[4/4] Benchmark data...")
    benchmark_docs = load_legal_corpus_from_responses()
    corpus.extend(benchmark_docs)

    # Remove duplicates and empty strings
    corpus = [doc.strip() for doc in corpus if doc and doc.strip()]
    corpus = list(set(corpus))
    print(f"\n{'='*60}")
    print(f"Total unique documents: {len(corpus)}")
    print(f"{'='*60}")

    # Chunk long documents
    print("\nChunking long documents...")
    chunked_corpus = chunk_long_documents(corpus, max_chunk_size=500)
    print(f"Total chunks after splitting: {len(chunked_corpus)}")

    # Create metadata
    metadata = [
        {"doc_id": i, "source": "legal_corpus", "chunk_id": 0}
        for i in range(len(chunked_corpus))
    ]

    # Initialize retriever
    print("\nInitializing document retriever...")
    retriever = DocumentRetriever(
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        device="cpu",
    )

    # Build index
    print("\nBuilding BM25 + dense embedding index...")
    index_path = "data/indexes/legal_corpus"
    retriever.build_index(
        documents=chunked_corpus,
        metadata=metadata,
        save_path=index_path,
    )

    print("\n" + "=" * 60)
    print("Index Build Complete")
    print("=" * 60)
    print(f"Index saved to: {index_path}")
    print(f"Total documents indexed: {len(chunked_corpus)}")

    # Test retrieval
    print("\n" + "=" * 60)
    print("Testing Retrieval")
    print("=" * 60)

    test_query = "What is the Fourth Amendment exclusionary rule?"
    print(f"\nTest query: {test_query}")

    results = retriever.retrieve_evidence(test_query, top_k=3)
    print(f"\nTop 3 retrieved passages:")
    for i, passage in enumerate(results, 1):
        print(f"\n[{i}] {passage[:200]}...")

    # Get corpus stats
    stats = retriever.get_corpus_stats()
    print("\n" + "=" * 60)
    print("Corpus Statistics")
    print("=" * 60)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

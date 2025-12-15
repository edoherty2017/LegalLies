"""
Script to generate Gold and Silver benchmark JSON files from lawyer annotations.

This script:
1. Loads annotations from lawyer_data_package/
2. Derives hallucination labels
3. Aggregates multi-annotator judgments with confidence weighting
4. Splits into Gold/Silver tiers based on evaluability
5. Exports to data/gold_standard/ for detector evaluation

Run: python scripts/generate_benchmarks.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.annotation_loader import AnnotationLoader
from src.evaluation.gold_silver_splitter import GoldSilverSplitter


def main() -> None:
    """Generate Gold and Silver benchmarks from lawyer annotations."""
    print("="  * 80)
    print("GOLD/SILVER BENCHMARK GENERATOR")
    print("=" * 80)
    print()

    # Paths
    data_dir = Path("lawyer_data_package/raw_annotations")
    output_dir = Path("data/gold_standard")

    print(f"Loading annotations from: {data_dir}")
    print()

    # Initialize loader
    loader = AnnotationLoader(data_dir=data_dir)

    # Load all rounds and aggregate
    print("Loading and aggregating annotations...")
    try:
        aggregated_df = loader.load_all_rounds(
            derive_hallucination=True, aggregate=True
        )
        print(f"[PASS] Loaded {len(aggregated_df)} aggregated annotations")
        print()
    except FileNotFoundError as e:
        print(f"[FAIL] Error: {e}")
        print("\nPlease ensure annotation files exist in lawyer_data_package/raw_annotations/")
        return

    # Add evaluability tiers
    print("Computing evaluability tiers...")
    aggregated_df = loader.add_evaluability_tiers(aggregated_df)
    print("[PASS] Evaluability tiers computed")
    print()

    # Show distribution
    tier_dist = aggregated_df["Evaluability_Tier"].value_counts()
    print("Evaluability Tier Distribution:")
    for tier, count in tier_dist.items():
        print(f"  {tier}: {count} annotations")
    print()

    # Create Gold/Silver splits
    print("Creating Gold/Silver splits...")
    splitter = GoldSilverSplitter()
    splits = splitter.create_all_splits(aggregated_df)

    print(f"[PASS] Gold tier: {len(splits['gold'])} annotations")
    print(f"[PASS] Silver tier: {len(splits['silver'])} annotations")
    print(f"[PASS] Excluded tier: {len(splits['excluded'])} annotations")
    print()

    # Generate split report
    print(splitter.generate_split_report(splits))
    print()

    # Export to JSON
    print(f"Exporting benchmarks to: {output_dir}")
    splitter.export_splits_to_json(splits, output_dir, include_excluded=True)
    print()

    print("=" * 80)
    print("BENCHMARK GENERATION COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review benchmarks in data/gold_standard/")
    print("2. Use Gold benchmark for threshold tuning")
    print("3. Use Silver benchmark for validation")
    print("4. Exclude low-confidence items from headline metrics")


if __name__ == "__main__":
    main()

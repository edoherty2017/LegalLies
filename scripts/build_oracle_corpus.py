#!/usr/bin/env python3
"""
Build Oracle Corpus from Harvard COLD Cases dataset.

Uses harvard-lil/cold-cases which is a cleaned, parquet-formatted
version of CourtListener data with 8.3M US legal decisions.

Filters for:
- SCOTUS + all 13 federal circuit courts
- 1990-2023 date range
- Published opinions only
- Minimum text length

Output: data/oracle_corpus/legal_oracle.jsonl
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm


# Court mappings - map court names to our standard IDs
SCOTUS_NAMES = [
    "Supreme Court of the United States",
    "SCOTUS",
]

CIRCUIT_COURT_MAPPINGS = {
    "ca1": ["First Circuit", "1st Circuit", "United States Court of Appeals for the First Circuit"],
    "ca2": ["Second Circuit", "2nd Circuit", "2d Circuit", "United States Court of Appeals for the Second Circuit"],
    "ca3": ["Third Circuit", "3rd Circuit", "3d Circuit", "United States Court of Appeals for the Third Circuit"],
    "ca4": ["Fourth Circuit", "4th Circuit", "United States Court of Appeals for the Fourth Circuit"],
    "ca5": ["Fifth Circuit", "5th Circuit", "United States Court of Appeals for the Fifth Circuit"],
    "ca6": ["Sixth Circuit", "6th Circuit", "United States Court of Appeals for the Sixth Circuit"],
    "ca7": ["Seventh Circuit", "7th Circuit", "United States Court of Appeals for the Seventh Circuit"],
    "ca8": ["Eighth Circuit", "8th Circuit", "United States Court of Appeals for the Eighth Circuit"],
    "ca9": ["Ninth Circuit", "9th Circuit", "United States Court of Appeals for the Ninth Circuit"],
    "ca10": ["Tenth Circuit", "10th Circuit", "United States Court of Appeals for the Tenth Circuit"],
    "ca11": ["Eleventh Circuit", "11th Circuit", "United States Court of Appeals for the Eleventh Circuit"],
    "cadc": ["D.C. Circuit", "DC Circuit", "District of Columbia Circuit", "United States Court of Appeals for the District of Columbia Circuit"],
    "cafed": ["Federal Circuit", "United States Court of Appeals for the Federal Circuit"],
}


def identify_court(court_name: str, court_type: str = None) -> Optional[str]:
    """Identify if court is SCOTUS or a federal circuit court."""
    if not court_name:
        return None

    court_upper = court_name.upper()

    # Check SCOTUS
    for name in SCOTUS_NAMES:
        if name.upper() in court_upper:
            return "scotus"

    # Check circuit courts
    for court_id, names in CIRCUIT_COURT_MAPPINGS.items():
        for name in names:
            if name.upper() in court_upper:
                return court_id

    # Also check by court_type if available
    if court_type:
        if court_type.lower() in ["federal appellate", "f"]:
            # Try to extract circuit from name
            for court_id, names in CIRCUIT_COURT_MAPPINGS.items():
                for name in names:
                    if name.upper() in court_upper:
                        return court_id

    return None


def extract_opinion_text(opinions: list) -> Optional[str]:
    """Extract the main opinion text from opinions list."""
    if not opinions:
        return None

    # Look for majority/lead opinion first
    for opinion in opinions:
        if isinstance(opinion, dict):
            opinion_type = opinion.get("type", "").lower()
            if opinion_type in ["majority", "lead", "020lead"]:
                text = opinion.get("opinion_text") or opinion.get("plain_text", "")
                if text and len(text) > 500:
                    return text

    # Fall back to first opinion with sufficient text
    for opinion in opinions:
        if isinstance(opinion, dict):
            text = opinion.get("opinion_text") or opinion.get("plain_text", "")
            if text and len(text) > 500:
                return text

    return None


def should_exclude(item: Dict[str, Any], text: str) -> bool:
    """Check if case should be excluded."""
    # Check precedential status
    status = item.get("precedential_status", "").lower()
    if status in ["unpublished", "errata", "separate", "in-chambers"]:
        return True

    # Check for per curiam without substance
    if "per curiam" in text[:1000].lower():
        # Allow if it has substantial content
        if len(text) < 2000:
            return True

    # Check for summary orders
    if "summary order" in text[:500].lower():
        return True

    return False


def generate_id(case_id: int, text: str) -> str:
    """Generate unique ID for document."""
    content = f"{case_id}:{text[:200]}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def process_case(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single case and return normalized record."""
    # Get court info
    court_name = item.get("court_full_name") or item.get("court_short_name", "")
    court_type = item.get("court_type", "")

    court_id = identify_court(court_name, court_type)
    if not court_id:
        return None

    # Get date
    date_filed = item.get("date_filed")
    if date_filed:
        if isinstance(date_filed, str):
            try:
                year = int(date_filed[:4])
            except:
                return None
        else:
            # datetime object
            try:
                year = date_filed.year
                date_filed = date_filed.strftime("%Y-%m-%d")
            except:
                return None
    else:
        return None

    # Filter by year
    if year < 1990 or year > 2023:
        return None

    # Get opinion text
    opinions = item.get("opinions", [])
    text = extract_opinion_text(opinions)
    if not text or len(text) < 500:
        return None

    # Check exclusions
    if should_exclude(item, text):
        return None

    # Build normalized record
    case_id = item.get("id", 0)
    case_name = item.get("case_name") or item.get("case_name_short") or "Unknown Case"

    record = {
        "id": generate_id(case_id, text),
        "title": str(case_name)[:300],
        "court": court_id,
        "date": str(date_filed),
        "year": year,
        "jurisdiction": "federal",
        "text": text,
        "source_url": f"https://www.courtlistener.com/opinion/{case_id}/",
        "precedential_status": item.get("precedential_status", ""),
        "citation_count": item.get("citation_count", 0),
    }

    return record


def main():
    """Build the oracle corpus."""
    output_path = Path("data/oracle_corpus/legal_oracle.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Harvard COLD Cases dataset...")
    print("This may take a while on first run...")

    # Load dataset in streaming mode
    try:
        dataset = load_dataset(
            "harvard-lil/cold-cases",
            split="train",
            streaming=True,
        )
        print("Dataset loaded in streaming mode.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

    # Track statistics
    stats = defaultdict(int)
    court_counts = defaultdict(int)
    year_counts = defaultdict(int)

    # Target counts per court (slightly higher to allow for filtering)
    target_counts = {
        "scotus": 3000,
        "ca1": 850,
        "ca2": 850,
        "ca3": 850,
        "ca4": 850,
        "ca5": 850,
        "ca6": 850,
        "ca7": 850,
        "ca8": 850,
        "ca9": 1100,  # Largest circuit
        "ca10": 850,
        "ca11": 850,
        "cadc": 850,
        "cafed": 850,
    }

    total_target = sum(target_counts.values())

    # Process cases
    processed = []
    seen_ids = set()

    print(f"Processing cases (target: {total_target:,})...")

    with tqdm(desc="Processing", total=total_target) as pbar:
        for item in dataset:
            stats["total"] += 1

            # Process the case
            record = process_case(item)

            if not record:
                stats["filtered"] += 1
                continue

            # Check for duplicates
            if record["id"] in seen_ids:
                stats["duplicates"] += 1
                continue

            # Check if we've hit target for this court
            court = record["court"]
            target = target_counts.get(court, 850)
            if court_counts[court] >= target:
                stats["over_quota"] += 1
                continue

            # Add record
            seen_ids.add(record["id"])
            processed.append(record)
            court_counts[court] += 1
            year_counts[record["year"]] += 1
            stats["kept"] += 1

            pbar.update(1)
            pbar.set_postfix({"kept": stats["kept"], "courts": len(court_counts)})

            # Check if we've hit overall target
            if stats["kept"] >= total_target:
                print(f"\nReached target of {total_target:,} cases")
                break

            # Progress update every 50k processed
            if stats["total"] % 50000 == 0:
                print(f"\n  Scanned: {stats['total']:,}, Kept: {stats['kept']:,}, Courts: {len(court_counts)}")

    # Write output
    print(f"\nWriting {len(processed):,} cases to {output_path}...")

    with open(output_path, "w") as f:
        for record in processed:
            f.write(json.dumps(record) + "\n")

    # Print summary
    print("\n" + "="*60)
    print("ORACLE CORPUS BUILD COMPLETE")
    print("="*60)
    print(f"\nTotal scanned: {stats['total']:,}")
    print(f"Filtered out: {stats['filtered']:,}")
    print(f"Duplicates: {stats['duplicates']:,}")
    print(f"Over quota: {stats['over_quota']:,}")
    print(f"Final corpus size: {stats['kept']:,}")

    print(f"\n{'Court':<10} {'Count':>8} {'Target':>8} {'%':>6}")
    print("-" * 36)
    for court in sorted(court_counts.keys()):
        target = target_counts.get(court, 850)
        count = court_counts[court]
        pct = count / target * 100 if target > 0 else 0
        print(f"{court:<10} {count:>8,} {target:>8,} {pct:>5.1f}%")

    print(f"\nYear distribution:")
    decades = defaultdict(int)
    for year, count in year_counts.items():
        decade = (year // 10) * 10
        decades[decade] += count
    for decade in sorted(decades.keys()):
        print(f"  {decade}s: {decades[decade]:,}")

    print(f"\nOutput: {output_path}")
    file_size = output_path.stat().st_size
    print(f"Size: {file_size / 1024 / 1024:.1f} MB")

    # Return stats for CLAUDE.md update
    return {
        "total_cases": stats["kept"],
        "court_counts": dict(court_counts),
        "year_counts": dict(year_counts),
    }


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build Oracle Corpus from Harvard COLD Cases dataset - V2 (Optimized).

Uses court_type='F' (Federal Appellate) filter for efficiency.
Downloads parquet files and filters locally for much faster processing.

Output: data/oracle_corpus/legal_oracle.jsonl
"""

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm


# Court mappings - map court_id field to our standard IDs
COURT_ID_MAPPINGS = {
    # SCOTUS
    "scotus": "scotus",
    # Circuit Courts (CourtListener uses these IDs)
    "ca1": "ca1",
    "ca2": "ca2",
    "ca3": "ca3",
    "ca4": "ca4",
    "ca5": "ca5",
    "ca6": "ca6",
    "ca7": "ca7",
    "ca8": "ca8",
    "ca9": "ca9",
    "ca10": "ca10",
    "ca11": "ca11",
    "cadc": "cadc",
    "cafc": "cafed",  # Federal Circuit
}

# Also check court names for matching
SCOTUS_PATTERNS = ["supreme court of the united states", "scotus"]
CIRCUIT_PATTERNS = {
    "ca1": ["first circuit", "1st circuit"],
    "ca2": ["second circuit", "2nd circuit", "2d circuit"],
    "ca3": ["third circuit", "3rd circuit", "3d circuit"],
    "ca4": ["fourth circuit", "4th circuit"],
    "ca5": ["fifth circuit", "5th circuit"],
    "ca6": ["sixth circuit", "6th circuit"],
    "ca7": ["seventh circuit", "7th circuit"],
    "ca8": ["eighth circuit", "8th circuit"],
    "ca9": ["ninth circuit", "9th circuit"],
    "ca10": ["tenth circuit", "10th circuit"],
    "ca11": ["eleventh circuit", "11th circuit"],
    "cadc": ["d.c. circuit", "dc circuit", "district of columbia circuit"],
    "cafed": ["federal circuit"],
}


def identify_court(item: Dict[str, Any]) -> Optional[str]:
    """Identify if court is SCOTUS or a federal circuit court."""
    # First try court_id (most reliable)
    court_id = (item.get("court_id") or "").lower().strip()
    if court_id in COURT_ID_MAPPINGS:
        return COURT_ID_MAPPINGS[court_id]

    # Fall back to court name matching
    court_name = (item.get("court_full_name") or item.get("court_short_name") or "").lower()

    # Check SCOTUS
    for pattern in SCOTUS_PATTERNS:
        if pattern in court_name:
            return "scotus"

    # Check circuits
    for court_id, patterns in CIRCUIT_PATTERNS.items():
        for pattern in patterns:
            if pattern in court_name:
                return court_id

    return None


def extract_opinion_text(opinions: list) -> Optional[str]:
    """Extract the main opinion text from opinions list."""
    if not opinions:
        return None

    # Look for majority/lead opinion first
    for opinion in opinions:
        if isinstance(opinion, dict):
            opinion_type = (opinion.get("type") or "").lower()
            if opinion_type in ["majority", "lead", "020lead"]:
                text = opinion.get("plain_text") or opinion.get("html", "")
                if text and len(text) > 500:
                    return text

    # Fall back to first opinion with sufficient text
    for opinion in opinions:
        if isinstance(opinion, dict):
            text = opinion.get("plain_text") or opinion.get("html", "")
            if text and len(text) > 500:
                return text

    return None


def should_exclude(item: Dict[str, Any], text: str) -> tuple[bool, str]:
    """Check if case should be excluded. Returns (exclude, reason)."""
    # Check precedential status
    status = (item.get("precedential_status") or "").lower()
    if status in ["unpublished", "errata", "separate", "in-chambers"]:
        return True, "unpublished"

    # Check for per curiam without substance
    text_lower = text[:1000].lower()
    if "per curiam" in text_lower and len(text) < 2000:
        return True, "per_curiam_short"

    # Check for summary orders
    if "summary order" in text[:500].lower():
        return True, "summary_order"

    # Check minimum length
    if len(text) < 500:
        return True, "too_short"

    return False, ""


def generate_id(case_id: int, text: str) -> str:
    """Generate unique ID for document."""
    content = f"{case_id}:{text[:200]}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def process_case(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single case and return normalized record."""
    # Get court info
    court_id = identify_court(item)
    if not court_id:
        return None

    # Get date
    date_filed = item.get("date_filed")
    if not date_filed:
        return None

    if isinstance(date_filed, str):
        try:
            year = int(date_filed[:4])
        except (ValueError, IndexError):
            return None
    else:
        try:
            year = date_filed.year
            date_filed = date_filed.strftime("%Y-%m-%d")
        except:
            return None

    # Filter by year
    if year < 1990 or year > 2023:
        return None

    # Get opinion text
    opinions = item.get("opinions") or []
    text = extract_opinion_text(opinions)
    if not text:
        return None

    # Check exclusions
    exclude, reason = should_exclude(item, text)
    if exclude:
        return None

    # Build normalized record
    case_id = item.get("id", 0)
    case_name = item.get("case_name") or item.get("case_name_short") or "Unknown"

    return {
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


def main():
    """Build the oracle corpus."""
    output_path = Path("data/oracle_corpus/legal_oracle.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Target counts per court
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
        "ca9": 1100,
        "ca10": 850,
        "ca11": 850,
        "cadc": 850,
        "cafed": 850,
    }
    total_target = sum(target_counts.values())

    print("="*60)
    print("ORACLE CORPUS BUILD - V2 (Optimized)")
    print("="*60)
    print(f"\nTarget: {total_target:,} cases")
    print("Filter: Federal Appellate (court_type='F') + SCOTUS")
    print("\nLoading Harvard COLD Cases dataset...")

    # Load with streaming, filtering for Federal courts only
    # court_type: F = Federal Appellate, FS = Federal Special (includes SCOTUS)
    dataset = load_dataset(
        "harvard-lil/cold-cases",
        split="train",
        streaming=True,
    )

    print("Dataset loaded. Filtering for federal appellate courts...\n")

    # Track statistics
    stats = defaultdict(int)
    court_counts = defaultdict(int)
    year_counts = defaultdict(int)
    seen_ids = set()

    # Open file for incremental writing
    with open(output_path, "w") as f:
        with tqdm(total=total_target, desc="Collecting cases") as pbar:
            for item in dataset:
                stats["scanned"] += 1

                # Quick filter: only federal courts
                court_type = item.get("court_type", "")
                if court_type not in ["F", "FS"]:  # F=Federal Appellate, FS=Federal Special
                    stats["non_federal"] += 1
                    continue

                # Process the case
                record = process_case(item)

                if not record:
                    stats["filtered"] += 1
                    continue

                # Check for duplicates
                if record["id"] in seen_ids:
                    stats["duplicates"] += 1
                    continue

                # Check quota
                court = record["court"]
                target = target_counts.get(court, 0)
                if target == 0:
                    stats["unknown_court"] += 1
                    continue

                if court_counts[court] >= target:
                    stats["over_quota"] += 1
                    # Check if all courts are full
                    if all(court_counts.get(c, 0) >= t for c, t in target_counts.items()):
                        print(f"\nAll court quotas filled!")
                        break
                    continue

                # Write record immediately
                f.write(json.dumps(record) + "\n")
                f.flush()

                seen_ids.add(record["id"])
                court_counts[court] += 1
                year_counts[record["year"]] += 1
                stats["kept"] += 1

                pbar.update(1)
                pbar.set_postfix({
                    "scanned": f"{stats['scanned']:,}",
                    "courts": len(court_counts)
                })

                # Progress update
                if stats["kept"] % 1000 == 0:
                    print(f"\n  Progress: {stats['kept']:,}/{total_target:,} kept, {stats['scanned']:,} scanned")
                    for c in sorted(court_counts.keys()):
                        t = target_counts.get(c, 0)
                        print(f"    {c}: {court_counts[c]}/{t}")

                # Check if done
                if stats["kept"] >= total_target:
                    print(f"\nReached target of {total_target:,} cases!")
                    break

    # Print summary
    print("\n" + "="*60)
    print("ORACLE CORPUS BUILD COMPLETE")
    print("="*60)
    print(f"\nTotal scanned: {stats['scanned']:,}")
    print(f"Non-federal skipped: {stats['non_federal']:,}")
    print(f"Filtered out: {stats['filtered']:,}")
    print(f"Duplicates: {stats['duplicates']:,}")
    print(f"Over quota: {stats['over_quota']:,}")
    print(f"Unknown court: {stats['unknown_court']:,}")
    print(f"\nFinal corpus size: {stats['kept']:,}")

    print(f"\n{'Court':<10} {'Count':>8} {'Target':>8} {'%':>6}")
    print("-" * 36)
    for court in sorted(target_counts.keys()):
        target = target_counts[court]
        count = court_counts.get(court, 0)
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


if __name__ == "__main__":
    main()

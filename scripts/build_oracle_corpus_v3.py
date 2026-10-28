#!/usr/bin/env python3
"""
Build Oracle Corpus from Harvard COLD Cases dataset - V3 (Parquet Native).

Downloads parquet files and uses pyarrow native filtering for speed.

Output: data/oracle_corpus/legal_oracle.jsonl
"""

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from collections import defaultdict

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm


# Court name patterns to identify courts
SCOTUS_PATTERNS = ["supreme court of the united states", "scotus"]

CIRCUIT_PATTERNS = {
    "ca1": ["first circuit", "1st cir"],
    "ca2": ["second circuit", "2nd cir", "2d cir"],
    "ca3": ["third circuit", "3rd cir", "3d cir"],
    "ca4": ["fourth circuit", "4th cir"],
    "ca5": ["fifth circuit", "5th cir"],
    "ca6": ["sixth circuit", "6th cir"],
    "ca7": ["seventh circuit", "7th cir"],
    "ca8": ["eighth circuit", "8th cir"],
    "ca9": ["ninth circuit", "9th cir"],
    "ca10": ["tenth circuit", "10th cir"],
    "ca11": ["eleventh circuit", "11th cir"],
    "cadc": ["d.c. circuit", "dc circuit", "district of columbia circuit"],
    "cafed": ["federal circuit", "fed. cir"],
}


def identify_court(court_short: str, court_full: str) -> Optional[str]:
    """Identify court from name strings."""
    combined = f"{court_short or ''} {court_full or ''}".lower()

    # Check SCOTUS
    for pattern in SCOTUS_PATTERNS:
        if pattern in combined:
            return "scotus"

    # Check circuits
    for court_id, patterns in CIRCUIT_PATTERNS.items():
        for pattern in patterns:
            if pattern in combined:
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


def should_exclude(precedential_status: str, text: str) -> bool:
    """Check if case should be excluded."""
    status = (precedential_status or "").lower()
    if status in ["unpublished", "errata", "separate", "in-chambers"]:
        return True

    text_lower = text[:1000].lower()
    if "per curiam" in text_lower and len(text) < 2000:
        return True
    if "summary order" in text[:500].lower():
        return True

    return False


def generate_id(case_id: int, text: str) -> str:
    content = f"{case_id}:{text[:200]}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def main():
    output_path = Path("data/oracle_corpus/legal_oracle.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Target counts
    target_counts = {
        "scotus": 3000,
        "ca1": 850, "ca2": 850, "ca3": 850, "ca4": 850,
        "ca5": 850, "ca6": 850, "ca7": 850, "ca8": 850,
        "ca9": 1100, "ca10": 850, "ca11": 850,
        "cadc": 850, "cafed": 850,
    }
    total_target = sum(target_counts.values())

    print("="*60)
    print("ORACLE CORPUS BUILD - V3 (Parquet Native)")
    print("="*60)
    print(f"\nTarget: {total_target:,} cases")
    print("\nStep 1: Listing parquet files...")

    # Get list of parquet files
    repo_id = "harvard-lil/cold-cases"
    files = list_repo_files(repo_id, repo_type="dataset")
    parquet_files = sorted([f for f in files if f.endswith(".parquet")])
    print(f"Found {len(parquet_files)} parquet files")

    # Stats
    stats = defaultdict(int)
    court_counts = defaultdict(int)
    year_counts = defaultdict(int)
    seen_ids = set()

    print(f"\nStep 2: Processing parquet files...")

    with open(output_path, "w") as out_f:
        for pq_idx, pq_file in enumerate(tqdm(parquet_files, desc="Parquet files")):
            # Check if we're done
            if all(court_counts.get(c, 0) >= t for c, t in target_counts.items()):
                print("\nAll quotas filled!")
                break

            # Download parquet file
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=pq_file,
                    repo_type="dataset",
                )
            except Exception as e:
                print(f"\nError downloading {pq_file}: {e}")
                continue

            # Read parquet file
            try:
                pq_table = pq.read_table(local_path)
            except Exception as e:
                print(f"\nError reading {pq_file}: {e}")
                continue

            num_rows = pq_table.num_rows
            print(f"\n  File {pq_idx+1}/{len(parquet_files)}: {num_rows:,} rows")

            # Process each row
            for i in range(num_rows):
                stats["scanned"] += 1

                # Quick court_type filter - F = Federal Appellate, FS = Federal Special
                court_type = pq_table["court_type"][i].as_py()
                if court_type not in ["F", "FS"]:
                    continue

                # Identify specific court
                court_short = pq_table["court_short_name"][i].as_py()
                court_full = pq_table["court_full_name"][i].as_py()
                court_id = identify_court(court_short, court_full)

                if not court_id:
                    stats["unknown_court"] += 1
                    continue

                # Check quota
                if court_counts[court_id] >= target_counts.get(court_id, 0):
                    stats["over_quota"] += 1
                    continue

                # Get date
                date_filed = pq_table["date_filed"][i].as_py()
                if not date_filed:
                    stats["no_date"] += 1
                    continue

                if hasattr(date_filed, 'year'):
                    year = date_filed.year
                    date_str = date_filed.strftime("%Y-%m-%d") if hasattr(date_filed, 'strftime') else str(date_filed)
                else:
                    try:
                        date_str = str(date_filed)
                        year = int(date_str[:4])
                    except:
                        continue

                if year < 1990 or year > 2023:
                    stats["out_of_range"] += 1
                    continue

                # Get opinions
                opinions = pq_table["opinions"][i].as_py()
                text = extract_opinion_text(opinions)
                if not text or len(text) < 500:
                    stats["no_text"] += 1
                    continue

                # Check exclusions
                prec_status = pq_table["precedential_status"][i].as_py()
                if should_exclude(prec_status, text):
                    stats["excluded"] += 1
                    continue

                # Generate ID and check dupe
                case_id = pq_table["id"][i].as_py() or 0
                doc_id = generate_id(case_id, text)
                if doc_id in seen_ids:
                    stats["duplicates"] += 1
                    continue

                # Build record
                case_name = pq_table["case_name"][i].as_py() or pq_table["case_name_short"][i].as_py() or "Unknown"
                citation_count = pq_table["citation_count"][i].as_py() or 0

                record = {
                    "id": doc_id,
                    "title": str(case_name)[:300],
                    "court": court_id,
                    "date": date_str,
                    "year": year,
                    "jurisdiction": "federal",
                    "text": text,
                    "source_url": f"https://www.courtlistener.com/opinion/{case_id}/",
                    "precedential_status": prec_status or "",
                    "citation_count": citation_count,
                }

                # Write immediately
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

                seen_ids.add(doc_id)
                court_counts[court_id] += 1
                year_counts[year] += 1
                stats["kept"] += 1

            # Progress after each file
            print(f"  Running totals - Kept: {stats['kept']:,}, Courts: {dict(court_counts)}")

    # Summary
    print("\n" + "="*60)
    print("ORACLE CORPUS BUILD COMPLETE")
    print("="*60)
    print(f"\nTotal scanned: {stats['scanned']:,}")
    print(f"Unknown court: {stats['unknown_court']:,}")
    print(f"Over quota: {stats['over_quota']:,}")
    print(f"Out of date range: {stats['out_of_range']:,}")
    print(f"No text: {stats['no_text']:,}")
    print(f"Excluded: {stats['excluded']:,}")
    print(f"Duplicates: {stats['duplicates']:,}")
    print(f"\nFinal corpus size: {stats['kept']:,}")

    print(f"\n{'Court':<10} {'Count':>8} {'Target':>8} {'%':>6}")
    print("-" * 36)
    for court in sorted(target_counts.keys()):
        target = target_counts[court]
        count = court_counts.get(court, 0)
        pct = count / target * 100 if target > 0 else 0
        print(f"{court:<10} {count:>8,} {target:>8,} {pct:>5.1f}%")

    total_kept = sum(court_counts.values())
    print(f"{'TOTAL':<10} {total_kept:>8,} {total_target:>8,} {total_kept/total_target*100:>5.1f}%")

    print(f"\nYear distribution:")
    decades = defaultdict(int)
    for year, count in year_counts.items():
        decade = (year // 10) * 10
        decades[decade] += count
    for decade in sorted(decades.keys()):
        print(f"  {decade}s: {decades[decade]:,}")

    print(f"\nOutput: {output_path}")
    if output_path.exists():
        file_size = output_path.stat().st_size
        print(f"Size: {file_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

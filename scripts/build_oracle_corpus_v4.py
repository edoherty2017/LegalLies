#!/usr/bin/env python3
"""
Build Oracle Corpus - V4 (Resume from file 9, process SCOTUS only).

Since circuit court quotas are filled, we only need more SCOTUS cases.
Downloads one parquet at a time, clears cache after each to avoid disk issues.

Appends to: data/oracle_corpus/legal_oracle.jsonl
"""

import json
import hashlib
import os
import shutil
from pathlib import Path
from typing import Optional
from collections import defaultdict

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

# Only looking for SCOTUS
SCOTUS_PATTERNS = ["supreme court of the united states", "scotus"]


def is_scotus(court_short: str, court_full: str) -> bool:
    """Check if this is a SCOTUS case."""
    combined = f"{court_short or ''} {court_full or ''}".lower()
    for pattern in SCOTUS_PATTERNS:
        if pattern in combined:
            return True
    return False


def extract_opinion_text(opinions: list) -> Optional[str]:
    """Extract the main opinion text."""
    if not opinions:
        return None
    for opinion in opinions:
        if isinstance(opinion, dict):
            opinion_type = (opinion.get("type") or "").lower()
            if opinion_type in ["majority", "lead", "020lead"]:
                text = opinion.get("opinion_text") or opinion.get("plain_text", "")
                if text and len(text) > 500:
                    return text
    for opinion in opinions:
        if isinstance(opinion, dict):
            text = opinion.get("opinion_text") or opinion.get("plain_text", "")
            if text and len(text) > 500:
                return text
    return None


def should_exclude(precedential_status: str, text: str) -> bool:
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

    # Load existing IDs to avoid duplicates
    seen_ids = set()
    scotus_count = 0
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line)
                seen_ids.add(rec["id"])
                if rec["court"] == "scotus":
                    scotus_count += 1

    scotus_target = 3000
    scotus_needed = scotus_target - scotus_count

    print("="*60)
    print("ORACLE CORPUS BUILD - V4 (SCOTUS completion)")
    print("="*60)
    print(f"\nExisting SCOTUS cases: {scotus_count}")
    print(f"SCOTUS target: {scotus_target}")
    print(f"SCOTUS needed: {scotus_needed}")

    if scotus_needed <= 0:
        print("\nSCOTUS quota already filled!")
        return

    print("\nListing parquet files...")
    repo_id = "harvard-lil/cold-cases"
    files = list_repo_files(repo_id, repo_type="dataset")
    parquet_files = sorted([f for f in files if f.endswith(".parquet")])
    print(f"Found {len(parquet_files)} parquet files")

    # Start from file 9 (0-indexed), since files 0-8 were already processed
    start_idx = 9
    print(f"Starting from file {start_idx} (skipping already processed)")

    # HF cache dir
    cache_base = os.path.expanduser("~/.cache/huggingface/hub")

    stats = defaultdict(int)

    with open(output_path, "a") as out_f:  # Append mode
        for pq_idx in range(start_idx, len(parquet_files)):
            if scotus_count >= scotus_target:
                print("\nSCOTUS quota filled!")
                break

            pq_file = parquet_files[pq_idx]
            print(f"\nFile {pq_idx+1}/{len(parquet_files)}: {pq_file}")

            # Download
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=pq_file,
                    repo_type="dataset",
                )
            except Exception as e:
                print(f"  Error downloading: {e}")
                continue

            # Process
            try:
                pq_table = pq.read_table(local_path)
            except Exception as e:
                print(f"  Error reading: {e}")
                continue

            num_rows = pq_table.num_rows
            file_scotus = 0

            for i in range(num_rows):
                stats["scanned"] += 1

                # Only federal special (FS) for SCOTUS
                court_type = pq_table["court_type"][i].as_py()
                if court_type != "FS":
                    continue

                court_short = pq_table["court_short_name"][i].as_py()
                court_full = pq_table["court_full_name"][i].as_py()

                if not is_scotus(court_short, court_full):
                    continue

                # Date filter
                date_filed = pq_table["date_filed"][i].as_py()
                if not date_filed:
                    continue

                if hasattr(date_filed, 'year'):
                    year = date_filed.year
                    date_str = str(date_filed)
                else:
                    try:
                        date_str = str(date_filed)
                        year = int(date_str[:4])
                    except:
                        continue

                if year < 1990 or year > 2023:
                    continue

                # Get text
                opinions = pq_table["opinions"][i].as_py()
                text = extract_opinion_text(opinions)
                if not text or len(text) < 500:
                    continue

                # Exclusions
                prec_status = pq_table["precedential_status"][i].as_py()
                if should_exclude(prec_status, text):
                    continue

                # Dupe check
                case_id = pq_table["id"][i].as_py() or 0
                doc_id = generate_id(case_id, text)
                if doc_id in seen_ids:
                    continue

                # Build and write
                case_name = pq_table["case_name"][i].as_py() or pq_table["case_name_short"][i].as_py() or "Unknown"
                citation_count = pq_table["citation_count"][i].as_py() or 0

                record = {
                    "id": doc_id,
                    "title": str(case_name)[:300],
                    "court": "scotus",
                    "date": date_str,
                    "year": year,
                    "jurisdiction": "federal",
                    "text": text,
                    "source_url": f"https://www.courtlistener.com/opinion/{case_id}/",
                    "precedential_status": prec_status or "",
                    "citation_count": citation_count,
                }

                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

                seen_ids.add(doc_id)
                scotus_count += 1
                file_scotus += 1
                stats["kept"] += 1

                if scotus_count >= scotus_target:
                    break

            print(f"  Rows: {num_rows:,}, SCOTUS found: {file_scotus}, Total SCOTUS: {scotus_count}/{scotus_target}")

            # Clear cache for this file to free disk
            try:
                # Find and remove the cached file
                for root, dirs, files_list in os.walk(cache_base):
                    for f in files_list:
                        if f.endswith(".parquet") and "cold-cases" in root:
                            fpath = os.path.join(root, f)
                            os.remove(fpath)
                            print(f"  Cleared cache: {f}")
            except Exception as e:
                print(f"  Cache cleanup error: {e}")

    # Summary
    print("\n" + "="*60)
    print("SCOTUS COMPLETION DONE")
    print("="*60)
    print(f"\nFinal SCOTUS count: {scotus_count}")
    print(f"Total cases in corpus: {len(seen_ids)}")

    if output_path.exists():
        file_size = output_path.stat().st_size
        print(f"\nOutput: {output_path}")
        print(f"Size: {file_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

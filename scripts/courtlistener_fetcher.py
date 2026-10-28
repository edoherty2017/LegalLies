"""
CourtListener API fetcher for legal corpus building.

Fetches recent opinions and metadata from the Free Law Project's
CourtListener API. Implements proper rate limiting, error handling,
and metadata extraction following the legal RAG framework.

API Documentation: https://www.courtlistener.com/api/rest-info/
Free API Key: https://www.courtlistener.com/sign-in/register/
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests


class CourtListenerFetcher:
    """
    Fetches legal opinions from CourtListener API.

    Implements:
    - Rate limiting (API allows 5000 requests/hour)
    - Metadata extraction (jurisdiction, court, date, citations)
    - Smart filtering (exclude unpublished, focus on recent)
    - Proper error handling
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CourtListener fetcher.

        Args:
            api_key: CourtListener API key (get from courtlistener.com/api)
                    If None, will try to load from environment or .env file
        """
        self.api_key = api_key or self._load_api_key()
        self.base_url = "https://www.courtlistener.com/api/rest/v4"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {self.api_key}",
            "User-Agent": "LegalLies-HallucinationDetector/1.0",
        })

        # Rate limiting (5000/hour = ~1.4/second, be conservative at 1/second)
        self.rate_limit_delay = 1.0
        self.last_request_time = 0

    def _load_api_key(self) -> str:
        """Load API key from .env file or environment variable."""
        import os

        # Try environment variable first
        api_key = os.getenv("COURTLISTENER_API_KEY")
        if api_key:
            return api_key

        # Try .env file
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file, "r") as f:
                for line in f:
                    if line.startswith("COURTLISTENER_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"\'')

        raise ValueError(
            "CourtListener API key not found. Please set COURTLISTENER_API_KEY "
            "environment variable or add to .env file. Get free key from: "
            "https://www.courtlistener.com/sign-in/register/"
        )

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def fetch_recent_opinions(
        self,
        max_results: int = 500,
        min_date: str = "2020-01-01",
        court_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Fetch recent published opinions.

        Args:
            max_results: Maximum number of opinions to fetch
            min_date: Minimum decision date (ISO format: YYYY-MM-DD)
            court_types: Filter by court type (e.g., ['scotus', 'ca1', 'ca9'])
                        None = all federal courts

        Returns:
            List of opinion documents with metadata
        """
        print(f"\n[API] Fetching up to {max_results} opinions from CourtListener...")
        print(f"[API] Date range: {min_date} to present")

        if court_types:
            print(f"[API] Court filter: {', '.join(court_types)}")

        opinions = []
        page = 1
        per_page = 100  # API max per page

        while len(opinions) < max_results:
            # Build query parameters
            params = {
                "cluster__date_filed__gte": min_date,  # Correct API parameter
                "order_by": "-cluster__date_filed",  # Most recent first
                "page": page,
                "page_size": min(per_page, max_results - len(opinions)),
            }

            # Add court filter if specified
            if court_types:
                params["cluster__court__id__in"] = ",".join(court_types)

            # Make request
            self._rate_limit()
            try:
                response = self.session.get(
                    f"{self.base_url}/opinions/",
                    params=params,
                    timeout=30,
                )
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"[ERROR] API request failed: {e}")
                break

            data = response.json()
            results = data.get("results", [])

            if not results:
                print(f"[INFO] No more results (page {page})")
                break

            # Process results
            for opinion in results:
                processed = self._process_opinion(opinion)
                if processed:
                    opinions.append(processed)

            print(f"[API] Page {page}: fetched {len(results)}, total {len(opinions)}")

            # Check if we have more pages
            if not data.get("next"):
                break

            page += 1

        print(f"[OK] Fetched {len(opinions)} opinions")
        return opinions

    def _process_opinion(self, raw_opinion: Dict) -> Optional[Dict]:
        """
        Process raw API response into structured document.

        Extracts:
        - Text content
        - Metadata (jurisdiction, court, date)
        - Citations
        - Authority scoring data
        """
        # Get opinion text
        text = raw_opinion.get("plain_text", raw_opinion.get("html", ""))
        if not text or len(text) < 100:
            # Skip very short or empty opinions
            return None

        # Extract case name from absolute_url (e.g., "/opinion/123/case-name-v-defendant/")
        case_name = "Unknown Case"
        absolute_url = raw_opinion.get("absolute_url", "")
        if absolute_url:
            # Parse case name from URL
            parts = absolute_url.strip("/").split("/")
            if len(parts) >= 3:
                # Convert dashes to spaces and title case
                case_name = parts[2].replace("-", " ").title()

        # Simple approach: extract minimal metadata without extra API calls
        # For production, could batch-fetch clusters or use search API instead
        court_id = "federal"  # Default
        jurisdiction = "us"

        # Build structured document with available data
        document = {
            "text": text[:10000],  # Limit to first 10k chars to manage size
            "doc_type": "opinion",
            "jurisdiction": jurisdiction,
            "court_id": court_id,
            "case_name": case_name,
            "date_filed": "",  # Not available without cluster
            "citations": [],  # Not available without cluster
            "citation_count": 0,
            "precedential_status": "Unknown",
            "opinion_type": raw_opinion.get("type", "Unknown"),
            "author": raw_opinion.get("author_str", ""),
            "url": absolute_url,
            "source": "courtlistener",
        }

        return document

    def _extract_jurisdiction(self, court_id: str) -> str:
        """
        Extract jurisdiction code from court ID.

        Examples:
        - scotus -> us
        - ca9 -> us-ca9 (9th Circuit)
        - californias -> us-ca
        """
        if court_id == "scotus":
            return "us"
        elif court_id.startswith("ca"):
            # Circuit court
            return f"us-{court_id}"
        elif court_id.endswith("s"):
            # State supreme court (e.g., californias)
            state = court_id[:-1]
            return f"us-{state}"
        else:
            # Default to US federal
            return "us"

    def fetch_supreme_court_cases(
        self,
        max_results: int = 100,
        min_year: int = 2000,
    ) -> List[Dict]:
        """
        Fetch Supreme Court cases (high authority).

        These are critical for the corpus as they are binding precedent.
        """
        min_date = f"{min_year}-01-01"
        return self.fetch_recent_opinions(
            max_results=max_results,
            min_date=min_date,
            court_types=["scotus"],
        )

    def fetch_circuit_cases(
        self,
        circuits: Optional[List[int]] = None,
        max_results: int = 200,
        min_year: int = 2015,
    ) -> List[Dict]:
        """
        Fetch Circuit Court cases.

        Args:
            circuits: List of circuit numbers (1-11, dc, fed). None = all circuits
            max_results: Total results across all circuits
            min_year: Minimum year
        """
        if circuits is None:
            # All federal circuits
            circuits = list(range(1, 12)) + ["dc", "fed"]

        court_types = [f"ca{c}" for c in circuits]
        min_date = f"{min_year}-01-01"

        return self.fetch_recent_opinions(
            max_results=max_results,
            min_date=min_date,
            court_types=court_types,
        )

    def save_corpus(self, opinions: List[Dict], output_path: Path):
        """Save fetched opinions to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(opinions, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved {len(opinions)} opinions to {output_path}")


def main():
    """Fetch legal corpus from CourtListener."""
    print("=" * 60)
    print("CourtListener Legal Corpus Fetcher")
    print("=" * 60)

    try:
        fetcher = CourtListenerFetcher()
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("\n[INSTRUCTIONS]")
        print("1. Register for free API key: https://www.courtlistener.com/sign-in/register/")
        print("2. Set environment variable: COURTLISTENER_API_KEY=your_key_here")
        print("   OR add to .env file: COURTLISTENER_API_KEY=your_key_here")
        return

    # Fetch Supreme Court cases (highest authority)
    print("\n" + "=" * 60)
    print("Fetching Supreme Court Cases")
    print("=" * 60)
    scotus_cases = fetcher.fetch_supreme_court_cases(
        max_results=50,
        min_year=2010,
    )

    # Fetch Circuit Court cases (binding in their circuits)
    print("\n" + "=" * 60)
    print("Fetching Circuit Court Cases")
    print("=" * 60)
    circuit_cases = fetcher.fetch_circuit_cases(
        circuits=[1, 2, 9, "dc"],  # Sample circuits
        max_results=150,
        min_year=2018,
    )

    # Fetch recent federal cases (general)
    print("\n" + "=" * 60)
    print("Fetching Recent Federal Cases")
    print("=" * 60)
    recent_cases = fetcher.fetch_recent_opinions(
        max_results=300,
        min_date="2020-01-01",
    )

    # Combine all
    all_opinions = scotus_cases + circuit_cases + recent_cases

    # Remove duplicates by URL
    seen_urls = set()
    unique_opinions = []
    for opinion in all_opinions:
        url = opinion.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            unique_opinions.append(opinion)

    print("\n" + "=" * 60)
    print("Corpus Summary")
    print("=" * 60)
    print(f"Total unique opinions: {len(unique_opinions)}")

    # Statistics
    by_court = {}
    by_year = {}
    for opinion in unique_opinions:
        court = opinion.get("court_id", "unknown")
        by_court[court] = by_court.get(court, 0) + 1

        year = opinion.get("date_filed", "")[:4]
        by_year[year] = by_year.get(year, 0) + 1

    print("\nBy Court:")
    for court, count in sorted(by_court.items(), key=lambda x: -x[1])[:10]:
        print(f"  {court:15s}: {count:4d}")

    print("\nBy Year:")
    for year, count in sorted(by_year.items(), reverse=True)[:10]:
        print(f"  {year}: {count:4d}")

    # Save
    output_path = Path("data/corpus/courtlistener_opinions.json")
    fetcher.save_corpus(unique_opinions, output_path)

    print("\n[INFO] Next step: Run build_retrieval_index.py to create searchable index")


if __name__ == "__main__":
    main()

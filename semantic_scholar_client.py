"""
Semantic Scholar API client - uses the bulk search API
"""

import asyncio
import aiohttp
import json
from typing import List, Optional
from datetime import datetime
from pathlib import Path
from models import Paper


class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self, calls_per_second: float = 0.1):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0

    async def wait(self):
        """Wait if needed to respect rate limit"""
        now = datetime.now().timestamp()
        time_since_last = now - self.last_call_time
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            await asyncio.sleep(wait_time)
        self.last_call_time = datetime.now().timestamp()


class SemanticScholarClient:
    """Wrapper for Semantic Scholar Bulk Search API"""

    def __init__(self, api_key: Optional[str] = None, save_raw_json: bool = True):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.api_key = api_key
        self.rate_limiter = RateLimiter(calls_per_second=0.1)
        self.max_retries = 5
        self.retry_delay = 2
        self.last_total = 0
        self.save_raw_json = save_raw_json
        self.raw_data_dir = Path("results/raw_api_responses")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        print("[INFO] Semantic Scholar Bulk API client initialized")
        if api_key:
            print("[INFO] Using API key for authentication")
        if save_raw_json:
            print(f"[INFO] Raw API responses will be saved to {self.raw_data_dir}")

    def _get_headers(self) -> dict:
        """Get request headers with API key if available"""
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def _make_request(
        self, url: str, params: dict, retry_count: int = 0
    ) -> Optional[dict]:
        """Make HTTP request with retry logic"""
        try:
            await self.rate_limiter.wait()

            timeout = aiohttp.ClientTimeout(total=30)
            headers = self._get_headers()

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        print(f"[WARNING] Rate limit. Waiting 15s...")
                        await asyncio.sleep(15)
                        if retry_count < self.max_retries:
                            return await self._make_request(
                                url, params, retry_count + 1
                            )
                        return None
                    elif response.status == 403:
                        text = await response.text()
                        print(f"[ERROR] Forbidden (403): {text[:200]}")
                        return None
                    else:
                        text = await response.text()
                        print(f"[ERROR] API returned {response.status}: {text[:100]}")
                        return None

        except Exception as e:
            print(f"[ERROR] Request failed: {e}")
            if retry_count < self.max_retries:
                await asyncio.sleep(self.retry_delay)
                return await self._make_request(url, params, retry_count + 1)
            return None

    async def search(
        self,
        query: str,
        max_results: int = 20,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        min_date: Optional[str] = None,
        fields_of_study: Optional[List[str]] = None,
        sort: str = "publicationDate",
        offset: int = 0,
    ) -> List[Paper]:
        """Search for papers using Bulk Search API"""

        url = f"{self.base_url}/paper/search/bulk"

        params = {
            "query": query,
            "fields": "title,abstract,url,venue,year,publicationDate,paperId,citationCount,authors,openAccessPdf",
            "limit": min(max_results, 1000),
            "sort": sort,
            "offset": offset,
        }

        if min_date:
            params["publicationDateOrYear"] = f"{min_date}:"
        elif year_from and year_to:
            params["year"] = f"{year_from}-{year_to}"
        elif year_from:
            params["year"] = f"{year_from}-"

        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        print(f"[Search] Query: {query[:50]}...")

        all_papers = []
        token = None
        retrieved_count = 0

        while retrieved_count < max_results:
            if token:
                params["token"] = token

            data = await self._make_request(url, params)

            if self.save_raw_json and data:
                self._save_raw_response(data, query, offset, retrieved_count)

            if not data:
                break

            if "error" in data:
                print(f"[ERROR] API error: {data.get('error', 'Unknown')}")
                break

            papers_data = data.get("data", [])
            if not papers_data:
                break

            total = data.get("total", 0)
            if total > 0:
                self.last_total = total
                if retrieved_count == 0:
                    print(f"[Search] Total found: {total}")

            for paper_data in papers_data:
                paper = self._parse_paper(paper_data)
                if paper:
                    all_papers.append(paper)
                    retrieved_count += 1

                if retrieved_count >= max_results:
                    break

            token = data.get("next")
            if not token:
                break

            await asyncio.sleep(1)

        print(f"[Search] Retrieved {len(all_papers)} papers")
        return all_papers[:max_results]

    def _save_raw_response(self, data: dict, query: str, offset: int, count: int):
        """Save raw API response to JSON file"""
        safe_query = "".join(c if c.isalnum() else "_" for c in query[:30])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_query}_offset{offset}_count{count}_{timestamp}.json"
        filepath = self.raw_data_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Raw API response saved to {filepath}")

    def _parse_paper(self, paper_data: dict) -> Optional[Paper]:
        """Parse paper data into Paper object"""

        pub_date = paper_data.get("publicationDate")
        if not pub_date:
            year_val = paper_data.get("year")
            if year_val:
                pub_date = f"{year_val}-01-01"
            else:
                pub_date = "Unknown"

        authors = []
        for author in paper_data.get("authors", []):
            if isinstance(author, dict) and "name" in author:
                authors.append(author["name"])
            elif isinstance(author, str):
                authors.append(author)

        open_access_url = None
        open_access = paper_data.get("openAccessPdf")
        if open_access:
            open_access_url = open_access.get("url")

        return Paper(
            title=paper_data.get("title", "Untitled"),
            authors=authors,
            abstract=paper_data.get("abstract") or "Abstract not available",
            publication_date=pub_date,
            url=paper_data.get(
                "url",
                f"https://www.semanticscholar.org/paper/{paper_data.get('paperId', '')}",
            ),
            venue=paper_data.get("venue", "Preprint"),
            paper_id=paper_data.get("paperId"),
            citation_count=paper_data.get("citationCount", 0),
            year=paper_data.get("year"),
            open_access_pdf=open_access_url,
        )


__all__ = ["SemanticScholarClient"]

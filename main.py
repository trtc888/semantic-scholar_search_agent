"""
Main entry point for the Paper Search Agent
Using CrewAI with Groq LLM
"""

import asyncio
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import config
from models import Paper, SearchResult
from semantic_scholar_client import SemanticScholarClient
from agents import generate_search_query, evaluate_paper, refine_keywords

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")


class PaperSearchAgent:
    """Main agent for searching academic papers"""

    def __init__(self, session_id: str = None):
        self.client = SemanticScholarClient()
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        self.result_json_path = self.results_dir / f"results_{self.session_id}.json"
        self.result_md_path = self.results_dir / f"results_{self.session_id}.md"
        self.matching_papers = []
        self._init_results_file("")

    async def search(
        self,
        user_query: str,
        on_paper_found=None,
        on_batch_complete=None,
        should_pause=None,
        provider: str = None,
        model: str = None,
        api_key: str = None,
        batch_size: int = 100,
        max_results: int = 100,
    ) -> SearchResult:
        """Execute the paper search workflow with batch processing"""

        self.matching_papers = []
        self.search_completed = False
        self.papers_processed = 0

        print("=" * 60)
        print("🔍 Paper Search Agent")
        print("=" * 60)
        print(f"Query: {user_query}")
        print()

        query_info = generate_search_query(
            user_query, provider=provider, model=model, api_key=api_key
        )
        query_string = query_info["keywords"]
        min_date = query_info["min_date"]
        domain = query_info["domain"]

        if min_date and "-" in min_date:
            min_year = int(min_date.split("-")[0])
        else:
            min_year = None

        print(f"Keywords: {query_string}")
        print(f"Domain: {domain}")
        print(f"Min Date: {min_date} (year: {min_year})")
        print()

        self._save_query_to_file(user_query, query_string, min_date, domain)
        self._init_results_file(user_query)

        min_year = min_year
        initial_papers = await self._search_papers(
            query_string, 1, min_year, sort="publicationDate"
        )

        total_found = 0
        if initial_papers:
            total_found = (
                self.client.last_total if hasattr(self.client, "last_total") else 0
            )

        max_results = min(total_found, max_results) if total_found > 0 else max_results
        print(
            f"Found {total_found} papers total. Will check up to {max_results} papers in batches of {batch_size}..."
        )

        papers = []
        all_filtered = []
        offset = 0

        while offset < max_results and not self.search_completed:
            batch_count = min(batch_size, max_results - offset)
            print(f"\n[Fetching] Papers {offset + 1} to {offset + batch_count}...")

            batch_papers = await self._search_papers(
                query_string,
                batch_count,
                min_year,
                min_date=min_date,
                sort="publicationDate",
                offset=offset,
            )

            if not batch_papers:
                break

            papers.extend(batch_papers)

            for paper in batch_papers:
                if min_date and not paper.is_after_date(min_date):
                    print(
                        f"✗ Filtered (date too early): {paper.title[:50]}... ({paper.publication_date} < {min_date})"
                    )
                else:
                    all_filtered.append(paper)

            offset += len(batch_papers)
            self.papers_processed = offset

            if on_batch_complete:
                on_batch_complete(len(self.matching_papers), offset, total_found)

            if offset < max_results and should_pause and should_pause():
                print(
                    f"\n[Pause] Paused after checking {offset} papers. {len(self.matching_papers)} matching so far."
                )
                break

        if all_filtered:
            print(f"\n[Evaluating] All {len(all_filtered)} date-matched papers...")
            await self._evaluate_papers(
                all_filtered,
                user_query,
                min_date,
                on_paper_found,
                provider,
                model,
                api_key,
            )

        if offset >= max_results or self.search_completed:
            self.search_completed = True

        matching_papers = self.matching_papers.copy()

        if len(matching_papers) < 5 and not self.search_completed:
            retry_count = 0
            max_retries = 5

            while (
                len(matching_papers) < 5
                and retry_count < max_retries
                and not self.search_completed
            ):
                print(
                    f"\n⚠ Only {len(matching_papers)} papers met criteria. Refining keywords..."
                )

                current_keywords = query_string.replace("+", " ")
                query_info = refine_keywords(
                    user_query,
                    current_keywords,
                    retry_count + 1,
                    min_date=min_date,
                    provider=provider,
                    model=model,
                    api_key=api_key,
                )
                query_string = query_info["keywords"]
                min_date = query_info["min_date"]
                domain = query_info["domain"]

                if min_date and "-" in min_date:
                    min_year = int(min_date.split("-")[0])
                else:
                    min_year = None

                print(f"New Keywords: {query_string}")
                print(f"New Min Date: {min_date} (year: {min_year})")

                self._save_query_to_file(
                    user_query, query_string, min_date, domain, version=retry_count + 2
                )

                offset = 0
                papers = []
                all_filtered = []

                while offset < max_results and not self.search_completed:
                    batch_count = min(batch_size, max_results - offset)
                    print(
                        f"\n[Retry Batch] Fetching papers {offset + 1} to {offset + batch_count}..."
                    )
                    batch_papers = await self._search_papers(
                        query_string,
                        batch_count,
                        min_year,
                        min_date=min_date,
                        sort="publicationDate",
                        offset=offset,
                    )

                    if not batch_papers:
                        break

                    papers.extend(batch_papers)

                    for paper in batch_papers:
                        if min_date and not paper.is_after_date(min_date):
                            print(
                                f"✗ Filtered (date too early): {paper.title[:50]}... ({paper.publication_date} < {min_date})"
                            )
                        else:
                            all_filtered.append(paper)

                    offset += len(batch_papers)
                    self.papers_processed = offset

                if all_filtered:
                    print(
                        f"\n[Retry Batch] Evaluating {len(all_filtered)} date-matched papers..."
                    )

                    await self._evaluate_papers(
                        all_filtered,
                        user_query,
                        min_date,
                        on_paper_found,
                        provider,
                        model,
                        api_key,
                    )

                    if on_batch_complete:
                        on_batch_complete(
                            len(self.matching_papers), offset, total_found
                        )

                    if offset < max_results and should_pause and should_pause():
                        print(f"\n[Pause] Paused after checking {offset} papers.")
                        break

                if offset >= max_results:
                    self.search_completed = True

                matching_papers = self.matching_papers.copy()
                retry_count += 1

        matching_papers.sort(key=lambda x: x.relevance_score, reverse=True)

        self.search_completed = True
        summary = self._generate_summary(user_query, papers, matching_papers)

        result = SearchResult(
            query=user_query,
            total_found=len(papers),
            papers=matching_papers,
            evaluation_summary=summary,
            generated_at=datetime.now().isoformat(),
        )

        return result

    async def _search_papers(
        self,
        query: str,
        max_results: int,
        min_year: Optional[int] = None,
        min_date: Optional[str] = None,
        sort: str = "relevance",
        offset: int = 0,
    ) -> List[Paper]:
        """Search for papers using API"""

        try:
            papers = await self.client.search(
                query,
                max_results,
                year_from=min_year,
                min_date=min_date,
                sort=sort,
                offset=offset,
            )
            return papers
        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            return []

    def _save_query_to_file(
        self,
        user_query: str,
        keywords: str,
        min_date: str,
        domain: str,
        version: int = 1,
    ):
        """Save the generated query to a markdown file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        content = f"""# Search Query

**Timestamp:** {timestamp}
**Version:** {version}

**User Query:**
{user_query}

**Generated Keywords:**
{keywords}

**Domain:**
{domain}

**Earliest Date:**
{min_date}

**Semantic Scholar Query URL:**
https://www.semanticscholar.org/search?q={keywords}
"""
        queries_dir = Path("queries")
        queries_dir.mkdir(exist_ok=True)

        filename = f"query_{timestamp}"
        if version > 1:
            filename = f"query_{timestamp}_v{version}"

        filepath = queries_dir / f"{filename}.md"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"✅ Query saved to {filepath}")

    async def _evaluate_papers(
        self,
        papers: List[Paper],
        user_query: str,
        min_date: str,
        on_paper_found=None,
        provider: str = None,
        model: str = None,
        api_key: str = None,
    ):
        """Evaluate papers using LLM (date filter already applied)"""

        match_count = len(self.matching_papers)

        for paper in papers:
            eval_result = evaluate_paper(
                paper.title,
                paper.abstract,
                paper.publication_date,
                user_query,
                min_date,
                provider=provider,
                model=model,
                api_key=api_key,
            )

            paper.relevance_score = eval_result.get("relevance_score", 0)
            paper.meets_criteria = eval_result.get("meets_criteria", False)

            if paper.meets_criteria:
                match_count += 1
                print(
                    f"✓ Match: {paper.title[:50]}... (score: {paper.relevance_score:.1f})"
                )
                saved_paper = self._append_paper_to_results(paper, match_count)

                if on_paper_found and saved_paper:
                    on_paper_found(saved_paper, match_count)

    def _init_results_file(self, user_query: str):
        """Initialize the results markdown file (only if not exists)"""
        if not self.result_md_path.exists():
            content = f"""# Paper Search Results

**Session ID**: {self.session_id}
**Generated At**: {datetime.now().isoformat()}

---

## Papers

"""
            with open(self.result_md_path, "w", encoding="utf-8") as f:
                f.write(content)

        if not self.result_json_path.exists():
            with open(self.result_json_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _append_paper_to_results(self, paper: Paper, count: int):
        """Append a paper to the results files"""
        self.matching_papers.append(paper)

        content = f"""
### {count}. {paper.title}

**Score**: {paper.relevance_score:.1f}/10
**Meets Criteria**: ✓ Yes
**Date**: {paper.publication_date}
**URL**: {paper.url}

**Authors**: {", ".join(paper.authors[:5])}{" et al." if len(paper.authors) > 5 else ""}
**Venue**: {paper.venue if paper.venue else "Preprint"}
**Citations**: {paper.citation_count if paper.citation_count else "N/A"}

**Abstract**: {paper.abstract[:500]}...

---
"""
        with open(self.result_md_path, "a", encoding="utf-8") as f:
            f.write(content)

        with open(self.result_json_path, "r", encoding="utf-8") as f:
            try:
                papers_list = json.load(f)
            except:
                papers_list = []

        papers_list.append(paper.to_dict())

        with open(self.result_json_path, "w", encoding="utf-8") as f:
            json.dump(papers_list, f, indent=2, ensure_ascii=False)

        return paper

    def _generate_summary(
        self, user_query: str, all_papers: List[Paper], matching_papers: List[Paper]
    ) -> str:
        """Generate evaluation summary"""

        if not matching_papers:
            return f"Found {len(all_papers)} papers, but none met the criteria."

        avg_score = sum(p.relevance_score for p in matching_papers) / len(
            matching_papers
        )

        summary = (
            f"Found {len(all_papers)} papers, {len(matching_papers)} met the criteria. "
        )
        summary += f"Average relevance score: {avg_score:.1f}/10. "

        if matching_papers:
            top_paper = matching_papers[0]
            summary += f"Top result: '{top_paper.title[:50]}...' with score {top_paper.relevance_score:.1f}/10."

        return summary


async def main():
    """Main function"""

    print("\n" + "=" * 60)
    print("🔬 Paper Search Agent")
    print("=" * 60)

    print("\nEnter your research query (press Enter when done):")
    query = input("> ").strip()

    if not query:
        print("No query entered. Exiting.")
        return

    agent = PaperSearchAgent()

    result = await agent.search(user_query=query)

    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)

    print(f"\nTotal Found: {result.total_found}")
    print(f"Meeting Criteria: {len(result.papers)}")
    print(f"\nSummary: {result.evaluation_summary}")

    print("\n" + "-" * 60)
    print("📄 PAPERS")
    print("-" * 60)

    for i, paper in enumerate(result.papers, 1):
        print(f"\n{i}. {paper.title}")
        print(f"   Score: {paper.relevance_score:.1f}/10")
        print(f"   Date: {paper.publication_date}")
        print(f"   URL: {paper.url}")

    output = {"json": result.to_dict()}

    timestamp = agent.timestamp
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / f"results_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(output["json"], f, indent=2, ensure_ascii=False)

    with open(results_dir / f"results_{timestamp}.md", "a", encoding="utf-8") as f:
        f.write(f"\n\n## Summary\n\n{result.evaluation_summary}\n")

    print(f"\n✅ Results saved to results_{timestamp}.json and results_{timestamp}.md")


if __name__ == "__main__":
    asyncio.run(main())

"""
Semantic Scholar scraper - fetches papers from search page URL
"""

import asyncio
import aiohttp
import re
import time
from typing import List, Optional
from bs4 import BeautifulSoup
from datetime import datetime
from models import Paper


class SemanticScholarScraper:
    """Scrape Semantic Scholar search results from URL"""
    
    def __init__(self):
        self.base_url = "https://www.semanticscholar.org/search"
        self.rate_limiter = RateLimiter(calls_per_second=0.5)
    
    async def search(
        self,
        query: str,
        max_results: int = 20,
        year_from: Optional[int] = None
    ) -> List[Paper]:
        """Search for papers using Semantic Scholar search URL"""
        papers = []
        
        params = {
            "q": query,
            "sort": "relevance"
        }
        
        if year_from:
            params["year"] = f"{year_from}-"
        
        await self.rate_limiter.wait()
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        print(f"[ERROR] Request failed with status {response.status}")
                        return []
                    
                    html = await response.text()
                    papers = self._parse_search_results(html)
                    
        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
        
        return papers[:max_results]
    
    def _parse_search_results(self, html: str) -> List[Paper]:
        """Parse HTML to extract paper information"""
        papers = []
        soup = BeautifulSoup(html, 'html.parser')
        
        paper_elements = soup.find_all('div', {'data-testid': 'result-card'})
        
        if not paper_elements:
            paper_elements = soup.find_all('div', class_=re.compile(r'card|paper|result'))
        
        for elem in paper_elements:
            try:
                paper = self._extract_paper_data(elem)
                if paper and paper.title:
                    papers.append(paper)
            except Exception as e:
                continue
        
        return papers
    
    def _extract_paper_data(self, element) -> Optional[Paper]:
        """Extract paper data from a result card element"""
        title = ""
        abstract = ""
        url = ""
        authors = []
        publication_date = ""
        venue = ""
        citation_count = None
        year = None
        paper_id = None
        
        title_elem = element.find('h2') or element.find('a', {'data-testid': 'title'})
        if title_elem:
            title = title_elem.get_text(strip=True)
        
        if not title:
            title_link = element.find('a', href=re.compile(r'/paper/'))
            if title_link:
                title = title_link.get_text(strip=True)
        
        if not title:
            return None
        
        link_elem = element.find('a', href=re.compile(r'/paper/'))
        if link_elem:
            href = link_elem.get('href', '')
            if href:
                if href.startswith('/'):
                    url = f"https://www.semanticscholar.org{href}"
                else:
                    url = href
                match = re.search(r'/paper/([^/?]+)', href)
                if match:
                    paper_id = match.group(1)
        
        abstract_elem = element.find('div', {'data-testid': 'abstract'})
        if abstract_elem:
            abstract = abstract_elem.get_text(strip=True)
        
        if not abstract:
            abstract_data = element.get('data-abstract')
            if abstract_data:
                abstract = abstract_data
        
        author_elems = element.find_all('a', {'data-testid': 'author'})
        for author_elem in author_elems:
            author_name = author_elem.get_text(strip=True)
            if author_name and author_name not in authors:
                authors.append(author_name)
        
        if not authors:
            author_span = element.find_all('span', class_=re.compile(r'author'))
            for span in author_span:
                name = span.get_text(strip=True)
                if name and name not in authors:
                    authors.append(name)
        
        date_elem = element.find('span', {'data-testid': 'date'}) or element.find('span', class_=re.compile(r'date|year'))
        if date_elem:
            publication_date = date_elem.get_text(strip=True)
            year_match = re.search(r'\d{4}', publication_date)
            if year_match:
                year = int(year_match.group())
        
        if not publication_date and year:
            publication_date = f"{year}-01-01"
        
        venue_elem = element.find('span', {'data-testid': 'venue'}) or element.find('span', class_=re.compile(r'venue|journal'))
        if venue_elem:
            venue = venue_elem.get_text(strip=True)
        
        citation_elem = element.find('span', {'data-testid': 'citations'}) or element.find('span', class_=re.compile(r'citation|cited'))
        if citation_elem:
            citation_text = citation_elem.get_text(strip=True)
            citation_match = re.search(r'(\d+)', citation_text)
            if citation_match:
                citation_count = int(citation_match.group(1))
        
        if not abstract:
            abstract = "Abstract not available"
        
        if not publication_date:
            publication_date = "Unknown"
        
        return Paper(
            title=title,
            authors=authors,
            abstract=abstract,
            publication_date=publication_date,
            url=url,
            venue=venue,
            paper_id=paper_id,
            citation_count=citation_count,
            year=year
        )


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, calls_per_second: float = 0.5):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0
    
    async def wait(self):
        """Wait if needed to respect rate limit"""
        now = time.time()
        time_since_last = now - self.last_call_time
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            await asyncio.sleep(wait_time)
        self.last_call_time = time.time()


__all__ = ['SemanticScholarScraper']

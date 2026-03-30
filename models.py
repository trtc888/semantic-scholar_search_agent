"""
Data models for the Paper Search Agent
Using Pydantic V2
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class Paper(BaseModel):
    """Structured paper metadata from Semantic Scholar"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    title: str
    authors: List[str]
    abstract: str
    publication_date: str
    url: str
    venue: str = ""
    paper_id: Optional[str] = None
    doi: Optional[str] = None
    citation_count: Optional[int] = None
    year: Optional[int] = None
    open_access_pdf: Optional[str] = None
    relevance_score: float = 0.0
    meets_criteria: bool = False
    full_text_available: bool = False
    
    def get_date_obj(self) -> Optional[datetime]:
        """Parse publication date to datetime object"""
        try:
            return datetime.fromisoformat(self.publication_date.replace("Z", "+00:00"))
        except:
            try:
                return datetime.strptime(self.publication_date, "%Y-%m-%d")
            except:
                return None
    
    def is_after_date(self, min_date: str) -> bool:
        """Check if paper is published after the given date"""
        try:
            pub_date = self.get_date_obj()
            if not pub_date:
                return False
            min_date_obj = datetime.strptime(min_date, "%Y-%m-%d")
            return pub_date >= min_date_obj
        except:
            return False
    
    def format_for_display(self) -> str:
        """Format paper for display in report"""
        return f"""
### {self.title}

**Authors**: {', '.join(self.authors[:5])}{' et al.' if len(self.authors) > 5 else ''}
**Venue**: {self.venue if self.venue else 'Preprint'}
**Date**: {self.publication_date}
**Citations**: {self.citation_count if self.citation_count else 'N/A'}
**Relevance Score**: {self.relevance_score:.1f}/10
**URL**: {self.url}
**Full Text Available**: {'Yes' if self.full_text_available else 'No'}

**Abstract**: {self.abstract[:500]}...
"""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output"""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "publication_date": self.publication_date,
            "url": self.url,
            "venue": self.venue,
            "paper_id": self.paper_id,
            "doi": self.doi,
            "citation_count": self.citation_count,
            "year": self.year,
            "open_access_pdf": self.open_access_pdf,
            "relevance_score": self.relevance_score,
            "meets_criteria": self.meets_criteria,
            "full_text_available": self.full_text_available
        }


class UserQuery(BaseModel):
    """User's search query parsed into structured format"""
    research_interest: str
    domain: str
    earliest_date: str
    additional_criteria: Optional[str] = None
    
    def to_prompt(self) -> str:
        """Convert to prompt string for LLM"""
        criteria = f" in {self.additional_criteria}" if self.additional_criteria else ""
        return f"{self.research_interest} for {self.domain}{criteria}, published after {self.earliest_date}"


class EvaluationResult(BaseModel):
    """Evaluation of a single paper"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    paper_title: str
    relevance_score: float = Field(ge=0, le=10, description="Quality score 0-10")
    reason: str = Field(description="Detailed evaluation rationale")
    meets_criteria: bool = Field(description="Whether paper meets all criteria")
    strengths: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    """Final search result with all papers"""
    query: str
    total_found: int
    papers: List[Paper]
    evaluation_summary: str
    generated_at: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output"""
        return {
            "query": self.query,
            "total_found": self.total_found,
            "papers": [p.to_dict() for p in self.papers],
            "evaluation_summary": self.evaluation_summary,
            "generated_at": self.generated_at
        }
    
    def to_report(self) -> str:
        """Generate human-readable report"""
        lines = [
            "# Paper Search Results",
            "",
            f"**Query**: {self.query}",
            f"**Total Found**: {self.total_found}",
            f"**Papers Meeting Criteria**: {len([p for p in self.papers if p.meets_criteria])}",
            f"**Generated At**: {self.generated_at}",
            "",
            "---",
            "",
            "## Summary",
            self.evaluation_summary,
            "",
            "---",
            "",
            "## Papers"
        ]
        
        for i, paper in enumerate(self.papers, 1):
            lines.append(f"\n### {i}. {paper.title}")
            lines.append(f"**Score**: {paper.relevance_score:.1f}/10")
            lines.append(f"**Meets Criteria**: {'✓ Yes' if paper.meets_criteria else '✗ No'}")
            lines.append(paper.format_for_display())
        
        return "\n".join(lines)

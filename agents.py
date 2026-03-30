"""
CrewAI Agents and Tools for Paper Search
Simplified version - LLM wrappers only
"""

import json
import re
from pathlib import Path


import config

llm = None
current_provider = None
current_model = None


def load_prompt() -> str:
    """Load the prompt template from prompt.md"""
    prompt_path = Path(__file__).parent / "prompts" / "prompt.md"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_llm(
    provider: str = None,
    model: str = None,
    api_key: str = None,
    force_new: bool = False,
):
    """Get or create LLM instance for specified provider"""
    global llm, current_provider, current_model

    provider = provider or config.CURRENT_PROVIDER
    model = model or config.CURRENT_MODEL

    if api_key is None:
        provider_config = config.LLM_PROVIDERS.get(provider, {})
        api_key = provider_config.get("api_key", "")

    if (
        not force_new
        and llm is not None
        and current_provider == provider
        and current_model == model
    ):
        return llm

    current_provider = provider
    current_model = model

    if not api_key:
        raise ValueError(f"API key not found for provider: {provider}")

    if provider == "groq":
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=model, api_key=api_key, temperature=0.3)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, api_key=api_key, temperature=0.3)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=model, anthropic_api_key=api_key, temperature=0.3)
    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/v1",
            temperature=0.3,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return llm


def generate_search_query(
    user_query: str,
    retry: int = 0,
    provider: str = None,
    model: str = None,
    api_key: str = None,
) -> dict:
    """Generate search query, domain and date using LLM"""

    prompt = load_prompt() + user_query

    try:
        llm_instance = get_llm(
            provider=provider, model=model, api_key=api_key, force_new=True
        )
        response = llm_instance.invoke(prompt)
        response_text = response.content.strip()

        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            result = json.loads(json_match.group())
            keywords = result.get("keywords", user_query)
            keywords = keywords.replace(" ", "+")

            return {
                "keywords": keywords,
                "domain": result.get("domain", ""),
                "min_date": result.get("min_date", ""),
            }

    except Exception as e:
        print(f"[WARNING] Query generation failed: {e}")

    keywords = user_query.replace(" ", "+")
    return {"keywords": keywords, "domain": "", "min_date": ""}


def refine_keywords(
    user_query: str,
    previous_keywords: str,
    retry: int = 1,
    min_date: str = None,
    provider: str = None,
    model: str = None,
    api_key: str = None,
) -> dict:
    """Refine keywords when not enough results found"""

    strategies = [
        "Use ONLY 2-4 most essential keywords. Make them SIMPLE and SHORT.",
        "Use completely different essential keywords. Think of alternative core terms.",
        "Use the most fundamental terms from the query. Max 5 keywords.",
    ]

    prompt = f"""Simplify keywords for academic paper search.

Previous Keywords: {previous_keywords}
Retry #{retry}

{strategies[min(retry - 1, len(strategies) - 1)]}

IMPORTANT: Use 2-3 keywords MAXIMUM. Fewer is better.

Return ONLY JSON:
{{
  "keywords": "keyword1 keyword2 keyword3 keyword4",
  "domain": "domain"
}}"""

    try:
        llm_instance = get_llm(
            provider=provider, model=model, api_key=api_key, force_new=True
        )
        response = llm_instance.invoke(prompt)
        response_text = response.content.strip()

        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            result = json.loads(json_match.group())
            keywords = result.get("keywords", previous_keywords)
            keywords = keywords.replace(" ", "+")

            return {
                "keywords": keywords,
                "domain": result.get("domain", ""),
                "min_date": min_date,
            }

    except Exception as e:
        print(f"[WARNING] Keyword refinement failed: {e}")

    words = previous_keywords.replace("+", " ").split()[:3]
    keywords = "+".join(words)

    return {"keywords": keywords, "domain": "", "min_date": min_date}


def evaluate_paper(
    paper_title: str,
    paper_abstract: str,
    pub_date: str,
    user_query: str,
    min_date: str,
    retry: int = 0,
    provider: str = None,
    model: str = None,
    api_key: str = None,
) -> dict:
    """Evaluate a single paper using LLM"""

    prompt = f"""Evaluate this paper for the user's research query.

User Query: {user_query}
Minimum Date Required: {min_date}

Paper Title: {paper_title}

Paper Abstract:
{paper_abstract[:1000]}

Publication Date: {pub_date}

Evaluate:
1. Does this paper match the user's research interests? (0-10)
2. Does it meet the date requirement ({min_date})?
3. Is the abstract relevant?

Return ONLY a JSON object:
{{
    "relevance_score": 8.5,
    "meets_criteria": true,
    "reason": "brief explanation"
}}"""

    try:
        llm_instance = get_llm(
            provider=provider, model=model, api_key=api_key, force_new=True
        )
        response = llm_instance.invoke(prompt)

        json_match = re.search(r"\{[\s\S]*\}", response.content)
        if json_match:
            result = json.loads(json_match.group())
            return result

    except Exception as e:
        if "429" in str(e) and retry < 3:
            import time

            wait_time = 2**retry
            print(f"[INFO] Rate limited, waiting {wait_time}s...")
            time.sleep(wait_time)
            return evaluate_paper(
                paper_title, paper_abstract, pub_date, user_query, min_date, retry + 1
            )
        print(f"[WARNING] Evaluation failed for '{paper_title}': {e}")

    return {
        "relevance_score": 5.0,
        "meets_criteria": False,
        "reason": "Evaluation failed",
    }

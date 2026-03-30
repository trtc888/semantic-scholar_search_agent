"""
Configuration for the Paper Search Agent
"""

import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDERS = {
    "groq": {
        "api_key": os.getenv("GROQ_API_KEY", ""),
        "default_model": "llama-3.1-8b-instant",
        "env_key": "GROQ_API_KEY"
    },
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "default_model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY"
    },
    "anthropic": {
        "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "default_model": "claude-3-haiku-20240307",
        "env_key": "ANTHROPIC_API_KEY"
    },
    "openrouter": {
        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
        "default_model": "google/gemini-flash-1.5-8b",
        "env_key": "OPENROUTER_API_KEY"
    }
}

CURRENT_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
CURRENT_MODEL = os.getenv("LLM_MODEL", LLM_PROVIDERS.get(CURRENT_PROVIDER, {}).get("default_model", "llama-3.1-8b-instant"))

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
SEMANTIC_SCHOLAR_SEARCH_URL = "https://www.semanticscholar.org/search"

DEFAULT_MAX_PAPERS = 20
DEFAULT_MIN_SCORE = 7.0

SEARCH_SETTINGS = {
    "max_results": 20,
    "sort": "relevance",
    "timeout": 30
}

RATE_LIMIT = {
    "min_interval": 1.0,
    "max_retries": 3
}

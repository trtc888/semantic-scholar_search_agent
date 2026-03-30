"""
Web UI for Paper Search Agent using Streamlit
"""

import streamlit as st
import asyncio
from datetime import datetime
from pathlib import Path
import json
import os
from dotenv import load_dotenv

import config
from main import PaperSearchAgent
from models import Paper


def init_session_state():
    if "search_done" not in st.session_state:
        st.session_state.search_done = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "found_papers" not in st.session_state:
        st.session_state.found_papers = []
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "groq"
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "llama-3.1-8b-instant"
    if "llm_api_key" not in st.session_state:
        st.session_state.llm_api_key = ""
    if "history" not in st.session_state:
        st.session_state.history = load_history()
    if "search_paused" not in st.session_state:
        st.session_state.search_paused = False
    if "papers_processed" not in st.session_state:
        st.session_state.papers_processed = 0
    if "total_papers" not in st.session_state:
        st.session_state.total_papers = 0


def save_history(history):
    """Save search history to file"""
    history_file = Path("results/history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def load_history():
    """Load search history from file"""
    history_file = Path("results/history.json")
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                return []
    return []


def display_paper(paper, index):
    with st.container():
        st.markdown(f"### {index}. {paper.get('title', 'Untitled')}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score", f"{paper.get('relevance_score', 0):.1f}/10")
        with col2:
            st.metric("Date", paper.get("publication_date", "N/A"))
        with col3:
            st.metric("Citations", paper.get("citation_count", "N/A") or "N/A")

        authors = paper.get("authors", [])
        st.markdown(
            f"**Authors**: {', '.join(authors[:5])}{' et al.' if len(authors) > 5 else ''}"
        )
        st.markdown(f"**Venue**: {paper.get('venue', 'Preprint')}")

        with st.expander("Abstract"):
            st.markdown(paper.get("abstract", "No abstract available"))

        st.link_button("View on Semantic Scholar", paper.get("url", ""))
        st.divider()


def load_existing_results(session_id: str):
    """Load existing results from JSON file"""
    results_dir = Path("results")
    json_path = results_dir / f"results_{session_id}.json"

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                return []
    return []


def main():
    init_session_state()

    st.set_page_config(
        page_title="Paper Search Agent",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔬 Paper Search Agent")
    st.markdown("Find relevant academic papers using AI-powered search")

    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("LLM Provider")
        provider = st.selectbox(
            "Provider",
            ["groq", "openai", "anthropic", "openrouter"],
            index=["groq", "openai", "anthropic", "openrouter"].index(
                st.session_state.llm_provider
            ),
            key="provider_select",
        )

        default_models = {
            "groq": "llama-3.1-8b-instant",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
            "openrouter": "google/gemini-flash-1.5-8b",
        }

        model = st.text_input(
            "Model Name",
            value=st.session_state.llm_model
            if st.session_state.llm_model
            else default_models.get(provider, ""),
            key="model_input",
        )

        saved_api_key = config.LLM_PROVIDERS.get(provider, {}).get("api_key", "")
        default_api_key = st.session_state.llm_api_key or saved_api_key
        st.session_state.saved_api_key = saved_api_key

        api_key = st.text_input(
            "API Key", value=default_api_key, type="password", key="api_key_input"
        )

        env_key = f"{provider.upper()}_API_KEY"
        env_file = Path(".env")
        if api_key and (not saved_api_key or api_key != saved_api_key):
            env_vars = {}
            if env_file.exists():
                with open(env_file, "r") as f:
                    for line in f:
                        if "=" in line:
                            k, v = line.strip().split("=", 1)
                            env_vars[k] = v
            env_vars[env_key] = api_key
            with open(env_file, "w") as f:
                for k, v in env_vars.items():
                    f.write(f"{k}={v}\n")
            load_dotenv(override=True)
            config.LLM_PROVIDERS[provider]["api_key"] = api_key
            st.session_state.saved_api_key = api_key
            st.rerun()

        if (
            provider != st.session_state.llm_provider
            or model != st.session_state.llm_model
            or api_key != st.session_state.llm_api_key
        ):
            st.session_state.llm_provider = provider
            st.session_state.llm_model = model
            st.session_state.llm_api_key = api_key

        st.divider()

        st.subheader("Search Query")
        query_input = st.text_area(
            "Research Query",
            height=100,
            placeholder="e.g., neural network solvers in computational chemistry",
            key="query_input",
        )

        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query_input.strip():
                if not api_key and not st.session_state.get("saved_api_key"):
                    st.error("Please enter API key")
                else:
                    st.session_state.current_query = query_input.strip()
                    st.session_state.session_id = datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )
                    st.session_state.found_papers = []
                    st.session_state.search_done = False

                    st.session_state.history.insert(
                        0,
                        {
                            "query": query_input.strip(),
                            "provider": provider,
                            "model": model,
                            "timestamp": st.session_state.session_id,
                            "papers_count": 0,
                        },
                    )
                    save_history(st.session_state.history)

                    st.rerun()
            else:
                st.warning("Please enter a query")

        st.divider()

        existing = load_existing_results(st.session_state.session_id)
        st.metric("Papers Found", len(existing))

        if existing:
            st.button(
                "🗑️ Clear Session",
                on_click=lambda: (
                    setattr(st.session_state, "found_papers", []),
                    setattr(st.session_state, "search_done", True),
                ),
            )

        st.divider()

        st.subheader("📜 History")
        if st.session_state.history:
            for i, item in enumerate(st.session_state.history[:10]):
                with st.container():
                    st.markdown(f"**{item.get('query', '')[:30]}...**")
                    st.caption(
                        f"{item.get('provider', '')} | {item.get('timestamp', '')}"
                    )
                    if st.button(f"Load", key=f"load_{i}"):
                        st.session_state.current_query = item.get("query", "")
                        st.session_state.session_id = item.get(
                            "timestamp", datetime.now().strftime("%Y%m%d_%H%M%S")
                        )
                        st.session_state.llm_provider = item.get("provider", "groq")
                        st.session_state.llm_model = item.get("model", "")
                        existing = load_existing_results(st.session_state.session_id)
                        st.session_state.found_papers = existing
                        st.session_state.search_done = True
                        st.rerun()
                    st.divider()

    if not st.session_state.search_done and st.session_state.current_query:
        agent = PaperSearchAgent(session_id=st.session_state.session_id)

        if st.session_state.search_paused:
            st.warning(
                f"⏸️ Search paused. Found {len(st.session_state.found_papers)} papers so far (checked {st.session_state.papers_processed}/{st.session_state.total_papers})."
            )
            if st.button("▶️ Continue Search", type="primary"):
                st.session_state.search_paused = False
                st.rerun()

        with st.spinner("Searching papers..."):
            existing = load_existing_results(st.session_state.session_id)
            st.session_state.found_papers = existing

            if existing:
                st.markdown(f"### Found {len(existing)} papers so far...")
                for i, paper in enumerate(existing, 1):
                    display_paper(paper, i)

            try:
                effective_api_key = (
                    st.session_state.llm_api_key
                    or st.session_state.get("saved_api_key", "")
                )

                pause_flag = {"paused": False}

                def should_pause():
                    return pause_flag["paused"]

                def on_batch(matched_count, processed, total):
                    st.session_state.papers_processed = processed
                    st.session_state.total_papers = total
                    pause_flag["paused"] = True

                result = asyncio.run(
                    agent.search(
                        st.session_state.current_query,
                        on_paper_found=lambda paper, count: (
                            st.session_state.found_papers.append(paper.to_dict())
                        ),
                        on_batch_complete=on_batch,
                        should_pause=should_pause,
                        provider=st.session_state.llm_provider,
                        model=st.session_state.llm_model,
                        api_key=effective_api_key,
                        batch_size=100,
                        max_results=500,
                    )
                )

                for item in st.session_state.history:
                    if item.get("timestamp") == st.session_state.session_id:
                        item["papers_count"] = len(st.session_state.found_papers)
                        break
                save_history(st.session_state.history)

                st.session_state.search_done = True
                st.session_state.search_paused = False
                st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.search_done = True

    elif st.session_state.found_papers:
        st.subheader(f"Results for: {st.session_state.current_query}")

        if st.session_state.search_paused:
            st.warning(
                f"⏸️ Search paused. Found {len(st.session_state.found_papers)} papers (checked {st.session_state.papers_processed}/{st.session_state.total_papers})."
            )
            if st.button("▶️ Continue Search", type="primary"):
                st.session_state.search_paused = False
                st.rerun()

        st.markdown(f"Found **{len(st.session_state.found_papers)}** papers")

        for i, paper in enumerate(st.session_state.found_papers, 1):
            display_paper(paper, i)

        st.divider()
        st.caption(
            f"Results saved to: results/results_{st.session_state.session_id}.md"
        )

    else:
        st.info("👈 Enter a research query and API key to get started")

        with st.expander("Example queries"):
            st.markdown("""
            - machine learning drug discovery
            - protein structure prediction
            - enzyme kinetics new formula
            - neural network solvers computational chemistry
            """)


if __name__ == "__main__":
    main()

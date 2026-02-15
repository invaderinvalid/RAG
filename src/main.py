#!/usr/bin/env python3
"""
Scientific Literature Explorer — main entry point.

Usage:
    python -m src.main ask "question"                # ask a research question (auto-discovers papers)
    python -m src.main ask "question" --session ID   # continue an existing session
    python -m src.main paper <arxiv_id> "question"   # deep-dive into a specific paper
    python -m src.main papers "query"                # search, list, and interactively select a paper to explore
    python -m src.main search "query"                # search ArXiv for papers
    python -m src.main ingest                        # download & index reference papers from ArXiv
    python -m src.main sessions                      # list all sessions
    python -m src.main workflow show                 # display current workflow config
    python -m src.main workflow toggle <stage> <on|off>
    python -m src.main workflow reorder <stage1,stage2,...>
    python -m src.main artifacts list                # list stored reasoning artifacts
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from src.core.config import ARTIFACTS_DIR, PROJECT_ROOT
from src.core.scaledown import ScaleDownClient, make_scaledown_llm_fn
from src.core.gemini import GeminiClient, GeminiRateLimitError, make_gemini_llm_fn, make_gemini_fast_fn
from src.core.llm import make_cot_handler, make_verify_handler, make_critique_handler
from src.core.session import Session, list_sessions, get_latest_session
from src.core.research_agent import ResearchAgent
from src.storage.artifact_store import ArtifactStore
from src.rag.pipeline import RAGPipeline
from src.papers.fetcher import PaperFetcher
from src.workflow.engine import ReasoningWorkflow, WorkflowStage

console = Console()

# Reference papers (ArXiv IDs only — always loaded as baseline)
REFERENCE_PAPERS = [
    "2511.14362",
]

WORKFLOW_CONFIG = PROJECT_ROOT / "workflow_config.json"


# ── Helpers ─────────────────────────────────────────────────────────────────── #

def _get_llm_fn():
    """Return the best available LLM function with automatic fallback.

    Priority:
      1. Gemini (primary) — wraps with auto-fallback to ScaleDown on 429
      2. ScaleDown compression-as-generation (fallback)
      3. None (template-based offline mode)
    """
    gemini_client = GeminiClient()
    sd_client = ScaleDownClient()

    if gemini_client.is_configured:
        gemini_fn = make_gemini_llm_fn()

        if sd_client.is_configured:
            # Wrap Gemini with automatic ScaleDown fallback on rate-limit
            sd_fn = make_scaledown_llm_fn()

            def resilient_llm_fn(system: str, user: str) -> str:
                try:
                    return gemini_fn(system, user)
                except GeminiRateLimitError:
                    console.print(
                        "[yellow]⚠ Gemini rate-limited — falling back to "
                        "ScaleDown compression-based generation[/yellow]"
                    )
                    return sd_fn(system, user)

            return resilient_llm_fn
        else:
            return gemini_fn

    if sd_client.is_configured:
        console.print("[dim]Gemini not configured — using ScaleDown fallback[/dim]")
        return make_scaledown_llm_fn()

    return None


def build_rag() -> RAGPipeline:
    """Construct the RAG pipeline with a shared ScaleDown client."""
    sd = ScaleDownClient()
    return RAGPipeline(scaledown=sd)


def build_workflow(context: str, llm_fn=None) -> ReasoningWorkflow:
    """Assemble the default reasoning workflow, loading saved config if any.

    Pass *llm_fn(system, user) -> str* to wire in any LLM backend.
    Without it, handlers produce template-based markdown.
    Uses a faster/cheaper LLM for verify & critique to reduce latency.
    """
    sd = ScaleDownClient()
    store = ArtifactStore(scaledown=sd)

    # Fast LLM for verify/critique (lower token cap + thinking budget)
    # Also wraps with ScaleDown fallback if both are configured
    fast_fn = None
    gemini_client = GeminiClient()
    sd_client_check = ScaleDownClient()
    if gemini_client.is_configured:
        gemini_fast = make_gemini_fast_fn()
        if sd_client_check.is_configured:
            sd_fn = make_scaledown_llm_fn()

            def resilient_fast_fn(system: str, user: str) -> str:
                try:
                    return gemini_fast(system, user)
                except GeminiRateLimitError:
                    return sd_fn(system, user)

            fast_fn = resilient_fast_fn
        else:
            fast_fn = gemini_fast
    elif sd_client_check.is_configured:
        fast_fn = make_scaledown_llm_fn()

    workflow = ReasoningWorkflow(store=store, compressor=sd)
    workflow.add_stage(WorkflowStage(
        name="cot",
        handler=make_cot_handler(context, llm_fn=llm_fn),
        description="Chain-of-thought reasoning over retrieved context",
    ))
    workflow.add_stage(WorkflowStage(
        name="self_verify",
        handler=make_verify_handler(context, llm_fn=fast_fn or llm_fn),
        description="Verify factual claims against source documents",
    ))
    workflow.add_stage(WorkflowStage(
        name="self_critique",
        handler=make_critique_handler(llm_fn=fast_fn or llm_fn),
        description="Critique completeness, accuracy, and clarity",
        enabled=False,  # disabled by default — enable with workflow toggle
    ))

    # Restore saved ordering / toggle state
    if WORKFLOW_CONFIG.exists():
        workflow.load_config(WORKFLOW_CONFIG)

    return workflow


# ── Commands ────────────────────────────────────────────────────────────────── #

def cmd_ingest() -> None:
    """Download reference papers via ArXiv API and index them."""
    fetcher = PaperFetcher()
    rag = build_rag()

    console.print(Panel("[bold]Ingesting ArXiv papers[/bold]"))
    for arxiv_id in REFERENCE_PAPERS:
        with console.status(f"Fetching arxiv:{arxiv_id} …"):
            try:
                text = fetcher.fetch_by_id(arxiv_id)
                n = rag.ingest(text, source=f"arxiv:{arxiv_id}")
                console.print(f"  ✓ [green]arxiv:{arxiv_id}[/green] — {n} chunks indexed")
            except Exception as exc:
                console.print(f"  ✗ [red]arxiv:{arxiv_id}[/red] — {exc}")

    console.print(f"\n[bold]Total chunks:[/bold] {len(rag.retriever.chunks)}")


def cmd_ask(question: str, session_id: str | None = None) -> None:
    """
    Smart pipeline:
      0. Classify whether question needs papers (triage — saves tokens!)
      1. Load/create session (with conversation history)
      2. If general/conceptual → answer directly via Gemini (skip paper fetch)
      3. Otherwise: extract keywords → search ArXiv → web fallback
      4. Compress context via ScaleDown
      5. Run reasoning workflow (COT → verify → critique) via Gemini
      6. Save turn to session
    """
    # ── Session ─────────────────────────────────────────────────────────── #
    if session_id:
        session = Session(session_id=session_id)
        console.print(f"[dim]Resuming session {session.session_id} ({session.turn_count} prior turns)[/dim]")
    else:
        latest = get_latest_session()
        if latest and latest.turn_count > 0:
            session = latest
            console.print(f"[dim]Continuing session {session.session_id} ({session.turn_count} prior turns)[/dim]")
        else:
            session = Session()
            console.print(f"[dim]New session {session.session_id}[/dim]")

    history_ctx = session.get_history_context(max_turns=5)

    console.print(Panel(f"[bold]Question:[/bold] {question}"))

    t_start = time.perf_counter()   # ── end-to-end latency tracking

    # ── Research Agent: triage + keyword extraction + discovery ─────────── #
    agent = ResearchAgent()
    rag = build_rag()

    with console.status("Classifying question & searching for papers …"):
        discovery = agent.discover(question, history=history_ctx, max_arxiv=3)

    triage = discovery.get("triage", {})
    complexity = triage.get("complexity", "research")
    console.print(f"[dim]Complexity: {complexity} — {triage.get('reason', '')}[/dim]")

    # ── DIRECT answer path (no papers needed) ───────────────────────────── #
    if discovery["source_type"] == "direct":
        llm_fn = _get_llm_fn()
        if llm_fn:
            console.print("[dim]General knowledge question — answering directly (no paper fetch)[/dim]")
            system = (
                "You are a scientific research assistant. Answer the question "
                "thoroughly in markdown with clear explanations, key concepts, "
                "and examples. Mention seminal papers or resources if relevant.\n"
            )
            if history_ctx:
                system += f"\n{history_ctx}\n"
            answer = llm_fn(system, question)
            console.print(Panel(Markdown(answer), title="[bold]ANSWER[/bold]"))
            session.add_turn(
                question, answer, sources=["gemini_direct"],
                metadata={"complexity": complexity, "source_type": "direct"},
            )
            console.print(f"\n[dim]Session {session.session_id} — turn {session.turn_count} saved[/dim]")
            return
        else:
            console.print("[yellow]Gemini not configured — falling through to paper search[/yellow]")

    kw = discovery["keywords"]
    console.print(f"[dim]Keywords: {', '.join(kw.get('keywords', []))}[/dim]")
    console.print(f"[dim]ArXiv query: {kw.get('arxiv_query', 'N/A')}[/dim]")

    sources_used: list[str] = []

    if discovery["source_type"] == "arxiv":
        table = Table(title="Discovered ArXiv Papers")
        table.add_column("ArXiv ID", style="cyan")
        table.add_column("Title", max_width=60)
        for p in discovery["arxiv_papers"]:
            table.add_row(p["arxiv_id"], p["title"])
        console.print(table)

        for aid, text in discovery["arxiv_texts"].items():
            n = rag.ingest(text, source=f"arxiv:{aid}")
            session.add_paper(aid)
            sources_used.append(f"arxiv:{aid}")
            console.print(f"  [green]✓ arxiv:{aid}[/green] — {n} chunks")

    elif discovery["source_type"] == "web":
        console.print("[yellow]No ArXiv papers found — using web search + GitHub[/yellow]")
        for wr in discovery["web_results"]:
            n = rag.ingest(wr["content"], source=wr["source"])
            sources_used.append(wr["source"])
            console.print(f"  [green]✓ Ingested {wr['source']}[/green] — {n} chunks")
    else:
        console.print("[yellow]No sources found. Asking Gemini directly.[/yellow]")

    # ── Also load any previously ingested session papers ─────────────────── #
    fetcher = PaperFetcher()
    for pid in session.ingested_papers:
        if pid not in [p.get("arxiv_id") for p in discovery.get("arxiv_papers", [])]:
            try:
                text = fetcher.fetch_by_id(pid)
                rag.ingest(text, source=f"arxiv:{pid}")
            except Exception:
                pass

    # ── Retrieve & compress ──────────────────────────────────────────────── #
    with console.status("Retrieving relevant context …"):
        result = rag.query(question)

    ctx = result["context"]

    if result["chunks"]:
        table = Table(title="Retrieved Context")
        table.add_column("Source", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Preview", max_width=60)
        for c in result["chunks"]:
            table.add_row(c["source"], f"{c['score']:.3f}", c["text"][:80] + "…")
        console.print(table)

    if result.get("compressed"):
        stats = result["compression_stats"]
        console.print(
            f"[dim]Compressed {stats['original_tokens']} → "
            f"{stats['compressed_tokens']} tokens "
            f"({stats['latency_ms']}ms)[/dim]\n"
        )

    # ── If still no context, ask Gemini directly with history ────────────── #
    if not ctx:
        llm_fn = _get_llm_fn()
        if llm_fn:
            console.print("[dim]No retrieval context — generating answer from Gemini knowledge…[/dim]")
            system = (
                "You are a scientific research assistant. Answer the question "
                "thoroughly in markdown. If you know relevant papers, GitHub repos, "
                "or documentation, include them.\n"
            )
            if history_ctx:
                system += f"\n{history_ctx}\n"
            answer = llm_fn(system, question)
            console.print(Panel(Markdown(answer), title="[bold]ANSWER[/bold]"))
            session.add_turn(question, answer, sources=sources_used)
            return
        else:
            console.print("[red]No context found and Gemini not configured.[/red]")
            return

    # ── Include conversation history in the context ──────────────────────── #
    if history_ctx:
        ctx = f"{history_ctx}\n\n---\n\n{ctx}"

    # ── Run reasoning workflow ───────────────────────────────────────────── #
    llm_fn = _get_llm_fn()
    workflow = build_workflow(ctx, llm_fn=llm_fn)

    # For conceptual questions, skip self_critique to save tokens
    if complexity == "conceptual":
        workflow.toggle_stage("self_critique", enabled=False)
        console.print("[dim]Conceptual question — skipping self_critique stage[/dim]")

    with console.status("Running reasoning pipeline (ScaleDown + Gemini) …"):
        artifacts = workflow.run(question, prompt_context=question)

    # Display results & collect final answer
    final_answer = ""
    for stage_name, meta in artifacts.items():
        content = Path(meta["storage"]).read_text(encoding="utf-8") if meta.get("storage") else ""
        subtitle_parts = []
        if meta.get("stage_latency_ms"):
            subtitle_parts.append(f"{meta['stage_latency_ms'] / 1000:.1f}s")
        if meta.get("compressed"):
            subtitle_parts.append("compressed")

        if stage_name == "cot":
            final_answer = content
            console.print(Panel(
                Markdown(content),
                title="[bold]ANSWER[/bold]",
                subtitle=" | ".join(subtitle_parts) if subtitle_parts else None,
            ))
        elif stage_name == "self_verify":
            summary, has_issues = _summarize_verification(content)
            latency = f" ({meta['stage_latency_ms'] / 1000:.1f}s)" if meta.get("stage_latency_ms") else ""
            console.print(f"\n{summary}{latency}")
            if has_issues:
                console.print(Panel(
                    Markdown(content),
                    title="[bold yellow]CITATION DETAILS[/bold yellow]",
                    border_style="yellow",
                ))
        elif stage_name == "self_critique":
            console.print(Panel(
                Markdown(content),
                title="[bold]CRITIQUE[/bold]",
                subtitle=" | ".join(subtitle_parts) if subtitle_parts else None,
                border_style="dim",
            ))
        else:
            console.print(Panel(
                Markdown(content),
                title=f"[bold]{stage_name.upper()}[/bold]",
                subtitle=" | ".join(subtitle_parts) if subtitle_parts else None,
            ))

    # ── Latency summary ──────────────────────────────────────────────────── #
    t_total = time.perf_counter() - t_start
    console.print(f"\n[dim]Total latency: {t_total:.1f}s[/dim]")

    # ── Save to session ──────────────────────────────────────────────────── #
    session.add_turn(
        question=question,
        answer=final_answer,
        sources=sources_used,
        metadata={
            "keywords": kw.get("keywords", []),
            "complexity": complexity,
            "source_type": discovery["source_type"],
            "artifacts": {k: v.get("id") for k, v in artifacts.items()},
        },
    )
    console.print(f"\n[dim]Session {session.session_id} — turn {session.turn_count} saved[/dim]")


def cmd_workflow(args: list[str]) -> None:
    """Manage workflow configuration."""
    workflow = build_workflow("")  # empty context for management commands

    if not args or args[0] == "show":
        table = Table(title="Reasoning Workflow Stages")
        table.add_column("#", justify="right")
        table.add_column("Stage")
        table.add_column("Enabled")
        table.add_column("Description")
        for i, s in enumerate(workflow.describe(), 1):
            status = "[green]✓[/green]" if s["enabled"] else "[red]✗[/red]"
            table.add_row(str(i), s["name"], status, s["description"])
        console.print(table)

    elif args[0] == "toggle" and len(args) >= 3:
        name, state = args[1], args[2].lower()
        workflow.toggle_stage(name, enabled=(state in ("on", "true", "1")))
        workflow.save_config(WORKFLOW_CONFIG)
        console.print(f"[green]Stage '{name}' → {'enabled' if state in ('on','true','1') else 'disabled'}[/green]")

    elif args[0] == "reorder" and len(args) >= 2:
        order = [s.strip() for s in args[1].split(",")]
        workflow.reorder(order)
        workflow.save_config(WORKFLOW_CONFIG)
        console.print(f"[green]Stages reordered: {order}[/green]")

    else:
        console.print("[yellow]Usage: workflow [show|toggle <stage> <on|off>|reorder <s1,s2,...>][/yellow]")


def cmd_artifacts(args: list[str]) -> None:
    """List stored reasoning artifacts."""
    store = ArtifactStore()
    atype = args[1] if len(args) > 1 else None

    items = store.list_artifacts(artifact_type=atype)
    if not items:
        console.print("[dim]No artifacts found.[/dim]")
        return

    table = Table(title="Stored Artifacts")
    table.add_column("ID")
    table.add_column("Type")
    table.add_column("Timestamp")
    table.add_column("Compressed")
    table.add_column("Tokens (orig→comp)")
    for m in items:
        table.add_row(
            m["id"],
            m["type"],
            m["timestamp"][:19],
            "✓" if m.get("compressed") else "✗",
            f"{m.get('original_tokens', '—')} → {m.get('compressed_tokens', '—')}",
        )
    console.print(table)


def cmd_sessions() -> None:
    """List all saved conversation sessions."""
    sessions = list_sessions()
    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Conversation Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Created")
    table.add_column("Updated")
    table.add_column("Turns", justify="right")
    table.add_column("Papers", justify="right")
    for s in sessions:
        table.add_row(
            s["session_id"],
            s["created_at"][:19],
            s["updated_at"][:19],
            str(s["turns"]),
            str(s["papers"]),
        )
    console.print(table)


def cmd_search(query: str) -> None:
    """Search ArXiv for papers matching a query."""
    fetcher = PaperFetcher()
    with console.status(f"Searching ArXiv for '{query}' …"):
        papers = fetcher.search(query, max_results=10)

    if not papers:
        console.print("[yellow]No papers found.[/yellow]")
        return

    table = Table(title=f"ArXiv results for '{query}'")
    table.add_column("#", justify="right")
    table.add_column("ArXiv ID", style="cyan")
    table.add_column("Title", max_width=60)
    table.add_column("Authors", max_width=30)
    table.add_column("Published")
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.authors[:2])
        if len(p.authors) > 2:
            authors += " et al."
        table.add_row(str(i), p.arxiv_id, p.title, authors, p.published[:10])
    console.print(table)


def _show_paper_list(papers: list) -> None:
    """Display the numbered paper list table."""
    table = Table(show_lines=True)
    table.add_column("#", justify="right", style="bold yellow", width=4)
    table.add_column("ArXiv ID", style="cyan", width=14)
    table.add_column("Title", max_width=55)
    table.add_column("Authors", max_width=25)
    table.add_column("Published", width=12)
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.authors[:2])
        if len(p.authors) > 2:
            authors += " et al."
        table.add_row(str(i), p.arxiv_id, p.title, authors, p.published[:10])
    console.print(table)


def _show_explorer_help() -> None:
    """Print the interactive command reference."""
    console.print(
        "\n[bold]Commands:[/bold]\n"
        "  [yellow]<number>[/yellow]   Select a paper by number\n"
        "  [yellow]<text>[/yellow]     Ask a follow-up about the current paper\n"
        "  [yellow]list[/yellow]       Show the paper list again\n"
        "  [yellow]back[/yellow]       De-select current paper and go back to list\n"
        "  [yellow]s[/yellow]          Search ArXiv for a new query\n"
        "  [yellow]q[/yellow]          Quit the explorer\n"
        "  [yellow]?[/yellow]          Show this help\n"
    )


def cmd_papers(query: str, session_id: str | None = None) -> None:
    """
    Interactive paper explorer with persistent session.

    Maintains a conversation session so you can ask follow-up questions
    about the same paper, switch between papers, or search again — all
    within one continuous session.
    """
    fetcher = PaperFetcher()

    # ── Session (shared across all interactions) ────────────────────────── #
    if session_id:
        session = Session(session_id=session_id)
    else:
        session = Session()
    console.print(f"[dim]Paper explorer session: {session.session_id}[/dim]")

    # Shared RAG instance & paper text cache to avoid refetching
    rag = build_rag()
    paper_cache: dict[str, str] = {}

    # ── Search ──────────────────────────────────────────────────────────── #
    with console.status(f"Searching ArXiv for '{query}' …"):
        papers = fetcher.search(query, max_results=10)

    if not papers:
        console.print("[yellow]No papers found on ArXiv for that query.[/yellow]")
        return

    console.print()
    _show_paper_list(papers)

    # ── State ────────────────────────────────────────────────────────────── #
    current_paper = None  # currently selected ArXivPaper

    def _prompt_label() -> str:
        if current_paper:
            short = current_paper.arxiv_id
            return f"  [{short}] → "
        return "  → "

    # ── Main interactive loop ────────────────────────────────────────────── #
    _show_explorer_help()

    while True:
        if current_paper:
            console.print(
                f"\n[bold]Current paper:[/bold] {current_paper.title}\n"
                "[dim]Type a question, a paper number, 'back', or '?' for help[/dim]"
            )
        else:
            console.print(
                "\n[bold]Select a paper by number[/bold] "
                "[dim](or 's' to search, '?' for help, 'q' to quit)[/dim]"
            )

        try:
            raw = input(_prompt_label()).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exiting paper explorer.[/dim]")
            return

        if not raw:
            continue

        low = raw.lower()

        # ── Global commands ──────────────────────────────────────────── #
        if low in ("q", "quit", "exit"):
            console.print(f"[dim]Session {session.session_id} saved ({session.turn_count} turns).[/dim]")
            return

        if low in ("?", "help"):
            _show_explorer_help()
            continue

        if low in ("list", "ls"):
            _show_paper_list(papers)
            continue

        if low in ("back", "b"):
            if current_paper:
                console.print(f"[dim]De-selected {current_paper.arxiv_id}[/dim]")
                current_paper = None
            else:
                console.print("[dim]No paper selected — nothing to go back from[/dim]")
            continue

        if low in ("s", "search"):
            try:
                new_query = input("  New search query: ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if not new_query:
                continue
            with console.status(f"Searching ArXiv for '{new_query}' …"):
                new_papers = fetcher.search(new_query, max_results=10)
            if new_papers:
                papers = new_papers
                current_paper = None
                console.print()
                _show_paper_list(papers)
            else:
                console.print("[yellow]No papers found.[/yellow]")
            continue

        # ── Select paper by number ───────────────────────────────────── #
        try:
            idx = int(raw)
            if 1 <= idx <= len(papers):
                current_paper = papers[idx - 1]
                console.print(f"\n[bold green]Selected:[/bold green] {current_paper.title}")
                console.print(f"[dim]ArXiv ID: {current_paper.arxiv_id}[/dim]")
                summary_preview = current_paper.summary[:250]
                if len(current_paper.summary) > 250:
                    summary_preview += "…"
                console.print(f"[dim]{summary_preview}[/dim]")
                console.print(
                    "\n[bold]Ask a question[/bold] (or press Enter for a general summary):"
                )
                try:
                    question = input("  → ").strip()
                except (EOFError, KeyboardInterrupt):
                    continue
                if not question:
                    question = (
                        "Provide a comprehensive summary of this paper, its key "
                        "contributions, methodology, results, and limitations."
                    )
                session = cmd_paper(
                    current_paper.arxiv_id, question,
                    session=session, _rag=rag, _paper_cache=paper_cache,
                )
                continue
            else:
                console.print(f"[red]Pick a number between 1 and {len(papers)}[/red]")
                continue
        except ValueError:
            pass

        # ── Follow-up question about current paper ───────────────────── #
        if current_paper:
            session = cmd_paper(
                current_paper.arxiv_id, raw,
                session=session, _rag=rag, _paper_cache=paper_cache,
            )
        else:
            console.print(
                "[red]Select a paper first (enter a number) or type '?' for help[/red]"
            )


def cmd_paper(
    arxiv_id: str,
    question: str,
    session_id: str | None = None,
    *,
    session: Session | None = None,
    _rag: object | None = None,
    _paper_cache: dict | None = None,
) -> Session:
    """
    Deep-dive into a specific paper with full anti-hallucination workflow.

    Downloads the paper, indexes it, and answers questions grounded in its content.
    Uses the full reasoning pipeline (COT → self-verify) to prevent hallucination,
    ensuring every claim is traced back to the actual paper text.

    Returns the session object for reuse in interactive loops.
    """
    # ── Session ─────────────────────────────────────────────────────────── #
    if session is None:
        if session_id:
            session = Session(session_id=session_id)
        else:
            latest = get_latest_session()
            session = latest if (latest and latest.turn_count > 0) else Session()

    console.print(f"[dim]Session {session.session_id}[/dim]")
    console.print(Panel(f"[bold]Paper:[/bold] arxiv:{arxiv_id}\n[bold]Question:[/bold] {question}"))

    t_start = time.perf_counter()

    # ── Fetch & ingest paper ────────────────────────────────────────────── #
    fetcher = PaperFetcher()
    rag = _rag or build_rag()

    # Reuse cached text if available
    if _paper_cache is not None and arxiv_id in _paper_cache:
        text = _paper_cache[arxiv_id]
    else:
        with console.status(f"Fetching arxiv:{arxiv_id} …"):
            try:
                text = fetcher.fetch_by_id(arxiv_id)
            except Exception as exc:
                console.print(f"[red]Could not fetch paper: {exc}[/red]")
                return session
        if _paper_cache is not None:
            _paper_cache[arxiv_id] = text

    n = rag.ingest(text, source=f"arxiv:{arxiv_id}")
    session.add_paper(arxiv_id)
    console.print(f"[green]✓ arxiv:{arxiv_id}[/green] — {n} chunks indexed")

    # ── Retrieve & compress ──────────────────────────────────────────────── #
    with console.status("Retrieving relevant sections …"):
        result = rag.query(question, top_k=8)  # more chunks for paper-specific questions

    ctx = result["context"]

    if result["chunks"]:
        table = Table(title="Relevant Sections")
        table.add_column("Score", justify="right")
        table.add_column("Preview", max_width=80)
        for c in result["chunks"]:
            table.add_row(f"{c['score']:.3f}", c["text"][:100] + "…")
        console.print(table)

    if result.get("compressed"):
        cstats = result["compression_stats"]
        console.print(
            f"[dim]Compressed {cstats['original_tokens']} → "
            f"{cstats['compressed_tokens']} tokens "
            f"({cstats['latency_ms']}ms)[/dim]\n"
        )

    if not ctx:
        console.print("[red]No relevant sections found in this paper.[/red]")
        return session

    # ── Include conversation history ─────────────────────────────────────── #
    history_ctx = session.get_history_context(max_turns=3)
    if history_ctx:
        ctx = f"{history_ctx}\n\n---\n\n{ctx}"

    # ── Run full anti-hallucination workflow (COT → verify) ──────────────── #
    #    Paper context is injected with strict grounding instructions so
    #    the model ONLY uses paper content, and verify checks every claim.
    paper_ctx = (
        "IMPORTANT: You are analysing a SPECIFIC research paper. "
        "ONLY use information from the paper excerpts below. "
        "Do NOT add information from your training data. "
        "If the paper does not mention something, say so. "
        "Cite specific sections, equations, figures, or tables.\n\n"
        f"## Paper Excerpts\n{ctx}"
    )

    llm_fn = _get_llm_fn()
    workflow = build_workflow(paper_ctx, llm_fn=llm_fn)
    # Always enable verify for paper analysis (anti-hallucination)
    workflow.toggle_stage("self_verify", enabled=True)

    with console.status("Running anti-hallucination pipeline (COT → verify) …"):
        artifacts = workflow.run(question, prompt_context=question)

    # Display results
    final_answer = ""
    for stage_name, meta in artifacts.items():
        content = Path(meta["storage"]).read_text(encoding="utf-8") if meta.get("storage") else ""
        subtitle_parts = []
        if meta.get("stage_latency_ms"):
            subtitle_parts.append(f"{meta['stage_latency_ms'] / 1000:.1f}s")

        if stage_name == "cot":
            final_answer = content
            console.print(Panel(
                Markdown(content),
                title="[bold]PAPER ANALYSIS[/bold]",
                subtitle=" | ".join(subtitle_parts) if subtitle_parts else None,
            ))
        elif stage_name == "self_verify":
            # Show compact summary instead of full table
            summary, has_issues = _summarize_verification(content)
            latency = f" ({meta['stage_latency_ms'] / 1000:.1f}s)" if meta.get("stage_latency_ms") else ""
            console.print(f"\n{summary}{latency}")
            if has_issues:
                console.print(Panel(
                    Markdown(content),
                    title="[bold yellow]CITATION DETAILS[/bold yellow]",
                    border_style="yellow",
                ))
        elif stage_name == "self_critique":
            console.print(Panel(
                Markdown(content),
                title="[bold]CRITIQUE[/bold]",
                subtitle=" | ".join(subtitle_parts) if subtitle_parts else None,
                border_style="dim",
            ))

    t_total = time.perf_counter() - t_start
    console.print(f"[dim]Total latency: {t_total:.1f}s[/dim]")

    session.add_turn(
        question=f"[paper:{arxiv_id}] {question}",
        answer=final_answer,
        sources=[f"arxiv:{arxiv_id}"],
        metadata={"command": "paper", "arxiv_id": arxiv_id,
                  "artifacts": {k: v.get("id") for k, v in artifacts.items()}},
    )
    console.print(f"\n[dim]Session {session.session_id} — turn {session.turn_count} saved[/dim]")
    return session


def _summarize_verification(content: str) -> tuple[str, bool]:
    """
    Parse verification markdown table and return a compact summary string
    plus a boolean indicating whether any issues were found.
    """
    supported = content.lower().count("supported")
    not_found = content.lower().count("not found")
    partial = content.lower().count("partially")
    # Deduct partial from supported since "partially supported" also contains "supported"
    supported = supported - partial - not_found  # "not found in sources" doesn't contain it
    total = supported + partial + not_found
    if total == 0:
        return "[dim]No claims to verify[/dim]", False

    has_issues = (not_found > 0 or partial > 0)
    parts = []
    if supported > 0:
        parts.append(f"[green]✓ {supported} supported[/green]")
    if partial > 0:
        parts.append(f"[yellow]~ {partial} partially supported[/yellow]")
    if not_found > 0:
        parts.append(f"[red]✗ {not_found} not found in sources[/red]")
    summary = f"Citation check ({total} claims): " + " | ".join(parts)
    return summary, has_issues


# ── Entry point ─────────────────────────────────────────────────────────────── #

def main() -> None:
    if len(sys.argv) < 2:
        console.print(__doc__)
        return

    cmd = sys.argv[1].lower()

    if cmd == "ingest":
        cmd_ingest()
    elif cmd == "papers":
        if len(sys.argv) < 3:
            console.print("[red]Provide a search query: python -m src.main papers \"your query\"[/red]")
            return
        # parse optional --session
        args = sys.argv[2:]
        session_id = None
        query_parts = []
        i = 0
        while i < len(args):
            if args[i] == "--session" and i + 1 < len(args):
                session_id = args[i + 1]
                i += 2
            else:
                query_parts.append(args[i])
                i += 1
        cmd_papers(" ".join(query_parts), session_id=session_id)
    elif cmd == "search":
        if len(sys.argv) < 3:
            console.print("[red]Provide a query: python -m src.main search \"your query\"[/red]")
            return
        cmd_search(" ".join(sys.argv[2:]))
    elif cmd == "paper":
        # python -m src.main paper <arxiv_id> "question" [--session ID]
        args = sys.argv[2:]
        if len(args) < 2:
            console.print("[red]Usage: python -m src.main paper <arxiv_id> \"question about paper\"[/red]")
            return
        paper_id = args[0]
        session_id = None
        question_parts = []
        i = 1
        while i < len(args):
            if args[i] == "--session" and i + 1 < len(args):
                session_id = args[i + 1]
                i += 2
            else:
                question_parts.append(args[i])
                i += 1
        cmd_paper(paper_id, " ".join(question_parts), session_id=session_id)
    elif cmd == "ask":
        # Parse --session flag
        args = sys.argv[2:]
        session_id = None
        question_parts = []
        i = 0
        while i < len(args):
            if args[i] == "--session" and i + 1 < len(args):
                session_id = args[i + 1]
                i += 2
            else:
                question_parts.append(args[i])
                i += 1
        if not question_parts:
            console.print("[red]Provide a question: python -m src.main ask \"your question\"[/red]")
            return
        cmd_ask(" ".join(question_parts), session_id=session_id)
    elif cmd == "sessions":
        cmd_sessions()
    elif cmd == "workflow":
        cmd_workflow(sys.argv[2:])
    elif cmd == "artifacts":
        cmd_artifacts(sys.argv[1:])
    else:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        console.print(__doc__)


if __name__ == "__main__":
    main()

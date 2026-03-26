import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

import asyncio
import atexit
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
)
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

load_dotenv()
console = Console()

# ── Models ────────────────────────────────────────────────────────────────────

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
model = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
    num_predict=1024,
    reasoning=True,
)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE = Path(__file__).parent.parent
CHROMA_PATH = str(BASE / "chroma_db")
GRAPH_INDEXED_FLAG = BASE / ".graph_indexed"
GRAPH_PROGRESS_FILE = BASE / ".graph_progress.json"

folders = [
    BASE / "langchain_md",
    BASE / "langgraph_md",
    BASE / "learn_md",
    BASE / "lang_integration_componet",
]

# ── Vector store ──────────────────────────────────────────────────────────────

if not Path(CHROMA_PATH).exists():
    console.print("[dim]Building vector knowledge base (first run)...[/dim]")
    docs = []
    for folder in folders:
        for path in Path(folder).rglob("*.md"):
            docs.append(Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": str(path)}
            ))
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vector_store = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )
    console.print(f"[dim]Indexed {len(chunks)} chunks → saved to disk[/dim]")
else:
    console.print("[dim]Loading vector knowledge base from disk...[/dim]")
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    console.print("[dim]Ready[/dim]")

# ── Knowledge graph (Graphiti + Neo4j) ───────────────────────────────────────

GRAPH_SEARCH_CONFIG = COMBINED_HYBRID_SEARCH_CROSS_ENCODER.model_copy(update={
                                                                      "limit": 5})
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Langgraph")


class GraphitiRuntime:
    """Own Graphiti and its async Neo4j driver on one dedicated event loop."""

    def __init__(self, uri: str, user: str, password: str):
        self._uri = uri
        self._user = user
        self._password = password
        self._loop: asyncio.AbstractEventLoop | None = None
        self._graphiti: Graphiti | None = None
        self._startup_error: BaseException | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="graphiti-runtime",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError(
                "Failed to initialize Graphiti runtime") from self._startup_error

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        try:
            self._graphiti = Graphiti(
                uri=self._uri,
                user=self._user,
                password=self._password,
            )
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            loop.close()
            return

        self._ready.set()

        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(
                    *pending, return_exceptions=True))
            loop.close()

    def run(self, coro):
        if self._loop is None:
            raise RuntimeError("Graphiti runtime loop is not available")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def init_graph(self) -> None:
        graphiti = self._require_graphiti()
        await graphiti.build_indices_and_constraints()

        if GRAPH_INDEXED_FLAG.exists():
            return

        # Load already-ingested files — first check Neo4j for existing episodes,
        # then fall back to the local progress file.
        done: set[str] = set()
        if GRAPH_PROGRESS_FILE.exists():
            done = set(json.loads(GRAPH_PROGRESS_FILE.read_text()))
        else:
            records, _, _ = await graphiti.driver.execute_query(
                "MATCH (e:Episodic) RETURN e.source_description AS src",
            )
            ingested = {r["src"] for r in records if r["src"]}
            all_paths_set = {
                p: str(p.relative_to(BASE))
                for folder in folders
                for p in Path(folder).rglob("*.md")
            }
            done = {str(p)
                    for p, rel in all_paths_set.items() if rel in ingested}
            if done:
                GRAPH_PROGRESS_FILE.write_text(json.dumps(list(done)))
                console.print(
                    f"[dim]Recovered {len(done)} already-ingested episodes from Neo4j[/dim]"
                )

        all_paths = [p for folder in folders for p in Path(
            folder).rglob("*.md")]
        remaining = [p for p in all_paths if str(p) not in done]

        if done:
            console.print(
                f"[dim]Resuming knowledge graph ({len(done)}/{len(all_paths)} already done)...[/dim]"
            )
        else:
            console.print(
                "[dim]Building knowledge graph (first run — this takes a few minutes)...[/dim]"
            )

        for path in remaining:
            for attempt in range(5):
                try:
                    await graphiti.add_episode(
                        name=path.stem,
                        episode_body=path.read_text(encoding="utf-8"),
                        source_description=str(path.relative_to(BASE)),
                        reference_time=datetime.now(timezone.utc),
                        source=EpisodeType.text,
                        group_id="langchain_docs",
                    )
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    wait = 60 * (attempt + 1)
                    console.print(
                        f"  [dim yellow]Rate limit hit, retrying in {wait}s...[/dim yellow]"
                    )
                    await asyncio.sleep(wait)
            done.add(str(path))
            GRAPH_PROGRESS_FILE.write_text(json.dumps(list(done)))
            await asyncio.sleep(30)

        GRAPH_INDEXED_FLAG.touch()
        console.print(
            f"[dim]Knowledge graph complete — {len(all_paths)} docs[/dim]")

    async def search(self, query: str) -> str:
        graphiti = self._require_graphiti()
        results = await graphiti.search_(
            query=query,
            config=GRAPH_SEARCH_CONFIG,
            group_ids=["langchain_docs"],
        )
        parts = []
        for edge in results.edges:
            parts.append(f"[fact] {edge.fact}")
        for node in results.nodes:
            parts.append(
                f"[entity] {node.name}: {getattr(node, 'summary', '')}")
        return "\n\n".join(parts) if parts else "No relevant graph facts found."

    def close(self) -> None:
        if self._loop is None or not self._thread.is_alive():
            return
        if self._graphiti is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._graphiti.close(), self._loop).result(timeout=5)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    def _require_graphiti(self) -> Graphiti:
        if self._graphiti is None:
            raise RuntimeError("Graphiti runtime is not initialized")
        return self._graphiti


graph_runtime = GraphitiRuntime(
    uri=NEO4J_URI,
    user=NEO4J_USER,
    password=NEO4J_PASSWORD,
)
atexit.register(graph_runtime.close)
graph_runtime.run(graph_runtime.init_graph())


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def search_langchain_docs(query: str) -> str:
    """Search LangChain/LangGraph documentation chunks for syntax, APIs, and how-to guides."""
    results = vector_store.similarity_search(query, k=3)
    return "\n\n---\n\n".join(
        f"[{r.metadata['source']}]\n{r.page_content}"
        for r in results
    )


@tool
def search_knowledge_graph(query: str) -> str:
    """Search the knowledge graph for entity relationships, concepts, and cross-topic facts."""
    return graph_runtime.run(graph_runtime.search(query))


# ── Agent ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a LangChain/LangGraph expert with two retrieval tools:\n"
    "- search_langchain_docs: exact documentation chunks, good for APIs and syntax\n"
    "- search_knowledge_graph: entity/relationship graph, good for concepts and cross-topic questions\n"
    "Always use at least one tool before answering. Never answer from memory alone."
)

agent = create_agent(
    model,
    tools=[search_langchain_docs, search_knowledge_graph],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
    middleware=[SummarizationMiddleware(
        model=model,
        trigger=("tokens", 4000),
        keep=("messages", 10),
    )]
)

# ── Streaming ─────────────────────────────────────────────────────────────────


def stream_response(user_message: str, config: dict) -> None:
    in_reasoning_block = False
    in_think_tag = False
    thinking_parts: list[str] = []
    final_answer_parts: list[str] = []
    last_thinking_flush = monotonic()
    last_thinking_char = ""
    last_answer_char = ""

    def normalized_chunk(last_char: str, chunk: str) -> str:
        if not last_char or not chunk:
            return chunk
        if last_char.isalnum() and chunk[:1].isspace():
            stripped = chunk.lstrip()
            if stripped and stripped[:1].islower():
                return " " + stripped
        return chunk

    def normalize_reasoning_text(text: str) -> str:
        # Ollama reasoning chunks sometimes contain single newlines in the middle
        # of a sentence ("The\\n user"). Collapse those while preserving blank lines.
        return re.sub(r"(?<!\n)\n(?!\n)\s*", " ", text)

    def flush_thinking(force: bool = False) -> None:
        nonlocal last_thinking_flush
        if not thinking_parts:
            return
        now = monotonic()
        pending = "".join(thinking_parts)
        if not force and len(pending) < 24 and now - last_thinking_flush < 0.06:
            return
        console.print(pending, end="", highlight=False,
                      style="dim italic yellow")
        thinking_parts.clear()
        last_thinking_flush = now

    def close_thinking_block() -> None:
        nonlocal in_reasoning_block, in_think_tag
        if in_reasoning_block or in_think_tag:
            flush_thinking(force=True)
            console.print()
            in_reasoning_block = False
            in_think_tag = False

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        stream_mode=["updates", "messages"],
        version="v2",
        config=config,
    ):
        if chunk["type"] == "updates":
            for step, data in chunk["data"].items():
                if not data or "messages" not in data:
                    continue
                last_message = data["messages"][-1]
                if step == "model":
                    tool_calls = list(
                        getattr(last_message, "tool_calls", None) or [])
                    if not tool_calls:
                        for block in getattr(last_message, "content_blocks", []) or []:
                            if isinstance(block, dict) and block.get("type") == "tool_call":
                                tool_calls.append({
                                    "name": block.get("name"),
                                    "args": block.get("args"),
                                })
                    if tool_calls:
                        close_thinking_block()
                        for call in tool_calls:
                            tool_name = call.get("name") or "tool"
                            tool_args = call.get("args")
                            if tool_args:
                                console.print(
                                    f"  [dim cyan]→ {tool_name}[/] {tool_args}")
                            else:
                                console.print(f"  [dim cyan]→ {tool_name}[/]")
                if step == "tools":
                    close_thinking_block()
                    tool_name = last_message.name
                    label = "[vector]" if tool_name == "search_langchain_docs" else "[graph]"
                    console.print(f"  [dim green]✓ {tool_name} {label}[/]")

        elif chunk["type"] == "messages":
            data = chunk.get("data")
            if not data:
                continue
            token, metadata = data
            if metadata.get("langgraph_node") != "model":
                continue

            reasoning_text = ""
            additional_kwargs = getattr(token, "additional_kwargs", {}) or {}
            if isinstance(additional_kwargs, dict):
                reasoning_text = additional_kwargs.get(
                    "reasoning_content", "") or ""
                reasoning_text = normalize_reasoning_text(reasoning_text)

            if reasoning_text:
                if not in_reasoning_block:
                    console.print("\n[dim]thinking:[/dim]", end=" ")
                    in_reasoning_block = True
                normalized = normalized_chunk(
                    last_thinking_char, reasoning_text)
                thinking_parts.append(normalized)
                if normalized:
                    last_thinking_char = normalized[-1]
                flush_thinking()

            text = getattr(token, "text", None)
            if text is None:
                text = getattr(token, "content", None)
            text = text or ""
            if not text and not reasoning_text:
                continue
            if text and in_reasoning_block:
                close_thinking_block()

            # Fast path for normal answer chunks with no inline think tags.
            if "<think>" not in text and "</think>" not in text and not in_think_tag:
                normalized = normalized_chunk(last_answer_char, text)
                final_answer_parts.append(normalized)
                if normalized:
                    last_answer_char = normalized[-1]
                continue

            # Fallback parser for models that emit inline think tags.
            i = 0
            plain_text: list[str] = []
            while i < len(text):
                if not in_think_tag and text[i:].startswith("<think>"):
                    if plain_text:
                        joined = "".join(plain_text)
                        joined = normalized_chunk(last_answer_char, joined)
                        final_answer_parts.append(joined)
                        if joined:
                            last_answer_char = joined[-1]
                        plain_text.clear()
                    in_think_tag = True
                    console.print("\n[dim]thinking:[/dim]", end=" ")
                    i += len("<think>")
                elif in_think_tag and text[i:].startswith("</think>"):
                    flush_thinking(force=True)
                    in_think_tag = False
                    console.print()  # newline after thinking block
                    i += len("</think>")
                elif in_think_tag:
                    normalized = normalized_chunk(last_thinking_char, text[i])
                    thinking_parts.append(normalized)
                    if normalized:
                        last_thinking_char = normalized[-1]
                    flush_thinking()
                    i += 1
                else:
                    plain_text.append(text[i])
                    i += 1
            if plain_text:
                joined = "".join(plain_text)
                joined = normalized_chunk(last_answer_char, joined)
                final_answer_parts.append(joined)
                if joined:
                    last_answer_char = joined[-1]

    close_thinking_block()
    if final_answer_parts:
        console.print()
        console.print("[dim]Assistant:[/dim]")
        console.print(Markdown("".join(final_answer_parts)))
        console.print()


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    try:
        console.print(Panel(
            f"[bold cyan]Hybrid RAG Chat[/bold cyan]\n"
            f"[dim]Ollama ({OLLAMA_MODEL}) · vector + knowledge graph[/dim]",
            subtitle="[dim]type 'exit' to quit[/dim]",
            border_style="cyan",
        ))
        console.print()

        while True:
            user_input = console.input("[bold green]You:[/] ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "bye"):
                console.print("\n[dim]Goodbye![/dim]")
                break

            stream_response(user_input, config)
    finally:
        graph_runtime.close()


if __name__ == "__main__":
    main()

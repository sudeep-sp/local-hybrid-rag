import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

import asyncio
import atexit
import json
import os
import re
import threading
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

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
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

load_dotenv()
console = Console()

# ── Models ────────────────────────────────────────────────────────────────────

def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


CHAT_BACKEND = os.getenv("CHAT_BACKEND", "ollama").strip().lower()
CHAT_MODEL = os.getenv("CHAT_MODEL") or os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0"))
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "1024"))
CHAT_OPENAI_BASE_URL = os.getenv("CHAT_OPENAI_BASE_URL") or os.getenv(
    "OPENAI_API_BASE"
)
CHAT_OPENAI_API_KEY = os.getenv("CHAT_OPENAI_API_KEY") or os.getenv(
    "OPENAI_API_KEY"
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL") or os.getenv(
    "OPENAI_API_BASE"
)
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
EMBEDDING_CHECK_CTX_LENGTH = env_flag(
    "EMBEDDING_CHECK_CTX_LENGTH",
    default=EMBEDDING_BASE_URL is None,
)

if CHAT_OPENAI_BASE_URL and not CHAT_OPENAI_API_KEY:
    CHAT_OPENAI_API_KEY = "omlx"
if EMBEDDING_BASE_URL and not EMBEDDING_API_KEY:
    EMBEDDING_API_KEY = "omlx"


class ReasoningChatOpenAI(ChatOpenAI):
    """Preserve reasoning deltas from OpenAI-compatible backends like oMLX."""

    @staticmethod
    def _coerce_reasoning_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            for key in (
                "reasoning_content",
                "reasoning",
                "thinking",
                "text",
                "content",
            ):
                text = ReasoningChatOpenAI._coerce_reasoning_text(value.get(key))
                if text:
                    return text
            return ""
        if isinstance(value, list):
            parts = [
                ReasoningChatOpenAI._coerce_reasoning_text(item) for item in value
            ]
            return "".join(part for part in parts if part)
        return ""

    @classmethod
    def _extract_reasoning_text(cls, payload: Any) -> str:
        if not isinstance(payload, Mapping):
            return ""

        for key in ("reasoning_content", "reasoning", "thinking"):
            text = cls._coerce_reasoning_text(payload.get(key))
            if text:
                return text

        content = payload.get("content")
        if not isinstance(content, list):
            return ""

        parts: list[str] = []
        for block in content:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") not in {"reasoning", "reasoning_content", "thinking"}:
                continue
            text = cls._coerce_reasoning_text(block)
            if text:
                parts.append(text)
        return "".join(parts)

    @classmethod
    def _extract_reasoning_from_chunk(cls, chunk: Mapping[str, Any]) -> str:
        choices = chunk.get("choices")
        if not choices:
            nested_chunk = chunk.get("chunk")
            if isinstance(nested_chunk, Mapping):
                choices = nested_chunk.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return ""

        choice = choices[0]
        if not isinstance(choice, Mapping):
            return ""

        return cls._extract_reasoning_text(choice.get("delta")) or cls._extract_reasoning_text(
            choice.get("message")
        )

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if generation_chunk is None:
            return None

        reasoning_text = self._extract_reasoning_from_chunk(chunk)
        if reasoning_text and isinstance(generation_chunk.message, AIMessageChunk):
            generation_chunk.message.additional_kwargs = {
                **(generation_chunk.message.additional_kwargs or {}),
                "reasoning_content": (
                    f"{generation_chunk.message.additional_kwargs.get('reasoning_content', '')}"
                    f"{reasoning_text}"
                ),
            }

        return generation_chunk

    def _create_chat_result(
        self,
        response: dict | Any,
        generation_info: dict | None = None,
    ) -> ChatResult:
        chat_result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response if isinstance(response, dict) else response.model_dump()
        )
        for generation, choice in zip(chat_result.generations, response_dict.get("choices", [])):
            if not isinstance(generation.message, AIMessage):
                continue
            reasoning_text = self._extract_reasoning_text(choice.get("message"))
            if reasoning_text:
                generation.message.additional_kwargs["reasoning_content"] = reasoning_text
        return chat_result


def build_chat_model():
    if CHAT_BACKEND == "ollama":
        return ChatOllama(
            model=CHAT_MODEL,
            temperature=CHAT_TEMPERATURE,
            num_predict=CHAT_MAX_TOKENS,
            reasoning=True,
        )

    if CHAT_BACKEND == "openai":
        return ReasoningChatOpenAI(
            model=CHAT_MODEL,
            temperature=CHAT_TEMPERATURE,
            max_completion_tokens=CHAT_MAX_TOKENS,
            base_url=CHAT_OPENAI_BASE_URL or "http://localhost:8000/v1",
            api_key=CHAT_OPENAI_API_KEY or "omlx",
        )

    raise ValueError(
        f"Unsupported CHAT_BACKEND={CHAT_BACKEND!r}. Use 'ollama' or 'openai'."
    )


def build_embeddings():
    kwargs = {
        "model": EMBEDDING_MODEL,
        "check_embedding_ctx_length": EMBEDDING_CHECK_CTX_LENGTH,
    }
    if EMBEDDING_BASE_URL:
        kwargs["base_url"] = EMBEDDING_BASE_URL
    if EMBEDDING_API_KEY:
        kwargs["api_key"] = EMBEDDING_API_KEY
    return OpenAIEmbeddings(**kwargs)


model = build_chat_model()
embeddings = build_embeddings()
CHAT_RUNTIME_LABEL = (
    f"oMLX/OpenAI-compatible ({CHAT_MODEL})"
    if CHAT_BACKEND == "openai"
    else f"Ollama ({CHAT_MODEL})"
)

MAX_DOC_SEARCHES_PER_TURN = int(os.getenv("MAX_DOC_SEARCHES_PER_TURN", "6"))
MAX_GRAPH_SEARCHES_PER_TURN = int(os.getenv("MAX_GRAPH_SEARCHES_PER_TURN", "4"))
_tool_state_lock = threading.Lock()
_tool_state = {
    "doc_queries": [],
    "doc_cache": {},
    "doc_calls": 0,
    "graph_queries": [],
    "graph_cache": {},
    "graph_calls": 0,
}


def reset_tool_state() -> None:
    with _tool_state_lock:
        _tool_state["doc_queries"] = []
        _tool_state["doc_cache"] = {}
        _tool_state["doc_calls"] = 0
        _tool_state["graph_queries"] = []
        _tool_state["graph_cache"] = {}
        _tool_state["graph_calls"] = 0


def normalize_query(query: str) -> str:
    query = query.lower()
    query = re.sub(r"[_\-:/()]+", " ", query)
    query = re.sub(r"[^a-z0-9\s]+", "", query)
    return " ".join(query.split())


def queries_are_similar(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))
    return overlap >= 0.8

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE = Path(__file__).parent.parent
CHROMA_PATH = str(Path(os.getenv("CHROMA_PATH", str(BASE / "chroma_db"))).expanduser())
GRAPH_INDEXED_FLAG = BASE / ".graph_indexed"
GRAPH_PROGRESS_FILE = BASE / ".graph_progress.json"

folders = [
    BASE / "langchain_md",
    BASE / "langgraph_md",
    BASE / "learn_md",
    BASE / "lang_integration_componet",
]

# ── Vector store ──────────────────────────────────────────────────────────────

if Path(CHROMA_PATH).exists() and (
    EMBEDDING_BASE_URL or EMBEDDING_MODEL != "text-embedding-3-small"
):
    console.print(
        "[dim yellow]Custom embeddings are enabled. If this Chroma DB was built "
        "with a different embedding model/backend, point CHROMA_PATH at a fresh "
        "directory or rebuild the index.[/dim yellow]"
    )

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
    normalized = normalize_query(query)
    with _tool_state_lock:
        for prev_query, cached in _tool_state["doc_cache"].items():
            if queries_are_similar(normalized, prev_query):
                return cached
        if _tool_state["doc_calls"] >= MAX_DOC_SEARCHES_PER_TURN:
            prior = ", ".join(_tool_state["doc_queries"]) or "none"
            return (
                "Search budget reached for this answer. Reuse the documentation already "
                f"retrieved instead of issuing more searches. Prior doc queries: {prior}"
            )
        _tool_state["doc_calls"] += 1
        _tool_state["doc_queries"].append(query)

    results = vector_store.similarity_search(query, k=3)
    output = "\n\n---\n\n".join(
        f"[{r.metadata['source']}]\n{r.page_content}"
        for r in results
    )
    with _tool_state_lock:
        _tool_state["doc_cache"][normalized] = output
    return output


@tool
def search_knowledge_graph(query: str) -> str:
    """Search the knowledge graph for entity relationships, concepts, and cross-topic facts."""
    normalized = normalize_query(query)
    with _tool_state_lock:
        for prev_query, cached in _tool_state["graph_cache"].items():
            if queries_are_similar(normalized, prev_query):
                return cached
        if _tool_state["graph_calls"] >= MAX_GRAPH_SEARCHES_PER_TURN:
            prior = ", ".join(_tool_state["graph_queries"]) or "none"
            return (
                "Knowledge-graph search budget reached for this answer. Reuse the graph "
                f"facts already retrieved instead of issuing more searches. Prior graph queries: {prior}"
            )
        _tool_state["graph_calls"] += 1
        _tool_state["graph_queries"].append(query)

    output = graph_runtime.run(graph_runtime.search(query))
    with _tool_state_lock:
        _tool_state["graph_cache"][normalized] = output
    return output


# ── Agent ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a LangChain/LangGraph expert assistant. You have two retrieval tools:\n"
    "- search_langchain_docs: searches exact documentation chunks — use this for import paths, "
    "API signatures, class names, method names, and code syntax\n"
    "- search_knowledge_graph: searches an entity/relationship graph — use this for concepts, "
    "architecture patterns, and cross-topic questions\n\n"
    "RULES:\n"
    "1. Always search before answering. Never write code from memory.\n"
    "2. For any code example, search for the exact import paths and class names first. "
    "Use only what the retrieved docs show — do not guess or fill in from memory.\n"
    "3. If the retrieved docs don't contain enough detail, say so explicitly rather than "
    "inventing plausible-looking code.\n"
    "4. Use at most 3 documentation searches and 2 graph searches per answer unless a tool says "
    "there were no relevant results.\n"
    "5. Never repeat the same or near-duplicate search query. Reuse prior tool results.\n"
    "6. For code, one search for imports/API names and one search for a usage example is usually enough.\n"
    "7. Base your answer strictly on retrieved content. If something contradicts what you "
    "know from training, trust the retrieved docs."
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
    reset_tool_state()
    in_reasoning_block = False
    in_think_tag = False
    thinking_parts: list[str] = []
    history_renderables: list[Any] = []
    final_answer_parts: list[str] = []
    last_thinking_flush = monotonic()
    last_thinking_char = ""
    last_answer_char = ""
    live = Live(console=console, refresh_per_second=15, auto_refresh=False)

    def normalized_chunk(last_char: str, chunk: str) -> str:
        if not last_char or not chunk:
            return chunk
        if last_char.isalnum() and chunk[:1].isspace():
            stripped = chunk.lstrip()
            if stripped and stripped[:1].islower():
                return " " + stripped
        return chunk

    def normalize_reasoning_text(text: str) -> str:
        # Some backends stream reasoning with single newlines mid-sentence
        # ("The\\n user"). Collapse those while preserving blank lines.
        return re.sub(r"(?<!\n)\n(?!\n)\s*", " ", text)

    def build_renderable() -> Group:
        items: list[Any] = list(history_renderables)
        if in_reasoning_block and thinking_parts:
            line = Text()
            line.append("thinking: ", style="dim")
            line.append("".join(thinking_parts), style="dim italic yellow")
            items.append(line)
        if final_answer_parts:
            if items:
                items.append(Text(""))
            items.append(Text("Assistant:", style="dim"))
            items.append(Markdown("".join(final_answer_parts)))
        if not items:
            items.append(Text(""))
        return Group(*items)

    def refresh_live() -> None:
        live.update(build_renderable(), refresh=True)

    def emit_answer(text: str) -> None:
        nonlocal last_answer_char
        if not text:
            return
        normalized = normalized_chunk(last_answer_char, text)
        final_answer_parts.append(normalized)
        if normalized:
            last_answer_char = normalized[-1]
        refresh_live()

    def flush_thinking(force: bool = False) -> None:
        nonlocal last_thinking_flush
        if not thinking_parts:
            return
        now = monotonic()
        pending = "".join(thinking_parts)
        if not force and len(pending) < 24 and now - last_thinking_flush < 0.06:
            return
        refresh_live()
        last_thinking_flush = now

    def close_thinking_block() -> None:
        nonlocal in_reasoning_block, in_think_tag
        if in_reasoning_block or in_think_tag:
            flush_thinking(force=True)
            if thinking_parts:
                line = Text()
                line.append("thinking: ", style="dim")
                line.append("".join(thinking_parts), style="dim italic yellow")
                history_renderables.append(line)
                thinking_parts.clear()
            in_reasoning_block = False
            in_think_tag = False
            refresh_live()

    with live:
        refresh_live()
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
                                    history_renderables.append(
                                        Text(f"  -> {tool_name} {tool_args}", style="dim cyan")
                                    )
                                else:
                                    history_renderables.append(
                                        Text(f"  -> {tool_name}", style="dim cyan")
                                    )
                            refresh_live()
                    if step == "tools":
                        close_thinking_block()
                        tool_name = last_message.name
                        label = "[vector]" if tool_name == "search_langchain_docs" else "[graph]"
                        history_renderables.append(
                            Text(f"  ok {tool_name} {label}", style="dim green")
                        )
                        refresh_live()

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
                    if text:
                        emit_answer(text)
                    continue

                # Fallback parser for models that emit inline think tags.
                i = 0
                plain_text: list[str] = []
                while i < len(text):
                    if not in_think_tag and text[i:].startswith("<think>"):
                        if plain_text:
                            emit_answer("".join(plain_text))
                            plain_text.clear()
                        in_think_tag = True
                        i += len("<think>")
                    elif in_think_tag and text[i:].startswith("</think>"):
                        close_thinking_block()
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
                    emit_answer("".join(plain_text))

        close_thinking_block()


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    try:
        console.print(Panel(
            f"[bold cyan]Hybrid RAG Chat[/bold cyan]\n"
            f"[dim]{CHAT_RUNTIME_LABEL} · vector + knowledge graph[/dim]",
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

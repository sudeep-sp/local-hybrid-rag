from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from dotenv import load_dotenv
import uuid
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
console = Console()

model = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
embeddings = OpenAIEmbeddings()

BASE = Path(__file__).parent.parent  # lang-tut/

CHROMA_PATH = str(BASE / "projects" / "chroma_db")

folders = [
    BASE / "langchain_md",
    BASE / "langgraph_md",
    BASE / "learn_md",
    BASE / "lang_integration_componet",
]


if not Path(CHROMA_PATH).exists():
    console.print("[dim]Building knowledge base for the first time...[/dim]")
    docs = []
    for folder in folders:
        for path in Path(folder).rglob("*.md"):
            docs.append(Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": str(path)}
            ))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vector_store = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    console.print(f"[dim]Indexed {len(chunks)} chunks — saved to disk[/dim]")
else:
    # Every run after: just load from disk, no re-embedding
    console.print("[dim]Loading knowledge base from disk...[/dim]")
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    console.print("[dim]Ready[/dim]")


@tool
def search_langchain_docs(query: str) -> str:
    """Search LangChain and LangGraph documentation and notes."""
    results = vector_store.similarity_search(query, k=3)
    return "\n\n---\n\n".join(
        f"[{r.metadata['source']}]\n{r.page_content}"
        for r in results
    )


agent = create_agent(
    model,
    tools=[search_langchain_docs],
    system_prompt="You are a LangChain/LangGraph expert. Always use search_langchain_docs to answer questions accurately from the documentation.",
    checkpointer=InMemorySaver(),
    middleware=[SummarizationMiddleware(
        model=model,
        trigger=("tokens", 100),
        keep=("messages", 10)
    )]
)


def stream_response(user_message: str, config: dict) -> None:
    response_tokens = []

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        stream_mode=["updates", "messages"],
        version="v2",
        config=config
    ):
        if chunk["type"] == "updates":
            for step, data in chunk["data"].items():
                if not data or "messages" not in data:
                    continue
                if step == "tools":
                    tool_name = data["messages"][-1].name
                    console.print(f"  [dim green]✓ {tool_name} done[/]")

        elif chunk["type"] == "messages":
            data = chunk.get("data")
            if not data:
                continue
            token, metadata = data
            # Only collect AI response tokens, skip tool result tokens
            if metadata.get("langgraph_node") == "model" and hasattr(token, "text") and token.text:
                response_tokens.append(token.text)

    if response_tokens:
        full_response = "".join(response_tokens)
        console.print()
        console.print(Markdown(full_response))
        console.print()


def main():
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    console.print(Panel(
        "[bold cyan]CLI Chat[/bold cyan]\n[dim]Powered by GPT + Langchain knowledge base[/dim]",
        subtitle="[dim]type 'exit' to quit[/dim]",
        border_style="cyan"
    ))
    console.print()

    while True:
        user_input = console.input("[bold green]You:[/] ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye"):
            console.print("\n[dim]Goodbye![/dim]")
            break

        console.print("[dim]Assistant:[/dim]")
        stream_response(user_input, config)


if __name__ == "__main__":
    main()

import argparse
import importlib
import json
import chromadb

from pathlib import Path
from uuid import uuid4

from docquery.embeddings.provider import get_embeddings
from langchain_unstructured import UnstructuredLoader
from langchain_chroma.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--db", default=None, help="Path to SQLite database (overrides DB_PATH env var)")


def _add_inference_args(parser: argparse.ArgumentParser) -> None:
    """Args that control retrieval and LLM behaviour for chat/query commands."""
    parser.add_argument("--top-k", type=int, default=None, metavar="K",
                        help="Number of similarity results to retrieve (overrides TOP_K env var)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="LLM sampling temperature 0–2 (overrides TEMPERATURE env var, default 0)")


def _add_prompt_args(parser: argparse.ArgumentParser) -> None:
    """--system-prompt and --preset for chat/query subcommands."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--system-prompt", default=None, metavar="TEXT",
                       help="Override the default system prompt with TEXT")
    group.add_argument("--preset", default=None, metavar="NAME",
                       help="Load a named system prompt from examples/queries.json")


def _resolve_system_prompt(args: argparse.Namespace) -> str | None:
    preset = getattr(args, "preset", None)
    if preset:
        presets_file = Path("examples/queries.json")
        if not presets_file.exists():
            print(f"Error: presets file not found at {presets_file.resolve()}\n"
                  f"Run docquery from the project root, or check that examples/queries.json exists.")
            raise SystemExit(1)
        data = json.loads(presets_file.read_text())
        if preset not in data:
            available = ", ".join(data.keys())
            print(f"Error: preset {preset!r} not found. Available: {available}")
            raise SystemExit(1)
        return data[preset]["system_prompt"]
    return getattr(args, "system_prompt", None)


def _apply_overrides(args: argparse.Namespace, settings) -> None:
    if args.db:
        settings.db_path = args.db
        settings.db_client = chromadb.PersistentClient(path=settings.db_path)
    else:
        # in-memory ChromaDB
        settings.db_path = None
        settings.db_client = chromadb.Client()

    if getattr(args, "top_k", None) is not None:
        settings.top_k = args.top_k
    if getattr(args, "temperature", None) is not None:
        settings.temperature = args.temperature
    
    embeddings = get_embeddings(settings)

    settings.vs = Chroma(
            client=settings.db_client,
            collection_name="db_knowledge",
            embedding_function=embeddings,
            )

def cmd_ingest(args: argparse.Namespace) -> None:
    from docquery.config import Settings

    settings = Settings()
    _apply_overrides(args, settings)

    docs = []

    loader = UnstructuredLoader(args.files,
                                chunking_strategy="by_title",
                                # chroma's chunk size limit is 5461
                                max_characters=2000
                                )

    # lazy load in case we get a lot of big docs
    docs_lazy = loader.lazy_load()

    for doc in docs_lazy:
        docs.append(
                doc
                )

    # filter metadata, chromadb doesn't handle complex metadata
    settings.vs.add_documents(documents=filter_complex_metadata(docs))

    print(f"Done - Docs added to {settings.vs.collection_name} collection: {len(docs)}")

def cmd_query(args: argparse.Namespace) -> None:
    from docquery.config import Settings

    settings = Settings()
    _apply_overrides(args, settings)

    system_prompt = _resolve_system_prompt(args)

    if not args.schema:
        # No Pydantic schema supplied — run a one-shot RAG answer
        from docquery.pipeline.chat import ChatAgent
        from docquery.tools.registry import ToolRegistry

        registry = ToolRegistry(settings.vs, settings)
        agent = ChatAgent(registry, settings, system_prompt=system_prompt)
        print(agent.chat(args.prompt))
        vs.close()
        return

    # Structured extraction mode: --schema must be a dotted Python class path
    if "." not in args.schema:
        print(
            f"Error: --schema must be a dotted Python class path, e.g. examples.arm_isa.ISAInstruction\n"
            f"Got: {args.schema!r}\n\n"
            f"To ask a free-form question without a schema, omit --schema entirely:\n"
            f"  docquery query \"your question\" --db {settings.db_path}"
        )
        raise SystemExit(1)

    from docquery.pipeline.extractor import ExtractionPipeline

    module_path, class_name = args.schema.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        print(f"Error: could not import module {module_path!r}. Check the --schema path.")
        raise SystemExit(1)
    output_model = getattr(module, class_name)

    pipeline = ExtractionPipeline(
        db_path=settings.db_path,
        output_model=output_model,
        system_prompt=system_prompt or f"Extract structured data matching the {class_name} schema.",
        settings=settings,
    )
    result = pipeline.run(args.prompt)
    print(result.model_dump_json(indent=2))


def cmd_chat(args: argparse.Namespace) -> None:
    from docquery.config import Settings
    from docquery.pipeline.chat import ChatAgent
    from docquery.tools.registry import ToolRegistry

    settings = Settings()
    _apply_overrides(args, settings)
    system_prompt = _resolve_system_prompt(args)

    registry = ToolRegistry(settings.vs, settings)
    agent = ChatAgent(registry, settings, system_prompt=system_prompt)

    print(f"RAG Chat — database: {settings.db_path}")
    print("Type 'exit' or Ctrl-D to quit. Type '/reset' to clear history.\n")
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.lower() == "exit":
                break
            if user_input == "/reset":
                agent.reset()
                print("(history cleared)")
                continue
            response = agent.chat(user_input)
            print(f"Agent: {response}\n")
    finally:
        vs.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="docquery", description="PDF → embeddings → LangGraph RAG pipeline")
    _add_common_args(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest one or more PDFs into the vector database")
    p_ingest.add_argument("files", nargs="+", metavar="FILE",
                          help="One or more PDF or plain-text/code files to ingest")
    p_ingest.add_argument("--chunk-size", type=int, default=None,
                          help="Characters per chunk (overrides CHUNK_SIZE env var)")
    p_ingest.add_argument("--chunk-overlap", type=int, default=None,
                          help="Overlap between chunks (overrides CHUNK_OVERLAP env var)")
    _add_common_args(p_ingest)

    # query (free-form answer or structured extraction)
    p_query = sub.add_parser(
        "query",
        help="Ask a one-shot question (omit --schema) or extract structured data (provide --schema)",
    )
    p_query.add_argument("prompt", help="Question or extraction instruction")
    p_query.add_argument(
        "--schema",
        default=None,
        metavar="MODULE.ClassName",
        help="Dotted path to a Pydantic model class for structured extraction, e.g. examples.arm_isa.ISAInstruction. "
             "Omit for a free-form RAG answer.",
    )
    _add_prompt_args(p_query)
    _add_common_args(p_query)
    _add_inference_args(p_query)

    # chat
    p_chat = sub.add_parser("chat", help="Interactive RAG chat session")
    _add_prompt_args(p_chat)
    _add_common_args(p_chat)
    _add_inference_args(p_chat)

    args = parser.parse_args()

    from docquery.logging_config import setup_logging
    level = getattr(args, "log_level", "INFO")
    setup_logging(level)

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "chat":
        cmd_chat(args)


if __name__ == "__main__":
    main()

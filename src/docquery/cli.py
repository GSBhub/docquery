import argparse
import importlib
import sys


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--db", default=None, help="Path to SQLite database (overrides DB_PATH env var)")


def _add_inference_args(parser: argparse.ArgumentParser) -> None:
    """Args that control retrieval and LLM behaviour for chat/query commands."""
    parser.add_argument("--top-k", type=int, default=None, metavar="K",
                        help="Number of similarity results to retrieve (overrides TOP_K env var)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="LLM sampling temperature 0–2 (overrides TEMPERATURE env var, default 0)")


def _apply_overrides(args: argparse.Namespace, settings) -> None:
    if args.db:
        settings.db_path = args.db
    if getattr(args, "top_k", None) is not None:
        settings.top_k = args.top_k
    if getattr(args, "temperature", None) is not None:
        settings.temperature = args.temperature


def cmd_ingest(args: argparse.Namespace) -> None:
    from docquery.config import Settings
    from docquery.embeddings.provider import get_embeddings
    from docquery.ingestion import chunker, pdf_loader
    from docquery.storage.vector_store import VectorStore

    settings = Settings()
    _apply_overrides(args, settings)
    if getattr(args, "chunk_size", None) is not None:
        settings.chunk_size = args.chunk_size
    if getattr(args, "chunk_overlap", None) is not None:
        settings.chunk_overlap = args.chunk_overlap

    embeddings = get_embeddings(settings)
    sample = embeddings.embed_query("probe")
    dim = len(sample)

    vs = VectorStore(settings.db_path, embedding_dim=dim)
    total_inserted = total_skipped = 0
    for pdf_path in args.pdfs:
        docs = pdf_loader.load(pdf_path)
        chunks = chunker.chunk(docs, settings)
        inserted, skipped = vs.add_chunks(chunks, embeddings, batch_size=settings.embed_batch_size)
        print(f"  {pdf_path}: {inserted} new chunks, {skipped} duplicates skipped")
        total_inserted += inserted
        total_skipped += skipped
    total = vs.get_chunk_count()
    vs.close()
    print(f"Done — {total_inserted} new, {total_skipped} skipped. DB total: {total} chunks ({settings.db_path})")


def cmd_query(args: argparse.Namespace) -> None:
    from docquery.config import Settings

    settings = Settings()
    _apply_overrides(args, settings)

    if not args.schema:
        # No Pydantic schema supplied — run a one-shot RAG answer
        from docquery.embeddings.provider import get_embeddings
        from docquery.pipeline.chat import ChatAgent
        from docquery.storage.vector_store import VectorStore
        from docquery.tools.registry import ToolRegistry

        embeddings = get_embeddings(settings)
        dim = len(embeddings.embed_query("probe"))
        vs = VectorStore(settings.db_path, embedding_dim=dim)
        registry = ToolRegistry(vs, embeddings, settings)
        agent = ChatAgent(registry, settings)
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
        system_prompt=args.system_prompt or f"Extract structured data matching the {class_name} schema.",
        settings=settings,
    )
    result = pipeline.run(args.prompt)
    print(result.model_dump_json(indent=2))


def cmd_chat(args: argparse.Namespace) -> None:
    from docquery.config import Settings
    from docquery.embeddings.provider import get_embeddings
    from docquery.pipeline.chat import ChatAgent
    from docquery.storage.vector_store import VectorStore
    from docquery.tools.registry import ToolRegistry

    settings = Settings()
    _apply_overrides(args, settings)

    embeddings = get_embeddings(settings)
    sample = embeddings.embed_query("probe")
    dim = len(sample)

    vs = VectorStore(settings.db_path, embedding_dim=dim)
    registry = ToolRegistry(vs, embeddings, settings)
    agent = ChatAgent(registry, settings)

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
    p_ingest.add_argument("pdfs", nargs="+", metavar="PDF", help="One or more PDF files to ingest")
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
    p_query.add_argument("--system-prompt", default=None)
    _add_common_args(p_query)
    _add_inference_args(p_query)

    # chat
    p_chat = sub.add_parser("chat", help="Interactive RAG chat session")
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

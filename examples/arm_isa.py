"""
Reference example: ARM ISA manual → structured ISAInstruction extraction + RAG chat.

Usage:
    # First ingest your ARM manual PDF:
    docquery ingest arm_manual.pdf --db arm_manual.db

    # Then run this script:
    python examples/arm_isa.py
"""
from pydantic import BaseModel

from docquery.config import Settings
from docquery.embeddings.provider import get_embeddings
from docquery.pipeline.chat import ChatAgent
from docquery.pipeline.extractor import ExtractionPipeline
from docquery.storage.vector_store import VectorStore
from docquery.tools.registry import ToolRegistry


class Operand(BaseModel):
    name: str
    type: str  # register, immediate, memory, label


class ISAInstruction(BaseModel):
    mnemonic: str
    operands: list[Operand]
    description: str
    encoding: str
    flags_affected: list[str]


SYSTEM_PROMPT = """You are an expert CPU architecture analyst.
Extract instruction details from the ARM Architecture Reference Manual.
Return ONLY valid JSON matching the provided schema. No markdown, no explanation."""

DB_PATH = "arm_manual.db"


def run_extraction():
    pipeline = ExtractionPipeline(
        db_path=DB_PATH,
        output_model=ISAInstruction,
        system_prompt=SYSTEM_PROMPT,
    )
    result = pipeline.run("Extract the LDR instruction: encoding, operands, and flags affected")
    print("=== Structured Extraction ===")
    print(result.model_dump_json(indent=2))


def run_chat():
    settings = Settings()
    settings.db_path = DB_PATH
    embeddings = get_embeddings(settings)
    sample = embeddings.embed_query("probe")
    vs = VectorStore(DB_PATH, embedding_dim=len(sample))
    registry = ToolRegistry(vs, embeddings, settings)
    agent = ChatAgent(registry, settings)

    print("\n=== RAG Chat ===")
    questions = [
        "What does the MOV instruction do? Give me an example.",
        "What opcode byte is 0xfe 0xed?",
        "What's on page 50 of the manual?",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {agent.chat(q)}\n")

    vs.close()


if __name__ == "__main__":
    run_extraction()
    run_chat()

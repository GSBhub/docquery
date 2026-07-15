"""
Reference example: ARM ISA manual → structured ISAInstruction extraction + RAG chat.

Usage:
    # First ingest your ARM manual PDF:
    docquery ingest arm_manual.pdf --db arm_manual.db

    # Then run this script:
    python examples/arm_isa.py
"""
from pydantic import BaseModel

import docquery
from docquery.config import Settings


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


def _settings() -> Settings:
    settings = Settings()
    settings.db_path = DB_PATH
    return settings


def run_extraction():
    result = docquery.query(
        "Extract the LDR instruction: encoding, operands, and flags affected",
        schema=ISAInstruction,
        system_prompt=SYSTEM_PROMPT,
        settings=_settings(),
    )
    print("=== Structured Extraction ===")
    print(result.model_dump_json(indent=2))


def run_chat():
    session = docquery.chat_session(settings=_settings())

    print("\n=== RAG Chat ===")
    questions = [
        "What does the MOV instruction do? Give me an example.",
        "What opcode byte is 0xfe 0xed?",
        "What's on page 50 of the manual?",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {session.chat(q)}\n")


if __name__ == "__main__":
    run_extraction()
    run_chat()

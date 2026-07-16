"""
Reference example: ISA reference manual -> grounded instruction-set extraction.

Same engine as examples/peripherals.py, different target: the schema declares
that ``instructions`` is an enumeration of the ``instruction`` entity type, so
docquery enumerates every instruction the manual tags and extracts each one's
operands / encoding / flags — membership from the document's tags, not the LLM.

Only two things are per-ISA, and both are the caller's: the entity rule that
tags instruction headings at ingest (heading conventions differ per manual),
and, if you want them, extra schema fields. The engine and this schema are not.

    # ARMv7-M ARM — instruction sections look like "A7.7.4  ADD (register)":
    docquery ingest armv7m.pdf --db arm.db \
        --entity 'instruction=^A7\\.7\\.\\d+\\s+([A-Z][A-Z0-9.]+)'

    # RISC-V unprivileged ISA — numbered instruction subsections:
    docquery ingest riscv.pdf --db riscv.db \
        --entity 'instruction=^\\d+\\.\\d+\\s+([A-Z][A-Za-z0-9.]+)'

    # TMS320C67x CPU reference — instruction pages headed by the mnemonic:
    docquery ingest c67x.pdf --db c67x.db \
        --entity 'instruction=^([A-Z][A-Z0-9]+)\\s+.*\\b[Ii]nstruction\\b'

    python -m examples.arm_isa --db arm.db
"""

from __future__ import annotations

import argparse
import sys

import docquery
from docquery.config import Settings
from pydantic import BaseModel, Field


class Operand(BaseModel):
    name: str
    # role (register / immediate / memory / label) is a judgement, not a
    # verbatim token, so exempt it from grounding.
    role: str | None = Field(default=None, json_schema_extra={"grounding": "off"})


class Instruction(BaseModel):
    mnemonic: str
    operands: list[Operand] = Field(default_factory=list)   # not an enumeration; extracted per instruction
    encoding: str | None = None
    flags_affected: str | None = None
    description: str | None = Field(default=None, json_schema_extra={"grounding": "off"})


class InstructionSet(BaseModel):
    instructions: list[Instruction] = Field(
        default_factory=list, json_schema_extra={"enumerate": "instruction"})


SYSTEM_PROMPT = """You are extracting CPU instruction facts.
Copy mnemonics, operand names, and encoding bit patterns VERBATIM from the context.
Use null for anything the context does not state; never fill from prior knowledge."""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Grounded instruction-set decode of an ingested ISA manual")
    ap.add_argument("--db", required=True,
                    help="store from `docquery ingest ... --entity instruction=<regex>`")
    args = ap.parse_args(argv)

    settings = Settings()
    settings.db_path = args.db

    isa = docquery.query(
        "Decode every instruction this ISA defines: mnemonic, operands, "
        "encoding, and flags affected.",
        schema=InstructionSet, system_prompt=SYSTEM_PROMPT, settings=settings)

    print(isa.model_dump_json(indent=2))
    if not isa.instructions:
        print("No instructions enumerated — was the manual ingested with an "
              "`--entity instruction=<regex>` rule?", file=sys.stderr)
    return 0 if isa.instructions else 1


if __name__ == "__main__":
    raise SystemExit(main())

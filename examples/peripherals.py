"""
Reference example: MCU reference manual → grounded peripheral/register/field
extraction.

Walks an ingested manual in three grounded stages — enumerate peripherals,
then each peripheral's registers, then each register's fields — because a
single query cannot retrieve enough context for a whole manual under TOP_K.
Every value is extracted under docquery's grounding checks: names, addresses,
offsets, and reset values must appear (correctly associated) in the retrieved
document text.

Usage:
    # First ingest your manual PDF:
    docquery ingest s3c2440_um.pdf --db s3c2440.db

    # Walk the whole manual (JSON on stdout, progress on stderr):
    python -m examples.peripherals --db s3c2440.db

    # Restrict the walk while iterating:
    python -m examples.peripherals --db s3c2440.db --peripheral "I/O ports" --max-registers 4

    # Or run a single stage through the CLI:
    docquery query "Registers of the UART peripheral with addresses and reset values" \
        --schema examples.peripherals.RegisterList --db s3c2440.db
"""

from __future__ import annotations

import argparse
import json
import sys

from pydantic import BaseModel, Field

import docquery
from docquery.config import Settings


class Peripheral(BaseModel):
    name: str = Field(description="Peripheral/module name exactly as printed")
    base_address: str | None = Field(
        default=None,
        description="Base or boundary address exactly as printed (e.g. 0x56000000); null if not stated",
    )


class PeripheralList(BaseModel):
    peripherals: list[Peripheral]


class Register(BaseModel):
    name: str = Field(description="Register name exactly as printed (keep placeholders like GPIOx_CFGR)")
    offset: str | None = Field(
        default=None, description="Address offset exactly as printed; null if not stated")
    address: str | None = Field(
        default=None, description="Absolute address exactly as printed; null if not stated")
    reset_value: str | None = Field(
        default=None, description="Reset/initial value exactly as printed; null if not stated")


class RegisterList(BaseModel):
    # The peripheral is named by the caller's query, not extracted from the
    # document text, so it is exempt from grounding enforcement.
    peripheral: str = Field(json_schema_extra={"grounding": "off"})
    registers: list[Register]


class RegisterField(BaseModel):
    name: str = Field(description="Field name exactly as printed")
    bits: str = Field(description="Bit or bit range exactly as printed (e.g. 31:16 or 3)")
    reset_value: str | None = Field(
        default=None, description="Field reset value exactly as printed; null if not stated")
    description: str | None = Field(
        default=None,
        description="What the field does (may be paraphrased)",
        json_schema_extra={"grounding": "off"},
    )


class RegisterFieldList(BaseModel):
    # Caller-supplied echo of which register was asked about (see above).
    register_name: str = Field(json_schema_extra={"grounding": "off"})
    fields: list[RegisterField]


SYSTEM_PROMPT = """You are extracting hardware documentation facts.
Copy names, addresses, offsets, bit ranges, and reset values VERBATIM from the
provided context — keep placeholders exactly as printed (GPIOx_CFGR, Bit 2y+1:2y).
Prefer TABLE/ROW/ENCODING lines as the source of values; they are authoritative.
Use null for any value the context does not state. Never fill gaps from prior
knowledge, and only list entries that appear in the context."""


def _query(prompt: str, schema: type[BaseModel], settings: Settings) -> BaseModel | None:
    """One grounded extraction; None (with a stderr note) when retries exhaust."""
    try:
        return docquery.query(prompt, schema=schema,
                              system_prompt=SYSTEM_PROMPT, settings=settings)
    except ValueError as exc:
        print(f"  ! {schema.__name__}: {str(exc)[:160]}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Grounded peripheral/register/field walk of an ingested manual")
    ap.add_argument("--db", required=True, help="ChromaDB store created by `docquery ingest`")
    ap.add_argument("--peripheral", help="Only walk this peripheral (stage-1 output filtered by substring)")
    ap.add_argument("--max-peripherals", type=int, default=0, help="Cap stage-2 fan-out (0 = all)")
    ap.add_argument("--max-registers", type=int, default=0, help="Cap stage-3 fan-out per peripheral (0 = all)")
    ap.add_argument("--skip-fields", action="store_true", help="Stop after registers (no stage 3)")
    args = ap.parse_args(argv)

    settings = Settings()
    settings.db_path = args.db

    print("stage 1: enumerating peripherals", file=sys.stderr)
    plist = _query(
        "List the peripherals (hardware modules) of this microcontroller and "
        "their base addresses, from the memory map / peripheral boundary "
        "address / special function register tables.",
        PeripheralList, settings)
    peripherals = plist.peripherals if plist else []
    if args.peripheral:
        needle = args.peripheral.lower()
        peripherals = [p for p in peripherals if needle in p.name.lower()]
    if args.max_peripherals:
        peripherals = peripherals[:args.max_peripherals]

    out: dict = {"peripherals": []}
    for p in peripherals:
        print(f"stage 2: registers of {p.name}", file=sys.stderr)
        entry: dict = {**p.model_dump(), "registers": []}
        rlist = _query(
            f"List the registers of the {p.name} peripheral with their "
            "addresses, address offsets, and reset values, from its register "
            "map / register overview table.",
            RegisterList, settings)
        registers = rlist.registers if rlist else []
        seen: set[str] = set()
        registers = [r for r in registers
                     if r.name and not (r.name in seen or seen.add(r.name))]
        if args.max_registers:
            registers = registers[:args.max_registers]
        for r in registers:
            reg_entry: dict = {**r.model_dump(), "fields": []}
            if not args.skip_fields:
                print(f"stage 3: fields of {r.name}", file=sys.stderr)
                flist = _query(
                    f"List the bit fields of the {r.name} register of the "
                    f"{p.name} peripheral: field names, bit ranges, reset "
                    "values, and what each field does.",
                    RegisterFieldList, settings)
                reg_entry["fields"] = [f.model_dump() for f in flist.fields] if flist else []
            entry["registers"].append(reg_entry)
        out["peripherals"].append(entry)

    json.dump(out, sys.stdout, indent=2)
    print(file=sys.stdout)
    return 0 if out["peripherals"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

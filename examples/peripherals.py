"""
Reference example: MCU reference manual -> grounded peripheral/register/field
extraction, driven entirely by an annotated Pydantic schema.

The schema below *declares* which list fields are enumerations of grounded
entity types (``enumerate=...``). docquery then drives extraction with its
generic decision node — grounded enumeration where the document tags the type,
a derivation where it does not, LLM generation (with a warning) where neither
is possible — so this file carries almost no logic: just the schema (what to
extract and how to format it) plus one small derivation strategy (how loose
registers cluster into peripherals on a manual with no peripheral table).

Usage:
    docquery ingest s3c2440_um.pdf --db s3c2440.db
    python -m examples.peripherals --db s3c2440.db

    # AT32 tags peripherals directly (a memory map), so no derivation is needed:
    docquery ingest at32f423.pdf --db at32f423.db
    python -m examples.peripherals --db at32f423.db
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import docquery
from docquery.config import Settings
from docquery._enumerate import EntityItem, enumerate_entities
from pydantic import BaseModel, Field


# --- the spec: what to extract from the document and how to format it ---------

class RegisterField(BaseModel):
    name: str
    bits: str
    reset_value: str | None = None
    description: str | None = Field(default=None, json_schema_extra={"grounding": "off"})


class Register(BaseModel):
    name: str
    offset: str | None = None
    address: str | None = None
    reset_value: str | None = None
    fields: list[RegisterField] = Field(
        default_factory=list, json_schema_extra={"enumerate": "register_field"})


class Peripheral(BaseModel):
    name: str
    base_address: str | None = None
    registers: list[Register] = Field(
        default_factory=list, json_schema_extra={"enumerate": "register"})


class Chip(BaseModel):
    # Enumerate 'peripheral' entities when the manual tags them (AT32's memory
    # map); otherwise derive this level by grouping 'register' entities with
    # the "base_address" strategy below.
    peripherals: list[Peripheral] = Field(
        default_factory=list,
        json_schema_extra={"enumerate": "peripheral",
                           "derive": {"from": "register", "by": "base_address"}})


SYSTEM_PROMPT = """You are extracting hardware documentation facts.
Copy names, addresses, offsets, bit ranges, and reset values VERBATIM from the
provided context — keep placeholders exactly as printed (GPIOx_CFGR, Bit 2y+1:2y).
Prefer TABLE/ROW/ENCODING lines as the source of values; they are authoritative.
Use null for any value the context does not state. Never fill gaps from prior
knowledge, and only list entries that appear in the context."""


# --- the only target-specific logic: cluster loose registers into peripherals -

_HEX_RE = re.compile(r"0[xX][0-9a-fA-F]+")


def _register_addresses(settings: Settings) -> dict[str, int]:
    """name -> numeric base address, from the recovered register_map rows."""
    addrs: dict[str, int] = {}
    for doc in docquery.structures("register_map", settings=settings):
        for tbl in doc.get("records", []):
            for row in tbl.get("rows", []):
                name = (row.get("register") or row.get("name") or "").strip()
                m = _HEX_RE.search(row.get("address") or "")
                if name and m and name not in addrs:
                    addrs[name] = int(m.group(), 16)
    return addrs


def group_by_base(registers: list[EntityItem], settings: Settings,
                  mask: int = 0xFFFF0000) -> list[dict]:
    """Cluster grounded register entities into peripherals by address base."""
    addrs = _register_addresses(settings)
    groups: dict[int, list[EntityItem]] = {}
    for r in registers:
        if r.name in addrs:
            groups.setdefault(addrs[r.name] & mask, []).append(r)
    out = []
    for base in sorted(groups):
        members = groups[base]
        prefix = re.sub(r"[^A-Za-z0-9]+$", "",
                        os.path.commonprefix([m.name for m in members]))
        out.append({
            "seed": {"name": prefix if len(prefix) >= 2 else f"peripheral_{base:#010x}",
                     "base_address": f"0x{base:08x}"},
            "members": members,
        })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Grounded peripheral/register/field decode of an ingested manual")
    ap.add_argument("--db", required=True, help="ChromaDB store created by `docquery ingest`")
    args = ap.parse_args(argv)

    settings = Settings()
    settings.db_path = args.db

    chip = docquery.query(
        "Decode every peripheral of this microcontroller, its registers, and "
        "their bit fields, with addresses, offsets, and reset values.",
        schema=Chip, system_prompt=SYSTEM_PROMPT, settings=settings,
        derivations={"base_address": group_by_base})

    print(chip.model_dump_json(indent=2))
    return 0 if chip.peripherals else 1


if __name__ == "__main__":
    raise SystemExit(main())

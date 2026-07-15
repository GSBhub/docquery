"""
Repro/diagnosis harness for grounded extraction on real reference manuals.

Two modes:

  --tables-only   No LLM or embeddings needed. Runs the deterministic table
                  recovery (docquery._tables) over each PDF and prints per-kind
                  line counts plus sample ROW lines — shows whether register
                  maps / memory maps / field tables survive ingest.

  (default)       Slices each manual to the page ranges below (full-manual
                  unstructured ingest is slow), ingests each slice into a
                  persistent per-manual store, prints structure coverage, then
                  runs the staged examples.peripherals extraction with
                  GROUNDING=warn while counting grounding-miss warnings.

Usage:
    python -m examples.grounding_repro --tables-only ~/Downloads/s3c2440_um.pdf
    python -m examples.grounding_repro ~/Downloads/at32f423.pdf --pages 80-95,145-165
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

# Page ranges (1-based, inclusive) that carry each manual's memory map and a
# representative peripheral chapter; used when --pages is not given.
DEFAULT_PAGES = {
    "s3c2440_um": "30-40,54-70,279-300",
    "at32f423": "80-95,145-165",
}


def _parse_pages(spec: str) -> list[int]:
    pages: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            pages.extend(range(int(lo), int(hi) + 1))
        else:
            pages.append(int(part))
    return pages


def tables_report(pdf: Path) -> None:
    from docquery.config import Settings
    from docquery._tables import extract_document_tables

    rules = Settings().structure_rules
    lines_by_page = extract_document_tables(pdf, rules)
    kinds: Counter[str] = Counter()
    samples: dict[str, list[str]] = {}
    for page, lines in lines_by_page.items():
        kind = None
        for ln in lines:
            if m := re.match(r"^TABLE (\S+):", ln.strip()):
                kind = m[1]
                kinds[kind] += 1
            elif kind and ln.startswith("ROW "):
                samples.setdefault(kind, []).append(f"p{page}: {ln}")
    print(f"\n== {pdf.name}: tables on {len(lines_by_page)} pages ==")
    for kind, n in kinds.most_common():
        print(f"  {kind}: {n} tables")
        for s in samples.get(kind, [])[:3]:
            print(f"    {s[:120]}")
    if not kinds:
        print("  (no rule-matching tables recovered)")


class _MissCounter(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.misses: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "Ungrounded extraction fields" in msg:
            self.misses.append(msg)


def staged_run(pdf: Path, pages_spec: str | None) -> None:
    import fitz  # pymupdf

    import docquery
    from docquery.config import Settings
    from examples.peripherals import main as walk

    stem = pdf.stem.lower()
    pages_spec = pages_spec or next(
        (v for k, v in DEFAULT_PAGES.items() if k in stem), None)

    db = f"{stem}.repro.db"
    if not Path(db).exists():
        src: Path = pdf
        if pages_spec:
            pages = _parse_pages(pages_spec)
            sliced = Path(tempfile.mkdtemp(prefix="docquery-repro-")) / pdf.name
            with fitz.open(pdf) as doc, fitz.open() as out:
                for p in pages:
                    out.insert_pdf(doc, from_page=p - 1, to_page=p - 1)
                out.save(sliced)
            src = sliced
            print(f"ingesting pages {pages_spec} of {pdf.name} -> {db}", file=sys.stderr)
        settings = Settings()
        settings.db_path = db
        docquery.ingest([str(src)], settings=settings)

    print(f"\n== {pdf.name}: structure coverage ==")
    for kind, info in sorted(docquery.structure_coverage(db).items()):
        print(f"  {kind}: {info['count']} docs on {info['pages']} pages")

    counter = _MissCounter()
    logging.getLogger("docquery").addHandler(counter)
    import os
    os.environ["GROUNDING"] = "warn"
    print(f"\n== {pdf.name}: staged extraction (GROUNDING=warn) ==")
    walk(["--db", db, "--max-peripherals", "3", "--max-registers", "3"])
    print(f"\n== {pdf.name}: {len(counter.misses)} grounding-miss warning(s) ==")
    for m in counter.misses[:10]:
        print(f"  {m[:160]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdfs", nargs="+", type=Path)
    ap.add_argument("--tables-only", action="store_true",
                    help="deterministic table recovery only (no LLM/embeddings)")
    ap.add_argument("--pages", help="page ranges to slice-ingest, e.g. 80-95,145-165")
    args = ap.parse_args(argv)

    for pdf in args.pdfs:
        if args.tables_only:
            tables_report(pdf.expanduser())
        else:
            staged_run(pdf.expanduser(), args.pages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

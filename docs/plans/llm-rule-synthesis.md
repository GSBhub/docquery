# Plan: LLM rule-synthesis pass at ingest (format-agnostic manual decoding)

**Status:** future / not started
**Depends on:** working LLM endpoint (open-webui bearer key in `.env`); the
grounded peripheral/register/field pipeline validated end-to-end on the
S3C2440 and AT32F423 manuals (2026-07 changes: `register_map`/`memory_map`
rules, block-scoped grounding units, spaced-hex normalization).

## Motivation

docquery's deterministic table recovery (`_tables.tables_from_words`) only
finds tables whose header matches a shipped `StructureRule`. The rules are the
format knowledge, and they are static: every new manual family (vendor,
layout generation, language) that words its headers differently silently
recovers nothing until someone hand-writes a rule.

Goal: use the LLM as the *glue* that unifies differently formatted — or
different-language — technical manuals into one canonical structured layer,
so backends can decode any manual into well-structured representations (SVD,
SystemRDL, semantic instruction lists, CPU register mappings, circuit design
specifications) without being tied to any one document format.

## The invariant that must hold

The LLM identifies **how to read the document** (which regions are tables,
what the columns mean, what vocabulary this manual uses). It must never be
the source of **what the values are**. Values always flow through the
deterministic geometry pass and the grounding gate. A hallucinated rule
proposal can recover real geometry or recover nothing — it cannot corrupt a
value. This preserves the property that grounding verifies LLM output against
document-derived text, never against other LLM output.

Three tiers, each independently verifiable:

| Tier | Actor | Verified by |
|---|---|---|
| Reading strategy (rules) | LLM proposes | deterministic re-extraction succeeds under existing gates |
| Values | geometry (`tables_from_words`) | exact by construction |
| Association / selection at query time | LLM | `ungrounded_fields` / `ungrounded_records` co-occurrence checks |

## Design: a small LangGraph at ingest

Mirrors the existing `extract → validate → should_retry` loop in
`src/docquery/_nodes.py` (same retry-with-errors pattern, same
`get_structured_llm` constrained-decoding path).

```
detect_candidates → propose_rule → verify_rule → (retry ≤ N | accept | skip)
```

1. **`detect_candidates` (deterministic, no LLM).** Find table-shaped regions
   that matched *no* existing rule: pymupdf's built-in `page.find_tables()`
   as a rule-free detector, or a relaxed geometric scan (aligned column
   clusters below a header-ish row). Emit per candidate: header band text, a
   few sample rows, and how often that header shape recurs across the
   document (recurrence = worth synthesizing; one-offs are not). Only run on
   pages with no rule match (cost control).
2. **`propose_rule` (LLM, constrained decoding).** Input: header/sample text.
   Output: a `StructureRule` proposal as a Pydantic schema — canonical `kind`
   (`register_map`, `register_fields`, `memory_map`, `interrupt_table`,
   `pin_map`, or a new kind), synonym groups mapping this manual's header
   vocabulary onto canonical column keys, `name_column`, `allow_empty_key`.
   Multi-language falls out here for free: a French `Registre | Adresse |
   Valeur initiale` header is just another synonym mapping; canonical keys
   stay `register | address | reset_value`.
3. **`verify_rule` (deterministic).** Apply the proposal by re-running
   `tables_from_words` on the actual page words. Accept only if it yields
   well-formed tables under the existing gates (row counts, key-cell shape,
   name-column fill) on the candidate pages — and ideally on the other pages
   where the header shape recurs. Reject/retry with the failure fed back as
   validation errors, capped like `Settings.max_retries`.
4. **Persist.** Accepted rules are stored with the ingest (rules are already
   env/CLI-mergeable by `kind`; scope synthesized rules to the document's
   store, not globally), so re-ingests and sibling manuals from the same
   vendor reuse them without another LLM call.

Same pattern extends to `EntityRule` synthesis: the LLM proposes the heading
regex for S3C2440-style ALL-CAPS unnumbered section headings; verification =
the regex must actually match ≥ N headings across the document.

## Surface

- `docquery ingest <pdf> --synthesize-rules` (opt-in flag; default off).
- New module, e.g. `src/docquery/_rule_synthesis.py` (graph + nodes), wired
  from `_ingest.ingest_documents` after the normal table pass reports its
  per-page match map.
- Proposal schema lives beside `StructureRule` in `config.py` (a Pydantic
  mirror of the dataclass, plus a `confidence`/`rationale` field the verify
  node ignores).
- Synthesized-rule provenance recorded in store metadata (rule JSON + source
  pages), inspectable via `docquery structures`.

## Downstream (separate follow-ons, enabled by this)

- **Canonical exporters** over the grounded structured layer: `structures()`
  already returns `{kind, records}` with stable column keys regardless of the
  manual's wording. An SVD exporter is a deterministic transform over the
  peripheral → register → field walk (`examples/peripherals.py` output);
  SystemRDL and instruction-list exporters follow the same shape.
- **Rule library growth**: synthesized rules that recur across manuals can be
  promoted (manually) into `_DEFAULT_STRUCTURE_RULES`.

## Risks / mitigations

- **Over-broad proposed rule** (matches prose as tables): the verify gates
  are the backstop; additionally scope synthesized rules to that document's
  store rather than merging into global defaults.
- **`find_tables()` cost**: slower than the word scan — run only on pages
  with zero rule matches.
- **Small-model proposal quality** (llama3.1:8b): constrained decoding
  guarantees schema shape; the verify loop converts weak proposals into
  retries or skips, never into bad data. Worst case: no rule synthesized,
  which is exactly today's behavior.
- **Kind proliferation**: prefer mapping onto existing canonical kinds;
  require a minimum recurrence count before accepting a novel kind.

## Acceptance criteria

- A manual whose register tables use header vocabulary no shipped rule knows
  (or a non-English manual) goes from zero recovered `register_map` docs to
  recovered ROW lines, with **no change to any extracted value's provenance**
  (values still verifiably from geometry).
- Strict-mode staged extraction (`examples/peripherals.py`) succeeds on such
  a manual without hand-written `STRUCTURE_RULES`.
- With `--synthesize-rules` off, behavior is byte-identical to today.
- Unit tests: proposal-schema round-trip; verify node rejects a deliberately
  over-broad rule; accepted-rule persistence and reuse on re-ingest.

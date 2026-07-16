"""Schema-driven structured extraction with grounded enumeration.

The single-shot extraction pipeline asks the LLM to produce a whole list of
items from one retrieval window — where it hallucinates list members, or
conflates one entity type with another. This module lets an output schema
instead *declare* that a list field is an enumeration of a grounded entity
type, and drives it with a decision:

    decide ─ grounded entities exist ──────→ enumerate + extract each item
           ├ none, but a derivation given ─→ derive groups from a child type
           └ neither ─────────────────────→ warn, fall back to LLM generation

Membership then comes from the deterministic entity tags (it cannot be
invented); only each item's own scalar fields are extracted by the LLM, still
under the per-field grounding checks. A ``list`` field opts in with
``Field(json_schema_extra={"enumerate": "<entity_type>"})`` and may add
``"derive": {"from": "<child_type>", "by": "<strategy>"}`` for aggregate
levels the document does not tag directly (grouping a child entity type into
parents). Target-agnostic: the entity types and strategies are the caller's,
nothing here names a domain.
"""

from __future__ import annotations

import logging
import typing
from typing import Any, Callable

from pydantic import BaseModel, create_model
from pydantic_core import PydanticUndefined

from docquery.config import Settings
from docquery._enumerate import EntityItem, available_entity_types, enumerate_entities

logger = logging.getLogger(__name__)

# A derivation strategy groups grounded child entities into parent items,
# given the child entities and the settings (for store access):
#   [{"seed": {field: value, ...}, "members": [EntityItem, ...]}, ...]
DerivationGroup = dict[str, Any]
Derivation = Callable[[list[EntityItem], Settings], list[DerivationGroup]]

_DEFAULT_PROMPT = "Extract the fields for {model}, copying values verbatim from the context."


def _list_item_model(annotation: Any) -> type[BaseModel] | None:
    """The BaseModel inside a ``list[X]`` annotation, else None."""
    if typing.get_origin(annotation) in (list, typing.List):
        (arg,) = typing.get_args(annotation) or (None,)
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return arg
    return None


def enumerable_field(model: type[BaseModel]) -> tuple[str, type[BaseModel], str, dict | None] | None:
    """``(field, item_model, entity_type, derive)`` for the model's enumerable
    list field, or None. A field opts in with ``json_schema_extra={"enumerate":
    ...}``; the first such field wins.
    """
    for name, info in model.model_fields.items():
        extra = info.json_schema_extra
        if not isinstance(extra, dict) or "enumerate" not in extra:
            continue
        item_model = _list_item_model(info.annotation)
        if item_model is not None:
            return name, item_model, str(extra["enumerate"]), extra.get("derive")
    return None


def has_enumerable(model: type[BaseModel]) -> bool:
    """True when *model* declares an enumerable list field."""
    return enumerable_field(model) is not None


def _name_field(item_model: type[BaseModel]) -> str | None:
    """The field that receives a grounded entity's name.

    ``name`` when present (registers, peripherals), else the first plain-``str``
    field (an instruction's ``mnemonic``, an interrupt's ``acronym``). Keeps
    the engine from hardcoding a domain's naming.
    """
    fields = item_model.model_fields
    if "name" in fields:
        return "name"
    for fname, info in fields.items():
        if info.annotation is str:
            return fname
    return None


def _reduced_model(model: type[BaseModel], exclude: set[str]) -> type[BaseModel] | None:
    """A copy of *model* for single-shot extraction of what the LLM should fill.

    Drops seeded fields (*exclude*) and the enumerable list field — that one is
    filled by recursion. Other fields are kept, including non-enumerable nested
    structures (e.g. an instruction's operand list). None when nothing remains.
    """
    ef = enumerable_field(model)
    drop = set(exclude) | ({ef[0]} if ef else set())
    fields: dict[str, Any] = {}
    for name, info in model.model_fields.items():
        if name in drop:
            continue
        default = info.default if info.default is not PydanticUndefined else None
        fields[name] = (info.annotation, default)
    if not fields:
        return None
    return create_model(f"{model.__name__}Scalars", **fields)


def _scalar_fields(
    model: type[BaseModel], settings: Settings, query: str, system_prompt: str,
    seed: dict, exclude: set[str], cache: dict,
) -> dict:
    """Extract *model*'s scalar fields (minus seed/enumerable), grounded."""
    reduced = _reduced_model(model, exclude)
    values = dict(seed)
    if reduced is not None:
        instance = _run(reduced, settings, query, system_prompt, cache)
        if instance is not None:
            for k, v in instance.model_dump().items():
                values.setdefault(k, v)
    # any non-seeded scalar the reduced extraction did not fill stays at default
    for name, info in model.model_fields.items():
        if name in values or _list_item_model(info.annotation) is not None:
            continue
        values[name] = info.default if info.default is not PydanticUndefined else None
    return values


def _run(model: type[BaseModel], settings: Settings, query: str,
         system_prompt: str, cache: dict) -> BaseModel | None:
    """Single-shot grounded extraction of *model*; None if it fails all retries."""
    from docquery._extractor import ExtractionPipeline
    key = (model, system_prompt)
    pipeline = cache.get(key)
    if pipeline is None:
        pipeline = ExtractionPipeline(output_model=model, system_prompt=system_prompt,
                                      settings=settings)
        cache[key] = pipeline
    try:
        return pipeline.run(query)
    except ValueError as exc:
        logger.info("structured: single-shot extraction of %s failed: %s",
                    model.__name__, str(exc)[:120])
        return None


def _item_query(entity_type: str, name: str, parent: str | None) -> str:
    scope = f" of {parent}" if parent else ""
    return (f"Extract the {entity_type} named {name}{scope}: its fields, copying "
            "every value verbatim from the context.")


def _scoped_list_query(entity_type: str, parent: str) -> str:
    return (f"List every {entity_type} of {parent}, copying each one's values "
            "verbatim from the context; include only entries that appear there.")


def _generate_list(item_model: type[BaseModel], settings: Settings, query: str,
                   system_prompt: str, cache: dict) -> list[BaseModel]:
    """LLM-generate a whole list of *item_model* (the ungrounded fallback)."""
    wrapper = create_model(f"{item_model.__name__}List",
                           items=(list[item_model], ...))  # type: ignore[valid-type]
    instance = _run(wrapper, settings, query, system_prompt, cache)
    return list(getattr(instance, "items", [])) if instance is not None else []


def _build(
    model: type[BaseModel], settings: Settings, query: str, system_prompt: str,
    derivations: dict[str, Derivation], cache: dict,
    seed: dict | None = None, members: list[EntityItem] | None = None,
    parent: str | None = None,
) -> BaseModel:
    """Recursively build one *model* instance: scalars + enumerable list."""
    seed = seed or {}
    ef = enumerable_field(model)
    exclude = set(seed) | ({ef[0]} if ef else set())
    values = _scalar_fields(model, settings, query, system_prompt, seed, exclude, cache)
    if ef is not None:
        values[ef[0]] = _resolve_list(
            ef, settings, query, system_prompt, derivations, cache,
            members=members, parent=values.get("name") or parent)
    return model(**values)


def _resolve_list(
    ef: tuple[str, type[BaseModel], str, dict | None], settings: Settings,
    query: str, system_prompt: str, derivations: dict[str, Derivation], cache: dict,
    members: list[EntityItem] | None, parent: str | None,
) -> list[BaseModel]:
    """Fill an enumerable list: grounded members -> enumerate -> derive -> warn."""
    _field, item_model, entity_type, derive = ef
    name_field = _name_field(item_model)

    def build_from(items: list[EntityItem]) -> list[BaseModel]:
        out = []
        for it in items:
            out.append(_build(
                item_model, settings, _item_query(entity_type, it.name, parent),
                system_prompt, derivations, cache,
                seed={name_field: it.name} if name_field else {}))
        return out

    # 1. members handed down by a derivation / parent scope — grounded
    if members is not None:
        return build_from(members)

    # A nested list (inside a parent item) must stay scoped to that parent; a
    # global entity enumeration would pull the whole document's entities under
    # one parent. Grounded enumeration only applies at the top level.
    if parent is not None:
        types = cache.setdefault("_types", set(available_entity_types(settings.vs)))
        if entity_type not in types:
            logger.warning(
                "structured: no %r entities are tagged anywhere; generating this "
                "list of %s ungrounded — likely empty under GROUNDING=strict and "
                "hallucination-prone under GROUNDING=off.", entity_type, parent)
        return _generate_list(item_model, settings,
                              _scoped_list_query(entity_type, parent), system_prompt, cache)

    # 2. top level, directly tagged — grounded enumeration
    items = enumerate_entities(settings.vs, entity_type)
    if items:
        return build_from(items)

    # 3. top level, not tagged, but derivable by grouping a child type
    if derive and derive.get("by") in derivations:
        children = enumerate_entities(settings.vs, str(derive.get("from", "")))
        if children:
            groups = derivations[derive["by"]](children, settings)
            if groups:
                out = []
                for g in groups:
                    seed = g.get("seed", {})
                    out.append(_build(
                        item_model, settings, query, system_prompt, derivations, cache,
                        seed=seed, members=g.get("members", []), parent=seed.get("name")))
                return out

    # 4. neither grounded enumeration nor derivation — warn loudly, then generate
    logger.warning(
        "structured: no grounded enumeration or derivation for entity type %r "
        "(available: %s). Falling back to LLM generation for this list — under "
        "GROUNDING=strict it will likely yield little, and under GROUNDING=off "
        "it is prone to hallucination.",
        entity_type, ", ".join(available_entity_types(settings.vs)) or "none")
    return _generate_list(item_model, settings, query, system_prompt, cache)


def extract_structured(
    model: type[BaseModel], settings: Settings, query: str,
    system_prompt: str | None = None,
    derivations: dict[str, Derivation] | None = None,
) -> BaseModel:
    """Extract *model* using grounded enumeration where its schema declares it."""
    prompt = system_prompt or _DEFAULT_PROMPT.format(model=model.__name__)
    return _build(model, settings, query, prompt, derivations or {}, cache={})

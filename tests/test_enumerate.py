"""Tests for core entity enumeration (docquery._enumerate)."""

from types import SimpleNamespace

from docquery._enumerate import available_entity_types, enumerate_entities


def _store(records):
    """A fake vector store whose _collection.get() returns *records*.

    Each record is ``(document, metadata)``.
    """
    docs = [d for d, _ in records]
    metas = [m for _, m in records]

    collection = SimpleNamespace(
        get=lambda include=None: {"documents": docs, "metadatas": metas})
    return SimpleNamespace(_collection=collection)


def test_distinct_entities_in_document_order():
    vs = _store([
        ("doc a", {"entity_register": "GPACON;GPADAT", "page": 5, "section_order": 1}),
        ("doc b", {"entity_register": "GPBCON", "page": 3, "section_order": 0}),
        ("doc c", {"entity_register": "GPACON", "page": 9, "section_order": 2}),  # dup
    ])
    items = enumerate_entities(vs, "register")
    assert [it.name for it in items] == ["GPBCON", "GPACON", "GPADAT"]  # section_order, then page
    assert items[0].page == 3


def test_reserved_and_markers_are_filtered():
    vs = _store([
        ("d", {"entity_register": "GPACON;Reserved;reserved;a.;12.;GPBCON"}),
    ])
    names = [it.name for it in enumerate_entities(vs, "register")]
    assert names == ["GPACON", "GPBCON"]


def test_kind_and_section_scoping():
    vs = _store([
        ("d1", {"entity_register": "R1", "kind": "register_map", "section": "GPIO"}),
        ("d2", {"entity_register": "R2", "kind": "register_fields", "section": "GPIO"}),
        ("d3", {"entity_register": "R3", "kind": "register_map", "section": "UART"}),
    ])
    assert [it.name for it in enumerate_entities(vs, "register", kind="register_map")] == ["R1", "R3"]
    assert [it.name for it in enumerate_entities(vs, "register", section="GPIO")] == ["R1", "R2"]


def test_unknown_type_returns_empty():
    vs = _store([("d", {"entity_register": "R1"})])
    assert enumerate_entities(vs, "peripheral") == []


def test_available_entity_types():
    vs = _store([
        ("d1", {"entity_register": "R1", "entity_pin": "PA0"}),
        ("d2", {"entity_interrupt": "NMI"}),
    ])
    assert available_entity_types(vs) == ["interrupt", "pin", "register"]

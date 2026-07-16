"""Tests for the schema-driven extraction engine (docquery._structured).

Cover the decision routing — grounded enumeration vs. derivation vs. the
warned LLM-generation fallback — plus the annotation/reduced-model helpers,
all without invoking an LLM (per-item build and generation are stubbed).
"""

import logging
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from docquery import _structured as st
from docquery._enumerate import EntityItem


class Fld(BaseModel):
    name: str
    bits: str = ""


class Reg(BaseModel):
    name: str
    address: str | None = None
    fields: list[Fld] = Field(default_factory=list,
                              json_schema_extra={"enumerate": "register_field"})


class Peri(BaseModel):
    name: str
    base_address: str | None = None
    registers: list[Reg] = Field(default_factory=list,
                                 json_schema_extra={"enumerate": "register"})


class Chip(BaseModel):
    peripherals: list[Peri] = Field(
        default_factory=list,
        json_schema_extra={"enumerate": "peripheral",
                           "derive": {"from": "register", "by": "base"}})


def _item(entity_type: str, name: str) -> EntityItem:
    return EntityItem(entity_type=entity_type, name=name, content="")


def _settings():
    return SimpleNamespace(vs=None)


# --- annotation / model helpers ----------------------------------------------

def test_enumerable_field():
    field, item_model, etype, derive = st.enumerable_field(Chip)
    assert (field, item_model, etype) == ("peripherals", Peri, "peripheral")
    assert derive == {"from": "register", "by": "base"}
    assert st.enumerable_field(Reg)[:3] == ("fields", Fld, "register_field")
    assert st.enumerable_field(Fld) is None


def test_has_enumerable():
    assert st.has_enumerable(Chip) and st.has_enumerable(Reg)
    assert not st.has_enumerable(Fld)


def test_list_item_model():
    assert st._list_item_model(list[Fld]) is Fld
    assert st._list_item_model(list[str]) is None
    assert st._list_item_model(str) is None


def test_reduced_model_drops_list_and_seed():
    reduced = st._reduced_model(Reg, exclude={"name"})
    assert list(reduced.model_fields) == ["address"]        # 'fields' list + seeded 'name' gone
    assert st._reduced_model(Chip, exclude=set()) is None   # only a list field


def test_reduced_model_keeps_non_enumerable_nested_list():
    # An instruction's operand list is not an enumeration; the LLM fills it in
    # the single-shot pass, so it must survive the reduction.
    class Op(BaseModel):
        name: str

    class Instr(BaseModel):
        mnemonic: str
        operands: list[Op] = Field(default_factory=list)
        encoding: str | None = None

    reduced = st._reduced_model(Instr, exclude={"mnemonic"})
    assert set(reduced.model_fields) == {"operands", "encoding"}


# --- routing ------------------------------------------------------------------

@pytest.fixture
def stub_build(monkeypatch):
    built: list[tuple] = []

    def fake_build(model, settings, query, sp, derivs, cache,
                   seed=None, members=None, parent=None):
        built.append((model.__name__, (seed or {}).get("name"),
                      None if members is None else len(members)))
        return model(name=(seed or {}).get("name") or "x")

    monkeypatch.setattr(st, "_build", fake_build)
    return built


def test_grounded_enumeration_path(monkeypatch, stub_build):
    monkeypatch.setattr(st, "enumerate_entities",
                        lambda vs, t, **k: [_item("peripheral", "GPIO"),
                                            _item("peripheral", "UART")] if t == "peripheral" else [])
    res = st._resolve_list(st.enumerable_field(Chip), _settings(), "q", "sys",
                           {}, {}, members=None, parent=None)
    assert len(res) == 2
    assert [b[1] for b in stub_build] == ["GPIO", "UART"]


def test_members_short_circuit_skips_enumeration(monkeypatch, stub_build):
    calls = {"n": 0}

    def enum(vs, t, **k):
        calls["n"] += 1
        return []

    monkeypatch.setattr(st, "enumerate_entities", enum)
    res = st._resolve_list(st.enumerable_field(Chip), _settings(), "q", "sys",
                           {}, {}, members=[_item("register", "R1")], parent=None)
    assert len(res) == 1 and calls["n"] == 0


def test_derivation_path(monkeypatch, stub_build):
    monkeypatch.setattr(st, "enumerate_entities",
                        lambda vs, t, **k: [_item("register", "GPACON"),
                                            _item("register", "GPBCON")] if t == "register" else [])
    derive = lambda children, settings: [{"seed": {"name": "GP"}, "members": children}]
    res = st._resolve_list(st.enumerable_field(Chip), _settings(), "q", "sys",
                           {"base": derive}, {}, members=None, parent=None)
    assert len(res) == 1
    assert stub_build[0][:3] == ("Peri", "GP", 2)   # derived group seed + member count


def test_degraded_path_warns_and_generates(monkeypatch):
    monkeypatch.setattr(st, "enumerate_entities", lambda vs, t, **k: [])
    monkeypatch.setattr(st, "available_entity_types", lambda vs: ["pin"])
    generated: list[str] = []
    monkeypatch.setattr(st, "_generate_list",
                        lambda im, s, q, sp, c: generated.append(im.__name__) or [])
    with pytest.MonkeyPatch.context():
        with _capture_warnings("docquery._structured") as records:
            res = st._resolve_list(st.enumerable_field(Chip), _settings(), "q", "sys",
                                   {}, {}, members=None, parent=None)
    assert res == []
    assert generated == ["Peri"]
    assert any("no grounded enumeration or derivation" in m for m in records)


class _capture_warnings:
    """Minimal caplog substitute so the test does not depend on the fixture."""

    def __init__(self, logger_name: str):
        self._logger = logging.getLogger(logger_name)
        self._records: list[str] = []

    def __enter__(self):
        self._handler = logging.Handler()
        self._handler.emit = lambda r: self._records.append(r.getMessage())
        self._handler.setLevel(logging.WARNING)
        self._logger.addHandler(self._handler)
        self._old = self._logger.level
        self._logger.setLevel(logging.WARNING)
        return self._records

    def __exit__(self, *exc):
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._old)
        return False

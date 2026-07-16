"""Tests for the peripherals example's schema + derivation strategy.

The enumeration/decision machinery is generic and lives in docquery._structured
(see test_structured.py); the example now carries only the annotated schema and
the base-address grouping strategy, which is what this file exercises.
"""

import examples.peripherals as ep
from docquery._enumerate import EntityItem
from docquery._structured import enumerable_field


def test_schema_declares_the_enumeration_tree():
    field, item_model, etype, derive = enumerable_field(ep.Chip)
    assert (field, etype) == ("peripherals", "peripheral")
    assert derive == {"from": "register", "by": "base_address"}
    assert enumerable_field(ep.Peripheral)[2] == "register"
    assert enumerable_field(ep.Register)[2] == "register_field"
    assert enumerable_field(ep.RegisterField) is None


def test_description_field_is_grounding_exempt():
    extra = ep.RegisterField.model_fields["description"].json_schema_extra
    assert extra == {"grounding": "off"}


def _structures_stub(rows):
    def structures(kind, settings=None):
        return [{"records": [{"columns": ["register", "address"], "rows": rows}]}]
    return structures


def test_group_by_base_clusters_registers(monkeypatch):
    monkeypatch.setattr(ep.docquery, "structures", _structures_stub([
        {"register": "GPACON", "address": "0x56000000"},
        {"register": "GPBCON", "address": "0x56000010"},
        {"register": "ADCCON", "address": "0x58000000"},
    ]))
    regs = [EntityItem("register", n, "") for n in ("GPACON", "GPBCON", "ADCCON")]
    groups = {g["seed"]["name"]: g for g in ep.group_by_base(regs, settings=None)}

    assert set(groups) == {"GP", "ADCCON"}
    assert groups["GP"]["seed"]["base_address"] == "0x56000000"
    assert len(groups["GP"]["members"]) == 2          # GPACON + GPBCON
    assert len(groups["ADCCON"]["members"]) == 1


def test_group_by_base_labels_prefixless_group_by_address(monkeypatch):
    monkeypatch.setattr(ep.docquery, "structures", _structures_stub([
        {"register": "FOO", "address": "0x40000000"},
        {"register": "BAR", "address": "0x40000004"},
    ]))
    regs = [EntityItem("register", n, "") for n in ("FOO", "BAR")]
    (group,) = ep.group_by_base(regs, settings=None)
    assert group["seed"]["name"] == "peripheral_0x40000000"


def test_group_by_base_ignores_registers_without_a_known_address(monkeypatch):
    monkeypatch.setattr(ep.docquery, "structures", _structures_stub([
        {"register": "WTCON", "address": "0x53000000"},
    ]))
    regs = [EntityItem("register", n, "") for n in ("WTCON", "MYSTERY")]
    (group,) = ep.group_by_base(regs, settings=None)
    assert [m.name for m in group["members"]] == ["WTCON"]

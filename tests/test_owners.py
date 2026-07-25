"""Tests for geometry-based structure→owner association (docquery._owners)."""

from __future__ import annotations

from docquery._owners import assign_owners, heading_positions


def _word(x: float, y: float, text: str, w: float = 8.0):
    return (x - w / 2, y, x + w / 2, y + 8, text)


# --- heading_positions -------------------------------------------------------

def test_heading_position_needs_anchor_token_not_just_the_name():
    """A running page header repeating the name must not win over the real heading."""
    words = [
        _word(300, 20, "ADD"),                     # running header (name only)
        _word(100, 200, "Syntax"), _word(160, 200, "ADD"),   # the real heading
    ]
    assert heading_positions(words, [("ADD", "Syntax   ADD")]) == [(200, "ADD")]


def test_heading_row_tolerates_baseline_wobble():
    """The label and its anchor often sit a point apart (real TI manuals do this:
    `ADD` at y=103 with `Syntax` at y=104) — they must still pair up."""
    words = [_word(160, 103, "ADD"), _word(100, 104, "Syntax")]
    assert heading_positions(words, [("ADD", "Syntax   ADD")]) == [(103, "ADD")]


def test_heading_positions_sorted_by_y():
    words = [
        _word(100, 400, "Syntax"), _word(160, 400, "SUB"),
        _word(100, 100, "Syntax"), _word(160, 100, "ADD"),
    ]
    got = heading_positions(words, [("SUB", "Syntax SUB"), ("ADD", "Syntax ADD")])
    assert got == [(100, "ADD"), (400, "SUB")]


def test_heading_positions_name_only_rule_matches_bare_name():
    words = [_word(100, 50, "LDW")]
    assert heading_positions(words, [("LDW", "LDW")]) == [(50, "LDW")]


def test_heading_matches_name_wrapped_in_punctuation():
    """Manuals delimit the name — `19.4.3 GPIO output register (GPIOx_OMODE)` —
    and the PDF keeps the parens attached to the word."""
    words = [
        _word(100, 200, "19.4.3"), _word(150, 200, "register"),
        _word(200, 200, "(GPIOx_OMODE)"),
    ]
    got = heading_positions(words, [("GPIOx_OMODE", "19.4.3 register (GPIOx_OMODE)")])
    assert got == [(200, "GPIOx_OMODE")]


def test_heading_positions_missing_name_is_skipped():
    assert heading_positions([_word(100, 50, "ADD")], [("SUB", "Syntax SUB")]) == []


# --- assign_owners -----------------------------------------------------------

def test_block_belongs_to_nearest_heading_above():
    blocks = {1: [(300, "encA"), (500, "encB")]}
    heads = {1: [(200, "ADD"), (400, "SUB")]}
    assert assign_owners(blocks, heads) == {1: [("encA", "ADD"), ("encB", "SUB")]}


def test_owner_carries_across_a_page_break():
    """A page that opens mid-description inherits the previous page's entity."""
    blocks = {1: [(300, "enc1")], 2: [(100, "enc2")]}
    heads = {1: [(200, "ADD")]}                       # page 2 has no heading
    assert assign_owners(blocks, heads) == {1: [("enc1", "ADD")], 2: [("enc2", "ADD")]}


def test_block_above_the_first_heading_uses_carried_owner():
    # page 2 continues ADD at the top, then starts SUB lower down
    blocks = {1: [(300, "enc1")], 2: [(100, "enc2"), (500, "enc3")]}
    heads = {1: [(200, "ADD")], 2: [(400, "SUB")]}
    assert assign_owners(blocks, heads) == {
        1: [("enc1", "ADD")],
        2: [("enc2", "ADD"), ("enc3", "SUB")],
    }


def test_carry_expires_so_a_stale_owner_cannot_spread():
    """An owner must not propagate indefinitely — mislabelling unrelated blocks
    is worse than leaving them unattributed, since consumers trust attribution."""
    blocks = {1: [(300, "enc1")], 2: [(100, "enc2")], 9: [(100, "far")]}
    heads = {1: [(200, "ADD")]}
    got = assign_owners(blocks, heads, max_carry_pages=2)
    assert got[1] == [("enc1", "ADD")]
    assert got[2] == [("enc2", "ADD")]      # within the carry window
    assert got[9] == [("far", None)]        # expired


def test_block_with_no_heading_anywhere_has_no_owner():
    assert assign_owners({1: [(300, "enc")]}, {}) == {1: [("enc", None)]}

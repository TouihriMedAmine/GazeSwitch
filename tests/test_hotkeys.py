"""Hotkey combo parsing tests (pure logic; no Win32 registration here)."""
from __future__ import annotations

from src.input_hotkeys import MOD_ALT, MOD_CONTROL, MOD_WIN, parse_combo


def test_ctrl_alt_letter():
    assert parse_combo("ctrl+alt+g") == (MOD_CONTROL | MOD_ALT, ord("G"))


def test_win_modifier():
    result = parse_combo("win+shift+k")
    assert result is not None
    mods, vk = result
    assert mods == MOD_WIN | 0x0004
    assert vk == ord("K")


def test_function_keys():
    assert parse_combo("ctrl+f5") == (MOD_CONTROL, 0x70 + 4)


def test_digits():
    assert parse_combo("alt+1") == (MOD_ALT, ord("1"))


def test_invalid_combos_return_none():
    assert parse_combo("") is None
    assert parse_combo("ctrl") is None          # no key
    assert parse_combo("ctrl+alt+bogus") is None
    assert parse_combo("hyper+g") is None

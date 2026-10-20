# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The generic registry: the rules every kind shares (keys, aliases, slots), whatever its entries are."""

from collections.abc import Callable

import pytest

from hpcagent_bench.registry import Kind, RegistryError, Slot


def test_add_returns_the_entry_and_lookup_resolves_keys_and_aliases() -> None:
    kind: Kind[str] = Kind("widgets")
    assert kind.add("a", "Alpha", order=0, aliases=("alpha",)) == "Alpha"
    assert kind["a"] == kind["alpha"] == "Alpha"
    assert "alpha" in kind
    assert "zzz" not in kind
    with pytest.raises(KeyError, match="widgets: 'zzz' is not registered"):
        kind["zzz"]


def test_a_taken_key_alias_or_order_is_refused_and_leaves_the_registry_as_it_was() -> None:
    kind: Kind[str] = Kind("widgets")
    kind.add("a", "A", order=0, aliases=("alpha",))
    for key, order, aliases, message in (
        ("a", 5, (), "registered twice"),
        ("alpha", 5, (), "registered twice"),
        ("b", 0, (), r"already 'a'.s slot"),
        ("b", 1, ("alpha",), "alias 'alpha'"),
        ("b", 1, ("x", "x"), "alias 'x'"),
        ("b", -1, (), "non-negative integer"),
    ):
        with pytest.raises(RegistryError, match=message):
            kind.add(key, "N", order=order, aliases=aliases)
    assert list(kind.entries) == ["a"]
    assert kind.aliases == {"alpha": "a"}


def test_an_entry_that_states_no_slot_takes_the_next_free_one() -> None:
    """A plugin's entry never fails over a colour: it gets the slot after the highest one taken."""
    kind: Kind[str] = Kind("widgets")
    kind.add("shipped", "S", order=7)
    kind.add("plugin", "P")
    kind.add("other", "O", order=Slot.NEXT)
    assert kind.slot("plugin") == 8
    assert kind.slot("other") == 9


def test_import_order_never_decides_the_slot_order() -> None:
    kind: Kind[str] = Kind("widgets")
    for key, order in (("c", 2), ("a", 0), ("b", 1)):
        kind.add(key, key.upper(), order=order)
    assert kind.keys() == ("a", "b", "c")
    assert tuple(kind) == ("a", "b", "c")
    assert kind.next_order() == 3
    assert kind.slot("b") == 1


def test_an_entry_without_a_slot_comes_first_and_does_not_count_toward_the_next_order() -> None:
    kind: Kind[str] = Kind("widgets")
    kind.add("x", "X", order=0)
    kind.add("", "control", order=None)
    assert kind.keys() == ("", "x")
    assert kind.next_order() == 1
    assert kind.slot("") is None


def test_canonical_resolves_an_alias_and_passes_an_unregistered_tag_through() -> None:
    kind: Kind[str] = Kind("widgets")
    kind.add("a", "A", order=0, aliases=("alpha",))
    assert kind.canonical("alpha") == "a"
    assert kind.canonical("zzz") == "zzz"


def test_register_adds_the_decorated_object_unchanged() -> None:
    kind: Kind[Callable[[int], int]] = Kind("functions")

    @kind.register("double")
    def double(value: int) -> int:
        return 2 * value

    assert kind["double"] is double
    assert double(3) == 6
    assert kind.slot("double") == 0


if __name__ == "__main__":
    test_add_returns_the_entry_and_lookup_resolves_keys_and_aliases()
    test_a_taken_key_alias_or_order_is_refused_and_leaves_the_registry_as_it_was()
    test_an_entry_that_states_no_slot_takes_the_next_free_one()
    test_import_order_never_decides_the_slot_order()
    test_an_entry_without_a_slot_comes_first_and_does_not_count_toward_the_next_order()
    test_canonical_resolves_an_alias_and_passes_an_unregistered_tag_through()
    test_register_adds_the_decorated_object_unchanged()

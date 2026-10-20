# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One ordered registry per kind of thing the repository names: models, packets, framework columns, anti-cheat
gates, ... Each is a :class:`Kind` holding typed entries (a frozen dataclass, a NamedTuple, a display name), added
by key::

    MODELS.add("qwen38", ModelEntry("Qwen3.8-27B", "Qwen/Qwen3.8-27B-FP8"), order=0)

An entry validates itself when it is built (its type's ``__post_init__``); the registry checks what spans
entries: a key, an alias or a slot taken twice.

``order`` is an entry's SLOT within its kind -- the colour and marker it is drawn with, the position a column is
listed at. A shipped entry states its slot, so no figure ever repaints when another entry is added
(``tests/test_vocabulary.py`` pins them). An entry a plugin brings takes the next free slot when it is loaded
(:data:`Slot.NEXT`, the default), so a plugin never fails to load over a colour. ``order=None`` registers an entry
with no slot (the no-packet control). Names and aliases share one namespace per kind.

Every kind is documented in ``docs/extending/registry.md``.
"""

import dataclasses
import enum
from collections.abc import Callable, Iterable, Iterator

__all__ = ["Kind", "RegistryError", "Slot"]


class RegistryError(ValueError):
    """An entry or its registration breaks the contract of its kind."""


class Slot(enum.Enum):
    """A slot an entry is registered at without naming one."""

    #: The next free slot of the kind, when the entry is added.
    NEXT = enum.auto()


@dataclasses.dataclass(slots=True)
class Kind[Entry]:
    """The ordered registry of one kind."""

    name: str
    entries: dict[str, Entry] = dataclasses.field(default_factory=dict[str, Entry])
    orders: dict[str, int | None] = dataclasses.field(default_factory=dict[str, int | None])
    aliases: dict[str, str] = dataclasses.field(default_factory=dict[str, str])

    def add(
        self, key: str, entry: Entry, *, order: int | Slot | None = Slot.NEXT, aliases: Iterable[str] = ()
    ) -> Entry:
        """Register ``entry`` under ``key`` at ``order`` and return it; raises when the key, an alias or a stated
        slot is taken."""
        slot = self.next_order() if order is Slot.NEXT else order
        if slot is not None and (isinstance(slot, bool) or slot < 0):
            raise RegistryError(f"{self.name} {key!r}: order must be a non-negative integer or None, got {slot!r}")
        if key in self.entries or key in self.aliases:
            raise RegistryError(f"{self.name}: {key!r} is registered twice")
        if slot is not None and slot in self.orders.values():
            holder = next(name for name, taken in self.orders.items() if taken == slot)
            raise RegistryError(f"{self.name} {key!r}: order {slot} is already {holder!r}'s slot")
        names = tuple(aliases)
        for alias in names:
            if alias in self.entries or alias in self.aliases or alias == key or names.count(alias) > 1:
                raise RegistryError(f"{self.name} {key!r}: alias {alias!r} is already a name or an alias")
        self.entries[key] = entry
        self.orders[key] = slot
        self.aliases.update(dict.fromkeys(names, key))
        return entry

    def register(
        self, key: str, *, order: int | Slot | None = Slot.NEXT, aliases: Iterable[str] = ()
    ) -> Callable[[Entry], Entry]:
        """A decorator adding the decorated object itself as the entry: for kinds whose entries are code (a
        function, a class)."""

        def apply(entry: Entry) -> Entry:
            self.add(key, entry, order=order, aliases=aliases)
            return entry

        return apply

    def next_order(self) -> int:
        """The next free slot: one past the highest registered."""
        return max((order for order in self.orders.values() if order is not None), default=-1) + 1

    def keys(self) -> tuple[str, ...]:
        """The registered keys in slot order; an entry with no slot comes first."""
        return tuple(sorted(self.entries, key=lambda key: -1 if (order := self.orders[key]) is None else order))

    def __iter__(self) -> Iterator[str]:
        """The keys, in :meth:`keys` order."""
        return iter(self.keys())

    def __getitem__(self, key: str) -> Entry:
        """The entry of ``key`` or of the alias ``key``; KeyError naming the kind when neither is registered."""
        try:
            return self.entries[self.canonical(key)]
        except KeyError:
            raise KeyError(f"{self.name}: {key!r} is not registered") from None

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.canonical(key) in self.entries

    def canonical(self, tag: str) -> str:
        """``tag`` with an alias resolved to the key it names; an unregistered tag passes through."""
        return self.aliases.get(tag, tag)

    def slot(self, key: str) -> int | None:
        """The order of a registered key (``None`` for a key without a slot); ``KeyError`` when unregistered."""
        return self.orders[key]

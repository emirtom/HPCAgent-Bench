# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Strong / weak scaling problem-size transforms for the distributed track (pure ``{symbol: value}`` maps).

``strong`` keeps the XL total; ``weak`` grows the decomposition-axis symbols so each rank keeps the 1-node work
(exact at ``R = m**k``, rounded otherwise and corrected by :func:`work_ratio`). No ``work_exponent`` = strong-only."""

from collections.abc import Iterable, Mapping

from hpcagent_bench.config import ScalingLaw
from hpcagent_bench.fuzz import FuzzValue

__all__ = [
    "RANK_BLOCK_QUANTUM",
    "Params",
    "ScalingLaw",
    "aligned_multiple",
    "aligned_symbols",
    "extent",
    "integer_kth_root",
    "sized_params",
    "strong",
    "weak",
    "weak_rounding_note",
    "work_ratio",
]

#: Every rank's block of an ALIGNED split size symbol is a multiple of this many elements at every graded rank count.
RANK_BLOCK_QUANTUM: int = 64


type Params = Mapping[str, FuzzValue]


def extent(params: Params, symbol: str) -> int:
    """The integer size of ``symbol`` in ``params``; a size symbol that is not a number is a manifest error."""
    value = params[symbol]
    if isinstance(value, (int, float, str)):
        return int(value)
    raise ValueError(f"size symbol {symbol!r} is {value!r}, not a number")


def strong(params: Params) -> dict[str, FuzzValue]:
    """Strong scaling: the total problem is fixed, so size is unchanged (a fresh dict)."""
    return dict(params)


def integer_kth_root(value: int, k: int) -> int | None:
    """The exact integer ``k``-th root of ``value``, or ``None`` (integer search: the ``R = m**k`` test must be
    exact, not float-close)."""
    if value < 1 or k < 1:
        return None
    lo, hi = 1, value
    while lo < hi:
        mid = (lo + hi) // 2
        if mid**k < value:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo**k == value else None


def aligned_multiple(value: float, quantum: int) -> int:
    """``value`` rounded to the NEAREST positive multiple of ``quantum`` (never below one quantum)."""
    return max(1, round(value / quantum)) * quantum


def weak(
    params: Params,
    axis_symbols: Iterable[str],
    ranks: int,
    work_exponent: int | None = None,
    aligned: Iterable[str] = (),
) -> dict[str, FuzzValue]:
    """Weak scaling: grow the total problem with ``ranks`` so each rank keeps the 1-node XL work.

    At ``R = m**k`` every decomposition-axis symbol is multiplied by ``m`` exactly; otherwise by ``R ** (1/k)``,
    rounded per symbol (:func:`work_ratio` recovers the realized ratio). ``work_exponent`` None or < 1 raises
    ``ValueError``, never defaulted. Symbols in ``aligned`` snap to multiples of ``RANK_BLOCK_QUANTUM * ranks``."""
    if work_exponent is None:
        raise ValueError(
            "weak scaling needs mpi.decomposition.work_exponent, the degree k of the work in the "
            "decomposition axis; the manifest declares none, so the kernel is strong-only"
        )
    k = int(work_exponent)
    if k < 1:
        raise ValueError(f"weak scaling needs a work_exponent k >= 1; the manifest declares k={k}")
    r = max(1, int(ranks))
    m = integer_kth_root(r, k)
    scaled = dict(params)
    for sym in set(axis_symbols):
        if sym not in scaled:
            continue
        if m is not None:
            scaled[sym] = extent(params, sym) * m
        else:
            # P is not a k-th power: the grown extent is rounded and eta corrects by the work ratio.
            scaled[sym] = max(1, round(extent(params, sym) * r ** (1.0 / k)))
    for sym in set(aligned) & set(axis_symbols):
        if sym in scaled:
            scaled[sym] = aligned_multiple(extent(scaled, sym), RANK_BLOCK_QUANTUM * r)
    return scaled


def work_ratio(base_params: Params, grown_params: Params, axis_symbols: Iterable[str], work_exponent: int) -> float:
    """The REALIZED work ratio ``W(N_P)/W(N_1)`` from the actual (rounded) per-symbol sizes.

    Exact for work homogeneous of degree ``k`` and symmetric over its ``d`` axis symbols; 1.0 with none declared,
    exactly ``P`` at ``P = m**k``."""
    axes = sorted(s for s in set(axis_symbols) if s in base_params and s in grown_params)
    if not axes:
        return 1.0
    k = int(work_exponent)
    d = len(axes)
    ratio = 1.0
    for sym in axes:
        base = extent(base_params, sym)
        if base <= 0:
            raise ValueError(f"work_ratio needs a positive base size for {sym!r}; got {base}")
        ratio *= extent(grown_params, sym) / base
    return ratio ** (k / d)


def weak_rounding_note(
    base_params: Params,
    grown_params: Params,
    axis_symbols: Iterable[str],
    ranks: int,
    work_exponent: int,
) -> str | None:
    """The disclosure for a weak size :func:`weak` rounded, or ``None`` at ``P = m**k``."""
    r, k = max(1, int(ranks)), int(work_exponent)
    m = integer_kth_root(r, k)
    axes = sorted(sym for sym in set(axis_symbols) if sym in grown_params and sym in base_params)
    if m is not None and all(extent(grown_params, sym) == extent(base_params, sym) * m for sym in axes):
        return None
    sizes = {sym: grown_params[sym] for sym in sorted(set(axis_symbols)) if sym in grown_params}
    ratio = work_ratio(base_params, grown_params, axis_symbols, k)
    why = "not a perfect k-th power; rounded" if m is None else f"aligned to {RANK_BLOCK_QUANTUM} per rank"
    return f"P={r}: k={k}, m={r ** (1.0 / k):.3f} -> sizes {sizes}, work ratio {ratio:.2f} ({why})"


def sized_params(
    params: Params,
    mode: ScalingLaw,
    axis_symbols: Iterable[str],
    ranks: int,
    work_exponent: int | None = None,
    aligned: Iterable[str] = (),
) -> dict[str, FuzzValue]:
    """Dispatch ``mode`` to the matching transform (strong ignores ``work_exponent`` and ``aligned``)."""
    match mode:
        case ScalingLaw.STRONG:
            return strong(params)
        case ScalingLaw.WEAK:
            return weak(params, axis_symbols, ranks, work_exponent, aligned)


def aligned_symbols(mpi: dict[str, object] | None) -> frozenset[str]:
    """The split size symbols the ML track keeps per-rank blocks of a multiple of :data:`RANK_BLOCK_QUANTUM`,
    minus ``mpi.rank_block_exempt``."""
    block = mpi or {}
    split = block.get("split") or {}
    exempt_raw = block.get("rank_block_exempt")
    exempt = {str(s) for s in exempt_raw} if isinstance(exempt_raw, (list, tuple)) else set()
    if not isinstance(split, dict):
        return frozenset()
    return frozenset(str(v) for v in split.values() if v is not None) - exempt

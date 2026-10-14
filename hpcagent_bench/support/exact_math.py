# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Elementary functions and reductions that give the same bits on numpy and on cupy, for kernel initializers.

glibc and a GPU's math library round ``exp``, ``log``, ``sin`` and ``pow`` differently in the last place, so an
initializer that calls them builds different inputs on the host and on the device. These versions use only
operations IEEE 754 rounds exactly the same everywhere -- ``+ - * /``, ``sqrt``, ``rint``, ``frexp``, ``ldexp``,
comparisons and ``where`` -- each as its own array operation, so no compiler fuses two of them into an FMA.
They are accurate to a few ulp of float64, which is all an input distribution needs. Arguments are converted to
float64 first; a float32 result is the float64 one rounded once.

A floating reduction is summed in an order the array library chooses (numpy's pairwise blocks, a GPU's tree), so
:func:`sum`, :func:`cumsum`, :func:`matvec` and :func:`norm` fix the order themselves: a binary tree of whole-array
additions (halves added elementwise until one row is left), and a Hillis-Steele scan. An integer or boolean
reduction is exact in any order and runs as the library's own.
"""

import math
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "EXP_LIMIT",
    "EXPM1_SERIES_LIMIT",
    "cos",
    "cumsum",
    "exp",
    "expm1",
    "ipow",
    "linspace",
    "log",
    "logspace",
    "matmul",
    "matvec",
    "mean",
    "norm",
    "pow",
    "sin",
    "sum",
    "tanh",
]

type Array = NDArray[Any]

#: ``expm1`` takes its own Taylor series below this magnitude, where ``exp(x) - 1`` would cancel.
EXPM1_SERIES_LIMIT = 0.25
#: ``exp`` saturates beyond this: float64 overflows above about 709.78 and underflows below about -745.
EXP_LIMIT = 708.0
#: ``ln 2`` split so that ``k * LN2_HI`` is exact for every ``|k| < 2**11``.
LN2_HI = 6.93147180369123816490e-01
LN2_LO = 1.90821492927058770002e-10
INV_LN2 = 1.0 / math.log(2.0)
#: ``pi / 2`` in three parts (Cody-Waite), exact products for ``|k| < 2**20``.
HALF_PI_1 = 1.5707963267341256e00
HALF_PI_2 = 6.077100506506192e-11
HALF_PI_3 = 2.0222662487959506e-21
INV_HALF_PI = 2.0 / math.pi
#: Taylor terms of ``exp`` on ``|r| <= ln2 / 2``, ``sin``/``cos`` on ``|r| <= pi / 4``, and ``atanh`` on
#: ``|s| <= 0.172`` (``log`` of a mantissa in ``[sqrt(1/2), sqrt(2))``): each truncation is below 2**-56.
EXP_TERMS = 14
TRIG_TERMS = 11
ATANH_TERMS = 12


def as_f64(x: Any, xp: ModuleType) -> Array:
    """``x`` as a float64 array of ``xp``; a complex ``x`` is refused rather than silently made real."""
    value = xp.asarray(x)
    if value.dtype.kind == "c":
        raise TypeError("exact_math: a complex argument (only exp takes one)")
    return value.astype(xp.float64, copy=False)


def horner(r: Array, coefficients: list[float], xp: ModuleType) -> Array:
    """``sum(c_i r**i)`` by Horner's rule, one exact multiply and one exact add per term."""
    total = xp.full(r.shape, coefficients[-1], dtype=xp.float64)
    for coefficient in reversed(coefficients[:-1]):
        total = total * r
        total = total + coefficient
    return total


def exp(x: Any, xp: ModuleType = np) -> Array:
    """``e**x``: ``x = k ln2 + r``, ``exp(r)`` by its Taylor series, then ``ldexp`` by ``k``. A complex ``x = a + ib``
    gives ``e**a (cos b + i sin b)`` as complex128."""
    value = xp.asarray(x)
    if value.dtype.kind == "c":
        magnitude = exp(value.real, xp)
        return magnitude * cos(value.imag, xp) + 1j * (magnitude * sin(value.imag, xp))
    value = xp.clip(as_f64(value, xp), -EXP_LIMIT, EXP_LIMIT)
    k = xp.rint(value * INV_LN2)
    r = value - k * LN2_HI
    r = r - k * LN2_LO
    series = horner(r, [1.0 / math.factorial(n) for n in range(EXP_TERMS)], xp)
    return xp.ldexp(series, k.astype(xp.int32))


def expm1(x: Any, xp: ModuleType = np) -> Array:
    """``e**x - 1``, without the cancellation of ``exp(x) - 1`` near 0 (the Taylor series ``x + x**2/2 + ...``
    there)."""
    value = as_f64(x, xp)
    series = value * horner(value, [1.0 / math.factorial(n + 1) for n in range(EXP_TERMS)], xp)
    return xp.where(xp.abs(value) < EXPM1_SERIES_LIMIT, series, exp(value, xp) - 1.0)


def log(x: Any, xp: ModuleType = np) -> Array:
    """Natural logarithm of a positive ``x``: ``x = m 2**e`` with ``m`` in ``[sqrt(1/2), sqrt(2))``, then
    ``log m = 2 atanh((m - 1) / (m + 1))``. ``0`` gives ``-inf``; a negative ``x`` gives ``nan``."""
    value = as_f64(x, xp)
    # A non-positive x takes the series at 1.0, whose result the last two lines replace.
    mantissa, exponent = xp.frexp(xp.where(value > 0.0, value, 1.0))
    low = mantissa < math.sqrt(0.5)
    mantissa = xp.where(low, mantissa * 2.0, mantissa)
    exponent = xp.where(low, exponent - 1, exponent).astype(xp.float64)
    s = (mantissa - 1.0) / (mantissa + 1.0)
    s2 = s * s
    series = horner(s2, [1.0 / (2 * n + 1) for n in range(ATANH_TERMS)], xp)
    result = 2.0 * s * series
    result = result + exponent * LN2_LO
    result = result + exponent * LN2_HI
    result = xp.where(value == 0.0, -xp.inf, result)
    return xp.where(value < 0.0, xp.nan, result)


def reduce_half_pi(x: Any, xp: ModuleType) -> tuple[Array, Array]:
    """``x = k pi/2 + r`` with ``|r| <= pi/4``: ``r`` and ``k mod 4`` (as float64)."""
    value = as_f64(x, xp)
    k = xp.rint(value * INV_HALF_PI)
    r = value - k * HALF_PI_1
    r = r - k * HALF_PI_2
    r = r - k * HALF_PI_3
    return r, k - 4.0 * xp.floor(k * 0.25)


def sin_cos_series(r: Array, xp: ModuleType) -> tuple[Array, Array]:
    """``sin r`` and ``cos r`` for ``|r| <= pi/4`` by their Taylor series in ``r**2``."""
    r2 = r * r
    sin_terms = [(-1.0) ** n / math.factorial(2 * n + 1) for n in range(TRIG_TERMS)]
    cos_terms = [(-1.0) ** n / math.factorial(2 * n) for n in range(TRIG_TERMS)]
    return r * horner(r2, sin_terms, xp), horner(r2, cos_terms, xp)


def sin(x: Any, xp: ModuleType = np) -> Array:
    """``sin x``, for ``|x| < 2**20`` (larger arguments lose the reduction's exactness)."""
    r, quadrant = reduce_half_pi(x, xp)
    s, c = sin_cos_series(r, xp)
    return xp.where(quadrant == 0.0, s, xp.where(quadrant == 1.0, c, xp.where(quadrant == 2.0, -s, -c)))


def cos(x: Any, xp: ModuleType = np) -> Array:
    """``cos x``, for ``|x| < 2**20``."""
    r, quadrant = reduce_half_pi(x, xp)
    s, c = sin_cos_series(r, xp)
    return xp.where(quadrant == 0.0, c, xp.where(quadrant == 1.0, -s, xp.where(quadrant == 2.0, -c, s)))


def tanh(x: Any, xp: ModuleType = np) -> Array:
    """``tanh x = (e**2x - 1) / (e**2x + 1)``, with the sign folded so the exponential never overflows."""
    value = as_f64(x, xp)
    e = exp(-2.0 * xp.abs(value), xp)
    magnitude = (1.0 - e) / (1.0 + e)
    return xp.where(value < 0.0, -magnitude, magnitude)


def ipow(x: Any, n: int, xp: ModuleType = np) -> Array:
    """``x**n`` for a non-negative integer ``n`` by repeated squaring (``x**2`` is one multiply)."""
    if n < 0:
        raise ValueError(f"ipow: negative exponent {n}; divide by ipow(x, {-n})")
    base = xp.asarray(x)
    result = xp.ones_like(base)
    while n:
        if n & 1:
            result = result * base
        n >>= 1
        if n:
            base = base * base
    return result


def pow(x: Any, y: Any, xp: ModuleType = np) -> Array:  # noqa: A001 -- the function's own name in every math library
    """``x**y`` for a positive ``x``: ``exp(y log x)``."""
    return exp(as_f64(y, xp) * log(x, xp), xp)


def linspace(start: float, stop: float, num: int, endpoint: bool = True, xp: ModuleType = np) -> Array:
    """``num`` evenly spaced float64 values from ``start``: ``start + (stop - start) * (i / d)`` with ``d`` the
    number of intervals, one exact operation at a time (numpy and cupy compute ``linspace`` differently). The last
    value is ``stop`` exactly when ``endpoint``."""
    count = int(num)
    intervals = max(count - 1, 1) if endpoint else max(count, 1)
    fraction = xp.arange(count, dtype=xp.float64) / intervals
    values = start + (stop - start) * fraction
    if endpoint and count > 1:
        values[-1] = stop
    return values


def logspace(start: float, stop: float, num: int, base: float = 10.0, xp: ModuleType = np) -> Array:
    """``base ** linspace(start, stop, num)``."""
    return pow(xp.full(int(num), base, dtype=xp.float64), linspace(start, stop, num, xp=xp), xp)


def exact_in_any_order(values: Array) -> bool:
    """Whether a reduction of ``values`` is exact whatever its order: integers and booleans."""
    return values.dtype.kind in "biu"


def sum(x: Any, axis: int | None = None, keepdims: bool = False, xp: ModuleType = np) -> Array:  # noqa: A001 -- numpy's name
    """``x`` summed over ``axis`` (every element for ``None``) in a fixed binary-tree order."""
    values = xp.asarray(x)
    if exact_in_any_order(values):
        return values.sum(axis=axis, keepdims=keepdims)
    if axis is None:
        flat = sum(values.reshape(-1), 0, False, xp)
        return flat.reshape((1,) * values.ndim) if keepdims else flat
    axis = axis % values.ndim
    rows = xp.moveaxis(values, axis, 0)
    if rows.shape[0] == 0:
        total = xp.zeros(rows.shape[1:], dtype=values.dtype)
    else:
        while rows.shape[0] > 1:
            half = rows.shape[0] // 2
            paired = rows[:half] + rows[half : 2 * half]
            rows = xp.concatenate((paired, rows[2 * half :]), axis=0) if rows.shape[0] % 2 else paired
        total = rows[0]
    return xp.expand_dims(total, axis) if keepdims else total


def mean(x: Any, axis: int | None = None, keepdims: bool = False, xp: ModuleType = np) -> Array:
    """:func:`sum` divided by the count, as float64 for an integer ``x``."""
    values = xp.asarray(x)
    count = values.size if axis is None else values.shape[axis]
    total = sum(values.astype(xp.float64) if exact_in_any_order(values) else values, axis, keepdims, xp)
    return total / max(count, 1)


def cumsum(x: Any, axis: int | None = None, xp: ModuleType = np) -> Array:
    """Running sum over ``axis`` (the flattened ``x`` for ``None``) by a Hillis-Steele scan: step ``d`` adds the
    element ``d`` back to every element, for ``d = 1, 2, 4, ...``."""
    values = xp.asarray(x)
    if exact_in_any_order(values):
        return xp.cumsum(values, axis=axis)
    if axis is None:
        values, axis = values.reshape(-1), 0
    rows = xp.moveaxis(values, axis % values.ndim, 0).copy()
    step = 1
    while step < rows.shape[0]:
        shifted = rows[:-step].copy()
        rows[step:] = rows[step:] + shifted
        step *= 2
    return xp.moveaxis(rows, 0, axis % values.ndim)


def matvec(a: Any, x: Any, xp: ModuleType = np) -> Array:
    """``a @ x`` for a matrix ``a`` and a vector ``x``: each row's products summed by :func:`sum`."""
    return sum(xp.asarray(a) * xp.asarray(x)[None, :], 1, False, xp)


def norm(x: Any, axis: int | None = None, keepdims: bool = False, xp: ModuleType = np) -> Array:
    """The Euclidean norm over ``axis``: ``sqrt`` of the :func:`sum` of squares (``|x|**2`` for complex)."""
    values = xp.asarray(x)
    squares = (values * values.conj()).real if values.dtype.kind == "c" else values * values
    return xp.sqrt(sum(squares, axis, keepdims, xp))


def matmul(a: Any, b: Any, xp: ModuleType = np) -> Array:
    """``a @ b`` for matrices: every product formed, then summed over the inner axis by :func:`sum`. Memory is
    ``m k n``, so this is for the small products of an initializer (a few projectors, three coordinates)."""
    left, right = xp.asarray(a), xp.asarray(b)
    return sum(left[:, :, None] * right[None, :, :], 1, False, xp)

# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Elementary functions that give the same bits on numpy and on cupy, for kernel initializers.

glibc and a GPU's math library round ``exp``, ``log``, ``sin`` and ``pow`` differently in the last place, so an
initializer that calls them builds different inputs on the host and on the device. These versions use only
operations IEEE 754 rounds exactly the same everywhere -- ``+ - * /``, ``sqrt``, ``rint``, ``frexp``, ``ldexp``,
comparisons and ``where`` -- each as its own array operation, so no compiler fuses two of them into an FMA.
They are accurate to a few ulp of float64, which is all an input distribution needs. Arguments are converted to
float64 first; a float32 result is the float64 one rounded once.
"""

import math
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "EXP_LIMIT",
    "cos",
    "exp",
    "ipow",
    "log",
    "pow",
    "sin",
    "tanh",
]

type Array = NDArray[Any]

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
    """``x`` as a float64 array of ``xp``."""
    return xp.asarray(x, dtype=xp.float64)


def horner(r: Array, coefficients: list[float], xp: ModuleType) -> Array:
    """``sum(c_i r**i)`` by Horner's rule, one exact multiply and one exact add per term."""
    total = xp.full(r.shape, coefficients[-1], dtype=xp.float64)
    for coefficient in reversed(coefficients[:-1]):
        total = total * r
        total = total + coefficient
    return total


def exp(x: Any, xp: ModuleType = np) -> Array:
    """``e**x``: ``x = k ln2 + r``, ``exp(r)`` by its Taylor series, then ``ldexp`` by ``k``."""
    value = xp.clip(as_f64(x, xp), -EXP_LIMIT, EXP_LIMIT)
    k = xp.rint(value * INV_LN2)
    r = value - k * LN2_HI
    r = r - k * LN2_LO
    series = horner(r, [1.0 / math.factorial(n) for n in range(EXP_TERMS)], xp)
    return xp.ldexp(series, k.astype(xp.int32))


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

# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""The harness's own precision casts for ``requests.precision``: an array is converted from the problem
precision into the requested one before every call and back after it, outside the score.

One conversion table serves every backend, so a cast is the same bits wherever it runs: the storage-only
formats (bf16, the fp8 pair) convert through float with the emitter's ``FP8_HELPERS`` (bit-exact against
ml_dtypes), the native ones by a C cast. The host runs them as one OpenMP loop per array from a library
built once per OpenMP context, so the timed process maps the submission's own runtime and no other
(:func:`host_library`); a device grade runs one cupy elementwise kernel per array (:func:`device_cast`).
An A/B on MI300A put threaded numpy at 26-32 GB/s against 150-200 GB/s for the OpenMP loop, and cupy at
2.4 TB/s.
"""

import ctypes
import functools
import hashlib
import os
import pathlib
import shutil
import subprocess
import tempfile
import types
from dataclasses import dataclass

import numpy as np

from hpcagent_bench import languages, omp_context
from hpcagent_bench.dtypes import c_type, canonical, is_storage_only
from hpcagent_bench.flags import Mode
from hpcagent_bench.harness import disk_cache
from hpcagent_bench.translators.numpyto_c.emit import FP8_HELPERS

__all__ = [
    "CAST_DTYPES",
    "CastPlan",
    "HostCasts",
    "cast_expr",
    "device_cast",
    "host_function",
    "host_library",
    "host_source",
    "prelude",
    "storage_view_dtype",
    "symbol",
]

#: Every element type a cast reads or writes: the :class:`~hpcagent_bench.precision.Precision` realizations.
CAST_DTYPES = ("float64", "float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2")
#: The ``__npb_<short>_to_f32`` / ``__npb_f32_to_<short>`` helper name of each storage-only format.
STORAGE_SHORT = {"bfloat16": "bf16", "float8_e4m3fn": "e4m3", "float8_e5m2": "e5m2"}
#: OpenMP context -> the toolchain family whose C compiler builds the host casts in it.
CONTEXT_FAMILY = {context: family for family, context in omp_context.FAMILY_CONTEXT.items()}


@dataclass(frozen=True, slots=True)
class CastPlan:
    """The casts one grade's calls run: per requested array its target dtype name, and the host cast
    library (:func:`host_library`) a host call loads; ``host_library`` is empty on a device grade."""

    targets: tuple[tuple[str, str], ...]
    host_library: str = ""

    def target(self, name: str) -> str | None:
        """The dtype ``name`` is cast to, or None when it is not requested."""
        return dict(self.targets).get(name)


def symbol(src: str, dst: str) -> str:
    """The C symbol of the ``src`` -> ``dst`` cast."""
    return f"hb_cast_{src}_{dst}"


def prelude(dtypes: tuple[str, ...] = CAST_DTYPES) -> str:
    """The storage typedefs and float conversions of every storage-only format in ``dtypes``."""
    return "".join(FP8_HELPERS[canonical(dt)].format(ct=c_type(dt)) for dt in dtypes if is_storage_only(dt))


def cast_expr(src: str, dst: str, value: str, half: str = "_Float16") -> str:
    """The C expression converting ``value`` (a ``src`` element) to a ``dst`` element. Storage-only formats
    go through float; ``half`` spells the native 2-byte float (cupy's ``float16`` on a device)."""
    if is_storage_only(src):
        value = f"__npb_{STORAGE_SHORT[src]}_to_f32({value})"
    if is_storage_only(dst):
        return f"__npb_f32_to_{STORAGE_SHORT[dst]}((float)({value}))"
    return f"({half if dst == 'float16' else c_type(dst)})({value})"


def host_function(src: str, dst: str) -> str:
    """The C function of the ``src`` -> ``dst`` cast: one OpenMP loop."""
    return (
        f"static void {symbol(src, dst)}(const {c_type(src)} *restrict in, {c_type(dst)} *restrict out, int64_t n) {{\n"
        f"#pragma omp parallel for simd schedule(static)\n"
        f"    for (int64_t i = 0; i < n; ++i) out[i] = {cast_expr(src, dst, 'in[i]')};\n"
        f"}}\n"
    )


def host_source() -> str:
    """The host cast library: one exported OpenMP loop per ordered pair of :data:`CAST_DTYPES`."""
    loops = (
        host_function(src, dst).removeprefix("static ") for src in CAST_DTYPES for dst in CAST_DTYPES if src != dst
    )
    return "#include <stdint.h>\n#include <string.h>\n" + prelude() + "".join(loops)


@functools.lru_cache(maxsize=4, typed=True)
def built_library(context: str) -> pathlib.Path:
    """The host cast library of OpenMP ``context``, built once per source and context and cached."""
    source = host_source()
    digest = hashlib.sha256(f"{context}\0{source}".encode()).hexdigest()[:16]
    cached = disk_cache.root() / "precision_casts" / f"hb_casts_{context or 'default'}_{digest}.so"
    if cached.is_file():
        return cached
    family = CONTEXT_FAMILY.get(context or omp_context.DEFAULT_CONTEXT, "gcc")
    compiler = languages.compiler_for_family("c", family)
    cached.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hb_casts_") as work:
        src = pathlib.Path(work, "hb_casts.c")
        src.write_text(source, encoding="utf-8")
        out = pathlib.Path(work, cached.name)
        env = {**os.environ, **omp_context.context_build_env(context)}
        for argv in languages.build_shared_lib_commands("c", src, out, mode=Mode.MULTI_CORE, compiler=compiler):
            subprocess.run(argv, check=True, capture_output=True, env=env, cwd=work)
        staged = cached.with_suffix(f".{os.getpid()}.tmp")
        shutil.copy2(out, staged)
        staged.replace(cached)  # a concurrent grade sees the whole library or none
    return cached


def host_library(context: str, beside: pathlib.Path) -> pathlib.Path:
    """The host cast library of ``context``, copied next to the submission's library ``beside`` so the
    sealed child (which keeps that directory) can load it."""
    target = beside.parent / built_library(context).name
    if not target.is_file():
        shutil.copy2(built_library(context), target)
    return target


class HostCasts:
    """The host cast library loaded in the grading child; :meth:`cast` converts one array in place of a copy."""

    def __init__(self, library: str) -> None:
        self.lib = ctypes.CDLL(library)

    def convert(self, src: np.ndarray, dst: np.ndarray, src_dtype: str, dst_dtype: str) -> None:
        """``dst[...] = src`` converted, both contiguous and of one size."""
        fn = self.lib[symbol(src_dtype, dst_dtype)]
        fn.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)
        fn.restype = None
        fn(src.ctypes.data, dst.ctypes.data, src.size)


def storage_view_dtype(dtype: str) -> np.dtype:
    """The dtype a device buffer of ``dtype`` is held in: cupy has no bf16/fp8, so their bits ride in
    same-width unsigned integers."""
    if is_storage_only(dtype):
        return np.dtype(f"uint{8 * np.dtype(dtype).itemsize}")
    return np.dtype(dtype)


@functools.lru_cache(maxsize=64, typed=True)
def device_kernel(cp: types.ModuleType, src: str, dst: str) -> object:
    """The cupy elementwise kernel of the ``src`` -> ``dst`` cast (compiled on first use, cached by cupy)."""
    in_t, out_t = storage_view_dtype(src).name, storage_view_dtype(dst).name
    # The helpers spell ``memcpy``; hipRTC and NVRTC take the builtin.
    preamble = "#define memcpy __builtin_memcpy\n" + prelude((src, dst))
    return cp.ElementwiseKernel(
        f"{in_t} x",
        f"{out_t} y",
        f"y = {cast_expr(src, dst, 'x', half='float16')};",
        symbol(src, dst),
        preamble=preamble,
    )


def device_cast(cp: types.ModuleType, src: object, dst: object, src_dtype: str, dst_dtype: str) -> None:
    """``dst[...] = src`` converted on the device; the dtypes are the logical element types (a storage-only
    buffer is an unsigned view, :func:`storage_view_dtype`)."""
    kernel = device_kernel(cp, src_dtype, dst_dtype)
    kernel(src, dst)  # type: ignore[operator]

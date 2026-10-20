"""Unified ``numpyto --target <lang> ...`` driver.

A thin dispatcher: ``--target`` picks the backend, and every other argument is
passed straight through to that backend's ``emit`` sub-command. So

    numpyto --target c        --kernel k_numpy.py --bench-info k.json --out DIR
    numpyto --target polly    --kernel ... --bench-info ... --out ...
    numpyto --target pluto    --kernel ... --bench-info ... --out ...
    numpyto --target fortran  --kernel ... --bench-info ... --out ...
    numpyto --target numba    --kernel ... --out ... [--fastmath] [--sanitize]

are equivalent to invoking each per-package CLI directly (the regen scripts call
those); this is the single front door over them.

``polly`` and ``pluto`` are the C-family polyhedral targets: a single
``numpyto_c`` emit already writes the C source, the C++ source, AND the
``#pragma scop``-wrapped Pluto input for the *whole* kernel, so all three share
one backend. ``polly`` reuses the same C/C++ source (Polly is a compile-flag
variant -- nothing changes in emission); ``pluto`` is the polycc input that
emit produces. They are distinct front-door names because the runtime exposes
them as flag presets.
"""

import argparse
import importlib
import sys
from collections.abc import Sequence

__all__ = ["PARALLEL_TARGETS", "TARGETS", "main"]

#: target name -> backend CLI module exposing ``main(argv)``. The C backend
#: backs three targets -- ``c`` / ``polly`` / ``pluto`` -- because one emit
#: produces the whole C-family (C, C++, and the Pluto ``#pragma scop`` input).
TARGETS = {
    "c": "hpcagent_bench.translators.numpyto_c.cli",
    "polly": "hpcagent_bench.translators.numpyto_c.cli",
    "pluto": "hpcagent_bench.translators.numpyto_c.cli",
    "fortran": "hpcagent_bench.translators.numpyto_fortran.cli",
    "numba": "hpcagent_bench.translators.numpyto_numba.cli",
    # OpenMP parallel-scope variants: same backend, ``--parallel`` injected. One C
    # emit produces both the C and C++ OpenMP sources, so ``c_omp`` and ``cpp_omp``
    # share the C backend (like ``c``/``polly``/``pluto``).
    "c_omp": "hpcagent_bench.translators.numpyto_c.cli",
    "cpp_omp": "hpcagent_bench.translators.numpyto_c.cli",
    "fortran_omp": "hpcagent_bench.translators.numpyto_fortran.cli",
}

#: Targets that inject ``--parallel`` into the backend emit (OpenMP variants).
PARALLEL_TARGETS = {"c_omp", "cpp_omp", "fortran_omp"}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog="numpyto", description="Unified NumpyToX emitter: emit a numpy kernel to a target language."
    )
    ap.add_argument("--target", "-t", required=True, choices=sorted(TARGETS), help="output language / framework")
    args, rest = ap.parse_known_args(argv)
    mod = importlib.import_module(TARGETS[args.target])
    if args.target in PARALLEL_TARGETS and "--parallel" not in rest:
        rest = ["--parallel", *rest]
    return mod.main(["emit", *rest])


if __name__ == "__main__":
    sys.exit(main())

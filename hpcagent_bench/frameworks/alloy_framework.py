# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Alloy (github.com/ThrudPrimrose/alloy) as a DaCe code generation backend: the flavor's DaCe pipeline gives the SDFG,
Alloy outlines it into regions, builds, verifies and times every region's candidates locally, and DaCe compiles the
program linking each region's winner. No judge and no ``/score``; the ``alloy-par`` / ``alloy-canon`` optimizers are
the judged path. Alloy is an optional install, imported on first use."""

import dace

from hpcagent_bench.frameworks import Benchmark
from hpcagent_bench.frameworks.dace_framework import DaceFramework, TimedCompiledSDFG
from hpcagent_bench.frameworks.framework import BenchData, KernelImpl, OutputValue
from hpcagent_bench.precision import precision_from_datatype
from hpcagent_bench.spec import BenchSpec

__all__ = ["AlloyFramework"]


class AlloyFramework(DaceFramework):
    """A DaCe flavor whose pipeline output is compiled by Alloy instead of DaCe's default CPU code generator."""

    __slots__ = ("spec",)

    def __init__(self, fname: str) -> None:
        super().__init__(fname)
        #: The kernel :meth:`compile_variants` hands Alloy (its manifest gives Alloy's verify and timing inputs).
        self.spec: BenchSpec | None = None

    def optimize(self, program: KernelImpl, bench: Benchmark, bdata: BenchData) -> TimedCompiledSDFG:
        self.spec = bench.spec
        return super().optimize(program, bench, bdata)

    def compile_variants(self, sdfgs: dict[str, dace.SDFG]) -> dict[str, TimedCompiledSDFG]:
        """Alloy's program for this flavor's pipeline output; a pipeline Alloy fails on is dropped with its reason."""
        from alloy import bench as alloy_bench  # pyright: ignore[reportMissingImports] -- Alloy is an optional install

        if self.spec is None:
            raise RuntimeError(f"{self.fname}: compile_variants runs inside optimize")
        precision = precision_from_datatype(self.datatype)
        compiled: dict[str, TimedCompiledSDFG] = {}
        for name, sdfg in sdfgs.items():
            try:
                program, dc_exec = alloy_bench.compiled_winners(sdfg, self.spec, precision)
            except Exception as exc:  # noqa: BLE001 -- a pipeline Alloy cannot build is recorded and skipped
                print(f"Alloy on {name} failed: {exc}")
                self._pipeline_errors.append(f"alloy {name}: {type(exc).__name__}: {exc}")
                continue
            compiled[name] = TimedCompiledSDFG(dc_exec, program, f"{name}_alloy")
        return compiled

    def strict_fp_or(
        self,
        name: str,
        fallback: TimedCompiledSDFG,
        sdfg: dace.SDFG,
        reference: list[OutputValue],
        bench: Benchmark,
        bdata: BenchData,
    ) -> TimedCompiledSDFG:
        """Alloy already verified every region at the FP level it built it with; a DaCe-side strict rebuild would
        recompile only the caller, not the region archives, so the verified program stands."""
        print(f"Alloy optimize: {name!r} failed the bench verify; Alloy's region builds are kept")
        return fallback

# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The framework columns: every way a kernel is built and run, one :class:`~hpcagent_bench.vocabulary.Column` each.

A name is ``<column>[_<flavor>]``: the column is ``<backend>`` or ``<backend>_<arch>``, a flavor (``column`` and
``flavor`` set) is the optimizer run on it. ``adapter`` names what runs a column, imported on first use. Registration
order is the listing order and each column's slot (colour and marker): a new column is added last.
``docs/extending/registry.md`` is the authoring guide.
"""

from hpcagent_bench.languages import gpu_backend
from hpcagent_bench.precision import Precision
from hpcagent_bench.vocabulary import FRAMEWORKS, TUNERS, Arch, Column, DaceStage, Transform, TunerEntry

__all__ = [
    "ALL_PRECISIONS",
    "DACE_PRECISIONS",
    "DACE_STAGES",
    "FRAMEWORKS",
    "IEEE_PRECISIONS",
    "add_dace",
    "dace_column",
]

#: The IEEE pair every non-ml_dtypes-aware framework can execute (C/C++/Fortran, Numba, Pythran).
IEEE_PRECISIONS = frozenset({Precision.FP32, Precision.FP64})

#: The full precision matrix (IEEE + fp16/bf16/fp8), for frameworks carrying low precision end to end.
ALL_PRECISIONS = frozenset(
    {
        Precision.FP64,
        Precision.FP32,
        Precision.FP16,
        Precision.BF16,
        Precision.FP8_E4M3,
        Precision.FP8_E5M2,
    }
)


#: A DaCe stage -> its label alone, its label beside a tuner, and its full-name word.
DACE_STAGES: dict[DaceStage, tuple[str, str, str]] = {
    DaceStage.CANON: ("canonicalized", "canon", "canonicalize"),
    DaceStage.PARALLEL: ("parallel", "parallel", "parallel"),
    DaceStage.AUTOOPT: ("auto-opt", "auto-opt", "auto_optimize"),
}
TUNERS.add(
    "alloy_profile",
    TunerEntry("Alloy profile", "Alloy (profiled)", "hpcagent_bench.harness.optimizers:AlloyProfileOptimizer"),
)
TUNERS.add(
    "alloy_heuristic",
    TunerEntry("Alloy heuristic", "Alloy (heuristic)", "hpcagent_bench.harness.optimizers:AlloyHeuristicOptimizer"),
)
DACE_PRECISIONS = frozenset({Precision.FP64, Precision.FP32, Precision.FP16})


def dace_column(arch: Arch, stage: DaceStage | None = None, tuner: str | None = None) -> Column:
    """The DaCe family's entry on ``arch``: the DB column every flavor groups under (no ``stage``; it runs nothing
    itself), a pipeline ``stage`` (``auto_optimize`` also runs on stock DaCe), or a stage followed by an
    autotuning ``tuner``, which runs as an Agent rather than in a sweep."""
    device = arch.text.upper()
    if stage is None:
        return Column(
            display=f"DaCe ({device})",
            adapter="hpcagent_bench.frameworks.dace_framework:DaceFramework",
            base="dace",
            full_name=f"DaCe {device}",
            postfix="dace",
            arch=arch,
            sweep_deterministic=False,
            precisions=DACE_PRECISIONS,
        )
    alone, beside, word = DACE_STAGES[stage]
    display, full_name, flavor = f"DaCe ({device}, {alone})", f"DaCe {device} {word}", stage.text
    if tuner is not None:
        tuned = TUNERS[tuner]
        display = f"DaCe ({device}, {beside}) + {tuned.display}"
        full_name = f"DaCe {device} {beside} + {tuned.full_name}"
        flavor = f"{stage.text}_{tuner}"
    return Column(
        display=display,
        adapter="hpcagent_bench.frameworks.dace_framework:DaceFramework" if tuner is None else TUNERS[tuner].adapter,
        base="dace",
        full_name=full_name,
        postfix="dace",
        arch=arch,
        sweep_deterministic=tuner is None,
        precisions=DACE_PRECISIONS if tuner is None else IEEE_PRECISIONS,
        pipelines=(f"{stage.text}_{arch.text}",),
        stage=stage,
        tuner=tuner,
        column=f"dace_{arch.text}",
        flavor=flavor,
    )


def add_dace(arch: Arch, stage: DaceStage | None = None, tuner: str | None = None) -> None:
    """Register the DaCe entry :func:`dace_column` builds, under ``dace_<arch>[_<stage>[_<tuner>]]``."""
    key = "_".join(part for part in (arch.text, stage.text if stage else None, tuner) if part is not None)
    FRAMEWORKS.add(f"dace_{key}", dace_column(arch, stage, tuner))


# Registration order is each column's slot (its colour and marker): a new column goes last.

FRAMEWORKS.add(
    "numpy",
    Column(
        display="NumPy",
        adapter="hpcagent_bench.frameworks.framework:Framework",
        base="numpy",
        full_name="NumPy",
        postfix="numpy",
        arch=Arch.CPU,
        sweep_deterministic=True,
        precisions=ALL_PRECISIONS,
    ),
)


FRAMEWORKS.add(
    "numba",
    Column(
        display="Numba",
        adapter="hpcagent_bench.frameworks.numba_framework:NumbaFramework",
        base="numba",
        full_name="Numba",
        postfix="numba",
        arch=Arch.CPU,
        sweep_deterministic=False,
        precisions=IEEE_PRECISIONS,
    ),
)


# Native backend: one flavor per (language, compiler), each building its own .so. ``polly`` is the C++
# flavor with a polyhedral flags preset; ``pluto`` is a separate source-to-source base.
FRAMEWORKS.add(
    "cc",
    Column(
        display="GCC",
        adapter="hpcagent_bench.frameworks.native_framework:NativeFramework",
        base="native",
        full_name="C (gcc)",
        postfix="cpp",
        arch=Arch.CPU,
        sweep_deterministic=True,
        precisions=IEEE_PRECISIONS,
        language="c",
    ),
    aliases=("c",),
)


add_dace(Arch.CPU)


FRAMEWORKS.add(
    "fortran",
    Column(
        display="gfortran",
        adapter="hpcagent_bench.frameworks.native_framework:NativeFramework",
        base="native",
        full_name="Fortran (gfortran)",
        postfix="cpp",
        arch=Arch.CPU,
        sweep_deterministic=True,
        precisions=IEEE_PRECISIONS,
        language="fortran",
    ),
)


# The gcc C++ column, completing gcc/g++/gfortran as one family (``llvm`` and ``polly`` are clang).
FRAMEWORKS.add(
    "cpp",
    Column(
        display="g++",
        adapter="hpcagent_bench.frameworks.native_framework:NativeFramework",
        base="native",
        full_name="C++ (g++)",
        postfix="cpp",
        arch=Arch.CPU,
        sweep_deterministic=True,
        precisions=IEEE_PRECISIONS,
        language="cpp",
        compiler="gpp",
    ),
)


FRAMEWORKS.add(
    "pythran",
    Column(
        display="Pythran",
        adapter="hpcagent_bench.frameworks.pythran_framework:PythranFramework",
        base="pythran",
        full_name="Pythran",
        postfix="pythran",
        arch=Arch.CPU,
        sweep_deterministic=False,
        precisions=IEEE_PRECISIONS,
    ),
)


# gcc's auto-parallelizer, the GCC half of the autopar axis clang already had via polly.
FRAMEWORKS.add(
    "cc_autopar",
    Column(
        display="GCC (autopar)",
        adapter="hpcagent_bench.frameworks.native_framework:NativeFramework",
        base="native",
        full_name="C autopar (gcc)",
        postfix="cpp",
        arch=Arch.CPU,
        sweep_deterministic=True,
        precisions=IEEE_PRECISIONS,
        language="c",
        flags="GCC_AUTOPAR",
    ),
    aliases=("c-autopar",),
)


add_dace(Arch.CPU, DaceStage.CANON)
add_dace(Arch.CPU, DaceStage.AUTOOPT)


FRAMEWORKS.add(
    "llvm",
    Column(
        display="Clang",
        adapter="hpcagent_bench.frameworks.native_framework:NativeFramework",
        base="native",
        full_name="C++ (clang)",
        postfix="cpp",
        arch=Arch.CPU,
        sweep_deterministic=True,
        precisions=IEEE_PRECISIONS,
        language="cpp",
        compiler="clangpp",
    ),
)


# Tiled OpenMP C. Pluto and PPCG share the pet/isl front end but run on different hardware, so they stay
# separate columns; polycc reads the C target's ``_pluto_input.c`` and writes C (VLA ``restrict`` parameters).
FRAMEWORKS.add(
    "pluto",
    Column(
        display="Pluto",
        adapter="hpcagent_bench.frameworks.pluto_framework:PlutoFramework",
        base="pluto",
        full_name="Polyhedral CPU (Pluto)",
        postfix="cpp",
        arch=Arch.CPU,
        sweep_deterministic=True,
        precisions=IEEE_PRECISIONS,
        language="c",
        compiler="clang-pluto",
        flags="PLUTO_PAR",
        autopar_gate="pluto_capability",
        transform=Transform.PLUTO,
    ),
)


FRAMEWORKS.add(
    "polly",
    Column(
        display="Polly",
        adapter="hpcagent_bench.frameworks.native_framework:NativeFramework",
        base="native",
        full_name="C++ Polly (clang)",
        postfix="cpp",
        arch=Arch.CPU,
        sweep_deterministic=True,
        precisions=IEEE_PRECISIONS,
        language="cpp",
        compiler="clangpp",
        flags="POLLY_PAR",
        autopar_gate="polly_capability",
    ),
)


FRAMEWORKS.add(
    "jax",
    Column(
        display="JAX",
        adapter="hpcagent_bench.frameworks.jax_framework:JaxFramework",
        base="jax",
        full_name="Jax",
        postfix="jax",
        arch=Arch.CPU,
        sweep_deterministic=False,
        precisions=ALL_PRECISIONS,
    ),
    aliases=("jax_cpu", "jax_gpu"),
)


# One base, two hardware flavors sharing the unified ``<kernel>_tvm.py`` (``tvm_build.active_kernel``).
FRAMEWORKS.add(
    "tvm_gpu",
    Column(
        display="TVM (GPU)",
        adapter="hpcagent_bench.frameworks.tvm_framework:TVMFramework",
        base="tvm",
        full_name="TVM (GPU)",
        postfix="tvm",
        arch=Arch.GPU,
        sweep_deterministic=False,
        precisions=ALL_PRECISIONS,
    ),
    aliases=("tvm",),
)


# The C family across the graded vendors, named ``cc_<vendor>`` (``llvm`` and ``polly`` already name the
# clang C++ columns).
FRAMEWORKS.add(
    "cc_llvm",
    Column(
        display="Clang",
        adapter="hpcagent_bench.frameworks.native_framework:NativeFramework",
        base="native",
        full_name="C (clang)",
        postfix="cpp",
        arch=Arch.CPU,
        sweep_deterministic=False,
        precisions=IEEE_PRECISIONS,
        language="c",
        compiler="clang",
    ),
)


FRAMEWORKS.add(
    "cc_llvm_autopar",
    Column(
        display="Clang (autopar)",
        adapter="hpcagent_bench.frameworks.native_framework:NativeFramework",
        base="native",
        full_name="C Polly (clang)",
        postfix="cpp",
        arch=Arch.CPU,
        sweep_deterministic=False,
        precisions=IEEE_PRECISIONS,
        language="c",
        compiler="clang",
        flags="POLLY_PAR",
        autopar_gate="polly_capability",
    ),
)


FRAMEWORKS.add(
    "cc_nvhpc",
    Column(
        display="NVHPC",
        adapter="hpcagent_bench.frameworks.native_framework:NativeFramework",
        base="native",
        full_name="C (nvc)",
        postfix="cpp",
        arch=Arch.CPU,
        sweep_deterministic=False,
        precisions=IEEE_PRECISIONS,
        language="c",
        compiler="nvc",
    ),
)


FRAMEWORKS.add(
    "cc_nvhpc_autopar",
    Column(
        display="NVHPC (autopar)",
        adapter="hpcagent_bench.frameworks.native_framework:NativeFramework",
        base="native",
        full_name="C autopar (nvc)",
        postfix="cpp",
        arch=Arch.CPU,
        sweep_deterministic=False,
        precisions=IEEE_PRECISIONS,
        language="c",
        compiler="nvc",
        flags="NVHPC_CONCUR",
        autopar_gate="nvhpc_autopar_capability",
    ),
)


# LLVM Fortran, the flang half of the gfortran/flang pair (declines cleanly if the driver is absent).
FRAMEWORKS.add(
    "flang",
    Column(
        display="Flang",
        adapter="hpcagent_bench.frameworks.native_framework:NativeFramework",
        base="native",
        full_name="Fortran (flang)",
        postfix="cpp",
        arch=Arch.CPU,
        sweep_deterministic=True,
        precisions=IEEE_PRECISIONS,
        language="fortran",
        compiler="flang",
    ),
)


# The Fortran half of the autopar axis (same emitted Fortran as ``fortran``, autopar flags differ).
FRAMEWORKS.add(
    "fortran_autopar",
    Column(
        display="gfortran (autopar)",
        adapter="hpcagent_bench.frameworks.native_framework:NativeFramework",
        base="native",
        full_name="Fortran autopar (gfortran)",
        postfix="cpp",
        arch=Arch.CPU,
        sweep_deterministic=True,
        precisions=IEEE_PRECISIONS,
        language="fortran",
        flags="GCC_AUTOPAR",
    ),
)


add_dace(Arch.GPU)
add_dace(Arch.GPU, DaceStage.AUTOOPT)
add_dace(Arch.GPU, DaceStage.CANON)


FRAMEWORKS.add(
    "cupy",
    Column(
        display="CuPy",
        adapter="hpcagent_bench.frameworks.cupy_framework:CupyFramework",
        base="cupy",
        full_name="CuPy",
        postfix="cupy",
        arch=Arch.GPU,
        sweep_deterministic=False,
        precisions=frozenset({Precision.FP64, Precision.FP32, Precision.FP16, Precision.BF16}),
    ),
)


# No fp64 path; runs the low-precision matrix instead.
FRAMEWORKS.add(
    "triton",
    Column(
        display="Triton",
        adapter="hpcagent_bench.frameworks.triton_framework:TritonFramework",
        base="triton",
        full_name="Triton",
        postfix="triton",
        arch=Arch.GPU,
        sweep_deterministic=False,
        precisions=frozenset({Precision.FP8_E5M2, Precision.FP8_E4M3, Precision.FP32, Precision.FP16, Precision.BF16}),
    ),
)


# ppcg emits CUDA; the compiled language is the local GPU toolchain's (hipify on ROCm,
# ``hpcagent_bench.ppcg_transform``), and compilers.yaml maps it to its compiler.
FRAMEWORKS.add(
    "ppcg",
    Column(
        display="PPCG",
        adapter="hpcagent_bench.frameworks.pluto_framework:PlutoFramework",
        base="pluto",
        full_name="Polyhedral GPU (PPCG)",
        postfix="cpp",
        arch=Arch.GPU,
        sweep_deterministic=False,
        precisions=IEEE_PRECISIONS,
        language=gpu_backend().text,
        emit_language="c",
        transform=Transform.PPCG,
    ),
)


# The ppcg transform with the GPU vendor pinned, so a row's vendor is a property of the column.
FRAMEWORKS.add(
    "ppcg_cuda",
    Column(
        display="PPCG (CUDA)",
        adapter="hpcagent_bench.frameworks.pluto_framework:PlutoFramework",
        base="pluto",
        full_name="Polyhedral GPU (PPCG, CUDA)",
        postfix="cpp",
        arch=Arch.GPU,
        sweep_deterministic=False,
        precisions=IEEE_PRECISIONS,
        column="ppcg",
        flavor="cuda",
        language="cuda",
        emit_language="c",
        transform=Transform.PPCG,
    ),
)


# ppcg's CUDA through hipify-perl, built by hipcc (``hpcagent_bench.ppcg_transform``). The label is NOT
# "PPCG (HIP)": ppcg has no AMD target (``--target`` takes c, cuda or opencl), so a reader comparing it with a
# hand-written HIP column is comparing against a translated source, not what a polyhedral compiler emits.
FRAMEWORKS.add(
    "ppcg_hip",
    Column(
        display="PPCG (CUDA via hipify)",
        adapter="hpcagent_bench.frameworks.pluto_framework:PlutoFramework",
        base="pluto",
        full_name="Polyhedral GPU (PPCG, CUDA via hipify)",
        postfix="cpp",
        arch=Arch.GPU,
        sweep_deterministic=False,
        precisions=IEEE_PRECISIONS,
        column="ppcg",
        flavor="hip",
        language="hip",
        emit_language="c",
        transform=Transform.PPCG,
    ),
)


FRAMEWORKS.add(
    "tvm_cpu",
    Column(
        display="TVM (CPU)",
        adapter="hpcagent_bench.frameworks.tvm_framework:TVMFramework",
        base="tvm",
        full_name="TVM (CPU)",
        postfix="tvm",
        arch=Arch.CPU,
        sweep_deterministic=False,
        precisions=ALL_PRECISIONS,
    ),
)


add_dace(Arch.CPU, DaceStage.PARALLEL)
add_dace(Arch.GPU, DaceStage.PARALLEL)
add_dace(Arch.CPU, DaceStage.CANON, "alloy_profile")
add_dace(Arch.CPU, DaceStage.CANON, "alloy_heuristic")
add_dace(Arch.CPU, DaceStage.PARALLEL, "alloy_profile")
add_dace(Arch.CPU, DaceStage.PARALLEL, "alloy_heuristic")
add_dace(Arch.GPU, DaceStage.CANON, "alloy_profile")
add_dace(Arch.GPU, DaceStage.CANON, "alloy_heuristic")
add_dace(Arch.GPU, DaceStage.PARALLEL, "alloy_profile")
add_dace(Arch.GPU, DaceStage.PARALLEL, "alloy_heuristic")


# Polygeist's cgeist lowers the C target's ``_pluto_input.c`` to affine MLIR, Polymer reschedules it with
# Pluto, MLIR parallelizes it to OpenMP, and clang compiles the LLVM IR (``hpcagent_bench.polygeist_transform``).
FRAMEWORKS.add(
    "polygeist_cpu",
    Column(
        display="Polygeist",
        adapter="hpcagent_bench.frameworks.pluto_framework:PlutoFramework",
        base="pluto",
        full_name="Polyhedral CPU (Polygeist)",
        postfix="cpp",
        arch=Arch.CPU,
        sweep_deterministic=True,
        precisions=IEEE_PRECISIONS,
        language="c",
        compiler="clang-polygeist",
        flags="POLYGEIST_PAR",
        transform=Transform.POLYGEIST,
    ),
)


# The same affine MLIR mapped to GPU blocks and threads and lowered through ROCDL; the arrays stay where the
# caller allocated them (``xnack+`` under ``HSA_XNACK=1``), so the pipeline has no data-movement step.
FRAMEWORKS.add(
    "polygeist_gpu",
    Column(
        display="Polygeist (GPU)",
        adapter="hpcagent_bench.frameworks.pluto_framework:PlutoFramework",
        base="pluto",
        full_name="Polyhedral GPU (Polygeist)",
        postfix="cpp",
        arch=Arch.GPU,
        sweep_deterministic=False,
        precisions=IEEE_PRECISIONS,
        language="c",
        compiler="clang-polygeist-gpu",
        transform=Transform.POLYGEIST,
    ),
)


add_dace(Arch.CPU, DaceStage.AUTOOPT, "alloy_profile")
add_dace(Arch.CPU, DaceStage.AUTOOPT, "alloy_heuristic")
add_dace(Arch.GPU, DaceStage.AUTOOPT, "alloy_profile")
add_dace(Arch.GPU, DaceStage.AUTOOPT, "alloy_heuristic")

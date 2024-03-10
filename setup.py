#!/usr/bin/env python3
from setuptools import setup, Extension
import numpy as np
from os import environ


def main():
    """
    Compile time options (set as environment variables):
    MKL:      use intel math kernel library
    CBLAS:    use C bindings of BLAS
    LAPACK_C: use C bindings of LAPACK
    PARALLEL: use some parallelization (not always helpful)
    PARALLEL_EXTRAPOLATION: use parallel matrix extrapolation when mitigating
              Floquet matrix truncation effects
    PARALLEL_EXTRA_DIMS: parallelize calculation for voltage-shifted replicas
    DEBUG:    enable debugging options
    ANALYZE:  enable benchmark/analysis options
    """
    compiler_args = ["-O3", "-Wall", "-Wextra", "-std=c11"]
    linker_args = []
    include_dirs = [np.get_include()]
    library_dirs = []
    libraries = []

    if "MKL" in environ:
        libraries += ["mkl_rt"]
        include_dirs += ["/opt/intel/mkl/include"]
        library_dirs += ["/opt/intel/mkl/lib/intel64"]
        compiler_args += ["-DCBLAS"]
        compiler_args += ["-DMKL"]
    else:
        libraries += ["lapack"]
        if "CBLAS" in environ:
            compiler_args += ["-DCBLAS"]
            libraries += ["cblas"]
        else:
            libraries += ["blas"]
        if "LAPACK_C" in environ:
            compiler_args += ["-DLAPACK_C"]

    parallel_modifiers = ("PARALLEL", "PARALLEL_EXTRAPOLATION", "PARALLEL_EXTRA_DIMS")
    need_omp = False
    for modifier in parallel_modifiers:
        if modifier in environ:
            compiler_args += ["-D" + modifier]
            need_omp = True
    if need_omp:
        compiler_args += ["-fopenmp"]
        linker_args += ["-fopenmp"]

    if "DEBUG" in environ:
        compiler_args += ["-DDEBUG"]

    if "ANALYZE" in environ:
        compiler_args += ["-DANALYZE"]

    module = Extension(
        "frtrg.rtrg_c",
        sources=["src/frtrg/rtrg_c.c"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=compiler_args,
        extra_link_args=linker_args,
    )
    setup(ext_modules=[module])


if __name__ == "__main__":
    main()

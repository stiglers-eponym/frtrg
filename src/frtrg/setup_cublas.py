# build with:
# python3 setup.py build_ext --inplace

"""
This file is only provided to illustrate how the CUBLAS version of
the library rtrg_c was built. This code has not been used for long
time. It did not turn out to be useful. Using it will require some
debugging.
"""


from distutils.core import setup, Extension
import numpy as np
from os import environ


def main():
    compiler_args = ["-O3", "-Wall", "-Wextra", "-std=c11"]
    linker_args = []
    include_dirs = [np.get_include(), "."]
    name = "rtrg_cublas"
    sources = ["rtrg_cublas.c"]

    include_dirs += [environ.get("CUDA_PATH") + "/include"]

    if "CBLAS" in environ:
        compiler_args += ["-DCBLAS"]
        libraries += ["cblas"]
        # libraries += ['mkl_rt']
    else:
        libraries += ["blas"]

    if "LAPACK_C" in environ:
        compiler_args += ["-DLAPACK_C"]

    if "DEBUG" in environ:
        print("using -DDEBUG")
        compiler_args += ["-DDEBUG"]

    parallel_modifiers = ("PARALLEL_EXTRAPOLATION", "PARALLEL_EXTRA_DIMS")

    need_omp = False
    for modifier in parallel_modifiers:
        if modifier in environ:
            compiler_args += ["-D" + modifier]
            need_omp = True

    if need_omp:
        compiler_args += ["-fopenmp"]
        linker_args += ["-fopenmp"]

    library_dirs = [".", environ.get("CUDA_PATH") + "/lib64"]
    libraries = ["cuda_helpers", "cublas", "cudart", "mkl_rt"]

    module = Extension(
        name,
        sources=sources,
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        extra_compile_args=compiler_args,
        extra_link_args=linker_args,
    )
    setup(ext_modules=[module])


if __name__ == "__main__":
    print(__doc__)
    # main()

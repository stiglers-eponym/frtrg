/* CONFIGURATION
 * The following variables can be defined:
 *   CBLAS: if defined, use CBLAS instead of directly calling BLAS functions
 *   LAPACK_C: include LAPACK C header instead of just linking to LAPACK
 *   PARALLEL_EXTRA_DIMS: use openmp to parallelize repeated operations over
 *          extra dimensions of arrays. Note that internal parallelization of
 *          BLAS functions might be faster.
 *   PARALLEL_EXTRAPOLATION: use OpenMP to parallelize the extrapolation loops.
 *          This is usually helpful except for small matrices.
 *   DEBUG: print debugging information to stderr. This is neither complete
 *          nor really useful.
 * The following macros can be redefined to optimize performance, adapt to your
 * BLAS and LAPACK installation, and adapt to the concrete mathematical problem.
 *   TRIANGULAR_OPTIMIZE_THRESHOLD: Threshold for subdividing multiplication
 *          of two triangular matrices (see below).
 *   extrapolate: function for extrapolation of unknown matrix elements
 *   complex_type, NPY_COMPLEX_TYPE: data type or basically everything
 *   gemm, trmm: (C)BLAS function names (not CUDA!), need to be adapted to complex_type.
 *   getrf, getri: LAPACK function names, need to be adapted to complex_type.
 */


/* Threshold for subdividing multiplication of two triangular matrices
 * (multiply_LU_inplace and multiply_UL_inplace).
 * The optimal value for this probably depends on the parallelization
 * used in BLAS functions. When using a GPU for matrix multiplication,
 * you should probably choose a large value here. Be aware that the
 * functions implemented here may be slow on a GPU.
 * If a matrix is smaller (less rows/columns) than this threshold, then
 * trmm from BLAS is used directly, which discards the fact that the
 * left matrix is also triangular. Otherwise the problem is recursively
 * subdivided. */
#define TRIANGULAR_OPTIMIZE_THRESHOLD 128
#define TRIANGULAR_OPTIMIZE_THRESHOLD_GPU 128
#define LU_ON_GPU_THRESHOLD 768

#define THRESHOLD_GEMM_GPU 512
#define THRESHOLD_TRMM_GPU 768

/* Simple linear extrapolation based on the last 3 elements.
 * Given the mapping {0:a, -1:b, -2:c} estimate the value at i. */
#define extrapolate(i, a, b, c) ((1 + 0.75*i)*(a) - 0.5*i*((b) + 0.5*(c)))


/* NOTE: complex_type and complex_type_cuda must be equivalent! */
#define complex_type complex double
#define complex_type_cuda cuDoubleComplex
#define NPY_COMPLEX_TYPE NPY_COMPLEX128
#define lapack_complex_double complex double

#define cu_gemm cublasZgemm
#define cu_trmm cublasZtrmm

#ifdef LAPACK_C
#define getrf LAPACK_zgetrf
#define getri LAPACK_zgetri
#else /* LAPACK_C */
#define getrf zgetrf_
#define getri zgetri_
#endif /* LAPACK_C */

#ifdef CBLAS
#define gemm cblas_zgemm
#define trmm cblas_ztrmm
#else /* CBLAS */
#define gemm zgemm_
#define trmm ztrmm_
#endif /* CBLAS */


enum
{
    UpperTriangular = 1 << 0,
    LeftMultiplication = 1 << 1,
    UnitTriangular = 1 << 2,
};

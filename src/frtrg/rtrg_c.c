/*
MIT License

Copyright (c) 2021 Valentin Bruch <valentin.bruch@rwth-aachen.de>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/**
 * @file
 * @section sec_config Configuration
 *
 * The following variables can be defined:
 * * MKL:   if defined, use intel MKL for CBLAS and LAPACK C header
 * * CBLAS: if defined, use CBLAS instead of directly calling BLAS functions
 * * LAPACK_C: include LAPACK C header instead of just linking to LAPACK
 * * PARALLEL_EXTRA_DIMS: use OpenMP to parallelize repeated operations over
 *          extra dimensions of arrays. Note that internal parallelization of
 *          BLAS functions might be faster.
 * * PARALLEL_EXTRAPOLATION: use OpenMP to parallelize the extrapolation loops.
 *          This is usually helpful except for small matrices.
 * * PARALLEL: Do some matrix products simultaneously (in parallel) using
 *          OpenMP. This can be useful, but might also decrease performance
 *          because internal parallelization of the BLAS functions is much
 *          more efficient.
 * * PARALLELIZE_CORRECTION_THREADS_THRESHOLD: number of threads at which
 *          maximal parallelization is used. In practice this maximal
 *          parallelization does not seem very useful and a high value
 *          is recommended.
 * * DEBUG: print debugging information to stderr. This is neither complete
 *          nor really useful.
 * * ANALYZE: collect some data for analyzing number of calls of specific
 *          functions and structure of input matrices for optimization.
 * The following macros can be redefined to optimize performance, adapt to your
 * BLAS and LAPACK installation, and adapt to the concrete mathematical problem.
 * * TRIANGULAR_OPTIMIZE_THRESHOLD: Threshold for subdividing multiplication
 *          of two triangular matrices (see below).
 * * extrapolate: function for extrapolation of unknown matrix elements
 * * complex_type, NPY_COMPLEX_TYPE: data type or basically everything
 * * gemm, trmm: (C)BLAS function names, need to be adapted to complex_type.
 * * getrf, getri: LAPACK function names, need to be adapted to complex_type.
 */


/**
 * Threshold for subdividing multiplication of two triangular matrices
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

#define PARALLELIZE_CORRECTION_THREADS_THRESHOLD 16

/**
 * Simple linear extrapolation based on the last 3 elements.
 * Given the mapping {0:a, -1:b, -2:c} estimate the value at i. */
#define extrapolate(i, a, b, c) ((1 + 0.75*i)*(a) - 0.5*i*((b) + 0.5*(c)))

/* Define data type and select CBLAS and LAPACK functions accordingly */
#include <complex.h>
#define complex_type complex double
#define NPY_COMPLEX_TYPE NPY_COMPLEX128
#define lapack_complex_double complex_type

#ifdef MKL

#include <mkl_lapack.h>
#include <mkl_cblas.h>
#define getrf zgetrf
#define getri zgetri
#define gemm cblas_zgemm
#define trmm cblas_ztrmm

#else /* MKL */

#ifdef LAPACK_C
#include <lapack.h>
#define getrf LAPACK_zgetrf
#define getri LAPACK_zgetri
#else /* LAPACK_C */
#define getrf zgetrf_
#define getri zgetri_
extern void zgetrf_(const int*, const int*, double complex*, const int*, int*, int*);
extern void zgetri_(const int*, double complex*, const int*, const int*, complex double*, const int*, int*);
#endif /* LAPACK_C */

#ifdef CBLAS
#include <cblas.h>
#define gemm cblas_zgemm
#define trmm cblas_ztrmm
#else /* CBLAS */
extern void zgemm_(const char*, const char*, const int*, const int*, const int*, const complex double*, const complex double*, const int*, const complex double*, const int*, const double complex*, double complex*, const int*);
extern void ztrmm_(const char*, const char*, const char*, const char*, const int*, const int*, const complex double*, const complex double*, const int*, complex double*, const int*);
#define gemm zgemm_
#define trmm ztrmm_
static const char N='N', L='L', R='R', U='U';
#endif /* CBLAS */

#endif /* MKL */

static const complex_type zero = 0.;
static const complex_type one = 1.;


#ifdef ANALYZE
static int TOTAL_MATRIX_MULTIPLICATIONS = 0;
#define MATRIX_MULTIPLICATIONS_SIZE 0x40
static int MATRIX_MULTIPLICATIONS[MATRIX_MULTIPLICATIONS_SIZE];
#define LEFT_F_CONTIGUOUS 0x1
#define RIGHT_F_CONTIGUOUS 0x2
#define LEFT_C_CONTIGUOUS 0x4
#define RIGHT_C_CONTIGUOUS 0x8
#define SYMMETRIC 0x10
#define TWO_DIM 0x20
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#endif

#define PY_SSIZE_T_CLEAN

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#ifdef DEBUG
#include <stdio.h>
#endif

/* It is easy to parallelize the iteration over extra dimensions.
 * But this seems to be quite inefficient and does not speed up the
 * calculations. If you want to use PARALLEL_EXTRA_DIMS, you should
 * first test whether it gives you an advantage.*/
#if defined(PARALLEL_EXTRA_DIMS) || defined(PARALLEL) || defined(PARALLEL_EXTRAPOLATION)
#include <omp.h>
#endif

/**
 * Flags for overwriting input in symmetric matrix multiplication.
 * Symmetric matrix multiplication can be faster if it is allowed to overwrite
 * the left matrix. This enumerator defines flags for this option. */
enum {
    OVERWRITE_LEFT = 1 << 0,
    OVERWRITE_RIGHT = 1 << 1,
};


/**
 * @file
 * @section sec_extrapolate_multiplication Extrapolate matrix for multiplication
 *
 * For example, for cutoff = 2 and a matrix
 *
 *          m00 m01 m02 m03 m04 m05
 *          m10 m11 m12 m13 m14 m15
 *          m20 m21 m22 m23 m24 m25
 *          m30 m31 m32 m33 m34 m35
 *          m40 m41 m42 m43 m44 m45
 *
 * the following functions fill the arrays t, l, r, b in the extrapolated
 * matrix
 *
 *          t00
 *          t10 t11
 *  l00 l01 m00 m01 m02 m03 m04 m05
 *      l11 m10 m11 m12 m13 m14 m15
 *          m20 m21 m22 m23 m24 m25
 *          m30 m31 m32 m33 m34 m35 r00
 *          m40 m41 m42 m43 m44 m45 r10 r11
 *                          b00 b01
 *                              b11
 *
 * In this representation, the pointers handed to the following functions have
 * the meaning:
 * extrapolate_top:    output=t00, input=m00
 * extrapolate_left:   output=l00, input=m00
 * extrapolate_bottom: output=b00, input=m45
 * extrapolate_right:  output=r00, input=m45
 *
 * @todo This could be parallelized.
 */

/**
 * Extrapolate matrix to the top.
 * input and output must be in columns major order (Fortran style).
 * output is treated as a square matrix, it must have at least cutoff
 * columns.
 *
 * The following requirements are not explicitly checked:
 * * out_rows >= cutoff
 * * rows_in >= 3
 * * cols_in >= cutoff + 3
 *
 * @see extrapolate_top_full()
 * @see extrapolate_bottom()
 * @see extrapolate_left()
 * @see extrapolate_right()
 */
static void extrapolate_top(
        const complex_type *input,
        const int rows_in,
        complex_type *output,
        const int cutoff,
        const int out_rows
        )
{
#ifdef DEBUG
    fprintf(stderr, "Starting extrapolate_top\n");
#endif
#ifdef PARALLEL_EXTRAPOLATION
#pragma omp parallel
    {
        complex_type row0, row1, row2;
        int chunk, i;
#pragma omp for
        for(chunk=1; chunk <= cutoff; chunk++)
        {
            row0 = input[chunk*rows_in];
            row1 = input[(chunk+1)*rows_in+1];
            row2 = input[(chunk+2)*rows_in+2];
            i = chunk + 1;
            while(--i > 0)
                output[cutoff-chunk+(chunk-i)*(out_rows+1)]
                    = extrapolate(i, row0, row1, row2);
        }
    }
#else /* PARALLEL_EXTRAPOLATION */
    input += rows_in;
    const complex_type *row1 = input + rows_in + 1;
    const complex_type *row2 = row1 + rows_in + 1;
    complex_type *write;
    int chunk = 1, i;
    while (chunk <= cutoff)
    {
        write = output + cutoff - chunk;
        i = ++chunk;
        while (--i > 0)
        {
            *write = extrapolate(i, *input, *row1, *row2);
            write += out_rows + 1;
        }
        input += rows_in;
        row1 += rows_in;
        row2 += rows_in;
    }
#endif /* PARALLEL_EXTRAPOLATION */
#ifdef DEBUG
    fprintf(stderr, "Done extrapolate_top\n");
#endif
}

/**
 * Extrapolate matrix to the bottom.
 * input and output must be in columns major order (Fortran style).
 * output is treated as a square matrix, it must have at least cutoff
 * columns.
 *
 * input points to the last element of the matrix!
 *
 * The following requirements are not explicitly checked:
 * * rows_in >= 3
 * * cols_in >= cutoff + 3
 * * out_rows >= cutoff
 *
 * @see extrapolate_top()
 * @see extrapolate_bottom_full()
 * @see extrapolate_left()
 * @see extrapolate_right()
 */
static void extrapolate_bottom(
        const complex_type *input,
        const int rows_in,
        complex_type *output,
        const int cutoff,
        const int out_rows
        )
{
#ifdef DEBUG
    fprintf(stderr, "Starting extrapolate_bottom\n");
#endif
#ifdef PARALLEL_EXTRAPOLATION
#pragma omp parallel
    {
        complex_type row0, row1, row2;
        int i, chunk;
#pragma omp for
        for(chunk=1; chunk <= cutoff; chunk++)
        {
            row0 = input[-chunk*rows_in];
            row1 = input[-(chunk+1)*rows_in-1];
            row2 = input[-(chunk+2)*rows_in-2];
            i = chunk + 1;
            while(--i > 0)
                output[chunk - cutoff + (cutoff - 1 - chunk + i)*(out_rows+1)]
                    = extrapolate(i, row0, row1, row2);
        }
    }
#else /* PARALLEL_EXTRAPOLATION */
    input -= rows_in;
    const complex_type *row1 = input - rows_in - 1;
    const complex_type *row2 = row1 - rows_in - 1;
    complex_type *write;
    int chunk = 1, i;
    while (chunk <= cutoff)
    {
        write = output + (cutoff - 1) * out_rows + chunk - 1;
        i = ++chunk;
        while (--i > 0)
        {
            *write = extrapolate(i, *input, *row1, *row2);
            write -= out_rows + 1;
        }
        input -= rows_in;
        row1 -= rows_in;
        row2 -= rows_in;
    }
#endif /* PARALLEL_EXTRAPOLATION */
#ifdef DEBUG
    fprintf(stderr, "Done extrapolate_bottom\n");
#endif
}

/**
 * Extrapolate matrix to the left.
 * input and output must be in columns major order (Fortran style).
 * output is treated as a square matrix, it must have at least cutoff
 * columns.
 * The input matrix must have at least 3 columns.
 *
 * The following requirements are not explicitly checked:
 * * rows_in >= cutoff + 3
 * * out_rows >= cutoff
 *
 * @see extrapolate_top()
 * @see extrapolate_bottom()
 * @see extrapolate_left_full()
 * @see extrapolate_right()
 */
static void extrapolate_left(
        const complex_type *input,
        const int rows_in,
        complex_type *output,
        const int cutoff,
        const int out_rows
        )
{
#ifdef DEBUG
    fprintf(stderr, "Starting extrapolate_left\n");
#endif
#ifdef PARALLEL_EXTRAPOLATION
#pragma omp parallel
    {
    int j, jmax;
#pragma omp for
        for (int i=1; i <= cutoff; i++)
        {
            jmax = cutoff - i;
            j = -1;
            while (++j <= jmax)
                output[out_rows * jmax + j] = extrapolate(i,
                        input[i + j],
                        input[i + j + rows_in + 1],
                        input[i + j + 2*rows_in + 2]);
        }
    }
#else /* PARALLEL_EXTRAPOLATION */
    ++input;
    const complex_type *col1 = input + rows_in + 1;
    const complex_type *col2 = col1 + rows_in + 1;
    output += out_rows * (cutoff - 1);
    int i=0, j, jmax;
    while (++i <= cutoff)
    {
        jmax = cutoff - i;
        j = -1;
        while (++j <= jmax)
            output[j] = extrapolate(i, input[j], col1[j], col2[j]);
        output -= out_rows;
        ++input;
        ++col1;
        ++col2;
    }
#endif /* PARALLEL_EXTRAPOLATION */
#ifdef DEBUG
    fprintf(stderr, "Done extrapolate_left\n");
#endif
}

/**
 * Extrapolate matrix to the right.
 * input and output must be in columns major order (Fortran style).
 * output is treated as a square matrix, it must have at least cutoff
 * columns.
 * The input matrix must have at least 3 columns.
 *
 * input points to the last element of the matrix!
 *
 * The following requirements are not explicitly checked:
 * rows_in >= cutoff + 3
 * out_rows >= cutoff
 *
 * @see extrapolate_top()
 * @see extrapolate_bottom()
 * @see extrapolate_left()
 * @see extrapolate_right_full()
 */
static void extrapolate_right(
        const complex_type *input,
        const int rows_in,
        complex_type *output,
        const int cutoff,
        const int out_rows
        )
{
#ifdef DEBUG
    fprintf(stderr, "Starting extrapolate_right\n");
#endif
    input -= cutoff;
#ifdef PARALLEL_EXTRAPOLATION
#pragma omp parallel
    {
    int j, jmax;
#pragma omp for
        for (int i=1; i <= cutoff; i++)
        {
            jmax = cutoff - i;
            j = -1;
            while (++j <= jmax)
                output[(out_rows+1) * (i-1) + j] = extrapolate(i,
                        input[j],
                        input[j - rows_in - 1],
                        input[j - 2*rows_in - 2]);
        }
    }
#else /* PARALLEL_EXTRAPOLATION */
    const complex_type *col1 = input - rows_in - 1;
    const complex_type *col2 = col1 - rows_in - 1;
    int i=0, j, jmax;
    while (i < cutoff)
    {
        jmax = cutoff - ++i;
        j = -1;
        while (++j <= jmax)
            output[j] = extrapolate(i, input[j], col1[j], col2[j]);
        output += out_rows + 1;
    }
#endif /* PARALLEL_EXTRAPOLATION */
#ifdef DEBUG
    fprintf(stderr, "Done extrapolate_right\n");
#endif
}


/**
 * @file
 * @section sec_extrapolate_inversion Extrapolate matrix for inversion
 *
 * For example, for cutoff = 2 and a matrix
 *
 *          m00 m01 m02 m03 m04 m05
 *          m10 m11 m12 m13 m14 m15
 *          m20 m21 m22 m23 m24 m25
 *          m30 m31 m32 m33 m34 m35
 *          m40 m41 m42 m43 m44 m45
 *
 * the following functions fill the arrays t, l, r, b in the extrapolated
 * matrix
 *
 *  l00 t01 t02
 *  l10 l11 t12 t13
 *  l20 l21 m00 m01 m02 m03 m04 m05
 *      l31 m10 m11 m12 m13 m14 m15
 *          m20 m21 m22 m23 m24 m25
 *          m30 m31 m32 m33 m34 m35 r00
 *          m40 m41 m42 m43 m44 m45 r10 r11
 *                          b00 b01 r20 r21
 *                              b11 b12 r31
 *
 * In this representation, the pointers handed to the following functions
 * have the meaning:
 * extrapolate_top:    output=t00, input=m00
 * extrapolate_left:   output=l00, input=m00
 * extrapolate_bottom: output=b00, input=m45
 * extrapolate_right:  output=r00, input=m45
 */

/**
 * Extrapolate matrix to the top.
 * input and output must be in columns major order (Fortran style).
 * output must have at least 2*cutoff columns.
 *
 * The following requirements are not explicitly checked:
 * * out_rows >= cutoff
 * * rows_in >= 3
 * * cols_in >= cutoff + 3
 *
 * @see extrapolate_top()
 * @see extrapolate_bottom_full()
 * @see extrapolate_left_full()
 * @see extrapolate_right_full()
 */
static void extrapolate_top_full(
        const complex_type *input,
        const int rows_in,
        complex_type *output,
        const int cutoff,
        const int out_rows
        )
{
#ifdef DEBUG
    fprintf(stderr, "Starting extrapolate_top_full\n");
#endif
    input += rows_in;
    const complex_type
        *row1 = input + rows_in + 1,
        *row2 = row1 + rows_in + 1,
        *endin = input + cutoff*rows_in;
    int i;
    output += out_rows;
    while (input < endin)
    {
        i = cutoff + 1;
        while (--i > 0)
        {
            *output = extrapolate(i, *input, *row1, *row2);
            output += out_rows + 1;
        }
        output -= out_rows * (cutoff - 1) + cutoff;
        input += rows_in;
        row1 += rows_in;
        row2 += rows_in;
    }
#ifdef DEBUG
    fprintf(stderr, "Done extrapolate_top_full\n");
#endif
}

/**
 * Extrapolate matrix to the bottom.
 * input and output must be in columns major order (Fortran style).
 * output must have at least 2*cutoff columns.
 *
 * input points to the last element of the matrix!
 *
 * The following requirements are not explicitly checked:
 * * rows_in >= 3
 * * cols_in >= cutoff + 3
 * * out_rows >= cutoff
 *
 * @see extrapolate_top_full()
 * @see extrapolate_bottom()
 * @see extrapolate_left_full()
 * @see extrapolate_right_full()
 */
static void extrapolate_bottom_full(
        const complex_type *input,
        const int rows_in,
        complex_type *output,
        const int cutoff,
        const int out_rows
        )
{
#ifdef DEBUG
    fprintf(stderr, "Starting extrapolate_bottom_full\n");
#endif
    input -= rows_in;
    const complex_type
        *row1 = input - rows_in - 1,
        *row2 = row1 - rows_in - 1,
        *endin = input - cutoff*rows_in;
    int i;
    output += (2*cutoff-2)*out_rows + cutoff - 1;
    while (input > endin)
    {
        i = cutoff + 1;
        while (--i > 0)
        {
            *output = extrapolate(i, *input, *row1, *row2);
            output -= out_rows + 1;
        }
        output += (cutoff - 1) * out_rows + cutoff;
        input -= rows_in;
        row1 -= rows_in;
        row2 -= rows_in;
    }
#ifdef DEBUG
    fprintf(stderr, "Done extrapolate_bottom_full\n");
#endif
}

/**
 * Extrapolate matrix to the left.
 * input and output must be in columns major order (Fortran style).
 * output must have at least 2*cutoff columns.
 * The input matrix must have at least 3 columns.
 *
 * The following requirements are not explicitly checked:
 * * rows_in >= cutoff + 3
 * * out_rows >= cutoff
 *
 * @see extrapolate_top_full()
 * @see extrapolate_bottom_full()
 * @see extrapolate_left()
 * @see extrapolate_right_full()
 */
static void extrapolate_left_full(
        const complex_type *input,
        const int rows_in,
        complex_type *output,
        const int cutoff,
        const int out_rows
        )
{
#ifdef DEBUG
    fprintf(stderr, "Starting extrapolate_left_full\n");
#endif
    const complex_type *col1 = input + rows_in + 1;
    const complex_type *col2 = col1 + rows_in + 1;
    output += out_rows * (cutoff - 1) + cutoff - 1;
    int i=0, j;
    while (++i <= cutoff)
    {
        j = -1;
        while (++j <= cutoff)
            output[j] = extrapolate(i, input[j], col1[j], col2[j]);
        output -= out_rows + 1;
    }
#ifdef DEBUG
    fprintf(stderr, "Done extrapolate_left_full\n");
#endif
}

/**
 * Extrapolate matrix to the right.
 * input and output must be in columns major order (Fortran style).
 * output must have at least 2*cutoff columns.
 * The input matrix must have at least 3 columns.
 *
 * input points to the last element of the matrix!
 *
 * The following requirements are not explicitly checked:
 * * rows_in >= cutoff + 3
 * * out_rows >= cutoff
 *
 * @see extrapolate_top_full()
 * @see extrapolate_bottom_full()
 * @see extrapolate_left_full()
 * @see extrapolate_right()
 */
static void extrapolate_right_full(
        const complex_type *input,
        const int rows_in,
        complex_type *output,
        const int cutoff,
        const int out_rows
        )
{
#ifdef DEBUG
    fprintf(stderr, "Starting extrapolate_right_full\n");
#endif
    input -= cutoff;
    const complex_type *col1 = input - rows_in - 1;
    const complex_type *col2 = col1 - rows_in - 1;
    int i=0, j;
    while (++i <= cutoff)
    {
        j = -1;
        while (++j <= cutoff)
            output[j] = extrapolate(i, input[j], col1[j], col2[j]);
        output += out_rows + 1;
    }
#ifdef DEBUG
    fprintf(stderr, "Done extrapolate_right_full\n");
#endif
}



/**
 * @file
 * @section sec_helper_functions Helper functions for multiplication
 */

/**
 * Multiply upper and lower triangular matrix.
 * @param A upper triangular matrix, fortran ordered. A will be overwritten by the prodyct AB.
 * @param B lower triangular matrix, fortran ordered.
 *
 * Both matrices are in Fortran order (columns major) not unit triangular.
 * Both matrices must have shape (size, size).
 *
 * @see multiply_LU_inplace()
 */
static void multiply_UL_inplace(
        const int size,
        complex_type *a,
        const int a_dim0,
        const complex_type *b,
        const int b_dim0
        )
{
#ifdef DEBUG
    fprintf(stderr, "Starting multiply_UL_inplace %d\n", size);
#endif
    if (size < TRIANGULAR_OPTIMIZE_THRESHOLD)
    {
#ifdef CBLAS
        trmm(
                CblasColMajor, // layout
                CblasRight, // order: this means B := B A
                CblasLower, // A is a lower triangular matrix
                CblasNoTrans, // A is not modified (no adjoint or transpose)
                CblasNonUnit, // A is not unit triangular
                size, // rows of B (int)
                size, // columns of B (int)
                &one, // global prefactor
                b, // matrix A
                b_dim0, // first dimension of A (int)
                a, // matrix B
                a_dim0  // first dimension of B (int)
                );
#else /* CBLAS */
        trmm(
                &R, // order: this means B := B A
                &L, // A is a lower triangular matrix
                &N, // A is not modified (no adjoint or transpose)
                &N, // A is not unit triangular
                &size, // rows of B (int)
                &size, // columns of B (int)
                &one, // global prefactor
                b, // matrix A
                &b_dim0, // first dimension of A (int)
                a, // matrix B
                &a_dim0  // first dimension of B (int)
                );
#endif /* CBLAS */
        return;
    }
    /* TODO: The following commands can be parallelized (partially) */

    /*
     * Matrices are labeled as follows:
     *
     * A = [ Aa  Ab ]  B = [ Ba  Bb ]
     *     [ Ac  Ad ]      [ Bc  Bd ]
     * Initially Ac = Bb = 0.
     */
    const int
        part1 = size / 2,
        part2 = size - part1;

    /* Step 1: overwrite Ac with Ad Bc.
     * This requires that first Bc is copied to Ac. */
    a += part1;
    b += part1;
    int i=-1;
    while (++i<part1)
        memcpy( a + i*a_dim0, b + i*b_dim0, part2 * sizeof(complex_type) );
    a -= part1;
    b -= part1;
#ifdef CBLAS
    trmm(
            CblasColMajor, // layout
            CblasLeft, // order: this means B := A B
            CblasUpper, // A is an upper triangular matrix
            CblasNoTrans, // A is not modified (no adjoint or transpose)
            CblasNonUnit, // A is not unit triangular
            part2, // rows of B (int)
            part1, // columns of B (int)
            &one, // global prefactor
            a + part1*(1 + a_dim0), // matrix A
            a_dim0, // first dimension of A (int)
            a + part1, // matrix B
            a_dim0  // first dimension of B (int)
            );
#else /* CBLAS */
    trmm(
            &L, // order: this means B := A B
            &U, // A is an upper triangular matrix
            &N, // A is not modified (no adjoint or transpose)
            &N, // A is not unit triangular
            &part2, // rows of B (int)
            &part1, // columns of B (int)
            &one, // global prefactor
            a + part1*(1 + a_dim0), // matrix A
            &a_dim0, // first dimension of A (int)
            a + part1, // matrix B
            &a_dim0  // first dimension of B (int)
            );
#endif /* CBLAS */

    /* Step 2: overwrite Ad with Ad Bd */
    multiply_UL_inplace(
            part2,
            a + (a_dim0+1)*part1,
            a_dim0,
            b + (b_dim0+1)*part1,
            b_dim0);
    /* Step 3: overwrite Aa with Aa Ba */
    multiply_UL_inplace(part1, a, a_dim0, b, b_dim0);
    /* Step 4: add Ab Bc to Aa */
#ifdef CBLAS
    gemm(
            CblasColMajor, // layout
            CblasNoTrans, // A is not modified (no adjoint or transpose)
            CblasNoTrans, // B is not modified (no adjoint or transpose)
            part1, // rows of A (int)
            part1, // columns of B (int)
            part2, // columns of A = rows of B (int)
            &one, // global prefactor
            a + part1*a_dim0, // matrix A
            a_dim0, // first dimension of A (int)
            b + part1, // matrix B
            b_dim0,  // first dimension of B (int)
            &one, // weight of C
            a, // matrix C
            a_dim0 // first dimension of C
            );
#else /* CBLAS */
    gemm(
            &N, // A is not modified (no adjoint or transpose)
            &N, // B is not modified (no adjoint or transpose)
            &part1, // rows of A (int)
            &part1, // columns of B (int)
            &part2, // columns of A = rows of B (int)
            &one, // global prefactor
            a + part1*a_dim0, // matrix A
            &a_dim0, // first dimension of A (int)
            b + part1, // matrix B
            &b_dim0,  // first dimension of B (int)
            &one, // weight of C
            a, // matrix C
            &a_dim0 // first dimension of C
            );
#endif /* CBLAS */

    /* Step 5: overwrite Ab with Ab Bd */
#ifdef CBLAS
    trmm(
            CblasColMajor, // layout
            CblasRight, // order: this means B := B A
            CblasLower, // A is a lower triangular matrix
            CblasNoTrans, // A is not modified (no adjoint or transpose)
            CblasNonUnit, // A is not unit triangular
            part1, // rows of B (int)
            part2, // columns of B (int)
            &one, // global prefactor
            b + (1 + b_dim0)*part1, // matrix A
            b_dim0, // first dimension of A (int)
            a + part1*a_dim0, // matrix B
            a_dim0  // first dimension of B (int)
            );
#else /* CBLAS */
    trmm(
            &R, // order: this means B := B A
            &L, // A is a lower triangular matrix
            &N, // A is not modified (no adjoint or transpose)
            &N, // A is not unit triangular
            &part1, // rows of B (int)
            &part2, // columns of B (int)
            &one, // global prefactor
            b + (1 + b_dim0)*part1, // matrix A
            &b_dim0, // first dimension of A (int)
            a + part1*a_dim0, // matrix B
            &a_dim0  // first dimension of B (int)
            );
#endif /* CBLAS */
}

/**
 * Multiply lower and upper triangular matrix.
 * @param A lower triangular matrix, fortran ordered. A will be overwritten by the prodyct AB.
 * @param B upper triangular matrix, fortran ordered.
 *
 * Both matrices are in Fortran order (columns major) not unit triangular.
 * Both matrices must have shape (size, size).
 *
 * @see multiply_UL_inplace()
 */
static void multiply_LU_inplace(
        const int size,
        complex_type *a,
        const int a_dim0,
        const complex_type *b,
        const int b_dim0
        )
{
#ifdef DEBUG
    fprintf(stderr, "Starting multiply_LU_inplace %d\n", size);
#endif
    if (size < TRIANGULAR_OPTIMIZE_THRESHOLD)
    {
#ifdef CBLAS
        trmm(
                CblasColMajor, // layout
                CblasRight, // order: this means B := B A
                CblasUpper, // A is an upper triangular matrix
                CblasNoTrans, // A is not modified (no adjoint or transpose)
                CblasNonUnit, // A is not unit triangular
                size, // rows of B (int)
                size, // columns of B (int)
                &one, // global prefactor
                b, // matrix A
                b_dim0, // first dimension of A (int)
                a, // matrix B
                a_dim0  // first dimension of B (int)
                );
#else /* CBLAS */
        trmm(
                &R, // order: this means B := B A
                &U, // A is a lower triangular matrix
                &N, // A is not modified (no adjoint or transpose)
                &N, // A is not unit triangular
                &size, // rows of B (int)
                &size, // columns of B (int)
                &one, // global prefactor
                b, // matrix A
                &b_dim0, // first dimension of A (int)
                a, // matrix B
                &a_dim0  // first dimension of B (int)
                );
#endif /* CBLAS */
        return;
    }
    /* TODO: The following commands can be parallelized (partially) */
    /*
     * Matrices are labeled as follows:
     *
     * A = [ Aa  Ab ]  B = [ Ba  Bb ]
     *     [ Ac  Ad ]      [ Bc  Bd ]
     * Initially Ab = Bc = 0.
     */
    const int
        part1 = size / 2,
        part2 = size - part1;

    /* Step 1: overwrite Ab with Aa Bb.
     * This requires that first Bb is copied to Ab. */
    a += a_dim0*part1;
    b += b_dim0*part1;
    int i=-1;
    while (++i<part2)
        memcpy( a + i*a_dim0, b + i*b_dim0, part1 * sizeof(complex_type) );
    a -= a_dim0*part1;
    b -= b_dim0*part1;

#ifdef CBLAS
    trmm(
            CblasColMajor, // layout
            CblasLeft, // order: this means B := A B
            CblasLower, // A is a lower triangular matrix
            CblasNoTrans, // A is not modified (no adjoint or transpose)
            CblasNonUnit, // A is not unit triangular
            part1, // rows of B (int)
            part2, // columns of B (int)
            &one, // global prefactor
            a, // matrix A
            a_dim0, // first dimension of A (int)
            a + a_dim0*part1, // matrix B
            a_dim0  // first dimension of B (int)
            );
#else /* CBLAS */
    trmm(
            &L, // order: this means B := A B
            &L, // A is a lower triangular matrix
            &N, // A is not modified (no adjoint or transpose)
            &N, // A is not unit triangular
            &part1, // rows of B (int)
            &part2, // columns of B (int)
            &one, // global prefactor
            a, // matrix A
            &a_dim0, // first dimension of A (int)
            a + a_dim0*part1, // matrix B
            &a_dim0  // first dimension of B (int)
            );
#endif /* CBLAS */

    /* Step 2: overwrite Aa with Aa Ba */
    multiply_LU_inplace(part1, a, a_dim0, b, b_dim0);
    /* Step 3: overwrite Ad with Ad Bd */
    multiply_LU_inplace(
            part2,
            a + (a_dim0+1)*part1,
            a_dim0,
            b + (b_dim0+1)*part1,
            b_dim0);
    /* Step 4: add Ac Bb to Ad */
#ifdef CBLAS
    gemm(
            CblasColMajor, // layout
            CblasNoTrans, // A is not modified (no adjoint or transpose)
            CblasNoTrans, // B is not modified (no adjoint or transpose)
            part2, // rows of A (int)
            part2, // columns of B (int)
            part1, // columns of A = rows of B (int)
            &one, // global prefactor
            a + part1, // matrix A
            a_dim0, // first dimension of A (int)
            b + part1*b_dim0, // matrix B
            b_dim0,  // first dimension of B (int)
            &one, // weight of C
            a + (a_dim0+1)*part1, // matrix C
            a_dim0 // first dimension of C
            );
#else /* CBLAS */
    gemm(
            &N, // A is not modified (no adjoint or transpose)
            &N, // B is not modified (no adjoint or transpose)
            &part2, // rows of A (int)
            &part2, // columns of B (int)
            &part1, // columns of A = rows of B (int)
            &one, // global prefactor
            a + part1, // matrix A
            &a_dim0, // first dimension of A (int)
            b + part1*b_dim0, // matrix B
            &b_dim0,  // first dimension of B (int)
            &one, // weight of C
            a + (a_dim0+1)*part1, // matrix C
            &a_dim0 // first dimension of C
            );
#endif /* CBLAS */

    /* Step 5: overwrite Ac with Ac Ba */
#ifdef CBLAS
    trmm(
            CblasColMajor, // layout
            CblasRight, // order: this means B := B A
            CblasUpper, // A is an upper triangular matrix
            CblasNoTrans, // A is not modified (no adjoint or transpose)
            CblasNonUnit, // A is not unit triangular
            part2, // rows of B (int)
            part1, // columns of B (int)
            &one, // global prefactor
            b, // matrix A
            b_dim0, // first dimension of A (int)
            a + part1, // matrix B
            a_dim0  // first dimension of B (int)
            );
#else /* CBLAS */
    trmm(
            &R, // order: this means B := B A
            &U, // A is an upper triangular matrix
            &N, // A is not modified (no adjoint or transpose)
            &N, // A is not unit triangular
            &part2, // rows of B (int)
            &part1, // columns of B (int)
            &one, // global prefactor
            b, // matrix A
            &b_dim0, // first dimension of A (int)
            a + part1, // matrix B
            &a_dim0  // first dimension of B (int)
            );
#endif /* CBLAS */
}


/**
 * @file
 * @section sec_extend Extend matrix
 */

/**
 * @param input in_rows × in_cols
 * @param output (in_rows + 2*cutoff) × (in_cols + 2*cutoff)
 */
static void extend_matrix_worker(
        const int nrow_in,
        const int ncol_in,
        const int cutoff,
        const complex_type *input,
        const int in_dim0,
        complex_type *output,
        const int out_dim0
        )
{
    /* Copy the matrix. */
    complex_type *auxptr = output + cutoff * (out_dim0 + 1);
    int i=-1;
    while (++i<ncol_in)
        memcpy( auxptr + i*out_dim0, input + i*in_dim0, nrow_in * sizeof(complex_type) );

    if (cutoff <= 0)
        return;

    /* outptr points to the first element of the original matrix in the extended matrix. */
#ifdef PARALLEL
#pragma omp sections
    {
#pragma omp section
#endif /* PARALLEL */
        extrapolate_top_full(
                auxptr,
                nrow_in+2*cutoff,
                output,
                cutoff,
                nrow_in+2*cutoff
                );
#ifdef PARALLEL
#pragma omp section
#endif
        extrapolate_left_full(
                auxptr,
                nrow_in+2*cutoff,
                output,
                cutoff,
                nrow_in+2*cutoff
                );
#ifdef PARALLEL
#pragma omp section
#endif
        extrapolate_bottom_full(
                auxptr + (ncol_in - 1) * out_dim0 + nrow_in - 1,
                nrow_in+2*cutoff,
                output + ncol_in * out_dim0 + nrow_in + cutoff,
                cutoff,
                nrow_in+2*cutoff
                );
#ifdef PARALLEL
#pragma omp section
#endif
        extrapolate_right_full(
                auxptr + (ncol_in - 1) * out_dim0 + nrow_in - 1,
                nrow_in+2*cutoff,
                output + (ncol_in + cutoff) * out_dim0 + nrow_in,
                cutoff,
                nrow_in+2*cutoff
                );
#ifdef PARALLEL
    }
#endif
}

/**
 * Given a Fortran-contiguous square matrix, extend it by linear extrapolation
 * in each direction by <cutoff> rows/columns.
 */
static PyArrayObject* extend_matrix_nd(PyArrayObject *input, const int cutoff)
{
    const int ndim = PyArray_NDIM(input);
    npy_intp *shape = malloc( ndim*sizeof(npy_intp) );
    memcpy( shape, PyArray_DIMS(input), ndim*sizeof(npy_intp) );
    shape[0] += 2*cutoff;
    shape[1] += 2*cutoff;
    PyArrayObject *output = (PyArrayObject*) PyArray_ZEROS(ndim, shape, NPY_COMPLEX_TYPE, 1);
    if (!output)
        return NULL;

    const int
        in_dim0 = PyArray_STRIDE(input, 1) / sizeof(complex_type),
        out_dim0 = PyArray_STRIDE(output, 1) / sizeof(complex_type),
        in_matrixstride = PyArray_STRIDE(input, 2),
        out_matrixstride = PyArray_STRIDE(output, 2);
    int i=1, nmatrices=1;
    while (++i<ndim)
        nmatrices *= shape[i];

    for (i=0; i<nmatrices; ++i)
        extend_matrix_worker(
                PyArray_DIM(input, 0),
                PyArray_DIM(input, 1),
                cutoff,
                PyArray_DATA(input) + i*in_matrixstride,
                in_dim0,
                PyArray_DATA(output) + i*out_matrixstride,
                out_dim0
                );

    return output;
}

#ifdef PARALLEL_EXTRA_DIMS
/**
 * Given a Fortran-contiguous square matrix, extend it by linear extrapolation
 * in each direction by <cutoff> rows/columns.
 */
static PyArrayObject* extend_matrix_nd_parallel(PyArrayObject *input, const int cutoff)
{
    const int ndim = PyArray_NDIM(input);
    npy_intp *shape = malloc( ndim*sizeof(npy_intp) );
    memcpy( shape, PyArray_DIMS(input), ndim*sizeof(npy_intp) );
    shape[0] += 2*cutoff;
    shape[1] += 2*cutoff;
    PyArrayObject *output = (PyArrayObject*) PyArray_ZEROS(ndim, shape, NPY_COMPLEX_TYPE, 1);
    if (!output)
        return NULL;

    const int
        in_dim0 = PyArray_STRIDE(input, 1) / sizeof(complex_type),
        out_dim0 = PyArray_STRIDE(output, 1) / sizeof(complex_type),
        in_matrixstride = PyArray_STRIDE(input, 2),
        out_matrixstride = PyArray_STRIDE(output, 2);
    int nmatrices=1;
    for (int j=1; ++j<ndim;)
        nmatrices *= shape[j];

#pragma omp for
    for (int i=0; i<nmatrices; ++i)
        extend_matrix_worker(
                PyArray_DIM(input, 0),
                PyArray_DIM(input, 1),
                cutoff,
                PyArray_DATA(input) + i*in_matrixstride,
                in_dim0,
                PyArray_DATA(output) + i*out_matrixstride,
                out_dim0
                );

    return output;
}
#endif /* PARALLEL_EXTRA_DIMS */

/**
 * Take an n×n Floquet matrix M and positive integer c as arguments; Extrapolate
 * M to shape (n+2c)×(n+2c).
 */
static PyObject* extend_matrix(PyObject *self, PyObject *args)
{
    PyArrayObject *input, *output;
    int cutoff;
    /* Parse the arguments: input should be an array and an integer. */
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &input, &cutoff))
        return NULL;

    if (PyArray_TYPE(input) != NPY_COMPLEX_TYPE)
        return PyErr_Format(
                PyExc_ValueError,
                "array of type complex128 required");
    if (PyArray_NDIM(input) < 2)
        return PyErr_Format(
                PyExc_ValueError,
                "1st argument must have at least 2 dimensions.");

    if (cutoff <= 0)
    {
        Py_INCREF(input);
        return (PyObject*) input;
    }

    if ((PyArray_DIM(input, 0) < cutoff + 3) || (PyArray_DIM(input, 1) < cutoff + 3))
        return PyErr_Format(
                PyExc_ValueError,
                "Matrix is too small or cutoff too large.");

    PyArrayObject *finput = (PyArrayObject*) PyArray_FromArray(
                input,
                PyArray_DescrFromType(NPY_COMPLEX_TYPE),
                NPY_ARRAY_WRITEABLE
                    | NPY_ARRAY_F_CONTIGUOUS
                    | NPY_ARRAY_ALIGNED
            );
    if (!finput)
        return PyErr_Format(PyExc_RuntimeError, "Failed to create array");

    if (PyArray_NDIM(finput) == 2)
    {
        npy_intp shape[] = {
            PyArray_DIM(finput, 0) + 2*cutoff,
            PyArray_DIM(finput, 1) + 2*cutoff
        };
        output = (PyArrayObject*) PyArray_ZEROS(2, shape, NPY_COMPLEX_TYPE, 1);
        if (output)
            extend_matrix_worker(
                    PyArray_DIM(finput, 0),
                    PyArray_DIM(finput, 1),
                    cutoff,
                    PyArray_DATA(finput),
                    PyArray_STRIDE(finput, 1)/sizeof(complex_type),
                    PyArray_DATA(output),
                    PyArray_STRIDE(output, 1)/sizeof(complex_type)
                    );
    }
#ifdef PARALLEL_EXTRA_DIMS
    else if (omp_get_max_threads() > 1)
        output = extend_matrix_nd_parallel(finput, cutoff);
#endif /* PARALLEL_EXTRA_DIMS */
    else
        output = extend_matrix_nd(finput, cutoff);
    Py_DECREF(finput);

    return (PyObject*) output;
}


/**
 * @file
 * @section sec_invert Invert matrix
 */

/**
 * matrix must be Fortran contiguous
 */
void invert_matrix(
        complex_type *matrix,
        const int size,
        const int dim0,
        int *status
        )
{
    int *const ipiv = malloc( size * sizeof(int) );
    if (!ipiv)
    {
        *status = 1;
        return;
    }

    getrf(&size, &size, matrix, &dim0, ipiv, status);
    if (*status != 0)
    {
        free( ipiv );
        return;
    }

    int lwork = -1;
    complex_type dummy_work;
    getri(&size, matrix, &dim0, ipiv, &dummy_work, &lwork, status);
    lwork = (int) dummy_work;
#ifdef DEBUG
    fprintf(stderr, "LWORK = %d\n", lwork);
#endif
    if (lwork < size)
        lwork = size;
    complex_type *work = malloc( lwork * sizeof(complex_type) );
    if (work)
    {
        getri(&size, matrix, &dim0, ipiv, work, &lwork, status);
        free( work );
    }
    else
        *status = 1;

    free( ipiv );
}


/**
 * @see invert_nd()
 * @see invert_matrix()
 */
static PyArrayObject *invert_2d(
        PyArrayObject *finput,
        const int cutoff,
        const int reduce_cutoff
        )
{
    const int
        size = PyArray_DIM(finput, 0),
        /* TODO: better aligned arrays for better performance */
        extended_stride = size + 2*cutoff;

    complex_type *extended = calloc( (size + 2*cutoff) * extended_stride, sizeof(complex_type) );
    if (!extended)
        return NULL;

    extend_matrix_worker(
            size,
            size,
            cutoff,
            PyArray_DATA(finput),
            PyArray_STRIDE(finput, 1)/sizeof(complex_type),
            extended,
            extended_stride
            );

    int status;
    invert_matrix(
            extended + reduce_cutoff * (extended_stride + 1),
            size + 2*(cutoff-reduce_cutoff),
            extended_stride,
            &status
            );
    PyArrayObject *output;
    if (status)
    {
        output = NULL;
        PyErr_SetString(PyExc_ValueError, "encountered singular matrix.");
    }
    else
    {
        output = (PyArrayObject*) PyArray_EMPTY(2, PyArray_DIMS(finput), NPY_COMPLEX_TYPE, 1);
        if (output)
            for (int i=0; i<size; ++i)
                memcpy(
                        PyArray_GETPTR2(output, 0, i),
                        extended + cutoff + (i+cutoff)*extended_stride,
                        size*sizeof(complex_type)
                        );
    }

    free( extended );
    return output;
}

/**
 * input must have shape (n, n, ...) with n > cutoff+2 and cutoff >= reduce_cutoff.
 * This is not cheked!
 *
 * @see invert_2d()
 * @see invert_nd_parallel()
 * @see invert_matrix()
 */
static PyArrayObject *invert_nd(
        PyArrayObject *input,
        const int cutoff,
        const int reduce_cutoff
        )
{
    const int
        size = PyArray_DIM(input, 0),
        /* TODO: better aligned arrays for better performance */
        extended_stride = size + 2*cutoff;

    const int ndim = PyArray_NDIM(input);
    PyArrayObject *output = (PyArrayObject*) PyArray_EMPTY(ndim, PyArray_DIMS(input), NPY_COMPLEX_TYPE, 1);
    if (!output)
        return NULL;

    const int
        in_matrixstride = PyArray_STRIDE(input, 2),
        out_matrixstride = PyArray_STRIDE(output, 2),
        out_colstride = PyArray_STRIDE(output, 1);
    int i=1, nmatrices=1, status, j;
    while (++i<ndim)
        nmatrices *= PyArray_DIM(input, i);

    complex_type *extended = malloc( (size + 2*cutoff) * extended_stride * sizeof(complex_type) );
    if (!extended)
    {
        Py_DECREF(output);
        return NULL;
    }

    void *outptr;
    for (i=0; i<nmatrices; ++i)
    {
        memset( extended, 0, (size + 2*cutoff) * extended_stride * sizeof(complex_type) );

        extend_matrix_worker(
                size,
                size,
                cutoff,
                PyArray_DATA(input) + i*in_matrixstride,
                PyArray_STRIDE(input, 1)/sizeof(complex_type),
                extended,
                extended_stride
                );

        invert_matrix(
                extended + reduce_cutoff * (extended_stride + 1),
                size + 2*(cutoff-reduce_cutoff),
                extended_stride,
                &status
                );
        if (status)
            PyErr_WarnEx(PyExc_RuntimeWarning, "encountered singular matrix.", 1);
        outptr = PyArray_DATA(output) + i*out_matrixstride;
        for (j=0; j<size; ++j)
            memcpy(
                    outptr + j*out_colstride,
                    extended + cutoff + (j+cutoff)*extended_stride,
                    size*sizeof(complex_type)
                    );
    }

    free( extended );
    return output;
}

#ifdef PARALLEL_EXTRA_DIMS
/**
 * input must have shape (n, n, ...) with n > cutoff+2 and cutoff >= reduce_cutoff.
 * This is not cheked!
 *
 * @see invert_nd()
 * @see invert_2d()
 * @see invert_matrix()
 */
static PyArrayObject *invert_nd_parallel(
        PyArrayObject *input,
        const int cutoff,
        const int reduce_cutoff
        )
{
    const int
        size = PyArray_DIM(input, 0),
        /* TODO: better aligned arrays for better performance */
        extended_stride = size + 2*cutoff;

    const int ndim = PyArray_NDIM(input);
    PyArrayObject *output = (PyArrayObject*) PyArray_EMPTY(ndim, PyArray_DIMS(input), NPY_COMPLEX_TYPE, 1);
    if (!output)
        return NULL;

    const int
        in_matrixstride = PyArray_STRIDE(input, 2),
        out_matrixstride = PyArray_STRIDE(output, 2),
        out_colstride = PyArray_STRIDE(output, 1);
    int nmatrices=1;
    for (int j=1; ++j<ndim;)
        nmatrices *= PyArray_DIM(input, j);

    void *outptr;

    char fatal_error = 0;
#pragma omp for
    for (int i=0; i<nmatrices; ++i)
    {
        if (fatal_error)
            continue;
        int status;
        complex_type *extended = calloc( (size + 2*cutoff) * extended_stride, sizeof(complex_type) );
        if (!extended)
        {
            fatal_error = 1;
            continue;
        }

        extend_matrix_worker(
                size,
                size,
                cutoff,
                PyArray_DATA(input) + i*in_matrixstride,
                PyArray_STRIDE(input, 1)/sizeof(complex_type),
                extended,
                extended_stride
                );

        invert_matrix(
                extended + reduce_cutoff * (extended_stride + 1),
                size + 2*(cutoff-reduce_cutoff),
                extended_stride,
                &status
                );
        if (status)
            PyErr_WarnEx(PyExc_RuntimeWarning, "encountered singular matrix.", 1);
        outptr = PyArray_DATA(output) + i*out_matrixstride;
        for (int j=0; j<size; ++j)
            memcpy(
                    outptr + j*out_colstride,
                    extended + cutoff + (j+cutoff)*extended_stride,
                    size*sizeof(complex_type)
                    );
        free( extended );
    }
    if (fatal_error)
    {
        Py_DECREF(output);
        return PyErr_Format(PyExc_RuntimeError, "Error in matrix inversion: memory allocation");
    }

    return output;
}
#endif /* PARALLEL_EXTRA_DIMS */

/**
 * @see invert_nd()
 * @see invert_2d()
 */
static PyObject* invert_extended(PyObject *self, PyObject *args)
{
    PyArrayObject *input, *output;
    int cutoff, reduce_cutoff = 0;
    /* Parse the arguments: input should be an array and an integer. */
    if (!PyArg_ParseTuple(args, "O!i|i", &PyArray_Type, &input, &cutoff, &reduce_cutoff))
        return NULL;

    if (PyArray_TYPE(input) != NPY_COMPLEX_TYPE)
        return PyErr_Format(PyExc_ValueError, "array of type complex128 required");
    if (PyArray_NDIM(input) < 2)
        return PyErr_Format(PyExc_ValueError, "1st argument must have at least 2 dimensions.");
    if (PyArray_DIM(input, 1) != PyArray_DIM(input, 0))
        return PyErr_Format(PyExc_ValueError, "1st argument must be a square matrix.");
    if ((cutoff < 0) || (reduce_cutoff >= cutoff))
    {
        cutoff = 0;
        reduce_cutoff = 0;
    }

    PyArrayObject *finput = (PyArrayObject*) PyArray_FromArray(
                input,
                PyArray_DescrFromType(NPY_COMPLEX_TYPE),
                NPY_ARRAY_WRITEABLE
                    | NPY_ARRAY_F_CONTIGUOUS
                    | NPY_ARRAY_ALIGNED
            );
    if (!finput)
        return PyErr_Format(PyExc_RuntimeError, "Failed to create array");

    if (PyArray_NDIM(finput) == 2)
        output = invert_2d(finput, cutoff, reduce_cutoff);
#ifdef PARALLEL_EXTRA_DIMS
    else if (omp_get_max_threads() > 1)
        output = invert_nd_parallel(finput, cutoff, reduce_cutoff);
#endif /* PARALLEL_EXTRA_DIMS */
    else
        output = invert_nd(finput, cutoff, reduce_cutoff);
    Py_DECREF(finput);


    return (PyObject*) output;
}


/**
 * @file
 * @section sec_multiply Multiply matrices
 */

static void correct_top_left(
        const int cutoff,
        const complex_type *left,
        const int left_dim0,
        const complex_type *right,
        const int right_dim0,
        complex_type *auxarr_left,
        complex_type *auxarr_right,
        const int aux_dim0
        )
{
#ifdef PARALLEL
#pragma omp sections
    {
#pragma omp section
#endif /* PARALLEL */
        extrapolate_left(left, left_dim0, auxarr_left, cutoff, aux_dim0);
#ifdef PARALLEL
#pragma omp section
#endif
        extrapolate_top(right, right_dim0, auxarr_right, cutoff, aux_dim0);
#ifdef PARALLEL
    }
#endif

    /* Calculate matrix product. This overwrites auxarr_left. */
#ifdef DEBUG
    fprintf(stderr, "Calling multiply_UL_inplace for extrapolated matrices\n");
#endif
    multiply_UL_inplace(
            cutoff, // size
            auxarr_left, // matrix A
            aux_dim0,  // first dimension of A (int)
            auxarr_right, // matrix B
            aux_dim0 // first dimension of B (int)
            );
}


static void correct_bottom_right(
        const int cutoff,
        const complex_type *left_last,
        const int left_dim0,
        const complex_type *right_last,
        const int right_dim0,
        complex_type *auxarr_left,
        complex_type *auxarr_right,
        const int aux_dim0
        )
{
#ifdef PARALLEL
#pragma omp sections
    {
#pragma omp section
#endif /* PARALLEL */
        extrapolate_right(left_last, left_dim0, auxarr_left, cutoff, aux_dim0);
#ifdef PARALLEL
#pragma omp section
#endif
        extrapolate_bottom(right_last, right_dim0, auxarr_right, cutoff, aux_dim0);
#ifdef PARALLEL
    }
#endif

    /* Calculate matrix product. This overwrites auxarr_left. */
#ifdef DEBUG
    fprintf(stderr, "Calling multiply_LU_inplace for extrapolated matrices\n");
#endif
    multiply_LU_inplace(
            cutoff,
            auxarr_left,
            aux_dim0,
            auxarr_right,
            aux_dim0
            );
}

/**
 * Helper function for multiply_extended().
 * auxarr1 and auxarr3 must be initialized to 0.
 * auxarr1 == auxarr3 and auxarr2 == auxarr4 is allowed and will avoid parallelization.
 * auxarr arrays must have size cutoff*aux_dim0 with aux_dim0 >= cutoff.
 */
static void multiply_extended_worker(
        const int cutoff,
        const int nrowl,
        const int ncoll,
        const int ncolr,
        const complex_type *left,
        const int left_dim0,
        const complex_type *right,
        const int right_dim0,
        complex_type *output,
        const int out_dim0,
        complex_type *auxarr1,
        complex_type *auxarr2,
#ifdef PARALLEL
        complex_type *auxarr3,
        complex_type *auxarr4,
#endif
        const int aux_dim0,
        const int clear_corners
        )
{
#ifdef CBLAS
    gemm(
            CblasColMajor, // layout
            CblasNoTrans, // A is not modified (no adjoint or transpose)
            CblasNoTrans, // B is not modified (no adjoint or transpose)
            nrowl, // rows of A (int)
            ncolr, // columns of B (int)
            ncoll, // columns of A = rows of B (int)
            &one, // global prefactor
            left, // matrix A
            left_dim0, // first dimension of A (int)
            right, // matrix B
            right_dim0,  // first dimension of B (int)
            &zero, // weight of C
            output, // matrix C
            out_dim0 // first dimension of C
            );
#else /* CBLAS */
    gemm(
            &N, // A is not modified (no adjoint or transpose)
            &N, // B is not modified (no adjoint or transpose)
            &nrowl, // rows of A (int)
            &ncolr, // columns of B (int)
            &ncoll, // columns of A = rows of B (int)
            &one, // global prefactor
            left, // matrix A
            &left_dim0, // first dimension of A (int)
            right, // matrix B
            &right_dim0,  // first dimension of B (int)
            &zero, // weight of C
            output, // matrix C
            &out_dim0 // first dimension of C
            );
#endif /* CBLAS */

    if (clear_corners > 0)
    {
#ifdef PARALLEL_EXTRAPOLATION
#pragma omp for
#endif
        for (int i=0; i<clear_corners; i++)
            memset(
                    output + (ncolr-clear_corners+i)*out_dim0,
                    0,
                    (i+1)*sizeof(complex_type)
                    );
#ifdef PARALLEL_EXTRAPOLATION
#pragma omp for
#endif
        for (int i=0; i<clear_corners; i++)
            memset(
                    output + i*(out_dim0+1) + nrowl - clear_corners,
                    0,
                    (clear_corners-i)*sizeof(complex_type)
                    );
    }

    if (cutoff <= 0)
        return;

#ifdef PARALLEL
    if (auxarr1 == auxarr3 || auxarr2 == auxarr4)
    {
#endif
        correct_top_left(cutoff, left, left_dim0, right, right_dim0, auxarr1, auxarr2, aux_dim0);

        /* Add result to top left of output array. */
#ifdef DEBUG
        fprintf(stderr, "Writing result to top left\n");
#endif
        const complex_type *read = auxarr1;
#ifdef PARALLEL
        const complex_type *end = auxarr1 + cutoff * aux_dim0;
#else
        const complex_type *const end = auxarr1 + cutoff * aux_dim0;
#endif
        complex_type *write = output;
        int i;
        while (read < end)
        {
            i = -1;
            while (++i < cutoff)
                write[i] += read[i];
            write += out_dim0;
            read += aux_dim0;
        }

#ifdef PARALLEL
        if (auxarr1 == auxarr3)
            memset( auxarr3, 0, cutoff*aux_dim0*sizeof(complex_type) );
#else
        memset( auxarr1, 0, cutoff*aux_dim0*sizeof(complex_type) );
#endif
        correct_bottom_right(
                cutoff,
                left + left_dim0*(ncoll-1) + nrowl-1,
                left_dim0,
                right + right_dim0*(ncolr-1) + ncoll-1,
                right_dim0,
#ifdef PARALLEL
                auxarr3,
                auxarr4,
#else
                auxarr1,
                auxarr2,
#endif
                aux_dim0);

        /* Add result to bottom right of output array. */
#ifdef DEBUG
        fprintf(stderr, "Writing result to bottom right\n");
#endif
#ifdef PARALLEL
        read = auxarr3;
        end = auxarr3 + cutoff * aux_dim0;
#else
        read = auxarr1;
#endif
        write = output + (ncolr - cutoff)*out_dim0 + nrowl - cutoff;
        while (read < end)
        {
            i = -1;
            while (++i < cutoff)
                write[i] += read[i];
            write += out_dim0;
            read += aux_dim0;
        }
        return;
#ifdef PARALLEL
    }
    correct_top_left(cutoff, left, left_dim0, right, right_dim0, auxarr1, auxarr2, aux_dim0);
    correct_bottom_right(
            cutoff,
            left + left_dim0*(ncoll-1) + nrowl-1,
            left_dim0,
            right + right_dim0*(ncolr-1) + ncoll-1,
            right_dim0,
            auxarr3,
            auxarr4,
            aux_dim0);

    /* Add result to top left of output array. */
#ifdef DEBUG
    fprintf(stderr, "Writing result to top left\n");
#endif
    const complex_type *read = auxarr1;
    const complex_type * end = auxarr1 + cutoff * aux_dim0;
    complex_type *write = output;
    int i;
    while (read < end)
    {
        i = -1;
        while (++i < cutoff)
            write[i] += read[i];
        write += out_dim0;
        read += aux_dim0;
    }

    /* Add result to bottom right of output array. */
#ifdef DEBUG
    fprintf(stderr, "Writing result to bottom right\n");
#endif
    read = auxarr3;
    end = auxarr3 + cutoff * aux_dim0;
    write = output + (ncolr - cutoff)*out_dim0 + nrowl - cutoff;
    while (read < end)
    {
        i = -1;
        while (++i < cutoff)
            write[i] += read[i];
        write += out_dim0;
        read += aux_dim0;
    }
#endif /* PARALLEL */
}


/**
 * left and right must be Fortran contiguous.
 */
static PyArrayObject* multiply_extended_2d(
        PyArrayObject *left,
        PyArrayObject *right,
        const int cutoff,
        const int clear_corners
        )
{
    /* First just ordinary matrix multiplication. */
    npy_intp dims[2] = {PyArray_DIM(left, 0), PyArray_DIM(right, 1)};
    PyArrayObject *out = (PyArrayObject*) PyArray_EMPTY(2, dims, NPY_COMPLEX_TYPE, 1);
    if (!out)
        return NULL;

    complex_type
        *auxarr1 = calloc( cutoff*cutoff, sizeof(complex_type) ),
        *auxarr2 = malloc( cutoff*cutoff* sizeof(complex_type) );
#ifdef PARALLEL
    complex_type *auxarr3, *auxarr4;
    if (omp_get_max_threads() >= PARALLELIZE_CORRECTION_THREADS_THRESHOLD)
    {
        auxarr3 = calloc( cutoff*cutoff, sizeof(complex_type) ),
        auxarr4 = malloc( cutoff*cutoff* sizeof(complex_type) );
    }
    else
    {
        auxarr3 = auxarr1;
        auxarr4 = auxarr2;
    }
#endif /* PARALLEL */

#ifdef PARALLEL
    if (auxarr1 && auxarr2 && auxarr3 && auxarr4)
#else /* PARALLEL */
    if (auxarr1 && auxarr2)
#endif /* PARALLEL */
    {
        multiply_extended_worker(
                cutoff,
                PyArray_DIM(left, 0),
                PyArray_DIM(left, 1),
                PyArray_DIM(right, 1),
                PyArray_DATA(left),
                PyArray_STRIDE(left, 1)/sizeof(complex_type),
                PyArray_DATA(right),
                PyArray_STRIDE(right, 1)/sizeof(complex_type),
                PyArray_DATA(out),
                PyArray_STRIDE(out, 1)/sizeof(complex_type),
                auxarr1,
                auxarr2,
#ifdef PARALLEL
                auxarr3,
                auxarr4,
#endif /* PARALLEL */
                cutoff,
                clear_corners
                );
    }
    else
    {
        Py_DECREF(out);
        out = NULL;
    }

#ifdef PARALLEL
    if (auxarr1 != auxarr3)
        free( auxarr3 );
    if (auxarr2 != auxarr4)
        free( auxarr4 );
#endif /* PARALLEL */
    free( auxarr1 );
    free( auxarr2 );
    return out;
}

/**
 * left and right must be Fortran contiguous.
 */
static PyArrayObject* multiply_extended_nd(
        PyArrayObject *left,
        PyArrayObject *right,
        const int cutoff,
        const int clear_corners
        )
{
    const int
        nrowl = PyArray_DIM(left, 0),
        ncoll = PyArray_DIM(left, 1),
        ncolr = PyArray_DIM(right, 1);


    const int ndim = PyArray_NDIM(left);
    npy_intp *shape = malloc( ndim*sizeof(npy_intp) );
    memcpy( shape, PyArray_DIMS(left), ndim*sizeof(npy_intp) );
    shape[1] = ncolr;
    PyArrayObject *out = (PyArrayObject*) PyArray_EMPTY(ndim, shape, NPY_COMPLEX_TYPE, 1);
    if (!out)
        return NULL;

    const int
        left_dim0 = PyArray_STRIDE(left, 1) / sizeof(complex_type),
        right_dim0 = PyArray_STRIDE(right, 1) / sizeof(complex_type),
        out_dim0 = PyArray_STRIDE(out, 1) / sizeof(complex_type),
        left_matrixstride = PyArray_STRIDE(left, 2),
        right_matrixstride = PyArray_STRIDE(right, 2),
        out_matrixstride = PyArray_STRIDE(out, 2);

    int i=1, nmatrices=1;
    while (++i<ndim)
        nmatrices *= shape[i];

    complex_type
        *auxarr1 = malloc( cutoff*cutoff * sizeof(complex_type) ),
        *auxarr2 = malloc( cutoff*cutoff * sizeof(complex_type) );
#ifdef PARALLEL
    complex_type *auxarr3, *auxarr4;
    if (omp_get_max_threads() >= PARALLELIZE_CORRECTION_THREADS_THRESHOLD)
    {
        auxarr3 = malloc( cutoff*cutoff* sizeof(complex_type) ),
        auxarr4 = malloc( cutoff*cutoff* sizeof(complex_type) );
    }
    else
    {
        auxarr3 = auxarr1;
        auxarr4 = auxarr2;
    }
#endif /* PARALLEL */

#ifdef PARALLEL
    if (auxarr1 && auxarr2 && auxarr3 && auxarr4)
#else /* PARALLEL */
    if (auxarr1 && auxarr2)
#endif /* PARALLEL */
    {
        for (i=0; i<nmatrices; ++i)
        {
            memset( auxarr1, 0, cutoff*cutoff*sizeof(complex_type) );
#ifdef PARALLEL
            if (auxarr1 != auxarr3)
                memset( auxarr3, 0, cutoff*cutoff*sizeof(complex_type) );
#endif /* PARALLEL */
            multiply_extended_worker(
                    cutoff,
                    nrowl,
                    ncoll,
                    ncolr,
                    PyArray_DATA(left) + i*left_matrixstride,
                    left_dim0,
                    PyArray_DATA(right) + i*right_matrixstride,
                    right_dim0,
                    PyArray_DATA(out) + i*out_matrixstride,
                    out_dim0,
                    auxarr1,
                    auxarr2,
#ifdef PARALLEL
                    auxarr3,
                    auxarr4,
#endif /* PARALLEL */
                    cutoff,
                    clear_corners
                    );
        }
    }
    else
    {
        Py_DECREF(out);
        out = NULL;
    }

#ifdef PARALLEL
    if (auxarr1 != auxarr3)
        free( auxarr3 );
    if (auxarr2 != auxarr4)
        free( auxarr4 );
#endif /* PARALLEL */
    free( auxarr1 );
    free( auxarr2 );
    return out;
}

#ifdef PARALLEL_EXTRA_DIMS
/**
 * left and right must be Fortran contiguous.
 */
static PyArrayObject* multiply_extended_nd_parallel(
        PyArrayObject *left,
        PyArrayObject *right,
        const int cutoff,
        const int clear_corners
        )
{
    const int
        nrowl = PyArray_DIM(left, 0),
        ncoll = PyArray_DIM(left, 1),
        ncolr = PyArray_DIM(right, 1);


    const int ndim = PyArray_NDIM(left);
    npy_intp *shape = malloc( ndim*sizeof(npy_intp) );
    memcpy( shape, PyArray_DIMS(left), ndim*sizeof(npy_intp) );
    shape[1] = ncolr;
    PyArrayObject *out = (PyArrayObject*) PyArray_EMPTY(ndim, shape, NPY_COMPLEX_TYPE, 1);
    if (!out)
        return NULL;

    const int
        left_dim0 = PyArray_STRIDE(left, 1) / sizeof(complex_type),
        right_dim0 = PyArray_STRIDE(right, 1) / sizeof(complex_type),
        out_dim0 = PyArray_STRIDE(out, 1) / sizeof(complex_type),
        left_matrixstride = PyArray_STRIDE(left, 2),
        right_matrixstride = PyArray_STRIDE(right, 2),
        out_matrixstride = PyArray_STRIDE(out, 2);

    int i=1, nmatrices=1;
    while (++i<ndim)
        nmatrices *= shape[i];

    char fatal_error = 0;
#pragma omp for
    for (i=0; i<nmatrices; ++i)
    {
        if (fatal_error)
            continue;
        complex_type
            *auxarr_left  = calloc( cutoff*cutoff,  sizeof(complex_type) ),
            *auxarr_right = malloc( cutoff*cutoff * sizeof(complex_type) );

        if (auxarr_left && auxarr_right)
            multiply_extended_worker(
                    cutoff,
                    nrowl,
                    ncoll,
                    ncolr,
                    PyArray_DATA(left) + i*left_matrixstride,
                    left_dim0,
                    PyArray_DATA(right) + i*right_matrixstride,
                    right_dim0,
                    PyArray_DATA(out) + i*out_matrixstride,
                    out_dim0,
                    auxarr_left,
                    auxarr_right,
#ifdef PARALLEL
                    auxarr_left,
                    auxarr_right,
#endif /* PARALLEL */
                    cutoff,
                    clear_corners
                    );
        else
            fatal_error = 1;

        free( auxarr_left );
        free( auxarr_right );
    }
    if (fatal_error)
    {
        Py_DECREF(out);
        out = NULL;
    }
    return out;
}
#endif /* PARALLEL_EXTRA_DIMS */


/**
 * left and right must be Fortran contiguous. right must not be a square matrix.
 * left is overwritten with the product of left and right.
 * left must be large enough to contain the matrix product left*right.
 */
static void multiply_symmetric_nonsquare(
        const int nrowl,
        const int ncoll,
        const int ncolr,
        complex_type *left,
        const int left_dim0,
        const complex_type *right,
        const int right_dim0,
        const int symmetry,
        const int clear_corners
        )
{
    /*
     * Calculate left := left @ right while assuming that right is a lower
     * triangular matrix (upper triangular matrix would also work).
     * Using this result and the symmetry of left and right, the correct matrix
     * product can be calculated from left.
     */
#ifdef DEBUG
    fprintf(stderr, "Calculating matrix product using trmm\n");
#endif
#ifdef CBLAS
    trmm(
            CblasColMajor, // layout
            CblasRight, // order: this means B := B A
            ncoll > ncolr ? CblasUpper : CblasLower, // A is treated as upper or lower triangular matrix, depending on its shape.
            CblasNoTrans, // A is not modified (no adjoint or transpose)
            CblasNonUnit, // A is not unit triangular
            nrowl, // rows of B (int)
            ncoll > ncolr ? ncolr : ncoll, // columns of B (int) = rows of A
            &one, // global prefactor
            right, // matrix A
            right_dim0, // first dimension of A (int)
            left, // matrix B
            left_dim0  // first dimension of B (int)
            );
#else /* CBLAS */
    trmm(
            &R, // order: this means B := B A
            ncoll > ncolr ? &U : &L, // A is treated as upper or lower triangular matrix, depending on its shape.
            &N, // A is not modified (no adjoint or transpose)
            &N, // A is not unit triangular
            &nrowl, // rows of B (int)
            ncoll > ncolr ? &ncolr : &ncoll, // columns of B (int) = rows of A
            &one, // global prefactor
            right, // matrix A
            &right_dim0, // first dimension of A (int)
            left, // matrix B
            &left_dim0  // first dimension of B (int)
            );
#endif /* CBLAS */

    /*
     * Use symmetry of the matrix to correct the result.
     */
#ifdef DEBUG
    fprintf(stderr, "Correct rest of the matrix\n");
#endif
    complex_type *fwd, *bck;
    int i=0, j;
    for (; i < ncolr - ncoll; ++i)
    {
        fwd = left + i*left_dim0;
        bck = left + (ncolr - i - 1)*left_dim0;
        j = nrowl;
        while (--j >= 0)
            bck[nrowl - j - 1] = symmetry * conj( fwd[j] );
    }
    for (; i < ncolr/2; ++i)
    {
        fwd = left + i*left_dim0;
        bck = left + (ncolr - i - 1)*left_dim0;
        j = nrowl;
        while (--j >= 0)
        {
            fwd[j] += symmetry * conj( bck[nrowl - j - 1] );
            bck[nrowl - j - 1] = symmetry * conj( fwd[j] );
        }
    }
    if (2*i < ncolr)
    {
        fwd = left + i*left_dim0;
        j = nrowl;
        while (--j >= nrowl/2)
        {
            fwd[j] += symmetry * conj( fwd[nrowl - j - 1] );
            fwd[nrowl - j - 1] = symmetry * conj( fwd[j] );
        }
    }

    if (clear_corners > 0)
    {
#ifdef PARALLEL_EXTRAPOLATION
#pragma omp for
#endif
        for (int i=0; i<clear_corners; i++)
            memset(
                    left + (ncolr-clear_corners+i)*left_dim0,
                    0,
                    (i+1)*sizeof(complex_type)
                    );
#ifdef PARALLEL_EXTRAPOLATION
#pragma omp for
#endif
        for (int i=0; i<clear_corners; i++)
            memset(
                    left + i*(left_dim0+1) + nrowl - clear_corners,
                    0,
                    (clear_corners-i)*sizeof(complex_type)
                    );
    }
}

/**
 * left and right must be Fortran contiguous. right must be a square matrix.
 * left is overwritten with the product of left and right.
 * Diagonal elements of right are temporarily overwritten but restored afterwards.
 */
static void multiply_symmetric_square(
        const int nrowl,
        const int ncolr,
        complex_type *left,
        const int left_dim0,
        complex_type *right,
        const int right_dim0,
        const int symmetry,
        const int clear_corners
        )
{
    /* multiply diagonal of right by 1/2 */
    int i = 0, j;
    while (i < ncolr)
        right[i++ * (right_dim0+1)] /= 2;

    /*
     * Calculate left := left @ right while assuming that right is a lower
     * triangular matrix (upper triangular matrix would also work).
     * Using this result and the symmetry of left and right, the correct matrix
     * product can be calculated from left.
     */
#ifdef DEBUG
    fprintf(stderr, "Calculating matrix product using trmm\n");
#endif
#ifdef CBLAS
    trmm(
            CblasColMajor, // layout
            CblasRight, // order: this means B := B A
            CblasLower, // A is interpreted as a lower triangular matrix
            CblasNoTrans, // A is not modified (no adjoint or transpose)
            CblasNonUnit, // A is not unit triangular
            nrowl, // rows of B (int)
            ncolr, // columns of B (int) = rows of A = columns of A
            &one, // global prefactor
            right, // matrix A
            right_dim0, // first dimension of A (int)
            left, // matrix B
            left_dim0 // first dimension of B (int)
            );
#else /* CBLAS */
    trmm(
            &R, // order: this means B := B A
            &L, // A is interpreted as a lower triangular matrix
            &N, // A is not modified (no adjoint or transpose)
            &N, // A is not unit triangular
            &nrowl, // rows of B (int)
            &ncolr, // columns of B (int) = rows of A = columns of A
            &one, // global prefactor
            right, // matrix A
            &right_dim0, // first dimension of A (int)
            left, // matrix B
            &left_dim0 // first dimension of B (int)
            );
#endif /* CBLAS */

    /* multiply diagonal of right by 2 */
#ifdef DEBUG
    fprintf(stderr, "Multiplying diagonal of right array by 2\n");
#endif
    i = 0;
    while (i < ncolr)
        right[i++ * (right_dim0+1)] *= 2;

    /*
     * Use symmetry of the matrix to correct the result.
     */
#ifdef DEBUG
    fprintf(stderr, "Correct rest of the matrix\n");
#endif
    i = 0;
    j = left_dim0 * (ncolr-1) + nrowl - 1;
    while (i <= j)
    {
        left[i] += symmetry * conj(left[j]);
        left[j--] = symmetry * conj(left[i++]);
    }

    if (clear_corners > 0)
    {
#ifdef PARALLEL_EXTRAPOLATION
#pragma omp for
#endif
        for (int i=0; i<clear_corners; i++)
            memset(
                    left + (ncolr-clear_corners+i)*left_dim0,
                    0,
                    (i+1)*sizeof(complex_type)
                    );
#ifdef PARALLEL_EXTRAPOLATION
#pragma omp for
#endif
        for (int i=0; i<clear_corners; i++)
            memset(
                    left + i*(left_dim0+1) + nrowl - clear_corners,
                    0,
                    (clear_corners-i)*sizeof(complex_type)
                    );
    }

#ifdef DEBUG
    fprintf(stderr, "Finished multiply_2d_symmetric\n");
#endif
}


/**
 * left and right must be Fortran contiguous. right must not be a square matrix.
 * left is overwritten with the product of left and right.
 * left must be large enough to contain the matrix product left*right.
 */
static void multiply_symmetric_worker(
        const int nrowl,
        const int ncoll,
        const int ncolr,
        const int symmetry,
        const int cutoff,
        complex_type *const left,
        const int left_dim0,
        complex_type *const right,
        const int right_dim0,
        complex_type *const auxmatrixl,
        complex_type *const auxmatrixr,
        const int aux_dim0,
        const int clear_corners
        )
{
    /* If cutoff == 0 (padding is disabled), just return the matrix product. */
    if (cutoff <= 0)
    {
        if (ncoll == ncolr)
            multiply_symmetric_square(
                    nrowl,
                    ncolr,
                    left,
                    left_dim0,
                    right,
                    right_dim0,
                    symmetry,
                    clear_corners
                    );
        else
            multiply_symmetric_nonsquare(
                    nrowl,
                    ncoll,
                    ncolr,
                    left,
                    left_dim0,
                    right,
                    right_dim0,
                    symmetry,
                    clear_corners
                    );
        return;
    }

#ifdef PARALLEL
#pragma omp sections
    {
#pragma omp section
#endif /* PARALLEL */
        {
        memset( auxmatrixl, 0, cutoff*aux_dim0*sizeof(complex_type) );
        /* Extrapolate left (upper part) of left matrix */
        extrapolate_left(left, nrowl, auxmatrixl, cutoff, cutoff);
        }
#ifdef PARALLEL
#pragma omp section
#endif
        /* Extrapolate top (left part) of right matrix */
        extrapolate_top(right, ncoll, auxmatrixr, cutoff, cutoff);
#ifdef PARALLEL
    }
#endif

    /* Calculate matrix product. This overwrites auxmatrixl. */
    if (ncoll == ncolr)
        multiply_symmetric_square(
                nrowl,
                ncolr,
                left,
                left_dim0,
                right,
                right_dim0,
                symmetry,
                clear_corners
                );
    else
        multiply_symmetric_nonsquare(
                nrowl,
                ncoll,
                ncolr,
                left,
                left_dim0,
                right,
                right_dim0,
                symmetry,
                clear_corners
                );

    multiply_UL_inplace(
            cutoff,
            auxmatrixl,
            aux_dim0,
            auxmatrixr,
            aux_dim0
            );

    /* Add result to top left of output array. */
    const complex_type *read = auxmatrixl;
    const complex_type * const end = auxmatrixl + cutoff * cutoff;
    complex_type *write = left;
    int i;
    while (read < end)
    {
        i = -1;
        while (++i < cutoff)
            write[i] += read[i];
        write += left_dim0;
        read += aux_dim0;
    }

    /* Add conjugate of result to bottom right of output array. */
    read = auxmatrixl;
    write = left + left_dim0*(ncolr - 1) + nrowl - cutoff - 1;
    while (read < end)
    {
        i = -1;
        while (++i < cutoff)
            write[cutoff-i] += symmetry * conj(read[i]);
        write -= left_dim0;
        read += aux_dim0;
    }
}

/**
 * multiply 2 matrices with the following requirements, which are not checked:
 * 1. left[::-1, ::-1] == s1 * left.conjugate() and right[::-1, ::-1] == s2 * right.conjugate()
 *    with symmetry = s1*s2, s1 and s2 must be +1 or -1.
 * 2. left.shape == (n, k), right.shape == (k, m) where
 *    n,k,m \in {N,N+1} for some integer N >= cutoff + 2
 *
 * left will be overwritten. right also needs to be writable.
 * Matrices are extended as defined by padding.
 */
static PyArrayObject* multiply_extended_2d_symmetric(
        PyArrayObject *left,
        PyArrayObject *right,
        const int cutoff,
        const int symmetry,
        const int clear_corners,
        const char flags
        )
{
#ifdef DEBUG
    fprintf(stderr, "Entering multiply_extended_2d_symmetric\n");
#endif
    const int
        nrowl = PyArray_DIM(left, 0),
        ncoll = PyArray_DIM(left, 1),
        ncolr = PyArray_DIM(right, 1);

    PyArrayObject *fright = (PyArrayObject*) PyArray_FromArray(
                right,
                PyArray_DescrFromType(NPY_COMPLEX_TYPE),
                NPY_ARRAY_WRITEABLE
                    | NPY_ARRAY_F_CONTIGUOUS
                    | NPY_ARRAY_ALIGNED
            );
    if (!fright)
        return NULL;
    PyArrayObject *fleft = (PyArrayObject*) PyArray_FromArray(
                left,
                PyArray_DescrFromType(NPY_COMPLEX_TYPE),
                NPY_ARRAY_WRITEABLE
                    | NPY_ARRAY_F_CONTIGUOUS
                    | NPY_ARRAY_ALIGNED
                    | ((flags & OVERWRITE_LEFT) ? 0 : NPY_ARRAY_ENSURECOPY)
            );
    if (!fleft)
    {
        Py_DECREF(fright);
        return NULL;
    }

    if (ncolr > ncoll)
    {
        npy_intp shape[] = {nrowl, ncolr};
        PyArray_Dims dims = {shape, 2};
        if (!PyArray_Resize(fleft, &dims, 1, NPY_FORTRANORDER))
        {
            Py_DECREF(fright);
            Py_DECREF(fleft);
            PyErr_SetString(
                    PyExc_RuntimeError,
                    "Failed to resize (enlargen) matrix.");
            return NULL;
        }
    }

    complex_type *const auxmatrixl = malloc( cutoff*cutoff * sizeof(complex_type) );
    complex_type *const auxmatrixr = malloc( cutoff*cutoff * sizeof(complex_type) );

    if (!auxmatrixl || !auxmatrixr)
    {
        free(auxmatrixl);
        free(auxmatrixr);
        Py_DECREF(fleft);
        Py_DECREF(fright);
        return NULL;
    }

    multiply_symmetric_worker(
            nrowl,
            ncoll,
            ncolr,
            symmetry,
            cutoff,
            PyArray_DATA(fleft),
            PyArray_STRIDE(fleft, 1) / sizeof(complex_type),
            PyArray_DATA(fright),
            PyArray_STRIDE(fright, 1) / sizeof(complex_type),
            auxmatrixl,
            auxmatrixr,
            cutoff,
            clear_corners
            );

    /* Free auxilliary arrays. */
    free(auxmatrixl);
    free(auxmatrixr);
    Py_DECREF(fright);

    if (ncolr < ncoll)
    {
        npy_intp shape[] = {nrowl, ncolr};
        PyArray_Dims dims = {shape, 2};
        if (!PyArray_Resize(fleft, &dims, 1, NPY_FORTRANORDER))
        {
            Py_DECREF(fleft);
            PyErr_SetString(
                    PyExc_RuntimeError,
                    "Failed to resize (truncate) matrix.");
            return NULL;
        }
    }

    return fleft;
}

/**
 * CAUTION: THIS DOES NOT DO WHAT IS REQUIRED FOR FRTRG!
 */
static PyArrayObject* multiply_extended_nd_symmetric(
        PyArrayObject *left,
        PyArrayObject *fright,
        const int cutoff,
        const int symmetry,
        const int clear_corners,
        const char flags
        )
{
    if (PyErr_WarnEx(NULL, "This function should probably not be used!", 1))
        return NULL;
#ifdef DEBUG
    fprintf(stderr, "Entering multiply_extended_nd_symmetric\n");
#endif
    const int
        nrowl = PyArray_DIM(left, 0),
        ncoll = PyArray_DIM(left, 1),
        ncolr = PyArray_DIM(fright, 1),
        ndim = PyArray_NDIM(left);

    int i=1, nmatrices=1;
    while (++i<ndim)
        nmatrices *= PyArray_DIM(left, i);

    void *left_data;
    PyArrayObject *fleft;
    int left_dim0, left_matrixstride;
    if (ncolr == ncoll)
    {
        fleft = (PyArrayObject*) PyArray_FromArray(
                    left,
                    PyArray_DescrFromType(NPY_COMPLEX_TYPE),
                    NPY_ARRAY_WRITEABLE
                        | NPY_ARRAY_F_CONTIGUOUS
                        | NPY_ARRAY_ALIGNED
                        | ((flags & OVERWRITE_LEFT) ? 0 : NPY_ARRAY_ENSURECOPY)
                );
        if (!fleft)
            return NULL;
        left_data = PyArray_DATA(fleft);
        left_dim0 = PyArray_STRIDE(fleft, 1) / sizeof(complex_type);
        left_matrixstride = PyArray_STRIDE(fleft, 2);
    }
    else
    {
        fleft = (PyArrayObject*) PyArray_FromArray(
                    left,
                    PyArray_DescrFromType(NPY_COMPLEX_TYPE),
                    NPY_ARRAY_WRITEABLE
                        | NPY_ARRAY_F_CONTIGUOUS
                        | NPY_ARRAY_ALIGNED
                );
        if (!fleft)
            return NULL;
        left_dim0 = nrowl;
        left_data = malloc(
                (ncoll > ncolr ? (nmatrices - 1) * ncolr + ncoll : nmatrices * ncolr)
                * left_dim0 * sizeof(complex_type) );
        if (!left_data)
        {
            Py_DECREF(fleft);
            return NULL;
        }
        left_matrixstride = nrowl * ncolr * sizeof(complex_type);
    }

    const int
        right_dim0 = PyArray_STRIDE(fright, 1) / sizeof(complex_type),
        right_matrixstride = PyArray_STRIDE(fright, 2);

    complex_type *const auxmatrixl = malloc( cutoff*cutoff* sizeof(complex_type) );
    complex_type *const auxmatrixr = malloc( cutoff*cutoff* sizeof(complex_type) );

    if (!auxmatrixl || !auxmatrixr)
    {
        if (ncolr != ncoll)
            free(left_data);
        free(auxmatrixl);
        free(auxmatrixr);
        Py_DECREF(fleft);
        return NULL;
    }

    for (i=0; i<nmatrices; ++i)
    {
        if (ncolr != ncoll)
            /* copy data */
            memcpy(
                    left_data + i*left_matrixstride,
                    PyArray_DATA(fleft) + i*PyArray_STRIDE(fleft, 2),
                    PyArray_STRIDE(fleft, 2)
                    );
        multiply_symmetric_worker(
                nrowl,
                ncoll,
                ncolr,
                symmetry,
                cutoff,
                left_data + i*left_matrixstride,
                left_dim0,
                PyArray_DATA(fright) + i*right_matrixstride,
                right_dim0,
                auxmatrixl,
                auxmatrixr,
                cutoff,
                clear_corners
                );
    }

    /* Free auxilliary arrays. */
    free(auxmatrixl);
    free(auxmatrixr);

    if (ncoll != ncolr)
    {
        Py_DECREF(fleft);
        npy_intp *shape = malloc( ndim*sizeof(npy_intp) );
        memcpy( shape, PyArray_DIMS(left), ndim*sizeof(npy_intp) );
        shape[1] = ncolr;
        fleft = (PyArrayObject*) PyArray_New(
                &PyArray_Type,
                ndim,
                shape,
                NPY_COMPLEX_TYPE, // data type
                NULL, // strides
                left_data, // data
                sizeof(complex_type), // item size
                NPY_ARRAY_F_CONTIGUOUS, // flags
                NULL // obj
                );
    }

    return fleft;
}



/**
 * Matrix multiplication function callable from python.
 */
static PyObject* multiply_extended(PyObject *self, PyObject *args)
{
    /* Define the arrays. */
    PyArrayObject *left, *right;
    int cutoff, symmetry = 0, clear_corners = 0;
    char flags = 0;

    /* Parse the arguments. symmetry and flags are optional. */
    if (!PyArg_ParseTuple(
                args,
                "O!O!i|iic",
                &PyArray_Type,
                &left,
                &PyArray_Type,
                &right,
                &cutoff,
                &symmetry,
                &clear_corners,
                &flags))
        return NULL;

    if (    PyArray_TYPE(left) != NPY_COMPLEX_TYPE
            || PyArray_TYPE(right) != NPY_COMPLEX_TYPE)
        return PyErr_Format(PyExc_ValueError, "arrays of type complex128 required");
    if (    PyArray_NDIM(left) != PyArray_NDIM(right) ||
            PyArray_NDIM(left) < 2 ||
            PyArray_DIM(left, 1) != PyArray_DIM(right, 0))
        return PyErr_Format(
                PyExc_ValueError,
                "Shape of the 2 matrices does not allow multiplication.");

    const int min_size = cutoff + (clear_corners > 4 ? clear_corners-1 : 3);
    if (cutoff > 0 && (
                PyArray_DIM(left, 0) < min_size ||
                PyArray_DIM(left, 1) < min_size ||
                PyArray_DIM(right, 1) < min_size))
        return PyErr_Format(
                PyExc_ValueError,
                "Matrices is too small (%d, %d, %d) or cutoff too large (%d).",
                PyArray_DIM(left, 0),
                PyArray_DIM(left, 1),
                PyArray_DIM(right, 1),
                cutoff);


    if (PyArray_NDIM(left) > 2 &&
            memcmp(
                PyArray_DIMS(left) + 2,
                PyArray_DIMS(right) + 2,
                (PyArray_NDIM(left)-2)*sizeof(npy_intp))
            )
        return PyErr_Format(PyExc_ValueError, "dimensions of matrices do not match");

    /* Get an F-contiguous version of right. */
    PyArrayObject *fright = (PyArrayObject*) PyArray_FromArray(
                right,
                PyArray_DescrFromType(NPY_COMPLEX_TYPE),
                NPY_ARRAY_WRITEABLE
                    | NPY_ARRAY_F_CONTIGUOUS
                    | NPY_ARRAY_ALIGNED
            );
    if (!fright)
        return PyErr_Format(PyExc_RuntimeError, "Failed to create array");

    if (symmetry && (
                   abs(((int) PyArray_DIM(left,  0)) - ((int) PyArray_DIM(left,  1))) > 1
                || abs(((int) PyArray_DIM(left,  0)) - ((int) PyArray_DIM(right, 1))) > 1
                || abs(((int) PyArray_DIM(right, 0)) - ((int) PyArray_DIM(right, 1))) > 1 ))
            symmetry = 0;

#ifdef ANALYZE
    int mm_flags = 0;
    if (PyArray_FLAGS(left) & NPY_ARRAY_F_CONTIGUOUS)
        mm_flags |= LEFT_F_CONTIGUOUS;
    else if (PyArray_FLAGS(left) & NPY_ARRAY_C_CONTIGUOUS)
        mm_flags |= LEFT_C_CONTIGUOUS;
    if (PyArray_FLAGS(right) & NPY_ARRAY_F_CONTIGUOUS)
        mm_flags |= RIGHT_F_CONTIGUOUS;
    else if (PyArray_FLAGS(right) & NPY_ARRAY_C_CONTIGUOUS)
        mm_flags |= RIGHT_C_CONTIGUOUS;
    if (symmetry)
        mm_flags |= SYMMETRIC;
    if (PyArray_NDIM(left) == 2)
        mm_flags |= TWO_DIM;
    TOTAL_MATRIX_MULTIPLICATIONS++;
    MATRIX_MULTIPLICATIONS[mm_flags]++;
#endif

    PyArrayObject *out;
    if (symmetry)
    {
        /* In this case, functions decide dyamically whether fleft needs to be copied.
         * Something like fleft will be created in these functions. */
        if (PyArray_NDIM(left) == 2)
            out = multiply_extended_2d_symmetric(
                    left,
                    fright,
                    cutoff,
                    symmetry,
                    clear_corners,
                    flags
                    );
        else
            out = multiply_extended_nd_symmetric(
                    left,
                    fright,
                    cutoff,
                    symmetry,
                    clear_corners,
                    flags
                    );
    }
    else
    {
        /* general matrix matrix multiplication does not overwrite the left array, no copy required.
         * Create an F-contiguous version of left. */
        PyArrayObject *fleft = (PyArrayObject*) PyArray_FromArray(
                    left,
                    PyArray_DescrFromType(NPY_COMPLEX_TYPE),
                    NPY_ARRAY_WRITEABLE
                        | NPY_ARRAY_F_CONTIGUOUS
                        | NPY_ARRAY_ALIGNED
                );
        if (!fleft)
        {
            Py_DECREF(fright);
            return PyErr_Format(PyExc_RuntimeError, "Failed to create array");
        }

        if (PyArray_NDIM(fleft) == 2)
            out = multiply_extended_2d(fleft, fright, cutoff, clear_corners);
#ifdef PARALLEL_EXTRA_DIMS
        else if (omp_get_max_threads() > 1)
            out = multiply_extended_nd_parallel(fleft, fright, cutoff, clear_corners);
#endif /* PARALLEL_EXTRA_DIMS */
        else
            out = multiply_extended_nd(fleft, fright, cutoff, clear_corners);

        Py_DECREF(fleft);
    }

    Py_DECREF(fright);
    return (PyObject*) out;
}

#ifdef ANALYZE
static PyObject* reset_statistics(PyObject *self, PyObject *args)
{
    TOTAL_MATRIX_MULTIPLICATIONS = 0;
    memset(&MATRIX_MULTIPLICATIONS, 0, MATRIX_MULTIPLICATIONS_SIZE*sizeof(int));
    Py_RETURN_NONE;
}
static PyObject* get_statistics(PyObject *self, PyObject *args)
{
    int flags = -1;
    if (!PyArg_ParseTuple(args, "|i", &flags))
        return NULL;
    if (flags >= 0 && flags < MATRIX_MULTIPLICATIONS_SIZE)
        return PyLong_FromLong(MATRIX_MULTIPLICATIONS[flags]);
    return PyLong_FromLong(TOTAL_MATRIX_MULTIPLICATIONS);
}
#endif


/**
 * @file
 * @section sec_module_setup Python module setup
 */


/** define functions in module */
static PyMethodDef FloquetMethods[] =
{
     {
         "extend_matrix",
         extend_matrix,
         METH_VARARGS,
         PyDoc_STR(
                 "Extrapolate matrix along diagonals.\n"
                 "Arguments:\n"
                 "  1. numpy array of type complex128 and shape (n,m,...)\n"
                 "  2. cutoff, positive integer fulfilling min(n,m)+2 > cutoff\n"
                 "Returns:\n"
                 "  numpy array of shape (n+2*cutoff, m+2*cutoff, ...)"
             )
     },
     {
         "multiply_extended",
         multiply_extended,
         METH_VARARGS,
         PyDoc_STR(
                 "Multiply virtually extended matrices.\n"
                 "Arguments:\n"
                 "  1. a, numpy array of type complex128 and shape (n,k,...)\n"
                 "  2. b, numpy array of type complex128 and shape (k,m,...)\n"
                 "  3. cutoff, positive integer fulfilling min(n,k,m)+2 > cutoff\n"
                 "  4. (optional) symmetry, integer, allowed values: {-1,0,+1}\n"
                 "     Speed up calculation by symmetry = s1*s2 if\n"
                 "         a[::-1,::-1].conjugate() == s1*a  and\n"
                 "         b[::-1,::-1].conjugate() == s2*b\n"
                 "  5. (optional) clear_corners: integer < 2*nmax+1 - cutoff\n"
                 "  6. (optional) flags, char, set to 1 to allow overwriting left matrix\n"
                 "Returns:\n"
                 "  numpy array of shape (n,m,...), product of a and b.\n"
                 "Note: the extra dimensions denoted ... must be the same for all matrices."
             )
     },
     {
         "invert_extended",
         invert_extended,
         METH_VARARGS,
         PyDoc_STR(
                 "Extrapolate matrix, invert it and reduce it to original size.\n"
                 "Arguments:\n"
                 "  1. numpy array of type complex128 and shape (n,n,...)\n"
                 "  2. cutoff, positive integer fulfilling n+2 > cutoff\n"
                 "  3. (optional) lazy_factor, int, 0 <= lazy_factor < cutoff:\n"
                 "     reduce matrix size by this amount in each direction after\n"
                 "     extrapolation but before inversion.\n"
                 "Returns:\n"
                 "  inverse, numpy array of shape (n,n,...)"
             )
     },
#ifdef ANALYZE
     {
         "reset_statistics",
         reset_statistics,
         METH_VARARGS,
         PyDoc_STR(
                 "Reset statistics."
             )
     },
     {
         "get_statistics",
         get_statistics,
         METH_VARARGS,
         PyDoc_STR(
                 "Get some statistics.\n"
                 "Argument: flags (int)\n"
                 "If no argument or an invalid argument is given,\n"
                 "the total number of matrix products is returned.\n"
                 "These flags can be combined:\n"
                 "    left array F contiguous:  " TOSTRING(LEFT_F_CONTIGUOUS) "\n"
                 "    right array F contiguous: " TOSTRING(RIGHT_F_CONTIGUOUS) "\n"
                 "    left array C contiguous:  " TOSTRING(LEFT_C_CONTIGUOUS) "\n"
                 "    right array C contiguous: " TOSTRING(RIGHT_C_CONTIGUOUS) "\n"
                 "    use symmetry in multiply: " TOSTRING(SYMMETRIC) "\n"
                 "    arrays are 2 dimensinoal: " TOSTRING(TWO_DIM) "\n"
             )
     },
#endif
     {NULL, NULL, 0, NULL}
};


/** module initialization (python 3) */
static struct PyModuleDef cModPyDem = {
    PyModuleDef_HEAD_INIT,
    "rtrg_c",
    PyDoc_STR(
            "Auxilliary functions for calculations with Floquet matrices.\n"
            "Floquet matrices are represented by square matrices, which should\n"
            "be extrapolated along the diagonal to avoid truncation effects.\n"
            "Non-square matrices may also be used to account for reduced matrices\n"
            "in special symmetric cases.\n\n"
            "NOTE:\n"
            "    Use\n"
            "        rtrg_c.multiply_extended(b.T, a.T, cutoff, ...).T\n"
            "        rtrg_c.invert_extended(a.T, cutoff, ...).T\n"
            "        rtrg_c.extend_matrix(a.T, cutoff).T\n"
            "    where the last 2 dimensions of a, b define Floquet matrices\n"
            "    instead of\n"
            "        rtrg_c.multiply_extended(a, b, cutoff, ...)\n"
            "        rtrg_c.invert_extended(a, cutoff, ...)\n"
            "        rtrg_c.extend_matrix(a, cutoff)\n"
            "    for better performance. This will pass F-contiguous arrays to\n"
            "    rtrg_c if original arrays were C-contiguous (standard in numpy).\n"
        ),
    -1,
    FloquetMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_rtrg_c(void)
{
    PyObject *module;
    module = PyModule_Create(&cModPyDem);
    if(module == NULL)
        return NULL;
    /* IMPORTANT: this must be called */
    import_array();
    if (PyErr_Occurred())
        return NULL;
    return module;
}

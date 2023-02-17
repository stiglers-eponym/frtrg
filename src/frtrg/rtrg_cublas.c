/*
MIT License

Copyright (c) 2021 Valentin Bruch

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



#include "rtrg_cublas.h"
/* Define data type and select CBLAS and LAPACK functions accordingly */
#include <complex.h>

#ifdef LAPACK_C
#include <lapack.h>
//#include <mkl_lapack.h>
#else /* LAPACK_C */
extern void zgetrf_(const int*, const int*, double complex*, const int*, int*, int*);
extern void zgetri_(const int*, double complex*, const int*, const int*, complex double*, const int*, int*);
#endif /* LAPACK_C */

#ifdef CBLAS
#include <cblas.h>
//#include <mkl_cblas.h>
#else /* CBLAS */
extern void zgemm_(const char*, const char*, const int*, const int*, const int*, const complex double*, const complex double*, const int*, const complex double*, const int*, const double complex*, double complex*, const int*);
extern void ztrmm_(const char*, const char*, const char*, const char*, const int*, const int*, const complex double*, const complex double*, const int*, complex double*, const int*);
static const char N='N', L='L', R='R', U='U';
#endif /* CBLAS */


extern char cuda_gemm(
        int nrowl,
        int ncolr,
        int ncoll,
        const void *prefactor,
        const void *left,
        int left_dim0,
        const void *right,
        int right_dim0,
        const void *weight_c,
        void *output,
        int out_dim0
        );

extern char cuda_trmm(
        const int flags,
        const int nrowb,
        const int ncolb,
        const void *prefactor,
        const void *a,
        const int a_dim0,
        void *b,
        const int b_dim0
        );

extern int multiply_UL_cuda(
        const int size,
        void *a,
        const int a_dim0,
        const void *b,
        const int b_dim0
        );

extern int multiply_LU_cuda(
        const int size,
        void *a,
        const int a_dim0,
        const void *b,
        const int b_dim0
        );


static const complex_type zero = 0.;
static const complex_type one = 1.;


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

/* Symmetric matrix multiplication can be faster if it is allowed to overwrite
 * the left matrix. This enum defines flags for this option. */
enum {
    OVERWRITE_LEFT = 1 << 0,
    OVERWRITE_RIGHT = 1 << 1,
};


/************************************************
 *     EXTRAPOLATE MATRIX FOR MULTIPLICATION    *
 ************************************************/

/* For example, for cutoff = 2 and a matrix
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
 * TODO: This could be parallelized.
 */

/**
 * Extrapolate matrix to the top.
 * input and output must be in columns major order (Fortran style).
 * output is treated as a square matrix, it must have at least cutoff
 * columns.
 *
 * The following requirements are not explicitly checked:
 * out_rows >= cutoff
 * rows_in >= 3
 * cols_in >= cutoff + 3
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
                output[cutoff-chunk+(chunk-i)*(out_rows+1)] = extrapolate(i, row0, row1, row2);
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
 * rows_in >= 3
 * cols_in >= cutoff + 3
 * out_rows >= cutoff
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
                output[chunk - cutoff + (cutoff - 1 - chunk + i)*(out_rows+1)] = extrapolate(i, row0, row1, row2);
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
 * rows_in >= cutoff + 3
 * out_rows >= cutoff
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
                output[out_rows * jmax + j] = extrapolate(i, input[i+j], input[i+j+rows_in+1], input[i+j+2*(rows_in+1)]);
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
                output[(out_rows+1) * (i-1) + j] = extrapolate(i, input[j], input[j-rows_in-1], input[j-2*rows_in-2]);
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


/************************************************
 *     EXTRAPOLATE MATRIX FOR INVERSION         *
 ************************************************/

/* For example, for cutoff = 2 and a matrix
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
 * In this representation, the pointers handed to the following functions have
 * the meaning:
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
 * out_rows >= cutoff
 * rows_in >= 3
 * cols_in >= cutoff + 3
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
 * rows_in >= 3
 * cols_in >= cutoff + 3
 * out_rows >= cutoff
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
 * rows_in >= cutoff + 3
 * out_rows >= cutoff
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
 * rows_in >= cutoff + 3
 * out_rows >= cutoff
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



/************************************************
 *     HELPER FUNCTIONS FOR MULTIPLICATION      *
 ************************************************/

/**
 * A is an upper triangular matrix.
 * B is a lower triangular matrix.
 * Both are in Fortran order (columns major) not unit triangular.
 * The product AB overwrites A.
 * Both matrices must have shape (size, size).
 */
static int multiply_UL_inplace(
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
        /* This is not done on the GPU since it is assumed that
         * TRIANGULAR_OPTIMIZE_THRESHOLD < THRESHOLD_TRMM_GPU. */
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
        return 0;
    }
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
    int i = -1;
    while (++i<part1)
        memcpy( a + i*a_dim0, b + i*b_dim0, part2 * sizeof(complex_type) );
    a -= part1;
    b -= part1;

    i = (part1 < THRESHOLD_TRMM_GPU) ? 1 :
            cuda_trmm(
                    LeftMultiplication | UpperTriangular,
                    part2, // rows of B (int)
                    part1, // columns of B (int)
                    &one, // prefactor
                    a + part1*(1 + a_dim0), // matrix A
                    a_dim0, // first dimension of A (int)
                    a + part1, // matrix B
                    a_dim0  // first dimension of B (int)
                    );
    if (i == 1)
    {
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
    }
    else if (i == 2)
        return 2;

    /* Step 2: overwrite Ad with Ad Bd */
    if (multiply_UL_inplace(part2, a + (a_dim0+1)*part1, a_dim0, b + (b_dim0+1)*part1, b_dim0))
        return 1;
    /* Step 3: overwrite Aa with Aa Ba */
    if (multiply_UL_inplace(part1, a, a_dim0, b, b_dim0))
        return 1;

    /* Step 4: add Ab Bc to Aa */
    if (part1 < THRESHOLD_GEMM_GPU
            || cuda_gemm(
                part1,
                part1,
                part2,
                &one, // global prefactor
                a + part1*a_dim0,
                a_dim0,
                b + part1,
                b_dim0,
                &one, // weight of C
                a,
                a_dim0
                ))
    {
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
    }

    /* Step 5: overwrite Ab with Ab Bd */
    i = (part1 < THRESHOLD_TRMM_GPU) ? 1 :
            cuda_trmm(
                    0,
                    part1, // rows of B (int)
                    part2, // columns of B (int)
                    &one, // prefactor
                    b + (1 + b_dim0)*part1, // matrix A
                    b_dim0, // first dimension of A (int)
                    a + part1*a_dim0, // matrix B
                    a_dim0  // first dimension of B (int)
                    );
    if (i == 1)
    {
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
        return 0;
    }
    return i;
}

/**
 * A is a lower triangular matrix.
 * B is an upper triangular matrix.
 * Both are in Fortran order (columns major) not unit triangular.
 * The product AB overwrites A.
 * Both matrices must have shape (size, size).
 */
static int multiply_LU_inplace(
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
        /* This is not done on the GPU since it is assumed that
         * TRIANGULAR_OPTIMIZE_THRESHOLD < THRESHOLD_TRMM_GPU. */
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
        return 0;
    }
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
    int i = -1;
    while (++i<part2)
        memcpy( a + i*a_dim0, b + i*b_dim0, part1 * sizeof(complex_type) );
    a -= a_dim0*part1;
    b -= b_dim0*part1;

    i = (part1 < THRESHOLD_TRMM_GPU) ? 1 :
            cuda_trmm(
                    LeftMultiplication,
                    part1, // rows of B (int)
                    part2, // columns of B (int)
                    &one, // prefactor
                    a, // matrix A
                    a_dim0, // first dimension of A (int)
                    a + a_dim0*part1, // matrix B
                    a_dim0  // first dimension of B (int)
                    );
    if (i == 1)
    {
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
    }
    else if (i == 2)
        return 2;

    /* Step 2: overwrite Aa with Aa Ba */
    if (multiply_LU_inplace(part1, a, a_dim0, b, b_dim0))
        return 1;
    /* Step 3: overwrite Ad with Ad Bd */
    if (multiply_LU_inplace(part2, a + (a_dim0+1)*part1, a_dim0, b + (b_dim0+1)*part1, b_dim0))
        return 1;

    /* Step 4: add Ac Bb to Ad */
    if (part1 < THRESHOLD_GEMM_GPU
            || cuda_gemm(
                part2,
                part2,
                part1,
                &one, // global prefactor
                a + part1,
                a_dim0,
                b + part1*b_dim0,
                b_dim0,
                &one, // weight of C
                a + (a_dim0+1)*part1,
                a_dim0
                ))
    {
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
    }

    /* Step 5: overwrite Ac with Ac Ba */
    i = (part1 < THRESHOLD_TRMM_GPU) ? 1 :
            cuda_trmm(
                    UpperTriangular,
                    part2, // rows of B (int)
                    part1, // columns of B (int)
                    &one, // prefactor
                    b, // matrix A
                    b_dim0, // first dimension of A (int)
                    a + part1, // matrix B
                    a_dim0  // first dimension of B (int)
                    );
    if (i == 1)
    {
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
        return 0;
    }
    return i;
}


/************************************************
 *               EXTEND MATRIX                  *
 ************************************************/

/**
 * input : in_rows × in_cols
 * output : (in_rows + 2*cutoff) × (in_cols + 2*cutoff)
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
    extrapolate_top_full(
            auxptr,
            nrow_in+2*cutoff,
            output,
            cutoff,
            nrow_in+2*cutoff
            );
    extrapolate_left_full(
            auxptr,
            nrow_in+2*cutoff,
            output,
            cutoff,
            nrow_in+2*cutoff
            );

    /* outptr points to the last element of the original matrix in the extended matrix. */
    auxptr += (ncol_in - 1) * out_dim0 + nrow_in - 1;
    extrapolate_bottom_full(
            auxptr,
            nrow_in+2*cutoff,
            output + ncol_in * out_dim0 + nrow_in + cutoff,
            cutoff,
            nrow_in+2*cutoff
            );
    extrapolate_right_full(
            auxptr,
            nrow_in+2*cutoff,
            output + (ncol_in + cutoff) * out_dim0 + nrow_in,
            cutoff,
            nrow_in+2*cutoff
            );
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

/**
 * Given a Fortran-contiguous square matrix, extend it by linear extrapolation
 * in each direction by <cutoff> rows/columns.
 */
#ifdef PARALLEL_EXTRA_DIMS
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
 * Take an n×n Floquet matrix M and positive integer c as arguments. Extrapolate
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
        return PyErr_Format(PyExc_ValueError, "array of type complex128 required");
    if (PyArray_NDIM(input) < 2)
        return PyErr_Format(PyExc_ValueError, "1st argument must have at least 2 dimensions.");

    if (cutoff <= 0)
    {
        Py_INCREF(input);
        return (PyObject*) input;
    }

    if ((PyArray_DIM(input, 0) < cutoff + 3) || (PyArray_DIM(input, 1) < cutoff + 3))
        return PyErr_Format(PyExc_ValueError, "Matrix is too small or cutoff too large.");

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
        npy_intp shape[] = {PyArray_DIM(finput, 0) + 2*cutoff, PyArray_DIM(finput, 1) + 2*cutoff};
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


/************************************************
 *               INVERT MATRIX                  *
 ************************************************/

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
                memcpy( PyArray_GETPTR2(output, 0, i), extended + cutoff + (i+cutoff)*extended_stride, size*sizeof(complex_type) );
    }

    free( extended );
    return output;
}

/**
 * input must have shape (n, n, ...) with n > cutoff+2 and cutoff >= reduce_cutoff.
 * This is not cheked!
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
            memcpy( outptr + j*out_colstride, extended + cutoff + (j+cutoff)*extended_stride, size*sizeof(complex_type) );
    }

    free( extended );
    return output;
}

/**
 * input must have shape (n, n, ...) with n > cutoff+2 and cutoff >= reduce_cutoff.
 * This is not cheked!
 */
#ifdef PARALLEL_EXTRA_DIMS
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
            memcpy( outptr + j*out_colstride, extended + cutoff + (j+cutoff)*extended_stride, size*sizeof(complex_type) );
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


/************************************************
 *             MULTIPLY MATRICES                *
 ************************************************/

/**
 * auxarr_left must be initialized to 0.
 * auxarr_left and auxarr_right must have size cutoff*aux_dim0 with aux_dim0 >= cutoff
 */
static int multiply_extended_worker(
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
        complex_type *auxarr_left,
        complex_type *auxarr_right,
        const int aux_dim0,
        const int clear_corners
        )
{
    if (nrowl < THRESHOLD_GEMM_GPU || ncoll < THRESHOLD_GEMM_GPU || ncolr < THRESHOLD_GEMM_GPU
            || cuda_gemm(
                nrowl,
                ncolr,
                ncoll,
                &one,
                left,
                left_dim0,
                right,
                right_dim0,
                &zero,
                output,
                out_dim0
            ) // If this fails, the CPU will take over the job.
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
    }

    if (clear_corners > 0)
    {
#ifdef PARALLEL_EXTRAPOLATION
#pragma omp for
#endif
        for (int i=0; i<clear_corners; i++)
            memset( output + (ncolr-clear_corners+i)*out_dim0, 0, (i+1)*sizeof(complex_type) );
#ifdef PARALLEL_EXTRAPOLATION
#pragma omp for
#endif
        for (int i=0; i<clear_corners; i++)
            memset( output + i*(out_dim0+1) + nrowl - clear_corners, 0, (clear_corners-i)*sizeof(complex_type) );
    }

    if (cutoff <= 0)
        return 0;

    extrapolate_left(left, left_dim0, auxarr_left, cutoff, aux_dim0);
    extrapolate_top(right, right_dim0, auxarr_right, cutoff, aux_dim0);

    /* Calculate matrix product. This overwrites auxarr_left. */
#ifdef DEBUG
    fprintf(stderr, "Calling multiply_UL_inplace for extrapolated matrices\n");
#endif
    if (cutoff < LU_ON_GPU_THRESHOLD
            ? multiply_UL_inplace(
                cutoff, // size
                auxarr_left, // matrix A
                aux_dim0,  // first dimension of A (int)
                auxarr_right, // matrix B
                aux_dim0 // first dimension of B (int)
                )
            : multiply_UL_cuda(
                cutoff, // size
                auxarr_left, // matrix A
                aux_dim0,  // first dimension of A (int)
                auxarr_right, // matrix B
                aux_dim0 // first dimension of B (int)
                )
            )
        return 1;

    /* Add result to top left of output array. */
#ifdef DEBUG
    fprintf(stderr, "Writing result to top left\n");
#endif
    const complex_type *read = auxarr_left;
    const complex_type *const end = auxarr_left + cutoff * aux_dim0;
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

    memset( auxarr_left, 0, cutoff*aux_dim0*sizeof(complex_type) );

    extrapolate_right(left + left_dim0*(ncoll-1) + nrowl-1, left_dim0, auxarr_left, cutoff, aux_dim0);
    extrapolate_bottom(right + right_dim0*(ncolr-1) + ncoll-1, right_dim0, auxarr_right, cutoff, aux_dim0);

    /* Calculate matrix product. This overwrites auxarr_left. */
#ifdef DEBUG
    fprintf(stderr, "Calling multiply_LU_inplace for extrapolated matrices\n");
#endif
    if (cutoff < LU_ON_GPU_THRESHOLD
            ? multiply_LU_inplace(
                cutoff,
                auxarr_left,
                aux_dim0,
                auxarr_right,
                aux_dim0
                )
            : multiply_LU_cuda(
                cutoff,
                auxarr_left,
                aux_dim0,
                auxarr_right,
                aux_dim0
                )
            )
        return 1;

    /* Add result to bottom right of output array. */
#ifdef DEBUG
    fprintf(stderr, "Writing result to bottom right\n");
#endif
    read = auxarr_left;
    write = output + (ncolr - cutoff)*out_dim0 + nrowl - cutoff;
    while (read < end)
    {
        i = -1;
        while (++i < cutoff)
            write[i] += read[i];
        write += out_dim0;
        read += aux_dim0;
    }
    return 0;
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
        *auxarr_left  = calloc( cutoff*cutoff, sizeof(complex_type) ),
        *auxarr_right = malloc( cutoff*cutoff* sizeof(complex_type) );

    if (!auxarr_left || !auxarr_right ||
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
                auxarr_left,
                auxarr_right,
                cutoff,
                clear_corners
                ))
    {
        Py_DECREF(out);
        out = NULL;
    }

    free( auxarr_left );
    free( auxarr_right );
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
        *auxarr_left  = malloc( cutoff*cutoff * sizeof(complex_type) ),
        *auxarr_right = malloc( cutoff*cutoff * sizeof(complex_type) );

    int status = 0;
    if (auxarr_left && auxarr_right)
    {
        for (i=0; i<nmatrices; ++i)
        {
            memset( auxarr_left, 0, cutoff*cutoff*sizeof(complex_type) );
            status = multiply_extended_worker(
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
                    cutoff,
                    clear_corners
                    );
            if (status)
            {
                Py_DECREF(out);
                out = NULL;
                break;
            }
        }
    }
    else
    {
        Py_DECREF(out);
        out = NULL;
    }

    free( auxarr_left );
    free( auxarr_right );
    return out;
}

/**
 * left and right must be Fortran contiguous.
 */
#ifdef PARALLEL_EXTRA_DIMS
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

        if (!auxarr_left || !auxarr_right ||
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
                        cutoff,
                        clear_corners
                        ))
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
static int multiply_symmetric_nonsquare(
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
    const int status = (nrowl < THRESHOLD_TRMM_GPU || ncoll < THRESHOLD_TRMM_GPU || ncolr < THRESHOLD_TRMM_GPU) ? 1 :
            cuda_trmm(
                    ncoll > ncolr ? UpperTriangular : 0,
                    nrowl, // rows of B (int)
                    ncoll > ncolr ? ncolr : ncoll, // columns of B (int) = rows of A
                    &one, // prefactor
                    right, // matrix A
                    right_dim0, // first dimension of A (int)
                    left, // matrix B
                    left_dim0  // first dimension of B (int)
                    );
    if (status == 1)
    {
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
    }
    else if (status == 2)
        return 2;

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
            memset( left + (ncolr-clear_corners+i)*left_dim0, 0, (i+1)*sizeof(complex_type) );
#ifdef PARALLEL_EXTRAPOLATION
#pragma omp for
#endif
        for (int i=0; i<clear_corners; i++)
            memset( left + i*(left_dim0+1) + nrowl - clear_corners, 0, (clear_corners-i)*sizeof(complex_type) );
    }
    return 0;
}

/**
 * left and right must be Fortran contiguous. right must be a square matrix.
 * left is overwritten with the product of left and right.
 * Diagonal elements of right are temporarily overwritten but restored afterwards.
 */
static int multiply_symmetric_square(
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
    i = (nrowl < THRESHOLD_TRMM_GPU || ncolr < THRESHOLD_TRMM_GPU) ? 1 :
            cuda_trmm(
                    0,
                    nrowl, // rows of B (int)
                    ncolr, // columns of B (int) = rows of A = columns of A
                    &one, // prefactor
                    right, // matrix A
                    right_dim0, // first dimension of A (int)
                    left, // matrix B
                    left_dim0  // first dimension of B (int)
                    );
    if (i == 1)
    {
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
    }
    else if (i == 2)
        return 2;

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
            memset( left + (ncolr-clear_corners+i)*left_dim0, 0, (i+1)*sizeof(complex_type) );
#ifdef PARALLEL_EXTRAPOLATION
#pragma omp for
#endif
        for (int i=0; i<clear_corners; i++)
            memset( left + i*(left_dim0+1) + nrowl - clear_corners, 0, (clear_corners-i)*sizeof(complex_type) );
    }

#ifdef DEBUG
    fprintf(stderr, "Finished multiply_2d_symmetric\n");
#endif

    return 0;
}


/**
 * left and right must be Fortran contiguous. right must not be a square matrix.
 * left is overwritten with the product of left and right.
 * left must be large enough to contain the matrix product left*right.
 */
static int multiply_symmetric_worker(
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
            return multiply_symmetric_square(
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
            return multiply_symmetric_nonsquare(
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
    }

    memset( auxmatrixl, 0, cutoff*aux_dim0*sizeof(complex_type) );
    /* Extrapolate left (upper part) of left matrix */
    extrapolate_left(left, nrowl, auxmatrixl, cutoff, cutoff);
    /* Extrapolate top (left part) of right matrix */
    extrapolate_top(right, ncoll, auxmatrixr, cutoff, cutoff);

    /* Calculate matrix product. This overwrites auxmatrixl. */
    int i = (ncoll == ncolr) ?
            multiply_symmetric_square(
                    nrowl,
                    ncolr,
                    left,
                    left_dim0,
                    right,
                    right_dim0,
                    symmetry,
                    clear_corners
                    ) :
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
    if (i)
        return i;

    if (cutoff < LU_ON_GPU_THRESHOLD
            ? multiply_UL_inplace(
                cutoff,
                auxmatrixl,
                aux_dim0,
                auxmatrixr,
                aux_dim0
                )
            : multiply_UL_cuda(
                cutoff,
                auxmatrixl,
                aux_dim0,
                auxmatrixr,
                aux_dim0
                )
            )
        return 1;

    /* Add result to top left of output array. */
    const complex_type *read = auxmatrixl;
    const complex_type * const end = auxmatrixl + cutoff * cutoff;
    complex_type *write = left;
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

    return 0;
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
            PyErr_SetString(PyExc_RuntimeError, "Failed to resize (enlargen) matrix.");
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

    const int status = multiply_symmetric_worker(
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

    if (status)
    {
        Py_DECREF(fleft);
        PyErr_SetString(PyExc_RuntimeError, "Error in symmetric matrix multiplication");
        return NULL;
    }

    if (ncolr < ncoll)
    {
        npy_intp shape[] = {nrowl, ncolr};
        PyArray_Dims dims = {shape, 2};
        if (!PyArray_Resize(fleft, &dims, 1, NPY_FORTRANORDER))
        {
            Py_DECREF(fleft);
            PyErr_SetString(PyExc_RuntimeError, "Failed to resize (truncate) matrix.");
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

    int status = 0;
    for (i=0; i<nmatrices; ++i)
    {
        if (ncolr != ncoll)
            /* copy data */
            memcpy( left_data + i*left_matrixstride, PyArray_DATA(fleft) + i*PyArray_STRIDE(fleft, 2), PyArray_STRIDE(fleft, 2) );
        status = multiply_symmetric_worker(
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
        if (status)
            break;
    }

    /* Free auxilliary arrays. */
    free(auxmatrixl);
    free(auxmatrixr);

    if (status)
    {
        if (ncoll != ncolr)
            free(left_data);
        Py_DECREF(fleft);
        PyErr_SetString(PyExc_RuntimeError, "Error in symmetric matrix multiplication");
        return NULL;
    }

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
    if (!PyArg_ParseTuple(args, "O!O!i|iic", &PyArray_Type, &left, &PyArray_Type, &right, &cutoff, &symmetry, &clear_corners, &flags))
        return NULL;

    if (PyArray_TYPE(left) != NPY_COMPLEX_TYPE || PyArray_TYPE(right) != NPY_COMPLEX_TYPE)
        return PyErr_Format(PyExc_ValueError, "arrays of type complex128 required");
    if (    PyArray_NDIM(left) != PyArray_NDIM(right) ||
            PyArray_NDIM(left) < 2 ||
            PyArray_DIM(left, 1) != PyArray_DIM(right, 0))
        return PyErr_Format(PyExc_ValueError, "Shape of the 2 matrices does not allow multiplication.");

    const int min_size = cutoff + (clear_corners > 4 ? clear_corners-1 : 3);
    if (cutoff > 0 && (
                PyArray_DIM(left, 0) < min_size ||
                PyArray_DIM(left, 1) < min_size ||
                PyArray_DIM(right, 1) < min_size))
        return PyErr_Format(PyExc_ValueError, "Matrices is too small (%d, %d, %d) or cutoff too large (%d).", PyArray_DIM(left, 0), PyArray_DIM(left, 1), PyArray_DIM(right, 1), cutoff);


    if (PyArray_NDIM(left) > 2 &&
            memcmp(PyArray_DIMS(left) + 2, PyArray_DIMS(right) + 2, (PyArray_NDIM(left)-2)*sizeof(npy_intp)))
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
                abs(((int) PyArray_DIM(left,  0)) - ((int) PyArray_DIM(left,  1))) > 1 ||
                abs(((int) PyArray_DIM(left,  0)) - ((int) PyArray_DIM(right, 1))) > 1 ||
                abs(((int) PyArray_DIM(right, 0)) - ((int) PyArray_DIM(right, 1))) > 1 ))
            symmetry = 0;

    PyArrayObject *out;
    if (symmetry)
    {
        /* In this case, functions decide dyamically whether fleft needs to be copied.
         * Something like fleft will be created in these functions. */
        if (PyArray_NDIM(left) == 2)
            out = multiply_extended_2d_symmetric(left, fright, cutoff, symmetry, clear_corners, flags);
        else
            out = multiply_extended_nd_symmetric(left, fright, cutoff, symmetry, clear_corners, flags);
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


/************************************************
 *           PYTHON MODULE SETUP                *
 ************************************************/


/* define functions in module */
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
     {NULL, NULL, 0, NULL}
};


/* module initialization (python 3) */
static struct PyModuleDef cModPyDem = {
    PyModuleDef_HEAD_INIT,
    "rtrg_cublas",
    PyDoc_STR(
            "Auxilliary functions for calculations with Floquet matrices.\n"
            "Floquet matrices are represented by square matrices, which should\n"
            "be extrapolated along the diagonal to avoid truncation effects.\n"
            "Non-square matrices may also be used to account for reduced matrices\n"
            "in special symmetric cases.\n\n"
            "NOTE:\n"
            "    Use\n"
            "        rtrg_cublas.multiply_extended(b.T, a.T, cutoff, ...).T\n"
            "        rtrg_cublas.invert_extended(a.T, cutoff, ...).T\n"
            "        rtrg_cublas.extend_matrix(a.T, cutoff).T\n"
            "    where the last 2 dimensions of a, b define Floquet matrices\n"
            "    instead of\n"
            "        rtrg_cublas.multiply_extended(a, b, cutoff, ...)\n"
            "        rtrg_cublas.invert_extended(a, cutoff, ...)\n"
            "        rtrg_cublas.extend_matrix(a, cutoff)\n"
            "    for better performance. This will pass F-contiguous arrays to\n"
            "    rtrg_cublas if original arrays were C-contiguous (standard in numpy).\n"
        ),
    -1,
    FloquetMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_rtrg_cublas(void)
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

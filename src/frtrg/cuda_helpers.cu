#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <complex.h>
#include "rtrg_cublas.h"

#ifdef __cplusplus
extern "C" {
#endif

char cuda_gemm(
        const int nrowl,
        const int ncolr,
        const int ncoll,
        const void *prefactor,
        const void *left,
        const int left_dim0,
        const void *right,
        const int right_dim0,
        const void *weight_c,
        void *output,
        const int out_dim0
        )
{
    complex_type_cuda *dev_left = NULL, *dev_right = NULL, *dev_out = NULL;
    cudaError_t cuda_err;
    cublasHandle_t handle;
    cublasStatus_t status;
    cuda_err = cudaMalloc(&dev_left, nrowl*ncoll*sizeof(complex_type_cuda));
    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "UNHANDLED ERROR in cuda_gemm: GPU memory allocation (left matrix)\n");
        return 1;
    }
    cuda_err = cudaMalloc(&dev_right, ncoll*ncolr*sizeof(complex_type_cuda));
    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "UNHANDLED ERROR in cuda_gemm: GPU memory allocation (right matrix)\n");
        goto cuda_error;
    }
    cuda_err = cudaMalloc(&dev_out, nrowl*ncolr*sizeof(complex_type_cuda));
    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "UNHANDLED ERROR in cuda_gemm: GPU memory allocation (output matrix)\n");
        goto cuda_error;
    }
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "UNHANDLED ERROR in cuda_gemm: cublas initialization\n");
        goto cuda_error;
    }
    status = cublasSetMatrix(nrowl, ncoll, sizeof(complex_type_cuda), left, left_dim0, dev_left, nrowl);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "UNHANDLED ERROR in cuda_gemm: set matrix (matrix A) %d %d %d %d\n", nrowl, ncoll, left_dim0, nrowl);
        goto cuda_error;
    }
    status = cublasSetMatrix(ncoll, ncolr, sizeof(complex_type_cuda), right, right_dim0, dev_right, ncoll);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "UNHANDLED ERROR in cuda_gemm: set matrix (matrix B) %d %d %d %d\n", ncoll, ncolr, right_dim0, ncoll);
        goto cuda_error;
    }
    /* Ugly implementation: cast to complex_type instead of complex_type_cuda because it works */
    if (*((const complex_type*) weight_c) != 0.)
    {
        status = cublasSetMatrix(nrowl, ncolr, sizeof(complex_type_cuda), output, out_dim0, dev_out, nrowl);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            fprintf(stderr, "UNHANDLED ERROR in cuda_gemm: set matrix (matrix C) %d %d %d %d\n", nrowl, ncolr, out_dim0, nrowl);
            goto cuda_error;
        }
    }

    status = cu_gemm(
            handle,
            CUBLAS_OP_N, // A is not modified (no adjoint or transpose)
            CUBLAS_OP_N, // B is not modified (no adjoint or transpose)
            nrowl, // rows of A (int)
            ncolr, // columns of B (int)
            ncoll, // columns of A = rows of B (int)
            (const complex_type_cuda*) prefactor, // global prefactor
            dev_left, // matrix A
            nrowl, // first dimension of A (int)
            dev_right, // matrix B
            ncoll,  // first dimension of B (int)
            (const complex_type_cuda*) weight_c, // weight of C
            dev_out, // matrix C
            nrowl // first dimension of C
            );
    cudaFree(dev_left);
    cudaFree(dev_right);
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasGetMatrix(nrowl, ncolr, sizeof(complex_type_cuda), dev_out, nrowl, output, out_dim0);
    else
        fprintf(stderr, "UNHANDLED ERROR in cuda_gemm: gemm\n");
    cudaFree(dev_out);
    cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "UNHANDLED ERROR in cuda_gemm: gemm or get matrix\n");
        return 2;
    }

    return 0;

cuda_error:
    if (dev_left)
        cudaFree(dev_left);
    if (dev_right)
        cudaFree(dev_right);
    if (dev_out)
        cudaFree(dev_out);
    /* handles is only created if memory for all arrays was allocated successfully */
    if (dev_left && dev_right && dev_out)
        cublasDestroy(handle);
    return 1;
}


char cuda_trmm(
        const int flags,
        const int nrowb,
        const int ncolb,
        const void *prefactor,
        const void *a,
        const int a_dim0,
        void *b,
        const int b_dim0
        )
{
    complex_type_cuda *dev_a = NULL, *dev_b = NULL;
    cudaError_t cuda_err;
    cublasHandle_t handle;
    cublasStatus_t status;
    const int sizea = (flags & LeftMultiplication) ? nrowb : ncolb;
    cuda_err = cudaMalloc(&dev_a, sizea*sizea*sizeof(complex_type_cuda));
    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "UNHANDLED ERROR in cuda_trmm: GPU memory allocation (matrix A)\n");
        return 1;
    }
    cuda_err = cudaMalloc(&dev_b, ncolb*nrowb*sizeof(complex_type_cuda));
    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "UNHANDLED ERROR in cuda_trmm: GPU memory allocation (matrix B)\n");
        goto cuda_error;
    }
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "UNHANDLED ERROR in cuda_trmm: cublas initialization\n");
        goto cuda_error;
    }
    status = cublasSetMatrix(sizea, sizea, sizeof(complex_type_cuda), a, a_dim0, dev_a, sizea);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "UNHANDLED ERROR in cuda_trmm: set matrix (matrix A)\n");
        goto cuda_error;
    }
    status = cublasSetMatrix(nrowb, ncolb, sizeof(complex_type_cuda), b, b_dim0, dev_b, nrowb);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "UNHANDLED ERROR in cuda_trmm: set matrix (matrix B)\n");
        goto cuda_error;
    }

    status = cu_trmm(
            handle,
            (flags & LeftMultiplication) ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT, // order AB or BA
            (flags & UpperTriangular) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER, // A is upper or lower triangular matrix
            CUBLAS_OP_N, // A is not modified (no adjoint or transpose)
            (flags & UnitTriangular) ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, // A is usually not unit triangular
            nrowb, // rows of B (int)
            ncolb, // columns of B (int)
            (const complex_type_cuda*) prefactor, // global prefactor
            dev_a, // matrix A
            sizea, // first dimension of A (int)
            dev_b, // matrix B (input)
            nrowb,  // first dimension of B (int)
            dev_b, // matrix B (output)
            nrowb  // first dimension of B (int)
            );

    cudaFree(dev_a);
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasGetMatrix(nrowb, ncolb, sizeof(complex_type_cuda), dev_b, nrowb, b, b_dim0);
    else
        fprintf(stderr, "UNHANDLED ERROR in cuda_trmm: trmm\n");
    cudaFree(dev_b);
    cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "UNHANDLED ERROR in cuda_trmm: trmm or get matrix\n");
        return 2;
    }

    return 0;

cuda_error:
    if (dev_a)
        cudaFree(dev_a);
    if (dev_b)
        cudaFree(dev_b);
    /* handles is only created if memory for all arrays was allocated successfully */
    if (dev_a && dev_b)
        cublasDestroy(handle);
    return 1;
}

/**
 * A is an upper triangular matrix.
 * B is a lower triangular matrix.
 * Both are in Fortran order (columns major) not unit triangular.
 * The product AB overwrites A.
 * Both matrices must have shape (size, size) and lie in device memory.
 */
cublasStatus_t multiply_UL_cuda_worker(
        cublasHandle_t handle,
        const int size,
        complex_type_cuda *a,
        const int a_dim0,
        const complex_type_cuda *b,
        const int b_dim0
        )
{
#ifdef DEBUG
    fprintf(stderr, "Starting multiply_UL_cuda_worker %d\n", size);
#endif
    static const complex_type_cuda one = make_cuDoubleComplex(1., 0.);
    if (size < TRIANGULAR_OPTIMIZE_THRESHOLD_GPU)
    {
#ifdef DEBUG
        fprintf(stderr, "multiply_UL_cuda_worker: final step %d\n", size);
#endif
        return cu_trmm(
                handle,
                CUBLAS_SIDE_RIGHT, // order: this means B := B A
                CUBLAS_FILL_MODE_LOWER, // A is a lower triangular matrix
                CUBLAS_OP_N, // A is not modified (no adjoint or transpose)
                CUBLAS_DIAG_NON_UNIT, // A is not unit triangular
                size, // rows of B (int)
                size, // columns of B (int)
                &one, // global prefactor
                b, // matrix A
                b_dim0, // first dimension of A (int)
                a, // matrix B
                a_dim0,  // first dimension of B (int)
                a, // matrix B
                a_dim0  // first dimension of B (int)
                );
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
#ifdef DEBUG
    fprintf(stderr, "multiply_UL_cuda_worker: memcpy %d\n", size);
#endif
    while (++i<part1)
        cudaMemcpy( a + i*a_dim0, b + i*b_dim0, part2 * sizeof(complex_type_cuda), cudaMemcpyDeviceToDevice );
    a -= part1;
    b -= part1;

#ifdef DEBUG
    fprintf(stderr, "multiply_UL_cuda_worker: step 1 %d\n", size);
#endif
    cublasStatus_t status = cu_trmm(
            handle,
            CUBLAS_SIDE_LEFT, // order: this means B := A B
            CUBLAS_FILL_MODE_UPPER, // A is an upper triangular matrix
            CUBLAS_OP_N, // A is not modified (no adjoint or transpose)
            CUBLAS_DIAG_NON_UNIT, // A is not unit triangular
            part2, // rows of B (int)
            part1, // columns of B (int)
            &one, // global prefactor
            a + part1*(1 + a_dim0), // matrix A
            a_dim0, // first dimension of A (int)
            a + part1, // matrix B
            a_dim0,  // first dimension of B (int)
            a + part1, // matrix B
            a_dim0  // first dimension of B (int)
            );
    if (status != CUBLAS_STATUS_SUCCESS)
        return status;

    /* Step 2: overwrite Ad with Ad Bd */
#ifdef DEBUG
    fprintf(stderr, "multiply_UL_cuda_worker: step 2 %d\n", size);
#endif
    status = multiply_UL_cuda_worker(handle, part2, a + (a_dim0+1)*part1, a_dim0, b + (b_dim0+1)*part1, b_dim0);
    if (status != CUBLAS_STATUS_SUCCESS)
        return status;

    /* Step 3: overwrite Aa with Aa Ba */
#ifdef DEBUG
    fprintf(stderr, "multiply_UL_cuda_worker: step 3 %d\n", size);
#endif
    status = multiply_UL_cuda_worker(handle, part1, a, a_dim0, b, b_dim0);
    if (status != CUBLAS_STATUS_SUCCESS)
        return status;

    /* Step 4: add Ab Bc to Aa */
#ifdef DEBUG
    fprintf(stderr, "multiply_UL_cuda_worker: step 4 %d\n", size);
#endif
    status = cu_gemm(
            handle,
            CUBLAS_OP_N, // A is not modified (no adjoint or transpose)
            CUBLAS_OP_N, // B is not modified (no adjoint or transpose)
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
    if (status != CUBLAS_STATUS_SUCCESS)
        return status;

    /* Step 5: overwrite Ab with Ab Bd */
#ifdef DEBUG
    fprintf(stderr, "multiply_UL_cuda_worker: step 5 %d\n", size);
#endif
    status = cu_trmm(
            handle,
            CUBLAS_SIDE_RIGHT, // order: this means B := B A
            CUBLAS_FILL_MODE_LOWER, // A is a lower triangular matrix
            CUBLAS_OP_N, // A is not modified (no adjoint or transpose)
            CUBLAS_DIAG_NON_UNIT, // A is not unit triangular
            part1, // rows of B (int)
            part2, // columns of B (int)
            &one, // global prefactor
            b + (1 + b_dim0)*part1, // matrix A
            b_dim0, // first dimension of A (int)
            a + part1*a_dim0, // matrix B
            a_dim0,  // first dimension of B (int)
            a + part1*a_dim0, // matrix B
            a_dim0  // first dimension of B (int)
            );
#ifdef DEBUG
    fprintf(stderr, "multiply_UL_cuda_worker: done %d\n", size);
#endif
    return status;
}

int multiply_UL_cuda(
        const int size,
        void *a,
        const int a_dim0,
        const void *b,
        const int b_dim0
        )
{
#ifdef DEBUG
    fprintf(stderr, "Entering multiply_UL_cuda %d %d %d\n", size, a_dim0, b_dim0);
#endif
    if (a_dim0 < size || b_dim0 < size)
        return 1;
    complex_type_cuda *dev_a = NULL, *dev_b = NULL;
    cudaError_t cuda_err;
    cublasHandle_t handle;
    cublasStatus_t status;
#ifdef DEBUG
    fprintf(stderr, "multiply_UL_cuda: allocating device memory\n");
#endif
    cuda_err = cudaMalloc(&dev_a, size*size*sizeof(complex_type_cuda));
    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "UNHANDLED ERROR in multiply_UL_cuda: GPU memory allocation (matrix A)\n");
        return 1;
    }
    cuda_err = cudaMalloc(&dev_b, size*size*sizeof(complex_type_cuda));
    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "UNHANDLED ERROR in multiply_UL_cuda: GPU memory allocation (matrix B)\n");
        goto cuda_error;
    }
#ifdef DEBUG
    fprintf(stderr, "multiply_UL_cuda: creating handle\n");
#endif
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "UNHANDLED ERROR in multiply_UL_cuda: cublas initialization\n");
        goto cuda_error;
    }
#ifdef DEBUG
    fprintf(stderr, "multiply_UL_cuda: copying data to device\n");
#endif
    status = cublasSetMatrix(size, size, sizeof(complex_type_cuda), a, a_dim0, dev_a, size);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "UNHANDLED ERROR in multiply_UL_cuda: set matrix (matrix A)\n");
        goto cuda_error;
    }
    status = cublasSetMatrix(size, size, sizeof(complex_type_cuda), b, b_dim0, dev_b, size);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "UNHANDLED ERROR in multiply_UL_cuda: set matrix (matrix B)\n");
        goto cuda_error;
    }
#ifdef DEBUG
    fprintf(stderr, "multiply_UL_cuda: calling worker\n");
#endif
    if (multiply_UL_cuda_worker(handle, size, dev_a, size, dev_b, size))
    {
        fprintf(stderr, "UNHANDLED ERROR in multiply_UL_cuda: worker\n");
        goto cuda_error;
    }
#ifdef DEBUG
    fprintf(stderr, "multiply_UL_cuda: copying data to host\n");
#endif
    status = cublasGetMatrix(size, size, sizeof(complex_type_cuda), dev_a, size, a, a_dim0);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "UNHANDLED ERROR in multiply_UL_cuda: get matrix\n");
        goto cuda_error;
    }

#ifdef DEBUG
    fprintf(stderr, "multiply_UL_cuda: cleaning up\n");
#endif
    cudaFree(dev_a);
    cudaFree(dev_b);
    cublasDestroy(handle);
#ifdef DEBUG
    fprintf(stderr, "multiply_UL_cuda: done\n");
#endif
    return 0;

cuda_error:
    if (dev_a)
        cudaFree(dev_a);
    if (dev_b)
        cudaFree(dev_b);
    if (dev_a && dev_b)
        cublasDestroy(handle);
    return 1;
}

/**
 * A is an upper triangular matrix.
 * B is a lower triangular matrix.
 * Both are in Fortran order (columns major) not unit triangular.
 * The product AB overwrites A.
 * Both matrices must have shape (size, size) and lie in device memory.
 */
cublasStatus_t multiply_LU_cuda_worker(
        cublasHandle_t handle,
        const int size,
        complex_type_cuda *a,
        const int a_dim0,
        const complex_type_cuda *b,
        const int b_dim0
        )
{
#ifdef DEBUG
    fprintf(stderr, "Starting multiply_LU_cuda_worker %d\n", size);
#endif
    static const complex_type_cuda one = make_cuDoubleComplex(1., 0.);
    if (size < TRIANGULAR_OPTIMIZE_THRESHOLD_GPU)
    {
#ifdef DEBUG
        fprintf(stderr, "multiply_LU_cuda_worker: final step %d\n", size);
#endif
        return cu_trmm(
                handle,
                CUBLAS_SIDE_RIGHT, // order: this means B := B A
                CUBLAS_FILL_MODE_UPPER, // A is an upper triangular matrix
                CUBLAS_OP_N, // A is not modified (no adjoint or transpose)
                CUBLAS_DIAG_NON_UNIT, // A is not unit triangular
                size, // rows of B (int)
                size, // columns of B (int)
                &one, // global prefactor
                b, // matrix A
                b_dim0, // first dimension of A (int)
                a, // matrix B
                a_dim0,  // first dimension of B (int)
                a, // matrix B
                a_dim0  // first dimension of B (int)
                );
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

    /* Step 1: overwrite Ab with Aa Bb.
     * This requires that first Bb is copied to Ab. */
    a += a_dim0*part1;
    b += b_dim0*part1;
    int i = -1;
#ifdef DEBUG
    fprintf(stderr, "multiply_LU_cuda_worker: memcpy %d\n", size);
#endif
    while (++i<part2)
        cudaMemcpy( a + i*a_dim0, b + i*b_dim0, part1 * sizeof(complex_type_cuda), cudaMemcpyDeviceToDevice );
    a -= a_dim0*part1;
    b -= b_dim0*part1;

#ifdef DEBUG
    fprintf(stderr, "multiply_LU_cuda_worker: step 1 %d\n", size);
#endif
    cublasStatus_t status = cu_trmm(
            handle,
            CUBLAS_SIDE_LEFT, // order: this means B := A B
            CUBLAS_FILL_MODE_LOWER, // A is a lower triangular matrix
            CUBLAS_OP_N, // A is not modified (no adjoint or transpose)
            CUBLAS_DIAG_NON_UNIT, // A is not unit triangular
            part1, // rows of B (int)
            part2, // columns of B (int)
            &one, // global prefactor
            a, // matrix A
            a_dim0, // first dimension of A (int)
            a + a_dim0*part1, // matrix B
            a_dim0,  // first dimension of B (int)
            a + a_dim0*part1, // matrix B
            a_dim0  // first dimension of B (int)
            );
    if (status != CUBLAS_STATUS_SUCCESS)
        return status;

    /* Step 2: overwrite Ad with Ad Bd */
#ifdef DEBUG
    fprintf(stderr, "multiply_LU_cuda_worker: step 2 %d\n", size);
#endif
    status = multiply_LU_cuda_worker(handle, part1, a, a_dim0, b, b_dim0);
    if (status != CUBLAS_STATUS_SUCCESS)
        return status;

    /* Step 3: overwrite Aa with Aa Ba */
#ifdef DEBUG
    fprintf(stderr, "multiply_LU_cuda_worker: step 3 %d\n", size);
#endif
    status = multiply_LU_cuda_worker(handle, part2, a + (a_dim0+1)*part1, a_dim0, b + (b_dim0+1)*part1, b_dim0);
    if (status != CUBLAS_STATUS_SUCCESS)
        return status;

    /* Step 4: add Ab Bc to Aa */
#ifdef DEBUG
    fprintf(stderr, "multiply_LU_cuda_worker: step 4 %d\n", size);
#endif
    status = cu_gemm(
            handle,
            CUBLAS_OP_N, // A is not modified (no adjoint or transpose)
            CUBLAS_OP_N, // B is not modified (no adjoint or transpose)
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
    if (status != CUBLAS_STATUS_SUCCESS)
        return status;

    /* Step 5: overwrite Ab with Ab Bd */
#ifdef DEBUG
    fprintf(stderr, "multiply_LU_cuda_worker: step 5 %d\n", size);
#endif
    status = cu_trmm(
            handle,
            CUBLAS_SIDE_RIGHT, // order: this means B := B A
            CUBLAS_FILL_MODE_UPPER, // A is an upper triangular matrix
            CUBLAS_OP_N, // A is not modified (no adjoint or transpose)
            CUBLAS_DIAG_NON_UNIT, // A is not unit triangular
            part2, // rows of B (int)
            part1, // columns of B (int)
            &one, // global prefactor
            b, // matrix A
            b_dim0, // first dimension of A (int)
            a + part1, // matrix B
            a_dim0,  // first dimension of B (int)
            a + part1, // matrix B
            a_dim0  // first dimension of B (int)
            );
#ifdef DEBUG
    fprintf(stderr, "multiply_LU_cuda_worker: done %d\n", size);
#endif
    return status;
}

int multiply_LU_cuda(
        const int size,
        void *a,
        const int a_dim0,
        const void *b,
        const int b_dim0
        )
{
#ifdef DEBUG
    fprintf(stderr, "Entering multiply_LU_cuda %d %d %d\n", size, a_dim0, b_dim0);
#endif
    if (a_dim0 < size || b_dim0 < size)
        return 1;
    complex_type_cuda *dev_a = NULL, *dev_b = NULL;
    cudaError_t cuda_err;
    cublasHandle_t handle;
    cublasStatus_t status;
#ifdef DEBUG
    fprintf(stderr, "multiply_LU_cuda: allocating device memory\n");
#endif
    cuda_err = cudaMalloc(&dev_a, size*size*sizeof(complex_type_cuda));
    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "UNHANDLED ERROR in multiply_LU_cuda: GPU memory allocation (matrix A)\n");
        return 1;
    }
    cuda_err = cudaMalloc(&dev_b, size*size*sizeof(complex_type_cuda));
    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "UNHANDLED ERROR in multiply_LU_cuda: GPU memory allocation (matrix B)\n");
        goto cuda_error;
    }
#ifdef DEBUG
    fprintf(stderr, "multiply_LU_cuda: creating handle\n");
#endif
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "UNHANDLED ERROR in multiply_LU_cuda: cublas initialization\n");
        goto cuda_error;
    }
#ifdef DEBUG
    fprintf(stderr, "multiply_LU_cuda: copying data to device\n");
#endif
    status = cublasSetMatrix(size, size, sizeof(complex_type_cuda), a, a_dim0, dev_a, size);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "UNHANDLED ERROR in multiply_LU_cuda: set matrix (matrix A)\n");
        goto cuda_error;
    }
    status = cublasSetMatrix(size, size, sizeof(complex_type_cuda), b, b_dim0, dev_b, size);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "UNHANDLED ERROR in multiply_LU_cuda: set matrix (matrix B)\n");
        goto cuda_error;
    }
#ifdef DEBUG
    fprintf(stderr, "multiply_LU_cuda: calling worker\n");
#endif
    if (multiply_LU_cuda_worker(handle, size, dev_a, size, dev_b, size))
    {
        fprintf(stderr, "UNHANDLED ERROR in multiply_LU_cuda: worker\n");
        goto cuda_error;
    }
#ifdef DEBUG
    fprintf(stderr, "multiply_LU_cuda: copying data to host\n");
#endif
    status = cublasGetMatrix(size, size, sizeof(complex_type_cuda), dev_a, size, a, a_dim0);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "UNHANDLED ERROR in multiply_LU_cuda: get matrix\n");
        goto cuda_error;
    }

#ifdef DEBUG
    fprintf(stderr, "multiply_LU_cuda: cleaning up\n");
#endif
    cudaFree(dev_a);
    cudaFree(dev_b);
    cublasDestroy(handle);
#ifdef DEBUG
    fprintf(stderr, "multiply_LU_cuda: done\n");
#endif
    return 0;

cuda_error:
    if (dev_a)
        cudaFree(dev_a);
    if (dev_b)
        cudaFree(dev_b);
    if (dev_a && dev_b)
        cublasDestroy(handle);
    return 1;
}


#ifdef __cplusplus
} /* extern "C" */
#endif

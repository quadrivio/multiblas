//
//  gemm_blas.cpp
//  multiBLAS.XC
//
//  Created by michael on 7/23/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#include "gemm_blas.h"

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

void gemm_blas_d(const double *inMatrixA, int nrowA, int ncolA, bool transposeA,
                 const double *inMatrixB, int nrowB, int ncolB, bool transposeB,
                 double alpha, double beta, double *outMatrix)
{
    const CBLAS_ORDER order = CblasColMajor;
    const CBLAS_TRANSPOSE transA = transposeA ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE transB = transposeB ? CblasTrans : CblasNoTrans;
    
    const int lda = nrowA;  // first dimension of A (rows), before any transpose
    const int ldb = nrowB;  // first dimension of B (rows), before any transpose
    const int ldc = transA ? ncolA : nrowA;  // first dimension of C (rows)
    
    const int M = transA ? ncolA : nrowA;    // rows in A (after transpose, if any) and C
    const int N = transB ? nrowB : ncolB;    // cols in B (after transpose, if any) and C
    const int K = transA ? nrowA : ncolA;    // cols in A and rows in B (after transposes, if any)
    
    cblas_dgemm(order, transA, transB, M, N, K, alpha, inMatrixA, lda, inMatrixB, ldb, beta, outMatrix, ldc);
}

void gemm_blas_f(const float *inMatrixA, int nrowA, int ncolA, bool transposeA,
                 const float *inMatrixB, int nrowB, int ncolB, bool transposeB,
                 float alpha, float beta, float *outMatrix)
{
    const CBLAS_ORDER order = CblasColMajor;
    const CBLAS_TRANSPOSE transA = transposeA ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE transB = transposeB ? CblasTrans : CblasNoTrans;
    
    const int lda = nrowA;  // first dimension of A (rows), before any transpose
    const int ldb = nrowB;  // first dimension of B (rows), before any transpose
    const int ldc = transA ? ncolA : nrowA;  // first dimension of C (rows)
    
    const int M = transA ? ncolA : nrowA;    // rows in A (after transpose, if any) and C
    const int N = transB ? nrowB : ncolB;    // cols in B (after transpose, if any) and C
    const int K = transA ? nrowA : ncolA;    // cols in A and rows in B (after transposes, if any)
    
    cblas_sgemm(order, transA, transB, M, N, K, alpha, inMatrixA, lda, inMatrixB, ldb, beta, outMatrix, ldc);
}

//
//  gemm_blas.cpp
//  multiBLAS.XC
//
//  Created by michael on 7/23/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#include "gemm_blas.h"
#include "nullptr.h"

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <cmath>
#include <cstddef>
#include <dlfcn.h>

typedef void (*cblas_dgemm_type)(const enum CBLAS_ORDER __Order,
                 const enum CBLAS_TRANSPOSE __TransA,
                 const enum CBLAS_TRANSPOSE __TransB, const int __M, const int __N,
                 const int __K, const double __alpha, const double *__A,
                 const int __lda, const double *__B, const int __ldb,
                 const double __beta, double *__C, const int __ldc);

typedef void (*cblas_sgemm_type)(const enum CBLAS_ORDER __Order,
                 const enum CBLAS_TRANSPOSE __TransA,
                 const enum CBLAS_TRANSPOSE __TransB, const int __M, const int __N,
                 const int __K, const float __alpha, const float *__A, const int __lda,
                 const float *__B, const int __ldb, const float __beta, float *__C,
                 const int __ldc);

cblas_dgemm_type cblas_dgemm_ptr = nullptr;
cblas_sgemm_type cblas_sgemm_ptr = nullptr;

bool cblas_sgemm_available()
{
    if (cblas_sgemm_ptr == nullptr) {
        cblas_sgemm_ptr = (cblas_sgemm_type)dlsym(RTLD_DEFAULT, "cblas_sgemm");
    }
    
    return cblas_sgemm_ptr != nullptr;
}

bool cblas_dgemm_available()
{
    if (cblas_dgemm_ptr == nullptr) {
        cblas_dgemm_ptr = (cblas_dgemm_type)dlsym(RTLD_DEFAULT, "cblas_dgemm");
    }
    
    return cblas_sgemm_ptr != nullptr;
}

void gemm_blas_d(const double *inMatrixA, int nrowA, int ncolA, bool transposeA,
                 const double *inMatrixB, int nrowB, int ncolB, bool transposeB,
                 double alpha, double beta, double *outMatrix)
{
    if (!cblas_dgemm_available()) {
        const int M = transposeA ? ncolA : nrowA;    // rows in A (after transpose, if any) and C
        const int N = transposeB ? nrowB : ncolB;    // cols in B (after transpose, if any) and C
        
        // double-precision not available
        for (int i = 0; i < M * N; i++) {
            outMatrix[i] = NAN;
        }
        
    } else {
        const int lda = nrowA;                  // first dimension of A (rows), before any transpose
        const int ldb = nrowB;                  // first dimension of B (rows), before any transpose
        const int ldc = transposeA ? ncolA : nrowA; // first dimension of C (rows)
        
        const int M = transposeA ? ncolA : nrowA;   // rows in A (after transpose, if any) and C
        const int N = transposeB ? nrowB : ncolB;   // cols in B (after transpose, if any) and C
        const int K = transposeA ? nrowA : ncolA;   // cols in A and rows in B (after transposes, if any)
        
        const CBLAS_ORDER order = CblasColMajor;
        const CBLAS_TRANSPOSE transA = transposeA ? CblasTrans : CblasNoTrans;
        const CBLAS_TRANSPOSE transB = transposeB ? CblasTrans : CblasNoTrans;
        
        (*cblas_dgemm_ptr)(order, transA, transB, M, N, K, alpha, inMatrixA, lda, inMatrixB, ldb, beta, outMatrix, ldc);
    }
}

void gemm_blas_f(const float *inMatrixA, int nrowA, int ncolA, bool transposeA,
                 const float *inMatrixB, int nrowB, int ncolB, bool transposeB,
                 float alpha, float beta, float *outMatrix)
{
    if (!cblas_sgemm_available()) {
        const int M = transposeA ? ncolA : nrowA;    // rows in A (after transpose, if any) and C
        const int N = transposeB ? nrowB : ncolB;    // cols in B (after transpose, if any) and C
        
        // single-precision not available
        for (int i = 0; i < M * N; i++) {
            outMatrix[i] = NAN;
        }
        
    } else {
        const int lda = nrowA;  // first dimension of A (rows), before any transpose
        const int ldb = nrowB;  // first dimension of B (rows), before any transpose
        const int ldc = transposeA ? ncolA : nrowA;  // first dimension of C (rows)
        
        const int M = transposeA ? ncolA : nrowA;    // rows in A (after transpose, if any) and C
        const int N = transposeB ? nrowB : ncolB;    // cols in B (after transpose, if any) and C
        const int K = transposeA ? nrowA : ncolA;    // cols in A and rows in B (after transposes, if any)
        
        const CBLAS_ORDER order = CblasColMajor;
        const CBLAS_TRANSPOSE transA = transposeA ? CblasTrans : CblasNoTrans;
        const CBLAS_TRANSPOSE transB = transposeB ? CblasTrans : CblasNoTrans;
        
        (*cblas_sgemm_ptr)(order, transA, transB, M, N, K, alpha, inMatrixA, lda, inMatrixB, ldb, beta, outMatrix, ldc);
    }
}

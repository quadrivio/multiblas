//
//  gemm_r.cpp
//  multiBLAS.XC
//
//  Created by michael on 7/23/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#include "shim.h"

#if RPACKAGE

#include "gemm_r.h"

#include <R_ext/BLAS.h>

void gemm_r_d(const double *inMatrixA, int nrowA, int ncolA, bool transposeA,
                 const double *inMatrixB, int nrowB, int ncolB, bool transposeB,
                 double alpha, double beta, double *outMatrix)
{
    const int lda = nrowA;                  // first dimension of A (rows), before any transpose
    const int ldb = nrowB;                  // first dimension of B (rows), before any transpose
    const int ldc = transposeA ? ncolA : nrowA; // first dimension of C (rows)
    
    const int M = transposeA ? ncolA : nrowA;   // rows in A (after transpose, if any) and C
    const int N = transposeB ? nrowB : ncolB;   // cols in B (after transpose, if any) and C
    const int K = transposeA ? nrowA : ncolA;   // cols in A and rows in B (after transposes, if any)
    
    const char transa = transposeA ? 'T' : 'N';
    const char transb = transposeB ? 'T' : 'N';
    
    F77_NAME(dgemm)(&transa, &transb, &M, &N, &K, &alpha, inMatrixA, &lda, inMatrixB, &ldb, &beta, outMatrix, &ldc);
}

#endif

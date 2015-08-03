//
//  crossprod_blas.cpp
//  OpenCL.ImpXC
//
//  Created by michael on 6/2/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#include "crossprod_blas.h"
#include "crossprod_naive.h"
#include "shim.h"

#include <cmath>

#if RPACKAGE
#include <R_ext/BLAS.h>
#elif defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

void crossprod_blas_d(double *inMatrix, double *outMatrix, int nrow, int ncol)
{
    
    const int lda = nrow;
    const int ldc = ncol;
    
    const double alpha = 1.0;
    
#if RPACKAGE
    const char uplo = 'U';
    const char trans = 'T';

    const double beta = 0.0;

    F77_NAME(dsyrk)(&uplo, &trans, &ncol, &nrow, &alpha, inMatrix, &lda, &beta, outMatrix, &ldc);
    
#else
    const CBLAS_ORDER order = CblasColMajor;
    const CBLAS_TRANSPOSE transA = CblasTrans;
    const CBLAS_UPLO uplo = CblasUpper;
    
    cblas_dsyrk(order, uplo, transA, ncol, nrow, alpha, inMatrix, lda, 0.0, outMatrix, ldc);
#endif
    
    symmetrizeSquare_d(outMatrix, ncol);
}

void crossprod_blas_f(float *inMatrix, float *outMatrix, int nrow, int ncol)
{
#if RPACKAGE
    // single-precision not available
    for (int i = 0; i < ncol * ncol; i++) {
        outMatrix[i] = NAN;
    }
    
#else
    const CBLAS_ORDER order = CblasColMajor;
    const CBLAS_TRANSPOSE transA = CblasTrans;
    
    const int lda = nrow;
    const int ldc = ncol;
    
    const float alpha = 1.0;
    
    const CBLAS_UPLO uplo = CblasUpper;
    
    cblas_ssyrk(order, uplo, transA, ncol, nrow, alpha, inMatrix, lda, 0.0, outMatrix, ldc);
    symmetrizeSquare_f(outMatrix, ncol);
#endif
}

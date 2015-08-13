//
//  crossprod_blas.cpp
//  OpenCL.ImpXC
//
//  Created by michael on 6/2/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#include "crossprod_blas.h"
#include "crossprod_naive.h"

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <cmath>
#include <dlfcn.h>

typedef void (*cblas_dsyrk_type)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                         const enum CBLAS_TRANSPOSE __Trans, const int __N, const int __K,
                         const double __alpha, const double *__A, const int __lda,
                         const double __beta, double *__C, const int __ldc);

typedef void (*cblas_ssyrk_type)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                         const enum CBLAS_TRANSPOSE __Trans, const int __N, const int __K,
                         const float __alpha, const float *__A, const int __lda,
                         const float __beta, float *__C, const int __ldc);

cblas_dsyrk_type cblas_dsyrk_ptr = nullptr;
cblas_ssyrk_type cblas_ssyrk_ptr = nullptr;

bool cblas_dsyrk_available()
{
    if (cblas_dsyrk_ptr == nullptr) {
        cblas_dsyrk_ptr = (cblas_dsyrk_type)dlsym(RTLD_DEFAULT, "cblas_dsyrk");
    }
    
    return cblas_ssyrk_ptr != nullptr;
}

bool cblas_ssyrk_available()
{
    if (cblas_ssyrk_ptr == nullptr) {
        cblas_ssyrk_ptr = (cblas_ssyrk_type)dlsym(RTLD_DEFAULT, "cblas_ssyrk");
    }
    
    return cblas_ssyrk_ptr != nullptr;
}

void crossprod_blas_d(double *inMatrix, double *outMatrix, int nrow, int ncol)
{
    if (!cblas_dsyrk_available()) {
        // double-precision not available
        for (int i = 0; i < ncol * ncol; i++) {
            outMatrix[i] = NAN;
        }
        
    } else {
        const int lda = nrow;
        const int ldc = ncol;
        
        const double alpha = 1.0;
        
        const CBLAS_ORDER order = CblasColMajor;
        const CBLAS_TRANSPOSE transA = CblasTrans;
        const CBLAS_UPLO uplo = CblasUpper;
        
        (*cblas_dsyrk_ptr)(order, uplo, transA, ncol, nrow, alpha, inMatrix, lda, 0.0, outMatrix, ldc);
        
        symmetrizeSquare_d(outMatrix, ncol);
    }
}

void crossprod_blas_f(float *inMatrix, float *outMatrix, int nrow, int ncol)
{
    if (!cblas_ssyrk_available()) {
        // single-precision not available
        for (int i = 0; i < ncol * ncol; i++) {
            outMatrix[i] = NAN;
        }
        
    } else {
        const CBLAS_ORDER order = CblasColMajor;
        const CBLAS_TRANSPOSE transA = CblasTrans;
        
        const int lda = nrow;
        const int ldc = ncol;
        
        const float alpha = 1.0;
        
        const CBLAS_UPLO uplo = CblasUpper;
        
        (*cblas_ssyrk_ptr)(order, uplo, transA, ncol, nrow, alpha, inMatrix, lda, 0.0, outMatrix, ldc);
        symmetrizeSquare_f(outMatrix, ncol);
    }
}

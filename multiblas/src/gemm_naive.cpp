//
//  gemm_naive.cpp
//  multiBLAS.XC
//
//  Created by michael on 7/23/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#include "gemm_naive.h"
#include "shim.h"

#include <cstdlib>
#include <cstring>
#include <string>

void gemm_naive_d(const double *inMatrixA, int nrowA, int ncolA, bool transposeA,
                  const double *inMatrixB, int nrowB, int ncolB, bool transposeB,
                  double alpha, double beta, double *outMatrix)
{
    size_t nrowC = transposeA ? ncolA : nrowA;
    size_t ncolC = transposeB ? nrowB : ncolB;
    size_t terms = transposeA ? nrowA : ncolA;

    for (size_t col = 0; col < ncolC; col++) {
        for (size_t row = 0; row < nrowC; row++) {
            const double *ptrA = transposeA ? &inMatrixA[nrowA * row] : &inMatrixA[row];
            const double *ptrB = transposeB ? &inMatrixB[row] : &inMatrixB[nrowB * col];
            double sum = 0.0;
            for (size_t k = 0; k < terms; k++) {
                sum += *ptrA * *ptrB;
                
                if (transposeA) {
                    ptrA++;
                    
                } else {
                    ptrA += nrowA;
                }
                
                if (transposeB) {
                    ptrB += nrowB;
                    
                } else {
                    ptrB++;
                }
            }
            
            outMatrix[nrowC * col + row] = alpha * sum + beta * outMatrix[nrowC * col + row];
        }
    }
}

void gemm_naive_f(const float *inMatrixA, int nrowA, int ncolA, bool transposeA,
                  const float *inMatrixB, int nrowB, int ncolB, bool transposeB,
                  float alpha, float beta, float *outMatrix)
{
    size_t nrowC = transposeA ? ncolA : nrowA;
    size_t ncolC = transposeB ? nrowB : ncolB;
    size_t terms = transposeA ? nrowA : ncolA;
    
    for (size_t col = 0; col < ncolC; col++) {
        for (size_t row = 0; row < nrowC; row++) {
            const float *ptrA = transposeA ? &inMatrixA[nrowA * row] : &inMatrixA[row];
            const float *ptrB = transposeB ? &inMatrixB[row] : &inMatrixB[nrowB * col];
            float sum = 0.0;
            for (size_t k = 0; k < terms; k++) {
                sum += *ptrA * *ptrB;
                
                if (transposeA) {
                    ptrA++;
                    
                } else {
                    ptrA += nrowA;
                }
                
                if (transposeB) {
                    ptrB += nrowB;
                    
                } else {
                    ptrB++;
                }
            }
            
            outMatrix[nrowC * col + row] = alpha * sum + beta * outMatrix[nrowC * col + row];
        }
    }
}

void transpose_f(float *x, int nrow, int ncol)
{
    float *copy = (float *)calloc(nrow * ncol, sizeof(float));
    if (copy == nullptr) {
#if RPACKAGE
        Rf_error("transpose_f: insufficient memory");
#else
        throw std::runtime_error("transpose_f: insufficient memory");
#endif
    }
    
    memcpy(copy, x, nrow * ncol * sizeof(float));
    transpose_f(copy, nrow, ncol, x);
//    float *q = copy;
//    for (size_t col = 0; col < ncol; col++) {
//        float *p = x + col;
//        for (size_t row = 0; row < nrow; row++) {
//            *p = *q++;
//            p += ncol;
//        }
//    }
    
    free(copy);
}

void transpose_f(const float *from, int nrow, int ncol, float *to)
{
    const float *q = from;
    for (int col = 0; col < ncol; col++) {
        float *p = to + col;
        for (int row = 0; row < nrow; row++) {
            *p = *q++;
            p += ncol;
        }
    }
}

void transpose_d(const double *from, int nrow, int ncol, double *to)
{
    const double *q = from;
    for (int col = 0; col < ncol; col++) {
        double *p = to + col;
        for (int row = 0; row < nrow; row++) {
            *p = *q++;
            p += ncol;
        }
    }
}

void transpose_d(double *x, int nrow, int ncol)
{
    double *copy = (double *)calloc(nrow * ncol, sizeof(double));
    if (copy == nullptr) {
#if RPACKAGE
        Rf_error("transpose_f: insufficient memory");
#else
        throw std::runtime_error("transpose_f: insufficient memory");
#endif
    }
    
    memcpy(copy, x, nrow * ncol * sizeof(double));
    transpose_d(copy, nrow, ncol, x);
//    double *q = copy;
//    for (size_t col = 0; col < ncol; col++) {
//        double *p = x + col;
//        for (size_t row = 0; row < nrow; row++) {
//            *p = *q++;
//            p += ncol;
//        }
//    }
    
    free(copy);
}


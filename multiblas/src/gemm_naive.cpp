//
//  gemm_naive.cpp
//  multiBLAS.XC
//
//  Created by michael on 7/23/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#include "gemm_naive.h"

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

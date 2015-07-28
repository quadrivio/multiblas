//
//  gemm_naive.h
//  multiBLAS.XC
//
//  Created by michael on 7/23/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#ifndef __multiBLAS_XC__gemm_naive__
#define __multiBLAS_XC__gemm_naive__

#include <stdio.h>

void gemm_naive_d(const double *inMatrixA, int nrowA, int ncolA, bool transposeA,
                 const double *inMatrixB, int nrowB, int ncolB, bool transposeB,
                 double alpha, double beta, double *outMatrix);

void gemm_naive_f(const float *inMatrixA, int nrowA, int ncolA, bool transposeA,
                 const float *inMatrixB, int nrowB, int ncolB, bool transposeB,
                 float alpha, float beta, float *outMatrix);

void transpose_f(const float *from, int nrow, int ncol, float *to);
void transpose_d(const double *from, int nrow, int ncol, double *to);

void transpose_f(float *x, int nrow, int ncol);
void transpose_d(double *x, int nrow, int ncol);

#endif /* defined(__multiBLAS_XC__gemm_naive__) */

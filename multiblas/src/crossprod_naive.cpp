//
//  crossprod_naive.cpp
//  OpenCL.ImpXC
//
//  Created by michael on 5/1/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#include "crossprod_naive.h"

void crossprod_naive(const double *inMatrix, double *outMatrix, size_t nrow, size_t ncol)
{
    for (size_t col = 0; col < ncol; col++) {
        for (size_t row = col; row < ncol; row++) {
            const double *colP1 = &inMatrix[nrow * col];
            const double *colP2 = &inMatrix[nrow * row];
            double sum = 0.0;
            for (size_t k = 0; k < nrow; k++) {
                sum += *colP1++ * *colP2++;
            }
            
            outMatrix[ncol * col + row] = sum;
            outMatrix[ncol * row + col] = sum;
        }
    }
}

void crossprod_naive(const float *inMatrix, float *outMatrix, size_t nrow, size_t ncol)
{
    for (size_t col = 0; col < ncol; col++) {
        for (size_t row = col; row < ncol; row++) {
            const float *colP1 = &inMatrix[nrow * col];
            const float *colP2 = &inMatrix[nrow * row];
            float sum = 0.0;
            for (size_t k = 0; k < nrow; k++) {
                sum += *colP1++ * *colP2++;
            }
            
            outMatrix[ncol * col + row] = sum;
            outMatrix[ncol * row + col] = sum;
        }
    }
}

void symmetrizeSquare_f(float *x, size_t dim)
{
    if (dim > 1) {
        size_t rowsToCopy = dim - 1;
        float *startFrom = x + dim * (dim - 1);
        float *startTo = x + dim - 1;
        while (rowsToCopy > 0) {
            float *from = startFrom;
            float *to = startTo;
            
            for (size_t k = rowsToCopy; k > 0; k--) {
                *to = *from++;
                to += dim;
            }
            
            rowsToCopy--;
            startFrom -= dim;
            startTo--;
        }
    }
}

void symmetrizeSquare_d(double *x, size_t dim)
{
    if (dim > 1) {
        size_t rowsToCopy = dim - 1;
        double *startFrom = x + dim * (dim - 1);
        double *startTo = x + dim - 1;
        while (rowsToCopy > 0) {
            double *from = startFrom;
            double *to = startTo;
            
            for (size_t k = rowsToCopy; k > 0; k--) {
                *to = *from++;
                to += dim;
            }
            
            rowsToCopy--;
            startFrom -= dim;
            startTo--;
        }
    }
}

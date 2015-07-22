//
//  crossprod_naive.h
//  OpenCL.ImpXC
//
//  Created by michael on 5/1/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#ifndef __OpenCL_ImpXC__crossprod_naive__
#define __OpenCL_ImpXC__crossprod_naive__

#include <cstddef>

void crossprod_naive(const double *inMatrix, double *outMatrix, size_t nrow, size_t ncol);
void crossprod_naive(const float *inMatrix, float *outMatrix, size_t nrow, size_t ncol);

void symmetrizeSquare_f(float *x, size_t dim);
void symmetrizeSquare_d(double *x, size_t dim);

#endif /* defined(__OpenCL_ImpXC__crossprod_naive__) */

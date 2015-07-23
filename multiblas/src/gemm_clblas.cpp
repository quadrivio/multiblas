//
//  gemm_clblas.cpp
//  multiBLAS.XC
//
//  Created by michael on 7/23/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#include "gemm_clblas.h"

ErrorStatus gemm_clblas_d(cl_device_id device, const double *inMatrixA, int nrowA, int ncolA, bool transposeA,
                          const double *inMatrixB, int nrowB, int ncolB, bool transposeB,
                          double alpha, double beta, double *outMatrix)
{
    ErrorStatus result = { 0, clblasNotImplemented };
    return result;
}

ErrorStatus gemm_clblas_f(cl_device_id device, const float *inMatrixA, int nrowA, int ncolA, bool transposeA,
                          const float *inMatrixB, int nrowB, int ncolB, bool transposeB,
                          float alpha, float beta, float *outMatrix)
{
    ErrorStatus result = { 0, clblasNotImplemented };
    return result;
}

ErrorStatus gemm_clblas(cl_device_id device, const void *inMatrixA, int nrowA, int ncolA, bool transposeA,
                        const void *inMatrixB, int nrowB, int ncolB, bool transposeB,
                        double alpha, double beta, void *outMatrix, bool use_float)
{
    ErrorStatus result = { 0, clblasNotImplemented };
    return result;
}

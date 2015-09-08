//
//  gemm_clblas.h
//  multiBLAS.XC
//
//  Created by michael on 7/23/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#ifndef __multiBLAS_XC__gemm_clblas__
#define __multiBLAS_XC__gemm_clblas__

#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include "clBLAS.h"
#include "utils_clblas.h"

ErrorStatus gemm_clblas_d(cl_device_id device, const double *inMatrixA, int nrowA, int ncolA, bool transposeA,
                 const double *inMatrixB, int nrowB, int ncolB, bool transposeB,
                 double alpha, double beta, double *outMatrix);

ErrorStatus gemm_clblas_f(cl_device_id device, const float *inMatrixA, int nrowA, int ncolA, bool transposeA,
                          const float *inMatrixB, int nrowB, int ncolB, bool transposeB,
                          float alpha, float beta, float *outMatrix);

ErrorStatus gemm_clblas(cl_device_id device, const void *inMatrixA, int nrowA, int ncolA, bool transposeA,
                          const void *inMatrixB, int nrowB, int ncolB, bool transposeB,
                          double alpha, double beta, void *outMatrix, bool use_float);

#endif /* defined(__multiBLAS_XC__gemm_clblas__) */

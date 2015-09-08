//
//  crossprod_clblas.h
//  template
//
//  Created by michael on 4/20/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#ifndef __template__crossprod_blas__
#define __template__crossprod_blas__

#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include "clBLAS.h"
#include "utils_clblas.h"

ErrorStatus crossprod_clblas_d(cl_device_id device, double *inMatrix, double *outMatrix, int nrow, int ncol);
ErrorStatus crossprod_clblas_f(cl_device_id device, float *inMatrix, float *outMatrix, int nrow, int ncol);
ErrorStatus crossprod_clblas(cl_device_id device, void *inMatrix, void *outMatrix, int nrow, int ncol, bool use_float);

#endif /* defined(__template__crossprod__) */

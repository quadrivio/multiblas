//
//  utils_clblas.h
//  multiBLAS.XC
//
//  Created by michael on 7/23/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#ifndef __multiBLAS_XC__utils_clblas__
#define __multiBLAS_XC__utils_clblas__

#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <string>

#include <clBLAS.h>

struct ErrorStatus {
    cl_int error;
    clblasStatus status;
};
typedef struct ErrorStatus ErrorStatus;

std::string clblasErrorToString(clblasStatus error);

#endif /* defined(__multiBLAS_XC__utils_clblas__) */

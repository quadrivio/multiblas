//
//  utils_clblas.cpp
//  multiBLAS.XC
//
//  Created by michael on 7/23/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#include "utils_clblas.h"
#include "opencl_info.h"

#include <iostream>
#include <sstream>

using namespace std;

std::string clblasErrorToString(clblasStatus error)
{
    switch(error) {
        case clblasSuccess: return "CL_SUCCESS";
        case clblasInvalidValue: return "CL_INVALID_VALUE";
        case clblasInvalidCommandQueue: return "CL_INVALID_COMMAND_QUEUE";
        case clblasInvalidContext: return "CL_INVALID_CONTEXT";
        case clblasInvalidMemObject: return " CL_INVALID_MEM_OBJECT";
        case clblasInvalidDevice: return "CL_INVALID_DEVICE";
        case clblasInvalidEventWaitList: return "CL_INVALID_EVENT_WAIT_LIST";
        case clblasOutOfResources: return "CL_OUT_OF_RESOURCES";
        case clblasOutOfHostMemory: return "CL_OUT_OF_HOST_MEMORY";
        case clblasInvalidOperation: return "CL_INVALID_OPERATION";
        case clblasCompilerNotAvailable: return "CL_COMPILER_NOT_AVAILABLE";
        case clblasBuildProgramFailure: return "CL_BUILD_PROGRAM_FAILURE";
            
            /* Extended error codes */
        case clblasNotImplemented: return "Functionality is not implemented";
        case clblasNotInitialized: return "clblas library is not initialized yet";
        case clblasInvalidMatA: return "Matrix A is not a valid memory object";
        case clblasInvalidMatB: return "Matrix B is not a valid memory object";
        case clblasInvalidMatC: return "Matrix C is not a valid memory object";
        case clblasInvalidVecX: return "Vector X is not a valid memory object";
        case clblasInvalidVecY: return "Vector Y is not a valid memory object";
        case clblasInvalidDim: return "An input dimension (M,N,K) is invalid";
        case clblasInvalidLeadDimA: return "Leading dimension A must not be less than the size of the first dimension";
        case clblasInvalidLeadDimB: return "Leading dimension B must not be less than the size of the second dimension";
        case clblasInvalidLeadDimC: return "Leading dimension C must not be less than the size of the third dimension";
        case clblasInvalidIncX: return "The increment for a vector X must not be 0";
        case clblasInvalidIncY: return "The increment for a vector Y must not be 0";
        case clblasInsufficientMemMatA: return "The memory object for Matrix A is too small";
        case clblasInsufficientMemMatB: return "The memory object for Matrix B is too small";
        case clblasInsufficientMemMatC: return "The memory object for Matrix C is too small";
        case clblasInsufficientMemVecX: return "The memory object for Vector X is too small";
        case clblasInsufficientMemVecY: return "The memory object for Vector Y is too small";
            
        default:
            std::stringstream result;
            result << "clblasStatus = " << (int)error;
            result << " ?(" << clErrorToString((cl_int)error) << ")";
            return result.str();
    }
}

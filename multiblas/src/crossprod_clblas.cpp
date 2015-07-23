//
//  crossprod_clblas.cpp
//  template
//
//  Created by michael on 4/20/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#include "crossprod_clblas.h"
#include "crossprod_naive.h"
#include "opencl_info.h"
#include "shim.h"

#include <iostream>
#include <sstream>
#include <vector>

//#if defined(__APPLE__)
//#include <OpenCL/opencl.h>
//#else
//#include <CL/opencl.h>
//#endif

const bool debug = false;

ErrorStatus crossprod_clblas_d(cl_device_id device, double *inMatrix, double *outMatrix, int nrow, int ncol)
{
    return crossprod_clblas(device, inMatrix, outMatrix, nrow, ncol, false);
}


ErrorStatus crossprod_clblas_f(cl_device_id device, float *inMatrix, float *outMatrix, int nrow, int ncol)
{
    return crossprod_clblas(device, inMatrix, outMatrix, nrow, ncol, true);
}

ErrorStatus crossprod_clblas(cl_device_id device, void *inMatrix, void *outMatrix, int nrow, int ncol, bool use_float)
{
    std::stringstream result;
    
    float *input_matrix_f = (float *)inMatrix;
    
    float *output_matrix_f = (float *)outMatrix;
    
    double *input_matrix_d = (double *)inMatrix;
    
    double *output_matrix_d = (double *)outMatrix;
    
    if (debug) {
        result << "crossprod_clblas( " << (use_float ? "FLOAT" : "DOUBLE") <<
        ", nrow = " << nrow << ", ncol = " << ncol << ")" << std::endl << std::endl;
    }
    
    cl_int err = CL_SUCCESS;

    clblasStatus status = clblasSetup();
    if (status != CL_SUCCESS) {
        if (debug) {
            result << "clblasSetup: " << clblasErrorToString(status) << std::endl;
        }
        
        err = CL_INVALID_OPERATION;
    }

    // get first platform
    cl_platform_id platform = NULL;
    if (err == CL_SUCCESS) {
        err = clGetPlatformIDs(1, &platform, NULL);
    }
    
    if (debug && err == CL_SUCCESS) {
        result << "Platform: " << getPlatformInfoString(platform, CL_PLATFORM_NAME) << std::endl;
        result << "Device: " << getDeviceInfoString(device, CL_DEVICE_NAME) << std::endl;
    }
    
    // context
    cl_context context = NULL;
    if (err == CL_SUCCESS) {
        if (debug) {
            result << "clCreateContext:" << std::endl;
        }
        
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    }
    
    // queue
    cl_command_queue queue = NULL;
    if (err == CL_SUCCESS) {
#ifdef CL_VERSION_2_0
        if (debug) {
            result << "clCreateCommandQueueWithProperties:" << std::endl;
        }
        
        queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
        
#else
        if (debug) {
            result << "clCreateCommandQueue:" << std::endl;
        }
        
        queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    }
    
    // buffers
    cl_mem cl_input_matrix = NULL;
    if (err == CL_SUCCESS) {
        if (debug) {
            result << "clCreateBuffer cl_input_matrix:" << std::endl;
        }
        
        if (use_float) {
            cl_input_matrix = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             nrow * ncol * sizeof(float), input_matrix_f, &err);
            
        } else {
            cl_input_matrix = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             nrow * ncol * sizeof(double), input_matrix_d, &err);
        }
    }
    
    cl_mem cl_output_matrix = NULL;
    if (err == CL_SUCCESS) {
        if (debug) {
            result << "clCreateBuffer cl_output_vector:" << std::endl;
        }

        if (use_float) {
            cl_output_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                              ncol * ncol * sizeof(float), output_matrix_f, &err);
            
        } else {
            cl_output_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                              ncol * ncol * sizeof(double), output_matrix_d, &err);
        }

    }
    
    // ++++++++++++
    const clblasOrder order = clblasColumnMajor;
    const clblasTranspose transA = clblasTrans;
//    const clblasTranspose transB = clblasNoTrans;
//    
//    const size_t rowsA = ncol;
//    const size_t colsB = ncol;
//    const size_t colsArowsB = nrow;
    
//    const size_t lda = nrow;
//    const size_t ldb = nrow;
//    const size_t ldc = ncol;
    
    const size_t lda = nrow;
    const size_t ldc = ncol;
    
    const cl_float alpha = 1.0;
    
    clblasUplo uplo = clblasUpper;
    
    cl_event event = NULL;
    
    if (err == CL_SUCCESS) {
        if (use_float) {
            if (debug) {
                result << "clblasSsyrk:" << std::endl;
            }
            
//            status = clblasSgemm(order, transA, transB, rowsA, colsB, colsArowsB,
//                              alpha, cl_input_matrix, 0, lda,
//                              cl_input_matrix, 0, ldb, 0.0,
//                              cl_output_matrix, 0, ldc,
//                              1, &queue, 0, NULL, &event);
            
            status = clblasSsyrk(order, uplo, transA, ncol, nrow, alpha, cl_input_matrix, 0, lda, 0.0,
                                 cl_output_matrix, 0, ldc, 1, &queue, 0, NULL, &event);
            
            if (status != CL_SUCCESS && debug) {
                result << "clblasSgemm error:" << clblasErrorToString(status) << std::endl;
            }

        } else {
            if (debug) {
                result << "clblasDsyrk:" << std::endl;
            }
            
//            status = clblasDgemm(order, transA, transB, rowsA, colsB, colsArowsB,
//                                 alpha, cl_input_matrix, 0, ncol,
//                                 cl_input_matrix, 0, nrow, 0.0,
//                                 cl_output_matrix, 0, ncol,
//                                 1, &queue, 0, NULL, &event);
            
            status = clblasDsyrk(order, uplo, transA, ncol, nrow, alpha, cl_input_matrix, 0, lda, 0.0,
                                 cl_output_matrix, 0, ldc, 1, &queue, 0, NULL, &event);
            
            if (status != CL_SUCCESS) {
                if (debug) {
                    result << "clblasDgemm error:" << clblasErrorToString(status) << std::endl;
                }
                
                err = status;
            }
        }
    }
    
    if (err == CL_SUCCESS) {
        /* Wait for calculations to be finished. */
        if (debug) {
            result << "clWaitForEvents:" << std::endl;
        }
        err = clWaitForEvents(1, &event);
    }
    
    // retrieve result
    if (err == CL_SUCCESS) {
        if (debug) {
            result << "Retrieve result:" << std::endl;
        }
        
        if (use_float) {
            clEnqueueReadBuffer(queue, cl_output_matrix, CL_TRUE, 0, ncol * ncol * sizeof(float), output_matrix_f, 0, NULL, NULL);
            symmetrizeSquare_f(output_matrix_f, ncol);
            
        } else {
            clEnqueueReadBuffer(queue, cl_output_matrix, CL_TRUE, 0, ncol * ncol * sizeof(double), output_matrix_d, 0, NULL, NULL);
            symmetrizeSquare_d(output_matrix_d, ncol);
        }
    }
    
    std::string err_str = clErrorToString(err);
    result << std::endl << err_str << std::endl;
    
    // cleanup
    clReleaseMemObject(cl_output_matrix);
    cl_output_matrix = NULL;
    
    clReleaseMemObject(cl_input_matrix);
    cl_input_matrix = NULL;
    
    clReleaseCommandQueue(queue);
    queue = NULL;
    
    clReleaseContext(context);
    context = NULL;
    
    if (debug) {
        CERR << result.str();
    }
    
    ErrorStatus errorStatus = { err, status };
    
//    return status != CL_SUCCESS ? clblasErrorToString(status) : clErrorToString(err);
    return errorStatus;
}


//void symmetrizeSquare_f(float *x, size_t dim)
//{
//    if (dim > 1) {
//        size_t rowsToCopy = dim - 1;
//        float *startFrom = x + dim;
//        float *to = x + 1;
//        while (rowsToCopy > 0) {
//            float *from = startFrom;
//            
//            for (size_t k = rowsToCopy; k > 0; k--) {
//                *to++ = *from;
//                from += dim;
//            }
//            
//            rowsToCopy--;
//            to += dim - rowsToCopy;
//            startFrom += 1 + dim;
//        }
//    }
//}
//
//void symmetrizeSquare_d(double *x, size_t dim)
//{
//    if (dim > 1) {
//        size_t rowsToCopy = dim - 1;
//        double *startFrom = x + dim;
//        double *to = x + 1;
//        while (rowsToCopy > 0) {
//            double *from = startFrom;
//            
//            for (size_t k = rowsToCopy; k > 0; k--) {
//                *to++ = *from;
//                from += dim;
//            }
//            
//            rowsToCopy--;
//            to += dim - rowsToCopy;
//            startFrom += 1 + dim;
//        }
//    }
//}

// =================================================================================================


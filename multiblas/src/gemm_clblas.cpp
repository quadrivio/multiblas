//
//  gemm_clblas.cpp
//  multiBLAS.XC
//
//  Created by michael on 7/23/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#include "gemm_clblas.h"

#include "opencl_info.h"
#include "shim.h"

#include <iostream>
#include <sstream>
#include <vector>

const bool debug = false;

ErrorStatus gemm_clblas_d(cl_device_id device, const double *inMatrixA, int nrowA, int ncolA, bool transposeA,
                          const double *inMatrixB, int nrowB, int ncolB, bool transposeB,
                          double alpha, double beta, double *outMatrix)
{
    return gemm_clblas(device, inMatrixA, nrowA, ncolA, transposeA, inMatrixB, nrowB, ncolB, transposeB, alpha, beta, outMatrix, false);
}

ErrorStatus gemm_clblas_f(cl_device_id device, const float *inMatrixA, int nrowA, int ncolA, bool transposeA,
                          const float *inMatrixB, int nrowB, int ncolB, bool transposeB,
                          float alpha, float beta, float *outMatrix)
{
    return gemm_clblas(device, inMatrixA, nrowA, ncolA, transposeA, inMatrixB, nrowB, ncolB, transposeB, alpha, beta, outMatrix, true);
}

ErrorStatus gemm_clblas(cl_device_id device, const void *inMatrixA, int nrowA, int ncolA, bool transposeA,
                        const void *inMatrixB, int nrowB, int ncolB, bool transposeB,
                        double alpha, double beta, void *outMatrix, bool use_float)
{
    std::stringstream result;
    
    float *input_matrixA_f = (float *)inMatrixA;
    float *input_matrixB_f = (float *)inMatrixB;
    
    float *output_matrix_f = (float *)outMatrix;
    
    double *input_matrixA_d = (double *)inMatrixA;
    double *input_matrixB_d = (double *)inMatrixB;
    
    double *output_matrix_d = (double *)outMatrix;
    
    if (debug) {
        result << "gemm_clblas( " << (use_float ? "FLOAT" : "DOUBLE") <<
        ")" << std::endl << std::endl;
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
    cl_mem cl_input_matrixA = NULL;
    if (err == CL_SUCCESS) {
        if (debug) {
            result << "clCreateBuffer cl_input_matrixA:" << std::endl;
        }
        
        if (use_float) {
            cl_input_matrixA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              nrowA * ncolA * sizeof(float), input_matrixA_f, &err);
            
        } else {
            cl_input_matrixA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              nrowA * ncolA * sizeof(double), input_matrixA_d, &err);
        }
    }
    
    cl_mem cl_input_matrixB = NULL;
    if (err == CL_SUCCESS) {
        if (debug) {
            result << "clCreateBuffer cl_input_matrixB:" << std::endl;
        }
        
        if (use_float) {
            cl_input_matrixB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              nrowB * ncolB * sizeof(float), input_matrixB_f, &err);
            
        } else {
            cl_input_matrixB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              nrowB * ncolB * sizeof(double), input_matrixB_d, &err);
        }
    }
    
    int nrowC = transposeA ? ncolA : nrowA;
    int ncolC = transposeB ? nrowB : ncolB;
    cl_mem cl_output_matrix = NULL;
    if (err == CL_SUCCESS) {
        if (debug) {
            result << "clCreateBuffer cl_output_vector:" << std::endl;
        }
        
        if (use_float) {
            cl_output_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                              nrowC * ncolC * sizeof(float), output_matrix_f, &err);
            
        } else {
            cl_output_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                              nrowC * ncolC * sizeof(double), output_matrix_d, &err);
        }
        
    }
    
    // ++++++++++++
    const int lda = nrowA;  // first dimension of A (rows), before any transpose
    const int ldb = nrowB;  // first dimension of B (rows), before any transpose
    const int ldc = nrowC;      // first dimension of C (rows)
    
    const int M = transposeA ? ncolA : nrowA;    // rows in A (after transpose, if any) and C
    const int N = transposeB ? nrowB : ncolB;    // cols in B (after transpose, if any) and C
    const int K = transposeA ? nrowA : ncolA;    // cols in A and rows in B (after transposes, if any)
    
    const clblasOrder order = clblasColumnMajor;
    const clblasTranspose transA = transposeA ? clblasTrans : clblasNoTrans;
    const clblasTranspose transB = transposeB ? clblasTrans : clblasNoTrans;
    
    cl_event event = NULL;
    
    if (err == CL_SUCCESS) {
        if (use_float) {
            if (debug) {
                result << "clblasSgemm:" << std::endl;
            }
            
            status = clblasSgemm(order, transA, transB, M, N, K,
                              alpha, cl_input_matrixA, 0, lda,
                              cl_input_matrixB, 0, ldb, beta,
                              cl_output_matrix, 0, ldc,
                              1, &queue, 0, NULL, &event);
            
            if (status != CL_SUCCESS && debug) {
                result << "clblasSgemm error:" << clblasErrorToString(status) << std::endl;
            }
            
        } else {
            if (debug) {
                result << "clblasDgemm:" << std::endl;
            }
            
            status = clblasDgemm(order, transA, transB, M, N, K,
                                 alpha, cl_input_matrixA, 0, lda,
                                 cl_input_matrixB, 0, ldb, beta,
                                 cl_output_matrix, 0, ldc,
                                 1, &queue, 0, NULL, &event);
            
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
            clEnqueueReadBuffer(queue, cl_output_matrix, CL_TRUE, 0, nrowC * ncolC * sizeof(float), output_matrix_f, 0, NULL, NULL);
            
        } else {
            clEnqueueReadBuffer(queue, cl_output_matrix, CL_TRUE, 0, nrowC * ncolC * sizeof(double), output_matrix_d, 0, NULL, NULL);
        }
    }
    
    std::string err_str = clErrorToString(err);
    result << std::endl << err_str << std::endl;
    
    // cleanup
    clReleaseMemObject(cl_output_matrix);
    cl_output_matrix = NULL;
    
    clReleaseMemObject(cl_input_matrixA);
    cl_input_matrixA = NULL;
    
    clReleaseMemObject(cl_input_matrixB);
    cl_input_matrixB = NULL;
    
    clReleaseCommandQueue(queue);
    queue = NULL;
    
    clReleaseContext(context);
    context = NULL;
    
    if (debug) {
        CERR << result.str();
    }
    
    ErrorStatus errorStatus = { err, status };
    
    return errorStatus;
}

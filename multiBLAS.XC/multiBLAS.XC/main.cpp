//
//  main.cpp
//  multiBLAS.XC
//
//  Created by michael on 6/25/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//  Copyright (c) 2017 Michael Budiansky. All rights reserved.
//

#include "opencl_info.h"
#include "gemm_naive.h"
#include "gemm_blas.h"
#include "gemm_clblas.h"
#include "gemm_opencl.h"
#include "nullptr.h"
#include "utils.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>
#include <vector>
#include <cstddef>

using namespace std;

std::string gInstPath("");

void printMatrix_f(float *x, size_t ncol, size_t nrow)
{
    for (size_t row = 0; row < nrow; row++) {
        for (size_t col = 0; col < ncol; col++) {
            cout << x[col * nrow + row] << "\t";
        }
        cout << endl;
    }
    cout << endl;
}

int main(int argc, const char * argv[]) {
    // find directory containing opencl files
    gInstPath = argv[0];
    gInstPath = gInstPath.substr(0, gInstPath.find_last_of("/\\"));
    gInstPath += "/inst";
    
    // prepare matrices
    size_t nrowA = 2;
    size_t ncolA = 3;
    float *a = (float *)calloc(sizeof(float), ncolA * nrowA);
    for (size_t k = 0; k < ncolA * nrowA; k++) a[k] = k + 1;
    
    cout << "a:" << endl;
    printMatrix_f(a, ncolA, nrowA);
    
    size_t nrowB = 3;
    size_t ncolB = 5;
    float *b = (float *)calloc(sizeof(float), ncolB * nrowB);
    for (size_t k = 0; k < ncolB * nrowB; k++) b[k] = k + 1;
    
    cout << "b:" << endl;
    printMatrix_f(b, ncolB, nrowB);
    
    size_t nrowC = nrowA;
    size_t ncolC = ncolB;
    float *c = (float *)calloc(sizeof(float), ncolC * nrowC);
    
    // Naive matrix multiplication
    
    cout << "gemm_naive_f result a * b =" << endl;
    memset(c, 0, ncolC * nrowC);
    
    gemm_naive_f(a, (int)nrowA, (int)ncolA, false, b, (int)nrowB, (int)ncolB, false, 1.0, 0.0, c);
    
    printMatrix_f(c, ncolC, nrowC);

    // BLAS matrix multiplication
    
    cout << "gemm_blas_f result a * b =" << endl;
    memset(c, 0, ncolC * nrowC);
    
    gemm_blas_f(a, (int)nrowA, (int)ncolA, false, b, (int)nrowB, (int)ncolB, false, 1.0, 0.0, c);
    
    printMatrix_f(c, ncolC, nrowC);
    
    // clBLAS matrix multiplication
 
    cout << "gemm_clblas_f result a * b =" << endl;
    memset(c, 0, ncolC * nrowC);

    cl_platform_id platform = NULL;
    clGetPlatformIDs(1, &platform, NULL);
    
    cl_device_id gpu_device = nullptr;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &gpu_device, NULL);
    
    gemm_clblas_f(gpu_device, a, (int)nrowA, (int)ncolA, false, b, (int)nrowB, (int)ncolB, false, 1.0, 0.0, c);
    
    printMatrix_f(c, 5, 2);

    // OpenCL matrix multiplication

    cout << "opencl_calc_gemm result a * b =" << endl;
    memset(c, 0, ncolC * nrowC);

    cl_context context = clCreateContext(NULL, 1, &gpu_device, NULL, NULL, NULL);

#ifdef CL_VERSION_2_0
    const cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, gpu_device, properties, NULL);
    
#else
    cl_command_queue queue = clCreateCommandQueue(context, gpu_device, CL_QUEUE_PROFILING_ENABLE, NULL);
#endif

    string path = gInstPath + "/gemm_f.cl";
    string source;
    fileToString(path, source);
    const char *src = source.c_str();
    
    cl_program program = clCreateProgramWithSource(context, 1, &src, NULL, NULL);
    clBuildProgram(program, 0, NULL, "", NULL, NULL);
    
    cl_kernel kernel_f = clCreateKernel(program, "gemm_f_naive", NULL);
    cl_kernel kernel_d = clCreateKernel(program, "gemm_d_naive", NULL);
    clReleaseProgram(program);

    bool verbose = false;

#if __cplusplus < 201103L
    std::vector<size_t> work_item_sizes;
    work_item_sizes.push_back(1);
    work_item_sizes.push_back(1);
    work_item_sizes.push_back(1);
#else
    std::vector<size_t> work_item_sizes = { 1, 1, 1 };
#endif

    opencl_calc_gemm(context, kernel_f, kernel_d, true, queue,
                     a, (int)nrowA, (int)ncolA, false,
                     b, (int)nrowB, (int)ncolB, false,
                     1.0, 0.0, c, work_item_sizes, 1, 1, 1, 1, verbose);
    
    printMatrix_f(c, 5, 2);

    // clean up
    
    free(a);
    a = nullptr;
    
    free(b);
    b = nullptr;
    
    free(c);
    c = nullptr;
    
}

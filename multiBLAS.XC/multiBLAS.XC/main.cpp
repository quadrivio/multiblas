//
//  main.cpp
//  multiBLAS.XC
//
//  Created by michael on 6/25/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#include "crossprod_opencl.h"
#include "opencl_info.h"
#include "gemm_naive.h"
#include "gemm_blas.h"
#include "gemm_opencl.h"
#include "utils.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace std::chrono;

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
    float *a = (float *)calloc(sizeof(float), 3 * 2);
    for (size_t k = 0; k < 3 * 4; k++) a[k] = k + 1;
    
    cout << "a:" << endl;
    printMatrix_f(a, 3, 2);
    
    float *b = (float *)calloc(sizeof(float), 5 * 3);
    for (size_t k = 0; k < 5 * 3; k++) b[k] = k + 7;
    
    cout << "b:" << endl;
    printMatrix_f(b, 5, 3);
    
    float *c = (float *)calloc(sizeof(float), 5 * 2);
//    gemm_naive_f(a, 2, 3, false, b, 3, 5, false, 1.0, 0.0, c);
//    gemm_blas_f(a, 2, 3, false, b, 3, 5, false, 1.0, 0.0, c);

    {
        gInstPath = argv[0];
        gInstPath = gInstPath.substr(0, gInstPath.find_last_of("/\\"));
        gInstPath += "/inst";
        
        cl_context context = nullptr;
        cl_kernel kernel_f = nullptr;
        cl_kernel kernel_d = nullptr;
        cl_command_queue queue = nullptr;
        size_t nrowA = 2;
        size_t ncolA = 3;
        size_t nrowB = 3;
        size_t ncolB = 5;
        bool verbose = false;
        
        std::vector<size_t> work_item_sizes;
        work_item_sizes.push_back(1);
        work_item_sizes.push_back(1);
        work_item_sizes.push_back(1);
        
        cl_platform_id platform = NULL;
        clGetPlatformIDs(1, &platform, NULL);
        
        cl_device_id gpu_device = nullptr;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &gpu_device, NULL);
        
        context = clCreateContext(NULL, 1, &gpu_device, NULL, NULL, NULL);
        
#ifdef CL_VERSION_2_0
        const cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        queue = clCreateCommandQueueWithProperties(context, gpu_device, properties, NULL);
        
#else
        queue = clCreateCommandQueue(context, gpu_device, CL_QUEUE_PROFILING_ENABLE, NULL);
#endif
        
        string path = gInstPath + "/gemm_f.cl";
        string source;
        fileToString(path, source);
        const char *src = source.c_str();
        
        cl_program program = clCreateProgramWithSource(context, 1, &src, NULL, NULL);
        clBuildProgram(program, 0, NULL, "", NULL, NULL);
        
        kernel_f = clCreateKernel(program, "gemm_f_naive", NULL);
        clReleaseProgram(program);
        
        cl_int err = CL_SUCCESS;
        for (int k = 0; k < 25; k++) {
            err = opencl_calc_gemm(context, kernel_f, kernel_d, true, queue,
                                   a, (int)nrowA, (int)ncolA, false,
                                   b, (int)nrowB, (int)ncolB, false,
                                   1.0, 0.0, c, work_item_sizes, 1, 1, 1, verbose);
        }
        
        cout << clErrorToString(err) << endl << endl;
    }
    
    cout << "result:" << endl;
    printMatrix_f(c, 5, 2);
}

#if 0

#if 0
std::string crossprod_cl_ckq_d(cl_context context, cl_kernel kernel, cl_command_queue queue,
                               void *inMatrix, void *outMatrix, int nrow, int ncol,
                               int row_tile_size, int col_tile_size, bool use_rect = false);
std::string crossprod_cl_ckq_f(cl_context context, cl_kernel kernel, cl_command_queue queue,
                               void *inMatrix, void *outMatrix, int nrow, int ncol,
                               int row_tile_size, int col_tile_size, bool use_rect = false);
std::string crossprod_cl_ckq(cl_context context, cl_kernel kernel, cl_command_queue queue,
                             void *inMatrix, void *outMatrix, int nrow, int ncol, bool use_float,
                             int row_tile_size, int col_tile_size, bool use_rect);

cl_int xopencl_calc_x(cl_context context, cl_kernel kernel_f, cl_kernel kernel_d, bool is_float,
                      cl_command_queue queue, void *inMatrix, void *outMatrix, size_t nrow, size_t ncol,
                      const std::vector<size_t>& work_item_sizes, int row_multiple, int col_multiple,
                      int row_tile_size, int col_tile_size, bool verbose);
#endif

std::vector<size_t> work_item_sizes;

int main(int argc, const char * argv[]) {
    gInstPath = argv[0];
    gInstPath = gInstPath.substr(0, gInstPath.find_last_of("/\\"));
    gInstPath += "/inst";
    
    cl_context context = nullptr;
    cl_kernel kernel_f = nullptr;
    cl_kernel kernel_d = nullptr;
    bool is_float = true;
    cl_command_queue queue = nullptr;
    float *inMatrix = nullptr;
    float *outMatrix = nullptr;
    size_t nrow = 16;
    size_t ncol = 32;
    int row_multiple = 16;
    int col_multiple = 1;
    int row_tile_size = 4;
    int col_tile_size = 4;
    bool verbose = true;
    
    work_item_sizes.push_back(1);
    work_item_sizes.push_back(1);
    work_item_sizes.push_back(1);

    cl_platform_id platform = NULL;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id gpu_device = nullptr;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &gpu_device, NULL);

    context = clCreateContext(NULL, 1, &gpu_device, NULL, NULL, NULL);

#ifdef CL_VERSION_2_0
    const cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    queue = clCreateCommandQueueWithProperties(context, gpu_device, properties, NULL);
    
#else
    queue = clCreateCommandQueue(context, gpu_device, CL_QUEUE_PROFILING_ENABLE, NULL);
#endif

    string path = gInstPath + "/crossprod_f.cl";
    string source;
    fileToString(path, source);
    const char *src = source.c_str();
    
    cl_program program = clCreateProgramWithSource(context, 1, &src, NULL, NULL);
    clBuildProgram(program, 0, NULL, "-DROW_TILE_SIZE=4 -DCOL_TILE_SIZE=4", NULL, NULL);
        
    kernel_f = clCreateKernel(program, "crossprod_f_dot4tile", NULL);
    clReleaseProgram(program);

    inMatrix = (float *)calloc(nrow * ncol, sizeof(float));
    for (size_t k = 0; k < nrow * ncol; k++) inMatrix[k] = 0.0;
    for (size_t col = 0; col < 4; col++) {
        for (size_t row = 0; row < 4; row++) {
            inMatrix[col * nrow + row] = col * 4 + row;
        }
    }

    outMatrix = (float *)calloc(ncol * ncol, sizeof(float));
    for (size_t k = 0; k < ncol * ncol; k++) outMatrix[k] = NAN;
    
    if (true) {
        cout << "inMatrix:" << endl;
        for (size_t row = 0; row < nrow; row++) {
            for (size_t col = 0; col < ncol; col++) {
                cout << inMatrix[col * nrow + row] << "\t";
            }
            cout << endl;
        }
        cout << endl;
    }

    {
        steady_clock::time_point start_time = steady_clock::now();
        
        cl_int err = opencl_calc_x(context, kernel_f, kernel_d, is_float, queue, inMatrix, outMatrix,
                                   (int)nrow, (int)ncol, work_item_sizes,
                                   row_multiple, col_multiple, row_tile_size, col_tile_size, verbose);
        
        steady_clock::time_point end_time = steady_clock::now();
        duration<double> time_span = duration_cast<duration<double> >(end_time - start_time);
        
        double nsec = std::chrono::duration_cast<std::chrono::nanoseconds> (end_time - start_time).count();
        double gflops = (2.0 * nrow * (ncol + 0.5 * (ncol - 1.0) * ncol)) / nsec;
        cout << "Elapsed: " << time_span.count() << " sec " << gflops << " GFLOPS" << std::endl;
        
        cout << clErrorToString(err) << endl << endl;
    }

    if (true) {
        cout << "outMatrix:" << endl;
        for (size_t row = 0; row < ncol; row++) {
            for (size_t col = 0; col < ncol; col++) {
                cout << outMatrix[col * ncol + row] << "\t";
            }
            cout << endl;
        }
        cout << endl;
    }

    // ---------------------------------------------------------------------------------------------
    
    /*
    bool use_rect = false;
    
    {
        steady_clock::time_point start_time = steady_clock::now();
        
        string result = crossprod_cl_ckq(context, kernel_f, queue,
                                         inMatrix, outMatrix, (int)nrow, (int)ncol, is_float,
                                         row_tile_size, col_tile_size, use_rect);
        
        steady_clock::time_point end_time = steady_clock::now();
        duration<double> time_span = duration_cast<duration<double> >(end_time - start_time);
        
        double nsec = std::chrono::duration_cast<std::chrono::nanoseconds> (end_time - start_time).count();
        double gflops = (2.0 * nrow * (ncol + 0.5 * (ncol - 1.0) * ncol)) / nsec;
        cout << "Elapsed: " << time_span.count() << " sec " << gflops << " GFLOPS" << std::endl;
        
        cout << result << endl;
    }
    */

    return 0;
}

#if 0
std::string crossprod_cl_ckq_f(cl_context context, cl_kernel kernel, cl_command_queue queue,
                               void *inMatrix, void *outMatrix, int nrow, int ncol,
                               int row_tile_size, int col_tile_size, bool use_rect)
{
    return crossprod_cl_ckq(context, kernel, queue, inMatrix, outMatrix, nrow, ncol, true, row_tile_size, col_tile_size, use_rect);
}

std::string crossprod_cl_ckq_d(cl_context context, cl_kernel kernel, cl_command_queue queue,
                               void *inMatrix, void *outMatrix, int nrow, int ncol,
                               int row_tile_size, int col_tile_size, bool use_rect)
{
    return crossprod_cl_ckq(context, kernel, queue, inMatrix, outMatrix, nrow, ncol, false, row_tile_size, col_tile_size, use_rect);
}

std::string crossprod_cl_ckq(cl_context context, cl_kernel kernel, cl_command_queue queue,
                             void *inMatrix, void *outMatrix, int nrow, int ncol, bool use_float,
                             int row_tile_size, int col_tile_size, bool use_rect)
{
    bool debug = false;
    // ======================
    
    std::stringstream result;
    
    float *input_matrix_f = (float *)inMatrix;
    
    float *output_matrix_f = (float *)outMatrix;
    
    double *input_matrix_d = (double *)inMatrix;
    
    double *output_matrix_d = (double *)outMatrix;
    
    if (debug) {
        result << "crossprod_cl_ckq( " << (use_float ? "FLOAT" : "DOUBLE") << ")" << std::endl << std::endl;
    }
    
    cl_int err = CL_SUCCESS;
    
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
        
        cl_output_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          use_float ? ncol * ncol * sizeof(float) :
                                          ncol * ncol * sizeof(double), NULL, &err);
    }
    
    // initiate calculation
    cl_event event = nullptr;
    if (err == CL_SUCCESS) {
        if (debug) {
            result << "Initiate calculation:" << std::endl;
        }
        cl_int cl_nrow = nrow;
        cl_int cl_ncol = ncol;
        clSetKernelArg(kernel, 0, sizeof(cl_nrow), &cl_nrow);
        clSetKernelArg(kernel, 1, sizeof(cl_ncol), &cl_ncol);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_input_matrix);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_output_matrix);
        
        //        std::vector<size_t> work_item_sizes = ckernel->work_item_sizes();
        
        cl_uint work_dim = 3;
        size_t work_rows = (size_t)ncol / row_tile_size;
        if (use_rect) {
            work_rows = (work_rows / 2) + 1;
            work_rows = work_item_sizes[1] * ((work_rows + work_item_sizes[1] - 1) / work_item_sizes[1]);
        }
        
        size_t global_work_sizes[] = { (size_t)ncol / col_tile_size, work_rows, 1 };
        
        std::cout << "work_item_sizes = (" << work_item_sizes[0] << ", " <<
            work_item_sizes[1] << ", " << work_item_sizes[2] << ")" << std::endl;
        
        std::cout << "global_work_sizes = (" << global_work_sizes[0] << ", " <<
        global_work_sizes[1] << ", " << global_work_sizes[2] << ")" << std::endl;
        
        err = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_sizes,
                                     work_item_sizes.size() == 0 ? NULL :  &work_item_sizes[0], 0, NULL, &event);
        
    } else {
        result << "NO Initiate calculation" << std::endl;
    }
    
    // retrieve result
    cl_event event2 = nullptr;
    if (err == CL_SUCCESS) {
        if (debug) {
            result << "Retrieve result:" << std::endl;
        }
        
        if (use_float) {
            err = clEnqueueReadBuffer(queue, cl_output_matrix, CL_TRUE, 0, ncol * ncol * sizeof(float), output_matrix_f, 1, &event, &event2);
            
        } else {
            err = clEnqueueReadBuffer(queue, cl_output_matrix, CL_TRUE, 0, ncol * ncol * sizeof(double), output_matrix_d, 1, &event, &event2);
        }
        
        if (event != nullptr) {
            
            //            if (false) {
            //                Profile enqueue_profile = getProfileTimes(event);
            //                CERR << "clEnqueueNDRangeKernel: " << enqueue_profile.queued << " Q msec; " <<
            //                enqueue_profile.pending << " P msec; " << enqueue_profile.exec << " E msec" << std::endl;
            //            }
            
            clReleaseEvent(event);
            
        } else {
            result << "event is null " << std::endl;
        }
        
        if (event2 != nullptr) {
            clWaitForEvents(1, &event2);
            
            //            if (false) {
            //                Profile enqueue_read = getProfileTimes(event2);
            //                CERR << "clEnqueueReadBuffer: " << enqueue_read.queued << " Q msec; " <<
            //                enqueue_read.pending << " P msec; " << enqueue_read.exec << " E msec" << std::endl;
            //            }
            
            clReleaseEvent(event2);
            
        } else {
            result << "event2 is null " << std::endl;
        }
        
    } else {
        result << "NO Retrieve result" << std::endl;
    }
    
    std::string err_str = clErrorToString(err);
    result << std::endl << err_str << std::endl;
    
    // cleanup
    if (cl_output_matrix != nullptr) {
        clReleaseMemObject(cl_output_matrix);
        cl_output_matrix = NULL;
    }
    
    if (cl_input_matrix != nullptr) {
        clReleaseMemObject(cl_input_matrix);
        cl_input_matrix = NULL;
    }
    
    if (debug) {
        CERR << result.str();
    }
    
    return clErrorToString(err);
}

std::string crossprod_cl_ckq0(cl_context context, cl_kernel kernel, cl_command_queue queue,
                             void *inMatrix, void *outMatrix, int nrow, int ncol, bool use_float,
                             int row_tile_size, int col_tile_size, bool use_rect)
{
    bool debug = false;
    // ======================
    
    std::stringstream result;
    
    float *input_matrix_f = (float *)inMatrix;
    
    float *output_matrix_f = (float *)outMatrix;
    
    double *input_matrix_d = (double *)inMatrix;
    
    double *output_matrix_d = (double *)outMatrix;
    
    if (debug) {
        result << "crossprod_cl_ckq( " << (use_float ? "FLOAT" : "DOUBLE") << ")" << std::endl << std::endl;
    }
    
    cl_int err = CL_SUCCESS;
    
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
        
        cl_output_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          use_float ? ncol * ncol * sizeof(float) :
                                          ncol * ncol * sizeof(double), NULL, &err);
    }
    
    // initiate calculation
    cl_event event = nullptr;
    if (err == CL_SUCCESS) {
        if (debug) {
            result << "Initiate calculation:" << std::endl;
        }
#if 0
        cl_int cl_nrow = nrow;
        cl_int cl_ncol = ncol;
        clSetKernelArg(kernel, 0, sizeof(cl_nrow), &cl_nrow);
        clSetKernelArg(kernel, 1, sizeof(cl_ncol), &cl_ncol);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_input_matrix);
        if (use_float) {
            clSetKernelArg(kernel, 3, 1024 * sizeof(float), NULL);
            
        } else {
            clSetKernelArg(kernel, 3, 512 * sizeof(double), NULL);
        }
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_output_matrix);
        
        cl_uint work_dim = 3;
        size_t ncol_round_up = 4 * ((ncol + 3) / 4);
        size_t global_work_size[] = { (size_t)ncol_round_up, (size_t)ncol_round_up, 1 };
        
        std::vector<size_t> work_item_sizes = ckernel->work_item_sizes();
        
        std::cout << global_work_size[0] << "\t" << work_item_sizes[0] << std::endl;
#else
        cl_int cl_nrow = nrow;
        cl_int cl_ncol = ncol;
        cl_int cl_row_tile_size = row_tile_size;
        cl_int cl_col_tile_size = col_tile_size;
        clSetKernelArg(kernel, 0, sizeof(cl_nrow), &cl_nrow);
        clSetKernelArg(kernel, 1, sizeof(cl_ncol), &cl_ncol);
        clSetKernelArg(kernel, 2, sizeof(cl_row_tile_size), &cl_row_tile_size);
        clSetKernelArg(kernel, 3, sizeof(cl_col_tile_size), &cl_col_tile_size);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_input_matrix);
        if (false) {
            //            clSetKernelArg(kernel, 5, ckernel->local_mem_per_workgroup(), NULL);
        } else {
            clSetKernelArg(kernel, 5, sizeof(cl_mem), NULL);
        }
        clSetKernelArg(kernel, 6, sizeof(cl_mem), &cl_output_matrix);
        
        //        std::vector<size_t> work_item_sizes = ckernel->work_item_sizes();
        
        cl_uint work_dim = 3;
        size_t work_rows = (size_t)ncol / row_tile_size;
        if (use_rect) {
            work_rows = (work_rows / 2) + 1;
            work_rows = work_item_sizes[1] * ((work_rows + work_item_sizes[1] - 1) / work_item_sizes[1]);
        }
        
        size_t global_work_sizes[] = { (size_t)ncol / col_tile_size, work_rows, 1 };
        
        //        std::cout << "work_item_sizes = (" << work_item_sizes[0] << ", " <<
        //            work_item_sizes[1] << ", " << work_item_sizes[2] << ")" << std::endl;
        
        std::cout << "global_work_sizes = (" << global_work_sizes[0] << ", " <<
        global_work_sizes[1] << ", " << global_work_sizes[2] << ")" << std::endl;
        
        
#endif
        
        err = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_sizes,
                                     work_item_sizes.size() == 0 ? NULL :  &work_item_sizes[0], 0, NULL, &event);
        
    } else {
        result << "NO Initiate calculation" << std::endl;
    }
    
    // retrieve result
    cl_event event2 = nullptr;
    if (err == CL_SUCCESS) {
        if (debug) {
            result << "Retrieve result:" << std::endl;
        }
        
        if (use_float) {
            err = clEnqueueReadBuffer(queue, cl_output_matrix, CL_TRUE, 0, ncol * ncol * sizeof(float), output_matrix_f, 1, &event, &event2);
            
        } else {
            err = clEnqueueReadBuffer(queue, cl_output_matrix, CL_TRUE, 0, ncol * ncol * sizeof(double), output_matrix_d, 1, &event, &event2);
        }
        
        if (event != nullptr) {
            
            //            if (false) {
            //                Profile enqueue_profile = getProfileTimes(event);
            //                CERR << "clEnqueueNDRangeKernel: " << enqueue_profile.queued << " Q msec; " <<
            //                enqueue_profile.pending << " P msec; " << enqueue_profile.exec << " E msec" << std::endl;
            //            }
            
            clReleaseEvent(event);
            
        } else {
            result << "event is null " << std::endl;
        }
        
        if (event2 != nullptr) {
            clWaitForEvents(1, &event2);
            
            //            if (false) {
            //                Profile enqueue_read = getProfileTimes(event2);
            //                CERR << "clEnqueueReadBuffer: " << enqueue_read.queued << " Q msec; " <<
            //                enqueue_read.pending << " P msec; " << enqueue_read.exec << " E msec" << std::endl;
            //            }
            
            clReleaseEvent(event2);
            
        } else {
            result << "event2 is null " << std::endl;
        }
        
    } else {
        result << "NO Retrieve result" << std::endl;
    }
    
    std::string err_str = clErrorToString(err);
    result << std::endl << err_str << std::endl;
    
    // cleanup
    if (cl_output_matrix != nullptr) {
        clReleaseMemObject(cl_output_matrix);
        cl_output_matrix = NULL;
    }
    
    if (cl_input_matrix != nullptr) {
        clReleaseMemObject(cl_input_matrix);
        cl_input_matrix = NULL;
    }
    
    if (debug) {
        CERR << result.str();
    }
    
    return clErrorToString(err);
}

cl_int xopencl_calc_x(cl_context context, cl_kernel kernel_f, cl_kernel kernel_d, bool is_float,
                     cl_command_queue queue, void *inMatrix, void *outMatrix, size_t nrow, size_t ncol,
                     const std::vector<size_t>& work_item_sizes, int row_multiple, int col_multiple,
                     int row_tile_size, int col_tile_size, bool verbose)
{
    bool gTrace = false;
    
    //    int local_mem_per_workgroup = 0;
    //===================================
    
    float *input_matrix_f = (float *)inMatrix;
    
    float *output_matrix_f = (float *)outMatrix;
    
    double *input_matrix_d = (double *)inMatrix;
    
    double *output_matrix_d = (double *)outMatrix;
    
    if (gTrace) {
        CERR << "xopencl_calc_x( " << (is_float ? "FLOAT" : "DOUBLE") << ")" << std::endl << std::endl;
    }
    
    cl_kernel kernel = is_float ? kernel_f : kernel_d;
    
    cl_int err = CL_SUCCESS;
    
    // buffers
    cl_mem cl_input_matrix = NULL;
    if (err == CL_SUCCESS) {
        if (gTrace) {
            CERR << "clCreateBuffer cl_input_matrix:" << std::endl;
        }
        
            if (is_float) {
                cl_input_matrix = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                 nrow * ncol * sizeof(float), input_matrix_f, &err);
                
            } else {
                cl_input_matrix = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                 nrow * ncol * sizeof(double), input_matrix_d, &err);
            }
    }
    
    const float zero_f = 0.0;
    const double zero_d = 0.0;
    
    
    cl_event write_event = nullptr;
    
    cl_mem cl_output_matrix = NULL;
    if (err == CL_SUCCESS) {
        if (gTrace) {
            CERR << "clCreateBuffer cl_output_vector:" << std::endl;
        }
        
        if (is_float) {
            cl_output_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              ncol * ncol * sizeof(float), NULL, &err);
            
        } else {
            cl_output_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              ncol * ncol * sizeof(double), NULL, &err);
        }
    }
    
    /*
    cl_event fill_out_event = nullptr;
    if (err == CL_SUCCESS) {
        if (is_float) {
            err = clEnqueueFillBuffer(queue, cl_output_matrix, &zero_f, sizeof(zero_f), 0,
                                      ncol * ncol * sizeof(float), 0, nullptr, &fill_out_event);
            
        } else {
            err = clEnqueueFillBuffer(queue, cl_output_matrix, &zero_d, sizeof(zero_d), 0,
                                      ncol * ncol * sizeof(double), 0, nullptr, &fill_out_event);
        }
    }
    */
    
    // initiate calculation
    cl_event kernel_event = nullptr;
    if (err == CL_SUCCESS) {
        if (gTrace) {
            CERR << "Initiate calculation:" << std::endl;
        }
        cl_int cl_nrow = (cl_int)nrow;
        cl_int cl_ncol = (cl_int)ncol;
        //        cl_int cl_row_tile_size = row_tile_size;
        //        cl_int cl_col_tile_size = col_tile_size;
        clSetKernelArg(kernel, 0, sizeof(cl_nrow), &cl_nrow);
        clSetKernelArg(kernel, 1, sizeof(cl_ncol), &cl_ncol);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_input_matrix);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_output_matrix);
        //        clSetKernelArg(kernel, 4, sizeof(cl_row_tile_size), &cl_row_tile_size);
        //        clSetKernelArg(kernel, 5, sizeof(cl_col_tile_size), &cl_col_tile_size);
        //        if (local_mem_per_workgroup > 0) {
        //            clSetKernelArg(kernel, 6, local_mem_per_workgroup, NULL);
        //        } else {
        //            clSetKernelArg(kernel, 6, sizeof(cl_mem), NULL);
        //        }
        
        cl_uint work_dim = 3;
        size_t global_work_sizes[] = { (size_t)ncol / col_tile_size, (size_t)ncol / row_tile_size, 1 };
        
//        cl_event events[] = { fill_out_event, write_event };
//        cl_int event_count = write_event == nullptr ? 1 : 2;
//        
//        err = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_sizes,
//                                     work_item_sizes.size() == 0 ? NULL :  &work_item_sizes[0],
//                                     event_count, events, &kernel_event);
        
        err = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_sizes,
                                     work_item_sizes.size() == 0 ? NULL :  &work_item_sizes[0],
                                     0, NULL, &kernel_event);
        
        if (verbose || gTrace || err == CL_INVALID_WORK_GROUP_SIZE || err == CL_INVALID_WORK_ITEM_SIZE) {
            CERR << "rows = " << nrow << ", cols = " << ncol << std::endl;
            
            CERR << "global_work_sizes = (" << global_work_sizes[0] << ", " <<
            global_work_sizes[1] << ", " << global_work_sizes[2] << ")" << std::endl;
            
            CERR << "work_item_sizes = (" << work_item_sizes[0] << ", " <<
            work_item_sizes[1] << ", " << work_item_sizes[2] << ")" << std::endl;
        }
        
    } else {
        if (gTrace) {
            CERR << "NO Initiate calculation" << std::endl;
        }
    }
    
//    if (fill_out_event != nullptr) {
//        clReleaseEvent(fill_out_event);
//        
//    } else {
//        if (gTrace) {
//            CERR << "fill_out_event is null " << std::endl;
//        }
//    }
    
    // retrieve result
    if (err == CL_SUCCESS) {
        if (gTrace) {
            CERR << "Retrieve result:" << std::endl;
        }
        
//        if (ncol == full_ncol && nrow == full_nrow) {
            if (is_float) {
                err = clEnqueueReadBuffer(queue, cl_output_matrix, CL_TRUE, 0, ncol * ncol * sizeof(float), output_matrix_f, 1, &kernel_event, nullptr /*&read_event*/);
                
            } else {
                err = clEnqueueReadBuffer(queue, cl_output_matrix, CL_TRUE, 0, ncol * ncol * sizeof(double), output_matrix_d, 1, &kernel_event, nullptr /*&read_event*/);
            }
            
//        } else {
//            size_t out_region_f[] = { ncol * sizeof(float), (size_t)ncol, 1 };
//            size_t out_region_d[] = { ncol * sizeof(double), (size_t)ncol, 1 };
//            
//            if (is_float) {
//                err = clEnqueueReadBufferRect(queue, cl_output_matrix, CL_TRUE, origin, origin,
//                                              out_region_f, full_ncol * sizeof(float),
//                                              0, ncol * sizeof(float), 0, output_matrix_f, 1,
//                                              &kernel_event, nullptr);
//                
//            } else {
//                err = clEnqueueReadBufferRect(queue, cl_output_matrix, CL_TRUE, origin, origin,
//                                              out_region_d, full_ncol * sizeof(double),
//                                              0, ncol * sizeof(double), 0, output_matrix_d, 1,
//                                              &kernel_event, nullptr);
//            }
//        }
        
        
        if (kernel_event != nullptr) {
            clReleaseEvent(kernel_event);
            
        } else {
            if (gTrace) {
                CERR << "kernel_event is null " << std::endl;
            }
        }
        
    } else {
        if (gTrace) {
            CERR << "NO Retrieve result" << std::endl;
        }
    }
    
    if (gTrace) {
        std::string err_str = clErrorToString(err);
        CERR << std::endl << err_str << std::endl;
    }
    
    // cleanup
    if (cl_output_matrix != nullptr) {
        clReleaseMemObject(cl_output_matrix);
        cl_output_matrix = NULL;
    }
    
    if (cl_input_matrix != nullptr) {
        clReleaseMemObject(cl_input_matrix);
        cl_input_matrix = NULL;
    }
    
    return err;
}
#endif
#endif


//
//  gemm_opencl.cpp
//  multiBLAS.XC
//
//  Created by michael on 7/23/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#define R_NO_REMAP  // needed if include fstream

#include "gemm_opencl.h"
#include "gemm_naive.h"
#include "opencl_info.h"
#include "shim.h"

#ifdef USE_TIMING
#include <chrono>
#endif

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#ifdef USE_TIMING
using namespace std::chrono;
#endif

// ========== Globals ==============================================================================

extern bool gTrace;    // for debugging

// ========== Local Functions ============================================================================

cl_int createAndWriteInput(cl_context context, cl_command_queue queue,
                           int nrow, int ncol, int row_multiple, int col_multiple,
                           int row_tile_size, int col_tile_size, const std::vector<size_t>& work_item_sizes,
                           const void *inMatrix, bool is_float,
                           size_t& full_nrow, size_t& full_ncol, cl_mem& cl_input_matrix,
                           cl_event& fill_in_event1, cl_event& fill_in_event2, cl_event& write_event);

// ========== Functions ============================================================================

cl_int opencl_calc_gemm(cl_context context, cl_kernel kernel_f, cl_kernel kernel_d, bool is_float,
                        cl_command_queue queue,
                        const void *inMatrixA, int nrowA, int ncolA, bool transposeA,
                        const void *inMatrixB, int nrowB, int ncolB, bool transposeB,
                        double alpha, double beta, void *outMatrix,
                        const std::vector<size_t>& work_item_sizes, int row_multiple, int col_multiple,
                        int row_tile_size, int col_tile_size, bool verbose)
{
#ifdef USE_TIMING
    steady_clock::time_point start_time = steady_clock::now();
#endif
    
    //    int local_mem_per_workgroup = 0;
    //===================================
    
    float *output_matrix_f = (float *)outMatrix;
    double *output_matrix_d = (double *)outMatrix;
    
    if (gTrace) {
        CERR << "opencl_calc_gemm( " << (is_float ? "FLOAT" : "DOUBLE") << ")" << std::endl << std::endl;
    }
    
    int nrowC = transposeA ? ncolA : nrowA;
    int ncolC = transposeB ? nrowB : ncolB;
    
//    if (is_float) {
//        for (size_t k = 0; k < nrowC * ncolC; k++) {
//            output_matrix_f[k] = k + 0.1;
//        }
//        
//    } else {
//        for (size_t k = 0; k < nrowC * ncolC; k++) {
//            output_matrix_d[k] = k + 0.1;
//        }
//    }
//    
//    return CL_SUCCESS;
    
    cl_kernel kernel = is_float ? kernel_f : kernel_d;
    
    cl_int err = CL_SUCCESS;
    
    // ----- inputA -----
    
    const void *tmA = inMatrixA;
    if (err == CL_SUCCESS && !transposeA) {
        if (is_float) {
            tmA = calloc(nrowA * ncolA, sizeof(float));
            if (tmA == nullptr) {
                err = CL_OUT_OF_HOST_MEMORY;
                
            } else {
                transpose_f((const float *)inMatrixA, nrowA, ncolA, (float *)tmA);
            }
            
        } else {
            tmA = calloc(nrowA * ncolA, sizeof(double));
            if (tmA == nullptr) {
                err = CL_OUT_OF_HOST_MEMORY;
                
            } else {
                transpose_d((const double *)inMatrixA, nrowA, ncolA, (double *)tmA);
            }
        }
    }
    
    size_t full_nrowA = 0;
    size_t full_ncolA = 0;
    cl_mem cl_input_matrixA = nullptr;
    cl_event fill_in_event1A = nullptr;
    cl_event fill_in_event2A = nullptr;
    cl_event write_eventA = nullptr;

    if (err == CL_SUCCESS) {
        err = createAndWriteInput(context, queue,
                                  nrowA, ncolA, row_multiple, col_multiple,
                                  row_tile_size, col_tile_size, work_item_sizes,
                                  tmA, is_float,
                                  full_nrowA, full_ncolA, cl_input_matrixA,
                                  fill_in_event1A, fill_in_event2A, write_eventA);
    }

    // ----- inputB -----

    const void *mB = inMatrixB;
    if (err == CL_SUCCESS && transposeB) {
        if (is_float) {
            mB = calloc(nrowB * ncolB, sizeof(float));
            if (mB == nullptr) {
                err = CL_OUT_OF_HOST_MEMORY;
                
            } else {
                transpose_f((const float *)inMatrixB, nrowB, ncolB, (float *)mB);
            }
            
        } else {
            mB = calloc(nrowB * ncolB, sizeof(double));
            if (mB == nullptr) {
                err = CL_OUT_OF_HOST_MEMORY;
                
            } else {
                transpose_d((const double *)inMatrixB, nrowB, ncolB, (double *)mB);
            }
        }
    }

    size_t full_nrowB = 0;
    size_t full_ncolB = 0;
    cl_mem cl_input_matrixB = nullptr;
    cl_event fill_in_event1B = nullptr;
    cl_event fill_in_event2B = nullptr;
    cl_event write_eventB = nullptr;
    
    if (err == CL_SUCCESS) {
        err = createAndWriteInput(context, queue,
                                  nrowB, ncolB, row_multiple, col_multiple,
                                  row_tile_size, col_tile_size, work_item_sizes,
                                  mB, is_float,
                                  full_nrowB, full_ncolB, cl_input_matrixB,
                                  fill_in_event1B, fill_in_event2B, write_eventB);
    }

    size_t full_nrowC = transposeA ? full_ncolA : full_nrowA;
    size_t full_ncolC = transposeB ? full_nrowB : full_ncolB;

    cl_mem cl_output_matrix = NULL;
    if (err == CL_SUCCESS) {
        if (gTrace) {
            CERR << "clCreateBuffer cl_output_vector:" << std::endl;
        }
        
        if (is_float) {
            cl_output_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              full_nrowC * full_ncolC * sizeof(float), NULL, &err);
            
        } else {
            cl_output_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              full_nrowC * full_ncolC * sizeof(double), NULL, &err);
        }
    }

    // initiate calculation
    cl_event kernel_event = nullptr;
    if (err == CL_SUCCESS) {
        if (gTrace) {
            CERR << "Initiate calculation:" << std::endl;
        }
        cl_int cl_nrowA = (cl_int)full_nrowA;
        cl_int cl_ncolA = (cl_int)full_ncolA;
        cl_int cl_nrowB = (cl_int)full_nrowB;
        cl_int cl_ncolB = (cl_int)full_ncolB;
        
        clSetKernelArg(kernel, 0, sizeof(cl_int), &cl_nrowA);
        clSetKernelArg(kernel, 1, sizeof(cl_int), &cl_ncolA);
        clSetKernelArg(kernel, 2, sizeof(cl_int), &cl_nrowB);
        clSetKernelArg(kernel, 3, sizeof(cl_int), &cl_ncolB);
        if (is_float) {
            float alpha_f = (float)alpha;
            float beta_f = (float)beta;
            clSetKernelArg(kernel, 4, sizeof(float), &alpha_f);
            clSetKernelArg(kernel, 5, sizeof(float), &beta_f);
            
        } else {
            clSetKernelArg(kernel, 4, sizeof(double), &alpha);
            clSetKernelArg(kernel, 5, sizeof(double), &beta);
        }
        clSetKernelArg(kernel, 6, sizeof(cl_mem), &cl_input_matrixA);
        clSetKernelArg(kernel, 7, sizeof(cl_mem), &cl_input_matrixB);
        clSetKernelArg(kernel, 8, sizeof(cl_mem), &cl_output_matrix);
        
        cl_uint work_dim = 3;
        size_t global_work_sizes[] = { (size_t)full_ncolC / col_tile_size, (size_t)full_nrowC / row_tile_size, 1 };
        
        //cl_event events[] = { /*fill_out_event,*/ write_event };
        std::vector<cl_event> events;
        if (fill_in_event1A != nullptr) events.push_back(fill_in_event1A);
        if (fill_in_event2A != nullptr) events.push_back(fill_in_event2A);
        if (fill_in_event1B != nullptr) events.push_back(fill_in_event1B);
        if (fill_in_event2B != nullptr) events.push_back(fill_in_event2B);
        if (write_eventA != nullptr) events.push_back(write_eventA);
        if (write_eventB != nullptr) events.push_back(write_eventB);
        cl_int event_count = (cl_int)events.size();
        
        err = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_sizes,
                                     work_item_sizes.size() == 0 ? NULL :  &work_item_sizes[0],
                                     event_count, event_count == 0 ? nullptr : &events[0], &kernel_event);
        
        if (verbose || gTrace || err == CL_INVALID_WORK_GROUP_SIZE || err == CL_INVALID_WORK_ITEM_SIZE) {
            CERR << "rows = " << full_nrowC << ", cols = " << full_ncolC << std::endl;
            
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
    
    
    if (write_eventA != nullptr) clReleaseEvent(write_eventA);
    if (write_eventB != nullptr) clReleaseEvent(write_eventB);
    
    if (fill_in_event1A != nullptr) clReleaseEvent(fill_in_event1A);
    if (fill_in_event2A != nullptr) clReleaseEvent(fill_in_event2A);
    if (fill_in_event1B != nullptr) clReleaseEvent(fill_in_event1B);
    if (fill_in_event2B != nullptr) clReleaseEvent(fill_in_event2B);
    
    // retrieve result
    if (err == CL_SUCCESS) {
        if (ncolC == full_ncolC && nrowC == full_nrowC) {
            if (gTrace) {
                CERR << "Retrieve result (clEnqueueReadBuffer):" << std::endl;
            }
            
            if (is_float) {
                err = clEnqueueReadBuffer(queue, cl_output_matrix, CL_TRUE, 0, nrowC * ncolC * sizeof(float), output_matrix_f, 1, &kernel_event, nullptr);
                
            } else {
                err = clEnqueueReadBuffer(queue, cl_output_matrix, CL_TRUE, 0, nrowC * ncolC * sizeof(double), output_matrix_d, 1, &kernel_event, nullptr);
            }
            
        } else {
            if (gTrace) {
                CERR << "Retrieve result (clEnqueueReadBufferRect):" << std::endl;
            }
            
            size_t origin[] = { 0, 0, 0 };
            size_t out_region_f[] = { nrowC * sizeof(float), (size_t)ncolC, 1 };
            size_t out_region_d[] = { nrowC * sizeof(double), (size_t)ncolC, 1 };
            
            if (is_float) {
                err = clEnqueueReadBufferRect(queue, cl_output_matrix, CL_TRUE, origin, origin,
                                              out_region_f, full_nrowC * sizeof(float),
                                              0, nrowC * sizeof(float), 0, output_matrix_f, 1,
                                              &kernel_event, nullptr);
                
            } else {
                err = clEnqueueReadBufferRect(queue, cl_output_matrix, CL_TRUE, origin, origin,
                                              out_region_d, full_nrowC * sizeof(double),
                                              0, nrowC * sizeof(double), 0, output_matrix_d, 1,
                                              &kernel_event, nullptr);
            }
        }
        
        
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
    
    if (cl_input_matrixA != nullptr) {
        clReleaseMemObject(cl_input_matrixA);
        cl_input_matrixA = NULL;
    }
    
    if (cl_input_matrixB != nullptr) {
        clReleaseMemObject(cl_input_matrixB);
        cl_input_matrixB = NULL;
    }
    
    if (tmA != nullptr && tmA != inMatrixA) {
        free((void *)tmA);
    }
    
    if (mB != nullptr && mB != inMatrixB) {
        free((void *)mB);
    }
    
#ifdef USE_TIMING
    steady_clock::time_point end_time = steady_clock::now();
    duration<double> time_span = duration_cast<duration<double> >(end_time - start_time);
    
    double nsec = std::chrono::duration_cast<std::chrono::nanoseconds> (end_time - start_time).count();
    double gflops = (2.0 * nrow * (ncol + 0.5 * (ncol - 1.0) * ncol)) / nsec;
    
    if (gTrace || verbose) {
        CERR << "opencl_calc_x Elapsed: " << time_span.count() << " sec " <<
        gflops << " GFLOPS" << std::endl;
    }
#endif
    
    return err;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cl_int createAndWriteInput(cl_context context, cl_command_queue queue,
                           int nrow, int ncol, int row_multiple, int col_multiple,
                           int row_tile_size, int col_tile_size, const std::vector<size_t>& work_item_sizes,
                           const void *inMatrix, bool is_float,
                           size_t& full_nrow, size_t& full_ncol, cl_mem& cl_input_matrix,
                           cl_event& fill_in_event1, cl_event& fill_in_event2, cl_event& write_event)
{
    cl_int err = CL_SUCCESS;

    float *input_matrix_f = (float *)inMatrix;
    double *input_matrix_d = (double *)inMatrix;

    full_nrow = row_multiple * ((nrow + row_multiple - 1) / row_multiple);
    
    // cheap way to find least-common-multiple; not terribly slow for small row_tile_size & col_tile_size
    size_t gcd = col_tile_size < row_tile_size ? col_tile_size : row_tile_size;
    while (gcd > 1 && col_tile_size % gcd != 0 && row_tile_size % gcd != 0) gcd--;
    size_t lcm = col_tile_size * row_tile_size / gcd;
    
    full_ncol = (ncol + lcm - 1) / lcm;
    
    full_ncol = work_item_sizes[0] * ((full_ncol + work_item_sizes[0] - 1) / work_item_sizes[0]);
    full_ncol = work_item_sizes[1] * ((full_ncol + work_item_sizes[1] - 1) / work_item_sizes[1]);
    
    full_ncol *= lcm;
    full_ncol = col_multiple * ((full_ncol + col_multiple - 1) / col_multiple);
    
    if (gTrace) {
        CERR << "createAndWriteInput: nrow = " << nrow << ", ncol = " << ncol << ", full_nrow = " << full_nrow << ", full_ncol = " << full_ncol << std::endl;
    }
    
    // buffers
    cl_input_matrix = NULL;
    if (err == CL_SUCCESS) {
        if (gTrace) {
            CERR << "clCreateBuffer cl_input_matrix:" << std::endl;
        }
        
        if (ncol == full_ncol && nrow == full_nrow) {
            if (is_float) {
                cl_input_matrix = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                 nrow * ncol * sizeof(float), input_matrix_f, &err);
                
            } else {
                cl_input_matrix = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                 nrow * ncol * sizeof(double), input_matrix_d, &err);
            }
            
        } else {
            if (is_float) {
                cl_input_matrix = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                                 full_nrow * full_ncol * sizeof(float), nullptr, &err);
                
            } else {
                cl_input_matrix = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                                 full_nrow * full_ncol * sizeof(double), nullptr, &err);
            }
        }
    }
    
    size_t origin[] = { 0, 0, 0 };
    size_t region_f[] = { nrow * sizeof(float), (size_t)ncol, 1 };
    size_t region_d[] = { nrow * sizeof(double), (size_t)ncol, 1 };
    
    fill_in_event1 = nullptr;
    fill_in_event2 = nullptr;
    if (err == CL_SUCCESS) {
        if (ncol != full_ncol || nrow != full_nrow) {
            if (full_nrow > nrow) {
                if (gTrace) {
                    CERR << "clEnqueueWriteBufferRect zeros1" << std::endl;
                }
                
                size_t origin1[] = { static_cast<size_t>(nrow), 0, 0 };
                if (is_float) {
                    float *zeros1 = (float *)calloc(ncol * (full_nrow - nrow), sizeof(float));
                    if (zeros1 == nullptr) {
                        err = CL_OUT_OF_HOST_MEMORY;
                        
                    } else {
                        memset(zeros1, 0, ncol * (full_nrow - nrow) * sizeof(float));
                        size_t region_f1[] = { (full_nrow - nrow) * sizeof(float), (size_t)ncol, 1 };
                        
                        err = clEnqueueWriteBufferRect(queue, cl_input_matrix, CL_FALSE, origin1, origin,
                                                       region_f1, full_nrow * sizeof(float), 0, (full_nrow - nrow) * sizeof(float),
                                                       0, zeros1,
                                                       0, nullptr, &fill_in_event1);
                        
                        free(zeros1);
                    }
                    
                    
                } else {
                    double *zeros1 = (double *)calloc(ncol * (full_nrow - nrow), sizeof(double));
                    if (zeros1 == nullptr) {
                        err = CL_OUT_OF_HOST_MEMORY;
                        
                    } else {
                        memset(zeros1, 0, ncol * (full_nrow - nrow) * sizeof(double));
                        size_t region_d1[] = { (full_nrow - nrow) * sizeof(double), (size_t)ncol, 1 };
                        
                        err = clEnqueueWriteBufferRect(queue, cl_input_matrix, CL_FALSE, origin1, origin,
                                                       region_d1, full_nrow * sizeof(double), 0, (full_nrow - nrow) * sizeof(double),
                                                       0, zeros1,
                                                       0, nullptr, &fill_in_event1);
                        
                        free(zeros1);
                    }
                }
            }
            
            if (full_ncol > ncol && err == CL_SUCCESS) {
                if (gTrace) {
                    CERR << "clEnqueueWriteBufferRect zeros2" << std::endl;
                }
                size_t origin2[] = { 0, static_cast<size_t>(ncol), 0 };
                if (is_float) {
                    float *zeros2 = (float *)calloc(full_nrow * (full_ncol - ncol), sizeof(float));
                    if (zeros2 == nullptr) {
                        err = CL_OUT_OF_HOST_MEMORY;
                        
                    } else {
                        memset(zeros2, 0, full_nrow * (full_ncol - ncol) * sizeof(float));
                        size_t region_f2[] = { full_nrow * sizeof(float), (size_t)(full_ncol - ncol), 1 };
                        
                        err = clEnqueueWriteBufferRect(queue, cl_input_matrix, CL_FALSE, origin2, origin,
                                                       region_f2, full_nrow * sizeof(float), 0, full_nrow * sizeof(float),
                                                       0, zeros2,
                                                       0, nullptr, &fill_in_event2);
                        
                        free(zeros2);
                    }
                    
                } else {
                    double *zeros2 = (double *)calloc(full_nrow * (full_ncol - ncol), sizeof(double));
                    if (zeros2 == nullptr) {
                        err = CL_OUT_OF_HOST_MEMORY;
                        
                    } else {
                        memset(zeros2, 0, full_nrow * (full_ncol - ncol) * sizeof(double));
                        size_t region_d2[] = { full_nrow * sizeof(double), (size_t)(full_ncol - ncol), 1 };
                        
                        err = clEnqueueWriteBufferRect(queue, cl_input_matrix, CL_FALSE, origin2, origin,
                                                       region_d2, full_nrow * sizeof(double), 0, full_nrow * sizeof(double),
                                                       0, zeros2,
                                                       0, nullptr, &fill_in_event2);
                        
                        free(zeros2);
                    }
                }
            }
        }
    }
    
    write_event = nullptr;
    if (err == CL_SUCCESS) {
        if (ncol != full_ncol || nrow != full_nrow) {
            if (is_float) {
                err = clEnqueueWriteBufferRect(queue, cl_input_matrix, CL_FALSE, origin, origin,
                                               region_f, full_nrow * sizeof(float), 0, nrow * sizeof(float),
                                               0, input_matrix_f,
                                               0, nullptr, &write_event);
                
            } else {
                err = clEnqueueWriteBufferRect(queue, cl_input_matrix, CL_FALSE, origin, origin,
                                               region_d, full_nrow * sizeof(double), 0, nrow * sizeof(double),
                                               0, input_matrix_d,
                                               0, nullptr, &write_event);
            }
        }
    }
    
    return err;
}
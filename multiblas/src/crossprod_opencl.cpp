//
//  crossprod_opencl.cpp
//  template
//
//  Created by michael on 4/20/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#define R_NO_REMAP  // needed if include fstream

#include "crossprod_opencl.h"
#include "nullptr.h"
#include "opencl_info.h"
#include "shim.h"

#ifdef USE_TIMING
#include <chrono>
#endif

#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstddef>

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

// ========== Functions ============================================================================

cl_int opencl_calc_x(cl_context context, cl_kernel kernel_f, cl_kernel kernel_d, bool is_float,
                     cl_command_queue queue, void *inMatrix, void *outMatrix, size_t nrow, size_t ncol,
                     const std::vector<size_t>& work_item_sizes, int vector_size, int row_multiple,
                     int row_tile_size, int col_tile_size, bool verbose)
{
#ifdef USE_TIMING
    steady_clock::time_point start_time = steady_clock::now();
#endif
    
    //===================================
    
    float *input_matrix_f = (float *)inMatrix;
    
    float *output_matrix_f = (float *)outMatrix;
    
    double *input_matrix_d = (double *)inMatrix;
    
    double *output_matrix_d = (double *)outMatrix;
    
    if (gTrace) {
        CERR << "opencl_calc_x( " << (is_float ? "FLOAT" : "DOUBLE") << ")" << std::endl << std::endl;
    }
    
    cl_kernel kernel = is_float ? kernel_f : kernel_d;
    
    cl_int err = CL_SUCCESS;

    size_t multiple = vector_size * row_multiple;
    size_t full_nrow = multiple * ((nrow + multiple - 1) / multiple);

    // cheap way to find least-common-multiple; not terribly slow for small row_tile_size & col_tile_size
    size_t gcd = col_tile_size < row_tile_size ? col_tile_size : row_tile_size;
    while (gcd > 1 && col_tile_size % gcd != 0 && row_tile_size % gcd != 0) gcd--;
    size_t lcm = col_tile_size * row_tile_size / gcd;
    
    size_t full_ncol = (ncol + lcm - 1) / lcm;
    
    full_ncol = work_item_sizes[0] * ((full_ncol + work_item_sizes[0] - 1) / work_item_sizes[0]);
    full_ncol = work_item_sizes[1] * ((full_ncol + work_item_sizes[1] - 1) / work_item_sizes[1]);

    full_ncol *= lcm;

    if (gTrace) {
        CERR << "opencl_calc_x: nrow = " << nrow << ", ncol = " << ncol << ", full_nrow = " << full_nrow << ", full_ncol = " << full_ncol << std::endl;
    }

    // buffers
    cl_mem cl_input_matrix = NULL;
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

#define USE_FILL 0
#define USE_MAP_IN 0
#define USE_MAP_OUT 0
    
#define FILL_NAN 0
#if FILL_NAN
    const float nan_f = (float)NAN;
    const double nan_d = NAN;
    
    cl_event fill_in_event = nullptr;
    if (err == CL_SUCCESS) {
        if (ncol != full_ncol || nrow != full_nrow) {
            if (gTrace) {
                CERR << "clEnqueueFillBuffer" << std::endl;
            }
            
            if (is_float) {
                err = clEnqueueFillBuffer(queue, cl_input_matrix, &nan_f, sizeof(nan_f), 0,
                                          full_nrow * full_ncol * sizeof(float), 0, nullptr, &fill_in_event);
                
            } else {
                err = clEnqueueFillBuffer(queue, cl_input_matrix, &nan_d, sizeof(nan_d), 0,
                                          full_nrow * full_ncol * sizeof(double), 0, nullptr, &fill_in_event);
            }
        }
    }
#endif
    
#if USE_FILL
    const float zero_f = 0.0;
    const double zero_d = 0.0;
    
    cl_event fill_in_event = nullptr;
    if (err == CL_SUCCESS) {
        if (ncol != full_ncol || nrow != full_nrow) {
            if (gTrace) {
                CERR << "clEnqueueFillBuffer" << std::endl;
            }
            
            if (is_float) {
                err = clEnqueueFillBuffer(queue, cl_input_matrix, &zero_f, sizeof(zero_f), 0,
                                          full_nrow * full_ncol * sizeof(float), 0, nullptr, &fill_in_event);
                
            } else {
                err = clEnqueueFillBuffer(queue, cl_input_matrix, &zero_d, sizeof(zero_d), 0,
                                          full_nrow * full_ncol * sizeof(double), 0, nullptr, &fill_in_event);
            }
        }
    }
#else
    cl_event fill_in_event1 = nullptr;
    cl_event fill_in_event2 = nullptr;
    if (err == CL_SUCCESS) {
        if (ncol != full_ncol || nrow != full_nrow) {
            if (full_nrow > nrow) {
                if (gTrace) {
                    CERR << "clEnqueueWriteBufferRect zeros1" << std::endl;
                }
                
                size_t origin1[] = { nrow, 0, 0 };
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
                size_t origin2[] = { 0, ncol, 0 };
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
#endif
    
    cl_event write_event = nullptr;
    if (err == CL_SUCCESS) {
        if (ncol != full_ncol || nrow != full_nrow) {
#if USE_FILL
            if (is_float) {
                err = clEnqueueWriteBufferRect(queue, cl_input_matrix, CL_FALSE, origin, origin,
                                               region_f, full_nrow * sizeof(float), 0, nrow * sizeof(float),
                                               0, input_matrix_f,
                                               1, &fill_in_event, &write_event);
                
            } else {
                err = clEnqueueWriteBufferRect(queue, cl_input_matrix, CL_FALSE, origin, origin,
                                               region_d, full_nrow * sizeof(double), 0, nrow * sizeof(double),
                                               0, input_matrix_d,
                                               1, &fill_in_event, &write_event);
            }
#else
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
#endif
        }
    }
    
    cl_mem cl_output_matrix = NULL;
    if (err == CL_SUCCESS) {
        if (gTrace) {
            CERR << "clCreateBuffer cl_output_vector:" << std::endl;
        }

        if (is_float) {
            cl_output_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              full_ncol * full_ncol * sizeof(float), NULL, &err);

        } else {
            cl_output_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              full_ncol * full_ncol * sizeof(double), NULL, &err);
        }
    }
    
#if FILL_NAN
    cl_event fill_out_event = nullptr;
    if (err == CL_SUCCESS) {
        if (is_float) {
            err = clEnqueueFillBuffer(queue, cl_output_matrix, &nan_f, sizeof(nan_f), 0,
                                      full_ncol * full_ncol * sizeof(float), 0, nullptr, &fill_out_event);
            
        } else {
            err = clEnqueueFillBuffer(queue, cl_output_matrix, &nan_d, sizeof(nan_d), 0,
                                      full_ncol * full_ncol * sizeof(double), 0, nullptr, &fill_out_event);
        }
    }
#endif
    
    // initiate calculation
    cl_event kernel_event = nullptr;
    if (err == CL_SUCCESS) {
        if (gTrace) {
            CERR << "Initiate calculation:" << std::endl;
        }
        cl_int cl_nrow = (cl_int)full_nrow;
        cl_int cl_ncol = (cl_int)full_ncol;
        clSetKernelArg(kernel, 0, sizeof(cl_nrow), &cl_nrow);
        clSetKernelArg(kernel, 1, sizeof(cl_ncol), &cl_ncol);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_input_matrix);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_output_matrix);
        
        cl_uint work_dim = 3;
        size_t global_work_sizes[] = { (size_t)full_ncol / col_tile_size, (size_t)full_ncol / row_tile_size, 1 };

#if USE_FILL
        cl_event *events = write_event == nullptr ? nullptr : &write_event;
        cl_int event_count = write_event == nullptr ? 0 : 1;
        
        err = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_sizes,
                                     work_item_sizes.size() == 0 ? NULL :  &work_item_sizes[0],
                                     event_count, events, &kernel_event);
#else
        std::vector<cl_event> events;
        if (fill_in_event1 != nullptr) events.push_back(fill_in_event1);
        if (fill_in_event2 != nullptr) events.push_back(fill_in_event2);
        if (write_event != nullptr) events.push_back(write_event);
#if FILL_NAN
        if (fill_out_event != nullptr) events.push_back(fill_out_event);
#endif
        cl_int event_count = (cl_int)events.size();
        
        err = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_sizes,
                                     work_item_sizes.size() == 0 ? NULL :  &work_item_sizes[0],
                                     event_count, event_count == 0 ? nullptr : &events[0], &kernel_event);
#endif
        
        if (verbose || gTrace || err == CL_INVALID_WORK_GROUP_SIZE || err == CL_INVALID_WORK_ITEM_SIZE) {
            CERR << "rows = " << full_nrow << ", cols = " << full_ncol << std::endl;
            
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

    
    if (write_event != nullptr) clReleaseEvent(write_event);
    
#if USE_FILL
#else
    if (fill_in_event1 != nullptr) clReleaseEvent(fill_in_event1);
    if (fill_in_event2 != nullptr) clReleaseEvent(fill_in_event2);
#endif
    
    // retrieve result
    if (err == CL_SUCCESS) {
        if (ncol == full_ncol && nrow == full_nrow) {
#if USE_MAP_OUT
            if (gTrace) {
                CERR << "Retrieve result (clEnqueueMapBuffer):" << std::endl;
            }
            
            if (is_float) {
                void *mapped = clEnqueueMapBuffer(queue, cl_output_matrix, CL_TRUE, CL_MAP_READ,
                                                  0, full_ncol * full_ncol * sizeof(float), 1, &kernel_event, nullptr, &err);
                
                if (err == CL_SUCCESS) {
                    memcpy(output_matrix_f, mapped, full_ncol * full_ncol * sizeof(float));
                    err = clEnqueueUnmapMemObject(queue, cl_output_matrix, mapped, 0, nullptr, nullptr);
                }
                
            } else {
                void *mapped = clEnqueueMapBuffer(queue, cl_output_matrix, CL_TRUE, CL_MAP_READ,
                                                  0, full_ncol * full_ncol * sizeof(double), 1, &kernel_event, nullptr, &err);
                
                if (err == CL_SUCCESS) {
                    memcpy(output_matrix_d, mapped, full_ncol * full_ncol * sizeof(double));
                    err = clEnqueueUnmapMemObject(queue, cl_output_matrix, mapped, 0, nullptr, nullptr);
                }
            }
#else
            if (gTrace) {
                CERR << "Retrieve result (clEnqueueReadBuffer):" << std::endl;
            }
            
            if (is_float) {
                err = clEnqueueReadBuffer(queue, cl_output_matrix, CL_TRUE, 0, ncol * ncol * sizeof(float), output_matrix_f, 1, &kernel_event, nullptr);
                
            } else {
                err = clEnqueueReadBuffer(queue, cl_output_matrix, CL_TRUE, 0, ncol * ncol * sizeof(double), output_matrix_d, 1, &kernel_event, nullptr);
            }
#endif

        } else {
            if (gTrace) {
                CERR << "Retrieve result (clEnqueueReadBufferRect):" << std::endl;
            }
            
            size_t out_region_f[] = { ncol * sizeof(float), (size_t)ncol, 1 };
            size_t out_region_d[] = { ncol * sizeof(double), (size_t)ncol, 1 };
            
            if (is_float) {
                err = clEnqueueReadBufferRect(queue, cl_output_matrix, CL_TRUE, origin, origin,
                                              out_region_f, full_ncol * sizeof(float),
                                              0, ncol * sizeof(float), 0, output_matrix_f, 1,
                                              &kernel_event, nullptr);
                
            } else {
                err = clEnqueueReadBufferRect(queue, cl_output_matrix, CL_TRUE, origin, origin,
                                              out_region_d, full_ncol * sizeof(double),
                                              0, ncol * sizeof(double), 0, output_matrix_d, 1,
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
    
    if (cl_input_matrix != nullptr) {
        clReleaseMemObject(cl_input_matrix);
        cl_input_matrix = NULL;
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

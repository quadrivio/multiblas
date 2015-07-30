//
//  crossprod_opencl.cpp
//  template
//
//  Created by michael on 4/20/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#define R_NO_REMAP  // needed if include fstream

#include "crossprod_opencl.h"
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
                     const std::vector<size_t>& work_item_sizes, int vector_size,
                     int row_tile_size, int col_tile_size, bool verbose)
{
#ifdef USE_TIMING
    steady_clock::time_point start_time = steady_clock::now();
#endif
    
//    int local_mem_per_workgroup = 0;
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

    size_t full_nrow = vector_size * ((nrow + vector_size - 1) / vector_size);

    // cheap way to find least-common-multiple; not terribly slow for small row_tile_size & col_tile_size
    size_t gcd = col_tile_size < row_tile_size ? col_tile_size : row_tile_size;
    while (gcd > 1 && col_tile_size % gcd != 0 && row_tile_size % gcd != 0) gcd--;
    size_t lcm = col_tile_size * row_tile_size / gcd;
    
    size_t full_ncol = (ncol + lcm - 1) / lcm;
    
    full_ncol = work_item_sizes[0] * ((full_ncol + work_item_sizes[0] - 1) / work_item_sizes[0]);
    full_ncol = work_item_sizes[1] * ((full_ncol + work_item_sizes[1] - 1) / work_item_sizes[1]);

    full_ncol *= lcm;
    //full_ncol = col_multiple * ((full_ncol + col_multiple - 1) / col_multiple);

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
    /*
    cl_event fill_out_event = nullptr;
    if (err == CL_SUCCESS) {
        if (is_float) {
            err = clEnqueueFillBuffer(queue, cl_output_matrix, &zero_f, sizeof(zero_f), 0,
                                      full_ncol * full_ncol * sizeof(float), 0, nullptr, &fill_out_event);
            
        } else {
            err = clEnqueueFillBuffer(queue, cl_output_matrix, &zero_d, sizeof(zero_d), 0,
                                      full_ncol * full_ncol * sizeof(double), 0, nullptr, &fill_out_event);
        }
    }
    */
    
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

        //cl_event events[] = { /*fill_out_event,*/ write_event };
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
    /*
    if (fill_out_event != nullptr) {
        clReleaseEvent(fill_out_event);
        
    } else {
        if (gTrace) {
            CERR << "fill_out_event is null " << std::endl;
        }
    }
    */
    
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

#if 0

const bool debug = false;

// =================================================================================================

CKernel::CKernel(cl_kernel kernel, size_t local_mem_per_workgroup)
{
    this->m_kernel = kernel;
    std::stringstream ss;
    ss << "-DLOCAL_MEM_PER_WORKGROUP=" << local_mem_per_workgroup;
    
    this->m_local_mem_per_workgroup = local_mem_per_workgroup;
    this->m_options = ss.str();
    this->m_max_work_group_size = 0;
    this->m_work_group_multiple = 0;
    
    this->m_work_item_sizes.push_back(1);
    this->m_work_item_sizes.push_back(1);
    this->m_work_item_sizes.push_back(1);
}

void CKernel::setKernel(cl_kernel kernel)
{
    if (m_kernel != nullptr) {
        throw "kernel in Kernel object can be set only once";
    }
    
    this->m_kernel = kernel;
}

void CKernel::setMaxWorkGroupSize(size_t max_work_group_size)
{
    this->m_max_work_group_size = max_work_group_size;

}

void CKernel::setWorkGroupMultiple(size_t work_group_multiple)
{
    this->m_work_group_multiple = work_group_multiple;
}

size_t CKernel::max_work_item_size(int dimension)
{
    size_t work_item_size = 0;
    
    if (dimension >= 0 && dimension < m_max_work_item_sizes.size()) {
        work_item_size = m_max_work_item_sizes[dimension];
    }
    
    return work_item_size;
}

std::vector<size_t> CKernel::work_item_sizes()
{
    return m_work_item_sizes;
}

void CKernel::setMaxWorkItemSizes(std::vector<size_t>& max_work_item_sizes)
{
    m_max_work_item_sizes = max_work_item_sizes;
    
    while (m_max_work_item_sizes.size() < 3) {
        m_max_work_item_sizes.push_back(1);
    }
}

void CKernel::setWorkItemSizes(std::vector<size_t>& work_item_sizes)
{
    m_work_item_sizes = work_item_sizes;
}

// =================================================================================================

//std::string crossprod_cl_d(cl_device_id device, double *inMatrix, double *outMatrix, int nrow, int ncol)
//{
//    return crossprod_cl(device, inMatrix, outMatrix, nrow, ncol, false);
//}
//
//
//std::string crossprod_cl_f(cl_device_id device, float *inMatrix, float *outMatrix, int nrow, int ncol)
//{
//    return crossprod_cl(device, inMatrix, outMatrix, nrow, ncol, true);
//}

CKernel *create_cl_kernel_from_path(cl_context context, cl_device_id device,
                                     std::string name, std::string path, int local_mem_per_workgroup,
                                     std::string *status)
{
    CERR << "local_mem_per_workgroup = " << local_mem_per_workgroup << std::endl;
    
    std::ifstream ifs(path.c_str());
    if (!ifs.is_open()) {
        if (status != nullptr) {
            *status = path + " FILE_NOT_FOUND: ";
        }
        return nullptr;
        
    } else {
        std::stringstream buffer;
        buffer << ifs.rdbuf();
        ifs.close();
        std::string source = buffer.str();

        return create_cl_kernel(context, device, name, source, local_mem_per_workgroup, status);
    }
}

CKernel *create_cl_kernel(cl_context context, cl_device_id device, std::string name,
                             std::string source, int local_mem_per_workgroup, std::string *status)
{
    std::stringstream result;
    CKernel *ckernel = new CKernel(nullptr, local_mem_per_workgroup);
    
    cl_int err = CL_SUCCESS;
    
    // program
    cl_program program = NULL;
    if (err == CL_SUCCESS) {
        if (debug) {
            result << "clCreateProgramWithSource:" << std::endl;
        }
        
        const char *src = source.c_str();
        program = clCreateProgramWithSource(context, 1, &src, NULL, &err);
    }
    
    if (err == CL_SUCCESS) {
        if (debug) {
            result << "clBuildProgram:" << std::endl;
        }
        
        err = clBuildProgram(program, 0, NULL, ckernel->options().c_str(), NULL, NULL);
        
        if (err != CL_SUCCESS) {
            const size_t log_size = 1024;
            char log[log_size];
            clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG,
                                   log_size, log, NULL);
            
            CERR << "clGetProgramBuildInfo:" << std::endl << log << std::endl;
            CERR << "Source:" << std::endl << source << std::endl;
        }
    }
    
    // kernel
    cl_kernel kernel = nullptr;
    if (err == CL_SUCCESS) {
        if (debug) {
            result << "clCreateKernel:" << std::endl;
        }
        
        kernel = clCreateKernel(program, name.c_str(), &err);
        ckernel->setKernel(kernel);
    }

    if (err == CL_SUCCESS) {
        size_t max_work_group_size = 0;
        err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
        ckernel->setMaxWorkGroupSize(max_work_group_size);
    }
    
    if (err == CL_SUCCESS) {
        size_t work_group_multiple = 0;
        err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &work_group_multiple, NULL);
        ckernel->setWorkGroupMultiple(work_group_multiple);
    }
    
    cl_uint work_item_dimensions = 0;
    if (err == CL_SUCCESS) {
        err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(work_item_dimensions), &work_item_dimensions, NULL);
    }
    
    std::vector<size_t> max_work_item_sizes;
    if (err == CL_SUCCESS) {
        max_work_item_sizes.resize(work_item_dimensions);
        err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, (size_t)work_item_dimensions * sizeof(size_t),
                              &max_work_item_sizes[0], NULL);
    }
    
    if (err == CL_SUCCESS) {
        ckernel->setMaxWorkItemSizes(max_work_item_sizes);
    }
    
    
    clReleaseProgram(program);
    program = NULL;

    if (status != NULL) {
        *status = clErrorToString(err);
    }
    
    return ckernel;
}

//std::string crossprod_cl_kernel(cl_context context, cl_device_id device, bool use_float, cl_kernel *kernel) {
//    // source
//    const char *source_f =
//    "__kernel void crossprod_f(__private int nrow, \n"
//    "                        __private int ncol, \n"
//    "                        __global float* matrix, \n"
//    "                        __global float* result) {\n"
//    "  int col = get_global_id(0);\n"
//    "  int row = get_global_id(1);\n"
//    
//    "  if (col >= row) {\n"
//    "    float sum = 0.0;\n"
//    "    int index1 = nrow * col;\n"
//    "    int index2 = nrow * row;\n"
//    "    for (int k = 0; k < nrow; k++) {\n"
//    "      sum += matrix[index1 + k] * matrix[index2 + k];\n"
//    "    }\n"
//    
//    "    result[row * ncol + col] = sum;\n"
//    "    result[col * ncol + row] = sum;\n"
//    "  }\n"
//    "}\n";
//    
//    const char *source_d =
//    "#ifdef cl_khr_fp64\n"
//    "    #pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
//    "#elif defined(cl_amd_fp64)\n"
//    "    #pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
//    "#else\n"
//    "    #error \"Double precision floating point not supported by OpenCL implementation.\"\n"
//    "#endif\n"
//    "__kernel void crossprod_d(__private int nrow, \n"
//    "                        __private int ncol, \n"
//    "                        __global double* matrix, \n"
//    "                        __global double* result) \n{"
//    "  int col = get_global_id(0);\n"
//    "  int row = get_global_id(1);\n"
//    
//    "  if (col >= row) {\n"
//    "    double sum = 0.0;\n"
//    "    int index1 = nrow * col;\n"
//    "    int index2 = nrow * row;\n"
//    "    for (int k = 0; k < nrow; k++) {\n"
//    "      sum += matrix[index1 + k] * matrix[index2 + k];\n"
//    "    }\n"
//    
//    "    result[row * ncol + col] = sum;\n"
//    "    result[col * ncol + row] = sum;\n"
//    "  }\n"
//    "}\n";
//
//    std::string status;
//    CKernel *ckernel = use_float ?
//        create_cl_kernel(context, device, "crossprod_f", source_f, 0, &status) :
//        create_cl_kernel(context, device, "crossprod_d", source_d, 0, &status);
//    
//    *kernel = ckernel->kernel();
//    
//    delete ckernel;
//    ckernel = nullptr;
//    
//    return status;
//}
//
//std::string crossprod_cl(cl_device_id device, void *inMatrix, void *outMatrix, int nrow, int ncol, bool use_float)
//{
//    std::stringstream result;
//    
//    float *input_matrix_f = (float *)inMatrix;
//    
//    float *output_matrix_f = (float *)outMatrix;
//    
//    double *input_matrix_d = (double *)inMatrix;
//    
//    double *output_matrix_d = (double *)outMatrix;
//    
//    if (debug) {
//        result << "crossprod_cl( " << (use_float ? "FLOAT" : "DOUBLE") << ")" << std::endl << std::endl;
//    }
//    
//    cl_int err = CL_SUCCESS;
//    
//    // context
//    cl_context context = NULL;
//    if (err == CL_SUCCESS) {
//        if (debug) {
//            result << "clCreateContext:" << std::endl;
//        }
//        
//        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
//    }
//    
//    // source
//    const char *source_f =
//    "__kernel void crossprod_f(__private int nrow, \n"
//    "                        __private int ncol, \n"
//    "                        __global float* matrix, \n"
//    "                        __global float* result) {\n"
//    "  int col = get_global_id(0);\n"
//    "  int row = get_global_id(1);\n"
//    
//    "  if (col >= row) {\n"
//    "    float sum = 0.0;\n"
//    "    int index1 = nrow * col;\n"
//    "    int index2 = nrow * row;\n"
//    "    for (int k = 0; k < nrow; k++) {\n"
//    "      sum += matrix[index1 + k] * matrix[index2 + k];\n"
//    "    }\n"
//    
//    "    result[row * ncol + col] = sum;\n"
//    "    result[col * ncol + row] = sum;\n"
//    "  }\n"
//    "}\n";
//    
//    const char *source_d =
//    "#ifdef cl_khr_fp64\n"
//    "    #pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
//    "#elif defined(cl_amd_fp64)\n"
//    "    #pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
//    "#else\n"
//    "    #error \"Double precision floating point not supported by OpenCL implementation.\"\n"
//    "#endif\n"
//    "__kernel void crossprod_d(__private int nrow, \n"
//    "                        __private int ncol, \n"
//    "                        __global double* matrix, \n"
//    "                        __global double* result) \n{"
//    "  int col = get_global_id(0);\n"
//    "  int row = get_global_id(1);\n"
//    
//    "  if (col >= row) {\n"
//    "    double sum = 0.0;\n"
//    "    int index1 = nrow * col;\n"
//    "    int index2 = nrow * row;\n"
//    "    for (int k = 0; k < nrow; k++) {\n"
//    "      sum += matrix[index1 + k] * matrix[index2 + k];\n"
//    "    }\n"
//    
//    "    result[row * ncol + col] = sum;\n"
//    "    result[col * ncol + row] = sum;\n"
//    "  }\n"
//    "}\n";
//    
//    // program
//    cl_program program = NULL;
//    if (err == CL_SUCCESS) {
//        if (debug) {
//            result << "clCreateProgramWithSource:" << std::endl;
//        }
//        
//        program = clCreateProgramWithSource(context, 1, use_float ? &source_f : &source_d, NULL, &err);
//    }
//    
//    if (err == CL_SUCCESS) {
//        if (debug) {
//            result << "clBuildProgram:" << std::endl;
//        }
//        
//        err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
//        
//        if (debug && err != CL_SUCCESS) {
//            const size_t log_size = 1024;
//            char log[log_size];
//            clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG,
//                                   log_size, log, NULL);
//            
//            result << "clGetProgramBuildInfo:" << std::endl << log << std::endl;
//            result << "Source:" << std::endl << (use_float ? source_f : source_d) << std::endl;
//        }
//    }
//    
//    // kernel
//    cl_kernel kernel = NULL;
//    if (err == CL_SUCCESS) {
//        if (debug) {
//            result << "clCreateKernel:" << std::endl;
//        }
//        
//        kernel = clCreateKernel(program, use_float ? "crossprod_f" : "crossprod_d", &err);
//    }
//    
//    clReleaseProgram(program);
//    program = NULL;
//    
//    // queue
//    cl_command_queue queue = NULL;
//    if (err == CL_SUCCESS) {
//#ifdef CL_VERSION_2_0
//        if (debug) {
//            result << "clCreateCommandQueueWithProperties:" << std::endl;
//        }
//        
//        queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
//        
//#else
//        if (debug) {
//            result << "clCreateCommandQueue:" << std::endl;
//        }
//        
//        queue = clCreateCommandQueue(context, device, 0, &err);
//#endif
//    }
//    
//    // buffers
//    cl_mem cl_input_matrix = NULL;
//    if (err == CL_SUCCESS) {
//        if (debug) {
//            result << "clCreateBuffer cl_input_matrix:" << std::endl;
//        }
//        
//        if (use_float) {
//            cl_input_matrix = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
//                                             nrow * ncol * sizeof(float), input_matrix_f, &err);
//            
//        } else {
//            cl_input_matrix = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
//                                             nrow * ncol * sizeof(double), input_matrix_d, &err);
//        }
//    }
//    
//    cl_mem cl_output_matrix = NULL;
//    if (err == CL_SUCCESS) {
//        if (debug) {
//            result << "clCreateBuffer cl_output_vector:" << std::endl;
//        }
//        
//        cl_output_matrix = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
//                                          use_float ? ncol * ncol * sizeof(float) :
//                                          ncol * ncol * sizeof(double), NULL, &err);
//    }
//    
//    // initiate calculation
//    if (err == CL_SUCCESS) {
//        if (debug) {
//            result << "Initiate calculation:" << std::endl;
//        }
//        
//        cl_int cl_nrow = nrow;
//        cl_int cl_ncol = ncol;
//        clSetKernelArg(kernel, 0, sizeof(cl_nrow), &cl_nrow);
//        clSetKernelArg(kernel, 1, sizeof(cl_ncol), &cl_ncol);
//        clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_input_matrix);
//        clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_output_matrix);
//        
//        cl_uint work_dim = 2;
//        size_t global_work_size[] = { (size_t)ncol, (size_t)ncol };
//        
//        clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_size, NULL, 0, NULL, NULL);
//    }
//    
//    // retrieve result
//    if (err == CL_SUCCESS) {
//        if (debug) {
//            result << "Retrieve result:" << std::endl;
//        }
//        
//        if (use_float) {
//            clEnqueueReadBuffer(queue, cl_output_matrix, CL_TRUE, 0, ncol * ncol * sizeof(float), output_matrix_f, 0, NULL, NULL);
//            
//        } else {
//            clEnqueueReadBuffer(queue, cl_output_matrix, CL_TRUE, 0, ncol * ncol * sizeof(double), output_matrix_d, 0, NULL, NULL);
//        }
//    }
//    
//    std::string err_str = clErrorToString(err);
//    result << std::endl << err_str << std::endl;
//    
//    // cleanup
//    clReleaseMemObject(cl_output_matrix);
//    cl_output_matrix = NULL;
//    
//    clReleaseMemObject(cl_input_matrix);
//    cl_input_matrix = NULL;
//    
//    clReleaseCommandQueue(queue);
//    queue = NULL;
//    
//    clReleaseKernel(kernel);
//    kernel = NULL;
//    
//    //    clReleaseProgram(program);
//    //    program = NULL;
//    
//    clReleaseContext(context);
//    context = NULL;
//    
//    if (debug) {
//        CERR << result.str();
//    }
//    
//    return clErrorToString(err);
//}

std::string crossprod_cl_ckq_f(cl_context context, CKernel *ckernel, cl_command_queue queue,
                               void *inMatrix, void *outMatrix, int nrow, int ncol,
                               int row_tile_size, int col_tile_size, bool use_rect)
{
    return crossprod_cl_ckq(context, ckernel, queue, inMatrix, outMatrix, nrow, ncol, true, row_tile_size, col_tile_size, use_rect);
}

std::string crossprod_cl_ckq_d(cl_context context, CKernel *ckernel, cl_command_queue queue,
                               void *inMatrix, void *outMatrix, int nrow, int ncol,
                               int row_tile_size, int col_tile_size, bool use_rect)
{
    return crossprod_cl_ckq(context, ckernel, queue, inMatrix, outMatrix, nrow, ncol, false, row_tile_size, col_tile_size, use_rect);
}

std::string crossprod_cl_ckq(cl_context context, CKernel *ckernel, cl_command_queue queue,
                             void *inMatrix, void *outMatrix, int nrow, int ncol, bool use_float,
                             int row_tile_size, int col_tile_size, bool use_rect)
{
    std::stringstream result;
    
    float *input_matrix_f = (float *)inMatrix;
    
    float *output_matrix_f = (float *)outMatrix;
    
    double *input_matrix_d = (double *)inMatrix;
    
    double *output_matrix_d = (double *)outMatrix;
    
    if (debug) {
        result << "crossprod_cl_ckq( " << (use_float ? "FLOAT" : "DOUBLE") << ")" << std::endl << std::endl;
    }
    
    cl_kernel kernel = ckernel->kernel();
    
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
        if (ckernel->local_mem_per_workgroup() > 0) {
            clSetKernelArg(kernel, 5, ckernel->local_mem_per_workgroup(), NULL);
        } else {
            clSetKernelArg(kernel, 5, sizeof(cl_mem), NULL);
        }
        clSetKernelArg(kernel, 6, sizeof(cl_mem), &cl_output_matrix);

        std::vector<size_t> work_item_sizes = ckernel->work_item_sizes();

        cl_uint work_dim = 3;
        size_t work_rows = (size_t)ncol / row_tile_size;
        if (use_rect) {
            work_rows = (work_rows / 2) + 1;
            work_rows = work_item_sizes[1] * ((work_rows + work_item_sizes[1] - 1) / work_item_sizes[1]);
        }
        
        size_t global_work_sizes[] = { (size_t)ncol / col_tile_size, work_rows, 1 };

//        std::cout << "work_item_sizes = (" << work_item_sizes[0] << ", " <<
//            work_item_sizes[1] << ", " << work_item_sizes[2] << ")" << std::endl;

        CERR << "global_work_sizes = (" << global_work_sizes[0] << ", " <<
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
            
            if (false) {
                Profile enqueue_profile = getProfileTimes(event);
                CERR << "clEnqueueNDRangeKernel: " << enqueue_profile.queued << " Q msec; " <<
                enqueue_profile.pending << " P msec; " << enqueue_profile.exec << " E msec" << std::endl;
            }

            clReleaseEvent(event);
   
        } else {
            result << "event is null " << std::endl;
        }
        
        if (event2 != nullptr) {
            clWaitForEvents(1, &event2);
            
            if (false) {
                Profile enqueue_read = getProfileTimes(event2);
                CERR << "clEnqueueReadBuffer: " << enqueue_read.queued << " Q msec; " <<
                enqueue_read.pending << " P msec; " << enqueue_read.exec << " E msec" << std::endl;
            }

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

// =================================================================================================

Profile getProfileTimes(const cl_event event)
{
    Profile profile;
    
    if (event == nullptr) {
        profile.queued = -1;
        profile.pending = -1;
        profile.exec = -1;
        
    } else {
        cl_ulong queued_time;
        cl_ulong submit_time;
        cl_ulong start_time;
        cl_ulong end_time;
        
        size_t foo;
        
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, &foo);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
        
        profile.queued = (submit_time - queued_time) / 1000000;
        profile.pending = (start_time - submit_time) / 1000000;
        profile.exec = (end_time - start_time) / 1000000;
    }
    
    return profile;
}

#endif

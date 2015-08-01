//
//  crossprod_opencl.h
//  template
//
//  Created by michael on 4/20/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#ifndef __template__crossprod__
#define __template__crossprod__

#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <vector>

cl_int opencl_calc_x(cl_context context, cl_kernel kernel_f, cl_kernel kernel_d, bool is_float,
                     cl_command_queue queue, void *inMatrix, void *outMatrix, size_t nrow, size_t ncol,
                     const std::vector<size_t>& work_item_sizes, int vector_size, int row_multiple,
                     int row_tile_size, int col_tile_size, bool verbose);


#if 0

#include <string>
#include <vector>

struct Profile {
    unsigned long long queued;
    unsigned long long pending;
    unsigned long long exec;
};
typedef struct Profile Profile;

class CKernel {
protected:
    cl_kernel m_kernel;
    std::string m_options;
    size_t m_local_mem_per_workgroup;
    size_t m_max_work_group_size;
    size_t m_work_group_multiple;
    std::vector<size_t> m_max_work_item_sizes;
    std::vector<size_t> m_work_item_sizes;

public:
    cl_kernel kernel() { return m_kernel; };
    std::string options() { return m_options; };
    size_t local_mem_per_workgroup() { return m_local_mem_per_workgroup; };
    size_t max_work_item_size(int dimension);

    std::vector<size_t> work_item_sizes();
    void setWorkItemSizes(std::vector<size_t>& work_item_sizes);

    // TODO combine into one one-time call
    void setKernel(cl_kernel kernel);
    void setMaxWorkGroupSize(size_t max_work_group_size);
    void setWorkGroupMultiple(size_t work_group_multiple);
    void setMaxWorkItemSizes(std::vector<size_t>& max_work_item_sizes);
    
//    CKernel(size_t local_mem_per_workgroup);
    CKernel(cl_kernel kernel, size_t local_mem_per_workgroup);
};

Profile getProfileTimes(const cl_event event);

CKernel *create_cl_kernel_from_path(cl_context context, cl_device_id device,
                                     std::string name, std::string path, int local_mem_per_workgroup,
                                     std::string *status);

CKernel *create_cl_kernel(cl_context context, cl_device_id device, std::string name,
                           std::string source, int local_mem_per_workgroup, std::string *status);

//std::string crossprod_cl_d(cl_device_id device, double *inMatrix, double *outMatrix, int nrow, int ncol);
//std::string crossprod_cl_f(cl_device_id device, float *inMatrix, float *outMatrix, int nrow, int ncol);
//std::string crossprod_cl(cl_device_id device, void *inMatrix, void *outMatrix, int nrow, int ncol, bool use_float);
//
//std::string crossprod_cl_kernel(cl_context context, cl_device_id device, bool use_float, cl_kernel *kernel);

std::string crossprod_cl_ckq_d(cl_context context, CKernel *ckernel, cl_command_queue queue,
                               void *inMatrix, void *outMatrix, int nrow, int ncol,
                               int row_tile_size, int col_tile_size, bool use_rect = false);
std::string crossprod_cl_ckq_f(cl_context context, CKernel *ckernel, cl_command_queue queue,
                               void *inMatrix, void *outMatrix, int nrow, int ncol,
                               int row_tile_size, int col_tile_size, bool use_rect = false);
std::string crossprod_cl_ckq(cl_context context, CKernel *ckernel, cl_command_queue queue,
                             void *inMatrix, void *outMatrix, int nrow, int ncol, bool use_float,
                             int row_tile_size, int col_tile_size, bool use_rect);
#endif

#endif /* defined(__template__crossprod__) */

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

#endif /* defined(__template__crossprod__) */

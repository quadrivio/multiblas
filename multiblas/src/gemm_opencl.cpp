//
//  gemm_opencl.cpp
//  multiBLAS.XC
//
//  Created by michael on 7/23/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#include "gemm_opencl.h"

cl_int opencl_calc_gemm(cl_context context, cl_kernel kernel_f, cl_kernel kernel_d, bool is_float,
                        cl_command_queue queue,
                        const void *inMatrixA, int nrowA, int ncolA, bool transposeA,
                        const void *inMatrixB, int nrowB, int ncolB, bool transposeB,
                        double alpha, double beta, void *outMatrix,
                        const std::vector<size_t>& work_item_sizes, int row_multiple, int col_multiple,
                        int row_tile_size, int col_tile_size, bool verbose)
{
    return CL_INVALID_VALUE;
}
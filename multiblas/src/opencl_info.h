//
//  opencl_info.h
//  template
//
//  Created by michael on 4/10/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#ifndef __template__opencl_info__
#define __template__opencl_info__

#include <cstddef>
#include <string>
#include <vector>
#include <limits.h>

#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

std::string report();

std::string getPlatformInfoString(cl_platform_id id, cl_platform_info param_name, cl_int *error = NULL);
std::string getDeviceInfoString(cl_device_id id, cl_device_info param_name, cl_int *error = NULL);
std::string getCommandQueueInfoString(cl_command_queue id, cl_command_queue_info param_name, cl_int *error = NULL);
std::string getKernelWorkGroupInfoString(cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info param_name, cl_int *error);

bool isContextValid(cl_context context);

std::string clErrorToString(cl_int error);

const uint32_t UNDEFINED_INFO = UINT_MAX;

cl_platform_info clStringToPlatformInfo(std::string name);
cl_device_info clStringToDeviceInfo(std::string name);

void getFullSizes(size_t& full_rowsA, size_t& full_colsA, size_t& full_colsB,
                  size_t rowsA,  size_t colsA,  size_t colsB,
                  size_t vector_size, size_t row_multiple, size_t row_tile_size, size_t col_tile_size,
                  const std::vector<size_t>& work_item_sizes);

void getFullSizes_atia(bool transposeA, bool transposeB,
                       size_t& full_rowsA, size_t& full_colsA, size_t& full_rowsB, size_t& full_colsB,
                       size_t rowsA,  size_t colsA, size_t rowsB, size_t colsB,
                       size_t vector_size, size_t row_multiple, size_t row_tile_size, size_t col_tile_size,
                       const std::vector<size_t>& work_item_sizes);

#endif /* defined(__template__opencl_info__) */

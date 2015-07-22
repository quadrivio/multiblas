//
//  opencl_info.h
//  template
//
//  Created by michael on 4/10/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#ifndef __template__opencl_info__
#define __template__opencl_info__

#include <string>
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

std::string clErrorToString(cl_int error);

const uint32_t UNDEFINED_INFO = UINT_MAX;

cl_platform_info clStringToPlatformInfo(std::string name);
cl_device_info clStringToDeviceInfo(std::string name);

#endif /* defined(__template__opencl_info__) */

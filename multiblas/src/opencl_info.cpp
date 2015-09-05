//
//  opencl_info.cpp
//  template
//
//  Created by michael on 4/10/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#include "opencl_info.h"
#include "shim.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

using namespace std;

// ========== Globals ==============================================================================

extern bool gTrace;    // for debugging

// ========== Functions ============================================================================

string report()
{
    stringstream result;

    result << "Begin" << endl << endl;
    
    cl_int err;
    
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    
    vector<cl_platform_id> platforms;
    if (err == CL_SUCCESS) {
        platforms.resize(num_platforms);
        err = clGetPlatformIDs(num_platforms, &platforms[0], &num_platforms);
    }
    
    if (err == CL_SUCCESS) {
        vector<cl_platform_id>::iterator platform_iter = platforms.begin();
        int platform_index = 1;
        while (platform_iter != platforms.end() && err == CL_SUCCESS) {
            result << "----- Platform " << platform_index++ << " -----" << endl;
            result << "Profile: " << getPlatformInfoString(*platform_iter, CL_PLATFORM_PROFILE) << endl;
            result << "Version: " << getPlatformInfoString(*platform_iter, CL_PLATFORM_VERSION) << endl;
            result << "Name: " << getPlatformInfoString(*platform_iter, CL_PLATFORM_NAME) << endl;
            result << "Vendor: " << getPlatformInfoString(*platform_iter, CL_PLATFORM_VENDOR) << endl;
            result << "Extensions: " << getPlatformInfoString(*platform_iter, CL_PLATFORM_EXTENSIONS) << endl;
            result << endl;
            
            cl_uint num_devices;
            err = clGetDeviceIDs(*platform_iter, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
            
            vector<cl_device_id> devices;
            if (err == CL_SUCCESS) {
                devices.resize(num_devices);
                err = clGetDeviceIDs(*platform_iter, CL_DEVICE_TYPE_ALL, num_devices, &devices[0], NULL);
            }
            
            if (err == CL_SUCCESS) {
                vector<cl_device_id>::iterator device_iter = devices.begin();
                int device_index = 1;
                while (device_iter != devices.end() && err == CL_SUCCESS) {
                    result << "----- Device " << device_index++ << " -----" << endl;
                    result << "Name: " << getDeviceInfoString(*device_iter, CL_DEVICE_NAME) << endl;
                    result << "Profile: " << getDeviceInfoString(*device_iter, CL_DEVICE_PROFILE) << endl;
                    result << "Vendor: " << getDeviceInfoString(*device_iter, CL_DEVICE_VENDOR) << endl;
                    result << "Vendor ID: " << getDeviceInfoString(*device_iter, CL_DEVICE_VENDOR_ID) << endl;
                    result << "Device Version: " << getDeviceInfoString(*device_iter, CL_DEVICE_VERSION) << endl;
                    result << "Driver Version: " << getDeviceInfoString(*device_iter, CL_DRIVER_VERSION) << endl;
                    result << "Max Clock MHz: " << getDeviceInfoString(*device_iter, CL_DEVICE_MAX_CLOCK_FREQUENCY) << endl;
                    result << "Max Compute Units: " << getDeviceInfoString(*device_iter, CL_DEVICE_MAX_COMPUTE_UNITS) << endl;
                    result << "Device Max Samplers: " << getDeviceInfoString(*device_iter, CL_DEVICE_MAX_SAMPLERS) << endl;
                    result << "Device Max Work Item Sizes: " << getDeviceInfoString(*device_iter, CL_DEVICE_MAX_WORK_ITEM_SIZES) << endl;
                    result << "Device Max Work Group Size: " << getDeviceInfoString(*device_iter, CL_DEVICE_MAX_WORK_GROUP_SIZE) << endl;
                    result << "Device Max Mem Alloc Size: " << getDeviceInfoString(*device_iter, CL_DEVICE_MAX_MEM_ALLOC_SIZE) << endl;
                    result << "Device Global Mem Size: " << getDeviceInfoString(*device_iter, CL_DEVICE_GLOBAL_MEM_SIZE) << endl;
                    result << "Device Local Mem Type: " << getDeviceInfoString(*device_iter, CL_DEVICE_LOCAL_MEM_TYPE) << endl;
                    result << "Device Local Mem Size: " << getDeviceInfoString(*device_iter, CL_DEVICE_LOCAL_MEM_SIZE) << endl;
                    result << "Double FP Config: " << getDeviceInfoString(*device_iter, CL_DEVICE_DOUBLE_FP_CONFIG) << endl;
                    result << "Preferred Char Vector Width: " << getDeviceInfoString(*device_iter, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR) << endl;
                    result << "Preferred Float Vector Width: " << getDeviceInfoString(*device_iter, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT) << endl;
                    result << "Preferred Double Vector Width: " << getDeviceInfoString(*device_iter, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE) << endl;
                    result << "Extensions: " << getDeviceInfoString(*device_iter, CL_DEVICE_EXTENSIONS) << endl;
                    result << endl;
                    
                    ++device_iter;
                }
            }
            
            ++platform_iter;
        }
    }
    
    return result.str();
}


string getPlatformInfoString(cl_platform_id id, cl_platform_info param_name, cl_int *error)
{
    string result("");
    cl_int err;
    
    size_t param_size;
    err = clGetPlatformInfo(id, param_name, 0, NULL, &param_size);
    
    char *buffer = NULL;
    if (err == CL_SUCCESS) {
        buffer = (char *)malloc(param_size);
        if (buffer == NULL) {
            err = CL_OUT_OF_HOST_MEMORY;
        }
    }
    
    if (err == CL_SUCCESS) {
        err = clGetPlatformInfo(id, param_name, param_size, buffer, NULL);
    }
    
    if (err == CL_SUCCESS && param_size > 1) {
        result.assign(buffer, param_size - 1);
    }
    
    if (buffer != NULL) {
        free(buffer);
        buffer = NULL;
    }
    
    if (error != NULL) {
        *error = err;
    }
    
    return(result);
}

string getDeviceInfoString(cl_device_id id, cl_device_info param_name, cl_int *error)
{
    stringstream result;
    cl_int err;
    
    size_t param_size;
    err = clGetDeviceInfo(id, param_name, 0, NULL, &param_size);
    
    void *buffer = NULL;
    if (err == CL_SUCCESS) {
        buffer = malloc(param_size);
        if (buffer == NULL) {
            err = CL_OUT_OF_HOST_MEMORY;
        }
    }
    
    if (err == CL_SUCCESS) {
        err = clGetDeviceInfo(id, param_name, param_size, buffer, NULL);
    }
    
    if (err == CL_SUCCESS) {
        switch (param_name) {
            case CL_DEVICE_EXTENSIONS:
            case CL_DEVICE_NAME:
            case CL_DEVICE_PROFILE:
            case CL_DEVICE_VENDOR:
            case CL_DEVICE_VERSION:
            case CL_DRIVER_VERSION:
                if (param_size > 1) {
                    result << (const char *)buffer;
                }
                break;
                
            case CL_DEVICE_ADDRESS_BITS:
            case CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:
            case CL_DEVICE_MAX_CLOCK_FREQUENCY:
            case CL_DEVICE_MAX_COMPUTE_UNITS:
            case CL_DEVICE_MAX_CONSTANT_ARGS:
            case CL_DEVICE_MAX_READ_IMAGE_ARGS:
            case CL_DEVICE_MAX_SAMPLERS:
            case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:
            case CL_DEVICE_MAX_WRITE_IMAGE_ARGS:
            case CL_DEVICE_MEM_BASE_ADDR_ALIGN:
            case CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE:
            case CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:
            case CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:
            case CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:
            case CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:
            case CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:
            case CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:
                result << *(cl_uint *)buffer;
                break;
                
            case CL_DEVICE_VENDOR_ID:
                result << "0x" << hex << uppercase << setfill('0') << setw(8) << *(cl_uint *)buffer << dec;
                break;
                
            case CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:
            case CL_DEVICE_GLOBAL_MEM_SIZE:
            case CL_DEVICE_LOCAL_MEM_SIZE:
            case CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:
            case CL_DEVICE_MAX_MEM_ALLOC_SIZE:
            {
                cl_ulong value = *(cl_ulong *)buffer;
                if (value >= 1024 * 1024) {
                    result << (value / (1024 * 1024)) << " MiB (" << value << ")";
                    
                } else if (value >= 1024) {
                    result << (value / 1024) << " kiB (" << value << ")";
                    
                } else {
                    result << value;
                }
            }
                break;
                
            case CL_DEVICE_AVAILABLE:
            case CL_DEVICE_COMPILER_AVAILABLE:
            case CL_DEVICE_ENDIAN_LITTLE:
            case CL_DEVICE_ERROR_CORRECTION_SUPPORT:
            case CL_DEVICE_IMAGE_SUPPORT:
                result << (*(cl_bool *)buffer != CL_FALSE);
                break;
                
            case CL_DEVICE_IMAGE2D_MAX_HEIGHT:
            case CL_DEVICE_IMAGE2D_MAX_WIDTH:
            case CL_DEVICE_IMAGE3D_MAX_DEPTH:
            case CL_DEVICE_IMAGE3D_MAX_HEIGHT:
            case CL_DEVICE_IMAGE3D_MAX_WIDTH:
            case CL_DEVICE_MAX_PARAMETER_SIZE:
            case CL_DEVICE_MAX_WORK_GROUP_SIZE:
            case CL_DEVICE_PROFILING_TIMER_RESOLUTION:
                result << *(size_t *)buffer;
                break;
                
            case CL_DEVICE_DOUBLE_FP_CONFIG:
            {
                cl_device_fp_config config = *(cl_device_fp_config *)buffer;
                if ((config & CL_FP_DENORM) != 0) result << "CL_FP_DENORM ";
                if ((config & CL_FP_INF_NAN) != 0) result << "CL_FP_INF_NAN ";
                if ((config & CL_FP_ROUND_TO_NEAREST) != 0) result << "CL_FP_ROUND_TO_NEAREST ";
                if ((config & CL_FP_ROUND_TO_ZERO) != 0) result << "CL_FP_ROUND_TO_ZERO ";
                if ((config & CL_FP_ROUND_TO_INF) != 0) result << "CL_FP_ROUND_TO_INF ";
                if ((config & CL_FP_FMA) != 0) result << "CL_FP_FMA ";
            }
                break;
                
            case CL_DEVICE_EXECUTION_CAPABILITIES:
            {
                cl_device_exec_capabilities capabilities = *(cl_device_exec_capabilities *)buffer;
                if ((capabilities & CL_EXEC_KERNEL) != 0) result << "CL_EXEC_KERNEL ";
                if ((capabilities & CL_EXEC_NATIVE_KERNEL) != 0) result << "CL_EXEC_NATIVE_KERNEL ";
            }
                break;
                
            case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:
                switch (*(cl_device_mem_cache_type *)buffer) {
                    case CL_NONE:               result << "CL_NONE";
                    case CL_READ_ONLY_CACHE:    result << "CL_READ_ONLY_CACHE";
                    case CL_READ_WRITE_CACHE:   result << "CL_READ_WRITE_CACHE";
                }
                break;
                
            case CL_DEVICE_HALF_FP_CONFIG:
            {
                cl_device_fp_config config = *(cl_device_fp_config *)buffer;
                if ((config & CL_FP_DENORM) != 0) result << "CL_FP_DENORM ";
                if ((config & CL_FP_INF_NAN) != 0) result << "CL_FP_INF_NAN ";
                if ((config & CL_FP_ROUND_TO_NEAREST) != 0) result << "CL_FP_ROUND_TO_NEAREST ";
                if ((config & CL_FP_ROUND_TO_ZERO) != 0) result << "CL_FP_ROUND_TO_ZERO ";
                if ((config & CL_FP_ROUND_TO_INF) != 0) result << "CL_FP_ROUND_TO_INF ";
                if ((config & CL_FP_FMA) != 0) result << "CL_FP_FMA ";
            }
                break;
                
            case CL_DEVICE_LOCAL_MEM_TYPE:
                switch (*(cl_device_local_mem_type *)buffer) {
                    case CL_LOCAL:      result << "CL_LOCAL";   break;
                    case CL_GLOBAL:     result << "CL_GLOBAL";   break;
                }
                break;
                
            case CL_DEVICE_MAX_WORK_ITEM_SIZES:
            {
                size_t *items = (size_t *)buffer;
                uint item_count; // = (int)(param_size / sizeof(size_t));
                err = clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(uint), &item_count, NULL);
                
                for (uint k = 0; k < item_count && err == CL_SUCCESS; k++) {
                    if (k > 0) result << ", ";
                    result << items[k];
                }
            }
                break;
                
            case CL_DEVICE_QUEUE_PROPERTIES:
            {
                cl_command_queue_properties properties = *(cl_command_queue_properties *)buffer;
                if ((properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0) {
                    result << "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ";
                }
                
                if ((properties & CL_QUEUE_PROFILING_ENABLE) != 0) result << "CL_QUEUE_PROFILING_ENABLE ";
            }
                break;
                
            case CL_DEVICE_SINGLE_FP_CONFIG:
            {
                cl_device_fp_config config = *(cl_device_fp_config *)buffer;
                if ((config & CL_FP_DENORM) != 0) result << "CL_FP_DENORM ";
                if ((config & CL_FP_INF_NAN) != 0) result << "CL_FP_INF_NAN ";
                if ((config & CL_FP_ROUND_TO_NEAREST) != 0) result << "CL_FP_ROUND_TO_NEAREST ";
                if ((config & CL_FP_ROUND_TO_ZERO) != 0) result << "CL_FP_ROUND_TO_ZERO ";
                if ((config & CL_FP_ROUND_TO_INF) != 0) result << "CL_FP_ROUND_TO_INF ";
                if ((config & CL_FP_FMA) != 0) result << "CL_FP_FMA ";
            }
                break;
                
            case CL_DEVICE_TYPE:
                switch (*(cl_device_type *)buffer) {
                    case CL_DEVICE_TYPE_CPU:            result << "CL_DEVICE_TYPE_CPU";         break;
                    case CL_DEVICE_TYPE_GPU:            result << "CL_DEVICE_TYPE_GPU";         break;
                    case CL_DEVICE_TYPE_ACCELERATOR:    result << "CL_DEVICE_TYPE_ACCELERATOR"; break;
                    case CL_DEVICE_TYPE_DEFAULT:        result << "CL_DEVICE_TYPE_DEFAULT";     break;
                }
                break;
                
            default:
                err = CL_INVALID_VALUE;
                break;
        }
    }
    
    if (buffer != NULL) {
        free(buffer);
        buffer = NULL;
    }
    
    if (error != NULL) {
        *error = err;
    }
    
    return(result.str());
}

std::string getCommandQueueInfoString(cl_command_queue id, cl_command_queue_info param_name, cl_int *error)
{
    stringstream result;
    cl_int err;
    
    size_t param_size;
    err = clGetCommandQueueInfo(id, param_name, 0, NULL, &param_size);
    
    void *buffer = NULL;
    if (err == CL_SUCCESS) {
        buffer = malloc(param_size);
        if (buffer == NULL) {
            err = CL_OUT_OF_HOST_MEMORY;
        }
    }
    
    if (err == CL_SUCCESS) {
        err = clGetCommandQueueInfo(id, param_name, param_size, buffer, NULL);
    }
    
    if (err != CL_SUCCESS) {
        result << clErrorToString(err);
        
    } else {
        switch (param_name) {
            case CL_QUEUE_CONTEXT:
            {
                cl_context *context = (cl_context *)buffer;
                result << hex << (unsigned long long)(void *)(*context) << dec;
            }
            break;
            
            case CL_QUEUE_DEVICE:
            {
                cl_device_id *device = (cl_device_id *)buffer;
                result << hex << (unsigned long long)(void *)(*device) << dec;
            }
            break;
            
            case CL_QUEUE_REFERENCE_COUNT:
            {
                cl_uint *count = (cl_uint *)buffer;
                result << *count;
            }
            break;
            
            case CL_QUEUE_PROPERTIES:
            {
                cl_command_queue_properties properties = *(cl_command_queue_properties *)buffer;
                if ((properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0) result << "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ";
                if ((properties & CL_QUEUE_PROFILING_ENABLE) != 0) result << "CL_QUEUE_PROFILING_ENABLE ";
            }
            break;
            
            default:
            err = CL_INVALID_VALUE;
            break;
        }
    }
    
    if (buffer != NULL) {
        free(buffer);
        buffer = NULL;
    }
    
    if (error != NULL) {
        *error = err;
    }
    
    return(result.str());
}

std::string getKernelWorkGroupInfoString(cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info param_name, cl_int *error)
{
    stringstream result;
    cl_int err;
    
    size_t param_size;
    err = clGetKernelWorkGroupInfo(kernel, device, param_name, 0, NULL, &param_size);
    
    void *buffer = NULL;
    if (err == CL_SUCCESS) {
        buffer = malloc(param_size);
        if (buffer == NULL) {
            err = CL_OUT_OF_HOST_MEMORY;
        }
    }
    
    if (err == CL_SUCCESS) {
        err = clGetKernelWorkGroupInfo(kernel, device, param_name, param_size, buffer, NULL);
    }
    
    if (err != CL_SUCCESS) {
        result << clErrorToString(err);
        
    } else {
        switch (param_name) {
            case CL_KERNEL_WORK_GROUP_SIZE:
            case CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE:
            {
                result << *(size_t *)buffer;
            }
            break;
            
            case CL_KERNEL_COMPILE_WORK_GROUP_SIZE:
            {
                size_t *sizes = (size_t *)buffer;
                result << "[" << sizes[0] << ", " << sizes[1] << ", " << sizes[2] << "]";
            }
            break;
            
            case CL_KERNEL_LOCAL_MEM_SIZE:
            case CL_KERNEL_PRIVATE_MEM_SIZE:
            {
                cl_ulong *size = (cl_ulong *)buffer;
                result << *size;
            }
            break;
            default:
            err = CL_INVALID_VALUE;
            break;
        }
    }
    
    if (buffer != NULL) {
        free(buffer);
        buffer = NULL;
    }
    
    if (error != NULL) {
        *error = err;
    }
    
    return(result.str());
}

std::string clErrorToString(cl_int error)
{
    std::string err_str;
    switch (error) {
        case CL_SUCCESS:                                    err_str = "CL_SUCCESS";                                     break;
            
        case CL_DEVICE_NOT_FOUND:                           err_str = "CL_DEVICE_NOT_FOUND";                            break;
        case CL_DEVICE_NOT_AVAILABLE:                       err_str = "CL_DEVICE_NOT_AVAILABLE";                        break;
        case CL_COMPILER_NOT_AVAILABLE:                     err_str = "CL_COMPILER_NOT_AVAILABLE";                      break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:              err_str = "CL_MEM_OBJECT_ALLOCATION_FAILURE";               break;
        case CL_OUT_OF_RESOURCES:                           err_str = "CL_OUT_OF_RESOURCES";                            break;
        case CL_OUT_OF_HOST_MEMORY:                         err_str = "CL_OUT_OF_HOST_MEMORY";                          break;
        case CL_PROFILING_INFO_NOT_AVAILABLE:               err_str = "CL_PROFILING_INFO_NOT_AVAILABLE";                break;
        case CL_MEM_COPY_OVERLAP:                           err_str = "CL_MEM_COPY_OVERLAP";                            break;
        case CL_IMAGE_FORMAT_MISMATCH:                      err_str = "CL_IMAGE_FORMAT_MISMATCH";                       break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:                 err_str = "CL_IMAGE_FORMAT_NOT_SUPPORTED";                  break;
        case CL_BUILD_PROGRAM_FAILURE:                      err_str = "CL_BUILD_PROGRAM_FAILURE";                       break;
        case CL_MAP_FAILURE:                                err_str = "CL_MAP_FAILURE";                                 break;
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:               err_str = "CL_MISALIGNED_SUB_BUFFER_OFFSET";                break;
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:  err_str = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";   break;
        case CL_COMPILE_PROGRAM_FAILURE:                    err_str = "CL_COMPILE_PROGRAM_FAILURE";                     break;
        case CL_LINKER_NOT_AVAILABLE:                       err_str = "CL_LINKER_NOT_AVAILABLE";                        break;
        case CL_LINK_PROGRAM_FAILURE:                       err_str = "CL_LINK_PROGRAM_FAILURE";                        break;
        case CL_DEVICE_PARTITION_FAILED:                    err_str = "CL_DEVICE_PARTITION_FAILED";                     break;
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:              err_str = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";               break;
            
        case CL_INVALID_VALUE:                              err_str = "CL_INVALID_VALUE";                               break;
        case CL_INVALID_DEVICE_TYPE:                        err_str = "CL_INVALID_DEVICE_TYPE";                         break;
        case CL_INVALID_PLATFORM:                           err_str = "CL_INVALID_PLATFORM";                            break;
        case CL_INVALID_DEVICE:                             err_str = "CL_INVALID_DEVICE";                              break;
        case CL_INVALID_CONTEXT:                            err_str = "CL_INVALID_CONTEXT";                             break;
        case CL_INVALID_QUEUE_PROPERTIES:                   err_str = "CL_INVALID_QUEUE_PROPERTIES";                    break;
        case CL_INVALID_COMMAND_QUEUE:                      err_str = "CL_INVALID_COMMAND_QUEUE";                       break;
        case CL_INVALID_HOST_PTR:                           err_str = "CL_INVALID_HOST_PTR";                            break;
        case CL_INVALID_MEM_OBJECT:                         err_str = "CL_INVALID_MEM_OBJECT";                          break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:            err_str = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";             break;
        case CL_INVALID_IMAGE_SIZE:                         err_str = "CL_INVALID_IMAGE_SIZE";                          break;
        case CL_INVALID_SAMPLER:                            err_str = "CL_INVALID_SAMPLER";                             break;
        case CL_INVALID_BINARY:                             err_str = "CL_INVALID_BINARY";                              break;
        case CL_INVALID_BUILD_OPTIONS:                      err_str = "CL_INVALID_BUILD_OPTIONS";                       break;
        case CL_INVALID_PROGRAM:                            err_str = "CL_INVALID_PROGRAM";                             break;
        case CL_INVALID_PROGRAM_EXECUTABLE:                 err_str = "CL_INVALID_PROGRAM_EXECUTABLE";                  break;
        case CL_INVALID_KERNEL_NAME:                        err_str = "CL_INVALID_KERNEL_NAME";                         break;
        case CL_INVALID_KERNEL_DEFINITION:                  err_str = "CL_INVALID_KERNEL_DEFINITION";                   break;
        case CL_INVALID_KERNEL:                             err_str = "CL_INVALID_KERNEL";                              break;
        case CL_INVALID_ARG_INDEX:                          err_str = "CL_INVALID_ARG_INDEX";                           break;
        case CL_INVALID_ARG_VALUE:                          err_str = "CL_INVALID_ARG_VALUE";                           break;
        case CL_INVALID_ARG_SIZE:                           err_str = "CL_INVALID_ARG_SIZE";                            break;
        case CL_INVALID_KERNEL_ARGS:                        err_str = "CL_INVALID_KERNEL_ARGS";                         break;
        case CL_INVALID_WORK_DIMENSION:                     err_str = "CL_INVALID_WORK_DIMENSION";                      break;
        case CL_INVALID_WORK_GROUP_SIZE:                    err_str = "CL_INVALID_WORK_GROUP_SIZE";                     break;
        case CL_INVALID_WORK_ITEM_SIZE:                     err_str = "CL_INVALID_WORK_ITEM_SIZE";                      break;
        case CL_INVALID_GLOBAL_OFFSET:                      err_str = "CL_INVALID_GLOBAL_OFFSET";                       break;
        case CL_INVALID_EVENT_WAIT_LIST:                    err_str = "CL_INVALID_EVENT_WAIT_LIST";                     break;
        case CL_INVALID_EVENT:                              err_str = "CL_INVALID_EVENT";                               break;
        case CL_INVALID_OPERATION:                          err_str = "CL_INVALID_OPERATION";                           break;
        case CL_INVALID_GL_OBJECT:                          err_str = "CL_INVALID_GL_OBJECT";                           break;
        case CL_INVALID_BUFFER_SIZE:                        err_str = "CL_INVALID_BUFFER_SIZE";                         break;
        case CL_INVALID_MIP_LEVEL:                          err_str = "CL_INVALID_MIP_LEVEL";                           break;
        case CL_INVALID_GLOBAL_WORK_SIZE:                   err_str = "CL_INVALID_GLOBAL_WORK_SIZE";                    break;
        case CL_INVALID_PROPERTY:                           err_str = "CL_INVALID_PROPERTY";                            break;
        case CL_INVALID_IMAGE_DESCRIPTOR:                   err_str = "CL_INVALID_IMAGE_DESCRIPTOR";                    break;
        case CL_INVALID_COMPILER_OPTIONS:                   err_str = "CL_INVALID_COMPILER_OPTIONS";                    break;
        case CL_INVALID_LINKER_OPTIONS:                     err_str = "CL_INVALID_LINKER_OPTIONS";                      break;
        case CL_INVALID_DEVICE_PARTITION_COUNT:             err_str = "CL_INVALID_DEVICE_PARTITION_COUNT";              break;
            
        default:
            err_str = "UNKNOWN CL ERROR";
            break;
    }
    
    return err_str;
}

cl_platform_info clStringToPlatformInfo(std::string name)
{
    cl_platform_info info = UNDEFINED_INFO;
    
    if (name == "CL_PLATFORM_PROFILE") info = CL_PLATFORM_PROFILE;
    else if (name == "CL_PLATFORM_VERSION") info = CL_PLATFORM_VERSION;
    else if (name == "CL_PLATFORM_NAME") info = CL_PLATFORM_NAME;
    else if (name == "CL_PLATFORM_VENDOR") info = CL_PLATFORM_VENDOR;
    else if (name == "CL_PLATFORM_EXTENSIONS") info = CL_PLATFORM_EXTENSIONS;
    
    return info;
}


cl_device_info clStringToDeviceInfo(std::string name)
{
    cl_device_info info = UNDEFINED_INFO;

    if (name == "CL_DEVICE_TYPE") info = CL_DEVICE_TYPE;
    else if (name == "CL_DEVICE_VENDOR_ID") info = CL_DEVICE_VENDOR_ID;
    else if (name == "CL_DEVICE_MAX_COMPUTE_UNITS") info = CL_DEVICE_MAX_COMPUTE_UNITS;
    else if (name == "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS") info = CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS;
    else if (name == "CL_DEVICE_MAX_WORK_GROUP_SIZE") info = CL_DEVICE_MAX_WORK_GROUP_SIZE;
    else if (name == "CL_DEVICE_MAX_WORK_ITEM_SIZES") info = CL_DEVICE_MAX_WORK_ITEM_SIZES;
    else if (name == "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR") info = CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR;
    else if (name == "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT") info = CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT;
    else if (name == "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT") info = CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT;
    else if (name == "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG") info = CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG;
    else if (name == "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT") info = CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT;
    else if (name == "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE") info = CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE;
    else if (name == "CL_DEVICE_MAX_CLOCK_FREQUENCY") info = CL_DEVICE_MAX_CLOCK_FREQUENCY;
    else if (name == "CL_DEVICE_ADDRESS_BITS") info = CL_DEVICE_ADDRESS_BITS;
    else if (name == "CL_DEVICE_MAX_READ_IMAGE_ARGS") info = CL_DEVICE_MAX_READ_IMAGE_ARGS;
    else if (name == "CL_DEVICE_MAX_WRITE_IMAGE_ARGS") info = CL_DEVICE_MAX_WRITE_IMAGE_ARGS;
    else if (name == "CL_DEVICE_MAX_MEM_ALLOC_SIZE") info = CL_DEVICE_MAX_MEM_ALLOC_SIZE;
    else if (name == "CL_DEVICE_IMAGE2D_MAX_WIDTH") info = CL_DEVICE_IMAGE2D_MAX_WIDTH;
    else if (name == "CL_DEVICE_IMAGE2D_MAX_HEIGHT") info = CL_DEVICE_IMAGE2D_MAX_HEIGHT;
    else if (name == "CL_DEVICE_IMAGE3D_MAX_WIDTH") info = CL_DEVICE_IMAGE3D_MAX_WIDTH;
    else if (name == "CL_DEVICE_IMAGE3D_MAX_HEIGHT") info = CL_DEVICE_IMAGE3D_MAX_HEIGHT;
    else if (name == "CL_DEVICE_IMAGE3D_MAX_DEPTH") info = CL_DEVICE_IMAGE3D_MAX_DEPTH;
    else if (name == "CL_DEVICE_IMAGE_SUPPORT") info = CL_DEVICE_IMAGE_SUPPORT;
    else if (name == "CL_DEVICE_MAX_PARAMETER_SIZE") info = CL_DEVICE_MAX_PARAMETER_SIZE;
    else if (name == "CL_DEVICE_MAX_SAMPLERS") info = CL_DEVICE_MAX_SAMPLERS;
    else if (name == "CL_DEVICE_MEM_BASE_ADDR_ALIGN") info = CL_DEVICE_MEM_BASE_ADDR_ALIGN;
    else if (name == "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE") info = CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE;
    else if (name == "CL_DEVICE_SINGLE_FP_CONFIG") info = CL_DEVICE_SINGLE_FP_CONFIG;
    else if (name == "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE") info = CL_DEVICE_GLOBAL_MEM_CACHE_TYPE;
    else if (name == "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE") info = CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE;
    else if (name == "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE") info = CL_DEVICE_GLOBAL_MEM_CACHE_SIZE;
    else if (name == "CL_DEVICE_GLOBAL_MEM_SIZE") info = CL_DEVICE_GLOBAL_MEM_SIZE;
    else if (name == "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE") info = CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE;
    else if (name == "CL_DEVICE_MAX_CONSTANT_ARGS") info = CL_DEVICE_MAX_CONSTANT_ARGS;
    else if (name == "CL_DEVICE_LOCAL_MEM_TYPE") info = CL_DEVICE_LOCAL_MEM_TYPE;
    else if (name == "CL_DEVICE_LOCAL_MEM_SIZE") info = CL_DEVICE_LOCAL_MEM_SIZE;
    else if (name == "CL_DEVICE_ERROR_CORRECTION_SUPPORT") info = CL_DEVICE_ERROR_CORRECTION_SUPPORT;
    else if (name == "CL_DEVICE_PROFILING_TIMER_RESOLUTION") info = CL_DEVICE_PROFILING_TIMER_RESOLUTION;
    else if (name == "CL_DEVICE_ENDIAN_LITTLE") info = CL_DEVICE_ENDIAN_LITTLE;
    else if (name == "CL_DEVICE_AVAILABLE") info = CL_DEVICE_AVAILABLE;
    else if (name == "CL_DEVICE_COMPILER_AVAILABLE") info = CL_DEVICE_COMPILER_AVAILABLE;
    else if (name == "CL_DEVICE_EXECUTION_CAPABILITIES") info = CL_DEVICE_EXECUTION_CAPABILITIES;
    else if (name == "CL_DEVICE_QUEUE_PROPERTIES") info = CL_DEVICE_QUEUE_PROPERTIES;
    else if (name == "CL_DEVICE_NAME") info = CL_DEVICE_NAME;
    else if (name == "CL_DEVICE_VENDOR") info = CL_DEVICE_VENDOR;
    else if (name == "CL_DRIVER_VERSION") info = CL_DRIVER_VERSION;
    else if (name == "CL_DEVICE_PROFILE") info = CL_DEVICE_PROFILE;
    else if (name == "CL_DEVICE_VERSION") info = CL_DEVICE_VERSION;
    else if (name == "CL_DEVICE_EXTENSIONS") info = CL_DEVICE_EXTENSIONS;
    else if (name == "CL_DEVICE_PLATFORM") info = CL_DEVICE_PLATFORM;
    else if (name == "CL_DEVICE_DOUBLE_FP_CONFIG") info = CL_DEVICE_DOUBLE_FP_CONFIG;
    else if (name == "CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF") info = CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF;
    else if (name == "CL_DEVICE_HOST_UNIFIED_MEMORY") info = CL_DEVICE_HOST_UNIFIED_MEMORY;
    else if (name == "CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR") info = CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR;
    else if (name == "CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT") info = CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT;
    else if (name == "CL_DEVICE_NATIVE_VECTOR_WIDTH_INT") info = CL_DEVICE_NATIVE_VECTOR_WIDTH_INT;
    else if (name == "CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG") info = CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG;
    else if (name == "CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT") info = CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT;
    else if (name == "CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE") info = CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE;
    else if (name == "CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF") info = CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF;
    else if (name == "CL_DEVICE_OPENCL_C_VERSION") info = CL_DEVICE_OPENCL_C_VERSION;
    else if (name == "CL_DEVICE_LINKER_AVAILABLE") info = CL_DEVICE_LINKER_AVAILABLE;
    else if (name == "CL_DEVICE_BUILT_IN_KERNELS") info = CL_DEVICE_BUILT_IN_KERNELS;
    else if (name == "CL_DEVICE_IMAGE_MAX_BUFFER_SIZE") info = CL_DEVICE_IMAGE_MAX_BUFFER_SIZE;
    else if (name == "CL_DEVICE_IMAGE_MAX_ARRAY_SIZE") info = CL_DEVICE_IMAGE_MAX_ARRAY_SIZE;
    else if (name == "CL_DEVICE_PARENT_DEVICE") info = CL_DEVICE_PARENT_DEVICE;
    else if (name == "CL_DEVICE_PARTITION_MAX_SUB_DEVICES") info = CL_DEVICE_PARTITION_MAX_SUB_DEVICES;
    else if (name == "CL_DEVICE_PARTITION_PROPERTIES") info = CL_DEVICE_PARTITION_PROPERTIES;
    else if (name == "CL_DEVICE_PARTITION_AFFINITY_DOMAIN") info = CL_DEVICE_PARTITION_AFFINITY_DOMAIN;
    else if (name == "CL_DEVICE_PARTITION_TYPE") info = CL_DEVICE_PARTITION_TYPE;
    else if (name == "CL_DEVICE_REFERENCE_COUNT") info = CL_DEVICE_REFERENCE_COUNT;
    else if (name == "CL_DEVICE_PREFERRED_INTEROP_USER_SYNC") info = CL_DEVICE_PREFERRED_INTEROP_USER_SYNC;
    else if (name == "CL_DEVICE_PRINTF_BUFFER_SIZE") info = CL_DEVICE_PRINTF_BUFFER_SIZE;
    else if (name == "CL_DEVICE_IMAGE_PITCH_ALIGNMENT") info = CL_DEVICE_IMAGE_PITCH_ALIGNMENT;
    else if (name == "CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT") info = CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT;
    
    return info;
}

void getFullSizes(size_t& full_rowsA, size_t& full_colsA, size_t& full_colsB,
                  size_t rowsA,  size_t colsA,  size_t colsB,
                  size_t vector_size, size_t row_multiple, size_t row_tile_size, size_t col_tile_size,
                  const std::vector<size_t>& work_item_sizes)
{
    size_t multiple = vector_size * row_multiple;
    full_colsA = multiple * ((colsA + multiple - 1) / multiple);
    
    size_t rowsA_multiple = row_tile_size * work_item_sizes[1];
    full_rowsA = rowsA_multiple * ((rowsA + rowsA_multiple - 1) / rowsA_multiple);
    
    size_t colsB_multiple = row_tile_size * work_item_sizes[1];
    full_colsB = colsB_multiple * ((colsB + colsB_multiple - 1) / colsB_multiple);
    
    if (gTrace) {
        CERR << "getFullSizes(full_rowsA=" << full_rowsA << ", full_colsA=" << full_colsA << ", full_colsB=" << full_colsB << "," << endl <<
        "    rowsA=" << rowsA << ", colsA=" << colsA << ", colsB=" << colsB << ", vector_size=" << vector_size<<
        ", row_multiple=" << row_multiple << ", " << endl <<
        "    row_tile_size=" << row_tile_size << ", col_tile_size=" << col_tile_size <<
        ", work_item_sizes=(" << work_item_sizes[0] << ", " << work_item_sizes[1] << ", " << work_item_sizes[2] << ")" << endl;
    }
}

void getFullSizes_atia(bool transposeA, bool transposeB,
                  size_t& full_rowsA, size_t& full_colsA, size_t& full_rowsB, size_t& full_colsB,
                  size_t rowsA,  size_t colsA, size_t rowsB, size_t colsB,
                  size_t vector_size, size_t row_multiple, size_t row_tile_size, size_t col_tile_size,
                  const std::vector<size_t>& work_item_sizes)
{
    // values after transposing, if any (atia)
    size_t atia_nrowA;
    size_t atia_ncolA;
    size_t atia_nrowB;
    size_t atia_ncolB;
    
    if (transposeA) {
        atia_nrowA = colsA;
        atia_ncolA = rowsA;
        
    } else {
        atia_nrowA = rowsA;
        atia_ncolA = colsA;
    }
    
    if (transposeB) {
        atia_nrowB = colsB;
        atia_ncolB = rowsB;
        
    } else {
        atia_nrowB = rowsB;
        atia_ncolB = colsB;
    }
    
    size_t atia_full_nrowA = 0;
    size_t atia_full_ncolA = 0;
    size_t atia_full_ncolB = 0;
    
    getFullSizes(atia_full_nrowA, atia_full_ncolA, atia_full_ncolB, atia_nrowA, atia_ncolA, atia_ncolB,
                 vector_size, row_multiple, row_tile_size, col_tile_size, work_item_sizes);
    
    if (transposeA) {
        full_rowsA = atia_full_ncolA;
        full_colsA = atia_full_nrowA;
        
    } else {
        full_rowsA = atia_full_nrowA;
        full_colsA = atia_full_ncolA;
    }
    
    if (transposeB) {
        full_colsB = atia_full_ncolA;
        full_rowsB = atia_full_ncolB;
        
    } else {
        full_colsB = atia_full_ncolB;
        full_rowsB = atia_full_ncolA;
    }
}

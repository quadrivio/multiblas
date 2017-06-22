//
//  multiblas.cpp
//  multiBLAS.XC
//
//  Created by michael on 6/29/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#include "multiblas.h"
#include "nullptr.h"
#include "opencl_info.h"
#include "crossprod_opencl.h"
#include "gemm_opencl.h"

#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

// ========== Globals ==============================================================================

bool gTrace = false;    // for debugging

#if RPACKAGE

#include "shim.h"

#include <algorithm>
#ifdef USE_TIMING
#include <chrono>
#endif
#include <cctype>
#include <iomanip>
#include <cstring>
#include <vector>
#include <iostream>
#include <sstream>
#include <cstddef>

using namespace std;
#ifdef USE_TIMING
using namespace std::chrono;
#endif

// ========== Local Headers ========================================================================

// ========== Functions ============================================================================

SEXP null_externalptr_C(SEXP s_externalptr)
{
    if (gTrace) CERR << "null_externalptr" << endl;
    
    int resultUnprotectCount = 0;
    
    // --------------- verify arg types ---------------
    
    // --------------- get args ---------------
    
    // --------------- calculate results ---------------
    
    SEXP result;
    PROTECT(result = R_MakeExternalPtr((void *)0, R_NilValue, R_NilValue));
    resultUnprotectCount++;
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << /*"return " << resultUnprotectCount <<*/ endl;
    
    return(result);
}

SEXP is_externalptr_null_C(SEXP s_externalptr)
{
    if (gTrace) CERR << "is_externalptr_null_C" << endl;
    
    int resultUnprotectCount = 0;
    
    // --------------- verify arg types ---------------
    
    if (gTrace) CERR << "verify arg types" << endl;
    
    {
        SEXP externalptrClass;
        PROTECT(externalptrClass = Rf_getAttrib(s_externalptr, R_ClassSymbol));
        
        if (!Rf_isNull(externalptrClass) && strcmp("externalptr", CHAR(STRING_ELT(externalptrClass, 0))) != 0) {
            error("is_externalptr_null_C: wrong externalptr class");
        }
        
        UNPROTECT(1);
    }
    
    // --------------- get args ---------------
    
    void *pointer = R_ExternalPtrAddr(s_externalptr);
    
    if (gTrace) CERR << "s_externalptr = " << hex << (unsigned long long)(void *)pointer << dec << endl;
    
    // --------------- calculate results ---------------
    
    SEXP result;
    PROTECT(result = Rf_allocVector(LGLSXP, 1));
    resultUnprotectCount++;
    
    *LOGICAL(result) = pointer == nullptr;
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << /*"return " << resultUnprotectCount <<*/ endl;
    
    return(result);
}

// call from R
SEXP opencl_platforms_C()
{
    if (gTrace) CERR << "opencl_platforms_C" << endl;
    
    // --------------- verify arg types ---------------
    
    // --------------- get args ---------------
    
    // --------------- calculate results ---------------
    
    cl_int err;
    
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    
    vector<cl_platform_id> platforms;
    if (err == CL_SUCCESS) {
        platforms.resize(num_platforms);
        err = clGetPlatformIDs(num_platforms, &platforms[0], &num_platforms);
    }
    
    vector<string> platform_names;
    if (err == CL_SUCCESS) {
        vector<cl_platform_id>::iterator platform_iter = platforms.begin();
        while (platform_iter != platforms.end() && err == CL_SUCCESS) {
            stringstream label;
            label << getPlatformInfoString(*platform_iter, CL_PLATFORM_VENDOR) << ": ";
            label << getPlatformInfoString(*platform_iter, CL_PLATFORM_NAME);
            
            platform_names.push_back(label.str());
            
            ++platform_iter;
        }
    }
    
    vector<string> platform_infos;
    if (err == CL_SUCCESS) {
        vector<cl_platform_id>::iterator platform_iter = platforms.begin();
        int platform_index = 1;
        while (platform_iter != platforms.end() && err == CL_SUCCESS) {
            stringstream info;
            
            info << "----- Platform " << platform_index++ << " -----" << endl;
            info << "Profile: " << getPlatformInfoString(*platform_iter, CL_PLATFORM_PROFILE) << endl;
            info << "Version: " << getPlatformInfoString(*platform_iter, CL_PLATFORM_VERSION) << endl;
            info << "Name: " << getPlatformInfoString(*platform_iter, CL_PLATFORM_NAME) << endl;
            info << "Vendor: " << getPlatformInfoString(*platform_iter, CL_PLATFORM_VENDOR) << endl;
            info << "Extensions: " << getPlatformInfoString(*platform_iter, CL_PLATFORM_EXTENSIONS) << endl;
            info << endl;
            
            platform_infos.push_back(info.str());
            
            ++platform_iter;
        }
    }
    
    // --------------- package results ---------------
    
    if (gTrace) CERR << "package results" << endl;
    int resultUnprotectCount = 0;
    
    SEXP result;
    PROTECT(result = Rf_allocVector(VECSXP, num_platforms));
    resultUnprotectCount++;
    
    SEXP className;
    PROTECT(className = Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(className, 0, Rf_mkChar("opencl.platforms"));
    Rf_classgets(result, className);
    resultUnprotectCount++;
    
    vector<SEXP> s_platforms(num_platforms);
    
    for (size_t k = 0; k < num_platforms; k++) {
        int itemUnprotectCount = 0;
        
        const int numItemFields = 4;
        
        PROTECT(s_platforms[k] = Rf_allocVector(VECSXP, numItemFields));
        itemUnprotectCount++;
        
        SEXP itemClassName;
        PROTECT(itemClassName = Rf_allocVector(STRSXP, 1));
        SET_STRING_ELT(itemClassName, 0, Rf_mkChar("opencl.platform"));
        Rf_classgets(s_platforms[k], itemClassName);
        itemUnprotectCount++;
        
        SET_VECTOR_ELT(result, (int)k, s_platforms[k]);
        
        SEXP field_names;
        PROTECT(field_names = Rf_allocVector(STRSXP, numItemFields));
        itemUnprotectCount++;
        
        int fieldNum;
        
        // --------------------------------------------------------------
        
        fieldNum = 0;
        
        SET_STRING_ELT(field_names, fieldNum, Rf_mkChar("name"));
        
        SEXP s_name;
        PROTECT(s_name = Rf_allocVector(STRSXP, 1));
        itemUnprotectCount++;
        
        SET_VECTOR_ELT(s_platforms[k], (int)fieldNum, s_name);
        SET_STRING_ELT(s_name, 0, Rf_mkChar(platform_names[k].c_str()));
        
        // --------------------------------------------------------------
        
        fieldNum = 1;
        
        SET_STRING_ELT(field_names, fieldNum, Rf_mkChar("index"));
        
        SEXP s_index;
        PROTECT(s_index = Rf_allocVector(INTSXP, 1));
        itemUnprotectCount++;
        
        SET_VECTOR_ELT(s_platforms[k], (int)fieldNum, s_index);
        *INTEGER(s_index) = (int)k + 1;
        
        // --------------------------------------------------------------
        
        if (gTrace) CERR << "cl_platform_id = " << hex << (unsigned long long)(void *)platforms[k] << dec << endl;
        
        fieldNum = 2;
        
        SET_STRING_ELT(field_names, fieldNum, Rf_mkChar("id"));
        
        SEXP s_id;
        PROTECT(s_id = R_MakeExternalPtr(platforms[k], R_NilValue, R_NilValue));
        itemUnprotectCount++;
        
        SET_VECTOR_ELT(s_platforms[k], (int)fieldNum, s_id);
        
        // --------------------------------------------------------------
        
        fieldNum = 3;
        
        SET_STRING_ELT(field_names, fieldNum, Rf_mkChar("info"));
        
        SEXP s_info;
        PROTECT(s_info = Rf_allocVector(STRSXP, 1));
        itemUnprotectCount++;
        
        SET_VECTOR_ELT(s_platforms[k], (int)fieldNum, s_info);
        SET_STRING_ELT(s_info, 0, Rf_mkChar(platform_infos[k].c_str()));
        
        // --------------------------------------------------------------
        
        Rf_setAttrib(s_platforms[k], R_NamesSymbol, field_names);
        
        UNPROTECT(itemUnprotectCount);
    }
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << /*"return " << resultUnprotectCount <<*/ endl;
    
    return(result);
}

SEXP opencl_devices_C(SEXP s_platform)
{
    if (gTrace) CERR << "opencl_devices_C" << endl;
    
    // --------------- verify arg types ---------------
    
    {
        SEXP platformClass;
        PROTECT(platformClass = Rf_getAttrib(s_platform, R_ClassSymbol));
        
        if (!Rf_isNull(platformClass) && strcmp("opencl.platform", CHAR(STRING_ELT(platformClass, 0))) != 0) {
            error("opencl_devices_C: wrong platform class");
        }
        
        UNPROTECT(1);
    }
    
    // --------------- get args ---------------
    
    const int idElementIndex = 2;
    SEXP s_id = VECTOR_ELT(s_platform, idElementIndex);
    
    cl_platform_id platform_id = (cl_platform_id)R_ExternalPtrAddr(s_id);
    
    if (platform_id == nullptr) {
        error("opencl_devices_C: null cl_platform_id");
    }
    
    if (gTrace) CERR << "cl_platform_id = " << hex << (unsigned long long)(void *)platform_id << dec << endl;
    
    // --------------- calculate results ---------------
    
    cl_int err;
    
    cl_uint num_devices;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    
    vector<cl_device_id> devices;
    if (err == CL_SUCCESS) {
        devices.resize(num_devices);
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, num_devices, &devices[0], NULL);
    }
    
    vector<string> device_names;
    vector<string> device_types;
    if (err == CL_SUCCESS) {
        vector<cl_device_id>::iterator device_iter = devices.begin();
        while (device_iter != devices.end() && err == CL_SUCCESS) {
            stringstream label;
            label << getDeviceInfoString(*device_iter, CL_DEVICE_VENDOR) << ": ";
            label << getDeviceInfoString(*device_iter, CL_DEVICE_NAME);
            
            device_names.push_back(label.str());
            
            device_types.push_back(getDeviceInfoString(*device_iter, CL_DEVICE_TYPE));
            
            ++device_iter;
        }
    }
    
    vector<string> device_infos;
    if (err == CL_SUCCESS) {
        vector<cl_device_id>::iterator device_iter = devices.begin();
        int device_index = 1;
        while (device_iter != devices.end() && err == CL_SUCCESS) {
            stringstream info;
            
            info << "----- Device " << device_index++ << " -----" << endl;
            info << "Name: " << getDeviceInfoString(*device_iter, CL_DEVICE_NAME) << endl;
            info << "Profile: " << getDeviceInfoString(*device_iter, CL_DEVICE_PROFILE) << endl;
            info << "Vendor: " << getDeviceInfoString(*device_iter, CL_DEVICE_VENDOR) << endl;
            info << "Vendor ID: " << getDeviceInfoString(*device_iter, CL_DEVICE_VENDOR_ID) << endl;
            info << "Device Version: " << getDeviceInfoString(*device_iter, CL_DEVICE_VERSION) << endl;
            info << "Driver Version: " << getDeviceInfoString(*device_iter, CL_DRIVER_VERSION) << endl;
            info << "Max Clock MHz: " << getDeviceInfoString(*device_iter, CL_DEVICE_MAX_CLOCK_FREQUENCY) << endl;
            info << "Max Compute Units: " << getDeviceInfoString(*device_iter, CL_DEVICE_MAX_COMPUTE_UNITS) << endl;
            info << "Max Samplers: " << getDeviceInfoString(*device_iter, CL_DEVICE_MAX_SAMPLERS) << endl;
            info << "Max Work Item Sizes: " << getDeviceInfoString(*device_iter, CL_DEVICE_MAX_WORK_ITEM_SIZES) << endl;
            info << "Double FP Config: " << getDeviceInfoString(*device_iter, CL_DEVICE_DOUBLE_FP_CONFIG) << endl;
            info << "Global Mem Size: " << getDeviceInfoString(*device_iter, CL_DEVICE_GLOBAL_MEM_SIZE) << endl;
            info << "Preferred Char Vector Width: " << getDeviceInfoString(*device_iter, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR) << endl;
            info << "Preferred Int Vector Width: " << getDeviceInfoString(*device_iter, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT) << endl;
            info << "Preferred Float Vector Width: " << getDeviceInfoString(*device_iter, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT) << endl;
            info << "Preferred Double Vector Width: " << getDeviceInfoString(*device_iter, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE) << endl;
            info << "Extensions: " << getDeviceInfoString(*device_iter, CL_DEVICE_EXTENSIONS) << endl;
            info << endl;
            
            device_infos.push_back(info.str());
            
            ++device_iter;
        }
    }
    
    // --------------- package results ---------------
    
    if (gTrace) CERR << "package results" << endl;
    int resultUnprotectCount = 0;
    
    SEXP result;
    PROTECT(result = Rf_allocVector(VECSXP, num_devices));
    resultUnprotectCount++;
    
    SEXP className;
    PROTECT(className = Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(className, 0, Rf_mkChar("opencl.devices"));
    Rf_classgets(result, className);
    resultUnprotectCount++;
    
    vector<SEXP> s_devices(num_devices);
    
    for (size_t k = 0; k < num_devices; k++) {
        int itemUnprotectCount = 0;
        
        const int numItemFields = 5;
        
        PROTECT(s_devices[k] = Rf_allocVector(VECSXP, numItemFields));
        itemUnprotectCount++;
        
        SEXP itemClassName;
        PROTECT(itemClassName = Rf_allocVector(STRSXP, 1));
        SET_STRING_ELT(itemClassName, 0, Rf_mkChar("opencl.device"));
        Rf_classgets(s_devices[k], itemClassName);
        itemUnprotectCount++;
        
        SET_VECTOR_ELT(result, (int)k, s_devices[k]);
        
        SEXP field_names;
        PROTECT(field_names = Rf_allocVector(STRSXP, numItemFields));
        itemUnprotectCount++;
        
        int fieldNum;
        
        // --------------------------------------------------------------
        
        fieldNum = 0;
        
        SET_STRING_ELT(field_names, fieldNum, Rf_mkChar("name"));
        
        SEXP s_name;
        PROTECT(s_name = Rf_allocVector(STRSXP, 1));
        itemUnprotectCount++;
        
        SET_VECTOR_ELT(s_devices[k], (int)fieldNum, s_name);
        SET_STRING_ELT(s_name, 0, Rf_mkChar(device_names[k].c_str()));
        
        // --------------------------------------------------------------
        
        fieldNum = 1;
        
        SET_STRING_ELT(field_names, fieldNum, Rf_mkChar("type"));
        
        SEXP s_type;
        PROTECT(s_type = Rf_allocVector(STRSXP, 1));
        itemUnprotectCount++;
        
        SET_VECTOR_ELT(s_devices[k], (int)fieldNum, s_type);
        SET_STRING_ELT(s_type, 0, Rf_mkChar(device_types[k].c_str()));
        
        // --------------------------------------------------------------
        
        fieldNum = 2;
        
        SET_STRING_ELT(field_names, fieldNum, Rf_mkChar("index"));
        
        SEXP s_index;
        PROTECT(s_index = Rf_allocVector(INTSXP, 1));
        itemUnprotectCount++;
        
        SET_VECTOR_ELT(s_devices[k], (int)fieldNum, s_index);
        *INTEGER(s_index) = (int)k + 1;
        
        // --------------------------------------------------------------
        
        if (gTrace) CERR << "cl_device_id = " << hex << (unsigned long long)(void *)devices[k] << dec << endl;
        
        fieldNum = 3;
        
        SET_STRING_ELT(field_names, fieldNum, Rf_mkChar("id"));
        
        SEXP s_id;
        PROTECT(s_id = R_MakeExternalPtr(devices[k], R_NilValue, R_NilValue));
        itemUnprotectCount++;
        
        SET_VECTOR_ELT(s_devices[k], (int)fieldNum, s_id);
        
        // --------------------------------------------------------------
        
        fieldNum = 4;
        
        SET_STRING_ELT(field_names, fieldNum, Rf_mkChar("info"));
        
        SEXP s_info;
        PROTECT(s_info = Rf_allocVector(STRSXP, 1));
        itemUnprotectCount++;
        
        SET_VECTOR_ELT(s_devices[k], (int)fieldNum, s_info);
        SET_STRING_ELT(s_info, 0, Rf_mkChar(device_infos[k].c_str()));
        
        // --------------------------------------------------------------
        
        Rf_setAttrib(s_devices[k], R_NamesSymbol, field_names);
        
        UNPROTECT(itemUnprotectCount);
    }
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << /*"return " << resultUnprotectCount <<*/ endl;
    
    return(result);
}

SEXP property_platform_C(SEXP s_platform, SEXP s_name)
{
    if (gTrace) CERR << "platform_property_C" << endl;
    
    // --------------- verify arg types ---------------
    
    {
        SEXP platformClass;
        PROTECT(platformClass = Rf_getAttrib(s_platform, R_ClassSymbol));
        
        if (!Rf_isNull(platformClass) && strcmp("opencl.platform", CHAR(STRING_ELT(platformClass, 0))) != 0) {
            error("platform_property_C: wrong platform class");
        }
        
        UNPROTECT(1);
    }
    
    if (!Rf_isString(s_name)) {
        error("platform_property_C: wrong name type");
    }
    
    // --------------- get args ---------------
    
    const int idElementIndex = 2;
    SEXP s_id = VECTOR_ELT(s_platform, idElementIndex);
    
    cl_platform_id platform_id = (cl_platform_id)R_ExternalPtrAddr(s_id);
    
    if (platform_id == nullptr) {
        error("platform_property_C: null cl_platform_id");
    }
    
    if (gTrace) CERR << "cl_platform_id = " << hex << (unsigned long long)(void *)platform_id << dec << endl;

    string name = CHAR(STRING_ELT(s_name, 0));
    
    if (gTrace) CERR << "name = " << name << endl;

    // --------------- calculate results ---------------
    
    cl_int err;
    string info;
    
    cl_platform_info platform_info = clStringToPlatformInfo(name);
    
    if (platform_info == UNDEFINED_INFO) {
        error("platform_property_C: undefined property name");
        
    } else {
        info = getPlatformInfoString(platform_id, platform_info, &err);
        if (err != CL_SUCCESS) {
            error("platform_property_C: get info error");
        }
    }
    
    // --------------- package results ---------------
    
    if (gTrace) CERR << "package results" << endl;
    int resultUnprotectCount = 0;
    
    SEXP result;
    PROTECT(result = Rf_allocVector(STRSXP, 1));
    resultUnprotectCount++;
 
    SET_STRING_ELT(result, 0, Rf_mkChar(info.c_str()));
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << /*"return " << resultUnprotectCount <<*/ endl;
    
    return(result);
}

SEXP property_device_C(SEXP s_device, SEXP s_name)
{
    if (gTrace) CERR << "device_property_C" << endl;
    
    // --------------- verify arg types ---------------
    
    {
        SEXP deviceClass;
        PROTECT(deviceClass = Rf_getAttrib(s_device, R_ClassSymbol));
        
        if (!Rf_isNull(deviceClass) && strcmp("opencl.device", CHAR(STRING_ELT(deviceClass, 0))) != 0) {
            error("device_property_C: wrong device class");
        }
        
        UNPROTECT(1);
    }
    
    if (!Rf_isString(s_name)) {
        error("device_property_C: wrong name type");
    }
    
    // --------------- get args ---------------
    
    const int idElementIndex = 3;
    SEXP s_id = VECTOR_ELT(s_device, idElementIndex);
    
    cl_device_id device_id = (cl_device_id)R_ExternalPtrAddr(s_id);
    
    if (device_id == nullptr) {
        error("device_property_C: null cl_device_id");
    }
    
    if (gTrace) CERR << "cl_device_id = " << hex << (unsigned long long)(void *)device_id << dec << endl;
    
    string name = CHAR(STRING_ELT(s_name, 0));
    
    if (gTrace) CERR << "name = " << name << endl;
    
    // --------------- calculate results ---------------
    
    cl_int err;
    string info;
    
    cl_device_info device_info = clStringToDeviceInfo(name);
    
    if (device_info == UNDEFINED_INFO) {
        error("device_property_C: undefined property name");
        
    } else {
        info = getDeviceInfoString(device_id, device_info, &err);
        if (err != CL_SUCCESS) {
            error("device_property_C: get info error");
        }
    }
    
    // --------------- package results ---------------
    
    if (gTrace) CERR << "package results" << endl;
    int resultUnprotectCount = 0;
    
    SEXP result;
    PROTECT(result = Rf_allocVector(STRSXP, 1));
    resultUnprotectCount++;
    
    SET_STRING_ELT(result, 0, Rf_mkChar(info.c_str()));
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << /*"return " << resultUnprotectCount <<*/ endl;
    
    return(result);
}

void context_finalizer(SEXP s_context)
{
    if (gTrace) CERR << "context_finalizer" << endl;

    // --------------- verify arg type ---------------
    
    {
        SEXP contextClass;
        PROTECT(contextClass = Rf_getAttrib(s_context, R_ClassSymbol));
        
        if (!Rf_isNull(contextClass) && strcmp("opencl.context", CHAR(STRING_ELT(contextClass, 0))) != 0) {
            error("context_finalizer: wrong context class");
        }
        
        UNPROTECT(1);
    }
    
    // --------------- get arg ---------------
    
    cl_context context = (cl_context)R_ExternalPtrAddr(s_context);
    
    // --------------- finalize ---------------
    
    if (context != nullptr) {
        clReleaseContext(context);
        R_ClearExternalPtr(s_context);
    }
}

SEXP opencl_context_C(SEXP s_device)
{
    if (gTrace) CERR << "opencl_context_C" << endl;
    
    // --------------- verify arg types ---------------
    
    {
        SEXP deviceClass;
        PROTECT(deviceClass = Rf_getAttrib(s_device, R_ClassSymbol));
        
        if (gTrace) CERR << "s_device class: '" << CHAR(STRING_ELT(deviceClass, 0)) << "'" << endl;
        
        if (!Rf_isNull(deviceClass) && strcmp("opencl.device", CHAR(STRING_ELT(deviceClass, 0))) != 0) {
            error("opencl_context_C: wrong device class");
        }
        
        UNPROTECT(1);
    }
    
    // --------------- get args ---------------
    
    const int idElementIndex = 3;
    SEXP s_id = VECTOR_ELT(s_device, idElementIndex);
    
    cl_device_id device = (cl_device_id)R_ExternalPtrAddr(s_id);
    
    if (gTrace) CERR << "cl_device_id = " << hex << (unsigned long long)(void *)device << dec << endl;
    
    if (device == nullptr) {
        error("opencl_context_C: null cl_device_id");
    }
    
    // --------------- calculate results ---------------
    
    cl_int err;
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    
    if (err != CL_SUCCESS) {
        error("opencl_context_C: cannot create context");
    }
    
    // --------------- package results ---------------
    
    if (gTrace) CERR << "package results" << endl;
    int resultUnprotectCount = 0;
    
    SEXP result;
    PROTECT(result = R_MakeExternalPtr(context, R_NilValue, R_NilValue));
    resultUnprotectCount++;
    
    R_RegisterCFinalizerEx(result, context_finalizer, RTRUE);
    
    SEXP className;
    PROTECT(className = Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(className, 0, Rf_mkChar("opencl.context"));
    Rf_classgets(result, className);
    resultUnprotectCount++;
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << /*"return " << resultUnprotectCount <<*/ endl;
    
    return(result);
}

void queue_finalizer(SEXP s_queue)
{
    if (gTrace) CERR << "queue_finalizer" << endl;
    
    // --------------- verify arg type ---------------
    
    {
        SEXP queueClass;
        PROTECT(queueClass = Rf_getAttrib(s_queue, R_ClassSymbol));
        
        if (!Rf_isNull(queueClass) && strcmp("opencl.queue", CHAR(STRING_ELT(queueClass, 0))) != 0) {
            error("queue_finalizer: wrong queue class");
        }
        
        UNPROTECT(1);
    }
    
    // --------------- get arg ---------------
    
    cl_command_queue queue = (cl_command_queue)R_ExternalPtrAddr(s_queue);
    
    // --------------- finalize ---------------
    
    if (queue != nullptr) {
        cl_int err = clReleaseCommandQueue(queue);
        R_ClearExternalPtr(s_queue);
        
        if (err != CL_SUCCESS) {
            error("queue_finalizer: cannot release queue");
        }
    }
}

SEXP opencl_queue_C(SEXP s_context, SEXP s_device)
{
    if (gTrace) CERR << "opencl_queue_C" << endl;
    
    // --------------- verify arg types ---------------
    
    {
        SEXP contextClass;
        PROTECT(contextClass = Rf_getAttrib(s_context, R_ClassSymbol));
        
        if (!Rf_isNull(contextClass) && strcmp("opencl.context", CHAR(STRING_ELT(contextClass, 0))) != 0) {
            error("opencl_queue_C: wrong context class");
        }
        
        UNPROTECT(1);
    }
    
    {
        SEXP deviceClass;
        PROTECT(deviceClass = Rf_getAttrib(s_device, R_ClassSymbol));
        
        if (!Rf_isNull(deviceClass) && strcmp("opencl.device", CHAR(STRING_ELT(deviceClass, 0))) != 0) {
            error("opencl_queue_C: wrong device class");
        }
        
        UNPROTECT(1);
    }
    
    // --------------- get args ---------------
    
    cl_context context = (cl_context)R_ExternalPtrAddr(s_context);
    
    if (context == nullptr) {
        error("opencl_queue_C: null context");
    }
    
    const int idElementIndex = 3;
    SEXP s_id = VECTOR_ELT(s_device, idElementIndex);
    
    cl_device_id device = (cl_device_id)R_ExternalPtrAddr(s_id);
    
    if (gTrace) CERR << "cl_device_id = " << hex << (unsigned long long)(void *)device << dec << endl;
    
    if (device == nullptr) {
        error("opencl_queue_C: null cl_device_id");
    }
    
    // --------------- calculate results ---------------
    
    cl_int err;
    cl_command_queue queue = NULL;
#ifdef CL_VERSION_2_0
    if (gTrace) CERR << "clCreateCommandQueueWithProperties:" << endl;
    
    queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    
#else
    if (gTrace) CERR << "clCreateCommandQueue:" << endl;
    
    queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    
    if (gTrace) CERR << "queue = " << hex << (unsigned long long)(void *)queue << dec << endl;
    
    if (err != CL_SUCCESS) {
        if (gTrace) {
            CERR << "device = " << hex << (unsigned long long)(void *)device << dec << endl;
            CERR << "context = " << hex << (unsigned long long)(void *)context << dec << endl;
            CERR << "Device: " << getDeviceInfoString(device, CL_DEVICE_NAME) << std::endl;
            CERR << "Context valid: " << (isContextValid(context) ? "TRUE" : "FALSE") << std::endl;
        }
        
        std::stringstream sst;
        sst << "opencl_queue_C: cannot create queue " << clErrorToString(err);
        error(sst.str().c_str());
    }
    
    // --------------- package results ---------------
    
    if (gTrace) CERR << "package results" << endl;
    int resultUnprotectCount = 0;
    
    SEXP result;
    PROTECT(result = R_MakeExternalPtr(queue, R_NilValue, R_NilValue));
    resultUnprotectCount++;
    
    R_RegisterCFinalizerEx(result, queue_finalizer, RTRUE);
    
    SEXP className;
    PROTECT(className = Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(className, 0, Rf_mkChar("opencl.queue"));
    Rf_classgets(result, className);
    resultUnprotectCount++;
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << /*"return " << resultUnprotectCount <<*/ endl;
    
    return(result);
}

void kernel_finalizer(SEXP s_kernel)
{
    if (gTrace) CERR << "kernel_finalizer" << endl;
    
    // --------------- verify arg type ---------------
    
    {
        SEXP kernelClass;
        PROTECT(kernelClass = Rf_getAttrib(s_kernel, R_ClassSymbol));
        
        if (!Rf_isNull(kernelClass) && strcmp("opencl.kernel", CHAR(STRING_ELT(kernelClass, 0))) != 0) {
            error("kernel_finalizer: wrong kernel class");
        }
        
        UNPROTECT(1);
    }
    
    // --------------- get arg ---------------
    
    cl_kernel kernel = (cl_kernel)R_ExternalPtrAddr(s_kernel);
    
    // --------------- finalize ---------------
    
    if (kernel != nullptr) {
        clReleaseKernel(kernel);
        R_ClearExternalPtr(s_kernel);
    }
}

// call from R
SEXP opencl_kernel_C(SEXP s_context, SEXP s_device, SEXP s_name, SEXP s_source, SEXP s_options,
                     SEXP s_verbose)
{
    if (gTrace) CERR << "opencl_kernel_C" << endl;
    
    // --------------- verify arg types ---------------
    
    {
        SEXP contextClass;
        PROTECT(contextClass = Rf_getAttrib(s_context, R_ClassSymbol));
        
        if (!Rf_isNull(contextClass) && strcmp("opencl.context", CHAR(STRING_ELT(contextClass, 0))) != 0) {
            error("opencl_kernel_C: wrong context class");
        }
        
        UNPROTECT(1);
    }
    
    {
        SEXP deviceClass;
        PROTECT(deviceClass = Rf_getAttrib(s_device, R_ClassSymbol));
        
        if (!Rf_isNull(deviceClass) && strcmp("opencl.device", CHAR(STRING_ELT(deviceClass, 0))) != 0) {
            error("opencl_kernel_C: wrong device class");
        }
        
        UNPROTECT(1);
    }
    
    if (!Rf_isString(s_name)) {
        error("opencl_kernel_C: wrong name type");
    }
    
    if (!Rf_isString(s_source)) {
        error("opencl_kernel_C: wrong source type");
    }
    
    if (!Rf_isString(s_options)) {
        error("opencl_kernel_C: wrong options type");
    }
    
    // --------------- get args ---------------
    
    cl_context context = (cl_context)R_ExternalPtrAddr(s_context);
    
    if (context == nullptr) {
        error("opencl_kernel_C: null context");
    }
    
    const int idElementIndex = 3;
    SEXP s_id = VECTOR_ELT(s_device, idElementIndex);
    
    cl_device_id device = (cl_device_id)R_ExternalPtrAddr(s_id);
    
    if (gTrace) CERR << "cl_device_id = " << hex << (unsigned long long)(void *)device << dec << endl;
    
    if (device == nullptr) {
        error("opencl_kernel_C: null cl_device_id");
    }
    
    string name = CHAR(STRING_ELT(s_name, 0));
    
    if (gTrace) CERR << "name = " << name << endl;
    
    string source = CHAR(STRING_ELT(s_source, 0));
    
    string options = CHAR(STRING_ELT(s_options, 0));
    
    if (gTrace) CERR << "options = \"" << options << "\"" << endl;
    
    bool verbose = !Rf_isNull(s_verbose) && *LOGICAL(s_verbose);
    
    // --------------- calculate results ---------------
    
    cl_int err = CL_SUCCESS;
    
    // program
    cl_program program = nullptr;
    if (err == CL_SUCCESS) {
        if (gTrace) {
            CERR << "clCreateProgramWithSource:" << std::endl;
        }
        
        const char *src = source.c_str();
        program = clCreateProgramWithSource(context, 1, &src, NULL, &err);
    }
    
    if (err == CL_SUCCESS) {
        if (gTrace) {
            CERR << "clBuildProgram:" << std::endl;
        }
        
        err = clBuildProgram(program, 0, NULL, options.c_str(), NULL, NULL);
        
        if (err != CL_SUCCESS || verbose) {
            const size_t log_size = 65536;
            char log[log_size];
            cl_int log_err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                   log_size, log, NULL);
            
            if (log_err != CL_SUCCESS) CERR << "log_err != CL_SUCCESS" << endl;
            
            if ((verbose || gTrace) && (log_err == CL_SUCCESS) && (strlen(log) > 0)) {
                CERR << "clGetProgramBuildInfo:" << std::endl << log << std::endl;
            }
            
            if (gTrace) {
                CERR << "Source:" << std::endl << source << std::endl;
            }
        }
    }
    
    // kernel
    cl_kernel kernel = nullptr;
    if (err == CL_SUCCESS) {
        if (gTrace) {
            CERR << "clCreateKernel:" << std::endl;
        }
        
        kernel = clCreateKernel(program, name.c_str(), &err);
    }
    
    if (program != nullptr) {
        clReleaseProgram(program);
        program = nullptr;
    }
    
    if (gTrace) CERR << "kernel = " << hex << (unsigned long long)(void *)kernel << dec << endl;
    
    if (err != CL_SUCCESS) {
        error(clErrorToString(err).c_str());
    }
    
    // --------------- package results ---------------
    
    if (gTrace) CERR << "package results" << endl;
    int resultUnprotectCount = 0;
    
    SEXP result;
    PROTECT(result = R_MakeExternalPtr(kernel, R_NilValue, R_NilValue));
    resultUnprotectCount++;
    
    R_RegisterCFinalizerEx(result, kernel_finalizer, RTRUE);
    
    SEXP className;
    PROTECT(className = Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(className, 0, Rf_mkChar("opencl.kernel"));
    Rf_classgets(result, className);
    resultUnprotectCount++;
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << /*"return " << resultUnprotectCount <<*/ endl;
    
    return(result);
}

SEXP opencl_calc_x_C(SEXP s_context, SEXP s_kernel_f, SEXP s_kernel_d, SEXP s_queue, SEXP s_x,
                     SEXP s_work_item_sizes, SEXP s_vector_size, SEXP s_row_multiple,
                     SEXP s_row_tile_size, SEXP s_col_tile_size, SEXP s_fill_on_host, SEXP s_verbose)
{
    
#ifdef USE_TIMING
    steady_clock::time_point start_time = steady_clock::now();
#endif
    
    if (gTrace) CERR << "opencl_calc_x_C" << endl;
    
    int resultUnprotectCount = 0;
    
    // --------------- verify arg types ---------------
    
    {
        SEXP contextClass;
        PROTECT(contextClass = Rf_getAttrib(s_context, R_ClassSymbol));
        
        if (!Rf_isNull(contextClass) && strcmp("opencl.context", CHAR(STRING_ELT(contextClass, 0))) != 0) {
            error("opencl_calc_x_C: wrong context class");
        }
        
        UNPROTECT(1);
    }
    
    if (!Rf_isNull(s_kernel_f))
    {
        SEXP kernelClass;
        PROTECT(kernelClass = Rf_getAttrib(s_kernel_f, R_ClassSymbol));
        
        if (!Rf_isNull(kernelClass) && strcmp("opencl.kernel", CHAR(STRING_ELT(kernelClass, 0))) != 0) {
            error("opencl_calc_x_C: wrong kernel_f class");
        }
        
        UNPROTECT(1);
    }
    
    if (!Rf_isNull(s_kernel_d))
    {
        SEXP kernelClass;
        PROTECT(kernelClass = Rf_getAttrib(s_kernel_d, R_ClassSymbol));
        
        if (!Rf_isNull(kernelClass) && strcmp("opencl.kernel", CHAR(STRING_ELT(kernelClass, 0))) != 0) {
            error("opencl_calc_x_C: wrong kernel_d class");
        }
        
        UNPROTECT(1);
    }
    
    {
        SEXP queueClass;
        PROTECT(queueClass = Rf_getAttrib(s_queue, R_ClassSymbol));
        
        if (!Rf_isNull(queueClass) && strcmp("opencl.queue", CHAR(STRING_ELT(queueClass, 0))) != 0) {
            error("opencl_calc_x_C: wrong queue class");
        }
        
        UNPROTECT(1);
    }
    if (Rf_isComplex(s_x)) {
        error("opencl_calc_x_C: complex x not implemented yet");
        
    } else if (Rf_isInteger(s_x)) {
        error("opencl_calc_x_C: integer x not implemented yet");
        
    } else if (!Rf_isReal(s_x) && !Rf_isInteger(s_x)) {
        error("opencl_calc_x_C: wrong x type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_x) = " << XLENGTH(s_x) << endl;
    
    SEXP s_dims;
    PROTECT(s_dims = Rf_getAttrib(s_x, R_DimSymbol));
    resultUnprotectCount++;
    
    if (Rf_isNull(s_dims)) {
        error("opencl_calc_x_C: no dimensions");
    }
    
    SEXP s_float;
    PROTECT(s_float = Rf_getAttrib(s_x, Rf_install("Csingle")));
    resultUnprotectCount++;
    
    bool isFloat = !Rf_isNull(s_float) && *LOGICAL(s_float);
    
    int dimCount = Rf_length(s_dims);
    int *dims = INTEGER(s_dims);
    
    if (gTrace) {
        CERR << "dims = ";
        for (int k = 0; k < dimCount; k++) {
            CERR << dims[k] << " ";
        }
        CERR << endl;
    }
    
    if (dimCount != 2) {
        error("opencl_calc_x_C: wrong dimension");
    }
    
    if (!Rf_isInteger(s_work_item_sizes)) {
        error("opencl_calc_x_C: wrong work_item_sizes class");
    }
    
    if (!Rf_isInteger(s_vector_size)) {
        error("opencl_calc_x_C: wrong vector_size class");
    }
    
    if (!Rf_isInteger(s_row_multiple)) {
        error("opencl_calc_x_C: wrong row_multiple class");
    }
    
    if (!Rf_isInteger(s_row_tile_size)) {
        error("opencl_calc_x_C: wrong row_tile.size class");
    }
    
    if (!Rf_isInteger(s_col_tile_size)) {
        error("opencl_calc_x_C: wrong col_tile.size class");
    }
    
    bool fillOnHost = !Rf_isNull(s_fill_on_host) && *LOGICAL(s_fill_on_host);
    
    bool verbose = !Rf_isNull(s_verbose) && *LOGICAL(s_verbose);
    
    // --------------- get args ---------------
    
    cl_context context = (cl_context)R_ExternalPtrAddr(s_context);
    
    if (context == nullptr) {
        error("opencl_calc_x_C: null context");
    }
    
    cl_kernel kernel_f = Rf_isNull(s_kernel_f) ? nullptr : (cl_kernel)R_ExternalPtrAddr(s_kernel_f);
    
    if (isFloat && kernel_f == nullptr) {
        error("opencl_calc_x_C: null kernel_f");
    }
    
    cl_kernel kernel_d = Rf_isNull(s_kernel_d) ? nullptr : (cl_kernel)R_ExternalPtrAddr(s_kernel_d);
    
    if (!isFloat && kernel_d == nullptr) {
        error("opencl_calc_x_C: null kernel_d");
    }
    
    cl_command_queue queue = (cl_command_queue)R_ExternalPtrAddr(s_queue);

    double *x = REAL(s_x);
    
    if (gTrace && XLENGTH(s_x) <= 256) {
        for (int k = 0; k < XLENGTH(s_x); k++) {
            CERR << "x[" << k << "] = " << x[k] << endl;
        }
    }
    
    int *p = INTEGER(s_work_item_sizes);
    vector<size_t> work_item_sizes;
    work_item_sizes.push_back(XLENGTH(s_work_item_sizes) > 0 ? p[0] : 1);
    work_item_sizes.push_back(XLENGTH(s_work_item_sizes) > 1 ? p[1] : 1);
    work_item_sizes.push_back(XLENGTH(s_work_item_sizes) > 2 ? p[2] : 1);
    
    int vector_size = *INTEGER(s_vector_size);
    int row_multiple = *INTEGER(s_row_multiple);
    
    int row_tile_size = *INTEGER(s_row_tile_size);
    int col_tile_size = *INTEGER(s_col_tile_size);
    
    // --------------- calculate results ---------------
    
    if (isFloat && kernel_f == nullptr) {
        error("float kernel not available");
    }
    
    if (!isFloat && kernel_d == nullptr) {
        error("double kernel not available");
    }
    
    size_t nrow = dims[0];
    size_t ncol = dims[1];
    
    SEXP result;
    PROTECT(result = Rf_allocVector(REALSXP, ncol * ncol));
    resultUnprotectCount++;
    
    SEXP s_out_dims;
    PROTECT(s_out_dims = Rf_allocVector(INTSXP, 2));
    resultUnprotectCount++;
    
    INTEGER(s_out_dims)[0] = (int)ncol;
    INTEGER(s_out_dims)[1] = (int)ncol;
    Rf_setAttrib(result, R_DimSymbol, s_out_dims);
    
    if (isFloat) {
        Rf_setAttrib(result, Rf_install("Csingle"), Rf_ScalarLogical(1));
    }
    
    cl_int err = CL_SUCCESS;
    
    size_t multiple = vector_size * row_multiple;
    size_t filled_nrow = multiple * ((nrow + multiple - 1) / multiple);
    
    // cheap way to find least-common-multiple; not terribly slow for small row_tile_size & col_tile_size
    size_t gcd = col_tile_size < row_tile_size ? col_tile_size : row_tile_size;
    while (gcd > 1 && col_tile_size % gcd != 0 && row_tile_size % gcd != 0) gcd--;
    size_t lcm = col_tile_size * row_tile_size / gcd;
    
    size_t filled_ncol = (ncol + lcm - 1) / lcm;
    
    filled_ncol = work_item_sizes[0] * ((filled_ncol + work_item_sizes[0] - 1) / work_item_sizes[0]);
    filled_ncol = work_item_sizes[1] * ((filled_ncol + work_item_sizes[1] - 1) / work_item_sizes[1]);
    
    filled_ncol *= lcm;

    if (gTrace) {
        CERR << "opencl_calc_x_C: nrow = " << nrow << ", ncol = " << ncol << ", filled_nrow = " << filled_nrow << ", filled_ncol = " << filled_ncol << std::endl;
    }

    bool needsFill = nrow != filled_nrow || ncol != filled_ncol;
    
    if (gTrace) {
        CERR << setprecision(12);
    }
    
    if (gTrace && nrow * ncol <= 256) {
        CERR << "x:" << endl;
        for (size_t row = 0; row < nrow; row++) {
            for (size_t col = 0; col < ncol; col++) {
                CERR << x[row + col * nrow] << "\t";
            }
            CERR << endl;
        }
        CERR << endl;
    }
    
    if (needsFill && fillOnHost) {

        if (isFloat) {
            if (gTrace) {
                CERR << "needsFill && fillOnHost && isFloat" << endl;
            }

            float *inMatrix = (float *)calloc(filled_nrow * filled_ncol, sizeof(float));
            float *outMatrix = (float *)calloc(filled_ncol * filled_ncol, sizeof(float));
            
            if (inMatrix == nullptr || outMatrix == nullptr) {
                error("crossprod_naive_C: insufficient memory");
                
            } else {
                for (size_t col = 0; col < ncol; col++) {
                    float *p = inMatrix + col * filled_nrow;
                    double *q = x + col * nrow;
                    for (size_t row = 0; row < nrow; row++) {
                        *p++ = (float)*q++;
                    }
                    for (size_t row = nrow; row < filled_nrow; row++) {
                        *p++ = 0.0;
                    }
                }
                for (size_t col = ncol; col < filled_ncol; col++) {
                    float *p = inMatrix + col * filled_nrow;
                    for (size_t row = 0; row < filled_nrow; row++) {
                        *p++ = 0;
                    }
                }
                
                if (gTrace && nrow * ncol <= 256) {
                    CERR << "inMatrix:" << endl;
                    for (size_t row = 0; row < filled_nrow; row++) {
                        for (size_t col = 0; col < filled_ncol; col++) {
                            CERR << inMatrix[row + col * filled_nrow] << "\t";
                        }
                        CERR << endl;
                    }
                    CERR << endl;
                }
                
                err = opencl_calc_x(context, kernel_f, kernel_d, true, queue, inMatrix, outMatrix,
                                    (int)filled_nrow, (int)filled_ncol, work_item_sizes,
                                    vector_size, row_multiple, row_tile_size, col_tile_size, verbose);

                if (gTrace && ncol <= 16) {
                    CERR << "outMatrix:" << endl;
                    for (size_t row = 0; row < filled_ncol; row++) {
                        for (size_t col = 0; col < filled_ncol; col++) {
                            CERR << outMatrix[row + col * filled_ncol] << "\t";
                        }
                        CERR << endl;
                    }
                    CERR << endl;
                }

                for (size_t col = 0; col < ncol; col++) {
                    float *p = outMatrix + col * filled_ncol;
                    double *q = REAL(result) + col * ncol;
                    for (size_t row = 0; row < ncol; row++) {
                        *q++ = (double)*p++;
                    }
                }
            }
            
            if (inMatrix != nullptr) {
                free(inMatrix);
                inMatrix = nullptr;
            }
            
            if (outMatrix != nullptr) {
                free(outMatrix);
                outMatrix = nullptr;
            }

        } else {
            if (gTrace) {
                CERR << "needsFill && fillOnHost && !isFloat" << endl;
            }
            
            double *inMatrix = (double *)calloc(filled_nrow * filled_ncol, sizeof(double));
            double *outMatrix = (double *)calloc(filled_ncol * filled_ncol, sizeof(double));
            
            if (inMatrix == nullptr || outMatrix == nullptr) {
                error("crossprod_naive_C: insufficient memory");
                
            } else {
                for (size_t col = 0; col < ncol; col++) {
                    double *p = inMatrix + col * filled_nrow;
                    double *q = x + col * nrow;
                    for (size_t row = 0; row < nrow; row++) {
                        *p++ = *q++;
                    }
                    for (size_t row = nrow; row < filled_nrow; row++) {
                        *p++ = 0.0;
                    }
                }
                for (size_t col = ncol; col < filled_ncol; col++) {
                    double *p = inMatrix + col * filled_nrow;
                    for (size_t row = 0; row < filled_nrow; row++) {
                        *p++ = 0;
                    }
                }
                
                if (gTrace && nrow * ncol <= 256) {
                    CERR << "inMatrix:" << endl;
                    for (size_t row = 0; row < filled_nrow; row++) {
                        for (size_t col = 0; col < filled_ncol; col++) {
                            CERR << inMatrix[row + col * filled_nrow] << "\t";
                        }
                        CERR << endl;
                    }
                    CERR << endl;
                }
                
                err = opencl_calc_x(context, kernel_f, kernel_d, false, queue, inMatrix, outMatrix,
                                    (int)filled_nrow, (int)filled_ncol, work_item_sizes,
                                    vector_size, row_multiple, row_tile_size, col_tile_size, verbose);
                
                if (gTrace && ncol <= 16) {
                    CERR << "outMatrix:" << endl;
                    for (size_t row = 0; row < filled_ncol; row++) {
                        for (size_t col = 0; col < filled_ncol; col++) {
                            CERR << outMatrix[row + col * filled_ncol] << "\t";
                        }
                        CERR << endl;
                    }
                    CERR << endl;
                }
                
                for (size_t col = 0; col < ncol; col++) {
                    double *p = outMatrix + col * filled_ncol;
                    double *q = REAL(result) + col * ncol;
                    for (size_t row = 0; row < ncol; row++) {
                        *q++ = *p++;
                    }
                }
            }
            
            if (inMatrix != nullptr) {
                free(inMatrix);
                inMatrix = nullptr;
            }
            
            if (outMatrix != nullptr) {
                free(outMatrix);
                outMatrix = nullptr;
            }
        }
        
    } else {
        if (isFloat) {
            if (gTrace) {
                CERR << "!(needsFill && fillOnHost) && isFloat" << endl;
            }
            
            size_t len = XLENGTH(s_x);
            
            float *inMatrix = (float *)calloc(len, sizeof(float));
            float *outMatrix = (float *)calloc(ncol * ncol, sizeof(float));
            
            if (inMatrix == nullptr || outMatrix == nullptr) {
                error("crossprod_naive_C: insufficient memory");
                
            } else {
                float *p = inMatrix;
                double *q = x;
                for (size_t k = 0; k < len; k++) {
                    *p++ = (float)*q++;
                }
                
                if (gTrace && nrow * ncol <= 256) {
                    CERR << "inMatrix:" << endl;
                    for (size_t row = 0; row < nrow; row++) {
                        for (size_t col = 0; col < ncol; col++) {
                            CERR << inMatrix[row + col * nrow] << "\t";
                        }
                        CERR << endl;
                    }
                    CERR << endl;
                }
                
                err = opencl_calc_x(context, kernel_f, kernel_d, true, queue, inMatrix, outMatrix,
                                    dims[0], dims[1], work_item_sizes, vector_size, row_multiple,
                                    row_tile_size, col_tile_size, verbose);
                
                if (gTrace && ncol <= 16) {
                    CERR << "outMatrix:" << endl;
                    for (size_t row = 0; row < ncol; row++) {
                        for (size_t col = 0; col < ncol; col++) {
                            CERR << outMatrix[row + col * ncol] << "\t";
                        }
                        CERR << endl;
                    }
                    CERR << endl;
                }
               
                p = outMatrix;
                q = REAL(result);
                for (size_t k = 0; k < ncol * ncol; k++) {
                    *q++ = (double)*p++;
                }
            }
            
            if (inMatrix != nullptr) {
                free(inMatrix);
                inMatrix = nullptr;
            }
            
            if (outMatrix != nullptr) {
                free(outMatrix);
                outMatrix = nullptr;
            }
            
        } else {
            if (gTrace) {
                CERR << "!(needsFill && fillOnHost) && !isFloat" << endl;
            }
            
            err = opencl_calc_x(context, kernel_f, kernel_d, false, queue, x, REAL(result), dims[0], dims[1],
                                work_item_sizes, vector_size, row_multiple, row_tile_size, col_tile_size, verbose);
        }
    }
    
    if (gTrace && ncol <= 16) {
        CERR << "result:" << endl;
        for (size_t row = 0; row < ncol; row++) {
            for (size_t col = 0; col < ncol; col++) {
                CERR << REAL(result)[row + col * ncol] << "\t";
            }
            CERR << endl;
        }
        CERR << endl;
    }

    if (err != CL_SUCCESS) {
        error(clErrorToString(err).c_str());
    }
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << /*"return " << resultUnprotectCount <<*/ endl;
    
#ifdef USE_TIMING
    steady_clock::time_point end_time = steady_clock::now();
    duration<double> time_span = duration_cast<duration<double> >(end_time - start_time);
    
    double nsec = std::chrono::duration_cast<std::chrono::nanoseconds> (end_time - start_time).count();
    double gflops = (2.0 * nrow * (ncol + 0.5 * (ncol - 1.0) * ncol)) / nsec;
    
    if (gTrace || verbose) {
        CERR << "opencl_calc_x_C Elapsed: " << time_span.count() << " sec " <<
        gflops << " GFLOPS" << std::endl;
    }
#endif
    
    return(result);
}

SEXP opencl_calc_gemm_C(SEXP s_context, SEXP s_kernel_f, SEXP s_kernel_d, SEXP s_queue,
                        SEXP s_A, SEXP s_transposeA, SEXP s_B, SEXP s_transposeB, SEXP s_C,
                        SEXP s_alpha, SEXP s_beta,
                        SEXP s_work_item_sizes, SEXP s_vector_size, SEXP s_row_multiple,
                        SEXP s_row_tile_size, SEXP s_col_tile_size, SEXP s_fill_on_host,
                        SEXP s_verbose)
{
#ifdef USE_TIMING
    steady_clock::time_point start_time = steady_clock::now();
#endif
    
    if (gTrace) CERR << "opencl_calc_x_C" << endl;
    
    int resultUnprotectCount = 0;
    
    // --------------- verify arg types ---------------
    
    {
        SEXP contextClass;
        PROTECT(contextClass = Rf_getAttrib(s_context, R_ClassSymbol));
        
        if (!Rf_isNull(contextClass) && strcmp("opencl.context", CHAR(STRING_ELT(contextClass, 0))) != 0) {
            error("opencl_calc_gemm_C: wrong context class");
        }
        
        UNPROTECT(1);
    }
    
    if (!Rf_isNull(s_kernel_f))
    {
        SEXP kernelClass;
        PROTECT(kernelClass = Rf_getAttrib(s_kernel_f, R_ClassSymbol));
        
        if (!Rf_isNull(kernelClass) && strcmp("opencl.kernel", CHAR(STRING_ELT(kernelClass, 0))) != 0) {
            error("opencl_calc_gemm_C: wrong kernel_f class");
        }
        
        UNPROTECT(1);
    }
    
    if (!Rf_isNull(s_kernel_d))
    {
        SEXP kernelClass;
        PROTECT(kernelClass = Rf_getAttrib(s_kernel_d, R_ClassSymbol));
        
        if (!Rf_isNull(kernelClass) && strcmp("opencl.kernel", CHAR(STRING_ELT(kernelClass, 0))) != 0) {
            error("opencl_calc_gemm_C: wrong kernel_d class");
        }
        
        UNPROTECT(1);
    }
    
    {
        SEXP queueClass;
        PROTECT(queueClass = Rf_getAttrib(s_queue, R_ClassSymbol));
        
        if (!Rf_isNull(queueClass) && strcmp("opencl.queue", CHAR(STRING_ELT(queueClass, 0))) != 0) {
            error("opencl_calc_gemm_C: wrong queue class");
        }
        
        UNPROTECT(1);
    }
    
    // s_A
    
    if (Rf_isComplex(s_A)) {
        error("opencl_calc_gemm_C: complex A not implemented yet");
        
    } else if (Rf_isInteger(s_A)) {
        error("opencl_calc_gemm_C: integer A not implemented yet");
        
    } else if (!Rf_isReal(s_A) && !Rf_isInteger(s_A)) {
        error("opencl_calc_gemm_C: wrong A type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_A) = " << XLENGTH(s_A) << endl;
    
    SEXP s_dims_A;
    PROTECT(s_dims_A = Rf_getAttrib(s_A, R_DimSymbol));
    resultUnprotectCount++;
    
    if (Rf_isNull(s_dims_A)) {
        error("opencl_calc_gemm_C: no A dimensions");
    }
    
    SEXP s_float_A;
    PROTECT(s_float_A = Rf_getAttrib(s_A, Rf_install("Csingle")));
    resultUnprotectCount++;
    
    bool isFloatA = !Rf_isNull(s_float_A) && *LOGICAL(s_float_A);
    
    int dimCountA = Rf_length(s_dims_A);
    int *dimsA = INTEGER(s_dims_A);
    
    if (gTrace) {
        CERR << "dimsA = ";
        for (int k = 0; k < dimCountA; k++) {
            CERR << dimsA[k] << " ";
        }
        CERR << endl;
    }
    
    if (dimCountA != 2) {
        error("opencl_calc_gemm_C: wrong A dimension");
    }
    
    // s_B
    
    if (Rf_isComplex(s_B)) {
        error("opencl_calc_gemm_C: complex B not implemented yet");
        
    } else if (Rf_isInteger(s_B)) {
        error("opencl_calc_gemm_C: integer B not implemented yet");
        
    } else if (!Rf_isReal(s_B) && !Rf_isInteger(s_B)) {
        error("opencl_calc_gemm_C: wrong B type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_B) = " << XLENGTH(s_B) << endl;
    
    SEXP s_dims_B;
    PROTECT(s_dims_B = Rf_getAttrib(s_B, R_DimSymbol));
    resultUnprotectCount++;
    
    if (Rf_isNull(s_dims_B)) {
        error("opencl_calc_gemm_C: no B dimensions");
    }
    
    SEXP s_float_B;
    PROTECT(s_float_B = Rf_getAttrib(s_B, Rf_install("Csingle")));
    resultUnprotectCount++;
    
    bool isFloatB = !Rf_isNull(s_float_B) && *LOGICAL(s_float_B);
    
    int dimCountB = Rf_length(s_dims_B);
    int *dimsB = INTEGER(s_dims_B);
    
    if (gTrace) {
        CERR << "dimsB = ";
        for (int k = 0; k < dimCountB; k++) {
            CERR << dimsB[k] << " ";
        }
        CERR << endl;
    }
    
    if (dimCountB != 2) {
        error("opencl_calc_gemm_C: wrong B dimension");
    }
    
    // s_C
    
    bool cIsNA = false;
    
    if (Rf_isLogical(s_C) && XLENGTH(s_C) > 0 && *LOGICAL(s_C) == NA_LOGICAL) {
        cIsNA = true;
        
    } else if (Rf_isComplex(s_C)) {
        error("opencl_calc_gemm_C: complex C not implemented yet");
        
    } else if (Rf_isInteger(s_C)) {
        error("opencl_calc_gemm_C: integer C not implemented yet");
        
    } else if (!Rf_isReal(s_C) && !Rf_isInteger(s_C)) {
        error("opencl_calc_gemm_C: wrong C type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_C) = " << XLENGTH(s_C) << endl;
    
    SEXP s_dims_C;
    PROTECT(s_dims_C = Rf_getAttrib(s_C, R_DimSymbol));
    resultUnprotectCount++;
    
    int dimCountC = Rf_isNull(s_dims_C) ? 0 : Rf_length(s_dims_C);
    int *dimsC = Rf_isNull(s_dims_C) ? nullptr : INTEGER(s_dims_C);
    
    if (!cIsNA && Rf_isNull(s_dims_C)) {
        error("opencl_calc_gemm_C: no C dimensions");
    }
    
    SEXP s_float_C;
    PROTECT(s_float_C = Rf_getAttrib(s_C, Rf_install("Csingle")));
    resultUnprotectCount++;
    
    bool isFloatC = !Rf_isNull(s_float_C) && *LOGICAL(s_float_C);
    
    if (gTrace) {
        if (cIsNA) {
            CERR << "C = NA" << endl;
            
        } else {
            CERR << "dimsC = ";
            for (int k = 0; k < dimCountC; k++) {
                CERR << dimsC[k] << " ";
            }
            CERR << endl;
        }
    }
    
    if (!cIsNA && dimCountC != 2) {
        error("opencl_calc_gemm_C: wrong C dimension");
    }
    
    // others
    
    if (!Rf_isReal(s_alpha)) {
        error("opencl_calc_gemm_C: wrong alpha type");
    }
    
    if (!Rf_isReal(s_beta)) {
        error("opencl_calc_gemm_C: wrong beta type");
    }
    
    if (!Rf_isLogical(s_transposeA)) {
        error("opencl_calc_gemm_C: wrong transposeA type");
    }
    
    if (!Rf_isLogical(s_transposeB)) {
        error("opencl_calc_gemm_C: wrong transposeB type");
    }
    
    if (!Rf_isInteger(s_work_item_sizes)) {
        error("opencl_calc_gemm_C: wrong work_item_sizes class");
    }
    
    if (!Rf_isInteger(s_vector_size)) {
        error("opencl_calc_gemm_C: wrong vector_size class");
    }
    
    if (!Rf_isInteger(s_row_multiple)) {
        error("opencl_calc_gemm_C: wrong row_multiple class");
    }
    
    if (!Rf_isInteger(s_row_tile_size)) {
        error("opencl_calc_gemm_C: wrong row_tile.size class");
    }
    
    if (!Rf_isInteger(s_col_tile_size)) {
        error("opencl_calc_gemm_C: wrong col_tile.size class");
    }
    
    bool fillOnHost = !Rf_isNull(s_fill_on_host) && *LOGICAL(s_fill_on_host);
    
    bool verbose = !Rf_isNull(s_verbose) && *LOGICAL(s_verbose);
    
    // --------------- get args ---------------
    
    bool isFloat = isFloatA || isFloatB || isFloatC;
    
    cl_context context = (cl_context)R_ExternalPtrAddr(s_context);
    
    if (context == nullptr) {
        error("opencl_calc_gemm_C: null context");
    }
    
    cl_kernel kernel_f = Rf_isNull(s_kernel_f) ? nullptr : (cl_kernel)R_ExternalPtrAddr(s_kernel_f);
    
    if (isFloat && kernel_f == nullptr) {
        error("opencl_calc_gemm_C: null kernel_f");
    }
    
    cl_kernel kernel_d = Rf_isNull(s_kernel_d) ? nullptr : (cl_kernel)R_ExternalPtrAddr(s_kernel_d);
    
    if (!isFloat && kernel_d == nullptr) {
        error("opencl_calc_gemm_C: null kernel_d");
    }
    
    cl_command_queue queue = (cl_command_queue)R_ExternalPtrAddr(s_queue);
    
    const double *A = REAL(s_A);
    const double *B = REAL(s_B);
    const double *C = cIsNA ? nullptr : REAL(s_C);
    
    if (gTrace && XLENGTH(s_A) <= 256) {
        for (int k = 0; k < XLENGTH(s_A); k++) {
            CERR << "A[" << k << "] = " << A[k] << endl;
        }
    }
    
    double alpha = *REAL(s_alpha);
    double beta = *REAL(s_beta);
    
    bool transposeA = !Rf_isNull(s_transposeA) && *LOGICAL(s_transposeA);
    bool transposeB = !Rf_isNull(s_transposeB) && *LOGICAL(s_transposeB);
    
    int *p = INTEGER(s_work_item_sizes);
    vector<size_t> work_item_sizes;
    work_item_sizes.push_back(XLENGTH(s_work_item_sizes) > 0 ? p[0] : 1);
    work_item_sizes.push_back(XLENGTH(s_work_item_sizes) > 1 ? p[1] : 1);
    work_item_sizes.push_back(XLENGTH(s_work_item_sizes) > 2 ? p[2] : 1);
    
    int vector_size = *INTEGER(s_vector_size);
    int row_multiple = *INTEGER(s_row_multiple);
    
    int row_tile_size = *INTEGER(s_row_tile_size);
    int col_tile_size = *INTEGER(s_col_tile_size);
    
    // --------------- calculate results ---------------
    
    if (isFloat && kernel_f == nullptr) {
        error("opencl_calc_gemm_C: float kernel not available");
    }
    
    if (!isFloat && kernel_d == nullptr) {
        error("opencl_calc_gemm_C: double kernel not available");
    }
    
    size_t nrowA = dimsA[0];
    size_t ncolA = dimsA[1];
    
    size_t nrowB = dimsB[0];
    size_t ncolB = dimsB[1];
    
    size_t outRow = transposeA ? ncolA : nrowA;
    size_t outCol = transposeB ? nrowB : ncolB;

    SEXP result;
    PROTECT(result = Rf_allocVector(REALSXP, outRow * outCol));
    resultUnprotectCount++;
    
    SEXP s_out_dims;
    PROTECT(s_out_dims = Rf_allocVector(INTSXP, 2));
    resultUnprotectCount++;
    
    INTEGER(s_out_dims)[0] = (int)outRow;
    INTEGER(s_out_dims)[1] = (int)outCol;
    Rf_setAttrib(result, R_DimSymbol, s_out_dims);
    
    if (isFloatA || isFloatB || isFloatC) {
        Rf_setAttrib(result, Rf_install("Csingle"), Rf_ScalarLogical(1));
    }
    
    cl_int err = CL_SUCCESS;
    
    // values after transposing, if any (atia)
    size_t full_nrowA = 0;
    size_t full_ncolA = 0;
    size_t full_nrowB = 0;
    size_t full_ncolB = 0;

    getFullSizes_atia(transposeA, transposeB,
                           full_nrowA, full_ncolA, full_nrowB, full_ncolB,
                           nrowA, ncolA, nrowB, ncolB,
                           vector_size, row_multiple, row_tile_size, col_tile_size,
                           work_item_sizes);
    
    if (gTrace) {
        CERR << "opencl_calc_gemm_C: nrowA = " << nrowA << ", ncolA = " << ncolA << ", full_nrowA = " << full_nrowA << ", full_ncolA = " << full_ncolA << std::endl;
    }
    
    if (gTrace) {
        CERR << "                    nrowB = " << nrowB << ", ncolB = " << ncolB << ", full_nrowB = " << full_nrowB << ", full_ncolB = " << full_ncolB << std::endl;
    }
    
    // matrix out
    
    size_t full_nrow_out = transposeA ? full_ncolA : full_nrowA;
    size_t full_ncol_out = transposeB ? full_nrowB : full_ncolB;

    bool needsFill = nrowA != full_nrowA || ncolA != full_ncolA || nrowB != full_nrowB || ncolB != full_ncolB;
    
    if (gTrace) {
        CERR << setprecision(12);
    }
    
    if (gTrace && nrowA * ncolA <= 256) {
        CERR << "A:" << endl;
        for (size_t row = 0; row < nrowA; row++) {
            for (size_t col = 0; col < ncolA; col++) {
                CERR << A[row + col * nrowA] << "\t";
            }
            CERR << endl;
        }
        CERR << endl;
    }
    
    if (needsFill && fillOnHost) {
        
        if (isFloat) {
            if (gTrace) {
                CERR << "needsFill && fillOnHost && isFloat" << endl;
            }
            
            float *inMatrixA = (float *)calloc(full_nrowA * full_ncolA, sizeof(float));
            float *inMatrixB = (float *)calloc(full_nrowB * full_ncolB, sizeof(float));
            float *outMatrix = (float *)calloc(full_nrow_out * full_ncol_out, sizeof(float));
            
            if (inMatrixA == nullptr || inMatrixB == nullptr || outMatrix == nullptr) {
                error("opencl_calc_gemm_C: insufficient memory");
                
            } else {
                for (size_t col = 0; col < ncolA; col++) {
                    float *p = inMatrixA + col * full_nrowA;
                    const double *q = A + col * nrowA;
                    for (size_t row = 0; row < nrowA; row++) {
                        *p++ = (float)*q++;
                    }
                    for (size_t row = nrowA; row < full_nrowA; row++) {
                        *p++ = 0.0;
                    }
                }
                for (size_t col = ncolA; col < full_ncolA; col++) {
                    float *p = inMatrixA + col * full_nrowA;
                    for (size_t row = 0; row < full_nrowA; row++) {
                        *p++ = 0;
                    }
                }
                
                for (size_t col = 0; col < ncolB; col++) {
                    float *p = inMatrixB + col * full_nrowB;
                    const double *q = B + col * nrowB;
                    for (size_t row = 0; row < nrowB; row++) {
                        *p++ = (float)*q++;
                    }
                    for (size_t row = nrowB; row < full_nrowB; row++) {
                        *p++ = 0.0;
                    }
                }
                for (size_t col = ncolB; col < full_ncolB; col++) {
                    float *p = inMatrixB + col * full_nrowB;
                    for (size_t row = 0; row < full_nrowB; row++) {
                        *p++ = 0;
                    }
                }
                
                if (C == nullptr) {
                    memset(outMatrix, 0, full_nrow_out * full_ncol_out * sizeof(float));
                    for (size_t k = 0; k < full_nrow_out * full_ncol_out; k++) outMatrix[k] = 0.001f;
                    
                } else {
                    for (size_t col = 0; col < outCol; col++) {
                        float *p = outMatrix + col * full_nrow_out;
                        const double *q = C + col * outRow;
                        for (size_t row = 0; row < outRow; row++) {
                            *p++ = (float)*q++;
                        }
                        for (size_t row = outRow; row < full_nrow_out; row++) {
                            *p++ = 0.0;
                        }
                    }
                    for (size_t col = outCol; col < full_ncol_out; col++) {
                        float *p = outMatrix + col * full_nrow_out;
                        for (size_t row = 0; row < full_nrow_out; row++) {
                            *p++ = 0;
                        }
                    }
                }
                
                if (gTrace && nrowA * ncolA <= 256) {
                    CERR << "inMatrixA:" << endl;
                    for (size_t row = 0; row < full_nrowA; row++) {
                        for (size_t col = 0; col < full_ncolA; col++) {
                            CERR << inMatrixA[row + col * full_nrowA] << "\t";
                        }
                        CERR << endl;
                    }
                    CERR << endl;
                }
                
                if (gTrace && nrowB * ncolB <= 256) {
                    CERR << "inMatrixB:" << endl;
                    for (size_t row = 0; row < full_nrowB; row++) {
                        for (size_t col = 0; col < full_ncolB; col++) {
                            CERR << inMatrixB[row + col * full_nrowB] << "\t";
                        }
                        CERR << endl;
                    }
                    CERR << endl;
                }
                
                err = opencl_calc_gemm(context, kernel_f, kernel_d, isFloat, queue,
                                       inMatrixA, (int)full_nrowA, (int)full_ncolA, transposeA,
                                       inMatrixB, (int)full_nrowB, (int)full_ncolB, transposeB,
                                       alpha, beta, outMatrix,
                                       work_item_sizes, vector_size, row_multiple,
                                       row_tile_size, col_tile_size, verbose);
                
                if (gTrace && outCol <= 16) {
                    CERR << "outMatrix:" << endl;
                    for (size_t row = 0; row < full_nrow_out; row++) {
                        for (size_t col = 0; col < full_ncol_out; col++) {
                            CERR << outMatrix[row + col * full_nrow_out] << "\t";
                        }
                        CERR << endl;
                    }
                    CERR << endl;
                }
                
                for (size_t col = 0; col < outCol; col++) {
                    float *p = outMatrix + col * full_nrow_out;
                    double *q = REAL(result) + col * outRow;
                    for (size_t row = 0; row < outRow; row++) {
                        *q++ = (double)*p++;
                    }
                }
            }
            
            if (inMatrixA != nullptr) {
                free(inMatrixA);
                inMatrixA = nullptr;
            }
            
            if (inMatrixB != nullptr) {
                free(inMatrixB);
                inMatrixB = nullptr;
            }
            
            if (outMatrix != nullptr) {
                free(outMatrix);
                outMatrix = nullptr;
            }
            
        } else {
            if (gTrace) {
                CERR << "needsFill && fillOnHost && !isFloat" << endl;
            }
            
            double *inMatrixA = (double *)calloc(full_nrowA * full_ncolA, sizeof(double));
            double *inMatrixB = (double *)calloc(full_nrowB * full_ncolB, sizeof(double));
            double *outMatrix = (double *)calloc(full_nrow_out * full_ncol_out, sizeof(double));
            
            if (inMatrixA == nullptr || inMatrixB == nullptr || outMatrix == nullptr) {
                error("opencl_calc_gemm_C: insufficient memory");
                
            } else {
                for (size_t col = 0; col < ncolA; col++) {
                    double *p = inMatrixA + col * full_nrowA;
                    const double *q = A + col * nrowA;
                    for (size_t row = 0; row < nrowA; row++) {
                        *p++ = *q++;
                    }
                    for (size_t row = nrowA; row < full_nrowA; row++) {
                        *p++ = 0.0;
                    }
                }
                for (size_t col = ncolA; col < full_ncolA; col++) {
                    double *p = inMatrixA + col * full_nrowA;
                    for (size_t row = 0; row < full_nrowA; row++) {
                        *p++ = 0;
                    }
                }
                
                for (size_t col = 0; col < ncolB; col++) {
                    double *p = inMatrixB + col * full_nrowB;
                    const double *q = B + col * nrowB;
                    for (size_t row = 0; row < nrowB; row++) {
                        *p++ = *q++;
                    }
                    for (size_t row = nrowB; row < full_nrowB; row++) {
                        *p++ = 0.0;
                    }
                }
                for (size_t col = ncolB; col < full_ncolB; col++) {
                    double *p = inMatrixB + col * full_nrowB;
                    for (size_t row = 0; row < full_nrowB; row++) {
                        *p++ = 0;
                    }
                }
                
                if (C == nullptr) {
                    memset(outMatrix, 0, full_nrow_out * full_ncol_out * sizeof(double));
                    
                } else {
                    for (size_t col = 0; col < outCol; col++) {
                        double *p = outMatrix + col * full_nrow_out;
                        const double *q = C + col * outRow;
                        for (size_t row = 0; row < outRow; row++) {
                            *p++ = *q++;
                        }
                        for (size_t row = outRow; row < full_nrow_out; row++) {
                            *p++ = 0.0;
                        }
                    }
                    for (size_t col = outCol; col < full_ncol_out; col++) {
                        double *p = outMatrix + col * full_nrow_out;
                        for (size_t row = 0; row < full_nrow_out; row++) {
                            *p++ = 0.0;
                        }
                    }
                }
                
                if (gTrace && nrowA * ncolA <= 256) {
                    CERR << "inMatrixA:" << endl;
                    for (size_t row = 0; row < full_nrowA; row++) {
                        for (size_t col = 0; col < full_ncolA; col++) {
                            CERR << inMatrixA[row + col * full_nrowA] << "\t";
                        }
                        CERR << endl;
                    }
                    CERR << endl;
                }
                
                if (gTrace && nrowB * ncolB <= 256) {
                    CERR << "inMatrixB:" << endl;
                    for (size_t row = 0; row < full_nrowB; row++) {
                        for (size_t col = 0; col < full_ncolB; col++) {
                            CERR << inMatrixB[row + col * full_nrowB] << "\t";
                        }
                        CERR << endl;
                    }
                    CERR << endl;
                }
                
                err = opencl_calc_gemm(context, kernel_f, kernel_d, isFloat, queue,
                                       inMatrixA, (int)full_nrowA, (int)full_ncolA, transposeA,
                                       inMatrixB, (int)full_nrowB, (int)full_ncolB, transposeB,
                                       alpha, beta, outMatrix,
                                       work_item_sizes, vector_size, row_multiple,
                                       row_tile_size, col_tile_size, verbose);
                
                if (gTrace && outCol <= 16) {
                    CERR << "outMatrix:" << endl;
                    for (size_t row = 0; row < full_nrow_out; row++) {
                        for (size_t col = 0; col < full_ncol_out; col++) {
                            CERR << outMatrix[row + col * full_nrow_out] << "\t";
                        }
                        CERR << endl;
                    }
                    CERR << endl;
                }
                
                for (size_t col = 0; col < outCol; col++) {
                    double *p = outMatrix + col * full_nrow_out;
                    double *q = REAL(result) + col * outRow;
                    for (size_t row = 0; row < outRow; row++) {
                        *q++ = (double)*p++;
                    }
                }
            }
            
            if (inMatrixA != nullptr) {
                free(inMatrixA);
                inMatrixA = nullptr;
            }
            
            if (inMatrixB != nullptr) {
                free(inMatrixB);
                inMatrixB = nullptr;
            }
            
            if (outMatrix != nullptr) {
                free(outMatrix);
                outMatrix = nullptr;
            }
        }
        
    } else {
        if (isFloat) {
            if (gTrace) {
                CERR << "!(needsFill && fillOnHost) && isFloat" << endl;
            }
            
            float *inMatrixA = (float *)calloc(nrowA * ncolA, sizeof(float));
            float *inMatrixB = (float *)calloc(nrowB * ncolB, sizeof(float));
            float *outMatrix = (float *)calloc(outRow * outCol, sizeof(float));
            
            if (inMatrixA == nullptr || inMatrixB == nullptr || outMatrix == nullptr) {
                error("opencl_calc_gemm_C: insufficient memory");
                
            } else {
                {
                    float *p = inMatrixA;
                    const double *q = A;
                    for (size_t k = 0; k < nrowA * ncolA; k++) {
                        *p++ = (float)*q++;
                    }
                }
                
                {
                    float *p = inMatrixB;
                    const double *q = B;
                    for (size_t k = 0; k < nrowB * ncolB; k++) {
                        *p++ = (float)*q++;
                    }
                }

                if (C == nullptr) {
                    memset(outMatrix, 0, outRow * outCol * sizeof(float));
                    
                } else {
                    float *p = outMatrix;
                    const double *q = C;
                    for (size_t k = 0; k < outRow * outCol; k++) {
                        *p++ = (float)*q++;
                    }
                }

                if (gTrace && nrowA * ncolA <= 256) {
                    CERR << "inMatrixA:" << endl;
                    for (size_t row = 0; row < nrowA; row++) {
                        for (size_t col = 0; col < ncolA; col++) {
                            CERR << inMatrixA[row + col * nrowA] << "\t";
                        }
                        CERR << endl;
                    }
                    CERR << endl;
                }
                
                if (gTrace && nrowB * ncolB <= 256) {
                    CERR << "inMatrixB:" << endl;
                    for (size_t row = 0; row < nrowB; row++) {
                        for (size_t col = 0; col < ncolB; col++) {
                            CERR << inMatrixB[row + col * nrowB] << "\t";
                        }
                        CERR << endl;
                    }
                    CERR << endl;
                }
                
                err = opencl_calc_gemm(context, kernel_f, kernel_d, isFloat, queue,
                                       inMatrixA, (int)nrowA, (int)ncolA, transposeA,
                                       inMatrixB, (int)nrowB, (int)ncolB, transposeB,
                                       alpha, beta, outMatrix,
                                       work_item_sizes, vector_size, row_multiple,
                                       row_tile_size, col_tile_size, verbose);
                
                if (gTrace && outCol <= 16) {
                    CERR << "outMatrix:" << endl;
                    for (size_t row = 0; row < outRow; row++) {
                        for (size_t col = 0; col < outCol; col++) {
                            CERR << outMatrix[row + col * outRow] << "\t";
                        }
                        CERR << endl;
                    }
                    CERR << endl;
                }
                
                for (size_t col = 0; col < outCol; col++) {
                    float *p = outMatrix + col * outRow;
                    double *q = REAL(result) + col * outRow;
                    for (size_t row = 0; row < outRow; row++) {
                        *q++ = (double)*p++;
                    }
                }
            }
            
            if (inMatrixA != nullptr) {
                free(inMatrixA);
                inMatrixA = nullptr;
            }
            
            if (inMatrixB != nullptr) {
                free(inMatrixB);
                inMatrixB = nullptr;
            }
            
            if (outMatrix != nullptr) {
                free(outMatrix);
                outMatrix = nullptr;
            }
            
        } else {
            if (gTrace) {
                CERR << "!(needsFill && fillOnHost) && !isFloat" << endl;
            }
            
            if (C == nullptr) {
                memset(REAL(result), 0, outRow * outCol * sizeof(double));
                
            } else {
                double *p = REAL(result);
                const double *q = C;
                for (size_t k = 0; k < outRow * outCol; k++) {
                    *p++ = *q++;
                }
            }
            
            err = opencl_calc_gemm(context, kernel_f, kernel_d, isFloat, queue,
                                   A, (int)nrowA, (int)ncolA, transposeA,
                                   B, (int)nrowB, (int)ncolB, transposeB,
                                   alpha, beta, REAL(result),
                                   work_item_sizes, vector_size, row_multiple,
                                   row_tile_size, col_tile_size, verbose);
        }
    }
    
    if (gTrace && outRow * outCol <= 256) {
        CERR << "result:" << endl;
        for (size_t row = 0; row < outRow; row++) {
            for (size_t col = 0; col < outCol; col++) {
                CERR << REAL(result)[row + col * outRow] << "\t";
            }
            CERR << endl;
        }
        CERR << endl;
    }
    
    if (err != CL_SUCCESS) {
        error(clErrorToString(err).c_str());
    }
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << /*"return " << resultUnprotectCount <<*/ endl;
    
#ifdef USE_TIMING
    steady_clock::time_point end_time = steady_clock::now();
    duration<double> time_span = duration_cast<duration<double> >(end_time - start_time);
    
    double nsec = std::chrono::duration_cast<std::chrono::nanoseconds> (end_time - start_time).count();
    double gflops = (2.0 * nrow * (ncol + 0.5 * (ncol - 1.0) * ncol)) / nsec;
    
    if (gTrace || verbose) {
        CERR << "opencl_calc_gemm_C Elapsed: " << time_span.count() << " sec " <<
        gflops << " GFLOPS" << std::endl;
    }
#endif
    
    return(result);
}

#endif


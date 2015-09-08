//
//  multiblas_crossprod.cpp
//  multiBLAS.XC
//
//  Created by michael on 7/2/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#include "multiblas_crossprod.h"
#include "opencl_info.h"
#include "crossprod_naive.h"
#include "crossprod_r.h"
#include "crossprod_blas.h"
#include "crossprod_clblas.h"

#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#if RPACKAGE

#include "shim.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <vector>
#include <cstddef>

using namespace std;

// ========== Local Headers ========================================================================

// ========== Globals ==============================================================================

extern bool gTrace;    // for debugging

// ========== Functions ============================================================================

SEXP crossprod_naive_C(SEXP s_x)
{
    if (gTrace) CERR << "crossprod_naive_C" << endl;
    
    int resultUnprotectCount = 0;
    
    // --------------- verify arg type ---------------
    
    if (gTrace) CERR << "verify arg type" << endl;
    
    if (Rf_isComplex(s_x)) {
        Rf_error("crossprod_naive_C: complex x not implemented yet");
        
    } else if (Rf_isInteger(s_x)) {
        Rf_error("crossprod_naive_C: integer x not implemented yet");
        
    } else if (!Rf_isReal(s_x) && !Rf_isInteger(s_x)) {
        Rf_error("crossprod_naive_C: wrong x type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_x) = " << XLENGTH(s_x) << endl;
    
    SEXP s_dims;
    PROTECT(s_dims = Rf_getAttrib(s_x, R_DimSymbol));
    resultUnprotectCount++;
    
    if (Rf_isNull(s_dims)) {
        Rf_error("crossprod_naive_C: no dimensions");
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
        Rf_error("crossprod_naive_C: wrong dimension");
    }
    
    // --------------- get arg ---------------
    
    double *x = REAL(s_x);
    
    if (gTrace && XLENGTH(s_x) <= 256) {
        for (int k = 0; k < XLENGTH(s_x); k++) {
            CERR << "x[" << k << "] = " << x[k] << endl;
        }
    }
    
    // --------------- calculate results ---------------
    
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
    
    if (isFloat) {
        size_t len = XLENGTH(s_x);
        
        float *inMatrix = (float *)calloc(len, sizeof(float));
        float *outMatrix = (float *)calloc(ncol * ncol, sizeof(float));
        
        if (inMatrix == nullptr || outMatrix == nullptr) {
            Rf_error("crossprod_naive_C: insufficient memory");
            
        } else {
            float *p = inMatrix;
            double *q = x;
            for (size_t k = 0; k < len; k++) {
                *p++ = (float)*q++;
            }
            
            crossprod_naive(inMatrix, outMatrix, dims[0], dims[1]);
            
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
        crossprod_naive(x, REAL(result), dims[0], dims[1]);
    }
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << endl;
    
    return(result);
}

SEXP crossprod_blas_C(SEXP s_x)
{
    if (gTrace) CERR << "crossprod_blas_C" << endl;
    
    int resultUnprotectCount = 0;
    
    // --------------- verify arg type ---------------
    
    if (gTrace) CERR << "verify arg type" << endl;
    
    if (Rf_isComplex(s_x)) {
        Rf_error("crossprod_blas_C: complex x not implemented yet");
        
    } else if (Rf_isInteger(s_x)) {
        Rf_error("crossprod_blas_C: integer x not implemented yet");
        
    } else if (!Rf_isReal(s_x) && !Rf_isInteger(s_x)) {
        Rf_error("crossprod_blas_C: wrong x type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_x) = " << XLENGTH(s_x) << endl;
    
    SEXP s_dims;
    PROTECT(s_dims = Rf_getAttrib(s_x, R_DimSymbol));
    resultUnprotectCount++;
    
    if (Rf_isNull(s_dims)) {
        Rf_error("crossprod_blas_C: no dimensions");
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
        Rf_error("crossprod_blas_C: wrong dimension");
    }
    
    // --------------- get arg ---------------
    
    double *x = REAL(s_x);
    
    if (gTrace && XLENGTH(s_x) <= 256) {
        for (int k = 0; k < XLENGTH(s_x); k++) {
            CERR << "x[" << k << "] = " << x[k] << endl;
        }
    }
    
    // --------------- calculate results ---------------
    
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
    
    if (isFloat) {
        if (!cblas_ssyrk_available()) {
            error("crossprod_blas_C: single-precision not available");
        }
        
        size_t len = XLENGTH(s_x);
        
        float *inMatrix = (float *)calloc(len, sizeof(float));
        float *outMatrix = (float *)calloc(ncol * ncol, sizeof(float));
        
        if (inMatrix == nullptr || outMatrix == nullptr) {
            Rf_error("crossprod_blas_C: insufficient memory");
            
        } else {
            float *p = inMatrix;
            double *q = x;
            for (size_t k = 0; k < len; k++) {
                *p++ = (float)*q++;
            }
            
            crossprod_blas_f(inMatrix, outMatrix, dims[0], dims[1]);
            
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
        if (!cblas_dsyrk_available()) {
            error("crossprod_blas_C: double-precision not available");
        }
        
        crossprod_blas_d(x, REAL(result), dims[0], dims[1]);
    }
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << /*"return " << resultUnprotectCount <<*/ endl;
    
    return(result);
}

SEXP crossprod_r_C(SEXP s_x)
{
    if (gTrace) CERR << "crossprod_r_C" << endl;
    
    int resultUnprotectCount = 0;
    
    // --------------- verify arg type ---------------
    
    if (gTrace) CERR << "verify arg type" << endl;
    
    if (Rf_isComplex(s_x)) {
        Rf_error("crossprod_r_C: complex x not implemented yet");
        
    } else if (Rf_isInteger(s_x)) {
        Rf_error("crossprod_r_C: integer x not implemented yet");
        
    } else if (!Rf_isReal(s_x) && !Rf_isInteger(s_x)) {
        Rf_error("crossprod_r_C: wrong x type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_x) = " << XLENGTH(s_x) << endl;
    
    SEXP s_dims;
    PROTECT(s_dims = Rf_getAttrib(s_x, R_DimSymbol));
    resultUnprotectCount++;
    
    if (Rf_isNull(s_dims)) {
        Rf_error("crossprod_r_C: no dimensions");
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
        Rf_error("crossprod_r_C: wrong dimension");
    }
    
    // --------------- get arg ---------------
    
    double *x = REAL(s_x);
    
    if (gTrace && XLENGTH(s_x) <= 256) {
        for (int k = 0; k < XLENGTH(s_x); k++) {
            CERR << "x[" << k << "] = " << x[k] << endl;
        }
    }
    
    // --------------- calculate results ---------------
    
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
    
    if (isFloat) {
        error("crossprod_r_C: single-precision not available");
        
    } else {
        crossprod_r_d(x, REAL(result), dims[0], dims[1]);
    }
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << /*"return " << resultUnprotectCount <<*/ endl;
    
    return(result);
}

SEXP crossprod_clblas_C(SEXP s_device, SEXP s_x)
{
    if (gTrace) CERR << "crossprod_clblas_C" << endl;
    
    int resultUnprotectCount = 0;
    
    // --------------- verify arg types ---------------
    
    if (gTrace) CERR << "verify arg types" << endl;
    
    {
        SEXP deviceClass;
        PROTECT(deviceClass = Rf_getAttrib(s_device, R_ClassSymbol));
        
        if (!Rf_isNull(deviceClass) && strcmp("opencl.device", CHAR(STRING_ELT(deviceClass, 0))) != 0) {
            Rf_error("crossprod_clblas_C: wrong device class");
        }
        
        UNPROTECT(1);
    }

    if (Rf_isComplex(s_x)) {
        Rf_error("crossprod_clblas_C: complex x not implemented yet");
        
    } else if (Rf_isInteger(s_x)) {
        Rf_error("crossprod_clblas_C: integer x not implemented yet");
        
    } else if (!Rf_isReal(s_x) && !Rf_isInteger(s_x)) {
        Rf_error("crossprod_clblas_C: wrong x type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_x) = " << XLENGTH(s_x) << endl;
    
    SEXP s_dims;
    PROTECT(s_dims = Rf_getAttrib(s_x, R_DimSymbol));
    resultUnprotectCount++;
    
    if (Rf_isNull(s_dims)) {
        Rf_error("crossprod_clblas_C: no dimensions");
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
        Rf_error("crossprod_clblas_C: wrong dimension");
    }
    
    // --------------- get arg ---------------
    
    const int idElementIndex = 3;
    SEXP s_id = VECTOR_ELT(s_device, idElementIndex);
    
    cl_device_id device_id = (cl_device_id)R_ExternalPtrAddr(s_id);
    
    if (device_id == nullptr) {
        Rf_error("crossprod_clblas_C: null cl_device_id");
    }
    
    if (gTrace) CERR << "cl_device_id = " << hex << (unsigned long long)(void *)device_id << dec << endl;
    
    double *x = REAL(s_x);
    
    if (gTrace && XLENGTH(s_x) <= 256) {
        for (int k = 0; k < XLENGTH(s_x); k++) {
            CERR << "x[" << k << "] = " << x[k] << endl;
        }
    }
    
    // --------------- calculate results ---------------
    
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
    
    ErrorStatus errorStatus;

    if (isFloat) {
        size_t len = XLENGTH(s_x);
        
        float *inMatrix = (float *)calloc(len, sizeof(float));
        float *outMatrix = (float *)calloc(ncol * ncol, sizeof(float));
        
        if (inMatrix == nullptr || outMatrix == nullptr) {
            Rf_error("crossprod_clblas_C: insufficient memory");
            
        } else {
            float *p = inMatrix;
            double *q = x;
            for (size_t k = 0; k < len; k++) {
                *p++ = (float)*q++;
            }
            
            errorStatus = crossprod_clblas_f(device_id, inMatrix, outMatrix, dims[0], dims[1]);
            
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
        errorStatus = crossprod_clblas_d(device_id, x, REAL(result), dims[0], dims[1]);
    }
    
    if (gTrace) {
        CERR << "ErrorStatus('" << clErrorToString(errorStatus.error) << "', '" <<
            clblasErrorToString(errorStatus.status) << "')" << endl;
    }
    
    if (errorStatus.error != CL_SUCCESS) {
        Rf_error(clErrorToString(errorStatus.error).c_str());
        
    } else if (errorStatus.status != clblasSuccess) {
        Rf_error(clblasErrorToString(errorStatus.status).c_str());
    }
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << endl;
    
    return(result);
}

#endif

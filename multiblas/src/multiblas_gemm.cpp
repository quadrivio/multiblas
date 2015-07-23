//
//  multiblas_gemm.cpp
//  multiBLAS.XC
//
//  Created by michael on 7/23/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#include "multiblas_gemm.h"
#include "opencl_info.h"
#include "gemm_naive.h"
#include "gemm_blas.h"
#include "gemm_clblas.h"

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

using namespace std;

// ========== Local Headers ========================================================================

// ========== Globals ==============================================================================

extern bool gTrace;    // for debugging

// ========== Functions ============================================================================

SEXP gemm_naive_C(SEXP s_A, SEXP s_transposeA, SEXP s_B, SEXP s_transposeB, SEXP s_C, SEXP s_alpha, SEXP s_beta)
{
    SEXP result = PROTECT(Rf_allocVector(INTSXP, 1));
    *INTEGER(result) = NA_INTEGER;
    UNPROTECT(1);

#if 0
    if (gTrace) CERR << "gemm_naive_C" << endl;
    
    int resultUnprotectCount = 0;
    
    // --------------- verify arg type ---------------
    
    if (gTrace) CERR << "verify arg type" << endl;
    
    if (Rf_isComplex(s_x)) {
        error("gemm_naive_C: complex x not implemented yet");
        
    } else if (Rf_isInteger(s_x)) {
        error("gemm_naive_C: integer x not implemented yet");
        
    } else if (!Rf_isReal(s_x) && !Rf_isInteger(s_x)) {
        error("gemm_naive_C: wrong x type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_x) = " << XLENGTH(s_x) << endl;
    
    SEXP s_dims;
    PROTECT(s_dims = Rf_getAttrib(s_x, R_DimSymbol));
    resultUnprotectCount++;
    
    if (Rf_isNull(s_dims)) {
        error("gemm_naive_C: no dimensions");
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
        error("gemm_naive_C: wrong dimension");
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
            error("gemm_naive_C: insufficient memory");
            
        } else {
            float *p = inMatrix;
            double *q = x;
            for (size_t k = 0; k < len; k++) {
                *p++ = (float)*q++;
            }
            
            gemm_naive(inMatrix, outMatrix, dims[0], dims[1]);
            
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
        gemm_naive(x, REAL(result), dims[0], dims[1]);
    }
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << /*"return " << resultUnprotectCount <<*/ endl;
#endif
    
    return(result);
}

SEXP gemm_blas_C(SEXP s_A, SEXP s_transposeA, SEXP s_B, SEXP s_transposeB, SEXP s_C, SEXP s_alpha, SEXP s_beta)
{
    SEXP result = PROTECT(Rf_allocVector(INTSXP, 1));
    *INTEGER(result) = NA_INTEGER;
    UNPROTECT(1);
    
#if 0
    if (gTrace) CERR << "gemm_blas_C" << endl;
    
    int resultUnprotectCount = 0;
    
    // --------------- verify arg type ---------------
    
    if (gTrace) CERR << "verify arg type" << endl;
    
    if (Rf_isComplex(s_x)) {
        error("gemm_blas_C: complex x not implemented yet");
        
    } else if (Rf_isInteger(s_x)) {
        error("gemm_blas_C: integer x not implemented yet");
        
    } else if (!Rf_isReal(s_x) && !Rf_isInteger(s_x)) {
        error("gemm_blas_C: wrong x type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_x) = " << XLENGTH(s_x) << endl;
    
    SEXP s_dims;
    PROTECT(s_dims = Rf_getAttrib(s_x, R_DimSymbol));
    resultUnprotectCount++;
    
    if (Rf_isNull(s_dims)) {
        error("gemm_blas_C: no dimensions");
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
        error("gemm_blas_C: wrong dimension");
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
            error("gemm_blas_C: insufficient memory");
            
        } else {
            float *p = inMatrix;
            double *q = x;
            for (size_t k = 0; k < len; k++) {
                *p++ = (float)*q++;
            }
            
            gemm_blas_f(inMatrix, outMatrix, dims[0], dims[1]);
            
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
        gemm_blas_d(x, REAL(result), dims[0], dims[1]);
    }
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << /*"return " << resultUnprotectCount <<*/ endl;
#endif
    
    return(result);
}

SEXP gemm_clblas_C(SEXP s_device, SEXP s_A, SEXP s_transposeA, SEXP s_B, SEXP s_transposeB, SEXP s_C, SEXP s_alpha, SEXP s_beta)
{
    SEXP result = PROTECT(Rf_allocVector(INTSXP, 1));
    *INTEGER(result) = NA_INTEGER;
    UNPROTECT(1);
    
#if 0
    if (gTrace) CERR << "gemm_clblas_C" << endl;
    
    int resultUnprotectCount = 0;
    
    // --------------- verify arg types ---------------
    
    if (gTrace) CERR << "verify arg types" << endl;
    
    {
        SEXP deviceClass;
        PROTECT(deviceClass = Rf_getAttrib(s_device, R_ClassSymbol));
        
        if (!Rf_isNull(deviceClass) && strcmp("opencl.device", CHAR(STRING_ELT(deviceClass, 0))) != 0) {
            error("gemm_clblas_C: wrong device class");
        }
        
        UNPROTECT(1);
    }
    
    if (Rf_isComplex(s_x)) {
        error("gemm_clblas_C: complex x not implemented yet");
        
    } else if (Rf_isInteger(s_x)) {
        error("gemm_clblas_C: integer x not implemented yet");
        
    } else if (!Rf_isReal(s_x) && !Rf_isInteger(s_x)) {
        error("gemm_clblas_C: wrong x type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_x) = " << XLENGTH(s_x) << endl;
    
    SEXP s_dims;
    PROTECT(s_dims = Rf_getAttrib(s_x, R_DimSymbol));
    resultUnprotectCount++;
    
    if (Rf_isNull(s_dims)) {
        error("gemm_clblas_C: no dimensions");
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
        error("gemm_clblas_C: wrong dimension");
    }
    
    // --------------- get arg ---------------
    
    const int idElementIndex = 3;
    SEXP s_id = VECTOR_ELT(s_device, idElementIndex);
    
    cl_device_id device_id = (cl_device_id)R_ExternalPtrAddr(s_id);
    
    if (device_id == nullptr) {
        error("gemm_clblas_C: null cl_device_id");
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
            error("gemm_naive_C: insufficient memory");
            
        } else {
            float *p = inMatrix;
            double *q = x;
            for (size_t k = 0; k < len; k++) {
                *p++ = (float)*q++;
            }
            
            errorStatus = gemm_clblas_f(device_id, inMatrix, outMatrix, dims[0], dims[1]);
            
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
        errorStatus = gemm_clblas_d(device_id, x, REAL(result), dims[0], dims[1]);
    }
    
    if (errorStatus.error != CL_SUCCESS) {
        error(clErrorToString(errorStatus.error).c_str());
        
    } else if (errorStatus.status != clblasSuccess) {
        error(clblasErrorToString(errorStatus.status).c_str());
    }
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << /*"return " << resultUnprotectCount <<*/ endl;
#endif
    
    return(result);
}

#endif

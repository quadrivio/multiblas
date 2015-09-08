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
#include "gemm_r.h"
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
#include <cstddef>

using namespace std;

// ========== Local Headers ========================================================================

// ========== Globals ==============================================================================

extern bool gTrace;    // for debugging

// ========== Functions ============================================================================

SEXP gemm_naive_C(SEXP s_A, SEXP s_transposeA, SEXP s_B, SEXP s_transposeB, SEXP s_C, SEXP s_alpha, SEXP s_beta)
{
    if (gTrace) CERR << "gemm_naive_C" << endl;
    
    int resultUnprotectCount = 0;
    
    // --------------- verify arg type ---------------
    
    if (gTrace) CERR << "verify arg type" << endl;
    
    // s_A
    
    if (Rf_isComplex(s_A)) {
        error("gemm_naive_C: complex A not implemented yet");
        
    } else if (Rf_isInteger(s_A)) {
        error("gemm_naive_C: integer A not implemented yet");
        
    } else if (!Rf_isReal(s_A) && !Rf_isInteger(s_A)) {
        error("gemm_naive_C: wrong A type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_A) = " << XLENGTH(s_A) << endl;
    
    SEXP s_dims_A;
    PROTECT(s_dims_A = Rf_getAttrib(s_A, R_DimSymbol));
    resultUnprotectCount++;
    
    if (Rf_isNull(s_dims_A)) {
        error("gemm_naive_C: no A dimensions");
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
        error("gemm_naive_C: wrong A dimension");
    }
    
    // s_B
    
    if (Rf_isComplex(s_B)) {
        error("gemm_naive_C: complex B not implemented yet");
        
    } else if (Rf_isInteger(s_B)) {
        error("gemm_naive_C: integer B not implemented yet");
        
    } else if (!Rf_isReal(s_B) && !Rf_isInteger(s_B)) {
        error("gemm_naive_C: wrong B type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_B) = " << XLENGTH(s_B) << endl;
    
    SEXP s_dims_B;
    PROTECT(s_dims_B = Rf_getAttrib(s_B, R_DimSymbol));
    resultUnprotectCount++;
    
    if (Rf_isNull(s_dims_B)) {
        error("gemm_naive_C: no B dimensions");
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
        error("gemm_naive_C: wrong B dimension");
    }
    
    // s_C
    
    bool cIsNA = false;
    
    if (Rf_isLogical(s_C) && XLENGTH(s_C) > 0 && *LOGICAL(s_C) == NA_LOGICAL) {
        cIsNA = true;
        
    } else if (Rf_isComplex(s_C)) {
        error("gemm_naive_C: complex C not implemented yet");
        
    } else if (Rf_isInteger(s_C)) {
        error("gemm_naive_C: integer C not implemented yet");
        
    } else if (!Rf_isReal(s_C) && !Rf_isInteger(s_C)) {
        error("gemm_naive_C: wrong C type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_C) = " << XLENGTH(s_C) << endl;
    
    SEXP s_dims_C;
    PROTECT(s_dims_C = Rf_getAttrib(s_C, R_DimSymbol));
    resultUnprotectCount++;
    
    int dimCountC = Rf_isNull(s_dims_C) ? 0 : Rf_length(s_dims_C);
    int *dimsC = Rf_isNull(s_dims_C) ? nullptr : INTEGER(s_dims_C);
    
    if (!cIsNA && Rf_isNull(s_dims_C)) {
        error("gemm_naive_C: no C dimensions");
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
        error("gemm_naive_C: wrong C dimension");
    }
    
    // others
    
    if (!Rf_isReal(s_alpha)) {
        error("gemm_naive_C: wrong alpha type");
    }
    
    if (!Rf_isReal(s_beta)) {
        error("gemm_naive_C: wrong beta type");
    }
    
    if (!Rf_isLogical(s_transposeA)) {
        error("gemm_naive_C: wrong transposeA type");
    }
    
    if (!Rf_isLogical(s_transposeB)) {
        error("gemm_naive_C: wrong transposeB type");
    }
    
    // --------------- get arg ---------------
    
    double *A = REAL(s_A);
    double *B = REAL(s_B);
    double *C = cIsNA ? nullptr : REAL(s_C);
    
    if (gTrace && XLENGTH(s_A) <= 256) {
        for (int k = 0; k < XLENGTH(s_A); k++) {
            CERR << "A[" << k << "] = " << A[k] << endl;
        }
    }
    
    double alpha = *REAL(s_alpha);
    double beta = *REAL(s_beta);
    
    bool transposeA = !Rf_isNull(s_transposeA) && *LOGICAL(s_transposeA);
    bool transposeB = !Rf_isNull(s_transposeB) && *LOGICAL(s_transposeB);
    
    // --------------- calculate results ---------------
    
    size_t outRow = transposeA ? dimsA[1] : dimsA[0];
    size_t outCol = transposeB ? dimsB[0] : dimsB[1];
    
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
    
    if (isFloatA || isFloatB || isFloatC) {
        float *inMatrixA = (float *)calloc(XLENGTH(s_A), sizeof(float));
        float *inMatrixB = (float *)calloc(XLENGTH(s_B), sizeof(float));
        float *outMatrix = (float *)calloc(outRow * outCol, sizeof(float));
        
        if (inMatrixA == nullptr || inMatrixB == nullptr || outMatrix == nullptr) {
            error("gemm_naive_C: insufficient memory");
            
        } else {
            float *p = inMatrixA;
            double *q = A;
            for (int k = 0; k < XLENGTH(s_A); k++) {
                *p++ = (float)*q++;
            }
            
            p = inMatrixB;
            q = B;
            for (int k = 0; k < XLENGTH(s_B); k++) {
                *p++ = (float)*q++;
            }
            
            if (!cIsNA && (size_t)XLENGTH(s_C) == outRow * outCol) {
                p = outMatrix;
                q = C;
                for (int k = 0; k < XLENGTH(s_C); k++) {
                    *p++ = (float)*q++;
                }
                
            } else {
                memset(outMatrix, 0, outRow * outCol * sizeof(float));
            }
            
            gemm_naive_f(inMatrixA, dimsA[0], dimsA[1], transposeA, inMatrixB, dimsB[0], dimsB[1], transposeB, alpha, beta, outMatrix);
            
            p = outMatrix;
            q = REAL(result);
            for (size_t k = 0; k < outRow * outCol; k++) {
                *q++ = (double)*p++;
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
        if (!cIsNA && (size_t)XLENGTH(s_C) == outRow * outCol) {
            memcpy(REAL(result), C, outRow * outCol * sizeof(double));
            
        } else {
            memset(REAL(result), 0, outRow * outCol * sizeof(double));
        }
        
        gemm_naive_d(A, dimsA[0], dimsA[1], transposeA, B, dimsB[0], dimsB[1], transposeB, alpha, beta, REAL(result));
    }
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << /*"return " << resultUnprotectCount <<*/ endl;
    
    return(result);
}

SEXP gemm_blas_C(SEXP s_A, SEXP s_transposeA, SEXP s_B, SEXP s_transposeB, SEXP s_C, SEXP s_alpha, SEXP s_beta)
{
    if (gTrace) CERR << "gemm_blas_C" << endl;
    
    int resultUnprotectCount = 0;
    
    // --------------- verify arg type ---------------
    
    if (gTrace) CERR << "verify arg type" << endl;
    
    // s_A
    
    if (Rf_isComplex(s_A)) {
        error("gemm_blas_C: complex A not implemented yet");
        
    } else if (Rf_isInteger(s_A)) {
        error("gemm_blas_C: integer A not implemented yet");
        
    } else if (!Rf_isReal(s_A) && !Rf_isInteger(s_A)) {
        error("gemm_blas_C: wrong A type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_A) = " << XLENGTH(s_A) << endl;
    
    SEXP s_dims_A;
    PROTECT(s_dims_A = Rf_getAttrib(s_A, R_DimSymbol));
    resultUnprotectCount++;
    
    if (Rf_isNull(s_dims_A)) {
        error("gemm_blas_C: no A dimensions");
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
        error("gemm_blas_C: wrong A dimension");
    }
    
    // s_B
    
    if (Rf_isComplex(s_B)) {
        error("gemm_blas_C: complex B not implemented yet");
        
    } else if (Rf_isInteger(s_B)) {
        error("gemm_blas_C: integer B not implemented yet");
        
    } else if (!Rf_isReal(s_B) && !Rf_isInteger(s_B)) {
        error("gemm_blas_C: wrong B type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_B) = " << XLENGTH(s_B) << endl;
    
    SEXP s_dims_B;
    PROTECT(s_dims_B = Rf_getAttrib(s_B, R_DimSymbol));
    resultUnprotectCount++;
    
    if (Rf_isNull(s_dims_B)) {
        error("gemm_blas_C: no B dimensions");
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
        error("gemm_blas_C: wrong B dimension");
    }
    
    // s_C
    
    bool cIsNA = false;
    
    if (Rf_isLogical(s_C) && XLENGTH(s_C) > 0 && *LOGICAL(s_C) == NA_LOGICAL) {
        cIsNA = true;
        
    } else if (Rf_isComplex(s_C)) {
        error("gemm_blas_C: complex C not implemented yet");
        
    } else if (Rf_isInteger(s_C)) {
        error("gemm_blas_C: integer C not implemented yet");
        
    } else if (!Rf_isReal(s_C) && !Rf_isInteger(s_C)) {
        error("gemm_blas_C: wrong C type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_C) = " << XLENGTH(s_C) << endl;
    
    SEXP s_dims_C;
    PROTECT(s_dims_C = Rf_getAttrib(s_C, R_DimSymbol));
    resultUnprotectCount++;
    
    int dimCountC = Rf_isNull(s_dims_C) ? 0 : Rf_length(s_dims_C);
    int *dimsC = Rf_isNull(s_dims_C) ? nullptr : INTEGER(s_dims_C);
    
    if (!cIsNA && Rf_isNull(s_dims_C)) {
        error("gemm_blas_C: no C dimensions");
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
        error("gemm_blas_C: wrong C dimension");
    }
    
    // others
    
    if (!Rf_isReal(s_alpha)) {
        error("gemm_blas_C: wrong alpha type");
    }
    
    if (!Rf_isReal(s_beta)) {
        error("gemm_blas_C: wrong beta type");
    }
    
    if (!Rf_isLogical(s_transposeA)) {
        error("gemm_blas_C: wrong transposeA type");
    }
    
    if (!Rf_isLogical(s_transposeB)) {
        error("gemm_blas_C: wrong transposeB type");
    }
    
    // --------------- get arg ---------------
    
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
    
    // --------------- calculate results ---------------
    
    size_t outRow = transposeA ? dimsA[1] : dimsA[0];
    size_t outCol = transposeB ? dimsB[0] : dimsB[1];
    
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
    
    if (isFloatA || isFloatB || isFloatC) {
        if (!cblas_sgemm_available()) {
            error("gemm_blas_C: single-precision not available");
        }
        
        float *inMatrixA = (float *)calloc(XLENGTH(s_A), sizeof(float));
        float *inMatrixB = (float *)calloc(XLENGTH(s_B), sizeof(float));
        float *outMatrix = (float *)calloc(outRow * outCol, sizeof(float));
        
        if (inMatrixA == nullptr || inMatrixB == nullptr || outMatrix == nullptr) {
            error("gemm_blas_C: insufficient memory");
            
        } else {
            float *p = inMatrixA;
            const double *q = A;
            for (int k = 0; k < XLENGTH(s_A); k++) {
                *p++ = (float)*q++;
            }
            
            p = inMatrixB;
            q = B;
            for (int k = 0; k < XLENGTH(s_B); k++) {
                *p++ = (float)*q++;
            }
            
            if (!cIsNA && (size_t)XLENGTH(s_C) == outRow * outCol) {
                p = outMatrix;
                q = C;
                for (int k = 0; k < XLENGTH(s_C); k++) {
                    *p++ = (float)*q++;
                }
                
            } else {
                memset(outMatrix, 0, outRow * outCol * sizeof(float));
            }
            
            gemm_blas_f(inMatrixA, dimsA[0], dimsA[1], transposeA, inMatrixB, dimsB[0], dimsB[1], transposeB, alpha, beta, outMatrix);
            
            p = outMatrix;
            double *qq = REAL(result);
            for (size_t k = 0; k < outRow * outCol; k++) {
                *qq++ = (double)*p++;
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
        if (!cblas_dgemm_available()) {
            error("gemm_blas_C: double-precision not available");
        }
        
        if (!cIsNA && (size_t)XLENGTH(s_C) == outRow * outCol) {
            memcpy(REAL(result), C, outRow * outCol * sizeof(double));
            
        } else {
            memset(REAL(result), 0, outRow * outCol * sizeof(double));
        }
        
        gemm_blas_d(A, dimsA[0], dimsA[1], transposeA, B, dimsB[0], dimsB[1], transposeB, alpha, beta, REAL(result));
    }
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << /*"return " << resultUnprotectCount <<*/ endl;
    
    return(result);
}

SEXP gemm_r_C(SEXP s_A, SEXP s_transposeA, SEXP s_B, SEXP s_transposeB, SEXP s_C, SEXP s_alpha, SEXP s_beta)
{
    if (gTrace) CERR << "gemm_r_C" << endl;
    
    int resultUnprotectCount = 0;
    
    // --------------- verify arg type ---------------
    
    if (gTrace) CERR << "verify arg type" << endl;
    
    // s_A
    
    if (Rf_isComplex(s_A)) {
        error("gemm_r_C: complex A not implemented yet");
        
    } else if (Rf_isInteger(s_A)) {
        error("gemm_r_C: integer A not implemented yet");
        
    } else if (!Rf_isReal(s_A) && !Rf_isInteger(s_A)) {
        error("gemm_r_C: wrong A type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_A) = " << XLENGTH(s_A) << endl;
    
    SEXP s_dims_A;
    PROTECT(s_dims_A = Rf_getAttrib(s_A, R_DimSymbol));
    resultUnprotectCount++;
    
    if (Rf_isNull(s_dims_A)) {
        error("gemm_r_C: no A dimensions");
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
        error("gemm_r_C: wrong A dimension");
    }
    
    // s_B
    
    if (Rf_isComplex(s_B)) {
        error("gemm_r_C: complex B not implemented yet");
        
    } else if (Rf_isInteger(s_B)) {
        error("gemm_r_C: integer B not implemented yet");
        
    } else if (!Rf_isReal(s_B) && !Rf_isInteger(s_B)) {
        error("gemm_r_C: wrong B type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_B) = " << XLENGTH(s_B) << endl;
    
    SEXP s_dims_B;
    PROTECT(s_dims_B = Rf_getAttrib(s_B, R_DimSymbol));
    resultUnprotectCount++;
    
    if (Rf_isNull(s_dims_B)) {
        error("gemm_r_C: no B dimensions");
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
        error("gemm_r_C: wrong B dimension");
    }
    
    // s_C
    
    bool cIsNA = false;
    
    if (Rf_isLogical(s_C) && XLENGTH(s_C) > 0 && *LOGICAL(s_C) == NA_LOGICAL) {
        cIsNA = true;
        
    } else if (Rf_isComplex(s_C)) {
        error("gemm_r_C: complex C not implemented yet");
        
    } else if (Rf_isInteger(s_C)) {
        error("gemm_r_C: integer C not implemented yet");
        
    } else if (!Rf_isReal(s_C) && !Rf_isInteger(s_C)) {
        error("gemm_r_C: wrong C type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_C) = " << XLENGTH(s_C) << endl;
    
    SEXP s_dims_C;
    PROTECT(s_dims_C = Rf_getAttrib(s_C, R_DimSymbol));
    resultUnprotectCount++;
    
    int dimCountC = Rf_isNull(s_dims_C) ? 0 : Rf_length(s_dims_C);
    int *dimsC = Rf_isNull(s_dims_C) ? nullptr : INTEGER(s_dims_C);
    
    if (!cIsNA && Rf_isNull(s_dims_C)) {
        error("gemm_r_C: no C dimensions");
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
        error("gemm_r_C: wrong C dimension");
    }
    
    // others
    
    if (!Rf_isReal(s_alpha)) {
        error("gemm_r_C: wrong alpha type");
    }
    
    if (!Rf_isReal(s_beta)) {
        error("gemm_r_C: wrong beta type");
    }
    
    if (!Rf_isLogical(s_transposeA)) {
        error("gemm_r_C: wrong transposeA type");
    }
    
    if (!Rf_isLogical(s_transposeB)) {
        error("gemm_r_C: wrong transposeB type");
    }
    
    // --------------- get arg ---------------
    
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
    
    // --------------- calculate results ---------------
    
    size_t outRow = transposeA ? dimsA[1] : dimsA[0];
    size_t outCol = transposeB ? dimsB[0] : dimsB[1];
    
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
    
    if (isFloatA || isFloatB || isFloatC) {
        error("gemm_r_C: single-precision not available");
        
    } else {
        if (!cIsNA && (size_t)XLENGTH(s_C) == outRow * outCol) {
            memcpy(REAL(result), C, outRow * outCol * sizeof(double));
            
        } else {
            memset(REAL(result), 0, outRow * outCol * sizeof(double));
        }
        
        gemm_r_d(A, dimsA[0], dimsA[1], transposeA, B, dimsB[0], dimsB[1], transposeB, alpha, beta, REAL(result));
    }
    
    // ---------------------------------------------------------------------------------------------
    
    UNPROTECT(resultUnprotectCount);
    
    if (gTrace) CERR << /*"return " << resultUnprotectCount <<*/ endl;
    
    return(result);
}

SEXP gemm_clblas_C(SEXP s_device, SEXP s_A, SEXP s_transposeA, SEXP s_B, SEXP s_transposeB, SEXP s_C, SEXP s_alpha, SEXP s_beta)
{
    if (gTrace) CERR << "gemm_clblas_C" << endl;
    
    int resultUnprotectCount = 0;
    
    // --------------- verify arg type ---------------
    
    if (gTrace) CERR << "verify arg type" << endl;
    
    {
        SEXP deviceClass;
        PROTECT(deviceClass = Rf_getAttrib(s_device, R_ClassSymbol));
        
        if (!Rf_isNull(deviceClass) && strcmp("opencl.device", CHAR(STRING_ELT(deviceClass, 0))) != 0) {
            error("gemm_clblas_C: wrong device class");
        }
        
        UNPROTECT(1);
    }
    
    // s_A
    
    if (Rf_isComplex(s_A)) {
        error("gemm_clblas_C: complex A not implemented yet");
        
    } else if (Rf_isInteger(s_A)) {
        error("gemm_clblas_C: integer A not implemented yet");
        
    } else if (!Rf_isReal(s_A) && !Rf_isInteger(s_A)) {
        error("gemm_clblas_C: wrong A type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_A) = " << XLENGTH(s_A) << endl;
    
    SEXP s_dims_A;
    PROTECT(s_dims_A = Rf_getAttrib(s_A, R_DimSymbol));
    resultUnprotectCount++;
    
    if (Rf_isNull(s_dims_A)) {
        error("gemm_clblas_C: no A dimensions");
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
        error("gemm_clblas_C: wrong A dimension");
    }
    
    // s_B
    
    if (Rf_isComplex(s_B)) {
        error("gemm_clblas_C: complex B not implemented yet");
        
    } else if (Rf_isInteger(s_B)) {
        error("gemm_clblas_C: integer B not implemented yet");
        
    } else if (!Rf_isReal(s_B) && !Rf_isInteger(s_B)) {
        error("gemm_clblas_C: wrong B type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_B) = " << XLENGTH(s_B) << endl;
    
    SEXP s_dims_B;
    PROTECT(s_dims_B = Rf_getAttrib(s_B, R_DimSymbol));
    resultUnprotectCount++;
    
    if (Rf_isNull(s_dims_B)) {
        error("gemm_clblas_C: no B dimensions");
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
        error("gemm_clblas_C: wrong B dimension");
    }
    
    // s_C
    
    bool cIsNA = false;
    
    if (Rf_isLogical(s_C) && XLENGTH(s_C) > 0 && *LOGICAL(s_C) == NA_LOGICAL) {
        cIsNA = true;
        
    } else if (Rf_isComplex(s_C)) {
        error("gemm_clblas_C: complex C not implemented yet");
        
    } else if (Rf_isInteger(s_C)) {
        error("gemm_clblas_C: integer C not implemented yet");
        
    } else if (!Rf_isReal(s_C) && !Rf_isInteger(s_C)) {
        error("gemm_clblas_C: wrong C type");
    }
    
    if (gTrace) CERR << "XLENGTH(s_C) = " << XLENGTH(s_C) << endl;
    
    SEXP s_dims_C;
    PROTECT(s_dims_C = Rf_getAttrib(s_C, R_DimSymbol));
    resultUnprotectCount++;
    
    int dimCountC = Rf_isNull(s_dims_C) ? 0 : Rf_length(s_dims_C);
    int *dimsC = Rf_isNull(s_dims_C) ? nullptr : INTEGER(s_dims_C);
    
    if (!cIsNA && Rf_isNull(s_dims_C)) {
        error("gemm_clblas_C: no C dimensions");
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
        error("gemm_clblas_C: wrong C dimension");
    }
    
    // others
    
    if (!Rf_isReal(s_alpha)) {
        error("gemm_clblas_C: wrong alpha type");
    }
    
    if (!Rf_isReal(s_beta)) {
        error("gemm_clblas_C: wrong beta type");
    }
    
    if (!Rf_isLogical(s_transposeA)) {
        error("gemm_clblas_C: wrong transposeA type");
    }
    
    if (!Rf_isLogical(s_transposeB)) {
        error("gemm_clblas_C: wrong transposeB type");
    }
    
    // --------------- get arg ---------------
    
    const int idElementIndex = 3;
    SEXP s_id = VECTOR_ELT(s_device, idElementIndex);
    
    cl_device_id device_id = (cl_device_id)R_ExternalPtrAddr(s_id);
    
    if (device_id == nullptr) {
        error("gemm_clblas_C: null cl_device_id");
    }
    
    if (gTrace) CERR << "cl_device_id = " << hex << (unsigned long long)(void *)device_id << dec << endl;
    
    double *A = REAL(s_A);
    double *B = REAL(s_B);
    double *C = cIsNA ? nullptr : REAL(s_C);
    
    if (gTrace && XLENGTH(s_A) <= 256) {
        for (int k = 0; k < XLENGTH(s_A); k++) {
            CERR << "A[" << k << "] = " << A[k] << endl;
        }
    }
    
    double alpha = *REAL(s_alpha);
    double beta = *REAL(s_beta);
    
    bool transposeA = !Rf_isNull(s_transposeA) && *LOGICAL(s_transposeA);
    bool transposeB = !Rf_isNull(s_transposeB) && *LOGICAL(s_transposeB);
    
    // --------------- calculate results ---------------
    
    size_t outRow = transposeA ? dimsA[1] : dimsA[0];
    size_t outCol = transposeB ? dimsB[0] : dimsB[1];
    
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
    
    ErrorStatus errorStatus;
    
    if (isFloatA || isFloatB || isFloatC) {
        float *inMatrixA = (float *)calloc(XLENGTH(s_A), sizeof(float));
        float *inMatrixB = (float *)calloc(XLENGTH(s_B), sizeof(float));
        float *outMatrix = (float *)calloc(outRow * outCol, sizeof(float));
        
        if (inMatrixA == nullptr || inMatrixB == nullptr || outMatrix == nullptr) {
            error("gemm_clblas_C: insufficient memory");
            
        } else {
            float *p = inMatrixA;
            double *q = A;
            for (int k = 0; k < XLENGTH(s_A); k++) {
                *p++ = (float)*q++;
            }
            
            p = inMatrixB;
            q = B;
            for (int k = 0; k < XLENGTH(s_B); k++) {
                *p++ = (float)*q++;
            }
            
            if (!cIsNA && (size_t)XLENGTH(s_C) == outRow * outCol) {
                p = outMatrix;
                q = C;
                for (int k = 0; k < XLENGTH(s_C); k++) {
                    *p++ = (float)*q++;
                }
                
            } else {
                memset(outMatrix, 0, outRow * outCol * sizeof(float));
            }
            
            errorStatus = gemm_clblas_f(device_id, inMatrixA, dimsA[0], dimsA[1], transposeA, inMatrixB, dimsB[0], dimsB[1], transposeB, alpha, beta, outMatrix);
            
            p = outMatrix;
            q = REAL(result);
            for (size_t k = 0; k < outRow * outCol; k++) {
                *q++ = (double)*p++;
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
        if (!cIsNA && (size_t)XLENGTH(s_C) == outRow * outCol) {
            memcpy(REAL(result), C, outRow * outCol * sizeof(double));
            
        } else {
            memset(REAL(result), 0, outRow * outCol * sizeof(double));
        }
        
        errorStatus = gemm_clblas_d(device_id, A, dimsA[0], dimsA[1], transposeA, B, dimsB[0], dimsB[1], transposeB, alpha, beta, REAL(result));
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
    
    if (gTrace) CERR << /*"return " << resultUnprotectCount <<*/ endl;
    
    return(result);
}

#endif

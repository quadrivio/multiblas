//
//  multiblas_gemm.h
//  multiBLAS.XC
//
//  Created by michael on 7/23/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#ifndef __multiBLAS_XC__multiblas_gemm__
#define __multiBLAS_XC__multiblas_gemm__

#include <stdio.h>

#ifndef RPACKAGE
#define RPACKAGE 1
#endif

#if RPACKAGE

// ---------- use this block to include R headers ----------
#ifndef NO_C_HEADERS
#define NO_C_HEADERS
#endif
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <cmath>
#include <iostream>
#include <sstream>
#include <R.h>
#define R_NO_REMAP
#include <Rinternals.h>
// ---------------------------------------------------------

// ========== Function Headers =====================================================================

extern "C" {
    // call from R
    SEXP gemm_naive_C(SEXP s_A, SEXP s_transposeA, SEXP s_B, SEXP s_transposeB, SEXP s_C, SEXP s_alpha, SEXP s_beta);
    
    // call from R
    SEXP gemm_blas_C(SEXP s_A, SEXP s_transposeA, SEXP s_B, SEXP s_transposeB, SEXP s_C, SEXP s_alpha, SEXP s_beta);
    
    // call from R
    SEXP gemm_clblas_C(SEXP s_device, SEXP s_A, SEXP s_transposeA, SEXP s_B, SEXP s_transposeB, SEXP s_C, SEXP s_alpha, SEXP s_beta);
}

#endif

#endif /* defined(__multiBLAS_XC__multiblas_gemm__) */

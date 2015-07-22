//
//  multiblas_crossprod.h
//  multiBLAS.XC
//
//  Created by michael on 7/2/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#ifndef __multiBLAS_XC__multiblas_crossprod__
#define __multiBLAS_XC__multiblas_crossprod__

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
    SEXP crossprod_naive_C(SEXP s_x);
    
    // call from R
    SEXP crossprod_blas_C(SEXP s_x);
    
    // call from R
    SEXP crossprod_clblas_C(SEXP s_device, SEXP s_x);
}

#endif

#endif /* defined(__multiBLAS_XC__multiblas_crossprod__) */

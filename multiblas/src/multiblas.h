//
//  multiblas.h
//  multiBLAS.XC
//
//  Created by michael on 6/29/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#ifndef __multiBLAS_XC__multiblas__
#define __multiBLAS_XC__multiblas__

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
    SEXP null_externalptr_C(SEXP s_externalptr);
    
    // call from R
    SEXP is_externalptr_null_C(SEXP s_externalptr);
    
    // call from R
    SEXP opencl_platforms_C();
    
    // call from R
    SEXP opencl_devices_C(SEXP s_platform);
    
    // call from R
    SEXP property_platform_C(SEXP s_platform, SEXP s_name);
    
    // call from R
    SEXP property_device_C(SEXP s_device, SEXP s_name);
    
    // call from R
    SEXP opencl_context_C(SEXP s_device);
    
    // call from R
    SEXP opencl_queue_C(SEXP s_context, SEXP s_device);

    // call from R
    SEXP opencl_kernel_C(SEXP s_context, SEXP s_device, SEXP s_name, SEXP s_source, SEXP s_options,
                         SEXP s_verbose);

    // call from R
    SEXP opencl_calc_x_C(SEXP s_context, SEXP s_kernel_f, SEXP s_kernel_d, SEXP s_queue, SEXP s_x,
                         SEXP s_work_item_sizes, SEXP s_vector_size, SEXP s_row_multiple,
                         SEXP s_row_tile_size, SEXP s_col_tile_size, SEXP s_fill_on_host,
                         SEXP s_verbose);
    
    // call from R
    SEXP opencl_calc_gemm_C(SEXP s_context, SEXP s_kernel_f, SEXP s_kernel_d, SEXP s_queue,
                            SEXP s_A, SEXP s_transposeA, SEXP s_B, SEXP s_transposeB, SEXP s_C,
                            SEXP s_alpha, SEXP s_beta,
                            SEXP s_work_item_sizes, SEXP s_vector_size, SEXP s_row_multiple,
                            SEXP s_row_tile_size, SEXP s_col_tile_size, SEXP s_fill_on_host,
                            SEXP s_verbose);
    
//    // call from R
//    SEXP crossprod_opencl_C(SEXP s_device, SEXP s_x, SEXP s_clblas);
//    
//    // call from R
//    SEXP opencl_kernel_from_path_C(SEXP s_context, SEXP s_device, SEXP s_name, SEXP s_path);
//    
//    // call from R
//    SEXP crossprod_opencl_kernel_C(SEXP s_context, SEXP s_device, SEXP s_use_float);
//    
//    // call from R
//    SEXP crossprod_opencl_ckq_C(SEXP s_context, SEXP s_kernel_f, SEXP s_kernel_d, SEXP s_queue, SEXP s_x);
    
}

#endif

#endif /* defined(__multiBLAS_XC__multiblas__) */






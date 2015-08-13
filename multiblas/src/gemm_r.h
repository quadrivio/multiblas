//
//  gemm_r.h
//  multiBLAS.XC
//
//  Created by michael on 7/23/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#ifndef __multiBLAS_XC__gemm_r__
#define __multiBLAS_XC__gemm_r__

#if RPACKAGE

void gemm_r_d(const double *inMatrixA, int nrowA, int ncolA, bool transposeA,
                 const double *inMatrixB, int nrowB, int ncolB, bool transposeB,
                 double alpha, double beta, double *outMatrix);

#endif

#endif /* defined(__multiBLAS_XC__gemm_r__) */

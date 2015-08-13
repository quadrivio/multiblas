//
//  crossprod_r.cpp
//  OpenCL.ImpXC
//
//  Created by michael on 6/2/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#include "shim.h"

#if RPACKAGE

#include "crossprod_r.h"
#include "crossprod_naive.h"

#include <R_ext/BLAS.h>

void crossprod_r_d(double *inMatrix, double *outMatrix, int nrow, int ncol)
{
    const int lda = nrow;
    const int ldc = ncol;
    
    const double alpha = 1.0;
    
    const char uplo = 'U';
    const char trans = 'T';

    const double beta = 0.0;

    F77_NAME(dsyrk)(&uplo, &trans, &ncol, &nrow, &alpha, inMatrix, &lda, &beta, outMatrix, &ldc);
    
    symmetrizeSquare_d(outMatrix, ncol);
}

#endif

//
//  crossprod_blas.h
//  OpenCL.ImpXC
//
//  Created by michael on 6/2/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#ifndef __OpenCL_ImpXC__crossprod_blas__
#define __OpenCL_ImpXC__crossprod_blas__

bool cblas_dsyrk_available();
bool cblas_ssyrk_available();

void crossprod_blas_d(double *inMatrix, double *outMatrix, int nrow, int ncol);
void crossprod_blas_f(float *inMatrix, float *outMatrix, int nrow, int ncol);

#endif /* defined(__OpenCL_ImpXC__crossprod_blas__) */

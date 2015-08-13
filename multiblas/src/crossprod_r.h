//
//  crossprod_r.h
//  OpenCL.ImpXC
//
//  Created by michael on 6/2/15.
//  Copyright (c) 2015 Quadrivio Corporation. All rights reserved.
//

#ifndef __OpenCL_ImpXC__crossprod_r__
#define __OpenCL_ImpXC__crossprod_r__

#if RPACKAGE

void crossprod_r_d(double *inMatrix, double *outMatrix, int nrow, int ncol);

#endif

#endif /* defined(__OpenCL_ImpXC__crossprod_r__) */

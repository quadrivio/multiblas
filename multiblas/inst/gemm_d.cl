#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define OPENCL_DOUBLE_SUPPORTED
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#define OPENCL_DOUBLE_SUPPORTED
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif

#ifdef OPENCL_DOUBLE_SUPPORTED

#ifndef VECTOR_SIZE
#define VECTOR_SIZE 1
#endif

#if VECTOR_SIZE == 1
#define doubleN double
#elif VECTOR_SIZE == 2
#define doubleN double2
#elif VECTOR_SIZE == 4
#define doubleN double4
#elif VECTOR_SIZE == 8
#define doubleN double8
#endif

__kernel void gemm_d_naive(__private int inputA_nrow,
                           __private int inputA_ncol,
                           __private int inputB_nrow,
                           __private int inputB_ncol,
                           __private double alpha,
                           __private double beta,
                           __global double* inputAT,
                           __global double* inputB,
                           __global double* output) {
    int output_col = get_global_id(0);
    int output_row = get_global_id(1);
    
    double sum = 0.0f;
    int index1 = inputA_ncol * output_row;
    int index2 = inputB_nrow * output_col;
    for (int k = 0; k < inputB_nrow; k++) {
        sum += inputAT[index1 + k] * inputB[index2 + k];
    }
    
    output[output_col * inputA_nrow + output_row] *= beta;
    output[output_col * inputA_nrow + output_row] += alpha * sum;
}

__kernel void gemm_d_dot4sum4x2(__private int inputA_nrow,
                                __private int inputA_ncol,
                                __private int inputB_nrow,
                                __private int inputB_ncol,
                                __private double alpha,
                                __private double beta,
                                __global double4* inputAT,
                                __global double4* inputB,
                                __global double* output) {
#undef VECTOR
#define VECTOR 4
    
#undef ROW_TILE_SIZE
#define ROW_TILE_SIZE 4
    
#undef COL_TILE_SIZE
#define COL_TILE_SIZE 2
    
    int output_col = COL_TILE_SIZE * get_global_id(0);
    int output_row = ROW_TILE_SIZE * get_global_id(1);
    
    double sum[COL_TILE_SIZE][ROW_TILE_SIZE];
    
    for (int i = 0; i < COL_TILE_SIZE; i++) {
        for (int j = 0; j < ROW_TILE_SIZE; j++) {
            sum[i][j] = 0.0f;
        }
    }
    
    int index1[ROW_TILE_SIZE];
    int index2[COL_TILE_SIZE];
    
    for (int k = 0; k < ROW_TILE_SIZE; k++) {
        index1[k] = inputA_ncol * (output_row + k) / VECTOR;
    }
    for (int k = 0; k < COL_TILE_SIZE; k++) {
        index2[k] = inputB_nrow * (output_col + k) / VECTOR;
    }
    
    for (int k = 0; k < inputB_nrow / VECTOR; k += 2) {
        
        sum[0][0] += dot(inputAT[index1[0] + k], inputB[index2[0] + k]) +
        dot(inputAT[index1[0] + k + 1], inputB[index2[0] + k + 1]);
        
        sum[0][1] += dot(inputAT[index1[1] + k], inputB[index2[0] + k]) +
        dot(inputAT[index1[1] + k + 1], inputB[index2[0] + k + 1]);
        
        sum[0][2] += dot(inputAT[index1[2] + k], inputB[index2[0] + k]) +
        dot(inputAT[index1[2] + k + 1], inputB[index2[0] + k + 1]);
        
        sum[0][3] += dot(inputAT[index1[3] + k], inputB[index2[0] + k]) +
        dot(inputAT[index1[3] + k + 1], inputB[index2[0] + k + 1]);
        
        
        sum[1][0] += dot(inputAT[index1[0] + k], inputB[index2[1] + k]) +
        dot(inputAT[index1[0] + k + 1], inputB[index2[1] + k + 1]);
        
        sum[1][1] += dot(inputAT[index1[1] + k], inputB[index2[1] + k]) +
        dot(inputAT[index1[1] + k + 1], inputB[index2[1] + k + 1]);
        
        sum[1][2] += dot(inputAT[index1[2] + k], inputB[index2[1] + k]) +
        dot(inputAT[index1[2] + k + 1], inputB[index2[1] + k + 1]);
        
        sum[1][3] += dot(inputAT[index1[3] + k], inputB[index2[1] + k]) +
        dot(inputAT[index1[3] + k + 1], inputB[index2[1] + k + 1]);
    }
    
    int index;
    
    index = (output_col + 0) * inputA_nrow + (output_row + 0);
    output[index] *= beta;
    output[index] += alpha * sum[0][0];
    
    index = (output_col + 0) * inputA_nrow + (output_row + 1);
    output[index] *= beta;
    output[index] += alpha * sum[0][1];
    
    index = (output_col + 0) * inputA_nrow + (output_row + 2);
    output[index] *= beta;
    output[index] += alpha * sum[0][2];
    
    index = (output_col + 0) * inputA_nrow + (output_row + 3);
    output[index] *= beta;
    output[index] += alpha * sum[0][3];
    
    
    index = (output_col + 1) * inputA_nrow + (output_row + 0);
    output[index] *= beta;
    output[index] += alpha * sum[1][0];
    
    index = (output_col + 1) * inputA_nrow + (output_row + 1);
    output[index] *= beta;
    output[index] += alpha * sum[1][1];
    
    index = (output_col + 1) * inputA_nrow + (output_row + 2);
    output[index] *= beta;
    output[index] += alpha * sum[1][2];
    
    index = (output_col + 1) * inputA_nrow + (output_row + 3);
    output[index] *= beta;
    output[index] += alpha * sum[1][3];
    
}

#endif

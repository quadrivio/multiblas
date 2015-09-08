#ifndef VECTOR_SIZE
#define VECTOR_SIZE 1
#endif

#if VECTOR_SIZE == 1
#define floatN float
#elif VECTOR_SIZE == 2
#define floatN float2
#elif VECTOR_SIZE == 4
#define floatN float4
#elif VECTOR_SIZE == 8
#define floatN float8
#endif


__kernel void gemm_f_naiveN(__private int inputA_nrow,
                           __private int inputA_ncol,
                           __private int inputB_nrow,
                           __private int inputB_ncol,
                           __private float alpha,
                           __private float beta,
                           __global floatN* inputAT,
                           __global floatN* inputB,
                           __global float* output) {
    int output_col = get_global_id(0);
    int output_row = get_global_id(1);
    
    float sum = 0.0f;
    int index1 = inputA_ncol * output_row / VECTOR_SIZE;
    int index2 = inputB_nrow * output_col / VECTOR_SIZE;
    for (int k = 0; k < inputB_nrow / VECTOR_SIZE; k++) {
#if VECTOR_SIZE == 1
        sum += inputAT[index1 + k] * inputB[index2 + k];
        
#elif (VECTOR_SIZE == 2) || (VECTOR_SIZE == 4)
        sum += dot(inputAT[index1 + k], inputB[index2 + k]);
        
#elif VECTOR_SIZE == 8
        sum += dot(inputAT[index1 + k].hi, inputB[index2 + k].hi) +
            dot(inputAT[index1 + k].lo, inputB[index2 + k].lo);
#endif

    }
    
    output[output_col * inputA_nrow + output_row] *= beta;
    output[output_col * inputA_nrow + output_row] += alpha * sum;
}

__kernel void gemm_f_naive(__private int inputA_nrow,
                           __private int inputA_ncol,
                           __private int inputB_nrow,
                           __private int inputB_ncol,
                           __private float alpha,
                           __private float beta,
                           __global float* inputAT,
                           __global float* inputB,
                           __global float* output) {
    int output_col = get_global_id(0);
    int output_row = get_global_id(1);
    
    float sum = 0.0f;
    int index1 = inputA_ncol * output_row;
    int index2 = inputB_nrow * output_col;
    for (int k = 0; k < inputB_nrow; k++) {
        sum += inputAT[index1 + k] * inputB[index2 + k];
    }
    
    output[output_col * inputA_nrow + output_row] *= beta;
    output[output_col * inputA_nrow + output_row] += alpha * sum;
}

__kernel void gemm_f_naive2(__private int inputA_nrow,
                            __private int inputA_ncol,
                            __private int inputB_nrow,
                            __private int inputB_ncol,
                            __private float alpha,
                            __private float beta,
                            __global float2* inputAT,
                            __global float2* inputB,
                            __global float* output) {
#undef VECTOR
#define VECTOR 2
    
    int output_col = get_global_id(0);
    int output_row = get_global_id(1);
    
    float sum = 0.0f;
    int index1 = inputA_ncol * output_row / VECTOR;
    int index2 = inputB_nrow * output_col / VECTOR;
    for (int k = 0; k < inputB_nrow / VECTOR; k++) {
        sum += dot(inputAT[index1 + k], inputB[index2 + k]);
    }
    
    output[output_col * inputA_nrow + output_row] *= beta;
    output[output_col * inputA_nrow + output_row] += alpha * sum;
}

__kernel void gemm_f_naive4(__private int inputA_nrow,
                            __private int inputA_ncol,
                            __private int inputB_nrow,
                            __private int inputB_ncol,
                            __private float alpha,
                            __private float beta,
                            __global float4* inputAT,
                            __global float4* inputB,
                            __global float* output) {
#undef VECTOR
#define VECTOR 4
    
    int output_col = get_global_id(0);
    int output_row = get_global_id(1);
    
    float sum = 0.0f;
    int index1 = inputA_ncol * output_row / VECTOR;
    int index2 = inputB_nrow * output_col / VECTOR;
    for (int k = 0; k < inputB_nrow / VECTOR; k++) {
        sum += dot(inputAT[index1 + k], inputB[index2 + k]);
    }
    
    output[output_col * inputA_nrow + output_row] *= beta;
    output[output_col * inputA_nrow + output_row] += alpha * sum;
}

__kernel void gemm_f_naive8(__private int inputA_nrow,
                            __private int inputA_ncol,
                            __private int inputB_nrow,
                            __private int inputB_ncol,
                            __private float alpha,
                            __private float beta,
                            __global float8* inputAT,
                            __global float8* inputB,
                            __global float* output) {
#undef VECTOR
#define VECTOR 8
    
    int output_col = get_global_id(0);
    int output_row = get_global_id(1);
    
    float sum = 0.0f;
    int index1 = inputA_ncol * output_row / VECTOR;
    int index2 = inputB_nrow * output_col / VECTOR;
    for (int k = 0; k < inputB_nrow / VECTOR; k++) {
        sum += dot(inputAT[index1 + k].hi, inputB[index2 + k].hi) +
               dot(inputAT[index1 + k].lo, inputB[index2 + k].lo);
    }
    
    output[output_col * inputA_nrow + output_row] *= beta;
    output[output_col * inputA_nrow + output_row] += alpha * sum;
}

__kernel void gemm_f_dot4sum4x2(__private int inputA_nrow,
                                __private int inputA_ncol,
                                __private int inputB_nrow,
                                __private int inputB_ncol,
                                __private float alpha,
                                __private float beta,
                                __global float4* inputAT,
                                __global float4* inputB,
                                __global float* output) {
#undef VECTOR
#define VECTOR 4
    
#undef ROW_TILE_SIZE
#define ROW_TILE_SIZE 4
    
#undef COL_TILE_SIZE
#define COL_TILE_SIZE 2
    
    int output_col = COL_TILE_SIZE * get_global_id(0);
    int output_row = ROW_TILE_SIZE * get_global_id(1);
    
    float sum[COL_TILE_SIZE][ROW_TILE_SIZE];
    
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


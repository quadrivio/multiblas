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

__kernel void gemm_f_dot4(__private int inputA_nrow,
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
    for (int k = 0; k < inputB_nrow / VECTOR; k += 2) {
        sum += dot(inputAT[index1 + k], inputB[index2 + k]) +
        dot(inputAT[index1 + k + 1], inputB[index2 + k + 1]);
    }
    
    output[output_col * inputA_nrow + output_row] *= beta;
    output[output_col * inputA_nrow + output_row] += alpha * sum;
}

#ifndef ROW_TILE_SIZE
#define ROW_TILE_SIZE 4
#endif

#ifndef COL_TILE_SIZE
#define COL_TILE_SIZE 2
#endif

__kernel void gemm_f_dot4tile(__private int inputA_nrow,
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
        for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
            for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
                sum[tile_col][tile_row] += dot(inputAT[index1[tile_row] + k],     inputB[index2[tile_col] + k]) +
                                           dot(inputAT[index1[tile_row] + k + 1], inputB[index2[tile_col] + k + 1]);
            }
        }
    }
    
    for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
        for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
            int index = (output_col + tile_col) * inputA_nrow + (output_row + tile_row);
            output[index] *= beta;
            output[index] += alpha * sum[tile_col][tile_row];
        }
    }
}



/*
__kernel void crossprod_f_dot4tile(__private int input_nrow,
                                     __private int input_ncol,
                                     __global float4* input,
                                     __global float* output) {
    int output_col = COL_TILE_SIZE * get_global_id(0);
    int output_row = ROW_TILE_SIZE * get_global_id(1);
    
    if (output_col >= output_row) {
        float sum[COL_TILE_SIZE][ROW_TILE_SIZE];
        
        for (int i = 0; i < COL_TILE_SIZE; i++) {
            for (int j = 0; j < ROW_TILE_SIZE; j++) {
                sum[i][j] = 0.0f;
            }
        }
        
        int index1[COL_TILE_SIZE];
        int index2[ROW_TILE_SIZE];
        
        for (int k = 0; k < COL_TILE_SIZE; k++) {
            index1[k] = input_nrow * (output_col + k) / 4;
        }
        for (int k = 0; k < ROW_TILE_SIZE; k++) {
            index2[k] = input_nrow * (output_row + k) / 4;
        }
        
        for (int k = 0; k < input_nrow / 4; k += 2) {
            for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
                    sum[tile_col][tile_row] += dot(input[index1[tile_col]+ k], input[index2[tile_row]+ k]) +
                    dot(input[index1[tile_col]+ k + 1], input[index2[tile_row]+ k + 1]);
                }
            }
        }

        for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
           for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
               output[(output_row + tile_row) * input_ncol + (output_col + tile_col)] = sum[tile_col][tile_row];
               output[(output_col + tile_col) * input_ncol + (output_row + tile_row)] = sum[tile_col][tile_row];
            }
        }
    }
}
 
 
__kernel void crossprod_f_dot4sum4x2(__private int input_nrow,
                                     __private int input_ncol,
                                     __global float4* input,
                                     __global float* output) {
#define SIDE1 2
#define SIDE2 4
    
    int output_col = SIDE1 * get_global_id(0);
    int output_row = SIDE2 * get_global_id(1);
    
    if (output_col >= output_row) {
        float sum[SIDE1][SIDE2];
        for (int i = 0; i < SIDE1; i++) {
            for (int j = 0; j < SIDE2; j++) {
                sum[i][j] = 0.0f;
            }
        }
        
        int index1[SIDE1];
        int index2[SIDE2];
        for (int k = 0; k < SIDE1; k++) {
            index1[k] = input_nrow * (output_col + k) / 4;
        }
        for (int k = 0; k < SIDE2; k++) {
            index2[k] = input_nrow * (output_row + k) / 4;
        }
        
        for (int k = 0; k < input_nrow / 4; k += 2) {
            sum[0][0] += dot(input[index1[0]+ k], input[index2[0]+ k]) +
            dot(input[index1[0]+ k + 1], input[index2[0]+ k + 1]);
            
            sum[1][0] += dot(input[index1[1]+ k], input[index2[0]+ k]) +
            dot(input[index1[1]+ k + 1], input[index2[0]+ k + 1]);
            
            sum[0][1] += dot(input[index1[0]+ k], input[index2[1]+ k]) +
            dot(input[index1[0]+ k + 1], input[index2[1]+ k + 1]);
            
            sum[1][1] += dot(input[index1[1]+ k], input[index2[1]+ k]) +
            dot(input[index1[1]+ k + 1], input[index2[1]+ k + 1]);
            
            sum[0][2] += dot(input[index1[0]+ k], input[index2[2]+ k]) +
            dot(input[index1[0]+ k + 1], input[index2[2]+ k + 1]);
            
            sum[1][2] += dot(input[index1[1]+ k], input[index2[2]+ k]) +
            dot(input[index1[1]+ k + 1], input[index2[2]+ k + 1]);
            
            sum[0][3] += dot(input[index1[0]+ k], input[index2[3]+ k]) +
            dot(input[index1[0]+ k + 1], input[index2[3]+ k + 1]);
            
            sum[1][3] += dot(input[index1[1]+ k], input[index2[3]+ k]) +
            dot(input[index1[1]+ k + 1], input[index2[3]+ k + 1]);
        }
        
        output[(output_row + 0)* input_ncol + (output_col + 0)] = sum[0][0];
        output[(output_col + 0) * input_ncol + (output_row + 0)] = sum[0][0];
        
        output[(output_row + 0)* input_ncol + (output_col + 1)] = sum[1][0];
        output[(output_col + 1) * input_ncol + (output_row + 0)] = sum[1][0];
        
        
        output[(output_row + 1)* input_ncol + (output_col + 0)] = sum[0][1];
        output[(output_col + 0) * input_ncol + (output_row + 1)] = sum[0][1];
        
        output[(output_row + 1)* input_ncol + (output_col + 1)] = sum[1][1];
        output[(output_col + 1) * input_ncol + (output_row + 1)] = sum[1][1];
        
        
        output[(output_row + 2)* input_ncol + (output_col + 0)] = sum[0][2];
        output[(output_col + 0) * input_ncol + (output_row + 2)] = sum[0][2];
        
        output[(output_row + 2)* input_ncol + (output_col + 1)] = sum[1][2];
        output[(output_col + 1) * input_ncol + (output_row + 2)] = sum[1][2];
        
        
        output[(output_row + 3)* input_ncol + (output_col + 0)] = sum[0][3];
        output[(output_col + 0) * input_ncol + (output_row + 3)] = sum[0][3];
        
        output[(output_row + 3)* input_ncol + (output_col + 1)] = sum[1][3];
        output[(output_col + 1) * input_ncol + (output_row + 3)] = sum[1][3];
    }
    
}
*/

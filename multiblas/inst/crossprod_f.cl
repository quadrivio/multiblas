#ifndef ROW_TILE_SIZE
#define ROW_TILE_SIZE 4
#endif

#ifndef COL_TILE_SIZE
#define COL_TILE_SIZE 2
#endif

__kernel void crossprod_f_naive(__private int input_nrow,
                                __private int input_ncol,
                                __global float* input,
                                __global float* output) {
    int output_col = get_global_id(0);
    int output_row = get_global_id(1);
    
    if (output_col >= output_row) {

        float sum = 0.0f;
        int index1 = input_nrow * output_col;
        int index2 = input_nrow * output_row;
        for (int k = 0; k < input_nrow; k++) {
            sum += input[index1 + k] * input[index2 + k];
        }
        
        output[output_row * input_ncol + output_col] = sum;
        output[output_col * input_ncol + output_row] = sum;
    }
}

__kernel void crossprod_f_naive2(__private int input_nrow,
                                 __private int input_ncol,
                                 __global float2* input,
                                 __global float* output) {
    int output_col = get_global_id(0);
    int output_row = get_global_id(1);
    
    if (output_col >= output_row) {
    
        float sum = 0.0f;
        int index1 = input_nrow * output_col / 2;
        int index2 = input_nrow * output_row / 2;
        for (int k = 0; k < input_nrow / 2; k++) {
            sum += dot(input[index1 + k], input[index2 + k]);
        }
        
        output[output_row * input_ncol + output_col] = sum;
        output[output_col * input_ncol + output_row] = sum;
    }
}

__kernel void crossprod_f_naive4(__private int input_nrow,
                                 __private int input_ncol,
                                 __global float4* input,
                                 __global float* output) {
    int output_col = get_global_id(0);
    int output_row = get_global_id(1);

    if (output_col >= output_row) {
    
        float sum = 0.0f;
        int index1 = input_nrow * output_col / 4;
        int index2 = input_nrow * output_row / 4;
        for (int k = 0; k < input_nrow / 4; k++) {
            sum += dot(input[index1 + k], input[index2 + k]);
        }
        
        output[output_row * input_ncol + output_col] = sum;
        output[output_col * input_ncol + output_row] = sum;
    }
}

__kernel void crossprod_f_naive8(__private int input_nrow,
                                 __private int input_ncol,
                                 __global float8* input,
                                 __global float* output) {
    int output_col = get_global_id(0);
    int output_row = get_global_id(1);
    
    if (output_col >= output_row) {
        
        float sum = 0.0f;
        int index1 = input_nrow * output_col / 8;
        int index2 = input_nrow * output_row / 8;
        for (int k = 0; k < input_nrow / 8; k++) {
            sum += dot(input[index1 + k].hi, input[index2 + k].hi) +
            dot(input[index1 + k].lo, input[index2 + k].lo);
        }
        
        output[output_row * input_ncol + output_col] = sum;
        output[output_col * input_ncol + output_row] = sum;
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

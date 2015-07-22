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

__kernel void crossprod_d_naive(__private int input_nrow,
                                __private int input_ncol,
                                __global double* input,
                                __global double* output) {
    int output_col = get_global_id(0);
    int output_row = get_global_id(1);
    
    if (output_col >= output_row) {
    
        double sum = 0.0;
        int index1 = input_nrow * output_col;
        int index2 = input_nrow * output_row;
        for (int k = 0; k < input_nrow; k++) {
            sum += input[index1 + k] * input[index2 + k];
        }
        
        output[output_row * input_ncol + output_col] = sum;
        output[output_col * input_ncol + output_row] = sum;
    }
}

__kernel void crossprod_d_naive2(__private int input_nrow,
                                 __private int input_ncol,
                                 __global double2* input,
                                 __global double* output) {
    int output_col = get_global_id(0);
    int output_row = get_global_id(1);
    
    if (output_col >= output_row) {
    
        double sum = 0.0;
        int index1 = input_nrow * output_col / 2;
        int index2 = input_nrow * output_row / 2;
        for (int k = 0; k < input_nrow / 2; k++) {
            sum += dot(input[index1 + k], input[index2 + k]);
        }
        
        output[output_row * input_ncol + output_col] = sum;
        output[output_col * input_ncol + output_row] = sum;
    }
}

__kernel void crossprod_d_naive4(__private int input_nrow,
                                 __private int input_ncol,
                                 __global double4* input,
                                 __global double* output) {
    int output_col = get_global_id(0);
    int output_row = get_global_id(1);
    
    if (output_col >= output_row) {
        
        double sum = 0.0;
        int index1 = input_nrow * output_col / 4;
        int index2 = input_nrow * output_row / 4;
        for (int k = 0; k < input_nrow / 4; k++) {
            sum += dot(input[index1 + k], input[index2 + k]);
        }
        
        output[output_row * input_ncol + output_col] = sum;
        output[output_col * input_ncol + output_row] = sum;
    }
}

/*
__kernel void crossprod_d_naive8(__private int input_nrow,
                                 __private int input_ncol,
                                 __private int block_dim_UNUSED,
                                 __private int strip_height_UNUSED,
                                 __global double8* input,
                                 __local double* work_UNUSED,
                                 __global double* output) {
    int output_col = get_global_id(0);
    int output_row = get_global_id(1);
    
    if (output_col >= output_row) {
        
        double sum = 0.0;
        int index1 = input_nrow * output_col / 8;
        int index2 = input_nrow * output_row / 8;
        for (int k = 0; k < input_nrow / 8; k++) {
            sum += dot(input[index1 + k], input[index2 + k]);
        }
        
        output[output_row * input_ncol + output_col] = sum;
        output[output_col * input_ncol + output_row] = sum;
    }
}
*/

__kernel void crossprod_d_dot4(__private int input_nrow,
                               __private int input_ncol,
                               __global double4* input,
                               __global double* output) {
    int output_col = get_global_id(0);
    int output_row = get_global_id(1);
    
    if (output_col >= output_row) {

        double sum = 0.0;
        int index1 = input_nrow * output_col / 4;
        int index2 = input_nrow * output_row / 4;
        for (int k = 0; k < input_nrow / 4; k += 2) {
            sum += dot(input[index1 + k], input[index2 + k]) +
            dot(input[index1 + k + 1], input[index2 + k + 1]);
        }
        
        output[output_row * input_ncol + output_col] = sum;
        output[output_col * input_ncol + output_row] = sum;
    }
    
}

/*
__kernel void crossprod_d_dot8(__private int input_nrow,
                               __private int input_ncol,
                               __private int block_dim_UNUSED,
                               __private int strip_height_UNUSED,
                               __global double8* input,
                               __local double* work_UNUSED,
                               __global double* output) {
    int output_col = get_global_id(0);
    int output_row = get_global_id(1);
    
    if (output_col >= output_row) {
        
        double sum = 0.0;
        int index1 = input_nrow * output_col / 8;
        int index2 = input_nrow * output_row / 8;
        for (int k = 0; k < input_nrow / 8; k += 2) {
            sum += dot(input[index1 + k].hi, input[index2 + k].hi) + dot(input[index1 + k].lo, input[index2 + k].lo) +
            dot(input[index1 + k + 1].hi, input[index2 + k + 1].hi) + dot(input[index1 + k + 1].lo, input[index2 + k + 1].lo);
        }
        
        output[output_row * input_ncol + output_col] = sum;
        output[output_col * input_ncol + output_row] = sum;
    }
    
}
*/

__kernel void crossprod_d_dot4sum4x2(__private int input_nrow,
                                     __private int input_ncol,
                                     __global double4* input,
                                     __global double* output) {
#define SIDE1 2
#define SIDE2 4
    
    int output_col = SIDE1 * get_global_id(0);
    int output_row = SIDE2 * get_global_id(1);
    
    if (output_col >= output_row) {
        double sum[SIDE1][SIDE2];
        for (int i = 0; i < SIDE1; i++) {
            for (int j = 0; j < SIDE2; j++) {
                sum[i][j] = 0.0;
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

#ifndef ROW_TILE_SIZE
#define ROW_TILE_SIZE 4
#endif

#ifndef COL_TILE_SIZE
#define COL_TILE_SIZE 2
#endif

__kernel void crossprod_d_dot4tile(__private int input_nrow,
                                   __private int input_ncol,
                                   __global double4* input,
                                   __global double* output) {
    int output_col = COL_TILE_SIZE * get_global_id(0);
    int output_row = ROW_TILE_SIZE * get_global_id(1);
    
    if (output_col >= output_row) {
        double sum[COL_TILE_SIZE][ROW_TILE_SIZE];
        
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
                output[(output_row + tile_row)* input_ncol + (output_col + tile_col)] = sum[tile_col][tile_row];
                output[(output_col + tile_col) * input_ncol + (output_row + tile_row)] = sum[tile_col][tile_row];
            }
        }
    }
}

#endif


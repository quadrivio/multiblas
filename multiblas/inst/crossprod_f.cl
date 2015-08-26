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

__kernel void crossprod_f_dot4(__private int input_nrow,
                               __private int input_ncol,
                               __global float4* input,
                               __global float* output) {
    int output_col = get_global_id(0);
    int output_row = get_global_id(1);
    
    if (output_col >= output_row) {
        
        float sum = 0.0f;
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

__kernel void crossprod_f_dot8(__private int input_nrow,
                               __private int input_ncol,
                               __global float8* input,
                               __global float* output) {
    int output_col = get_global_id(0);
    int output_row = get_global_id(1);
    
    if (output_col >= output_row) {
        
        float sum = 0.0f;
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

__kernel void crossprod_f_dot4sum4x4(__private int input_nrow,
                                     __private int input_ncol,
                                     __global float4* input,
                                     __global float* output) {
#define SIDE1 4
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
            
            sum[2][0] += dot(input[index1[2]+ k], input[index2[0]+ k]) +
            dot(input[index1[2]+ k + 1], input[index2[0]+ k + 1]);
            
            sum[3][0] += dot(input[index1[3]+ k], input[index2[0]+ k]) +
            dot(input[index1[3]+ k + 1], input[index2[0]+ k + 1]);
            
            // -----
            
            sum[0][1] += dot(input[index1[0]+ k], input[index2[1]+ k]) +
            dot(input[index1[0]+ k + 1], input[index2[1]+ k + 1]);
            
            sum[1][1] += dot(input[index1[1]+ k], input[index2[1]+ k]) +
            dot(input[index1[1]+ k + 1], input[index2[1]+ k + 1]);
            
            sum[2][1] += dot(input[index1[2]+ k], input[index2[1]+ k]) +
            dot(input[index1[2]+ k + 1], input[index2[1]+ k + 1]);
            
            sum[3][1] += dot(input[index1[3]+ k], input[index2[1]+ k]) +
            dot(input[index1[3]+ k + 1], input[index2[1]+ k + 1]);
            
            // -----
            
            sum[0][2] += dot(input[index1[0]+ k], input[index2[2]+ k]) +
            dot(input[index1[0]+ k + 1], input[index2[2]+ k + 1]);
            
            sum[1][2] += dot(input[index1[1]+ k], input[index2[2]+ k]) +
            dot(input[index1[1]+ k + 1], input[index2[2]+ k + 1]);
            
            sum[2][2] += dot(input[index1[2]+ k], input[index2[2]+ k]) +
            dot(input[index1[2]+ k + 1], input[index2[2]+ k + 1]);
            
            sum[3][2] += dot(input[index1[3]+ k], input[index2[2]+ k]) +
            dot(input[index1[3]+ k + 1], input[index2[2]+ k + 1]);
            
            // -----
            
            sum[0][3] += dot(input[index1[0]+ k], input[index2[3]+ k]) +
            dot(input[index1[0]+ k + 1], input[index2[3]+ k + 1]);
            
            sum[1][3] += dot(input[index1[1]+ k], input[index2[3]+ k]) +
            dot(input[index1[1]+ k + 1], input[index2[3]+ k + 1]);
            
            sum[2][3] += dot(input[index1[2]+ k], input[index2[3]+ k]) +
            dot(input[index1[2]+ k + 1], input[index2[3]+ k + 1]);
            
            sum[3][3] += dot(input[index1[3]+ k], input[index2[3]+ k]) +
            dot(input[index1[3]+ k + 1], input[index2[3]+ k + 1]);
        }
        
        output[(output_row + 0)* input_ncol + (output_col + 0)] = sum[0][0];
        output[(output_col + 0) * input_ncol + (output_row + 0)] = sum[0][0];
        
        output[(output_row + 0)* input_ncol + (output_col + 1)] = sum[1][0];
        output[(output_col + 1) * input_ncol + (output_row + 0)] = sum[1][0];
        
        output[(output_row + 0)* input_ncol + (output_col + 2)] = sum[2][0];
        output[(output_col + 2) * input_ncol + (output_row + 0)] = sum[2][0];
        
        output[(output_row + 0)* input_ncol + (output_col + 3)] = sum[3][0];
        output[(output_col + 3) * input_ncol + (output_row + 0)] = sum[3][0];
        
        // -----
        
        output[(output_row + 1)* input_ncol + (output_col + 0)] = sum[0][1];
        output[(output_col + 0) * input_ncol + (output_row + 1)] = sum[0][1];
        
        output[(output_row + 1)* input_ncol + (output_col + 1)] = sum[1][1];
        output[(output_col + 1) * input_ncol + (output_row + 1)] = sum[1][1];
        
        output[(output_row + 1)* input_ncol + (output_col + 2)] = sum[2][1];
        output[(output_col + 2) * input_ncol + (output_row + 1)] = sum[2][1];
        
        output[(output_row + 1)* input_ncol + (output_col + 3)] = sum[3][1];
        output[(output_col + 3) * input_ncol + (output_row + 1)] = sum[3][1];
        
        // -----
        
        output[(output_row + 2)* input_ncol + (output_col + 0)] = sum[0][2];
        output[(output_col + 0) * input_ncol + (output_row + 2)] = sum[0][2];
        
        output[(output_row + 2)* input_ncol + (output_col + 1)] = sum[1][2];
        output[(output_col + 1) * input_ncol + (output_row + 2)] = sum[1][2];
        
        output[(output_row + 2)* input_ncol + (output_col + 2)] = sum[2][2];
        output[(output_col + 2) * input_ncol + (output_row + 2)] = sum[2][2];
        
        output[(output_row + 2)* input_ncol + (output_col + 3)] = sum[3][2];
        output[(output_col + 3) * input_ncol + (output_row + 2)] = sum[3][2];
        
        // -----
        
        output[(output_row + 3)* input_ncol + (output_col + 0)] = sum[0][3];
        output[(output_col + 0) * input_ncol + (output_row + 3)] = sum[0][3];
        
        output[(output_row + 3)* input_ncol + (output_col + 1)] = sum[1][3];
        output[(output_col + 1) * input_ncol + (output_row + 3)] = sum[1][3];
        
        output[(output_row + 3)* input_ncol + (output_col + 2)] = sum[2][3];
        output[(output_col + 2) * input_ncol + (output_row + 3)] = sum[2][3];
        
        output[(output_row + 3)* input_ncol + (output_col + 3)] = sum[3][3];
        output[(output_col + 3) * input_ncol + (output_row + 3)] = sum[3][3];
    }
    
}

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

__kernel void crossprod_f_dot4tileB(__private int input_nrow,
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
            for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
                for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                    sum[tile_col][tile_row] += dot(input[index1[tile_col]+ k], input[index2[tile_row]+ k]) +
                    dot(input[index1[tile_col]+ k + 1], input[index2[tile_row]+ k + 1]);
                }
            }
        }
        
        for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
            for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                output[(output_row + tile_row) * input_ncol + (output_col + tile_col)] = sum[tile_col][tile_row];
                output[(output_col + tile_col) * input_ncol + (output_row + tile_row)] = sum[tile_col][tile_row];
            }
        }
    }
}

__kernel void crossprod_f_dot4tileC(__private int input_nrow,
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
            for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
                for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                    sum[tile_col][tile_row] += dot(input[index1[tile_col]+ k], input[index2[tile_row]+ k]) +
                    dot(input[index1[tile_col]+ k + 1], input[index2[tile_row]+ k + 1]);
                }
            }
        }
        
//        for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
//            for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
//                output[(output_row + tile_row) * input_ncol + (output_col + tile_col)] = sum[tile_col][tile_row];
//                output[(output_col + tile_col) * input_ncol + (output_row + tile_row)] = sum[tile_col][tile_row];
//            }
//        }
        
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

__kernel void crossprod_f_dot4tileD(__private int input_nrow,
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
//            for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
//                for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
//                    sum[tile_col][tile_row] += dot(input[index1[tile_col]+ k], input[index2[tile_row]+ k]) +
//                    dot(input[index1[tile_col]+ k + 1], input[index2[tile_row]+ k + 1]);
//                }
//            }

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
        
        for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
            for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                output[(output_row + tile_row) * input_ncol + (output_col + tile_col)] = sum[tile_col][tile_row];
                output[(output_col + tile_col) * input_ncol + (output_row + tile_row)] = sum[tile_col][tile_row];
            }
        }
    }
}

__kernel void crossprod_f_dot4tileE(__private int input_nrow,
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
            //            for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
            //                for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
            //                    sum[tile_col][tile_row] += dot(input[index1[tile_col]+ k], input[index2[tile_row]+ k]) +
            //                    dot(input[index1[tile_col]+ k + 1], input[index2[tile_row]+ k + 1]);
            //                }
            //            }
            
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
        
//        for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
//            for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
//                output[(output_row + tile_row) * input_ncol + (output_col + tile_col)] = sum[tile_col][tile_row];
//                output[(output_col + tile_col) * input_ncol + (output_row + tile_row)] = sum[tile_col][tile_row];
//            }
//        }

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

__kernel void crossprod_f_dot4tileF(__private int input_nrow,
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
            //            for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
            //                for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
            //                    sum[tile_col][tile_row] += dot(input[index1[tile_col]+ k], input[index2[tile_row]+ k]) +
            //                    dot(input[index1[tile_col]+ k + 1], input[index2[tile_row]+ k + 1]);
            //                }
            //            }
            
            for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
                    sum[tile_col][tile_row] += dot(input[index1[tile_col]+ k], input[index2[tile_row]+ k]) +
                    dot(input[index1[tile_col]+ k + 1], input[index2[tile_row]+ k + 1]);
                }
            }
            
//            sum[0][0] += dot(input[index1[0]+ k], input[index2[0]+ k]) +
//            dot(input[index1[0]+ k + 1], input[index2[0]+ k + 1]);
//            
//            sum[1][0] += dot(input[index1[1]+ k], input[index2[0]+ k]) +
//            dot(input[index1[1]+ k + 1], input[index2[0]+ k + 1]);
//            
//            sum[0][1] += dot(input[index1[0]+ k], input[index2[1]+ k]) +
//            dot(input[index1[0]+ k + 1], input[index2[1]+ k + 1]);
//            
//            sum[1][1] += dot(input[index1[1]+ k], input[index2[1]+ k]) +
//            dot(input[index1[1]+ k + 1], input[index2[1]+ k + 1]);
//            
//            sum[0][2] += dot(input[index1[0]+ k], input[index2[2]+ k]) +
//            dot(input[index1[0]+ k + 1], input[index2[2]+ k + 1]);
//            
//            sum[1][2] += dot(input[index1[1]+ k], input[index2[2]+ k]) +
//            dot(input[index1[1]+ k + 1], input[index2[2]+ k + 1]);
//            
//            sum[0][3] += dot(input[index1[0]+ k], input[index2[3]+ k]) +
//            dot(input[index1[0]+ k + 1], input[index2[3]+ k + 1]);
//            
//            sum[1][3] += dot(input[index1[1]+ k], input[index2[3]+ k]) +
//            dot(input[index1[1]+ k + 1], input[index2[3]+ k + 1]);
            
        }
        
        for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
            for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                output[(output_row + tile_row) * input_ncol + (output_col + tile_col)] = sum[tile_col][tile_row];
                output[(output_col + tile_col) * input_ncol + (output_row + tile_row)] = sum[tile_col][tile_row];
            }
        }
    }
}

__kernel void crossprod_f_dot4tileG(__private int input_nrow,
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
            //            for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
            //                for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
            //                    sum[tile_col][tile_row] += dot(input[index1[tile_col]+ k], input[index2[tile_row]+ k]) +
            //                    dot(input[index1[tile_col]+ k + 1], input[index2[tile_row]+ k + 1]);
            //                }
            //            }
            
            for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
//                for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
//                    sum[tile_col][tile_row] += dot(input[index1[tile_col]+ k], input[index2[tile_row]+ k]) +
//                    dot(input[index1[tile_col]+ k + 1], input[index2[tile_row]+ k + 1]);
//                }

                sum[0][tile_row] += dot(input[index1[0]+ k], input[index2[tile_row]+ k]) +
                dot(input[index1[0]+ k + 1], input[index2[0]+ k + 1]);

                sum[1][tile_row] += dot(input[index1[1]+ k], input[index2[tile_row]+ k]) +
                dot(input[index1[1]+ k + 1], input[index2[tile_row]+ k + 1]);
            }
            
            //            sum[0][0] += dot(input[index1[0]+ k], input[index2[0]+ k]) +
            //            dot(input[index1[0]+ k + 1], input[index2[0]+ k + 1]);
            //
            //            sum[1][0] += dot(input[index1[1]+ k], input[index2[0]+ k]) +
            //            dot(input[index1[1]+ k + 1], input[index2[0]+ k + 1]);
            //
            //            sum[0][1] += dot(input[index1[0]+ k], input[index2[1]+ k]) +
            //            dot(input[index1[0]+ k + 1], input[index2[1]+ k + 1]);
            //
            //            sum[1][1] += dot(input[index1[1]+ k], input[index2[1]+ k]) +
            //            dot(input[index1[1]+ k + 1], input[index2[1]+ k + 1]);
            //
            //            sum[0][2] += dot(input[index1[0]+ k], input[index2[2]+ k]) +
            //            dot(input[index1[0]+ k + 1], input[index2[2]+ k + 1]);
            //
            //            sum[1][2] += dot(input[index1[1]+ k], input[index2[2]+ k]) +
            //            dot(input[index1[1]+ k + 1], input[index2[2]+ k + 1]);
            //
            //            sum[0][3] += dot(input[index1[0]+ k], input[index2[3]+ k]) +
            //            dot(input[index1[0]+ k + 1], input[index2[3]+ k + 1]);
            //
            //            sum[1][3] += dot(input[index1[1]+ k], input[index2[3]+ k]) +
            //            dot(input[index1[1]+ k + 1], input[index2[3]+ k + 1]);
            
        }
        
        for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
            for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                output[(output_row + tile_row) * input_ncol + (output_col + tile_col)] = sum[tile_col][tile_row];
                output[(output_col + tile_col) * input_ncol + (output_row + tile_row)] = sum[tile_col][tile_row];
            }
        }
    }
}

__kernel void crossprod_f_dot4tileH(__private int input_nrow,
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
#pragma unroll
            for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
#pragma unroll
                for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                    sum[tile_col][tile_row] += dot(input[index1[tile_col]+ k], input[index2[tile_row]+ k]) +
                    dot(input[index1[tile_col]+ k + 1], input[index2[tile_row]+ k + 1]);
                }
            }
        }
        
        for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
            for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                output[(output_row + tile_row) * input_ncol + (output_col + tile_col)] = sum[tile_col][tile_row];
                output[(output_col + tile_col) * input_ncol + (output_row + tile_row)] = sum[tile_col][tile_row];
            }
        }
    }
}

__kernel void crossprod_f_dot4tileI(__private int input_nrow,
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
        
        float *sump;
        
        for (int k = 0; k < input_nrow / 4; k += 2) {
            sump = &sum[0][0];
            for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
                for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                    *sump += dot(input[index1[tile_col]+ k], input[index2[tile_row]+ k]) +
                    dot(input[index1[tile_col]+ k + 1], input[index2[tile_row]+ k + 1]);
                    
                    sump++;
                }
            }
        }
        
        sump = &sum[0][0];

        for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
            for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                output[(output_row + tile_row) * input_ncol + (output_col + tile_col)] = *sump;
                output[(output_col + tile_col) * input_ncol + (output_row + tile_row)] = *sump++;
            }
        }
    }
}

__kernel void crossprod_f_dot4tileJ(__private int input_nrow,
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
#pragma unroll
            for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
                int index1_tile_col = index1[tile_col];
#pragma unroll
                for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                    int index2_tile_row = index2[tile_row];
                    
                    sum[tile_col][tile_row] += dot(input[index1_tile_col + k], input[index2_tile_row + k]) +
                    dot(input[index1_tile_col + k + 1], input[index2_tile_row + k + 1]);
                }
            }
        }
        
        for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
            for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                output[(output_row + tile_row) * input_ncol + (output_col + tile_col)] = sum[tile_col][tile_row];
                output[(output_col + tile_col) * input_ncol + (output_row + tile_row)] = sum[tile_col][tile_row];
            }
        }
    }
}

__kernel void crossprod_f_dot4tileK(__private int input_nrow,
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
        
        for (int k = 0; k < input_nrow / 4; k += 2) {
            for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
                int index1_tile_col = input_nrow * (output_col + tile_col) / 4;

                for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                    int index2_tile_row = input_nrow * (output_row + tile_row) / 4;
                    
                    sum[tile_col][tile_row] += dot(input[index1_tile_col + k], input[index2_tile_row + k]) +
                    dot(input[index1_tile_col + k + 1], input[index2_tile_row + k + 1]);
                }
            }
        }
        
        for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
            for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                output[(output_row + tile_row) * input_ncol + (output_col + tile_col)] = sum[tile_col][tile_row];
                output[(output_col + tile_col) * input_ncol + (output_row + tile_row)] = sum[tile_col][tile_row];
            }
        }
    }
}

__kernel void crossprod_f_dot4tileL(__private int input_nrow,
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
            //            for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
            //                for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
            //                    sum[tile_col][tile_row] += dot(input[index1[tile_col]+ k], input[index2[tile_row]+ k]) +
            //                    dot(input[index1[tile_col]+ k + 1], input[index2[tile_row]+ k + 1]);
            //                }
            //            }
            
            float4 input_index1_0_k = input[index1[0]+ k];
            float4 input_index1_0_k_1 = input[index1[0]+ k + 1];
            float4 input_index1_1_k = input[index1[1]+ k];
            float4 input_index1_1_k_1 = input[index1[1]+ k + 1];
            
            sum[0][0] += dot(input_index1_0_k, input[index2[0]+ k]) +
            dot(input_index1_0_k_1, input[index2[0]+ k + 1]);
            
            sum[1][0] += dot(input_index1_1_k, input[index2[0]+ k]) +
            dot(input_index1_1_k_1, input[index2[0]+ k + 1]);
            
            sum[0][1] += dot(input_index1_0_k, input[index2[1]+ k]) +
            dot(input_index1_0_k_1, input[index2[1]+ k + 1]);
            
            sum[1][1] += dot(input_index1_1_k, input[index2[1]+ k]) +
            dot(input_index1_1_k_1, input[index2[1]+ k + 1]);
            
            sum[0][2] += dot(input_index1_0_k, input[index2[2]+ k]) +
            dot(input_index1_0_k_1, input[index2[2]+ k + 1]);
            
            sum[1][2] += dot(input_index1_1_k, input[index2[2]+ k]) +
            dot(input_index1_1_k_1, input[index2[2]+ k + 1]);
            
            sum[0][3] += dot(input_index1_0_k, input[index2[3]+ k]) +
            dot(input_index1_0_k_1, input[index2[3]+ k + 1]);
            
            sum[1][3] += dot(input_index1_1_k, input[index2[3]+ k]) +
            dot(input_index1_1_k_1, input[index2[3]+ k + 1]);
            
        }
        
        for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
            for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                output[(output_row + tile_row) * input_ncol + (output_col + tile_col)] = sum[tile_col][tile_row];
                output[(output_col + tile_col) * input_ncol + (output_row + tile_row)] = sum[tile_col][tile_row];
            }
        }
    }
}

__kernel void crossprod_f_dot4tileM(__private int input_nrow,
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
            float4 input_index1_k[COL_TILE_SIZE];
            float4 input_index1_k_1[COL_TILE_SIZE];
            for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
                input_index1_k[tile_col] = input[index1[tile_col] + k];
                input_index1_k_1[tile_col] = input[index1[tile_col] + k + 1];
            }

            float4 input_index2_k[ROW_TILE_SIZE];
            float4 input_index2_k_1[ROW_TILE_SIZE];
            for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                input_index2_k[tile_row] = input[index2[tile_row] + k];
                input_index2_k_1[tile_row] = input[index2[tile_row] + k + 1];
            }

            for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
                for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                    sum[tile_col][tile_row] += dot(input_index1_k[tile_col], input_index2_k[tile_row]) +
                        dot(input_index1_k_1[tile_col], input_index2_k_1[tile_row]);
                }
            }
        }
        
        for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
            for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                output[(output_row + tile_row) * input_ncol + (output_col + tile_col)] = sum[tile_col][tile_row];
                output[(output_col + tile_col) * input_ncol + (output_row + tile_row)] = sum[tile_col][tile_row];
            }
        }
    }
}


__kernel void crossprod_f_dot8sum4x2(__private int input_nrow,
                                     __private int input_ncol,
                                     __global float8* input,
                                     __global float* output) {
#define SIDE1 2
#define SIDE2 4
    
    int output_col = SIDE1 * get_global_id(0);
    int output_row = SIDE2 * get_global_id(1);
    
    if (output_col >= output_row) {
        float sum[SIDE1][SIDE2];
#pragma unroll SIDE1
        for (int i = 0; i < SIDE1; i++) {
#pragma unroll SIDE2
            for (int j = 0; j < SIDE2; j++) {
                sum[i][j] = 0.0f;
            }
        }
        
        int index1[SIDE1];
        int index2[SIDE2];
        for (int k = 0; k < SIDE1; k++) {
            index1[k] = input_nrow * (output_col + k) / 8;
        }
        for (int k = 0; k < SIDE2; k++) {
            index2[k] = input_nrow * (output_row + k) / 8;
        }
        for (int k = 0; k < input_nrow / 8; k += 2) {
            sum[0][0] += dot(input[index1[0] + k].hi, input[index2[0] + k].hi) +
                dot(input[index1[0] + k].lo, input[index2[0] + k].lo) +
                dot(input[index1[0] + k + 1].hi, input[index2[0] + k + 1].hi) +
                dot(input[index1[0] + k + 1].lo, input[index2[0] + k + 1].lo);
            
            sum[1][0] += dot(input[index1[1] + k].hi, input[index2[0] + k].hi) +
                dot(input[index1[1] + k].lo, input[index2[0] + k].lo) +
                dot(input[index1[1] + k + 1].hi, input[index2[0] + k + 1].hi) +
                dot(input[index1[1] + k + 1].lo, input[index2[0] + k + 1].lo);
            
            sum[0][1] += dot(input[index1[0] + k].hi, input[index2[1] + k].hi) +
                dot(input[index1[0] + k].lo, input[index2[1] + k].lo) +
                dot(input[index1[0] + k + 1].hi, input[index2[1] + k + 1].hi) +
                dot(input[index1[0] + k + 1].lo, input[index2[1] + k + 1].lo);
            
            sum[1][1] += dot(input[index1[1] + k].hi, input[index2[1] + k].hi) +
                dot(input[index1[1] + k].lo, input[index2[1] + k].lo) +
                dot(input[index1[1] + k + 1].hi, input[index2[1] + k + 1].hi) +
                dot(input[index1[1] + k + 1].lo, input[index2[1] + k + 1].lo);
            
            sum[0][2] += dot(input[index1[0] + k].hi, input[index2[2] + k].hi) +
                dot(input[index1[0] + k].lo, input[index2[2] + k].lo) +
                dot(input[index1[0] + k + 1].hi, input[index2[2] + k + 1].hi) +
                dot(input[index1[0] + k + 1].lo, input[index2[2] + k + 1].lo);
            
            sum[1][2] += dot(input[index1[1] + k].hi, input[index2[2] + k].hi) +
                dot(input[index1[1] + k].lo, input[index2[2] + k].lo) +
                dot(input[index1[1] + k + 1].hi, input[index2[2] + k + 1].hi) +
                dot(input[index1[1] + k + 1].lo, input[index2[2] + k + 1].lo);
            
            sum[0][3] += dot(input[index1[0] + k].hi, input[index2[3] + k].hi) +
                dot(input[index1[0] + k].lo, input[index2[3] + k].lo) +
                dot(input[index1[0] + k + 1].hi, input[index2[3] + k + 1].hi) +
                dot(input[index1[0] + k + 1].lo, input[index2[3] + k + 1].lo);
            
            sum[1][3] += dot(input[index1[1] + k].hi, input[index2[3] + k].hi) +
                dot(input[index1[1] + k].lo, input[index2[3] + k].lo) +
                dot(input[index1[1] + k + 1].hi, input[index2[3] + k + 1].hi) +
                dot(input[index1[1] + k + 1].lo, input[index2[3] + k + 1].lo);
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

__kernel void crossprod_f_dot8tile(__private int input_nrow,
                                     __private int input_ncol,
                                     __global float8* input,
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
            index1[k] = input_nrow * (output_col + k) / 8;
        }
        for (int k = 0; k < ROW_TILE_SIZE; k++) {
            index2[k] = input_nrow * (output_row + k) / 8;
        }
        
        for (int k = 0; k < input_nrow / 8; k += 2) {
            for (int tile_row = 0; tile_row < ROW_TILE_SIZE; tile_row++) {
                for (int tile_col = 0; tile_col < COL_TILE_SIZE; tile_col++) {
                    sum[tile_col][tile_row] += dot(input[index1[tile_col] + k].hi, input[index2[tile_row] + k].hi) +
                        dot(input[index1[tile_col] + k].lo, input[index2[tile_row] + k].lo) +
                        dot(input[index1[tile_col] + k + 1].hi, input[index2[tile_row] + k + 1].hi) +
                        dot(input[index1[tile_col] + k + 1].lo, input[index2[tile_row] + k + 1].lo);
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

__kernel void crossprod_f_units8dreg(__private int input_nrow,
                                     __private int input_ncol,
                                     __global float8* input,
                                     __global float* output) {
#define ZIDE1 2
#define ZIDE2 4
    
    int output_col = ZIDE1 * get_global_id(0);
    int output_row = ZIDE2 * get_global_id(1);
    
    if (output_col >= output_row) {
        float sum[ZIDE1][ZIDE2];
        for (int i = 0; i < ZIDE1; i++) {
            for (int j = 0; j < ZIDE2; j++) {
                sum[i][j] = 0.0f;
            }
        }
        
        int index1[ZIDE1];
        int index2[ZIDE2];
        for (int k = 0; k < ZIDE1; k++) {
            index1[k] = input_nrow * (output_col + k) / 8;
        }
        for (int k = 0; k < ZIDE2; k++) {
            index2[k] = input_nrow * (output_row + k) / 8;
        }
        
        for (int k = 0; k < input_nrow / 8; k += 2) {
            for (int i = 0; i < ZIDE1; i++) {
                for (int j = 0; j < ZIDE2; j++) {
                    sum[i][j] += dot(input[index1[i] + k].hi, input[index2[j] + k].hi) + dot(input[index1[i] + k].lo, input[index2[j] + k].lo) +
                    dot(input[index1[i] + k + 1].hi, input[index2[j] + k + 1].hi) + dot(input[index1[i] + k + 1].lo, input[index2[j] + k + 1].lo);
                }
            }
        }
        
        for (int i = 0; i < ZIDE1; i++) {
            for (int j = 0; j < ZIDE2; j++) {
                output[(output_row + j)* input_ncol + (output_col + i)] = sum[i][j];
                output[(output_col + i) * input_ncol + (output_row + j)] = sum[i][j];
            }
        }
    }
    
}

// -------------------------------------------------------------------------------------------------

__kernel void crossprod_f_units8d4x2(__private int input_nrow,
                                     __private int input_ncol,
                                     __global float8* input,
                                     __global float* output) {
#define SIDE1 2
#define SIDE2 4
    
    int output_col = SIDE1 * get_global_id(0);
    int output_row = SIDE2 * get_global_id(1);
    
    if (output_col >= output_row) {
        float sum[SIDE1][SIDE2];
#pragma unroll SIDE1
        for (int i = 0; i < SIDE1; i++) {
#pragma unroll SIDE2
            for (int j = 0; j < SIDE2; j++) {
                sum[i][j] = 0.0f;
            }
        }
        
        int index1[SIDE1];
        int index2[SIDE2];
        for (int k = 0; k < SIDE1; k++) {
            index1[k] = input_nrow * (output_col + k) / 8;
        }
        for (int k = 0; k < SIDE2; k++) {
            index2[k] = input_nrow * (output_row + k) / 8;
        }
        
        for (int k = 0; k < input_nrow / 8; k += 2) {
            sum[0][0] += dot(input[index1[0] + k].hi, input[index2[0] + k].hi) + dot(input[index1[0] + k].lo, input[index2[0] + k].lo) +
            dot(input[index1[0] + k + 1].hi, input[index2[0] + k + 1].hi) + dot(input[index1[0] + k + 1].lo, input[index2[0] + k + 1].lo);
            
            sum[1][0] += dot(input[index1[1] + k].hi, input[index2[0] + k].hi) + dot(input[index1[1] + k].lo, input[index2[0] + k].lo) +
            dot(input[index1[1] + k + 1].hi, input[index2[0] + k + 1].hi) + dot(input[index1[1] + k + 1].lo, input[index2[0] + k + 1].lo);
            
            sum[0][1] += dot(input[index1[0] + k].hi, input[index2[1] + k].hi) + dot(input[index1[0] + k].lo, input[index2[1] + k].lo) +
            dot(input[index1[0] + k + 1].hi, input[index2[1] + k + 1].hi) + dot(input[index1[0] + k + 1].lo, input[index2[1] + k + 1].lo);
            
            sum[1][1] += dot(input[index1[1] + k].hi, input[index2[1] + k].hi) + dot(input[index1[1] + k].lo, input[index2[1] + k].lo) +
            dot(input[index1[1] + k + 1].hi, input[index2[1] + k + 1].hi) + dot(input[index1[1] + k + 1].lo, input[index2[1] + k + 1].lo);
            
            sum[0][2] += dot(input[index1[0] + k].hi, input[index2[2] + k].hi) + dot(input[index1[0] + k].lo, input[index2[2] + k].lo) +
            dot(input[index1[0] + k + 1].hi, input[index2[2] + k + 1].hi) + dot(input[index1[0] + k + 1].lo, input[index2[2] + k + 1].lo);
            
            sum[1][2] += dot(input[index1[1] + k].hi, input[index2[2] + k].hi) + dot(input[index1[1] + k].lo, input[index2[2] + k].lo) +
            dot(input[index1[1] + k + 1].hi, input[index2[2] + k + 1].hi) + dot(input[index1[1] + k + 1].lo, input[index2[2] + k + 1].lo);
            
            sum[0][3] += dot(input[index1[0] + k].hi, input[index2[3] + k].hi) + dot(input[index1[0] + k].lo, input[index2[3] + k].lo) +
            dot(input[index1[0] + k + 1].hi, input[index2[3] + k + 1].hi) + dot(input[index1[0] + k + 1].lo, input[index2[3] + k + 1].lo);
            
            sum[1][3] += dot(input[index1[1] + k].hi, input[index2[3] + k].hi) + dot(input[index1[1] + k].lo, input[index2[3] + k].lo) +
            dot(input[index1[1] + k + 1].hi, input[index2[3] + k + 1].hi) + dot(input[index1[1] + k + 1].lo, input[index2[3] + k + 1].lo);
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


__kernel void crossprod_f_units8d4x2_0(__private int input_nrow,
                                     __private int input_ncol,
                                     __private int block_dim_UNUSED,
                                     __private int strip_height_UNUSED,
                                     __global float8* input,
                                     __local float* work_UNUSED,
                                     __global float* output) {
#define SIDE1 2
#define SIDE2 4
    
    int output_col = SIDE1 * get_global_id(0);
    int output_row = SIDE2 * get_global_id(1);
    
    if (output_col >= output_row) {
        float sum[SIDE1][SIDE2];
#pragma unroll SIDE1
        for (int i = 0; i < SIDE1; i++) {
#pragma unroll SIDE2
            for (int j = 0; j < SIDE2; j++) {
                sum[i][j] = 0.0f;
            }
        }
        
        //        sum[0][0] = 0.0;
        //        sum[1][0] = 0.0;
        //
        //        sum[0][1] = 0.0;
        //        sum[1][1] = 0.0;
        //
        //        sum[0][2] = 0.0;
        //        sum[1][2] = 0.0;
        //
        //        sum[0][3] = 0.0;
        //        sum[1][3] = 0.0;
        
        int index1[SIDE1];
        int index2[SIDE2];
        for (int k = 0; k < SIDE1; k++) {
            index1[k] = input_nrow * (output_col + k) / 8;
        }
        for (int k = 0; k < SIDE2; k++) {
            index2[k] = input_nrow * (output_row + k) / 8;
        }
        //        index1[0] = input_nrow * (output_col + 0) / 8;
        //        index1[1] = input_nrow * (output_col + 1) / 8;
        //        index2[0] = input_nrow * (output_row + 0) / 8;
        //        index2[1] = input_nrow * (output_row + 1) / 8;
        //        index2[2] = input_nrow * (output_row + 2) / 8;
        //        index2[3] = input_nrow * (output_row + 3) / 8;
        
        for (int k = 0; k < input_nrow / 8; k += 2) {
            //            float8 value1[SIDE];
            //            float8 value1p1[SIDE];
            //            float8 value2[SIDE];
            //            float8 value2p1[SIDE];
            //
            //            for (int i = 0; i < SIDE; i++) {
            //                value1[i] = input[index1[i] + k];
            //                value1p1[i] = input[index1[i] + k + 1];
            //                value2[i] = input[index2[i] + k];
            //                value2p1[i] = input[index2[i] + k + 1];
            //            }
            
            sum[0][0] += dot(input[index1[0] + k].hi, input[index2[0] + k].hi) + dot(input[index1[0] + k].lo, input[index2[0] + k].lo) +
            dot(input[index1[0] + k + 1].hi, input[index2[0] + k + 1].hi) + dot(input[index1[0] + k + 1].lo, input[index2[0] + k + 1].lo);
            
            sum[1][0] += dot(input[index1[1] + k].hi, input[index2[0] + k].hi) + dot(input[index1[1] + k].lo, input[index2[0] + k].lo) +
            dot(input[index1[1] + k + 1].hi, input[index2[0] + k + 1].hi) + dot(input[index1[1] + k + 1].lo, input[index2[0] + k + 1].lo);
            
            sum[0][1] += dot(input[index1[0] + k].hi, input[index2[1] + k].hi) + dot(input[index1[0] + k].lo, input[index2[1] + k].lo) +
            dot(input[index1[0] + k + 1].hi, input[index2[1] + k + 1].hi) + dot(input[index1[0] + k + 1].lo, input[index2[1] + k + 1].lo);
            
            sum[1][1] += dot(input[index1[1] + k].hi, input[index2[1] + k].hi) + dot(input[index1[1] + k].lo, input[index2[1] + k].lo) +
            dot(input[index1[1] + k + 1].hi, input[index2[1] + k + 1].hi) + dot(input[index1[1] + k + 1].lo, input[index2[1] + k + 1].lo);
            
            sum[0][2] += dot(input[index1[0] + k].hi, input[index2[2] + k].hi) + dot(input[index1[0] + k].lo, input[index2[2] + k].lo) +
            dot(input[index1[0] + k + 1].hi, input[index2[2] + k + 1].hi) + dot(input[index1[0] + k + 1].lo, input[index2[2] + k + 1].lo);
            
            sum[1][2] += dot(input[index1[1] + k].hi, input[index2[2] + k].hi) + dot(input[index1[1] + k].lo, input[index2[2] + k].lo) +
            dot(input[index1[1] + k + 1].hi, input[index2[2] + k + 1].hi) + dot(input[index1[1] + k + 1].lo, input[index2[2] + k + 1].lo);
            
            sum[0][3] += dot(input[index1[0] + k].hi, input[index2[3] + k].hi) + dot(input[index1[0] + k].lo, input[index2[3] + k].lo) +
            dot(input[index1[0] + k + 1].hi, input[index2[3] + k + 1].hi) + dot(input[index1[0] + k + 1].lo, input[index2[3] + k + 1].lo);
            
            sum[1][3] += dot(input[index1[1] + k].hi, input[index2[3] + k].hi) + dot(input[index1[1] + k].lo, input[index2[3] + k].lo) +
            dot(input[index1[1] + k + 1].hi, input[index2[3] + k + 1].hi) + dot(input[index1[1] + k + 1].lo, input[index2[3] + k + 1].lo);
            
            //            for (int i = 0; i < SIDE1; i++) {
            //                for (int j = 0; j < SIDE2; j++) {
            //                    //                    sum[i][j] += dot(value1[i].hi, value2[j].hi) + dot(value1[i].lo, value2[j].lo) +
            //                    //                    dot(value1p1[i].hi, value2p1[j].hi) + dot(value1p1[i].lo, value2p1[j].lo);
            //
            //                    sum[i][j] += dot(input[index1[i] + k].hi, input[index2[j] + k].hi) + dot(input[index1[i] + k].lo, input[index2[j] + k].lo) +
            //                    dot(input[index1[i] + k + 1].hi, input[index2[j] + k + 1].hi) + dot(input[index1[i] + k + 1].lo, input[index2[j] + k + 1].lo);
            //                }
            //            }
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
        
        //        for (int i = 0; i < SIDE1; i++) {
        //            for (int j = 0; j < SIDE2; j++) {
        //                output[(output_row + j)* input_ncol + (output_col + i)] = sum[i][j];
        //                output[(output_col + i) * input_ncol + (output_row + j)] = sum[i][j];
        //            }
        //        }
    }
    
}


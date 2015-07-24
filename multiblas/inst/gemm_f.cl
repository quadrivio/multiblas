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

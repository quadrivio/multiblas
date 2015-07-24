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
    output[output_col * inputA_nrow + output_row] += alpha * sum];
}

#endif

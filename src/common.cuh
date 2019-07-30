/**
 * GNMT LSTM GEMM Optimization.
 * Part of the code comes from `https://github.com/tbennun/cudnn-training`.
 *
 * Author: Wil Kong
 * Date: 01/22/2018 Mon
 */

#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>
#include <cublas_v2.h>

#define ITERATIONS 100

//////////////////////////////////////////////////////////////////////////////
// Error handling
// Adapted from the CUDNN classification code 
// sample: https://developer.nvidia.com/cuDNN

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCublasErrors(status) do {                                 \
    std::stringstream _error;                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                             \
      _error << "CUBLAS failure: " << status;                          \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != cudaSuccess) {                                       \
      _error << "Cuda failure(" << status << "): "                     \
        << cudaGetErrorString(status);                                 \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCUErrors(status) do {                                     \
    std::stringstream _error;                                          \
    const char** pStr = NULL;                                          \
    if (status != 0) {                                                 \
      cuGetErrorString(status, pStr);                                  \
      _error << "Cuda failure: " << pStr;                              \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCurandErrors(status) do {                                 \
    std::stringstream _error;                                          \
    if (status != CURAND_STATUS_SUCCESS) {                             \
      _error << "Curand failure: " << status;                          \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

//////////////////////////////////////////////////////////////////////////////
// GPU kernels


//////////////////////////////////////////////////////////////////////////////
// Utils

#if __CUDA_ARCH__ >= 350
__launch_bounds__(512, 4)
#endif
__global__ void
fp16tofp32(float *dst, __half *src, size_t size) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)  return;

    dst[idx] = __half2float(src[idx]);
}

#if __CUDA_ARCH__ >= 350
__launch_bounds__(512, 4)
#endif
__global__ void
fp32tofp16(__half *dst, float *src, size_t size) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)  return;

    dst[idx] = __float2half(src[idx]);
}

__global__ void initGPUData_ker(__half *data, uint64_t numElements, int seed) {
   uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < numElements) {
      data[idx] = __float2half(((float)(((idx+seed)%253)+1))/256.0);
   } 
}
      
__global__ void initGPUData_ker(float *data, uint64_t numElements, int seed) {
   uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < numElements) {
      data[idx] = ((float)(((idx+seed)%253)+1))/256.0;
   } 
}

template <typename TensorType>
void initGPUData(TensorType *data, uint64_t numElements, float seed) {
   dim3 gridDim;
   dim3 blockDim;
   
   blockDim.x = 1024;
   gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;
      
   initGPUData_ker<<< gridDim, blockDim >>>(data, numElements, seed);
}

__global__ void print4dTensor(__half *tensor, int N, int C, int H, int W) {
    for (int n=0; n<N; n++) {
        for (int c=0; c<C; c++) {
            printf("===%d  %d===\n", n, c);
            for (int h=0; h<H; h++) {
                for (int w=0; w<W; w++) {
                    printf("%.4f ", __half2float(tensor[n*C*H*W + c*H*W + h*W + w]));
                    //printf("%04x ", *(uint16_t *)(&tensor[n*C*H*W + c*H*W + h*W + w]));
                }
                printf("\n");
            }
        }
    }
}

__global__ void print4dTensor(float *tensor, int N, int C, int H, int W) {
    for (int n=0; n<N; n++) {
        for (int c=0; c<C; c++) {
            printf("===%d  %d===\n", n, c);
            for (int h=0; h<H; h++) {
                for (int w=0; w<W; w++) {
                    printf("%.4f ", tensor[n*C*H*W + c*H*W + h*W + w]);
                }
                printf("\n");
            }
        }
    }
}

__global__ void checkEquality(__half *a, __half *b, uint64_t size, float epislon, float *err) {
    uint64_t absCount = 0;
    float max_diff = 0.0f;
    __half max_a, max_b;

    for (int64_t i=0; i<size; i++)  {
        float diff = abs(__half2float(a[i]) - __half2float(b[i]));
        err[i] = (__half2float(a[i]) - __half2float(b[i])) / __half2float(a[i]);
        //if (abs(__half2float(a[i]) - __half2float(b[i])) > epislon)
        if (diff > epislon)
            absCount++;
        if (diff > max_diff) {
            max_diff = diff;
            max_a = a[i];
            max_b = b[i];
        }
    }
    printf("Number of mismatch: %ld\n", absCount);
    printf("Maximum difference: %.4f, a: %.4f (%#04x), b: %.4f (%#04x)\n", max_diff,
        __half2float(max_a), *(uint16_t *)(&max_a),
        __half2float(max_b), *(uint16_t *)(&max_b));
}

__global__ void checkEquality(float *a, float *b, uint64_t size, float epislon, float *err) {
    uint64_t absCount = 0;
    float max_diff = 0.0f, max_a, max_b;

    for (uint64_t i=0; i<size; i++)  {
        float diff = abs(a[i] - b[i]);
        err[i] = (a[i] - b[i]) / a[i];
        if (diff > epislon)
            absCount++;
        if (diff > max_diff) {
            max_diff = diff;
            max_a = a[i];
            max_b = b[i];
        }
    }
    printf("Number of mismatch: %ld\n", absCount);
    printf("Maximum difference: %.4f, a: %.4f (%#08x), b: %.4f (%#08x)\n", max_diff,
        max_a, *(uint32_t *)(&max_a), max_b, *(uint32_t *)(&max_b));
}

__global__ void checkEquality(float *a, __half *b, uint64_t size, float epislon, float *err) {
    uint64_t absCount = 0;
    float max_diff = 0.0f, max_a;
    __half max_b;

    for (int64_t i=0; i<size; i++)  {
        float diff = abs(a[i] - __half2float(b[i]));
        err[i] = (a[i] - __half2float(b[i])) / a[i];
        if (diff > epislon)
            absCount++;
        if (diff > max_diff) {
            max_diff = diff;
            max_a = a[i];
            max_b = b[i];
        }
    }
    printf("Number of mismatch: %ld\n", absCount);
    printf("Maximum difference: %.4f, a: %.4f (%#08x), b: %.4f (%#04x)\n", max_diff,
        max_a, *(uint32_t *)(&max_a),
        __half2float(max_b), *(uint16_t *)(&max_b));
}


/**
 * GNMT LSTM GEMM Optimization.
 * Part of the code comes from `https://github.com/tbennun/cudnn-training`.
 *
 * Author: Wil Kong
 * Date: 01/22/2018 Mon
 */

#include "common.cuh"

//////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])  {
    uint64_t seq_length = 38, input_size = 1024, hidden_size = 1024, batch_size = 32;
    uint64_t num_layer = 3;
    uint64_t num_mat = 4;  // lstm
    bool graph_launch = false;

    /** Setup environments. */
    if (argc >= 7) {
        graph_launch = static_cast<bool>(atoi(argv[1]));
        batch_size = static_cast<uint64_t>(atoi(argv[2]));
        seq_length = static_cast<uint64_t>(atoi(argv[3]));
        input_size = static_cast<uint64_t>(atoi(argv[4]));
        hidden_size = static_cast<uint64_t>(atoi(argv[5]));
        num_layer = static_cast<uint64_t>(atoi(argv[6]));
    }
   
    /** Declare handle and status. */ 
    //cudaError_t cuda_status;    
    cublasStatus_t cublas_status;
    cublasHandle_t cublas_handle;

    /** Declare cublas algorithms. */
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    /** Setup GPU property. */
    int gpuid = 0;
    checkCudaErrors(cudaSetDevice(gpuid));

    /** Initialize the cublas handle. */
    checkCublasErrors(cublasCreate(&cublas_handle));
   
    /** Setup math property. */
    typedef __half io_type;
    typedef float math_type;
    cublasMath_t cublas_math_mode = CUBLAS_TENSOR_OP_MATH;
    cublasSetMathMode(cublas_handle, cublas_math_mode); 

    /** Setup algorithm. */
    if (argc >= 8) {
        algo = static_cast<cublasGemmAlgo_t>(atoi(argv[7]));
    }

    /** Create sperate stream for layer gemm and recurrent gemm(high priority). */
    cudaStream_t stream[num_layer];
    for (int i=0; i<num_layer; i++)  {
        checkCudaErrors(cudaStreamCreateWithPriority(&stream[i],
            cudaStreamNonBlocking, -1));
    }

    /** Create events to synchronize layer gemm and recurrent gemm. */
    cudaEvent_t events[num_layer][seq_length];
    for (int i=0; i<num_layer; i++)  {
        for (int j=0; j<seq_length; j++)  {
            checkCudaErrors(cudaEventCreate(&events[i][j]));
        }
    }

    /** Create events to monitor time elapsed. */
    float milliseconds = 0;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    /** Setup gemm dimensions. */
    int m = num_mat * hidden_size;
    int k = input_size + hidden_size;
    int n = batch_size;

    /**
     * Allocate matrix used by layer gemm and recurrent gemm.
     * Just to see the perf, initialization doesn't matter.
     */
    float *mw[num_layer];
    float *mi[num_layer];
    float *mo[num_layer];
    for (int i=0; i<num_layer; i++)  {
        checkCudaErrors(cudaMalloc((void **)&mw[i], sizeof(io_type) * m * k));
        checkCudaErrors(cudaMalloc((void **)&mi[i], sizeof(io_type) * k * n));
        checkCudaErrors(cudaMalloc((void **)&mo[i], sizeof(io_type) * m * n));
    }
    
    /** Cudnn 7.1 use nn gemm for fprop. */
    cublasOperation_t transa = CUBLAS_OP_N, transb = CUBLAS_OP_N;

    /** Setup constant. */
    float alpha = 1.0f, beta = 0.0f;

    /** Graph launch. */
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    uint64_t num_iter;
    if (graph_launch)  {
        num_iter = 1;
        checkCudaErrors(cudaGraphCreate(&graph, 0));
        checkCudaErrors(cudaStreamBeginCapture(stream[0], cudaStreamCaptureModeGlobal));
    } else {
        num_iter = ITERATIONS;
        checkCudaErrors(cudaEventRecord(start));
    }

    /** Emulate the cudnn lstm fprop. */
    for (int iter=0; iter<num_iter; iter++) {
        for (int d=0; d<num_layer+seq_length-1; d++)  {
            for (int i=0; i<seq_length; i++)  {
                /** Boundary check. */
                int l = d - i;
                if (l < 0 || l >= num_layer)  {
                    continue;
                }

                /** Layer limitation. */
                if (l > 0)  {
                    checkCudaErrors(cudaStreamWaitEvent(stream[l], events[l-1][i], 0));
                }

                /** Combined GEMM. */
                checkCublasErrors(cublasSetStream(cublas_handle, stream[l]));
                cublas_status = cublasGemmEx(cublas_handle,
                                transa, transb,
                                m, n, k,
                                &alpha,
                                mw[l], CUDA_R_16F, transa == CUBLAS_OP_N ? m : k,
                                mi[l], CUDA_R_16F, k,
                                &beta,
                                mo[l], CUDA_R_16F, m,
                                CUDA_R_32F,
                                algo);
                checkCublasErrors(cublas_status);

                /** Element-wise operation. */
                checkCudaErrors(cudaEventRecord(events[l][i], stream[l]));
            }

        }

        /**
         * Make sure all works done.
         * Stream capture requires same stream in, same stream out.
         */
        for (int i=0; i<num_layer; i++)  {
            checkCudaErrors(cudaStreamWaitEvent(stream[0],
                events[i][seq_length-1], 0));
        }
    }

    if (graph_launch)  {
        checkCudaErrors(cudaStreamEndCapture(stream[0], &graph));
        checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

        checkCudaErrors(cudaEventRecord(start));
        for (int i = 0; i < ITERATIONS; i++)  {
            checkCudaErrors(cudaGraphLaunch(graphExec, 0));
        }
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    //checkCudaErrors(cudaStreamSynchronize(0));
    //checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    /** Print time and gflops. */
    double fma = ITERATIONS * num_layer * num_mat * hidden_size * (input_size + hidden_size) * batch_size * seq_length;
    std::printf("CUDA elapsed: %.3f ms, %.3f Tflops\n", \
        milliseconds/ITERATIONS, fma*2/milliseconds/1e9);
    
    /** Free the device memory. */
    for (int i=0; i<num_layer; i++)  {
        checkCudaErrors(cudaFree(mw[i]));
        checkCudaErrors(cudaFree(mi[i]));
        checkCudaErrors(cudaFree(mo[i]));
    }

    /** Destroy events. */
    for (int i=0; i<num_layer; i++)  {
        for (int j=0; j<seq_length; j++)  {
            checkCudaErrors(cudaEventDestroy(events[i][j]));
        }
    }
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    /** Destory streams. */
    for (int i=0; i<num_layer; i++)  {
        checkCudaErrors(cudaStreamDestroy(stream[i]));
    }

    /** Destroy CUDA graph. */
    if (graph_launch) {
        checkCudaErrors(cudaGraphDestroy(graph));
        checkCudaErrors(cudaGraphExecDestroy(graphExec));
    }

    /** Free the cublas handle. */
    checkCublasErrors(cublasDestroy(cublas_handle));

    return 0;
}

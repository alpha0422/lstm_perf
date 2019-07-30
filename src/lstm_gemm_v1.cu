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
    uint64_t fuse_factor = 8;  // timestep fusion factor
    uint64_t reuse_length = 8;  // workspace limitation
    bool graph_launch = false;

    /** Setup environments. */
    if (argc >= 9) {
        graph_launch = static_cast<bool>(atoi(argv[1]));
        batch_size = static_cast<uint64_t>(atoi(argv[2]));
        seq_length = static_cast<uint64_t>(atoi(argv[3]));
        input_size = static_cast<uint64_t>(atoi(argv[4]));
        hidden_size = static_cast<uint64_t>(atoi(argv[5]));
        num_layer = static_cast<uint64_t>(atoi(argv[6]));
        fuse_factor = static_cast<uint64_t>(atoi(argv[7]));
        reuse_length = static_cast<uint64_t>(atoi(argv[8]));
    }
   
    /** Declare handle and status. */ 
    //cudaError_t cuda_status;    
    cublasStatus_t cublas_status;
    cublasHandle_t cublas_handle;

    /** Declare cublas algorithms. */
    cublasGemmAlgo_t lalgo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    cublasGemmAlgo_t ralgo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

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
    if (argc >= 11) {
        lalgo = static_cast<cublasGemmAlgo_t>(atoi(argv[9]));
        ralgo = static_cast<cublasGemmAlgo_t>(atoi(argv[10]));
    }

    /** Create sperate stream for layer gemm and recurrent gemm(high priority). */
    cudaStream_t lstream[num_layer], rstream[num_layer];
    for (int i=0; i<num_layer; i++)  {
        checkCudaErrors(cudaStreamCreateWithPriority(&lstream[i],
            cudaStreamNonBlocking, -1));
        checkCudaErrors(cudaStreamCreateWithPriority(&rstream[i],
            cudaStreamNonBlocking, -1));
    }

    /** Create events to synchronize layer gemm and recurrent gemm. */
    cudaEvent_t revents[num_layer][seq_length], levents[num_layer][seq_length];
    for (int i=0; i<num_layer; i++)  {
        for (int j=0; j<seq_length; j++)  {
            checkCudaErrors(cudaEventCreate(&revents[i][j]));
            checkCudaErrors(cudaEventCreate(&levents[i][j]));
        }
    }

    /** Create events to monitor time elapsed. */
    float milliseconds = 0;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    /** Setup gemm dimensions. */
    int lm = num_mat * hidden_size;
    int lk = input_size;
    int ln = batch_size * fuse_factor;
    int rm = num_mat * hidden_size;
    int rk = hidden_size;
    int rn = batch_size;

    /**
     * Allocate matrix used by layer gemm and recurrent gemm.
     * Just to see the perf, initialization doesn't matter.
     */
    float *lw[num_layer], *rw[num_layer];
    float *li[num_layer], *ri[num_layer];
    float *lo[num_layer], *ro[num_layer];
    for (int i=0; i<num_layer; i++)  {
        checkCudaErrors(cudaMalloc((void **)&lw[i], sizeof(io_type) * lm * lk));
        checkCudaErrors(cudaMalloc((void **)&li[i], sizeof(io_type) * lk * ln));
        checkCudaErrors(cudaMalloc((void **)&lo[i], sizeof(io_type) * lm * ln));
        checkCudaErrors(cudaMalloc((void **)&rw[i], sizeof(io_type) * rm * rk));
        checkCudaErrors(cudaMalloc((void **)&ri[i], sizeof(io_type) * rk * rn));
        checkCudaErrors(cudaMalloc((void **)&ro[i], sizeof(io_type) * rm * rn));
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
        checkCudaErrors(cudaStreamBeginCapture(lstream[0], cudaStreamCaptureModeGlobal));
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

                /** Workspace limitation. */
                if (i >= reuse_length)  {
                    checkCudaErrors(cudaStreamWaitEvent(lstream[l],
                        revents[l][i-reuse_length], 0));
                }

                /** Layer limitation. */
                if (l > 0)  {
                    checkCudaErrors(cudaStreamWaitEvent(lstream[l], revents[l-1][i], 0));
                }

                /** Layer GEMM. */
                if (i % fuse_factor == 0)  {
                    checkCublasErrors(cublasSetStream(cublas_handle, lstream[l]));
                    cublas_status = cublasGemmEx(cublas_handle,
                                    transa, transb,
                                    lm, ln, lk,
                                    &alpha,
                                    lw[l], CUDA_R_16F, transa == CUBLAS_OP_N ? lm : lk,
                                    li[l], CUDA_R_16F, lk,
                                    &beta,
                                    lo[l], CUDA_R_16F, lm,
                                    CUDA_R_32F,
                                    lalgo);
                    checkCublasErrors(cublas_status);
                    for (int n=0; n<fuse_factor; n++)  {
                        if (i + n < seq_length)  {
                            checkCudaErrors(cudaEventRecord(levents[l][i+n], lstream[l]));
                        }
                    }
                }

                /** Wait previous timestep's layer GEMM. */
                checkCudaErrors(cudaStreamWaitEvent(rstream[l], levents[l][i], 0));

                /** Recurrent GEMM. */
                checkCublasErrors(cublasSetStream(cublas_handle, rstream[l]));
                cublas_status = cublasGemmEx(cublas_handle,
                                transa, transb,
                                rm, rn, rk,
                                &alpha,
                                rw[l], CUDA_R_16F, transa == CUBLAS_OP_N ? rm : rk,
                                ri[l], CUDA_R_16F, rk,
                                &beta,
                                ro[l], CUDA_R_16F, rm,
                                CUDA_R_32F,
                                ralgo);
                checkCublasErrors(cublas_status);

                /** Element-wise operation. */
                checkCudaErrors(cudaEventRecord(revents[l][i], rstream[l]));
            }

        }

        /**
         * Make sure all works done.
         * Stream capture requires same stream in, same stream out.
         */
        for (int i=0; i<num_layer; i++)  {
            checkCudaErrors(cudaStreamWaitEvent(lstream[0],
                levents[i][seq_length-1], 0));
            checkCudaErrors(cudaStreamWaitEvent(lstream[0],
                revents[i][seq_length-1], 0));
        }
    }

    if (graph_launch)  {
        checkCudaErrors(cudaStreamEndCapture(lstream[0], &graph));
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
        checkCudaErrors(cudaFree(lw[i]));
        checkCudaErrors(cudaFree(li[i]));
        checkCudaErrors(cudaFree(lo[i]));
        checkCudaErrors(cudaFree(rw[i]));
        checkCudaErrors(cudaFree(ri[i]));
        checkCudaErrors(cudaFree(ro[i]));
    }

    /** Destroy events. */
    for (int i=0; i<num_layer; i++)  {
        for (int j=0; j<seq_length; j++)  {
            checkCudaErrors(cudaEventDestroy(revents[i][j]));
            checkCudaErrors(cudaEventDestroy(levents[i][j]));
        }
    }
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    /** Destory streams. */
    for (int i=0; i<num_layer; i++)  {
        checkCudaErrors(cudaStreamDestroy(lstream[i]));
        checkCudaErrors(cudaStreamDestroy(rstream[i]));
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

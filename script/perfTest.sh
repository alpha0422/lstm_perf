#!/usr/bin/env bash

./build/lstm_gemm_v1 0 32 38 1024 1024 1 8 8
./build/lstm_gemm_v2 0 32 38 1024 1024 1
./build/lstm_gemm_v1 1 32 38 1024 1024 1 8 8 103 102
./build/lstm_gemm_v2 1 32 38 1024 1024 1 102

./build/lstm_gemm_v1 0 256 38 1024 1024 1 2 8
./build/lstm_gemm_v2 0 256 38 1024 1024 1
./build/lstm_gemm_v1 1 256 38 1024 1024 1 2 8 103 103
./build/lstm_gemm_v2 1 256 38 1024 1024 1 103

./build/lstm_gemm_v1 0 32 38 2048 1024 1 8 8
./build/lstm_gemm_v2 0 32 38 2048 1024 1
./build/lstm_gemm_v1 1 32 38 2048 1024 1 8 8 103 102
./build/lstm_gemm_v2 1 32 38 2048 1024 1 102

./build/lstm_gemm_v1 0 256 38 2048 1024 1 2 8
./build/lstm_gemm_v2 0 256 38 2048 1024 1
./build/lstm_gemm_v1 1 256 38 2048 1024 1 2 8 103 103
./build/lstm_gemm_v2 1 256 38 2048 1024 1 103

./build/lstm_gemm_v1 0 32 38 1024 1024 3 8 8
./build/lstm_gemm_v2 0 32 38 1024 1024 3
./build/lstm_gemm_v1 1 32 38 1024 1024 3 8 8 103 102
./build/lstm_gemm_v2 1 32 38 1024 1024 3 102

./build/lstm_gemm_v1 0 256 38 1024 1024 3 2 8
./build/lstm_gemm_v2 0 256 38 1024 1024 3
./build/lstm_gemm_v1 1 256 38 1024 1024 3 2 8 103 103
./build/lstm_gemm_v2 1 256 38 1024 1024 3 103

./build/lstm_gemm_v1 0 32 38 2048 1024 3 8 8
./build/lstm_gemm_v2 0 32 38 2048 1024 3
./build/lstm_gemm_v1 1 32 38 2048 1024 3 8 8 103 102
./build/lstm_gemm_v2 1 32 38 2048 1024 3 102

./build/lstm_gemm_v1 0 256 38 2048 1024 3 2 8
./build/lstm_gemm_v2 0 256 38 2048 1024 3
./build/lstm_gemm_v1 1 256 38 2048 1024 3 2 8 103 103
./build/lstm_gemm_v2 1 256 38 2048 1024 3 103


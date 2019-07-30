# lstm\_perf

LSTM GEMM performance investigation on GPU.

* lstm\_gemm\_v1: CuDNN pattern;
* lstm\_gemm\_v2: Google/Intel patter;

Examples:

```
./build/lstm_gemm_v1 1 32 38 1024 1024 3 8 8 103 102
./build/lstm_gemm_v2 1 32 38 1024 1024 3 102
```


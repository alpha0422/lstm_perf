
NVCC = nvcc

CFLAGS = -g -I/usr/local/cuda/include/
LDFLAGS = -lcuda -lcublas -L/usr/local/cuda/lib64/

default: all

all: lstm_gemm_v1 lstm_gemm_v2 

%: src/%.cu
	@mkdir -p build
	$(NVCC) -arch=sm_70 $(CFLAGS) -o build/$@ $(LDFLAGS) $^

.PHONY: clean

clean:
	@rm -rf build core.*

#!/usr/bin/env bash

H=1024
T=38
L=3

for I in 1024 2048
do
    for N in 32 256
    do
        echo "I:" ${I} "N:" ${N}

        # (4H, H+I, NT)
        cublasTest -Rgemm -b -K -Pinh -Pouth -Ps -m$((4*H)) -n$((N*T)) -k$((H+I)) -na -nb -T100 -algorithm99 | grep 'CUDA'
        # (4H, H, NT)
        cublasTest -Rgemm -b -K -Pinh -Pouth -Ps -m$((4*H)) -n$((N*T)) -k$((H)) -na -nb -T100 -algorithm99 | grep 'CUDA'
        # (4H, H+I, N)
        cublasTest -Rgemm -b -K -Pinh -Pouth -Ps -m$((4*H)) -n$((N)) -k$((H+I)) -na -nb -T100 -algorithm99 | grep 'CUDA'
        # (4H, H, N)
        cublasTest -Rgemm -b -K -Pinh -Pouth -Ps -m$((4*H)) -n$((N)) -k$((H)) -na -nb -T100 -algorithm99 | grep 'CUDA'

        # (2L, 4H, H, NT)
        cublasTest -Rgemm_batch -b -K -N$((2*L)) -Y -m$((4*H)) -n$((N*T)) -k$((H)) -T100 -na -nb -A1 -B0 -d0 -mathMode1 -Ps -Pinh -Pouth -algorithm99 | grep 'CUDA'
        # (2LT, 4H, H, N)
        cublasTest -Rgemm_batch -b -K -N$((2*L*T)) -Y -m$((4*H)) -n$((N)) -k$((H)) -T100 -na -nb -A1 -B0 -d0 -mathMode1 -Ps -Pinh -Pouth -algorithm99 | grep 'CUDA'
        # (LT, 4H, H+I, N)
        cublasTest -Rgemm_batch -b -K -N$((L*T)) -Y -m$((4*H)) -n$((N)) -k$((H+I)) -T100 -na -nb -A1 -B0 -d0 -mathMode1 -Ps -Pinh -Pouth -algorithm99 | grep 'CUDA'
        # (L, 4H, H+I, N)
        cublasTest -Rgemm_batch -b -K -N$((L)) -Y -m$((4*H)) -n$((N)) -k$((H+I)) -T100 -na -nb -A1 -B0 -d0 -mathMode1 -Ps -Pinh -Pouth -algorithm99 | grep 'CUDA'

        # LSTM GEMM 1 Layer
        cudnnTest -RRNNf -b -mode2 -rnnInputSize${I} -rnnHiddenSize${H} -rnnSeqLength${T} -rnnNumLayers1 -rnnMiniBatch${N} -rnnPersistent0 -Ph -T100 -Pmath1 -d0 | grep 'CUDA'
        # LSTM Persist 1 Layer
        cudnnTest -RRNNf -b -mode2 -rnnInputSize${I} -rnnHiddenSize${H} -rnnSeqLength${T} -rnnNumLayers1 -rnnMiniBatch${N} -rnnPersistent1 -Ph -T100 -Pmath1 -d0 | grep 'CUDA'

        # LSTM GEMM
        cudnnTest -RRNNf -b -mode2 -rnnInputSize${I} -rnnHiddenSize${H} -rnnSeqLength${T} -rnnNumLayers${L} -rnnMiniBatch${N} -rnnPersistent0 -Ph -T100 -Pmath1 -d0 | grep 'CUDA'
        # LSTM Persist
        cudnnTest -RRNNf -b -mode2 -rnnInputSize${I} -rnnHiddenSize${H} -rnnSeqLength${T} -rnnNumLayers${L} -rnnMiniBatch${N} -rnnPersistent1 -Ph -T100 -Pmath1 -d0 | grep 'CUDA'
    done
done


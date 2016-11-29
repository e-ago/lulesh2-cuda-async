#!/bin/bash

function run() {
    local A=$1
    local B=$2
    local C=$3
    local D=$4
    local NP=$5
    local COMM=$6
    local ASYNC=$7
    shift 7
    local PAR=$@
    date
    (
        echo; \
        mpirun  \
        \
        -x COMM_ENABLE_DEBUG=0 \
        -x COMM_USE_COMM=$COMM  -x COMM_USE_ASYNC=$ASYNC   -x COMM_USE_GPU_COMM=0 -x COMM_USE_GDRDMA \
        \
        -x MP_ENABLE_DEBUG=0 \
        -x GDS_ENABLE_DEBUG=0 \
        -x ENABLE_DEBUG_MSG=0 \
        -x GDS_CQ_MAP_SMART=0 \
        \
        -x MLX5_DEBUG_MASK=0 \
        -x MLX5_FREEZE_ON_ERROR_CQE=0 \
        \
        -x MP_DBREC_ON_GPU=0 \
        -x MP_RX_CQ_ON_GPU=0 \
        -x MP_TX_CQ_ON_GPU=0 \
        \
        -x MP_EVENT_ASYNC=0 \
        -x MP_GUARD_PROGRESS=1 \
    	-x MP_ENABLE_WARN \
    	-x MP_DBREC_ON_GPU=0 \
        \
        -x GDS_DISABLE_WRITE64=1           \
        -x GDS_SIMULATE_WRITE64=$A         \
        -x GDS_DISABLE_INLINECOPY=$B       \
        -x GDS_ENABLE_WEAK_CONSISTENCY=$C \
        -x GDS_DISABLE_MEMBAR=$D           \
        -x CUDA_VISIBLE_DEVICES=0 \
        -x CUDA_DISABLE_UNIFIED_MEMORY=1 \
	--mca btl_openib_want_cuda_gdr 1 --map-by node  -np $NP -mca btl_openib_warn_default_gid_prefix 0 $PREFIX/src/scripts/wrapper.sh $PREFIX/src/lulesh-cuda-async/src/lulesh $PAR ) 2>&1 | tee -a run.log


#nvprof -o nvprof-kernel.%q{OMPI_COMM_WORLD_RANK}.nvprof
#../scripts/wrapper.sh ./bin/CoMD-cuda-mpi $PAR ) 2>&1 | tee -a run.log
    date
}

set -x

echo "CWD=$PWD"

#run 0 0 1 1 1 0 0 -s 45 $@
run 0 0 1 1 8 0 0 -s 45 $@
run 0 0 1 1 8 1 0 -s 45 $@

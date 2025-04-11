#!/bin/bash

# matmul
# commands=(
#     "benchmark_matmul.py --libtpu_args=--xla_tpu_scoped_vmem_limit_kib=65536 --trace_matcher="jit_matmul.*" --output=/tmp/matmul_result.tsv --clear_caches --dtype=bf16 --dim"
#     "benchmark_matmul.py --libtpu_args=--xla_tpu_scoped_vmem_limit_kib=65536 --trace_matcher="jit_matmul.*" --output=/tmp/matmul_result.tsv --clear_caches --dtype=fp8_e5m2 --dim"
#     "benchmark_matmul.py --libtpu_args=--xla_tpu_scoped_vmem_limit_kib=65536 --trace_matcher="jit_matmul.*" --output=/tmp/matmul_result.tsv --clear_caches --dtype=int8 --dim"
# )

# input_args=(
#   "16384 8192 1280"
#   "16384 1024 8192"
#   "16384 8192 7168"
#   "16384 3584 8192"
#   "8192 8192 8192"
# )

# HBM
commands=(
    "benchmark_hbm.py --trace_matcher="jit_my_copy.*" --output=/tmp/hbm_result.tsv --clear_caches --dtype=float32 --num_elements"
)

input_args=($(for i in $(seq 10 31); do echo $((2**i)); done))
input_args+=($((15*10**9/4)))

for cmd in "${commands[@]}"; do
    for args in "${input_args[@]}"; do
        python $cmd $args 2>/dev/null
    done
done
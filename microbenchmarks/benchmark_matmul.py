r"""Benchmark for matrix multiplication.

Sample usage (on TPU vm):
  $ python benchmark_matmul.py \
  --dim 8192 8192 8192 \
  --libtpu_args=--xla_tpu_scoped_vmem_limit_kib=65536 \
  --trace_matcher="jit_matmul.*"
"""

import argparse
import os
import re
from typing import Any
from benchmark_utils import run_bench
import jax
import jax.numpy as jnp
import pandas as pd
import sys
from datetime import datetime


def matmul(a, b):
  return a @ b


def get_dtype(dtype: str):
  if dtype == "float32":
    return jnp.float32
  if dtype == "bf16":
    return jnp.bfloat16
  if dtype == "fp8_e5m2":
    return jnp.float8_e5m2
  if dtype == "fp8_e4m3":
    return jnp.float8_e4m3fn
  if dtype == "int8":
    return jnp.int8
  raise ValueError(f"Invalid data type: {dtype}")


def main():
  """Benchmark for matrix multiplication."""
  parser = argparse.ArgumentParser(
      description="Run matrix multiplication benchmark."
  )

  parser.add_argument(
      "--dtype",
      type=str,
      choices=["float32", "bf16", "fp8_e5m2", "fp8_e4m3", "int8"],
      default="bf16",
      help="Data type of the matrix elements.",
  )
  parser.add_argument(
      "--libtpu_args",
      type=str,
      required=False,
      help=(
          "LIBTPU_INIT_ARGS environment variable, e.g."
          " '--xla_tpu_scoped_vmem_limit_kib=65536'."
      ),
  )
  parser.add_argument(
      "--dim",
      type=int,
      required=True,
      nargs=3,
      help=(
          "Dimensions of the two matices to be multiplied. 3 integers, m, n, k"
          " must be specified, implying (m, n) * (n, k)."
      ),
  )
  parser.add_argument(
      "--num_iter",
      type=int,
      default=200,
      help="Number of times the benchmark function will be run.",
  )
  parser.add_argument(
      "--warmup_iter",
      type=int,
      default=30,
      help=(
          "Number of times the benchmark function will be run to warm up before"
          " the actual timing measurement starts."
      ),
  )
  parser.add_argument(
      "--log_dir",
      type=str,
      default="/tmp/matmul",
      help="The directory to save the profiler trace to.",
  )
  parser.add_argument(
      "--label",
      type=str,
      default="my_func",
      help=(
          "A label used to name the function to be benchmarked in the trace"
          " events."
      ),
  )
  parser.add_argument(
      "--trace_matcher",
      type=str,
      required=False,
      help=(
          "A regex-based string matcher to filter the trace events eligible for"
          " benchmarking. If a matcher is specified, the timing result will be"
          " derived from the profiler trace. Otherwise, the result will be"
          " derived from the time() wrapper."
      ),
  )
  parser.add_argument(
      "--clear_caches",
      action=argparse.BooleanOptionalAction,
      help=(
          "If set, jax.clear_caches() will be invoked every time before the"
          " benchmark function is executed, which clears all compilation and"
          " staging caches."
      ),
  )
  parser.add_argument(
      "--output",
      type=str,
      required=False,
      help=(
          "write results to an output file."
      ),
  )

  args = parser.parse_args()

  if args.libtpu_args:
    os.environ["LIBTPU_INIT_ARGS"] = args.libtpu_args

  dtype = get_dtype(args.dtype)
  m, n, k = args.dim[0], args.dim[1], args.dim[2]
  a = jax.random.normal(jax.random.key(0), (m, n)).astype(dtype)
  b = jax.random.normal(jax.random.key(0), (n, k)).astype(dtype)

  compiled = jax.jit(matmul).lower(a, b).compile()
  matcher = re.compile(args.trace_matcher) if args.trace_matcher else None
  result = run_bench(
      compiled,
      a,
      b,
      num_iter=args.num_iter,
      warmup_iter=args.warmup_iter,
      log_dir=args.log_dir,
      func_label=args.label,
      trace_matcher=matcher,
      clear_caches=args.clear_caches,
  )

  # 2 ops (multiply and add)
  compute = m * n * k * 2
  tflops = compute / result.time_median / 1e12

  print(
      f"dtype: {dtype.__name__}, matrix dimensions: ({m}, {n}, {k}), time taken"
      f" (median, ms): {result.time_median * 1e3}, TFLOPS: {tflops}"
  )

  if args.output:
    record = {
      "dtype": dtype.__name__,
      "dimensions": (m, n, k),
      "time_secs_median": result.time_median,
      "tflops_median": tflops,
    }
    if os.path.exists(args.output):
      pd.DataFrame([record]).to_csv(args.output, sep="\t", index=False, mode="a", header=False)
    else:
      pd.DataFrame([record]).to_csv(args.output, sep="\t", index=False)
    
    # To be uploaded to DB
    cmd = " ".join([v for v in sys.argv if not v.startswith("--output") and "/dev/null" not in v])
    entry = {
      "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
      "workload_id": "matmul_single_chip",
      "workload_parameters": {
          "data_type": dtype.__name__,
          "matrix_dimensions": [m, n, k],
      },
      "hardware_id": "v6e-1",
      "hardware_num_chips": 1,
      "hardware_num_nodes": 1,
      "result_success": True,
      "configs_num_iterations": args.num_iter,
      "configs_other": {
          "jax_version": "0.5.2",
          "warmup_iter": args.warmup_iter,
          "command": cmd,
      },
      "benchmarker_ldap": "chishuen",
      "metrics_type": "throughput_top_level_xla_module" if args.trace_matcher else "throughput_on_host",
      "metrics_unit": "tera_ops_per_second" if args.dtype == "int8" else "tera_flops_per_second",
      "metrics_p50": tflops,
      "metrics_other": {"total_ops": compute},
    }
    if args.libtpu_args:
      entry["configs_other"]["libtpu_init_args"] = args.libtpu_args
    if args.clear_caches:
      entry["configs_other"]["clear_caches"] = True

    dirname = os.path.dirname(args.output)
    basename = os.path.basename(args.output)
    name, ext = os.path.splitext(basename)
    new_name = f"{name}_db{ext}"
    output_db = os.path.join(dirname, new_name)
    if os.path.exists(output_db):
      pd.DataFrame([entry]).to_csv(output_db, sep="\t", index=False, mode="a", header=False)
    else:
      pd.DataFrame([entry]).to_csv(output_db, sep="\t", index=False)


if __name__ == "__main__":
  main()

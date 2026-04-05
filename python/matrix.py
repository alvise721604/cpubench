
#!/usr/bin/env python3

# Download Conda from: https://conda-forge.org/download/
# Execute with: conda Miniforge...
# logout/login
# conda create -c conda-forge -n hpc_env python=3.11 --override-channels
# conda activate hpc_env
# conda install cupy numpy "blas=*=openblas" -c conda-forge --override-channels pyopencl

# On RUSTICL platform run with: RUSTICL_ENABLE=v3d python matrix.py -B opencl -s 100

"""
matrix.py

A script for matrix multiply benchmark that can run on:
  - CPU via NumPy (backend="cpu")
  - NVIDIA GPU via CuPy/CUDA (backend="cuda")
  - Apple Silicon (M1, M2, M3, M4, M5, ...) GPU via PyTorch MPS (backend="mps")
  - Raspberry 5 V3d GPU via RUSTICL/OpenCL (backend=opencl)

Examples
--------
CPU (NumPy):
  python matrix.py --backend cpu -n 16 -s 6000 --dtype float32

NVIDIA GPU (CuPy):
  python matrix.py --backend cuda -s 8000 --dtype float16

Apple Silicon GPU (MPS):
  python matrix.py --backend mps -s 6000 --dtype float16

Note: "mps" backend in CuPy (for Apple Silicon GPU) doesn't support float64
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Literal

Backend = Literal["cpu", "cuda", "mps", "opencl"]

def parse_args():
    p = argparse.ArgumentParser(description="CPU/CUDA/MPS matrix multiply benchmark")
    p.add_argument("-B", "--backend", choices=["cpu", "cuda", "mps", "opencl"], default="cpu",
                   help="cpu=NumPy, cuda=CuPy (NVIDIA), mps=PyTorch on Apple Silicon")
    p.add_argument("-s", "--size", type=int, default=5000, help="Matrix dimension N for NxN")
    p.add_argument("-D", "--dtype", choices=["float16", "float32", "float64"], default="float32",
                   help="Data type. CPU supports float32/float64 only.")
    p.add_argument("-n", "--threads", type=int, default=0,
                   help="CPU threads (cpu backend). 0 = don't override")
    p.add_argument("-W", "--warmup", type=int, default=1, help="Warmup iterations (not timed)")
    p.add_argument("-I", "--iters", type=int, default=1, help="Timed iterations")
    p.add_argument("-d", "--device", type=int, default=0, help="CUDA device index (cuda backend)")
    p.add_argument("-K", "--kmp-duplicate-lib-ok", action="store_true",
                   help="(macOS) Unsafe workaround for OpenMP duplicate libomp error (#15).")
    p.add_argument("--cl-platform", type=int, default=0, help="OpenCL platform index (opencl backend)")
    p.add_argument("--cl-device", type=int, default=0, help="OpenCL device index (opencl backend)")
    p.add_argument("--cl-tile", type=int, default=16, help="Tile size / local workgroup edge (opencl backend)")
    p.add_argument("--cl-prefer-gpu", action="store_true", help="Prefer a GPU device if available (opencl backend)")
    p.add_argument("--cl-list", action="store_true", help="List OpenCL platforms/devices and exit")
    p.add_argument("--rusticl-enable", type=str, default="", help="Set RUSTICL_ENABLE (e.g. v3d) before OpenCL init")
    return p.parse_args()


def main():
    args = parse_args()
    n = int(args.size)
    if n <= 0:
        raise SystemExit("Error: size must be > 0")
    if int(args.warmup) < 0 or int(args.iters) <= 0:
        raise SystemExit("Error warmup must be >= 0 and iters must be > 0")

    if args.backend == "cpu":
        import cpu
        cpu.run_cpu(n=n, dtype=args.dtype, threads=int(args.threads), warmup=int(args.warmup), iters=int(args.iters))
    elif args.backend == "cuda":
        import cuda
        cuda.run_cuda(n=n, dtype=args.dtype, warmup=int(args.warmup), iters=int(args.iters), device=int(args.device))
    elif args.backend == "mps":
        if args.dtype == "float64":
            raise SystemExit("[mps] dtype 'float64' not supported. Use float32 or float64 for MPS backend")
        import mps
        mps.run_mps(n=n, dtype=args.dtype, warmup=int(args.warmup), iters=int(args.iters), kmp_duplicate_lib_ok=bool(args.kmp_duplicate_lib_ok))
    elif args.backend == "opencl":
        import opencl
        opencl.run_opencl(
            n=n,
            dtype=args.dtype,
            warmup=int(args.warmup),
            iters=int(args.iters),
            platform=int(args.cl_platform),
            device=int(args.cl_device),
            tile=int(args.cl_tile),
            rusticl_enable=(args.rusticl_enable or None),
            prefer_gpu=bool(args.cl_prefer_gpu),
        )
    else:
        raise SystemExit(f"Error: Unknown backend {args.backend}")

if __name__ == "__main__":
    main()

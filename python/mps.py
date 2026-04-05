import stats
import time
import os

#______________________________________________________________________________
def _dtype_torch(name: str):
    import torch
    return {"float16": torch.float16, "float32": torch.float32, "float64": torch.float64}.get(name)

#______________________________________________________________________________
def run_mps(n: int, dtype: str, warmup: int, iters: int, kmp_duplicate_lib_ok: bool):
    # If you see: "OMP: Error #15 ... libomp.dylib already initialized",
    # it usually means *two different* libomp.dylib copies are being loaded (e.g. Homebrew + Conda + PyTorch).
    # The proper fix is to ensure only one OpenMP runtime is on your DYLD search path.
    # As an *unsafe workaround*, you can enable this flag to let the process continue.
    if kmp_duplicate_lib_ok:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    try:
        import torch
    except Exception as e:
        raise SystemExit(
            "[mps] PyTorch not available. Install:\n"
            "  python -m pip install torch\n"
            f"Original error: {e}"
        )

    if not torch.backends.mps.is_available():
        raise SystemExit("[mps] MPS backend not available. Requires Apple Silicon + compatible macOS/PyTorch.")

    dt = _dtype_torch(dtype)
    if dt is None:
        raise SystemExit(f"[mps] dtype '{dtype}' not supported. Use float16/float32 for MPS backend.")

    device = torch.device("mps")

    print(f"--> Backend: Apple Silicon GPU / PyTorch MPS")
    print(f"--> Matrix size: {n} x {n}  dtype={dtype}")

    t0 = time.perf_counter()
    A = torch.rand((n, n), device=device, dtype=dt)
    B = torch.rand((n, n), device=device, dtype=dt)
    torch.mps.synchronize()
    t1 = time.perf_counter()
    print(f"--> {t1-t0:.3f}s for matrix init with random numbers")  


    t0 = time.perf_counter()
    for _ in range(max(0, warmup)):
        _ = A @ B
    torch.mps.synchronize()
    t1 = time.perf_counter()
    print(f"--> {t1-t0:.3f}s for matrix warmup")

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        C = A @ B
        torch.mps.synchronize()
        t1 = time.perf_counter()
        _ = float(C[0, 0].to("cpu"))
        times.append(t1 - t0)

    stats.print_stats(n, times)
import stats
import time
import os

#______________________________________________________________________________
def _set_cpu_threads(n: int):
    """Best effort thread control for common BLAS backends."""
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(n))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(n))

    try:
        from threadpoolctl import threadpool_limits
        try:
            threadpool_limits(limits=n)
        except Exception:
            pass
    except Exception:
        pass

#______________________________________________________________________________
def _dtype_cpu(name: str):
    import numpy as np
    return {"float32": np.float32, "float64": np.float64}.get(name)

#______________________________________________________________________________
def run_cpu(n: int, dtype: str, threads: int, warmup: int, iters: int):
    if threads > 0:
        _set_cpu_threads(threads)

    import numpy as np

    dt = _dtype_cpu(dtype)
    if dt is None:
        raise SystemExit(f"[cpu] dtype '{dtype}' not supported. Use float32 or float64 for CPU backend.")

    print(f"--> Backend: CPU / NumPy")
    print(f"--> Threads: {threads if threads > 0 else 'default'}")
    print(f"--> Matrix size: {n} x {n}  dtype={dtype}")


    t0 = time.perf_counter()
    A = np.random.random((n, n)).astype(dt, copy=False)
    B = np.random.random((n, n)).astype(dt, copy=False)
    t1 = time.perf_counter()
    print(f"--> {t1-t0:.3f}s for matrix init with random numbers")

    t0 = time.perf_counter()
    for _ in range(max(0, warmup)):
        _ = A @ B
    t1 = time.perf_counter()
    print(f"--> {t1-t0:.3f}s for matrix warmup")

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        C = A @ B
        t1 = time.perf_counter()
        _ = float(C[0, 0])
        times.append(t1 - t0)

    stats.print_stats(n, times)
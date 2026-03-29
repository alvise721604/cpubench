
import stats
import time

#______________________________________________________________________________
def _dtype_cupy(name: str):
    import cupy as cp
    return {"float16": cp.float16, "float32": cp.float32, "float64": cp.float64}.get(name)

#______________________________________________________________________________
def run_cuda(n: int, dtype: str, warmup: int, iters: int, device: int):
    try:
        import cupy as cp
    except Exception as e:
        raise SystemExit(
            "[cuda] CuPy not available. Install (typical):\n"
            "  python -m pip install cupy-cuda12x\n"
            f"Original error: {e}"
        )

    cp.cuda.Device(device).use()

    dt = _dtype_cupy(dtype)
    if dt is None:
        raise SystemExit(f"[cuda] dtype '{dtype}' not supported. Use float16/float32/float64 for CUDA backend.")

    props = cp.cuda.runtime.getDeviceProperties(device)
    raw_name = props.get("name", b"")
    name = raw_name.decode(errors="ignore") if isinstance(raw_name, (bytes, bytearray)) else str(raw_name)

    print(f"--> Backend: NVIDIA GPU / CuPy (CUDA)")
    print(f"--> Device: {device}  {name}")
    print(f"--> Matrix size: {n} x {n}  dtype={dtype}")

    t0 = time.perf_counter()
    A = cp.random.random((n, n), dtype=dt)
    B = cp.random.random((n, n), dtype=dt)
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    print(f"--> {t1-t0:.3f}s for matrix init with random numbers")

    t0 = time.perf_counter()
    for _ in range(max(0, warmup)):
        _ = A @ B
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    print(f"--> {t1-t0:.3f}s for matrix warmup")

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        C = A @ B
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        _ = float(C[0, 0].get())
        times.append(t1 - t0)

    stats.print_stats(n, times)
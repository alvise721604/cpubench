import stats 
import time

#______________________________________________________________________________
def list_opencl_devices() -> None:
    """
    Print available OpenCL platforms/devices (PyOpenCL).

    Tip for Raspberry Pi 5 (V3D / Rusticl):
      RUSTICL_ENABLE=v3d clinfo
      RUSTICL_ENABLE=v3d python matrix_universal_opencl.py --backend opencl --cl-platform 0 --cl-device 0

    If you don't see the V3D device on Pi 5, try:
      export RUSTICL_ENABLE=v3d
      export LIBGL_ALWAYS_SOFTWARE=0
    """
    try:
        import pyopencl as cl
    except Exception as e:
        raise SystemExit(
            "[opencl] PyOpenCL not available. Install:\n"
            "  python -m pip install pyopencl\n"
            "Also ensure an OpenCL ICD is installed (e.g. mesa-opencl-icd / pocl-opencl-icd) and 'clinfo' works.\n"
            f"Original error: {e}"
        )

    plats = cl.get_platforms()
    if not plats:
        raise SystemExit("[opencl] No OpenCL platforms found. (On Pi 5 you may need: RUSTICL_ENABLE=v3d)")

    print("--> OpenCL platforms/devices:")
    for pi, p in enumerate(plats):
        print(f"  [P{pi}] {p.name}  ({p.vendor})  version={p.version}")
        devs = p.get_devices()
        for di, d in enumerate(devs):
            dtype = "GPU" if d.type & cl.device_type.GPU else ("CPU" if d.type & cl.device_type.CPU else "OTHER")
            print(f"      [D{di}] {d.name}  type={dtype}  compute_units={d.max_compute_units}  "
                  f"global_mem={d.global_mem_size/1024/1024:.1f} MiB")

#______________________________________________________________________________
def run_opencl(
    n: int,
    dtype: str,
    warmup: int,
    iters: int,
    platform: int,
    device: int,
    tile: int,
    rusticl_enable: str | None,
    prefer_gpu: bool,
) -> None:
    """
    OpenCL backend using PyOpenCL.

    Notes:
      - Uses a simple tiled matmul kernel (not a vendor BLAS).
      - For Raspberry Pi 5 GPU via Mesa/Rusticl, you often must set: RUSTICL_ENABLE=v3d
    """
    if rusticl_enable:
        # Must be set before importing pyopencl in most setups
        os.environ.setdefault("RUSTICL_ENABLE", rusticl_enable)

    try:
        import numpy as np
        import pyopencl as cl
    except Exception as e:
        raise SystemExit(
            "[opencl] Missing dependency. Install:\n"
            "  python -m pip install pyopencl numpy\n"
            "Also ensure an OpenCL ICD is installed and 'clinfo' works.\n"
            f"Original error: {e}"
        )

    if dtype != "float32":
        raise SystemExit("[opencl] For now only float32 is enabled (use --dtype float32).")

    if tile <= 0 or tile > 64:
        raise SystemExit("[opencl] --cl-tile must be in a reasonable range (e.g. 8, 16, 32).")

    plats = cl.get_platforms()
    if not plats:
        raise SystemExit("[opencl] No OpenCL platforms found. (On Pi 5 you may need: RUSTICL_ENABLE=v3d)")

    if platform < 0 or platform >= len(plats):
        raise SystemExit(f"[opencl] Invalid --cl-platform {platform}. Use --cl-list to see options.")
    p = plats[platform]
    devs = p.get_devices()
    if not devs:
        raise SystemExit(f"[opencl] No devices on platform {platform}.")

    # Choose device (optionally prefer GPU)
    d = None
    if prefer_gpu:
        gpu_devs = [x for x in devs if x.type & cl.device_type.GPU]
        if gpu_devs:
            if 0 <= device < len(gpu_devs):
                d = gpu_devs[device]
            else:
                d = gpu_devs[0]
    if d is None:
        if device < 0 or device >= len(devs):
            raise SystemExit(f"[opencl] Invalid --cl-device {device}. Use --cl-list to see options.")
        d = devs[device]

    ctx = cl.Context(devices=[d])
    queue = cl.CommandQueue(ctx)

    dt = np.float32

    print(f"--> Backend: OpenCL / PyOpenCL")
    print(f"--> Platform: {p.name}  ({p.vendor})")
    print(f"--> Device:   {d.name}")
    print(f"--> Matrix size: {n} x {n}  dtype=float32")
    print(f"--> Tile: {tile} (local workgroup {tile}x{tile})")

    # Host matrices
    A_h = np.random.random((n, n)).astype(dt, copy=False)
    B_h = np.random.random((n, n)).astype(dt, copy=False)

    mf = cl.mem_flags
    A_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_h)
    B_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_h)
    C_g = cl.Buffer(ctx, mf.WRITE_ONLY, size=A_h.nbytes)

    kernel_src = f"""
    #define TS {tile}
    __kernel void matmul_tiled(const int N,
                               __global const float* A,
                               __global const float* B,
                               __global float* C)
    {{
        const int row = get_global_id(0);
        const int col = get_global_id(1);
        const int lrow = get_local_id(0);
        const int lcol = get_local_id(1);

        __local float As[TS*TS];
        __local float Bs[TS*TS];

        float sum = 0.0f;

        const int tiles = (N + TS - 1) / TS;
        for (int t = 0; t < tiles; t++) {{
            const int a_col = t*TS + lcol;
            const int b_row = t*TS + lrow;

            As[lrow*TS + lcol] = (row < N && a_col < N) ? A[row*N + a_col] : 0.0f;
            Bs[lrow*TS + lcol] = (b_row < N && col < N) ? B[b_row*N + col] : 0.0f;

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int k = 0; k < TS; k++) {{
                sum += As[lrow*TS + k] * Bs[k*TS + lcol];
            }}

            barrier(CLK_LOCAL_MEM_FENCE);
        }}

        if (row < N && col < N) {{
            C[row*N + col] = sum;
        }}
    }}
    """
    try:
        prg = cl.Program(ctx, kernel_src).build()
    except Exception as e:
        raise SystemExit(f"[opencl] Failed to build kernel. Error: {e}")

    knl = cl.Kernel(prg, "matmul_tiled")

    # Padded global size
    g0 = ((n + tile - 1) // tile) * tile
    g1 = ((n + tile - 1) // tile) * tile
    global_size = (g0, g1)
    local_size = (tile, tile)

    # Warmup
    for _ in range(max(0, warmup)):
        #prg.matmul_tiled(queue, global_size, local_size, np.int32(n), A_g, B_g, C_g)
        knl(queue, global_size, local_size, np.int32(n), A_g, B_g, C_g)
    queue.finish()

    scalar = np.empty(1, dtype=dt)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        #prg.matmul_tiled(queue, global_size, local_size, np.int32(n), A_g, B_g, C_g)
        knl(queue, global_size, local_size, np.int32(n), A_g, B_g, C_g)
        queue.finish()
        t1 = time.perf_counter()
        cl.enqueue_copy(queue, scalar, C_g, src_offset=0).wait()
        _ = float(scalar[0])
        times.append(t1 - t0)

    stats.print_stats(n, times)
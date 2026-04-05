
def print_stats(n: int, times: list[float]):
    times_sorted = sorted(times)
    t_min = times_sorted[0]
    t_med = times_sorted[len(times_sorted) // 2]
    t_avg = sum(times_sorted) / len(times_sorted)
    t_max = times_sorted[-1]

    flops = 2 * (n ** 3)
    gflops_min = (flops / t_min) / 1e9
    gflops_med = (flops / t_med) / 1e9
    gflops_avg = (flops / t_avg) / 1e9

    print(f"--> Iterations: {len(times_sorted)}")
    print(f"--> Time (s): min={t_min:.6f}  med={t_med:.6f}  avg={t_avg:.6f}  max={t_max:.6f}")
    print(f"--> Throughput (GFLOPS): min={gflops_min:.2f}  med={gflops_med:.2f}  avg={gflops_avg:.2f}")

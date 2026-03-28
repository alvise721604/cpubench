import time
import math
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor
import os

#___________________________________________________________________________________________________________________
def simpson_block(start_idx: int, end_idx: int, step: float) -> Tuple[float, float]:
    result = 0.0
    x = start_idx * step

    for _ in range(start_idx, end_idx):
        if x != 0.0:
            result += (math.sin(x) / x) * step
        else:
            result += 1.0 * step
        x += step

    return result

#___________________________________________________________________________________________________________________
def gaussian_block(start_idx: int, end_idx: int, step: float) -> Tuple[float, float]:
    result = 0.0
    x = start_idx * step

    for _ in range(start_idx, end_idx):
        result += math.exp(-(x * x)) * step
        x += step

    return result

#___________________________________________________________________________________________________________________
def make_ranges(total_iterations: int, chunk_size: int):
    ranges = []
    start = 0
    while start < total_iterations:
        end = min(start + chunk_size, total_iterations)
        ranges.append((start, end))
        start = end
    return ranges

#___________________________________________________________________________________________________________________
def simpson_multiprocess(limit: float, step: float, chunk_size: int = 200_000) -> Tuple[float, float]:
    start_time = time.time()
    iterations = int(limit / step)
    ranges = make_ranges(iterations, chunk_size)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(simpson_block, start, end, step)
            for start, end in ranges
        ]
        result = sum(f.result() for f in futures)
    duration = time.time() - start_time
    return duration, result * 2.0

#___________________________________________________________________________________________________________________
def gaussian_integral_multiprocess(limit: float, step: float, chunk_size: int = 200_000) -> Tuple[float, float]:
    start_time = time.time()
    iterations = int(limit / step)
    ranges = make_ranges(iterations, chunk_size)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(gaussian_block, start, end, step)
            for start, end in ranges
        ]
        result = sum(f.result() for f in futures)

    result *= 2.0
    duration = time.time() - start_time
    return duration,result * result
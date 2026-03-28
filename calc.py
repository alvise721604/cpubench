import time
import math
from typing import Tuple

#___________________________________________________________________________________________________________________
def calculate_point_integral_sin(x: float) -> float:
    if x != 0.0:
        return math.sin(x) / x # faster than numpy, as numpy is only optimized for calculus on vectors
    else:
        return 1.0

#___________________________________________________________________________________________________________________
def gaussian_integral( iterations: int, step: float ) -> Tuple[float, float]:
    result = 0.0
    x = 0.0
    start_time = time.time()
    for k in range(iterations):
        result += math.exp( -(x*x) ) * step
        x += step
    result *= 2.0
    result = math.pow(result, 2)
    duration = time.time() - start_time
    return duration, result

#___________________________________________________________________________________________________________________
def riemann_sinx_integral( iterations: int, step: float) -> Tuple[float, float]:
    result = 0.0
    x = 0.0
    start_time = time.time()
    for k in range(iterations):
        result += calculate_point_integral_sin(x) * step
        x += step
    result = result * 2.0
    duration = time.time() - start_time
    return duration, result
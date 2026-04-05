import math
import numpy as np
from typing import Tuple
import multiprocessing as mp

#___________________________________________________________________________________________________________________
def _leibniz_chunk_mp(args) -> float:
    start, end = args
    parziale = 0.0
    for n in range(start, end):
        segno = 1.0 if n % 2 == 0 else -1.0
        parziale += segno / (2 * n + 1)
    return parziale

#___________________________________________________________________________________________________________________
def _euler_chunk_mp(args) -> float:
    start, end = args
    sum = 0.0
    for n in range(start, end):
        sum += 1 / (n*n)
    return sum

#___________________________________________________________________________________________________________________
def pi_fabrice_bellard( iterations: int ) -> float:
    sign = 1.0
    result = 0.0
    factor = 1.0
    for n in range(iterations):
        result += sign * factor * ( 1/(10.0*n+9.0) 
                                    - 4.0/(10.0*n+7.0) 
                                    - 4.0/(10.0*n+5.0) 
                                    - 64.0/(10.0*n+3.0) 
                                    + 256.0/(10.0*n+1.0) 
                                    - 1.0/(4.0*n+3.0) 
                                    - 32.0/(4.0*n+1.0)
                                    ) 
        sign = -sign
        factor /= 1024.0
    return result / 64.0

#___________________________________________________________________________________________________________________
def pi_leibniz_multiprocessing(iterations: int, num_procs: int = 4) -> float:
    chunk_size = (iterations + num_procs - 1) // num_procs
    intervals = []
    for i in range(num_procs):
        start = i * chunk_size
        end = min(start + chunk_size, iterations)
        if start < end:
            intervals.append((start, end))

    with mp.Pool(processes=num_procs) as pool:
        risultati = pool.map(_leibniz_chunk_mp, intervals)

    return 4.0 * sum(risultati)

#___________________________________________________________________________________________________________________
def pi_euler_multiprocessing(iterations: int, num_procs: int = 4) -> float:
    if iterations < 1:
        return 0.0

    chunk_size = (iterations + num_procs - 1) // num_procs
    intervals = []

    for i in range(num_procs):
        start = 1 + i * chunk_size
        end = min(start + chunk_size, iterations + 1)  # +1 perché range esclude l'ultimo
        if start < end:
            intervals.append((start, end))

    with mp.Pool(processes=num_procs) as pool:
        results = pool.map(_euler_chunk_mp, intervals)

    total = sum(results)
    return math.sqrt(6.0 * total)

#___________________________________________________________________________________________________________________
def gaussian_integral( iterations: int, step: float ) -> float:
    result = 0.0
    x = 0.0
    for k in range(iterations):
        result += math.exp( -(x*x) ) * step
        x = (k+1)*step
    result *= 2.0
    
    return math.pow(result, 2)

#___________________________________________________________________________________________________________________
def gaussian_integral_numpy_vectorized(limit: float, step: float) -> float:
    x = np.arange(0.0, limit, step, dtype=np.float64)
    integral_half = np.exp(-(x * x)).sum() * step
    return (2.0 * integral_half) ** 2
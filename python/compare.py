import math
import time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor


def pi_leibniz_while(iterazioni: int) -> float:
    somma = 0.0
    segno = 1.0
    n = 0
    while n < iterazioni:
        somma += segno / (2 * n + 1)
        segno *= -1.0
        n += 1
    return 4.0 * somma


def _leibniz_chunk(start: int, end: int) -> float:
    parziale = 0.0
    for n in range(start, end):
        segno = 1.0 if n % 2 == 0 else -1.0
        parziale += segno / (2 * n + 1)
    return parziale


def pi_leibniz_threading(iterazioni: int, num_threads: int = 4) -> float:
    chunk_size = (iterazioni + num_threads - 1) // num_threads
    tasks = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            start = i * chunk_size
            end = min(start + chunk_size, iterazioni)
            if start < end:
                tasks.append(executor.submit(_leibniz_chunk, start, end))
    return 4.0 * sum(task.result() for task in tasks)


def _leibniz_chunk_mp(args) -> float:
    start, end = args
    parziale = 0.0
    for n in range(start, end):
        segno = 1.0 if n % 2 == 0 else -1.0
        parziale += segno / (2 * n + 1)
    return parziale


def pi_leibniz_multiprocessing(iterazioni: int, num_processi: int = 4) -> float:
    chunk_size = (iterazioni + num_processi - 1) // num_processi
    intervalli = []
    for i in range(num_processi):
        start = i * chunk_size
        end = min(start + chunk_size, iterazioni)
        if start < end:
            intervalli.append((start, end))

    with mp.Pool(processes=num_processi) as pool:
        risultati = pool.map(_leibniz_chunk_mp, intervalli)

    return 4.0 * sum(risultati)


def pi_leibniz_numpy(iterazioni: int) -> float:
    n = np.arange(iterazioni, dtype=np.float64)
    segni = np.where(n % 2 == 0, 1.0, -1.0)
    termini = segni / (2.0 * n + 1.0)
    return 4.0 * np.sum(termini)

def pi_leibniz_numpy_chunked(iterazioni: int, chunk_size: int = 1_000_000) -> float:
    somma = 0.0

    for start in range(0, iterazioni, chunk_size):
        end = min(start + chunk_size, iterazioni)
        n = np.arange(start, end, dtype=np.float64)
        termini = np.where(n % 2 == 0, 1.0, -1.0) / (2.0 * n + 1.0)
        somma += np.sum(termini)

    return 4.0 * somma

def benchmark(nome, funzione, *args, **kwargs):
    t0 = time.perf_counter()
    val = funzione(*args, **kwargs)
    dt = time.perf_counter() - t0
    err = abs(math.pi - val)
    print(f"{nome:<18} pi≈{val:.15f}  errore={err:.3e}  tempo={dt:.6f}s")


if __name__ == "__main__":
    n = 400_000_000
    
    benchmark("while", pi_leibniz_while, n)
    benchmark("threading", pi_leibniz_threading, n, 4)
    benchmark("multiprocessing", pi_leibniz_multiprocessing, n, 4)
    benchmark("numpy", pi_leibniz_numpy, n)
    benchmark("numpy chunked", pi_leibniz_numpy_chunked, n)

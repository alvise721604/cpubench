g++-15 -O3 -march=native -fopenmp -std=c++20 src/main_cli.cpp src/calc.cpp -o bench_cpu
g++-15 -O3 -march=native -fopenmp -std=c++20 src/stream.cpp -o bench_mem
g++-15 -O3 -march=native -fopenmp -std=c++20 src/stream_omp.cpp -o bench_mem_omp


#include <iostream>
#include <random>
#include <memory>
#include <algorithm> // std::fill
#ifdef _OPENMP
  #include <omp.h>
#endif

inline double& at(double* M, std::size_t N, std::size_t i, std::size_t j) {
    return M[i * N + j];
}

inline double at_const(const double* M, std::size_t N, std::size_t i, std::size_t j) {
    return M[i * N + j];
}

void multiply(const double* __restrict A,
              const double* __restrict B,
              double* __restrict C,
              std::size_t N)
{
    A = static_cast<const double*>(
        __builtin_assume_aligned(A, 64));
    B = static_cast<const double*>(
        __builtin_assume_aligned(B, 64));
    C = static_cast<double*>(
        __builtin_assume_aligned(C, 64));
    
    std::fill(C, C + N * N, 0.0);
    
    #pragma omp parallel for
    for (std::size_t i = 0; i < N; ++i) {
        double* __restrict c_row = C + i * N;  

        for (std::size_t k = 0; k < N; ++k) {
            const double aik = A[i * N + k]; 
            const double* __restrict b_row = B + k * N;

            #pragma GCC ivdep
            for (std::size_t j = 0; j < N; ++j) {
                c_row[j] += aik * b_row[j];
            }
        }
    }
}

int main() {
    const std::size_t N = 4000;

    std::unique_ptr<double[]> A(new double[N * N]);
    std::unique_ptr<double[]> B(new double[N * N]);
    std::unique_ptr<double[]> C(new double[N * N]);

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            at(A.get(), N, i, j) = dist(rng);
            at(B.get(), N, i, j) = dist(rng);
        }
    }

    std::cout << "Matrices A and B " << N << "x" << N
              << " allocated as linear buffers (" << (N * N)
              << " elements each).\n";

    double start = omp_get_wtime();
    multiply(A.get(), B.get(), C.get(), N);
    double end = omp_get_wtime();
    double seconds = end-start;
    std::cout << "Elapsed time: " << seconds << std::endl;
    std::cout << "First 2x2 elements of C = A * B:\n";
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            std::cout << at_const(C.get(), N, i, j) << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

using clock_type = std::chrono::steady_clock;

struct Result {
    std::string name;
    double seconds;
    double bytes_moved;
    double bandwidth_gbs;
};

template <typename T>
inline void do_not_optimize(const T& value) {
#if defined(__clang__) || defined(__GNUC__)
    asm volatile("" : : "g"(value) : "memory");
#else
    (void)value;
#endif
}

inline void clobber_memory() {
#if defined(__clang__) || defined(__GNUC__)
    asm volatile("" : : : "memory");
#endif
}

template <typename F>
double measure_best_seconds(F&& func, int repeats) {
    double best = 1e100;

    for (int i = 0; i < repeats; ++i) {
        clobber_memory();
        const auto t0 = clock_type::now();
        func();
        const auto t1 = clock_type::now();
        clobber_memory();

        const double dt = std::chrono::duration<double>(t1 - t0).count();
        best = std::min(best, dt);
    }

    return best;
}

Result bench_read(const std::vector<double>& a, int repeats) {
    double final_sum = 0.0;
    const std::size_t n = a.size();

    auto fn = [&]() {
        double sum = 0.0;

        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
            sum += a[static_cast<std::size_t>(i)];
        }

        do_not_optimize(sum);
        final_sum = sum;
        clobber_memory();
    };

    const double sec = measure_best_seconds(fn, repeats);
    do_not_optimize(final_sum);

    const double bytes = static_cast<double>(n) * sizeof(double);
    return {"read", sec, bytes, bytes / sec / 1e9};
}

Result bench_write(std::vector<double>& a, int repeats) {
    const std::size_t n = a.size();

    auto fn = [&]() {
        #pragma omp parallel for schedule(static)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
            a[static_cast<std::size_t>(i)] = 1.2345;
        }

        clobber_memory();
        do_not_optimize(a.data());
    };

    const double sec = measure_best_seconds(fn, repeats);
    do_not_optimize(a[n / 2]);

    const double bytes = static_cast<double>(n) * sizeof(double);
    return {"write", sec, bytes, bytes / sec / 1e9};
}

Result bench_copy(const std::vector<double>& src, std::vector<double>& dst, int repeats) {
    const std::size_t n = src.size();

    auto fn = [&]() {
        #pragma omp parallel for schedule(static)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
            const std::size_t j = static_cast<std::size_t>(i);
            dst[j] = src[j];
        }

        clobber_memory();
        do_not_optimize(dst.data());
    };

    const double sec = measure_best_seconds(fn, repeats);
    do_not_optimize(dst[n / 2]);

    const double bytes = static_cast<double>(n) * sizeof(double) * 2.0;
    return {"copy", sec, bytes, bytes / sec / 1e9};
}

Result bench_triad(const std::vector<double>& b,
                   const std::vector<double>& c,
                   std::vector<double>& a,
                   double scalar,
                   int repeats) {
    const std::size_t n = a.size();

    auto fn = [&]() {
        #pragma omp parallel for schedule(static)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
            const std::size_t j = static_cast<std::size_t>(i);
            a[j] = b[j] + scalar * c[j];
        }

        clobber_memory();
        do_not_optimize(a.data());
    };

    const double sec = measure_best_seconds(fn, repeats);
    do_not_optimize(a[n / 2]);

    const double bytes = static_cast<double>(n) * sizeof(double) * 3.0;
    return {"triad", sec, bytes, bytes / sec / 1e9};
}

int main(int argc, char* argv[]) {
    std::size_t mib = 512;
    int repeats = 5;
    int threads = omp_get_max_threads();

    if (argc > 1) {
        mib = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    }
    if (argc > 2) {
        repeats = std::atoi(argv[2]);
    }
    if (argc > 3) {
        threads = std::atoi(argv[3]);
    }

    if (threads < 1) {
        std::cerr << "Invalid thread count.\n";
        return 1;
    }

    omp_set_num_threads(threads);

    const std::size_t bytes_per_array = mib * 1024ULL * 1024ULL;
    const std::size_t n = bytes_per_array / sizeof(double);

    if (n == 0) {
        std::cerr << "Array size too small.\n";
        return 1;
    }

    std::cout << "OpenMP threads: " << threads << "\n";
    std::cout << "Allocating " << mib << " MiB per array\n";
    std::cout << "Elements per array: " << n << "\n";
    std::cout << "Repeats: " << repeats << "\n\n";

    std::vector<double> a(n, 1.0);
    std::vector<double> b(n, 2.0);
    std::vector<double> c(n, 3.0);

    #pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        const std::size_t j = static_cast<std::size_t>(i);
        a[j] = b[j] + 0.5 * c[j];
    }

    clobber_memory();

    const auto r_read  = bench_read(a, repeats);
    const auto r_write = bench_write(a, repeats);
    const auto r_copy  = bench_copy(b, a, repeats);
    const auto r_triad = bench_triad(b, c, a, 3.0, repeats);

    const std::vector<Result> results = {r_read, r_write, r_copy, r_triad};

    std::cout << std::fixed;
    std::cout << std::left
              << std::setw(10) << "Test"
              << std::setw(12) << "Time(s)"
              << std::setw(18) << "Bytes moved"
              << std::setw(12) << "GB/s"
              << "\n";

    for (const auto& r : results) {
        std::cout << std::left
                  << std::setw(10) << r.name
                  << std::setw(12) << std::setprecision(6) << r.seconds
                  << std::setw(18) << std::setprecision(0) << r.bytes_moved
                  << std::setw(12) << std::setprecision(3) << r.bandwidth_gbs
                  << "\n";
    }

    return 0;
}

#include "calc.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <stdexcept>
#include <vector>


namespace {

    unsigned int sanitize_thread_count(unsigned int num_threads) {
        return num_threads == 0 ? 1u : num_threads;
    }

    template <typename Function>
    double parallel_sum(std::size_t begin, std::size_t end, unsigned int num_threads, Function fn) {
        num_threads = sanitize_thread_count(num_threads);
        const std::size_t total = end - begin;
        const std::size_t chunk_size = (total + num_threads - 1) / num_threads;

        std::vector<std::future<double>> futures;
        futures.reserve(num_threads);

        for (unsigned int i = 0; i < num_threads; ++i) {
            const std::size_t chunk_begin = begin + static_cast<std::size_t>(i) * chunk_size;
            const std::size_t chunk_end = std::min(chunk_begin + chunk_size, end);
            if (chunk_begin >= chunk_end) {
                break;
            }

            futures.push_back(std::async(std::launch::async, [chunk_begin, chunk_end, &fn]() {
                return fn(chunk_begin, chunk_end);
            }));
        }

        double total_sum = 0.0;
        for (auto &future : futures) {
            total_sum += future.get();
        }
        return total_sum;
    }

} // namespace

namespace calc {

    double pi_fabrice_bellard(int iterations) {
        if (iterations < 0) {
            throw std::invalid_argument("iterations deve essere >= 0");
        }

        double sign = 1.0;
        double result = 0.0;
        double factor = 1.0;

        for (int n = 0; n < iterations; ++n) {
            result += sign * factor * (
                1.0 / (10.0 * n + 9.0)
                - 4.0 / (10.0 * n + 7.0)
                - 4.0 / (10.0 * n + 5.0)
                - 64.0 / (10.0 * n + 3.0)
                + 256.0 / (10.0 * n + 1.0)
                - 1.0 / (4.0 * n + 3.0)
                - 32.0 / (4.0 * n + 1.0)
            );
            sign = -sign;
            factor /= 1024.0;
        }

        return result / 64.0;
    }

    double pi_leibniz_parallel(std::size_t iterations, unsigned int num_threads) {
        const double sum = parallel_sum(0, iterations, num_threads, [](std::size_t start, std::size_t end) {
            double partial = 0.0;
            for (std::size_t n = start; n < end; ++n) {
                const double sign = (n % 2 == 0) ? 1.0 : -1.0;
                partial += sign / (2.0 * static_cast<double>(n) + 1.0);
            }
            return partial;
        });

        return 4.0 * sum;
    }

    double pi_euler_parallel(std::size_t iterations, unsigned int num_threads) {
        const double sum = parallel_sum(1, iterations + 1, num_threads, [](std::size_t start, std::size_t end) {
            double partial = 0.0;
            for (std::size_t n = start; n < end; ++n) {
                const double d = static_cast<double>(n);
                partial += 1.0 / (d * d);
            }
            return partial;
        });

        return std::sqrt(6.0 * sum);
    }

    double gaussian_integral(std::size_t iterations, double step) {
        double result = 0.0;
        double x = 0.0;

        for (std::size_t k = 0; k < iterations; ++k) {
            result += std::exp(-(x * x)) * step;
            x = static_cast<double>(k + 1) * step;
        }

        result *= 2.0;
        return result * result;
    }

    double gaussian_integral_parallel(double limit, double step, unsigned int num_threads) {
        const std::size_t iterations = static_cast<std::size_t>(limit / step);
        const double half_integral = parallel_sum(0, iterations, num_threads, [step](std::size_t start, std::size_t end) {
            double partial = 0.0;
            for (std::size_t k = start; k < end; ++k) {
                const double x = static_cast<double>(k) * step;
                partial += std::exp(-(x * x)) * step;
            }
            return partial;
        });

        const double full_integral = 2.0 * half_integral;
        return full_integral * full_integral;
    }

} // namespace calc

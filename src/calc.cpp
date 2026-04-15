#include "calc.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace calc {

    /*
        Single-CPU LEIBNIZ
    */
    long double pi_leibniz(const std::size_t iterations) {
        double sum = 0.0;

        for (std::size_t n = 0; n < iterations; ++n) {
            //long double sign = (n % 2 == 0) ? 1.0 : -1.0;
            long double sign = (n & 1) ? -1.0 : 1.0; // bitwise è più veloce
            sum += sign / (2.0L * static_cast<double>(n) + 1.0L);
        }
        return 4.0L * sum;
    }

    /*
        OMP LEIBNIZ
    */
    long double pi_leibniz_omp( const  std::size_t iterations) {
        long double sum = 0.0;
        #pragma omp parallel for simd reduction(+:sum)
        for (std::size_t n = 0; n < iterations; ++n) {
            //long double sign = (n % 2 == 0) ? 1.0 : -1.0;
            long double sign = (n & 1) ? -1.0 : 1.0; // bitwise è più veloce
            sum += sign / (2.0L * static_cast<long double>(n) + 1.0L);
        }
        return 4.0L * sum;
    }

    /*
        Single-CPU EULER
    */
    long double pi_euler( const std::size_t iterations) {
        long double sum = 0.0;
        for (unsigned long long n = 1; n < iterations; ++n) {
            const long double d = static_cast<long double>(n);
            sum += 1.0L / (d * d);
        }
        return std::sqrt(6.0L * sum);
    }

    /*
        OMP EULER
    */
    long double pi_euler_omp( const std::size_t iterations) {
        long double sum = 0.0;

        #pragma omp parallel for simd reduction(+:sum)
        for (unsigned long long n = 1; n < iterations; ++n) {
            const long double d = static_cast<long double>(n);
            sum += 1.0L / (d * d);
        }
        return std::sqrt(6.0L * sum);
    }

    /*
    	GAUSSIAN
    */
    long double gaussian_integral( const std::size_t iterations, const long double step) {
        long double result = 0.0;
        long double x = 0.0;
        for (std::size_t k = 0; k < iterations; ++k) {
            result += std::exp(-(x * x)) * step;
            x = static_cast<long double>(k + 1) * step;
        }
        result *= 2.0L;
        return result * result;
    }

    /*
    	OMP GAUSSIAN
    */
    long double gaussian_integral_omp( const std::size_t iterations, const long double step) {
        long double result = 0.0L;
        long double x = 0.0L;
        #pragma omp parallel for simd reduction(+:result)
        for (unsigned long long k = 0; k < iterations; ++k) {
            result += std::exp(-(x * x)) * step;
            x = static_cast<long double>(k + 1) * step;
        }
        result *= 2.0L;
        return result * result;
    }

    /*
      Wallis
    */
    long double pi_wallis( const std::size_t iterations ) {
        long double prod = 1.0L;
        for (unsigned long long k = 1; k < iterations; ++k) {
            prod *= 4*k*k / (4*k*k-1);
        } 

        return 2.0L * prod;
    }

    /*
      Wallis OMP
    */
    long double pi_wallis_omp( const std::size_t iterations ) {
        long double prod = 1.0L;
        #pragma omp parallel for simd reduction(+:prod)
        for (unsigned long long k = 1; k < iterations; ++k) {
            prod *= 4*k*k / (4*k*k-1);
        } 

        return 2.0L * prod;
    }

} // namespace calc

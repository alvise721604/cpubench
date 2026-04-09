#include "calc.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace calc {

    /*
        FABRICE BELLARD
     */
    double pi_fabrice_bellard(int iterations) {

        //double sign = 1.0;
        double result = 0.0;
        double factor = 1.0;

        for (int n = 0; n < iterations-1; n+=2) {
            result += factor * (
                1.0 / (10.0 * n + 9.0)
                - 4.0 / (10.0 * n + 7.0)
                - 4.0 / (10.0 * n + 5.0)
                - 64.0 / (10.0 * n + 3.0)
                + 256.0 / (10.0 * n + 1.0)
                - 1.0 / (4.0 * n + 3.0)
                - 32.0 / (4.0 * n + 1.0)
            );
            factor /= 1048576.0;
        }
        for (int n = 1; n < iterations-1; n+=2) {
            result += -factor * (
                1.0 / (10.0 * n + 9.0)
                - 4.0 / (10.0 * n + 7.0)
                - 4.0 / (10.0 * n + 5.0)
                - 64.0 / (10.0 * n + 3.0)
                + 256.0 / (10.0 * n + 1.0)
                - 1.0 / (4.0 * n + 3.0)
                - 32.0 / (4.0 * n + 1.0)
            );
            factor /= 1048576.0;
        }

        return result / 64.0;
    }

    /*
        Single-CPU LEIBNIZ
    */
    double pi_leibniz(std::size_t iterations) {
        double sum = 0.0;
        for (std::size_t n = 0; n < iterations; n+=2) {
            sum += 1.0 / (2.0 * static_cast<double>(n) + 1.0);
        }
        for (std::size_t n = 1; n < iterations; n+=2) {
            sum += (-1.0) / (2.0 * static_cast<double>(n) + 1.0);
        }
        return 4.0 * sum;
    }

    /*
        OMP LEIBNIZ
    */
    double pi_leibniz_omp(std::size_t iterations) {
        double sum = 0.0;
        #pragma omp parallel for simd reduction(+:sum)
        for (unsigned long long n = 0; n < iterations; n+=2) {
            sum += 1.0 / (2.0 * static_cast<double>(n) + 1.0);
        }
        #pragma omp parallel for simd reduction(+:sum)
        for (unsigned long long n = 1; n < iterations; n+=2) {
            sum += (-1.0) / (2.0 * static_cast<double>(n) + 1.0);
        }
        return 4.0 * sum;
    }

    /*
        Single-CPU EULER
    */
    double pi_euler(std::size_t iterations) {
        double sum = 0.0;
        for (unsigned long long n = 1; n < iterations; ++n) {
            const double d = static_cast<double>(n);
            sum += 1.0 / (d * d);
        }
        return std::sqrt(6.0 * sum);
    }

    /*
        OMP EULER
    */
    double pi_euler_omp(std::size_t iterations) {
        double sum = 0.0;

        #pragma omp parallel for simd reduction(+:sum)
        
        for (unsigned long long n = 1; n < iterations; ++n) {
            const double d = static_cast<double>(n);
            sum += 1.0 / (d * d);
        }
        return std::sqrt(6.0 * sum);
    }

    /*
    	GAUSSIAN
    */
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

    /*
    	OMP GAUSSIAN
    */
    double gaussian_integral_omp(std::size_t iterations, double step) {
        double result = 0.0;
        
        #pragma omp parallel for simd reduction(+:result)
        for (unsigned long long k = 0; k < iterations; ++k) {
            double x = static_cast<double>(k + 1) * step;
            result += std::exp(-(x * x)) * step;
        }
        result *= 2.0;
        return result * result;
    }

    /*
      Wallis
    */
    double wallis( std::size_t iterations ) {
        double sum = 0.0;
        for (unsigned long long k = 1; k < iterations; ++k) {
            sum *= 4*k*k / (4*k*k-1);
        } 

        return 2.0*sum;
    }

    /*
      Wallis OMP
    */
    double wallis_omp( std::size_t iterations ) {
        double sum = 0.0;
        #pragma omp parallel for simd reduction(+:sum)
        for (unsigned long long k = 1; k < iterations; ++k) {
            sum *= 4*k*k / (4*k*k-1);
        } 

        return 2.0*sum;
    }

} // namespace calc

#include "calc.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace calc {

    /*
        Single-CPU NEPER
    */
    long double e_neper( const int iterations ) {
        long double _sum = 0.0L;
        for ( std::size_t n = 0; n < iterations; ++n) {
            long double sum = 0.0L;
            unsigned long int factorial = 1UL;
            for ( std::size_t m = 0; m < 1000; ++m) {
                sum += 1.0L / factorial;
                factorial *= (m+1);
            }
            _sum = sum;
        }
        return _sum;
    }

    /*
        Single-CPU LEIBNIZ
    */
    long double pi_leibniz(const std::size_t iterations) {
        long double sum = 0.0L;
        for ( std::size_t n = 0; n < iterations; ++n ) {
            const long double sign = (n & 1) ? -1.0L : 1.0L; // bitwise è più veloce
            sum += sign / (2.0L * n + 1.0L);
        }
        return 4.0L * sum;


    }

    /*
        OMP LEIBNIZ
    */
    long double pi_leibniz_omp( const  std::size_t iterations) {
        long double sum = 0.0L;
        #pragma omp parallel for simd reduction(+:sum)
        for ( std::size_t n = 0; n < iterations; ++n ) {
            //long double sign = (n % 2 == 0) ? 1.0 : -1.0;
            long double sign = (n & 1) ? -1.0 : 1.0; // bitwise è più veloce
            sum += sign / (2.0L * n + 1.0L);
        }
        return 4.0L * sum;
    }

    /*
        Single-CPU EULER
    */
    long double pi_euler( const std::size_t iterations) {
        long double sum = 0.0L;
        for ( std::size_t n = 1; n < iterations; ++n ) {
            sum += 1.0L / ( n*n );
        }
        return std::sqrt(6.0L * sum);
    }

    /*
        OMP EULER
    */
    long double pi_euler_omp( const std::size_t iterations) {
        long double sum = 0.0L;
        #pragma omp parallel for simd reduction(+:sum)
        for ( std::size_t n = 1; n < iterations; ++n ) {
            sum += 1.0L / ( n*n );
        }
        return std::sqrt(6.0L * sum);
    }

    /*
    	GAUSSIAN
    */
    long double gauss_integral( const std::size_t iterations, const long double step) {
        long double result = 0.0L;
        long double x = 0.0L;
        for ( std::size_t k = 0; k < iterations; ++k ) {
            result += std::exp(-(x * x)) * step;
            x = k * step;
        }
        result *= 2.0L;
        return result*result;
    }

    /*
    	OMP GAUSSIAN
    */
    long double gauss_integral_omp( const std::size_t iterations, const long double step) {
        long double result = 0.0L;
        long double x = 0.0L;
        #pragma omp parallel for simd reduction(+:result)
        for ( std::size_t k = 0; k < iterations; ++k ) {
            long double x = k * step;
            result += std::exp(-(x*x)) * step;
        }
        result *= 2.0L;
        return result*result;
    }

    /*
      Wallis
    */
    long double pi_wallis( const std::size_t iterations ) {
        long double prod = 1.0L;
        for ( std::size_t k = 1; k < iterations; ++k ) {
            //long double fk = static_cast<long double>(k);
            prod *= 4.0L * k * k / (4.0L * k * k - 1.0L);
        } 

        return 2.0L * prod;
    }

    /*
      Wallis OMP
    */
    long double pi_wallis_omp( const std::size_t iterations ) {
        long double prod = 1.0L;
        #pragma omp parallel for simd reduction(*:prod)
        //#pragma omp parallel for reduction(*:prod)
        for ( std::size_t k = 1; k < iterations; ++k ) {
            //long double fk = static_cast<long double>(k);
            prod *= 4.0L * k * k / (4.0L * k * k - 1.0L);
        }
        return 2.0L * prod;
    }

} // namespace calc

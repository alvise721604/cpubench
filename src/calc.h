#pragma once

#include <cstddef>
#include <string>

namespace calc {

long double pi_wallis( const std::size_t iterations );
long double pi_wallis_omp( const std::size_t iterations );

long double pi_leibniz(const std::size_t iterations);
long double pi_leibniz_omp(const std::size_t iterations);

long double pi_euler(const std::size_t iterations);
long double pi_euler_omp(const std::size_t iterations);

long double gaussian_integral(const std::size_t iterations, const long double step);
long double gaussian_integral_omp(const std::size_t iterations, const long double step);



} // namespace calc

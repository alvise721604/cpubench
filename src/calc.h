#pragma once

#include <cstddef>
#include <string>

namespace calc {

double pi_fabrice_bellard(int iterations);
double pi_leibniz_omp(std::size_t iterations);
double pi_leibniz(std::size_t iterations);
double pi_leibniz_omp(std::size_t iterations);
double pi_euler_omp(std::size_t iterations);
double pi_euler(std::size_t iterations);
double gaussian_integral(std::size_t iterations, double step);
double gaussian_integral_omp(std::size_t iterations, double step);

} // namespace calc

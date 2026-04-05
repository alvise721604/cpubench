#pragma once

#include <cstddef>
#include <string>

namespace calc {

double pi_fabrice_bellard(int iterations);
double pi_leibniz_parallel(std::size_t iterations, unsigned int num_threads);
double pi_euler_parallel(std::size_t iterations, unsigned int num_threads);
double gaussian_integral(std::size_t iterations, double step);
double gaussian_integral_parallel(double limit, double step, unsigned int num_threads);

} // namespace calc

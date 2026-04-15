#include "calc.h"

#include <cstdlib>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using clock_type = std::chrono::steady_clock;

constexpr std::size_t LEIBNIZ_ITERATIONS = 8000'000'000ULL;
constexpr std::size_t EULER_ITERATIONS   = 8000'000'000ULL;
constexpr double      RIEMANN_GAUSS_LIMIT = 1000.0;
constexpr long double      RIEMANN_STEP        = 1e-07L;
//constexpr int         FBELLARD_ITERATIONS = 100;

void print_usage(const char* progname) {
    std::cerr
        << "Uso:\n"
        << "  " << progname << " --algo <algorithm> [--multicore] [--iter <iterations>>]\n"

        << "Algorithm:\n"
        << "  leibniz | euler | wallis | gaussian\n";
}

std::string to_lower(std::string s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return s;
}

struct Options {
    std::string algo;
    bool multicore = false;
    std::string iter;
};

Options parse_args(int argc, char* argv[]) {
    Options opt;

    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];

        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("Manca il valore per ") + name);
            }
            return to_lower(argv[++i]);
        };

        if (key == "--algo") {
            opt.algo = require_value("--algo");
        } else if (key == "--iter") {
            opt.iter = require_value("--iter");
        } else if (key == "--multicore") {
            opt.multicore = true;
        } else if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("Argomento sconosciuto: " + key);
        }
    }

    if (opt.algo.empty()) {
        throw std::invalid_argument("Devi specificare --algo");
    }

    if (opt.algo != "leibniz" &&
        opt.algo != "euler" &&
        opt.algo != "wallis" &&
        opt.algo != "gaussian") {
        throw std::invalid_argument("Valore non valido per --algo: " + opt.algo);
    }

    return opt;
}

std::size_t parse_size_t_or_throw(const std::string& s, const char* name) {
    if (s.empty()) {
        throw std::invalid_argument(std::string("Valore vuoto per ") + name);
    }

    std::size_t pos = 0;
    unsigned long long value = 0;

    try {
        value = std::stoull(s, &pos, 10);
    } catch (...) {
        throw std::invalid_argument(std::string("Valore non valido per ") + name + ": " + s);
    }

    if (pos != s.size()) {
        throw std::invalid_argument(std::string("Valore non valido per ") + name + ": " + s);
    }

    return static_cast<std::size_t>(value);
}

int parse_int_or_throw(const std::string& s, const char* name) {
    if (s.empty()) {
        throw std::invalid_argument(std::string("Valore vuoto per ") + name);
    }

    std::size_t pos = 0;
    long value = 0;

    try {
        value = std::stol(s, &pos, 10);
    } catch (...) {
        throw std::invalid_argument(std::string("Valore non valido per ") + name + ": " + s);
    }

    if (pos != s.size() || value < 0) {
        throw std::invalid_argument(std::string("Valore non valido per ") + name + ": " + s);
    }

    return static_cast<int>(value);
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const Options opt = parse_args(argc, argv);

        double result = 0.0;

        const auto start_time = clock_type::now();

        if (opt.algo == "leibniz") {
            const std::size_t iterations =
                opt.iter.empty() ? LEIBNIZ_ITERATIONS : parse_size_t_or_throw(opt.iter, "--iter");

            result = opt.multicore
                ? calc::pi_leibniz_omp(iterations)
                : calc::pi_leibniz(iterations);

        } else if (opt.algo == "euler") {
            const std::size_t iterations =
                opt.iter.empty() ? EULER_ITERATIONS : parse_size_t_or_throw(opt.iter, "--iter");

            result = opt.multicore
                ? calc::pi_euler_omp(iterations)
                : calc::pi_euler(iterations);

        } else if (opt.algo == "wallis") {
            const std::size_t iterations =
                opt.iter.empty() ? EULER_ITERATIONS : parse_size_t_or_throw(opt.iter, "--iter");

            result = opt.multicore
                ? calc::pi_wallis_omp(iterations)
                : calc::pi_wallis(iterations);

        } else if (opt.algo == "gaussian") {
            if (!opt.iter.empty()) {
                std::cerr << "Nota: --iter viene ignorato per gaussian.\n";
            }

            const auto iterations = static_cast<std::size_t>(RIEMANN_GAUSS_LIMIT / RIEMANN_STEP);

            result = opt.multicore
                ? calc::gaussian_integral_omp(iterations, RIEMANN_STEP)
                : calc::gaussian_integral(iterations, RIEMANN_STEP);

        } else {
            throw std::invalid_argument("Algoritmo non supportato: " + opt.algo);
        }

        std::cout << "Test: cpu\n";
        std::cout << "Algorithm: " << opt.algo << "\n";
        std::cout << "Multicore: " << (opt.multicore ? "yes" : "no") << "\n";
        if (!opt.iter.empty()) {
            std::cout << "Iterations override: " << opt.iter << "\n";
        }
        std::cout << "Result: " << std::fixed << std::setprecision(8) << result << "\n";

        const auto end_time = clock_type::now();
        const double duration = std::chrono::duration<double>(end_time - start_time).count();

        if (duration < 1.0) {
            std::cout << "Duration [s]: " << std::fixed << std::setprecision(6) << duration << "\n";
        } else {
            std::cout << "Duration [s]: " << std::fixed << std::setprecision(3) << duration << "\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Errore: " << e.what() << "\n\n";
        print_usage(argv[0]);
        return 1;
    } catch (...) {
        std::cerr << "Errore sconosciuto\n";
        return 2;
    }
}

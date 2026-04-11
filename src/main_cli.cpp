#include "calc.h"
#include "mem.h"
#include "memutil.h"

#include <cstdlib>
#include <chrono>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

constexpr std::size_t LEIBNIZ_ITERATIONS_PARAL = 8000'000'000ULL;
constexpr std::size_t LEIBNIZ_ITERATIONS = 8000'000'000ULL;
constexpr std::size_t EULER_ITERATIONS = 8000'000'000ULL;
constexpr std::size_t EULER_ITERATIONS_PARAL = 8000'000'000ULL;
constexpr double RIEMANN_GAUSS_LIMIT = 1000.0;
constexpr double RIEMANN_STEP = 1e-07;
constexpr int FBELLARD_ITERATIONS = 100'000'000;
constexpr int WALLIS_ITERATIONS = 100'000'000;
constexpr int WALLIS_ITERATIONS_PARAL = 1000'000'000;
constexpr unsigned int MEMTEST_ITERATIONS = 500U;

double seconds_between(clock_type::time_point a, clock_type::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

void print_usage(const char* progname) {
    std::cerr
        << "Uso:\n"
        << "  " << progname << " --test cpu --algo leibniz --engine single\n"
        << "  " << progname << " --test mem --engine multi\n\n"
        << "Argomenti:\n"
        << "  --test    cpu | mem\n"
        << "  --algo    leibniz | euler | bellard | gaussian | wallis   (solo per --test cpu)\n"
        << "  --engine  single | multi\n";
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
    std::string test;
    std::string algo;
    std::string engine;
    std::string memiter;
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

        if (key == "--test") {
            opt.test = require_value("--test");
        } else if (key == "--algo") {
            opt.algo = require_value("--algo");
        } else if (key == "--engine") {
            opt.engine = require_value("--engine");
        } else if (key == "--memiter") {
            opt.memiter = require_value("--memiter");
        } else if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("Argomento sconosciuto: " + key);
        }
    }

    if (opt.test.empty()) {
        throw std::invalid_argument("Devi specificare --test cpu oppure --test mem");
    }
    if (opt.engine.empty()) {
        throw std::invalid_argument("Devi specificare --engine single oppure --engine multi");
    }

    if (opt.test != "cpu" && opt.test != "mem") {
        throw std::invalid_argument("Valore non valido per --test: " + opt.test);
    }
    if (opt.engine != "single" && opt.engine != "multi") {
        throw std::invalid_argument("Valore non valido per --engine: " + opt.engine);
    }

    if ( opt.memiter.empty() ) 
        opt.memiter = MEMTEST_ITERATIONS;
    
    if (opt.test == "cpu") {
        if (opt.algo.empty()) {
            throw std::invalid_argument("Per --test cpu devi specificare anche --algo");
        }
        if (opt.algo != "leibniz" &&
            opt.algo != "euler" &&
            opt.algo != "bellard" &&
            opt.algo != "gaussian" &&
            opt.algo != "wallis") {
            throw std::invalid_argument("Valore non valido per --algo: " + opt.algo);
        }
    } else {
        opt.algo.clear();
    }

    return opt;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const Options opt = parse_args(argc, argv);

        double result = 0.0;
        const bool omp = (opt.engine == "multi");

        const auto start_time = clock_type::now();

        if (opt.test == "mem") {
            const std::size_t buffer_size = memutil::get_installed_ram_bytes() * 0.2;
            if ( buffer_size == 0 ) {
                std::cerr << "Error: cannot determine system RAM. Stop!";
                std::exit(1);
            }
            //const std::size_t buffer_size = 5ull * 1024ull * 1024ull * 1024ull;
            std::vector<char> buf(buffer_size);

            mem::mem_test_init(buf);
            const auto t0 = clock_type::now();

            if (omp) {
                mem::mem_test_write_omp(buf, std::atol(opt.memiter.c_str()) );
            } else {
                mem::mem_test_write(buf, std::atol(opt.memiter.c_str()) );
            }

            const auto t1 = clock_type::now();
            const double write_seconds = seconds_between(t0, t1);
            const double total_written = static_cast<double>(buffer_size) * MEMTEST_ITERATIONS;
            result = total_written / write_seconds / 1e9;

            std::cout << "Test: mem\n";
            std::cout << "Engine: " << opt.engine << "\n";
            std::cout << "Write bandwidth [GB/s]: " << std::fixed << std::setprecision(8) << result << "\n";
        } else {
            if (opt.algo == "leibniz") {
                result = omp ? calc::pi_leibniz_omp(LEIBNIZ_ITERATIONS_PARAL)
                             : calc::pi_leibniz(LEIBNIZ_ITERATIONS);
            } else if (opt.algo == "euler") {
                result = omp ? calc::pi_euler_omp(EULER_ITERATIONS_PARAL)
                             : calc::pi_euler(EULER_ITERATIONS);
            } else if (opt.algo == "bellard") {
                if (omp) {
                    std::cerr << "Nota: bellard usa comunque l'implementazione non-OMP disponibile nel progetto.\n";
                }
                result = calc::pi_fabrice_bellard(FBELLARD_ITERATIONS);
            } else if (opt.algo == "gaussian") {
                const auto iterations = static_cast<std::size_t>(RIEMANN_GAUSS_LIMIT / RIEMANN_STEP);
                result = omp ? calc::gaussian_integral_omp(iterations, RIEMANN_STEP)
                             : calc::gaussian_integral(iterations, RIEMANN_STEP);
            } else if (opt.algo == "wallis") {
                result = omp ? calc::wallis_omp(WALLIS_ITERATIONS_PARAL)
                             : calc::wallis(WALLIS_ITERATIONS);
            } else {
                throw std::invalid_argument("Algoritmo non supportato: " + opt.algo);
            }

            std::cout << "Test: cpu\n";
            std::cout << "Algorithm: " << opt.algo << "\n";
            std::cout << "Engine: " << opt.engine << "\n";
            std::cout << "Result: " << std::fixed << std::setprecision(8) << result << "\n";
        }

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

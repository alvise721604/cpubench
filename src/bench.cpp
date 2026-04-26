#include <string>
#include <algorithm>
#include <iostream>
#include <cstddef>
#include <thread>
#include <vector>
#ifdef _OPENMP
  #include <omp.h>
#endif

using namespace std;

#define DX 0.00001
#define DEFAULT_ITERATIONS 20
#define DEFAULT_SIZE 200000000
struct Options {
    int threads = 0;             
    string test = "all";    
};

void compute_pi( const unsigned long long N );
void test_flops(size_t n, size_t i, int nthreads);
void test_mips(size_t n, size_t i, int nthreads);
void test_mem_bw(size_t n, size_t i, int nthreads);
inline double f( const double x ) { return 1.0f/(x*x + 1.0f); }
inline size_t parse_size_t(const string& s, const char* what);
inline int parse_int(const string& s, const char* what);



[[noreturn]] inline void usage(const char* prog, ostream& os, int code) {
    os <<
        "Usage: " << prog << " [options]\n"
        "Options:\n"
        //"  -n, --iterations N   Number of iteration to average out 'noise' and outliers (default: 10)\n"
        //"  -s, --size N Size of array on which execute computation (default: 200000000 elements)\n"
        "  -T, --threads T      Number of threads (default: auto)\n"
        "  -t, --test {all|flops|mips|mem|PI}\n"
        "  -h, --help           Show this help\n";
    exit(code);
}

inline Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];

        auto need_value = [&](const char* name) {
            if (i + 1 >= argc) {
                cerr << "Error: missing value for " << name << "\n";
                exit(2);
            }
        };

        if (a == "-h" || a == "--help") {
            usage(argv[0], cout, 0);
        } else if (a == "-T" || a == "--threads") {
            need_value(a.c_str());
            opt.threads = parse_int(argv[++i], "threads");
        }  else if (a == "-t" || a == "--test") {
            need_value(a.c_str());
            opt.test = argv[++i];
            transform(opt.test.begin(), opt.test.end(), opt.test.begin(), ::tolower);
            if (opt.test != "all" &&
                opt.test != "flops" &&
                opt.test != "mips"  &&
                opt.test != "mem"  &&
                opt.test != "PI" )
            {
                cerr << "Error: --test must be one of {all|flops|mips|mem|PI}\n";
                exit(2);
            }
        } else {
            cerr << "Unknown argument: " << a << "\n";
            usage(argv[0], cerr, 2);
        }
    }
    return opt;
}

// -----------------------------------------------------------
//        M A I N
// -----------------------------------------------------------
int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);

    vector<int> thread_set;
#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
#else
    int max_threads = max(1u, thread::hardware_concurrency());
#endif

    if (opt.threads > 0) {
        thread_set = { opt.threads };
#ifdef _OPENMP
        omp_set_num_threads(opt.threads);
#endif
    } else {
        thread_set = { 1, max_threads };
    }

    auto run_flops = [&](int th) {
        test_flops(DEFAULT_SIZE, DEFAULT_ITERATIONS, th);
    };
    auto run_mips = [&](int th) {
        test_mips(DEFAULT_SIZE, DEFAULT_ITERATIONS, th);
    };
    auto run_mem  = [&](int th) {
        test_mem_bw(DEFAULT_SIZE, DEFAULT_ITERATIONS, th);
    };
    auto run_pi = [&](int th) {
        compute_pi( 1000000000 );
    };
    
    for (int threads : thread_set) {
#ifdef _OPENMP
        if (opt.threads <= 0) omp_set_num_threads(threads);
#endif
        cout << "\n=== Using " << threads << " thread(s) ===\n";

        if (opt.test == "all" || opt.test == "flops") run_flops(threads);
        if (opt.test == "all" || opt.test == "mips")  run_mips(threads);
        if (opt.test == "all" || opt.test == "mem")   run_mem(threads);
        if (opt.test == "all" || opt.test == "PI")    run_pi(threads);
    }

    return 0;
}

//______________________________________________________________________________________________
void compute_pi( const unsigned long long N ) {
    double sum = 0.0f;
    double x   = 0.0f;
    
    // Warm-up
    for(size_t i = 0; i<1000; i++) {
        x = i * DX;
        sum += f( x ) * DX;
    }
    
    sum = 0.0f;
    double start = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum)
    for(size_t i = 0; i<N; i++) {
        x = i * DX;
        sum += f( x ) * DX;
    } 
    double end = omp_get_wtime();
    double seconds = end - start;
    cout << "---------  PI   ---------" << endl;
    cout << "  Time = " << seconds  << endl;
    cout << "  PI = " << 2.0*sum  << endl;
}

//______________________________________________________________________________________________
void test_flops(size_t n, size_t iter, int nthreads )
{
    vector<double> a(n, 1.123), 
                        b(n, 2.123), 
                        c(n, 3.123), 
                        d(n, 0.0);

    // Warm-up
    for (size_t i = 0; i < n; ++i) {
        d[i] = a[i] * b[i] + c[i];
    }

    double start = omp_get_wtime();
    #pragma omp parallel for num_threads(nthreads)
    for (size_t k = 0; k < iter; ++k) {
        #pragma omp parallel for num_threads(nthreads)
        for (size_t i = 0; i < n; ++i) {
            // 2 FLOPS per iteration: 1 mul + 1 add
            d[i] = a[i] * b[i] + c[i];
        }
    }
    double end = omp_get_wtime();
    double seconds = end-start;

    // 2 FLOPS per element * n elements * iter iterations
    double total_flops = 2.0 * static_cast<double>(n) * static_cast<double>(iter);
    double flops_per_sec = total_flops / seconds;
    cout << "--------- FLOPS ---------" << endl;
    cout << "  Time = " << seconds << endl;
    cout << "  d[0] = " << d[0]  << endl;
}

//______________________________________________________________________________________________
void test_mips(size_t n, size_t iter, int nthreads )
{
    vector<uint64_t> a(n, 1), b(n, 3);
    
    // Warm-up
    for (size_t i = 0; i < n; ++i) {
        uint64_t x = a[i];
        uint64_t y = b[i];

        // ~20 integer operations per iteration (add, mul, shift, xor, sub)

        x = x * 3 + 7;        // 2 ops (mul + add)
        x ^= (x >> 2);        // 2 ops (shift + xor)
        x = (x << 5) - x;     // 2 ops (shift + sub)
        x ^= (x >> 11);       // 2 ops
        x = x * 5 + 11;       // 2 ops

        y = y * 9 + 13;       // 2 ops
        y ^= (y << 3);        // 2 ops
        y = (y << 7) - (y >> 3); // 3 ops (2 shift + sub)
        y ^= x;               // 1 op (xor)

        a[i] = y;             // store 
    }
    
    double start = omp_get_wtime();
    #pragma omp parallel for num_threads(nthreads)
    for (size_t k = 0; k<iter; k++) {        
        #pragma omp parallel for num_threads(nthreads)
        for (size_t i = 0; i < n; ++i) {
         uint64_t x = a[i];
         uint64_t y = b[i];

          x = x * 3 + 7;        // 2 ops (mul + add)
          x ^= (x >> 2);        // 2 ops (shift + xor)
          x = (x << 5) - x;     // 2 ops (shift + sub)
          x ^= (x >> 11);       // 2 ops
          x = x * 5 + 11;       // 2 ops

          y = y * 9 + 13;       // 2 ops
         y ^= (y << 3);        // 2 ops
          y = (y << 7) - (y >> 3); // 3 ops (2 shift + sub)
          y ^= x;               // 1 op (xor)

          a[i] = y;             // store (non counted as “ALU op”)
     }
    }
    
    double end = omp_get_wtime();
    double seconds = end-start;
    double mips = (20.0 * n * iter) / seconds;

    cout << "--------- GIOPS ---------" << endl;
    cout << "  Time = " << seconds  << endl;
    cout << "  a[0] = " << a[0]  << endl;
}

//______________________________________________________________________________________________
void test_mem_bw(size_t n, size_t iter, int nthreads )
{
    vector<double> a(n, 1.1), 
                        b(n, 2.2), 
                        c(n, 0.0);

    // Warm-up
    #pragma omp parallel for num_threads(nthreads)
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }

    double start = omp_get_wtime();
    #pragma omp parallel for num_threads(nthreads)
    for (size_t k = 0; k < iter; ++k) {
       
        for (size_t i = 0; i < n; ++i) {
            c[i] = a[i] + b[i];  // 2 read + 1 write per elemento
        }
    }
    double end = omp_get_wtime();
    double seconds = end-start;

    // 3 * 8 byte per element per iteration(2 load + 1 store)
    double bytes = 3.0 * 8.0 * static_cast<double>(n) * static_cast<double>(iter);
    double gb_per_s = (bytes / seconds) / 1e9;  // GB/s

    cout << "--------- MEM BW ---------" << endl;
    cout << "  Time   = " << seconds << endl;
    cout << "  c[0]   = " << c[0] << endl;
    cout << "  MEM BW = " << gb_per_s << " GBytes/s" << endl;

}

inline size_t parse_size_t(const string& s, const char* what) {
    try {
        size_t pos = 0;
        unsigned long long v = stoull(s, &pos, 10);
        if (pos != s.size()) throw invalid_argument("trailing");
        return static_cast<size_t>(v);
    } catch (...) {
        cerr << "Error: invalid value for " << what << ": '" << s << "'\n";
        exit(2);
    }
}

inline int parse_int(const string& s, const char* what) {
    try {
        size_t pos = 0;
        long v = stol(s, &pos, 10);
        if (pos != s.size() || v < 0) throw invalid_argument("bad");
        return static_cast<int>(v);
    } catch (...) {
        cerr << "Error: invalid value for " << what << ": '" << s << "'\n";
        exit(2);
    }
}
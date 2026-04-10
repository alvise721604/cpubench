#include "mem.h"
#include <omp.h>

void mem::mem_test_init( std::vector<char> &buf ) {
    std::memset(buf.data(), 2, buf.size());
}

void mem::mem_test_write( std::vector<char> &buf, const unsigned int iterations ) {
    for (int i = 0; i < iterations; ++i) {
        std::memset(buf.data(), 0, buf.size());
    }
}

void mem::mem_test_write_omp(std::vector<char>& buf, const unsigned int iterations) {
    char* data = buf.data();
    const std::size_t n = buf.size();

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nth = omp_get_num_threads();

        const std::size_t chunk = n / nth;
        const std::size_t begin = tid * chunk;
        const std::size_t end = (tid == nth - 1) ? n : begin + chunk;
        const std::size_t len = end - begin;

        char* ptr = data + begin;

        for (unsigned int i = 0; i < iterations; ++i) {
            std::memset(ptr, 0, len);
        }
    }
}

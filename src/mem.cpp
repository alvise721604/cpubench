#include "mem.h"

void mem_test_init( std::vector<char> &buf, const std::size_t bufsize ) {
    std::memset(buf.data(), 2, buf.size());
}

void mem_test_write( std::vector<char> &buf, const unsigned int iterations ) {
    for (int i = 0; i < iterations; ++i) {
        std::memset(buf.data(), 0, buf.size());
    }
}

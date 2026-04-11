#include "mem.h"
#include <omp.h>
#include <cstring>

#if defined(_WIN32)
    #define NOMINMAX
    #include <windows.h>

    std::uint64_t get_installed_ram_bytes() {
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (GlobalMemoryStatusEx(&status)) {
            return static_cast<std::uint64_t>(status.ullTotalPhys);
        }
        return 0;
    }

#elif defined(__APPLE__)
    #include <sys/types.h>
    #include <sys/sysctl.h>

    std::uint64_t get_installed_ram_bytes() {
        std::uint64_t mem = 0;
        size_t len = sizeof(mem);
        if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) {
            return mem;
        }
        return 0;
    }

#elif defined(__linux__)
    #include <sys/sysinfo.h>

    std::uint64_t get_installed_ram_bytes() {
        struct sysinfo info {};
        if (sysinfo(&info) == 0) {
            return static_cast<std::uint64_t>(info.totalram) * info.mem_unit;
        }
        return 0;
    }

#else
    std::uint64_t get_installed_ram_bytes() {
        return 0; // non supportato
    }
#endif


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

#include "memutil.h"

#if defined(_WIN32)
    #define NOMINMAX
    #include <windows.h>

    std::uint64_t memutil::get_installed_ram_bytes() {
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

    std::uint64_t memutil::get_installed_ram_bytes() {
        std::uint64_t mem = 0;
        size_t len = sizeof(mem);
        if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) {
            return mem;
        }
        return 0;
    }

#elif defined(__linux__)
    #include <sys/sysinfo.h>

    std::uint64_t memutil::get_installed_ram_bytes() {
        struct sysinfo info {};
        if (sysinfo(&info) == 0) {
            return static_cast<std::uint64_t>(info.totalram) * info.mem_unit;
        }
        return 0;
    }

#else
    std::uint64_t memutil::get_installed_ram_bytes() {
        return 0; // non supportato
    }
#endif


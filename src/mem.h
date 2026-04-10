
#include <vector>

namespace mem {

    void mem_test_init( std::vector<char> & );
    void mem_test_write( std::vector<char> &, const unsigned int );
    void mem_test_write_omp(std::vector<char>&, const unsigned int);

}

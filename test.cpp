#include <iostream>
#include <oneapi/tbb.h>

int main() {
    std::cout << "TBB Version: " << TBB_VERSION_MAJOR << "." << TBB_VERSION_MINOR << std::endl;
    return 0;
}

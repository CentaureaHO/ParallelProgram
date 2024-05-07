#include <omp.h>
#include <iostream>
#include <chrono>

void compute() {
    // 一个假定的计算密集型任务
    double result = 0.0;
    for (int i = 0; i < 100000000; ++i) {
        result += i * i;
    }
}

int main() {
    for (int num_threads = 1; num_threads <= 16; ++num_threads) {
        omp_set_num_threads(num_threads);
        auto start = std::chrono::high_resolution_clock::now();
        
        #pragma omp parallel
        {
            compute();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Threads: " << num_threads << " Time: " << diff.count() << " s\n";
    }
    return 0;
}

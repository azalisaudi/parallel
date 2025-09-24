#include <tbb/tbb.h>
#include <iostream>
#include <vector>

int main() {
    const int N = 16;
    std::vector<int> arr(N);

    // Initialize array with values 1..16
    for (int i = 0; i < N; i++) {
        arr[i] = i + 1;
    }

    // Parallel sum using parallel_reduce
    int global_sum = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, N),  // range to process
        0,                              // identity value (initial sum)
        [&](tbb::blocked_range<int> r, int local_sum) {
            for (int i = r.begin(); i < r.end(); i++) {
                local_sum += arr[i];
            }
            return local_sum;            // return partial sum
        },
        [](int x, int y) {
            return x + y;                // combine partial sums
        }
    );

    std::cout << "Final Global Sum = " << global_sum << std::endl;
    return 0;
}

#include <stdio.h>
#include <omp.h>

#define N 16

int main() {
    int arr[N];
    int global_sum = 0;

    // Initialize array
    for (int i = 0; i < N; i++) {
        arr[i] = i + 1; // values 1..16
    }

    // Parallel region
    #pragma omp parallel for reduction(+:global_sum)
    for (int i = 0; i < N; i++) {
        global_sum += arr[i];
    }

    printf("Final Global Sum = %d\n", global_sum);
    return 0;
}

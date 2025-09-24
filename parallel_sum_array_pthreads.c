#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 4
#define N 16

int arr[N];
int global_sum = 0;
pthread_mutex_t lock;

// Function each thread will run
void* partial_sum(void* arg) {
    int thread_id = *(int*)arg;
    int chunk_size = N / NUM_THREADS;
    int start = thread_id * chunk_size;
    int end = start + chunk_size;

    int local_sum = 0;
    for (int i = start; i < end; i++) {
        local_sum += arr[i];
    }

    // Synchronize update to global sum
    pthread_mutex_lock(&lock);
    global_sum += local_sum;
    pthread_mutex_unlock(&lock);

    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];

    // Initialize array
    for (int i = 0; i < N; i++) {
        arr[i] = i + 1;  // values 1..16
    }

    pthread_mutex_init(&lock, NULL);

    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, partial_sum, &thread_ids[i]);
    }

    // Join threads (wait for them to finish)
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&lock);

    printf("Final Global Sum = %d\n", global_sum);
    return 0;
}

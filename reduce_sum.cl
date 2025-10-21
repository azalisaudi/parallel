// File: reduce_sum.cl
__kernel void reduce_sum(
    __global const int *input,   // Input data array (Global Memory)
    __global int *partial_sums,  // Output array for block results (Global Memory)
    __local int *local_sums      // Local Shared Memory for reduction
) {
    // 1. Setup IDs
    const uint local_id = get_local_id(0);    // Thread index within the work-group
    const uint global_id = get_global_id(0);  // Global index for input access
    const uint group_size = get_local_size(0); // Work-group size

    // 2. Load Global Data to Local Memory
    local_sums[local_id] = input[global_id];

    // Wait for all threads to finish loading data
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3. Logarithmic Reduction (Tree Sum)
    // 's' starts as half the work-group size and halves each iteration.
    for (uint s = group_size / 2; s > 0; s /= 2) {
        // Only the first half of threads participate in the sum at each step
        if (local_id < s) {
            local_sums[local_id] += local_sums[local_id + s];
        }
        // Synchronize before the next stride reduction
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 4. Write Partial Sum to Global Output
    // Only the first thread (local_id 0) writes the final sum for the work-group.
    if (local_id == 0) {
        // group_id is used as the index in the output array
        partial_sums[get_group_id(0)] = local_sums[0];
    }
}

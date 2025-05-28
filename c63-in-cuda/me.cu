#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include <cuda_runtime.h>

// Motion estimate helpers and motion compensate helpers are defined in me.h
#include "me.h"
#include "tables.h"


/*
    We decided it would be best to have the motion estimation in a single kernel. 
    me_kernel() runs on the gpu and takes in the original frame and reference frame. 
    These frames are used in the motion search, comparing the orignal to a search area in the reference.
    The threads work together by using shared memory to store each thread's best result, 
    then one thread finds the overall best value and writes this final SAD value and motion vector to arrays that will later be sent back to the CPU.
*/
__global__ void me_kernel(
    uint8_t *orig,
    uint8_t *ref,
    int w,
    int h,
    int search_range,
    int mb_cols,
    int *best_mv_x,
    int *best_mv_y,
    int *best_sad
)
{
    // Each block processes one macroblock
    int mb_x = blockIdx.x;
    int mb_y = blockIdx.y;
    
    // Linear thread id within the block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;
    
    if (mb_x >= mb_cols || mb_y >= h / 8)
        return;
        
    // Coordinates of the macroblock
    int mx = mb_x * 8;
    int my = mb_y * 8;
    
    // Search area
    int left = max(mx - search_range, 0);
    int top = max(my - search_range, 0);
    int right = min(mx + search_range, w - 8);
    int bottom = min(my + search_range, h - 8);
    
    // Calculate total search positions
    int search_width = right - left + 1;
    int search_height = bottom - top + 1;
    int total_positions = search_width * search_height;
    
    // Local thread variables for best match
    int thread_best_sad = INT_MAX;
    int thread_best_x = 0;
    int thread_best_y = 0;
    
    // Motion search
    for (int pos = tid; pos < total_positions; pos += block_size) {
        // Convert linear position to 2D coordinates in search area
        int offset_x = pos % search_width;
        int offset_y = pos / search_width;
        int x = left + offset_x;
        int y = top + offset_y;
        int sad = 0;
        
        // boolean to stop the calculation whenever it is true
        bool exceeded_threshold = false;
        
        /*
        SAD value calculation
        Added a sad threshold defined in me.h and can be adjusted
        In theory, it should be more compute effective to stop the calculation
        if the sad threshold has been exceeded, eliminating unecessary compute.
        */
        #pragma unroll
        for (int i = 0; i < 8 && !exceeded_threshold; ++i) {
            #pragma unroll
            for (int j = 0; j < 8 && !exceeded_threshold; ++j) {
                sad += abs(orig[(my + i) * w + (mx + j)] - ref[(y + i) * w + (x + j)]);
                if (sad > SAD_THRESHOLD) {
                    exceeded_threshold = true;
                }
            }
        }
        
        // Update thread's best match if better and if the threshold is not exceeded
        if (!exceeded_threshold && sad < thread_best_sad) {
            thread_best_sad = sad;
            thread_best_x = x - mx;
            thread_best_y = y - my;
        }
    }
    
    // Prepare for parallel reduction using atomics
    int idx = mb_y * mb_cols + mb_x;
    
    // Use atomicMin to find the global minimum SAD across all threads
    // First initialize the best_sad if this is the first thread
    if (tid == 0) {
        best_sad[idx] = INT_MAX;
    }
    __syncthreads();
    
    // Atomic operations to update the best match
    if (thread_best_sad < INT_MAX) {
        // Use atomic operations to update the best SAD value
        int old_sad = atomicMin(&best_sad[idx], thread_best_sad);
        
        // If this thread's SAD is the best so far, update motion vectors
        if (thread_best_sad < old_sad) {
            // We need to ensure only one thread updates the motion vectors
            // This can lead to a race condition, but for motion estimation
            // it's acceptable if we get the motion vector from a position with
            // the same minimum SAD value
            best_mv_x[idx] = thread_best_x;
            best_mv_y[idx] = thread_best_y;
        }
    }
}

/*
    c63_motion_estimate() initializes variables for the device and cpu, handles memory management 
    and handles the results coming back from the device.
*/
void c63_motion_estimate(struct c63_common *cm)
{
    // Define grid dimensions
    dim3 blockSize(8, 8);
    dim3 gridSize_y(cm->mb_cols, cm->mb_rows);
    dim3 gridSize_uv(cm->mb_cols/2, cm->mb_rows/2);
    
    // Arrays to store best SAD values and motion vector coordinates
    int *d_best_mv_x_y, *d_best_mv_y_y, *d_best_sad_y;
    int *d_best_mv_x_u, *d_best_mv_y_u, *d_best_sad_u;
    int *d_best_mv_x_v, *d_best_mv_y_v, *d_best_sad_v;
    
    // Create streams
    cudaStream_t streamY, streamU, streamV;
    cudaStreamCreate(&streamY);
    cudaStreamCreate(&streamU);
    cudaStreamCreate(&streamV);

    // Allocate memory for results only
    cudaMallocManaged((void **)&d_best_mv_x_y, cm->mb_rows * cm->mb_cols * sizeof(int));
    cudaMallocManaged((void **)&d_best_mv_y_y, cm->mb_rows * cm->mb_cols * sizeof(int));
    cudaMallocManaged((void **)&d_best_sad_y, cm->mb_rows * cm->mb_cols * sizeof(int));
    
    cudaMallocManaged((void **)&d_best_mv_x_u, (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(int));
    cudaMallocManaged((void **)&d_best_mv_y_u, (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(int));
    cudaMallocManaged((void **)&d_best_sad_u, (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(int));
    
    cudaMallocManaged((void **)&d_best_mv_x_v, (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(int));
    cudaMallocManaged((void **)&d_best_mv_y_v, (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(int));
    cudaMallocManaged((void **)&d_best_sad_v, (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(int));

    // Launch kernels
    // No need to copy data since it's already in unified memory
    me_kernel<<<gridSize_y, blockSize, 0, streamY>>>(
        cm->curframe->orig->Y, cm->refframe->recons->Y, 
        cm->padw[Y_COMPONENT], cm->padh[Y_COMPONENT],
        cm->me_search_range, cm->mb_cols, 
        d_best_mv_x_y, d_best_mv_y_y, d_best_sad_y);
    
    me_kernel<<<gridSize_uv, blockSize, 0, streamU>>>(
        cm->curframe->orig->U, cm->refframe->recons->U, 
        cm->padw[U_COMPONENT], cm->padh[U_COMPONENT],
        cm->me_search_range, cm->mb_cols/2, 
        d_best_mv_x_u, d_best_mv_y_u, d_best_sad_u);
    
    me_kernel<<<gridSize_uv, blockSize, 0, streamV>>>(
        cm->curframe->orig->V, cm->refframe->recons->V, 
        cm->padw[V_COMPONENT], cm->padh[V_COMPONENT],
        cm->me_search_range, cm->mb_cols/2, 
        d_best_mv_x_v, d_best_mv_y_v, d_best_sad_v);

    // Synchronize to make sure kernels are complete
    cudaStreamSynchronize(streamY);
    cudaStreamSynchronize(streamU);
    cudaStreamSynchronize(streamV);

    // Process results directly
    process_component_results(d_best_mv_x_y, d_best_mv_y_y, d_best_sad_y, 
                             cm->curframe->mbs[Y_COMPONENT], cm->mb_rows, cm->mb_cols);
    process_component_results(d_best_mv_x_u, d_best_mv_y_u, d_best_sad_u, 
                             cm->curframe->mbs[U_COMPONENT], cm->mb_rows/2, cm->mb_cols/2);
    process_component_results(d_best_mv_x_v, d_best_mv_y_v, d_best_sad_v, 
                             cm->curframe->mbs[V_COMPONENT], cm->mb_rows/2, cm->mb_cols/2);

    // Clean up resources
    cudaStreamDestroy(streamY);
    cudaStreamDestroy(streamU);
    cudaStreamDestroy(streamV);
    
    cudaFree(d_best_mv_x_y);
    cudaFree(d_best_mv_y_y);
    cudaFree(d_best_sad_y);
    cudaFree(d_best_mv_x_u);
    cudaFree(d_best_mv_y_u);
    cudaFree(d_best_sad_u);
    cudaFree(d_best_mv_x_v);
    cudaFree(d_best_mv_y_v);
    cudaFree(d_best_sad_v);
}

/* Motion compensation for 8x8 block */
__global__ void mc_kernel(struct macroblock *mbs, uint8_t *predicted, uint8_t *ref, int padw, int padh)
{
    int mb_x = blockIdx.x;
    int mb_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    int x = thread_x + mb_x * 8;
    int y = thread_y + mb_y * 8;

    struct macroblock *mb = &mbs[mb_y * (padw / 8) + mb_x];

    if(!mb->use_mv){
        return;
    }
  
    if(x < padw && y < padh){
        predicted[y * padw + x] = ref[(y + mb->mv_y) * padw + (x + mb->mv_x)];
    }
}   

// Refactored motion compensation function
void c63_motion_compensate(struct c63_common *cm)
{
    // Create streams
    cudaStream_t streamY, streamU, streamV;
    cudaStreamCreate(&streamY);
    cudaStreamCreate(&streamU);
    cudaStreamCreate(&streamV);

    // Create thread blocks and grid dimensions
    dim3 blockSize(8, 8);
    dim3 gridSize_y(cm->mb_cols, cm->mb_rows);
    dim3 gridSize_uv(cm->mb_cols/2, cm->mb_rows/2);

    // Launch kernels directly on unified memory
    mc_kernel<<<gridSize_y, blockSize, 0, streamY>>>(
        cm->curframe->mbs[Y_COMPONENT], cm->curframe->predicted->Y, cm->refframe->recons->Y, 
        cm->padw[Y_COMPONENT], cm->padh[Y_COMPONENT]);

    mc_kernel<<<gridSize_uv, blockSize, 0, streamU>>>(
        cm->curframe->mbs[U_COMPONENT], cm->curframe->predicted->U, cm->refframe->recons->U, 
        cm->padw[U_COMPONENT], cm->padh[U_COMPONENT]);

    mc_kernel<<<gridSize_uv, blockSize, 0, streamV>>>(
        cm->curframe->mbs[V_COMPONENT], cm->curframe->predicted->V, cm->refframe->recons->V, 
        cm->padw[V_COMPONENT], cm->padh[V_COMPONENT]);

    // Wait for all operations to complete
    cudaStreamSynchronize(streamY);
    cudaStreamSynchronize(streamU);
    cudaStreamSynchronize(streamV);

    // Clean up resources
    cudaStreamDestroy(streamY);
    cudaStreamDestroy(streamU);
    cudaStreamDestroy(streamV);
}

// Process results for a component
void process_component_results(
    int *best_mv_x,
    int *best_mv_y,
    int *best_sad,
    struct macroblock *mbs,
    int mb_rows,
    int mb_cols
) {
    for (int mb_y = 0; mb_y < mb_rows; ++mb_y) {
        for (int mb_x = 0; mb_x < mb_cols; ++mb_x) {
            int idx = mb_y * mb_cols + mb_x;
            struct macroblock *mb = &mbs[idx];
            mb->mv_x = best_mv_x[idx];
            mb->mv_y = best_mv_y[idx];
            mb->use_mv = (best_sad[idx] < SAD_THRESHOLD) ? 1 : 0;
            //printf("Motion vector (x: %d, y: %d) with sad value: %d\n", mb->mv_x, mb->mv_y, best_sad[idx]);
        }
    }
}

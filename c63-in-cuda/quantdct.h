#ifndef QUANT_DCT_H
#define QUANT_DCT_H

#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

// Task types
typedef enum {
    TASK_DCT_QUANTIZE,   // For quantization
    TASK_DEQUANTIZE_IDCT // For dequantization
} task_type_t;

// Structure of tasks for the thread pool
typedef struct task_t {
    task_type_t type;    // Either quantize or dequantize
    
    // Common parameters for both task types
    uint32_t width;
    uint32_t height;
    uint8_t *quantization;
    uint8_t *prediction;
    int start_row;
    int num_rows;
    
    // Parameters for quantization
    uint8_t *dct_in_data;
    int16_t *dct_out_data;
    
    // Parameters for dequantization
    int16_t *idct_in_data;
    uint8_t *idct_out_data;
    
    // Task synchronization
    pthread_mutex_t *done_mutex;
    pthread_cond_t *done_cond;
    int *done_count;
    int total_tasks;
    
    // Pointer to linked list
    struct task_t *next;
} task_t;

// Thread pool structure, a pool of workers
typedef struct {
    pthread_t *threads;      // Array of pthreads
    int num_threads;         // Number of threads
    int shutdown;            // Termination flag
    
    // For thread synchronization
    pthread_mutex_t mutex;
    pthread_cond_t task_cond;
    task_t *task_queue;      // Linked list of tasks
    task_t *task_queue_tail; // Tail pointer in order to add tasks to queue in O(1)
} thread_pool_t;

// Task pool structure for pre-allocated tasks
typedef struct {
    task_t *tasks;           // Array of pre-allocated tasks
    int max_tasks;           // Maximum number of tasks the pool can hold
    int initialized;         // Flag to indicate if task pool is initialized
} task_pool_t;

// External global thread pool variable
extern thread_pool_t *pool;

// External global task pool variable
extern task_pool_t *task_pool;

/* Gets the number of threads for the used system. Calls sysconf, and returns the number of cores.
   If number of cores is 0 it returns 1. */
int get_optimal_thread_count();

/*Function to initialize the thread pool. Function calls get_optimal_thread_count() to get number of threads.
  Allocates a thread_pool_t structure, and allocates a list containing the threads (pool->threads).
  Finally it creates the threads using pthread_create().
  The function prints a confirmation message with the number of threads.*/
void thread_pool_init();

/*Function to destroy and clean up the thread pool. This function signalises shutdown to the threads, 
  then waits for the threads to terminate, and frees the tasks.
  Finally the function destroys the mutex and condition variable used for synchronization, 
  and frees allocated thread array, and the thread pool struct.
  The function prints a confirmation message telling the user that the thread pool has been destroyed.*/
void thread_pool_destroy();

/*Function to initialize the task pool. This function takes an uint32_t parameter to identify 
  the maximum height of the frame in order to identify how many tasks is needed
  for processing the frame. The function allocates memory for the task_pool struct 
  and an array of tasks. Then it initializes the values of the struct and prints 
  a confirmation message with the number of tasks. */
void task_pool_init(uint32_t max_height);

/*Function to destroy and clean up the task pool. This function frees the allocated 
  array of tasks, then frees the task_pool struct. Finally it resets the global 
  pointer and prints a confirmation message. */
void task_pool_destroy();

/* Function performs DCT and quantization on images using multiple threads. 
   Requires a thread- and task pool to be initialized before calling */
void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width,
                  uint32_t height, int16_t *out_data, uint8_t *quantization);

/* Function performs IDCT and Dequantization on compressed image data using multiple threads. 
   Requires a thread- and task pool to be initialized before calling */
void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width,
                     uint32_t height, uint8_t *out_data, uint8_t *quantization);

/* Function can quantize multiple or singular macroblock rows. 
   Has been modified using ARM NEON SIMD instructions. */
void dct_quantize_row(uint8_t *in_data, uint8_t *prediction, int w, int h,
                      int16_t *out_data, uint8_t *quantization);

/* Function can dequantize multiple or singular macroblock rows. 
   Has been modified using ARM NEON SIMD instructions. */
void dequantize_idct_row(int16_t *in_data, uint8_t *prediction, int w, int h,
                         uint8_t *out_data, uint8_t *quantization);

                         
#endif
#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h> // For sysconf(_SC_NPROCESSORS_ONLN)

#include "common.h"
#include "tables.h"
#include <arm_neon.h>

#include "quantdct.h"

#define ISQRT2 0.70710678118654f

#define ROWS_PER_TASK 1 // Determines how many rows should be quanized and dequantized per task

// Global thread pool
thread_pool_t *pool = NULL;
// Global task pool
task_pool_t *task_pool = NULL;

// Function to get optimal number of threads based on CPU cores
int get_optimal_thread_count() { 
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN) - 3; 
    return (num_cores > 0) ? num_cores : 1; // Defaults to 1 core
}

/* -------------------------------------------------------------------------------------------------------------
                                        TASK POOL FUNCTIONS
----------------------------------------------------------------------------------------------------------------*/ 

void task_pool_init(uint32_t max_height) {
    // Safety check - return if already initialized
    if (task_pool != NULL) {
        return;
    }
    
    // Calculate maximum number of tasks based on max height
    int macroblock_rows = max_height / 8;
    int max_tasks = (macroblock_rows + ROWS_PER_TASK - 1) / ROWS_PER_TASK;
    
    // Allocate the task pool structure
    task_pool = (task_pool_t *)malloc(sizeof(task_pool_t));
    if (!task_pool) {
        fprintf(stderr, "Failed to allocate task pool structure\n");
        exit(1);
    }
    
    // Allocate the tasks array
    task_pool->tasks = (task_t *)calloc(max_tasks, sizeof(task_t));
    if (!task_pool->tasks) {
        fprintf(stderr, "Failed to allocate task pool memory\n");
        free(task_pool);  // Free the pool structure if task allocation fails
        task_pool = NULL;
        exit(1);
    }
    
    // Initialize the structure fields
    task_pool->max_tasks = max_tasks;
    task_pool->initialized = 1;
    
    printf("Task pool is initialized with %d tasks\n", max_tasks);
}

// Clean-up task pool
void task_pool_destroy() {
    // Safety check
    if (task_pool == NULL) {
        return;
    }
    
    // Free the tasks array if it exists
    if (task_pool->tasks != NULL) {
        free(task_pool->tasks);
    }
    
    // Free the task pool structure itself
    free(task_pool);
    
    // Reset the global pointer
    task_pool = NULL;
    
    printf("Task pool destroyed\n");
}

/* -------------------------------------------------------------------------------------------------------------
                                        THREAD POOL FUNCTIONS
----------------------------------------------------------------------------------------------------------------*/ 

// Main function for workers
static void* thread_worker(void *arg) {
    thread_pool_t *pool = (thread_pool_t *)arg;
    
    while (1) {
        task_t *task = NULL;
        
        // Fetch a task from the queueu
        pthread_mutex_lock(&pool->mutex);
        while (pool->task_queue == NULL && !pool->shutdown) {
            pthread_cond_wait(&pool->task_cond, &pool->mutex); // Thread waits if no tasks and pool is up
        }
        
        // If shutdown and no more tasks, the thread exits the loop
        if (pool->shutdown && pool->task_queue == NULL) {
            pthread_mutex_unlock(&pool->mutex); // Release the mutex
            break;
        }
        
        // Remove task from queue
        task = pool->task_queue;
        pool->task_queue = task->next;
        
        // Update tail of queue
        if (pool->task_queue == NULL) {
            pool->task_queue_tail = NULL;
        }
        
        pthread_mutex_unlock(&pool->mutex);
        
        // Process the task depending on its type
        if (task->type == TASK_DCT_QUANTIZE) {
            dct_quantize_row(
                task->dct_in_data, 
                task->prediction, 
                task->width, 
                task->height, 
                task->dct_out_data, 
                task->quantization
            );
        } else if (task->type == TASK_DEQUANTIZE_IDCT) {
            dequantize_idct_row(
                task->idct_in_data, 
                task->prediction, 
                task->width, 
                task->height, 
                task->idct_out_data, 
                task->quantization
            );
        }
        
        // Marking the task as completed
        pthread_mutex_lock(task->done_mutex);
        (*task->done_count)++;
        if (*task->done_count == task->total_tasks) {
            pthread_cond_broadcast(task->done_cond);
        }

        // Release the mutex
        pthread_mutex_unlock(task->done_mutex);
        
    }
    
    return NULL;
}

// Function to initialize the thread pool
void thread_pool_init() {
    // Safety check
    if (pool != NULL) {
        return;
    }
    
    int num_threads = get_optimal_thread_count(); // Gets number of threads based on the system
    
    // Allocates memory and sets values for the thread pool
    pool = (thread_pool_t *)malloc(sizeof(thread_pool_t));
    pool->num_threads = num_threads;
    pool->shutdown = 0;
    pool->task_queue = NULL;
    pool->task_queue_tail = NULL;
    
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->task_cond, NULL);
    
    // Allocates memory for the threads
    pool->threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    
    // Create all the worker threads
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&pool->threads[i], NULL, thread_worker, pool);
    }

    printf("Thread pool initialized with %d threads\n", num_threads);
}

// Add a task to the thread pool
static void thread_pool_add_task(task_t *task) {
    // Safety check
    if (!pool || !task) {
        if (task) {
            free(task); // If task exits, free it
        }
            return;
    }

    // Lock before adding task
    pthread_mutex_lock(&pool->mutex);
    
    // Add task to queue at the end
    if (pool->task_queue == NULL) {
        pool->task_queue = task;
        pool->task_queue_tail = task;
    } else {
        pool->task_queue_tail->next = task;
        pool->task_queue_tail = task;
    }
    
    // Send signal that the task is available
    pthread_cond_broadcast(&pool->task_cond);

    // Release the mutex
    pthread_mutex_unlock(&pool->mutex);
}

// Block untill all tasks are submitted
static void thread_pool_wait_completion(pthread_mutex_t *done_mutex, pthread_cond_t *done_cond, int *done_count, int total_tasks) {
    pthread_mutex_lock(done_mutex);

    // Check if all tasks are completed
    while (*done_count < total_tasks) {
        pthread_cond_wait(done_cond, done_mutex); // Waiting
    }

    pthread_mutex_unlock(done_mutex);
}

// Clean-up of the thread pool
void thread_pool_destroy() {
    // Safety check
    if (pool == NULL) {
        return;
    }
    
    // Send signal to shut down
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->task_cond);
    pthread_mutex_unlock(&pool->mutex);
    
    // Wait for all threads to finish their tasks
    for (int i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    
    // Free the remaining tasks in the queue
    task_t *task = pool->task_queue;
    while (task != NULL) {
        task_t *tmp = task;
        task = task->next;
        free(tmp);
    }
    
    // Clean-up
    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->task_cond);
    
    // Free
    free(pool->threads);
    free(pool);
    
    // Reset the global pointer
    pool = NULL;

    printf("Thread pool destroyed\n");
}

/* -------------------------------------------------------------------------------------------------------------
                                arm neon optimizations
----------------------------------------------------------------------------------------------------------------*/ 


static void dct_2d(const float* in, float* out)
{
    // Loop through all elements of the block
    for(int v = 0; v < 8; v++)
    {
        for(int u = 0; u < 8; u++)
        {
            // initialize 2 vectors for dct accumulations with values of 0.0f
            float32x4_t dct_acc_1 = vdupq_n_f32(0.0f);
            float32x4_t dct_acc_2 = vdupq_n_f32(0.0f);
            
            for(int y = 0; y < 8; y++)
            {
                // load dctlookup[y][v]
                float y_v = dctlookup[y][v];
                float32x4_t load_y_v = vdupq_n_f32(y_v);
                
                // load all 8 input values
                float32x4_t load_in_1 = vld1q_f32(&in[y*8]);
                float32x4_t load_in_2 = vld1q_f32(&in[y*8 + 4]);
                
                // load x_u lookup
                float x_u_1[4] = {
                    dctlookup[0][u],
                    dctlookup[1][u],
                    dctlookup[2][u],
                    dctlookup[3][u]
                };
                
                float x_u_2[4] = {
                    dctlookup[4][u],
                    dctlookup[5][u],
                    dctlookup[6][u],
                    dctlookup[7][u]
                };
                
                float32x4_t load_x_u_1 = vld1q_f32(x_u_1);
                float32x4_t load_x_u_2 = vld1q_f32(x_u_2);
                
                // first multiplications
                float32x4_t mul_1 = vmulq_f32(load_in_1, load_x_u_1);
                float32x4_t mul_2 = vmulq_f32(load_in_2, load_x_u_2);
                
                // multiply by y_v lookup and accumulate
                dct_acc_1 = vmlaq_f32(dct_acc_1, mul_1, load_y_v);
                dct_acc_2 = vmlaq_f32(dct_acc_2, mul_2, load_y_v);
            }
            
            // combine results and store result to out
            float32x4_t combined = vaddq_f32(dct_acc_1, dct_acc_2);
            out[v*8 + u] = vaddvq_f32(combined);
        }
    }
}

static void idct_2d(const float* in, float* out)
{
    // Loop through all elements of the block
    for(int v = 0; v < 8; v++)
    {
        for(int u = 0; u < 8; u++)
        {
            // initialize 2 vectors for dct accumulations with values of 0.0f
            float32x4_t dct_acc_1 = vdupq_n_f32(0.0f);
            float32x4_t dct_acc_2 = vdupq_n_f32(0.0f);

            for(int y = 0; y < 8; y++)
            {
                // load dctlookup[y][v]
                float v_y = dctlookup[v][y];
                float32x4_t load_v_y = vdupq_n_f32(v_y);
                
                // load all 8 input values
                float32x4_t load_in_1 = vld1q_f32(&in[y*8]);
                float32x4_t load_in_2 = vld1q_f32(&in[y*8 + 4]);
                
                // load u_x lookup
                float u_x_1[4] = {
                    dctlookup[u][0],
                    dctlookup[u][1],
                    dctlookup[u][2],
                    dctlookup[u][3]
                };
                
                float u_x_2[4] = {
                    dctlookup[u][4],
                    dctlookup[u][5],
                    dctlookup[u][6],
                    dctlookup[u][7]
                };
                
                float32x4_t load_u_x_1 = vld1q_f32(u_x_1);
                float32x4_t load_u_x_2 = vld1q_f32(u_x_2);
                
                // first multiplications
                float32x4_t mul_1 = vmulq_f32(load_in_1, load_u_x_1);
                float32x4_t mul_2 = vmulq_f32(load_in_2, load_u_x_2);
                
                // multiply by v_y lookup and accumulate
                dct_acc_1 = vmlaq_f32(dct_acc_1, mul_1, load_v_y);
                dct_acc_2 = vmlaq_f32(dct_acc_2, mul_2, load_v_y);
            }
            
            // combine results and store result to out
            float32x4_t combined = vaddq_f32(dct_acc_1, dct_acc_2);
            out[v*8 + u] = vaddvq_f32(combined);
        }
    }
}

static void scale_block(float *in_data, float *out_data)
{
  int u, v;

  for (v = 0; v < 8; ++v)
  {

    // if v == 0 create 4 lanes with ISQRT2 values, else store lanes with 1.0f
    float32x4_t a2 = vdupq_n_f32(1.0f);
    if(!v){
      a2 = vdupq_n_f32(ISQRT2);
    }

    for (u = 0; u < 8; u+=4)
    {
      // load in_data
      float32x4_t load_in_data = vld1q_f32(&in_data[v*8+u]);

      // if u == 0 set the first lane to ISQRT2, else set all to 1.0f
      float32x4_t a1 = vdupq_n_f32(1.0f);
      if(!u){
        a1 = vsetq_lane_f32(ISQRT2, a1, 0);
      }

      /// perform first multiplication
      float32x4_t first_multiplication = vmulq_f32(load_in_data, a1);

      // store the second multiplicaiton to results
      float32x4_t result = vmulq_f32(first_multiplication, a2);

      // store the result to out_data
      vst1q_f32(&out_data[v*8+u], result);
    }
  }
}

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; zigzag+=4)
  {
    // initialize v and u with increments
    uint8_t v1 = zigzag_V[zigzag], v2 = zigzag_V[zigzag+1], v3 = zigzag_V[zigzag+2], v4 = zigzag_V[zigzag+3];
    uint8_t u1 = zigzag_U[zigzag], u2 = zigzag_U[zigzag+1], u3 = zigzag_U[zigzag+2], u4 = zigzag_U[zigzag+3];

    // initialize in_data with increments
    float value1 = in_data[v1*8+u1];
    float value2 = in_data[v2*8+u2];
    float value3 = in_data[v3*8+u3];
    float value4 = in_data[v4*8+u4];

    // store the in_data values to vector
    float32x4_t load_in_data = {value1, value2, value3, value4};
    
    // initialize quant table with increments
    float quant1 = (float)quant_tbl[zigzag];
    float quant2 = (float)quant_tbl[zigzag+1];
    float quant3 = (float)quant_tbl[zigzag+2];
    float quant4 = (float)quant_tbl[zigzag+3];
    
    // store the quant tables to vector
    float32x4_t quant_vec = {quant1, quant2, quant3, quant4};

    // create a vector with value 0.25 in each lane
    float32x4_t multiplication_factor = vdupq_n_f32(0.25);
    // multiply with 0.25 = divide in data with 4
    float32x4_t in_divided = vmulq_f32(load_in_data, multiplication_factor);
    // divide the result from in_divided with the quant vector
    float32x4_t quantized = vdivq_f32(in_divided, quant_vec);

    // round the final result
    float32x4_t rounded = vrndiq_f32(quantized);
    
    // store results to outdata
    vst1q_f32(&out_data[zigzag], rounded);
  }
}

static void dequantize_block(float *in_data, float *out_data,
    uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; zigzag+=4)
  {
    // initialize v and u with increments
    uint8_t v1 = zigzag_V[zigzag], v2 = zigzag_V[zigzag+1], v3 = zigzag_V[zigzag+2], v4 = zigzag_V[zigzag+3];
    uint8_t u1 = zigzag_U[zigzag], u2 = zigzag_U[zigzag+1], u3 = zigzag_U[zigzag+2], u4 = zigzag_U[zigzag+3];

    // load in_data to vector
    float32x4_t load_in_data = vld1q_f32(&in_data[zigzag]);
    
    // initialize quant table with increments
    float quant1 = (float)quant_tbl[zigzag];
    float quant2 = (float)quant_tbl[zigzag+1];
    float quant3 = (float)quant_tbl[zigzag+2];
    float quant4 = (float)quant_tbl[zigzag+3];
    
    // store the quant tables to vector
    float32x4_t quant_vec = {quant1, quant2, quant3, quant4};

    // create a vector with value 0.25 in each lane
    float32x4_t multiplication_factor = vdupq_n_f32(0.25);
    
    // multiply indata with quant
    float32x4_t in_quant = vmulq_f32(load_in_data, quant_vec);

    // multiply with 0.25 = divide with 4
    float32x4_t divided = vmulq_f32(in_quant, multiplication_factor);

    // round the final result
    float32x4_t rounded = vrndiq_f32(divided);
    
    // store result to out data for each increment
    out_data[v1*8+u1] = vgetq_lane_f32(rounded, 0);
    out_data[v2*8+u2] = vgetq_lane_f32(rounded, 1);
    out_data[v3*8+u3] = vgetq_lane_f32(rounded, 2);
    out_data[v4*8+u4] = vgetq_lane_f32(rounded, 3);
  }
}

static void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

  for( int i = 0; i < 64; i++ ) { mb[i] = in_data[i]; }

  dct_2d(mb, mb2);
  scale_block(mb2, mb);
  quantize_block(mb, mb2, quant_tbl);

  for( int i = 0; i < 64; i++ ) { out_data[i] = mb2[i]; }
}

static void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

  for( int i = 0; i < 64; i++ ) { mb[i] = in_data[i]; }

  dequantize_block(mb, mb2, quant_tbl);
  scale_block(mb2, mb);
  idct_2d(mb, mb2);

  for( int i = 0; i < 64; i++ ) { out_data[i] = mb2[i]; }
}

void dequantize_idct_row(int16_t *in_data, uint8_t *prediction, int w, int h,
    uint8_t *out_data, uint8_t *quantization)
{
   // Process all rows in the given task
   for (int row = 0; row < h; row += 8) {
       int row_offset = row * w;
       
       // Process each 8x8 block in this row
       for(int x = 0; x < w; x += 8) {
           int16_t block[8*8];
           
           // Process one 8x8 block
           dequant_idct_block_8x8(in_data + row_offset + (x*8), block, quantization);
           
           for (int i = 0; i < 8; ++i) {
               // Define offsets
               int block_row = i * 8;
               int data_row = row_offset + i * w + x;
               
               // Load the row into 8 lanes
               int16x8_t block_vec = vld1q_s16(&block[block_row]);
               
               // Load 8 values from prediction 
               uint8x8_t pred_vec = vld1_u8(&prediction[data_row]);
               
               // Convert prediction to uint16
               uint16x8_t pred_16u = vmovl_u8(pred_vec);
               
               // Add them together
               int16x8_t sum = vaddq_s16(
                   block_vec, 
                   vreinterpretq_s16_u16(pred_16u)
               );
               
               // Saturate and convert to uint8
               uint8x8_t result = vqmovun_s16(sum);
               
               // Store all 8 results at once
               vst1_u8(&out_data[data_row], result);
           }
       }
   }
}

void dct_quantize_row(uint8_t *in_data, uint8_t *prediction, int w, int h,
   int16_t *out_data, uint8_t *quantization)
{
   // Process all rows in the given task
   for (int row = 0; row < h; row += 8) {
       int row_offset = row * w;
       
       // Process each 8x8 block in this row
       for(int x = 0; x < w; x += 8) {
           int16_t block[8*8];
           
           // Compute difference for one 8x8 block
           for (int i = 0; i < 8; ++i) {
               // Define offset
               int block_row = i * 8;
               int data_row = row_offset + i * w + x;
               
               // Load 8 predictions and in data
               uint8x8_t pred_vec = vld1_u8(&prediction[data_row]);
               uint8x8_t in_data_vec = vld1_u8(&in_data[data_row]);
               
               // Convert to uint16 vectors
               uint16x8_t pred_16 = vmovl_u8(pred_vec);
               uint16x8_t in_data_16 = vmovl_u8(in_data_vec);
               
               // Subtract in_data with pred
               int16x8_t subtraction_result = vsubq_s16(
                   vreinterpretq_s16_u16(in_data_16),
                   vreinterpretq_s16_u16(pred_16)
               );
               
               // Store the subtraction results to block
               vst1q_s16(&block[block_row], subtraction_result);
           }
           
           // Process the block
           dct_quant_block_8x8(block, out_data + row_offset + (x*8), quantization);
       }
   }
}


/* -------------------------------------------------------------------------------------------------------------
                                    Quantize and dequantize  
----------------------------------------------------------------------------------------------------------------*/ 

void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width,
    uint32_t height, int16_t *out_data, uint8_t *quantization)
{
    // Check if thread pool is initialized
    if (pool == NULL) {
        thread_pool_init();
    }
    
    // Calculate number of macroblock rows
    int macroblock_rows = height / 8;
    
    // Calculate number of tasks
    int num_tasks = (macroblock_rows + ROWS_PER_TASK - 1) / ROWS_PER_TASK;
    
    // Ensure that we have enough tasks in the pool
    if (num_tasks > task_pool->max_tasks) {
        fprintf(stderr, "Error: Not enough tasks in the task pool (required %d, have %d)\n", 
                num_tasks, task_pool->max_tasks);
        return;
    }
    
    // Synchronization variables
    pthread_mutex_t done_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t done_cond = PTHREAD_COND_INITIALIZER;
    int done_count = 0;
    
    // We submit the tasks, each potentially processing multiple rows
    for (int y = 0; y < height; y += 8 * ROWS_PER_TASK) {
        // Get the id of the task
        int task_idx = y / (8 * ROWS_PER_TASK);
        // Get a task from the pool
        task_t *task = &task_pool->tasks[task_idx];
        
        // Initialize the values
        task->type = TASK_DCT_QUANTIZE;
        task->dct_in_data = in_data + y * width;
        task->prediction = prediction + y * width;
        task->width = width;
        
        int rows_this_task = ROWS_PER_TASK;
        if (y + 8 * ROWS_PER_TASK > height) {
            rows_this_task = (height - y) / 8;
        }
         // Height in pixels
        task->height = 8 * rows_this_task;
        task->dct_out_data = out_data + y * width;
        task->quantization = quantization;
        task->done_mutex = &done_mutex;
        task->done_cond = &done_cond;
        task->done_count = &done_count;
        task->total_tasks = num_tasks;
        task->next = NULL;

        // Add task to the thread pool
        thread_pool_add_task(task);
    }
    
    // Wait for all tasks to complete
    thread_pool_wait_completion(&done_mutex, &done_cond, &done_count, num_tasks);
    
    // Cleanup
    pthread_mutex_destroy(&done_mutex);
    pthread_cond_destroy(&done_cond);
}

void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width,
    uint32_t height, uint8_t *out_data, uint8_t *quantization)
{
    // Check if thread pool is initialized
    if (pool == NULL) {
        thread_pool_init();
    }
    
    // Calculate number of macroblock rows
    int macroblock_rows = height / 8; 
    
    // Calculate number of tasks
    int num_tasks = (macroblock_rows + ROWS_PER_TASK - 1) / ROWS_PER_TASK;
    
    // Ensure we have enough tasks in the pool
    if (num_tasks > task_pool->max_tasks) {
        fprintf(stderr, "Error: Not enough tasks in the task pool (required %d, have %d)\n", 
                num_tasks, task_pool->max_tasks);
        return;
    }
    
    // Synchronization variables
    pthread_mutex_t done_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t done_cond = PTHREAD_COND_INITIALIZER;
    int done_count = 0;
    
    // We submit the tasks, each potentially processing multiple rows
    for (int y = 0; y < height; y += 8 * ROWS_PER_TASK) {
        // Get the id of the task
        int task_idx = y / (8 * ROWS_PER_TASK);
        // Get a task from the pool
        task_t *task = &task_pool->tasks[task_idx];
        
        // Initialize the values
        task->type = TASK_DEQUANTIZE_IDCT;
        task->idct_in_data = in_data + y * width;
        task->prediction = prediction + y * width;
        task->width = width;
        
        int rows_this_task = ROWS_PER_TASK;
        if (y + 8 * ROWS_PER_TASK > height) {
            rows_this_task = (height - y) / 8;
        }
         // Height in pixels
        task->height = 8 * rows_this_task;
        task->idct_out_data = out_data + y * width;
        task->quantization = quantization;
        task->done_mutex = &done_mutex;
        task->done_cond = &done_cond;
        task->done_count = &done_count;
        task->total_tasks = num_tasks;
        task->next = NULL;
        
        thread_pool_add_task(task);
    }
    
    // Wait for all tasks to complete
    thread_pool_wait_completion(&done_mutex, &done_cond, &done_count, num_tasks);
    
    // Cleanup
    pthread_mutex_destroy(&done_mutex);
    pthread_cond_destroy(&done_cond);
}
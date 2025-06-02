#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <stdbool.h>
#include <unistd.h>

#include "c63.h"
#include "quantdct.h"
#include "common.h"
#include "me.h"
#include "tables.h"

#include <cuda_runtime.h>
#include <sisci_error.h>
#include <sisci_api.h>

/* Struct to hold frame-data and helper values */
typedef struct {
    uint8_t *yuv_data;
    size_t data_size;
    int frame_number;
    bool valid; // Value to identify if the struct contains data to be encoded
} frame_queue_entry_t;

/* Frame queue to enable the 3-frame pipeline. Array to hold the frame-data. Helper values */
typedef struct {
    frame_queue_entry_t entries[MAX_FRAMES];
    int head, tail, count; // Head is the index of the first to be proccessed, tail is the index of the next frame insertion
    
    /* Values for prints */
    int frames_received;
    int frames_encoded;
    int frames_sent;

    int next_frame_to_encode; // Helper value to keep track of which frame to encode next
    
    /* The frame currently being encoded and values related to the frame */
    yuv_t current_frame;
    char *encoded_data; // Buffer holding the encoded data
    size_t encoded_size;
    int current_frame_number;
    bool current_keyframe;
    bool frame_ready_to_send;
    
    bool quit_requested; // Signal to shutdown
    
    pthread_mutex_t mutex; // Mutual exclusion lock to prevent race conditions
    pthread_cond_t frame_available; // Signal to the encoder thread
    pthread_cond_t encode_done; // Signal to the sender thread
} frame_queue_t;

/* Struct containing all information relevant to c63server */
static struct {
    uint32_t width, height;
    uint32_t remote_node;
    struct c63_common *cm;
    frame_queue_t frame_queue;
    
    /* Resources related to SISCI */
    volatile struct recv_segment *recv_seg;
    volatile struct send_segment *send_seg;
    volatile struct send_segment *client_send;
    volatile struct recv_segment *client_recv;
    sci_dma_queue_t dma_queue;
    sci_local_segment_t recv_segment, send_segment;
    sci_remote_segment_t remote_client_send, remote_client_recv;
} g;

/* Function to initialize the frame_queue_t and threading*/
void frame_queue_init(frame_queue_t *queue, size_t frame_size) {
    memset(queue, 0, sizeof(frame_queue_t));
    
    /* Initialize the entries */
    for (int i = 0; i < MAX_FRAMES; i++) {
        queue->entries[i].yuv_data = (uint8_t*)malloc(frame_size);        
        queue->entries[i].valid = false;
    }
    
    queue->encoded_data = (char*)malloc(MESSAGE_SIZE); // Allocate buffer based on variable MESSAGE_SIZE

    queue->head = 0;
    queue->tail = 0;
    queue->count = 0;
    queue->next_frame_to_encode = 0;
    queue->frame_ready_to_send = false;
    
    pthread_mutex_init(&queue->mutex, NULL);
    pthread_cond_init(&queue->frame_available, NULL);
    pthread_cond_init(&queue->encode_done, NULL);
}

/* Function to clean up the frame_queue_t */
void frame_queue_destroy(frame_queue_t *queue) {
    pthread_mutex_lock(&queue->mutex);
    
    /* Free all the entries */
    for (int i = 0; i < MAX_FRAMES; i++) {
        if (queue->entries[i].yuv_data) {
            free(queue->entries[i].yuv_data);
        }
    }
    
    free(queue->encoded_data); // Free the buffer
    
    /* Cudafree on yuv_t */
    if (queue->current_frame.Y) cudaFree(queue->current_frame.Y);
    if (queue->current_frame.U) cudaFree(queue->current_frame.U);
    if (queue->current_frame.V) cudaFree(queue->current_frame.V);
    
    pthread_mutex_unlock(&queue->mutex);
    
    /* Clean up pthreads */
    pthread_mutex_destroy(&queue->mutex);
    pthread_cond_destroy(&queue->frame_available);
    pthread_cond_destroy(&queue->encode_done);
}

/* Function called by receiver_thread to add frames to queue
    - Locks mutex
    - Checks whether to exit early
    - Adds the frame to the tail
    - Allocates memory and sets appropriate values
    - Advance the pointer and increase counter
    - Signals the encoder thread
    */
bool frame_queue_add(frame_queue_t *queue, const void *yuv_data, size_t data_size) {
    pthread_mutex_lock(&queue->mutex);

    if (queue->quit_requested) {
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    
    if (queue->count >= MAX_FRAMES) {
        printf("Server: Frame queue is full, dropping frame\n");
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    
    frame_queue_entry_t *entry = &queue->entries[queue->tail];
    
    memcpy(entry->yuv_data, yuv_data, data_size);
    entry->data_size = data_size;
    entry->frame_number = queue->frames_received++;
    entry->valid = true;
    
    printf("Server: Frame %d added to queue (position %d, queue size: %d)\n", 
           entry->frame_number, queue->tail, queue->count + 1);
    
    queue->tail = (queue->tail + 1) % MAX_FRAMES;
    queue->count++;
    
    pthread_cond_signal(&queue->frame_available);
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

/* Function called by the encoder thread to get the next frame:
    - Locks mutex 
    - Searches for the next frame to encode
    - Copies YUV data and sets metadata
    - Updates the queue structure 
    - If no frame is found we call pthread_cond_wait
    - Unlock mutex*/
bool frame_queue_get_next_encode_frame(frame_queue_t *queue) {
    pthread_mutex_lock(&queue->mutex);
    
    while (!queue->quit_requested) {
        frame_queue_entry_t *next_frame = NULL;
        int next_index = -1;
        
        for (int i = 0; i < queue->count; i++) {
            int index = (queue->head + i) % MAX_FRAMES;
            frame_queue_entry_t *entry = &queue->entries[index];
            
            if (entry->valid && entry->frame_number == queue->next_frame_to_encode) {
                next_frame = entry;
                next_index = index;
                break;
            }
        }
        
        if (next_frame) {            
            size_t y_size = g.width * g.height;
            size_t u_size = (g.width * g.height) / 4;
            
            const uint8_t *src = next_frame->yuv_data;
            memcpy(queue->current_frame.Y, src, y_size);
            memcpy(queue->current_frame.U, src + y_size, u_size);
            memcpy(queue->current_frame.V, src + y_size + u_size, u_size);
            
            queue->current_frame_number = next_frame->frame_number; // Update metadata
            
            printf("Server: Frame %d ready for encoding\n", queue->current_frame_number);
            
            next_frame->valid = false; // Remove from queue by marking false
            
            /* If this is the head we advance head pointer */
            if (next_index == queue->head) {
                queue->head = (queue->head + 1) % MAX_FRAMES;
                queue->count--;
            } else {
                /* Move the head frame to this position to fill the gap */
                frame_queue_entry_t *head_entry = &queue->entries[queue->head];
                queue->entries[next_index] = *head_entry;
                head_entry->valid = false;
                head_entry->yuv_data = NULL;
                queue->head = (queue->head + 1) % MAX_FRAMES;
                queue->count--;
            }
            
            pthread_mutex_unlock(&queue->mutex);
            return true;
        }
        
        pthread_cond_wait(&queue->frame_available, &queue->mutex); // We wait for the next frame
    }
    
    pthread_mutex_unlock(&queue->mutex);
    return false;
}

/* Function called by encoder_thread to mark the frame as encoded */
void frame_queue_mark_encoded(frame_queue_t *queue) {
    pthread_mutex_lock(&queue->mutex);
    
    queue->frames_encoded++;
    queue->next_frame_to_encode++;
    queue->frame_ready_to_send = true;
    
    printf("Server: Frame %d encoded and ready to send\n", queue->current_frame_number);
    
    pthread_cond_signal(&queue->encode_done);
    pthread_mutex_unlock(&queue->mutex);
}

/*Function called by sender_thread to mark the frame as sent*/
void frame_queue_mark_sent(frame_queue_t *queue) {
    pthread_mutex_lock(&queue->mutex);
    
    queue->frames_sent++;
    queue->frame_ready_to_send = false;
    
    printf("Server: Frame %d sent\n", queue->current_frame_number);
    
    pthread_mutex_unlock(&queue->mutex);
}

// Request shutdown
void frame_queue_request_quit(frame_queue_t *queue) {
    pthread_mutex_lock(&queue->mutex);
    queue->quit_requested = true;
    pthread_cond_broadcast(&queue->frame_available);
    pthread_cond_broadcast(&queue->encode_done);
    pthread_mutex_unlock(&queue->mutex);
}

/* Initialization function for cm */
struct c63_common *init_encoder(int width, int height) {
    struct c63_common *cm = (struct c63_common*)calloc(1, sizeof(struct c63_common));
    
    cm->width = width;
    cm->height = height;
    cm->padw[Y_COMPONENT] = cm->ypw = (uint32_t)(ceil(width / 16.0f) * 16);
    cm->padh[Y_COMPONENT] = cm->yph = (uint32_t)(ceil(height / 16.0f) * 16);
    cm->padw[U_COMPONENT] = cm->upw = (uint32_t)(ceil(width * UX / (YX * 8.0f)) * 8);
    cm->padh[U_COMPONENT] = cm->uph = (uint32_t)(ceil(height * UY / (YY * 8.0f)) * 8);
    cm->padw[V_COMPONENT] = cm->vpw = (uint32_t)(ceil(width * VX / (YX * 8.0f)) * 8);
    cm->padh[V_COMPONENT] = cm->vph = (uint32_t)(ceil(height * VY / (YY * 8.0f)) * 8);
    
    cm->mb_cols = cm->ypw / 8;
    cm->mb_rows = cm->yph / 8;
    cm->qp = 25;
    cm->me_search_range = 16;
    cm->keyframe_interval = 100;
    
    for (int i = 0; i < 64; ++i) {
        cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
        cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
        cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    }
    
    return cm;
}

// Encode frame with queue
void encode_frame_with_queue(struct c63_common *cm, frame_queue_t *queue) {
    printf("Server: Encoding frame %d\n", queue->current_frame_number);
    
    // Update frame references
    destroy_frame(cm->refframe);
    cm->refframe = cm->curframe;
    cm->curframe = create_frame(cm, &queue->current_frame);
    
    // Determine if keyframe
    if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval) {
        cm->curframe->keyframe = 1;
        cm->frames_since_keyframe = 0;
        queue->current_keyframe = true;
    } else {
        cm->curframe->keyframe = 0;
        queue->current_keyframe = false;
    }
    
    // Motion estimation for non-keyframes
    if (!cm->curframe->keyframe) {
        c63_motion_estimate(cm);
        c63_motion_compensate(cm);
    }
    
    // DCT and quantization
    dct_quantize(queue->current_frame.Y, cm->curframe->predicted->Y, cm->padw[Y_COMPONENT],
                 cm->padh[Y_COMPONENT], cm->curframe->residuals->Ydct, cm->quanttbl[Y_COMPONENT]);
    dct_quantize(queue->current_frame.U, cm->curframe->predicted->U, cm->padw[U_COMPONENT],
                 cm->padh[U_COMPONENT], cm->curframe->residuals->Udct, cm->quanttbl[U_COMPONENT]);
    dct_quantize(queue->current_frame.V, cm->curframe->predicted->V, cm->padw[V_COMPONENT],
                 cm->padh[V_COMPONENT], cm->curframe->residuals->Vdct, cm->quanttbl[V_COMPONENT]);
    
    // Dequantization and IDCT for reconstruction
    dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y, cm->ypw, cm->yph, 
                    cm->curframe->recons->Y, cm->quanttbl[Y_COMPONENT]);
    dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U, cm->upw, cm->uph, 
                    cm->curframe->recons->U, cm->quanttbl[U_COMPONENT]);
    dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V, cm->vpw, cm->vph, 
                    cm->curframe->recons->V, cm->quanttbl[V_COMPONENT]);
    
    // Pack encoded data
    char *ptr = queue->encoded_data;
    
    // Keyframe flag
    *((int*)ptr) = queue->current_keyframe ? 1 : 0;
    ptr += sizeof(int);
    
    // DCT coefficients
    size_t ydct_size = cm->ypw * cm->yph * sizeof(int16_t);
    memcpy(ptr, cm->curframe->residuals->Ydct, ydct_size);
    ptr += ydct_size;
    
    size_t udct_size = cm->upw * cm->uph * sizeof(int16_t);
    memcpy(ptr, cm->curframe->residuals->Udct, udct_size);
    ptr += udct_size;
    
    size_t vdct_size = cm->vpw * cm->vph * sizeof(int16_t);
    memcpy(ptr, cm->curframe->residuals->Vdct, vdct_size);
    ptr += vdct_size;
    
    // Macroblock data  
    size_t mby_size = cm->mb_rows * cm->mb_cols * sizeof(struct macroblock);
    memcpy(ptr, cm->curframe->mbs[Y_COMPONENT], mby_size);
    ptr += mby_size;
    
    size_t mbu_size = (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(struct macroblock);
    memcpy(ptr, cm->curframe->mbs[U_COMPONENT], mbu_size);
    ptr += mbu_size;
    
    size_t mbv_size = (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(struct macroblock);
    memcpy(ptr, cm->curframe->mbs[V_COMPONENT], mbv_size);
    
    queue->encoded_size = sizeof(int) + ydct_size + udct_size + vdct_size + 
                          mby_size + mbu_size + mbv_size;
    
    // Update frame counters
    cm->framenum++;
    cm->frames_since_keyframe++;
    if (cm->curframe->keyframe) {
        cm->frames_since_keyframe = 0;
    }
    
    printf("Server: Frame %d encoded (%zu bytes, %s)\n", 
           queue->current_frame_number, queue->encoded_size, 
           queue->current_keyframe ? "keyframe" : "non-keyframe");
}

/* Receiver thread 
    - Loops, Waits for command CMD_YUV_DATA indicating that we have received data from client
    - Extract frame data
    - Sends ack to client: Clear receiving command, flush, setting client ack, flush
    - Calls frame_queue_add()
    - Continues the loop */
void *receiver_thread(void *arg) {
    printf("Server: Receiver thread started\n");
    
    while (!g.frame_queue.quit_requested) {
        // Wait for YUV data command
        while (g.recv_seg->packet.cmd != CMD_YUV_DATA && !g.frame_queue.quit_requested) {
            usleep(1000);
        }
        
        if (g.frame_queue.quit_requested) break;
        
        size_t data_size = g.recv_seg->packet.data_size;
        const void *yuv_data = (const void*)g.recv_seg->message_buffer;
        
        printf("Server: Receiving YUV data (%zu bytes)\n", data_size);
        
        g.recv_seg->packet.cmd = CMD_INVALID;
        SCIFlush(NULL, NO_FLAGS);
        g.client_recv->packet.cmd = CMD_YUV_DATA_ACK;
        SCIFlush(NULL, NO_FLAGS);
        
        // Add frame to queue
        if (!frame_queue_add(&g.frame_queue, yuv_data, data_size)) {
            printf("Server: Failed to add frame to queue\n");
        }
    }
    
    printf("Server: Receiver thread finished\n");
    return NULL;
}

/* Encoder thread
    - Loops, receives the next frame from frame_queue_get_next_encode_frame() which blocks until a frame is available or shutdown is requested
    - Calls  encode_frame_with_queue(), and frame_queue_mark_encoded()
    - Exits when frame_queue_get_next_encode_frame() returns false*/
void *encoder_thread(void *arg) {
    printf("Server: Encoder thread started\n");
    
    while (true) {
        if (!frame_queue_get_next_encode_frame(&g.frame_queue)) break;
        
        // Encode the frame
        encode_frame_with_queue(g.cm, &g.frame_queue);
        
        // Mark as encoded
        frame_queue_mark_encoded(&g.frame_queue);
    }
    
    printf("Server: Encoder thread finished\n");
    return NULL;
}

/* Sender thread
    - Waits for encoded frames by Locking the mutex and checking frame_ready_to_send flag, using pthread_cond_wait() to sleep until encoder thread signals completion
    -  Extracts metadata about frames
    - Copies the encoded buffer
    - Transfer with DMA
    - Notifies the client with command CMD_ENCODED_DATA
    - Waits for ack
    - Calls frame_queue_mark_sent()
    - Exits when g.frame_queue.quit_requested is true */
void *sender_thread(void *arg) {
    printf("Server: Sender thread started\n");
    
    while (!g.frame_queue.quit_requested) {
        // Wait for encoded frame
        pthread_mutex_lock(&g.frame_queue.mutex);
        while (!g.frame_queue.frame_ready_to_send && !g.frame_queue.quit_requested) {
            pthread_cond_wait(&g.frame_queue.encode_done, &g.frame_queue.mutex);
        }
        
        if (g.frame_queue.quit_requested) {
            pthread_mutex_unlock(&g.frame_queue.mutex);
            break;
        }
        
        int frame_number = g.frame_queue.current_frame_number;
        size_t encoded_size = g.frame_queue.encoded_size;
        pthread_mutex_unlock(&g.frame_queue.mutex);
        
        printf("Server: Sending frame %d (%zu bytes)\n", frame_number, encoded_size);
        
        memcpy((void*)g.send_seg->message_buffer, g.frame_queue.encoded_data, encoded_size);
        
        sci_error_t error;
        SCIStartDmaTransfer(g.dma_queue, g.send_segment, g.remote_client_recv,
                           offsetof(struct send_segment, message_buffer),
                           encoded_size,
                           offsetof(struct recv_segment, message_buffer),
                           NO_CALLBACK, NULL, NO_FLAGS, &error);
        
        if (error == SCI_ERR_OK) {
            SCIWaitForDMAQueue(g.dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
        }
        
        if (error != SCI_ERR_OK) {
            printf("Server: DMA transfer failed for frame %d\n", frame_number);
            frame_queue_mark_sent(&g.frame_queue);
            continue;
        }
        
        /* Signal client */
        SCIFlush(NULL, NO_FLAGS);
        g.client_recv->packet.data_size = encoded_size;
        g.client_recv->packet.cmd = CMD_ENCODED_DATA;
        SCIFlush(NULL, NO_FLAGS);
        
        // Wait for acknowledgment
        time_t start = time(NULL);
        while (g.send_seg->packet.cmd != CMD_ENCODED_DATA_ACK) {
            if (time(NULL) - start > TIMEOUT_SECONDS) {
                printf("Server: Timeout waiting for ACK for frame %d\n", frame_number);
                break;
            }
            usleep(1000);
        }
        
        if (g.send_seg->packet.cmd == CMD_ENCODED_DATA_ACK) {
            printf("Server: Frame %d acknowledged\n", frame_number);
            g.send_seg->packet.cmd = CMD_INVALID;
        }
        
        frame_queue_mark_sent(&g.frame_queue);
    }
    
    printf("Server: Sender thread finished\n");
    return NULL;
}

/* Main loop
    - Waits for dimmensions from client
    - Initialize encoder and frame queue
    - Allocate cuda memory 
    - Initialize thread and task pool
    - Launches the three threads
    - Notifies client that dimmensions has been received
    - Monitors for quit command from client
    - Coordinates shutdown of all threads
    - Cleans up resources used
    */
int main_loop() { 
    pthread_t receiver_tid, encoder_tid, sender_tid;
    bool threads_started = false;
    
    printf("Server: Waiting for dimensions...\n");
    
    // Wait for dimensions command only
    while (g.recv_seg->packet.cmd != CMD_DIMENSIONS) {
        usleep(1000);
    }
    
    struct dimensions_data dim_data;
    memcpy(&dim_data, (const void*)g.recv_seg->message_buffer, sizeof(dim_data));
    g.recv_seg->packet.cmd = CMD_INVALID;
    
    printf("Server: Received dimensions %ux%u\n", dim_data.width, dim_data.height);
    
    g.width = dim_data.width;
    g.height = dim_data.height;

    size_t max_frame_size = g.width * g.height * 3 / 2;
    
    // Initialize encoder and frame queue
    g.cm = init_encoder(g.width, g.height);
    frame_queue_init(&g.frame_queue, max_frame_size);

    cudaMallocManaged((void**)&g.frame_queue.current_frame.Y, g.cm->padw[Y_COMPONENT] * g.cm->padh[Y_COMPONENT]);
    cudaMallocManaged((void**)&g.frame_queue.current_frame.U, g.cm->padw[U_COMPONENT] * g.cm->padh[U_COMPONENT]);
    cudaMallocManaged((void**)&g.frame_queue.current_frame.V, g.cm->padw[V_COMPONENT] * g.cm->padh[V_COMPONENT]);

    // Initialize thread pools
    thread_pool_init();
    task_pool_init(g.cm->padh[Y_COMPONENT]);
    
    // Start processing threads
    pthread_create(&receiver_tid, NULL, receiver_thread, NULL);
    pthread_create(&encoder_tid, NULL, encoder_thread, NULL);
    pthread_create(&sender_tid, NULL, sender_thread, NULL);
    threads_started = true;

    SCIFlush(NULL, NO_FLAGS);
    g.client_recv->packet.cmd = CMD_DIMENSIONS_ACK;
    SCIFlush(NULL, NO_FLAGS);
    
    printf("Server: Dimensions acknowledged, processing threads started\n");
    
    // Wait for quit command or completion
    while (true) {
        if (g.recv_seg->packet.cmd == CMD_QUIT) {
            printf("Server: Quit command received\n");
            g.recv_seg->packet.cmd = CMD_INVALID;
            break;
        }
        usleep(10000);
    }
    
    if (threads_started) {
        frame_queue_request_quit(&g.frame_queue);
        pthread_join(receiver_tid, NULL);
        pthread_join(encoder_tid, NULL);
        pthread_join(sender_tid, NULL);
        
        printf("Server: Final statistics:\n");
        printf("  Frames received: %d\n", g.frame_queue.frames_received);
        printf("  Frames encoded: %d\n", g.frame_queue.frames_encoded);
        printf("  Frames sent: %d\n", g.frame_queue.frames_sent);
    }

    /* Clean up */
    if (g.cm) {
        frame_queue_destroy(&g.frame_queue);
        free(g.cm);
        task_pool_destroy();
        thread_pool_destroy();
    }
    
    return g.frame_queue.frames_encoded;
}

/* Initialize SISCI relevant */
bool init_sisci() {
    sci_error_t error;
    unsigned int localAdapterNo = 0;
    
    SCIInitialize(NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    sci_desc_t sd;
    SCIOpen(&sd, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    /* Create server segments */
    SCICreateSegment(sd, &g.recv_segment, SEGMENT_SERVER_RECV, sizeof(struct recv_segment),
                     NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    SCICreateSegment(sd, &g.send_segment, SEGMENT_SERVER_SEND, sizeof(struct send_segment),
                     NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    /* Prepare segments */
    SCIPrepareSegment(g.recv_segment, localAdapterNo, NO_FLAGS, &error);
    SCIPrepareSegment(g.send_segment, localAdapterNo, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    /* Make segments available */
    SCISetSegmentAvailable(g.recv_segment, localAdapterNo, NO_FLAGS, &error);
    SCISetSegmentAvailable(g.send_segment, localAdapterNo, NO_FLAGS, &error);
    
    SCICreateDMAQueue(sd, &g.dma_queue, localAdapterNo, 1, NO_FLAGS, &error); // Create DMA queue
    if (error != SCI_ERR_OK) return false;
    
    sci_map_t recv_map, send_map; // Map local segments
    g.recv_seg = (volatile struct recv_segment *)SCIMapLocalSegment(
        g.recv_segment, &recv_map, 0, sizeof(struct recv_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    g.send_seg = (volatile struct send_segment *)SCIMapLocalSegment(
        g.send_segment, &send_map, 0, sizeof(struct send_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    g.recv_seg->packet.cmd = CMD_INVALID;
    g.send_seg->packet.cmd = CMD_INVALID;
    
    printf("Server: Waiting for client connection...\n");
    
    /* Connect to client segments */
    do {
        SCIConnectSegment(sd, &g.remote_client_send, g.remote_node, SEGMENT_CLIENT_SEND,
                         localAdapterNo, NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    } while (error != SCI_ERR_OK);
    
    do {
        SCIConnectSegment(sd, &g.remote_client_recv, g.remote_node, SEGMENT_CLIENT_RECV,
                         localAdapterNo, NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    } while (error != SCI_ERR_OK);
    
    sci_map_t client_send_map, client_recv_map; // Map client segments
    g.client_send = (volatile struct send_segment *)SCIMapRemoteSegment(
        g.remote_client_send, &client_send_map, 0, sizeof(struct send_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    g.client_recv = (volatile struct recv_segment *)SCIMapRemoteSegment(
        g.remote_client_recv, &client_recv_map, 0, sizeof(struct recv_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    printf("Server: Connected to client, ready for processing\n");
    return true;
}

/* Main function, calls main_loop()*/
int main(int argc, char **argv) {
    int c;
    
    while ((c = getopt(argc, argv, "r:")) != -1) {
        switch (c) {
            case 'r':
                g.remote_node = atoi(optarg);
                break;
            default:
                break;
        }
    }
    
    if (g.remote_node == 0) {
        fprintf(stderr, "Remote node-id not specified. Use -r <remote node-id>\n");
        exit(EXIT_FAILURE);
    }
    
    if (!init_sisci()) {
        fprintf(stderr, "SISCI initialization failed\n");
        exit(EXIT_FAILURE);
    }
    
    int frames_processed = main_loop();
    
    printf("Server: Processed %d frames, exiting\n", frames_processed);
    
    return 0;
}
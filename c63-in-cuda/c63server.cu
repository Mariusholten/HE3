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

// Frame buffer for sequential processing
typedef struct {
    yuv_t image;               // YUV data (CUDA managed memory)
    char *encoded_data;        // Encoded result buffer
    size_t encoded_size;       // Size of encoded data
    int frame_number;          // Frame sequence number
    bool has_yuv_data;         // Whether YUV data is available
    bool is_encoded;           // Whether frame is encoded
    bool keyframe;             // Whether this is a keyframe
} frame_buffer_t;

// Simple frame manager for sequential processing
typedef struct {
    frame_buffer_t frames[MAX_FRAMES];
    
    int next_frame_to_encode;  // Next frame number to encode (sequential)
    int frames_received;       // Total frames received
    int frames_encoded;        // Total frames encoded
    int frames_sent;           // Total frames sent
    
    bool quit_requested;
    
    pthread_mutex_t mutex;
    pthread_cond_t frame_ready, encode_done;
} frame_manager_t;

// Global state
static struct {
    uint32_t width, height;
    uint32_t remote_node;
    struct c63_common *cm;
    frame_manager_t frame_mgr;
    
    // SISCI resources
    volatile struct recv_segment *recv_seg;
    volatile struct send_segment *send_seg;
    volatile struct send_segment *client_send;
    volatile struct recv_segment *client_recv;
    sci_dma_queue_t dma_queue;
    sci_local_segment_t recv_segment, send_segment;
    sci_remote_segment_t remote_client_send, remote_client_recv;
} g;

// Initialize frame manager
void frame_manager_init(frame_manager_t *mgr) {
    memset(mgr, 0, sizeof(frame_manager_t));
    
    // Allocate encoded data buffers
    for (int i = 0; i < MAX_FRAMES; i++) {
        mgr->frames[i].encoded_data = (char*)malloc(MESSAGE_SIZE);
        mgr->frames[i].has_yuv_data = false;
        mgr->frames[i].is_encoded = false;
        mgr->frames[i].image.Y = NULL;
        mgr->frames[i].image.U = NULL;
        mgr->frames[i].image.V = NULL;
    }
    
    mgr->next_frame_to_encode = 0;
    mgr->quit_requested = false;
    
    pthread_mutex_init(&mgr->mutex, NULL);
    pthread_cond_init(&mgr->frame_ready, NULL);
    pthread_cond_init(&mgr->encode_done, NULL);
}

// Cleanup frame manager
void frame_manager_destroy(frame_manager_t *mgr) {
    pthread_mutex_lock(&mgr->mutex);
    
    for (int i = 0; i < MAX_FRAMES; i++) {
        free(mgr->frames[i].encoded_data);
        
        // Clean up CUDA memory
        if (mgr->frames[i].image.Y) cudaFree(mgr->frames[i].image.Y);
        if (mgr->frames[i].image.U) cudaFree(mgr->frames[i].image.U);
        if (mgr->frames[i].image.V) cudaFree(mgr->frames[i].image.V);
    }
    
    pthread_mutex_unlock(&mgr->mutex);
    
    pthread_mutex_destroy(&mgr->mutex);
    pthread_cond_destroy(&mgr->frame_ready);
    pthread_cond_destroy(&mgr->encode_done);
}

// Add received frame
bool frame_manager_add_frame(frame_manager_t *mgr, const void *yuv_data, size_t data_size) {
    pthread_mutex_lock(&mgr->mutex);
    
    if (mgr->quit_requested) {
        pthread_mutex_unlock(&mgr->mutex);
        return false;
    }
    
    // Find available slot
    int slot_idx = -1;
    for (int i = 0; i < MAX_FRAMES; i++) {
        if (!mgr->frames[i].has_yuv_data && !mgr->frames[i].is_encoded) {
            slot_idx = i;
            break;
        }
    }
    
    if (slot_idx == -1) {
        printf("Server: No available slots for frame\n");
        pthread_mutex_unlock(&mgr->mutex);
        return false;
    }
    
    frame_buffer_t *frame = &mgr->frames[slot_idx];
    
    // Allocate CUDA memory if needed
    if (!frame->image.Y) {
        cudaMallocManaged((void**)&frame->image.Y, g.cm->padw[Y_COMPONENT] * g.cm->padh[Y_COMPONENT]);
        cudaMallocManaged((void**)&frame->image.U, g.cm->padw[U_COMPONENT] * g.cm->padh[U_COMPONENT]);
        cudaMallocManaged((void**)&frame->image.V, g.cm->padw[V_COMPONENT] * g.cm->padh[V_COMPONENT]);
    }
    
    // Copy YUV data
    size_t y_size = g.width * g.height;
    size_t u_size = (g.width * g.height) / 4;
    
    const char *src = (const char*)yuv_data;
    memcpy(frame->image.Y, src, y_size);
    memcpy(frame->image.U, src + y_size, u_size);
    memcpy(frame->image.V, src + y_size + u_size, u_size);
    
    frame->frame_number = mgr->frames_received++;
    frame->has_yuv_data = true;
    frame->is_encoded = false;
    
    printf("Server: Frame %d received in slot %d\n", frame->frame_number, slot_idx);
    
    pthread_cond_signal(&mgr->frame_ready);
    pthread_mutex_unlock(&mgr->mutex);
    return true;
}

// Get next frame to encode (sequential)
frame_buffer_t* frame_manager_get_encode_frame(frame_manager_t *mgr) {
    pthread_mutex_lock(&mgr->mutex);
    
    frame_buffer_t *frame = NULL;
    
    while (!mgr->quit_requested) {
        // Look for the next frame in sequence
        for (int i = 0; i < MAX_FRAMES; i++) {
            if (mgr->frames[i].has_yuv_data && 
                !mgr->frames[i].is_encoded &&
                mgr->frames[i].frame_number == mgr->next_frame_to_encode) {
                frame = &mgr->frames[i];
                break;
            }
        }
        
        if (frame) break;
        
        // Wait for the next frame to arrive
        pthread_cond_wait(&mgr->frame_ready, &mgr->mutex);
    }
    
    pthread_mutex_unlock(&mgr->mutex);
    return frame;
}

// Mark frame as encoded
void frame_manager_frame_encoded(frame_manager_t *mgr, frame_buffer_t *frame) {
    pthread_mutex_lock(&mgr->mutex);
    
    frame->is_encoded = true;
    mgr->frames_encoded++;
    mgr->next_frame_to_encode++;
    
    printf("Server: Frame %d encoded\n", frame->frame_number);
    
    pthread_cond_signal(&mgr->encode_done);
    pthread_mutex_unlock(&mgr->mutex);
}

// Get next encoded frame to send
frame_buffer_t* frame_manager_get_send_frame(frame_manager_t *mgr) {
    pthread_mutex_lock(&mgr->mutex);
    
    frame_buffer_t *frame = NULL;
    
    while (!mgr->quit_requested) {
        // Find oldest encoded frame
        int oldest_frame = INT_MAX;
        int oldest_idx = -1;
        
        for (int i = 0; i < MAX_FRAMES; i++) {
            if (mgr->frames[i].is_encoded && 
                mgr->frames[i].frame_number < oldest_frame) {
                oldest_frame = mgr->frames[i].frame_number;
                oldest_idx = i;
            }
        }
        
        if (oldest_idx != -1) {
            frame = &mgr->frames[oldest_idx];
            break;
        }
        
        // Wait for encoded frame
        pthread_cond_wait(&mgr->encode_done, &mgr->mutex);
    }
    
    pthread_mutex_unlock(&mgr->mutex);
    return frame;
}

// Mark frame as sent and cleanup
void frame_manager_frame_sent(frame_manager_t *mgr, frame_buffer_t *frame) {
    pthread_mutex_lock(&mgr->mutex);
    
    frame->has_yuv_data = false;
    frame->is_encoded = false;
    mgr->frames_sent++;
    
    printf("Server: Frame %d sent\n", frame->frame_number);
    
    pthread_mutex_unlock(&mgr->mutex);
}

// Request shutdown
void frame_manager_request_quit(frame_manager_t *mgr) {
    pthread_mutex_lock(&mgr->mutex);
    mgr->quit_requested = true;
    pthread_cond_broadcast(&mgr->frame_ready);
    pthread_cond_broadcast(&mgr->encode_done);
    pthread_mutex_unlock(&mgr->mutex);
}

// Initialize encoder
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

// Encode frame - sequential as required
void encode_frame(struct c63_common *cm, frame_buffer_t *frame) {
    printf("Server: Encoding frame %d\n", frame->frame_number);
    
    // Update frame references
    destroy_frame(cm->refframe);
    cm->refframe = cm->curframe;
    cm->curframe = create_frame(cm, &frame->image);
    
    // Determine if keyframe
    if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval) {
        cm->curframe->keyframe = 1;
        cm->frames_since_keyframe = 0;
        frame->keyframe = true;
    } else {
        cm->curframe->keyframe = 0;
        frame->keyframe = false;
    }
    
    // Motion estimation for non-keyframes
    if (!cm->curframe->keyframe) {
        c63_motion_estimate(cm);
        c63_motion_compensate(cm);
    }
    
    // DCT and quantization
    dct_quantize(frame->image.Y, cm->curframe->predicted->Y, cm->padw[Y_COMPONENT],
                 cm->padh[Y_COMPONENT], cm->curframe->residuals->Ydct, cm->quanttbl[Y_COMPONENT]);
    dct_quantize(frame->image.U, cm->curframe->predicted->U, cm->padw[U_COMPONENT],
                 cm->padh[U_COMPONENT], cm->curframe->residuals->Udct, cm->quanttbl[U_COMPONENT]);
    dct_quantize(frame->image.V, cm->curframe->predicted->V, cm->padw[V_COMPONENT],
                 cm->padh[V_COMPONENT], cm->curframe->residuals->Vdct, cm->quanttbl[V_COMPONENT]);
    
    // Dequantization and IDCT for reconstruction
    dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y, cm->ypw, cm->yph, 
                    cm->curframe->recons->Y, cm->quanttbl[Y_COMPONENT]);
    dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U, cm->upw, cm->uph, 
                    cm->curframe->recons->U, cm->quanttbl[U_COMPONENT]);
    dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V, cm->vpw, cm->vph, 
                    cm->curframe->recons->V, cm->quanttbl[V_COMPONENT]);
    
    // Pack encoded data
    char *ptr = frame->encoded_data;
    
    // Keyframe flag for client
    *((int*)ptr) = frame->keyframe ? 1 : 0;
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
    
    frame->encoded_size = sizeof(int) + ydct_size + udct_size + vdct_size + 
                         mby_size + mbu_size + mbv_size;
    
    // Update frame counters
    cm->framenum++;
    cm->frames_since_keyframe++;
    if (cm->curframe->keyframe) {
        cm->frames_since_keyframe = 0;
    }
    
    printf("Server: Frame %d encoded (%zu bytes, %s)\n", 
           frame->frame_number, frame->encoded_size, 
           frame->keyframe ? "keyframe" : "non-keyframe");
}

// Receiver thread - handles YUV data from client
void *receiver_thread(void *arg) {
    printf("Server: Receiver thread started\n");
    
    while (!g.frame_mgr.quit_requested) {
        // Wait for YUV data command
        while (g.recv_seg->packet.cmd != CMD_YUV_DATA && !g.frame_mgr.quit_requested) {
            usleep(1000);
        }
        
        if (g.frame_mgr.quit_requested) break;
        
        size_t data_size = g.recv_seg->packet.data_size;
        const void *yuv_data = (const void*)g.recv_seg->message_buffer;
        
        printf("Server: Receiving YUV data (%zu bytes)\n", data_size);
        
        // Add frame to manager
        if (frame_manager_add_frame(&g.frame_mgr, yuv_data, data_size)) {
            // Send acknowledgment
            g.recv_seg->packet.cmd = CMD_INVALID;
            SCIFlush(NULL, NO_FLAGS);
            g.client_recv->packet.cmd = CMD_YUV_DATA_ACK;
            SCIFlush(NULL, NO_FLAGS);
            
            // Signal encoder thread that new frame is available
            pthread_mutex_lock(&g.frame_mgr.mutex);
            pthread_cond_signal(&g.frame_mgr.frame_ready);
            pthread_mutex_unlock(&g.frame_mgr.mutex);
        } else {
            printf("Server: Failed to add frame\n");
            g.recv_seg->packet.cmd = CMD_INVALID;
        }
    }
    
    printf("Server: Receiver thread finished\n");
    return NULL;
}

// Encoder thread - sequential encoding
void *encoder_thread(void *arg) {
    printf("Server: Encoder thread started\n");
    
    while (true) {
        frame_buffer_t *frame = frame_manager_get_encode_frame(&g.frame_mgr);
        if (!frame) break;
        
        // Encode the frame
        encode_frame(g.cm, frame);
        
        // Mark as encoded
        frame_manager_frame_encoded(&g.frame_mgr, frame);
    }
    
    printf("Server: Encoder thread finished\n");
    return NULL;
}

// Sender thread - sends results back to client
void *sender_thread(void *arg) {
    printf("Server: Sender thread started\n");
    
    while (true) {
        frame_buffer_t *frame = frame_manager_get_send_frame(&g.frame_mgr);
        if (!frame) break;
        
        printf("Server: Sending frame %d (%zu bytes)\n", 
               frame->frame_number, frame->encoded_size);
        
        // Copy to send buffer
        memcpy((void*)g.send_seg->message_buffer, frame->encoded_data, frame->encoded_size);
        
        // DMA transfer to client
        sci_error_t error;
        SCIStartDmaTransfer(g.dma_queue, g.send_segment, g.remote_client_recv,
                           offsetof(struct send_segment, message_buffer),
                           frame->encoded_size,
                           offsetof(struct recv_segment, message_buffer),
                           NO_CALLBACK, NULL, NO_FLAGS, &error);
        
        if (error == SCI_ERR_OK) {
            SCIWaitForDMAQueue(g.dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
        }
        
        if (error != SCI_ERR_OK) {
            printf("Server: DMA transfer failed for frame %d\n", frame->frame_number);
            frame_manager_frame_sent(&g.frame_mgr, frame);
            continue;
        }
        
        // Signal client
        SCIFlush(NULL, NO_FLAGS);
        g.client_recv->packet.data_size = frame->encoded_size;
        g.client_recv->packet.cmd = CMD_ENCODED_DATA;
        SCIFlush(NULL, NO_FLAGS);
        
        // Wait for acknowledgment
        time_t start = time(NULL);
        while (g.send_seg->packet.cmd != CMD_ENCODED_DATA_ACK) {
            if (time(NULL) - start > TIMEOUT_SECONDS) {
                printf("Server: Timeout waiting for ACK for frame %d\n", frame->frame_number);
                break;
            }
            usleep(1000);
        }
        
        if (g.send_seg->packet.cmd == CMD_ENCODED_DATA_ACK) {
            printf("Server: Frame %d acknowledged\n", frame->frame_number);
            g.send_seg->packet.cmd = CMD_INVALID;
        }
        
        // Mark frame as sent and free up slot for new frames
        frame_manager_frame_sent(&g.frame_mgr, frame);
        
        // Signal receiver thread that slot is available for new frames
        pthread_mutex_lock(&g.frame_mgr.mutex);
        pthread_cond_signal(&g.frame_mgr.frame_ready);
        pthread_mutex_unlock(&g.frame_mgr.mutex);
    }
    
    printf("Server: Sender thread finished\n");
    return NULL;
}

// Main processing loop
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
    
    // Initialize encoder and frame manager
    g.cm = init_encoder(g.width, g.height);
    frame_manager_init(&g.frame_mgr);
    
    // Initialize thread pools
    thread_pool_init();
    task_pool_init(g.cm->padh[Y_COMPONENT]);
    
    // Start processing threads
    pthread_create(&receiver_tid, NULL, receiver_thread, NULL);
    pthread_create(&encoder_tid, NULL, encoder_thread, NULL);
    pthread_create(&sender_tid, NULL, sender_thread, NULL);
    threads_started = true;
    
    // Send acknowledgment
    sci_error_t error;
    memcpy((void*)g.send_seg->message_buffer, &dim_data, sizeof(dim_data));
    SCIStartDmaTransfer(g.dma_queue, g.send_segment, g.remote_client_recv,
                       offsetof(struct send_segment, message_buffer),
                       sizeof(dim_data),
                       offsetof(struct recv_segment, message_buffer),
                       NO_CALLBACK, NULL, NO_FLAGS, &error);
    SCIWaitForDMAQueue(g.dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    
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
        usleep(10000); // 10ms
    }
    
    if (threads_started) {
        frame_manager_request_quit(&g.frame_mgr);
        pthread_join(receiver_tid, NULL);
        pthread_join(encoder_tid, NULL);
        pthread_join(sender_tid, NULL);
        
        printf("Server: Final statistics:\n");
        printf("  Frames received: %d\n", g.frame_mgr.frames_received);
        printf("  Frames encoded: %d\n", g.frame_mgr.frames_encoded);
        printf("  Frames sent: %d\n", g.frame_mgr.frames_sent);
    }
    
    // Cleanup
    if (g.cm) {
        frame_manager_destroy(&g.frame_mgr);
        free(g.cm);
        task_pool_destroy();
        thread_pool_destroy();
    }
    
    return g.frame_mgr.frames_encoded;
}

// Initialize SISCI
bool init_sisci() {
    sci_error_t error;
    unsigned int localAdapterNo = 0;
    
    SCIInitialize(NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    sci_desc_t sd;
    SCIOpen(&sd, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    // Create server segments
    SCICreateSegment(sd, &g.recv_segment, SEGMENT_SERVER_RECV, sizeof(struct recv_segment),
                     NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    SCICreateSegment(sd, &g.send_segment, SEGMENT_SERVER_SEND, sizeof(struct send_segment),
                     NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    // Prepare segments
    SCIPrepareSegment(g.recv_segment, localAdapterNo, NO_FLAGS, &error);
    SCIPrepareSegment(g.send_segment, localAdapterNo, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    // Make segments available
    SCISetSegmentAvailable(g.recv_segment, localAdapterNo, NO_FLAGS, &error);
    SCISetSegmentAvailable(g.send_segment, localAdapterNo, NO_FLAGS, &error);
    
    // Create DMA queue
    SCICreateDMAQueue(sd, &g.dma_queue, localAdapterNo, 1, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    // Map local segments
    sci_map_t recv_map, send_map;
    g.recv_seg = (volatile struct recv_segment *)SCIMapLocalSegment(
        g.recv_segment, &recv_map, 0, sizeof(struct recv_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    g.send_seg = (volatile struct send_segment *)SCIMapLocalSegment(
        g.send_segment, &send_map, 0, sizeof(struct send_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    g.recv_seg->packet.cmd = CMD_INVALID;
    g.send_seg->packet.cmd = CMD_INVALID;
    
    printf("Server: Waiting for client connection...\n");
    
    // Connect to client segments
    do {
        SCIConnectSegment(sd, &g.remote_client_send, g.remote_node, SEGMENT_CLIENT_SEND,
                         localAdapterNo, NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    } while (error != SCI_ERR_OK);
    
    do {
        SCIConnectSegment(sd, &g.remote_client_recv, g.remote_node, SEGMENT_CLIENT_RECV,
                         localAdapterNo, NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    } while (error != SCI_ERR_OK);
    
    // Map client segments
    sci_map_t client_send_map, client_recv_map;
    g.client_send = (volatile struct send_segment *)SCIMapRemoteSegment(
        g.remote_client_send, &client_send_map, 0, sizeof(struct send_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    g.client_recv = (volatile struct recv_segment *)SCIMapRemoteSegment(
        g.remote_client_recv, &client_recv_map, 0, sizeof(struct recv_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    printf("Server: Connected to client, ready for processing\n");
    return true;
}

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
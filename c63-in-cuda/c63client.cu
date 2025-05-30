#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <stdbool.h>

#include <sisci_error.h>
#include <sisci_api.h>

#include "c63.h"
#include "c63_write.h"
#include "common.h"
#include "tables.h"

// Frame buffer for 3-frame pipeline
typedef struct {
    yuv_t *yuv_data;           // YUV frame data
    char *encoded_data;        // Encoded result from server
    size_t encoded_size;       // Size of encoded data
    int frame_number;          // Sequential frame number
    bool ready_to_send;        // Ready for sending to server
    bool result_received;      // Result received from server
    bool keyframe;             // Whether this is a keyframe
} frame_buffer_t;

// Pipeline state for 3-frame architecture
typedef struct {
    frame_buffer_t frames[MAX_FRAMES];  // 3 frame buffers
    
    int next_read_frame;       // Next frame number to read
    int next_send_frame;       // Next frame number to send
    int next_write_frame;      // Next frame number to write
    
    int frames_read;           // Total frames read
    int frames_sent;           // Total frames sent to server
    int frames_received;       // Total results received
    int frames_written;        // Total frames written to output
    
    bool finished_reading;     // EOF reached
    bool quit_requested;       // Shutdown requested
    
    pthread_mutex_t mutex;
    pthread_cond_t frame_ready_to_send;
    pthread_cond_t result_ready_to_write;
    pthread_cond_t slot_available;
} pipeline_t;

// Global state
static struct {
    char *input_file, *output_file;
    uint32_t remote_node, width, height;
    int limit_frames;
    FILE *infile, *outfile;
    pipeline_t pipeline;
    struct c63_common *cm;
    
    // SISCI resources
    volatile struct send_segment *send_seg;
    volatile struct recv_segment *recv_seg;
    volatile struct recv_segment *server_recv;
    volatile struct send_segment *server_send;
    sci_dma_queue_t dma_queue;
    sci_local_segment_t send_segment, recv_segment;
    sci_remote_segment_t remote_server_recv, remote_server_send;
} g;

// Initialize pipeline
void pipeline_init(pipeline_t *p) {
    memset(p, 0, sizeof(pipeline_t));
    
    for (int i = 0; i < MAX_FRAMES; i++) {
        p->frames[i].encoded_data = (char*)malloc(MESSAGE_SIZE);
        p->frames[i].yuv_data = NULL;
        p->frames[i].ready_to_send = false;
        p->frames[i].result_received = false;
        p->frames[i].frame_number = -1;
    }
    
    pthread_mutex_init(&p->mutex, NULL);
    pthread_cond_init(&p->frame_ready_to_send, NULL);
    pthread_cond_init(&p->result_ready_to_write, NULL);
    pthread_cond_init(&p->slot_available, NULL);
}

// Cleanup pipeline
void pipeline_destroy(pipeline_t *p) {
    for (int i = 0; i < MAX_FRAMES; i++) {
        if (p->frames[i].yuv_data) {
            free(p->frames[i].yuv_data->Y);
            free(p->frames[i].yuv_data->U);
            free(p->frames[i].yuv_data->V);
            free(p->frames[i].yuv_data);
        }
        free(p->frames[i].encoded_data);
    }
    
    pthread_mutex_destroy(&p->mutex);
    pthread_cond_destroy(&p->frame_ready_to_send);
    pthread_cond_destroy(&p->result_ready_to_write);
    pthread_cond_destroy(&p->slot_available);
}

// Read YUV frame from file
static yuv_t *read_yuv_frame() {
    yuv_t *image = (yuv_t*)malloc(sizeof(*image));
    
    image->Y = (uint8_t*)calloc(1, g.cm->padw[Y_COMPONENT] * g.cm->padh[Y_COMPONENT]);
    image->U = (uint8_t*)calloc(1, g.cm->padw[U_COMPONENT] * g.cm->padh[U_COMPONENT]);
    image->V = (uint8_t*)calloc(1, g.cm->padw[V_COMPONENT] * g.cm->padh[V_COMPONENT]);
    
    size_t len = 0;
    len += fread(image->Y, 1, g.width * g.height, g.infile);
    len += fread(image->U, 1, (g.width * g.height) / 4, g.infile);
    len += fread(image->V, 1, (g.width * g.height) / 4, g.infile);
    
    if (ferror(g.infile) || feof(g.infile) || len != g.width * g.height * 1.5) {
        free(image->Y);
        free(image->U);
        free(image->V);
        free(image);
        return NULL;
    }
    
    return image;
}

// Wait for command with timeout
bool wait_for_command(uint32_t expected_cmd, volatile struct recv_segment *seg, int timeout) {
    time_t start = time(NULL);
    while (seg->packet.cmd != expected_cmd) {
        if (time(NULL) - start > timeout) return false;
        usleep(1000);
    }
    return true;
}

// Send DMA data to server
bool send_dma_data(void *data, size_t size) {
    sci_error_t error;
    
    memcpy((void*)g.send_seg->message_buffer, data, size);
    
    SCIStartDmaTransfer(g.dma_queue, g.send_segment, g.remote_server_recv,
                       offsetof(struct send_segment, message_buffer),
                       size, offsetof(struct recv_segment, message_buffer),
                       NO_CALLBACK, NULL, NO_FLAGS, &error);
    
    if (error != SCI_ERR_OK) return false;
    
    SCIWaitForDMAQueue(g.dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    return error == SCI_ERR_OK;
}

// Thread 1: Read frames and send to server
void *read_send_thread(void *arg) {
    printf("ReadSend: Thread started\n");
    
    while (true) {
        // Step 1: Read frames
        pthread_mutex_lock(&g.pipeline.mutex);
        
        // Check if we should stop reading
        if (g.pipeline.quit_requested || 
            (g.limit_frames && g.pipeline.frames_read >= g.limit_frames)) {
            g.pipeline.finished_reading = true;
            pthread_cond_broadcast(&g.pipeline.frame_ready_to_send);
            pthread_cond_broadcast(&g.pipeline.result_ready_to_write);
            pthread_mutex_unlock(&g.pipeline.mutex);
            break;
        }
        
        // Find available slot for new frame
        int read_slot = -1;
        for (int i = 0; i < MAX_FRAMES; i++) {
            if (g.pipeline.frames[i].frame_number == -1) {
                read_slot = i;
                break;
            }
        }
        
        if (read_slot == -1) {
            // Wait for slot to become available
            pthread_cond_wait(&g.pipeline.slot_available, &g.pipeline.mutex);
            pthread_mutex_unlock(&g.pipeline.mutex);
            continue;
        }
        
        pthread_mutex_unlock(&g.pipeline.mutex);
        
        // Read frame outside of lock
        yuv_t *image = read_yuv_frame();
        if (!image) {
            pthread_mutex_lock(&g.pipeline.mutex);
            g.pipeline.finished_reading = true;
            pthread_cond_broadcast(&g.pipeline.frame_ready_to_send);
            pthread_cond_broadcast(&g.pipeline.result_ready_to_write);
            pthread_mutex_unlock(&g.pipeline.mutex);
            printf("ReadSend: End of file reached\n");
            break;
        }
        
        pthread_mutex_lock(&g.pipeline.mutex);
        
        // Store frame in pipeline
        frame_buffer_t *frame = &g.pipeline.frames[read_slot];
        frame->yuv_data = image;
        frame->frame_number = g.pipeline.frames_read++;
        frame->ready_to_send = true;
        frame->result_received = false;
        
        printf("ReadSend: Read frame %d into slot %d\n", frame->frame_number, read_slot);
        pthread_cond_signal(&g.pipeline.frame_ready_to_send);
        pthread_mutex_unlock(&g.pipeline.mutex);
        
        // Step 2: Send frames to server
        while (true) {
            pthread_mutex_lock(&g.pipeline.mutex);
            
            // Look for frame ready to send (in sequential order)
            frame_buffer_t *send_frame = NULL;
            for (int i = 0; i < MAX_FRAMES; i++) {
                if (g.pipeline.frames[i].ready_to_send && 
                    !g.pipeline.frames[i].result_received &&
                    g.pipeline.frames[i].frame_number == g.pipeline.next_send_frame) {
                    send_frame = &g.pipeline.frames[i];
                    break;
                }
            }
            
            if (!send_frame) {
                // Check if we're done
                if (g.pipeline.finished_reading && g.pipeline.frames_sent >= g.pipeline.frames_read) {
                    pthread_mutex_unlock(&g.pipeline.mutex);
                    printf("ReadSend: All frames sent\n");
                    return NULL;
                }
                
                // Wait for frame to become ready to send
                pthread_cond_wait(&g.pipeline.frame_ready_to_send, &g.pipeline.mutex);
                pthread_mutex_unlock(&g.pipeline.mutex);
                break; // Go back to reading
            }
            
            pthread_mutex_unlock(&g.pipeline.mutex);
            
            // Send frame to server
            yuv_t *send_image = send_frame->yuv_data;
            int frame_number = send_frame->frame_number;
            
            printf("ReadSend: Sending frame %d\n", frame_number);
            
            // Pack YUV data
            size_t y_size = g.width * g.height;
            size_t u_size = (g.width * g.height) / 4;
            size_t v_size = (g.width * g.height) / 4;
            size_t total_size = y_size + u_size + v_size;
            
            memcpy((void*)g.send_seg->message_buffer, send_image->Y, y_size);
            memcpy((void*)(g.send_seg->message_buffer + y_size), send_image->U, u_size);
            memcpy((void*)(g.send_seg->message_buffer + y_size + u_size), send_image->V, v_size);
            
            // Send YUV data via DMA
            if (!send_dma_data((void*)g.send_seg->message_buffer, total_size)) {
                printf("ReadSend: DMA transfer failed for frame %d\n", frame_number);
                continue;
            }
            
            // Signal server about YUV data
            SCIFlush(NULL, NO_FLAGS);
            g.server_recv->packet.cmd = CMD_YUV_DATA;
            g.server_recv->packet.data_size = total_size;
            SCIFlush(NULL, NO_FLAGS);
            
            // Wait for YUV acknowledgment
            if (!wait_for_command(CMD_YUV_DATA_ACK, g.recv_seg, TIMEOUT_SECONDS)) {
                printf("ReadSend: Timeout waiting for YUV ACK for frame %d\n", frame_number);
                continue;
            }
            g.recv_seg->packet.cmd = CMD_INVALID;
            
            pthread_mutex_lock(&g.pipeline.mutex);
            g.pipeline.frames_sent++;
            g.pipeline.next_send_frame++;
            pthread_mutex_unlock(&g.pipeline.mutex);
            
            printf("ReadSend: Frame %d sent successfully\n", frame_number);
            
            // Continue sending more frames if available
        }
    }
    
    printf("ReadSend: Thread finished\n");
    return NULL;
}

// Thread 2: Receive results and write to output
void *receive_write_thread(void *arg) {
    printf("ReceiveWrite: Thread started\n");
    
    while (true) {
        // Step 1: Receive encoded data from server
        if (!wait_for_command(CMD_ENCODED_DATA, g.recv_seg, TIMEOUT_SECONDS * 4)) {
            pthread_mutex_lock(&g.pipeline.mutex);
            if (g.pipeline.finished_reading && 
                g.pipeline.frames_received >= g.pipeline.frames_sent) {
                pthread_mutex_unlock(&g.pipeline.mutex);
                printf("ReceiveWrite: All results received\n");
                break;
            }
            pthread_mutex_unlock(&g.pipeline.mutex);
            usleep(10000); // 10ms
            continue;
        }
        
        // Process encoded data
        size_t data_size = g.recv_seg->packet.data_size;
        char *encoded_data = (char*)g.recv_seg->message_buffer;
        
        // Find the frame this result belongs to
        pthread_mutex_lock(&g.pipeline.mutex);
        frame_buffer_t *result_frame = NULL;
        int expected_frame = g.pipeline.frames_received;
        
        for (int i = 0; i < MAX_FRAMES; i++) {
            if (g.pipeline.frames[i].frame_number == expected_frame) {
                result_frame = &g.pipeline.frames[i];
                break;
            }
        }
        
        if (!result_frame) {
            pthread_mutex_unlock(&g.pipeline.mutex);
            printf("ReceiveWrite: Could not find frame for result %d\n", expected_frame);
            g.recv_seg->packet.cmd = CMD_INVALID;
            continue;
        }
        
        pthread_mutex_unlock(&g.pipeline.mutex);
        
        // Extract keyframe flag and copy encoded data
        result_frame->keyframe = *((int*)encoded_data);
        encoded_data += sizeof(int);
        
        memcpy(result_frame->encoded_data, encoded_data, data_size - sizeof(int));
        result_frame->encoded_size = data_size - sizeof(int);
        
        // Acknowledge encoded data
        g.recv_seg->packet.cmd = CMD_INVALID;
        SCIFlush(NULL, NO_FLAGS);
        g.server_send->packet.cmd = CMD_ENCODED_DATA_ACK;
        SCIFlush(NULL, NO_FLAGS);

        pthread_cond_signal(&g.pipeline.slot_available);
        pthread_cond_signal(&g.pipeline.frame_ready_to_send); // Signal read/send thread
        
        pthread_mutex_lock(&g.pipeline.mutex);
        result_frame->result_received = true;
        g.pipeline.frames_received++;
        pthread_cond_signal(&g.pipeline.result_ready_to_write);
        pthread_mutex_unlock(&g.pipeline.mutex);
        
        printf("ReceiveWrite: Result for frame %d received\n", result_frame->frame_number);
        
        // Step 2: Write frames to output (process all available frames)
        while (true) {
            pthread_mutex_lock(&g.pipeline.mutex);
            
            // Look for frame ready to write (in sequential order)
            frame_buffer_t *frame_to_write = NULL;
            for (int i = 0; i < MAX_FRAMES; i++) {
                if (g.pipeline.frames[i].result_received && 
                    g.pipeline.frames[i].frame_number == g.pipeline.next_write_frame) {
                    frame_to_write = &g.pipeline.frames[i];
                    break;
                }
            }
            
            if (!frame_to_write) {
                // Check if we're done
                if (g.pipeline.finished_reading && 
                    g.pipeline.frames_written >= g.pipeline.frames_read) {
                    pthread_mutex_unlock(&g.pipeline.mutex);
                    printf("ReceiveWrite: All frames written\n");
                    return NULL;
                }
                
                // No frame ready to write, go back to receiving
                pthread_mutex_unlock(&g.pipeline.mutex);
                break;
            }
            
            pthread_mutex_unlock(&g.pipeline.mutex);
            
            printf("ReceiveWrite: Writing frame %d\n", frame_to_write->frame_number);
            
            // Reconstruct frame data from encoded data
            char *ptr = frame_to_write->encoded_data;
            
            // Copy DCT coefficients
            size_t ydct_size = g.cm->ypw * g.cm->yph * sizeof(int16_t);
            memcpy(g.cm->curframe->residuals->Ydct, ptr, ydct_size);
            ptr += ydct_size;
            
            size_t udct_size = g.cm->upw * g.cm->uph * sizeof(int16_t);
            memcpy(g.cm->curframe->residuals->Udct, ptr, udct_size);
            ptr += udct_size;
            
            size_t vdct_size = g.cm->vpw * g.cm->vph * sizeof(int16_t);
            memcpy(g.cm->curframe->residuals->Vdct, ptr, vdct_size);
            ptr += vdct_size;
            
            // Copy macroblock data
            size_t mby_size = g.cm->mb_rows * g.cm->mb_cols * sizeof(struct macroblock);
            memcpy(g.cm->curframe->mbs[Y_COMPONENT], ptr, mby_size);
            ptr += mby_size;
            
            size_t mbu_size = (g.cm->mb_rows/2) * (g.cm->mb_cols/2) * sizeof(struct macroblock);
            memcpy(g.cm->curframe->mbs[U_COMPONENT], ptr, mbu_size);
            ptr += mbu_size;
            
            size_t mbv_size = (g.cm->mb_rows/2) * (g.cm->mb_cols/2) * sizeof(struct macroblock);
            memcpy(g.cm->curframe->mbs[V_COMPONENT], ptr, mbv_size);
            
            // Set keyframe flag
            g.cm->curframe->keyframe = frame_to_write->keyframe;
            
            // Write frame to output
            write_frame(g.cm);
            g.cm->framenum++;
            g.cm->frames_since_keyframe++;
            if (g.cm->curframe->keyframe) {
                g.cm->frames_since_keyframe = 0;
            }
            
            printf("ReceiveWrite: Frame %d written successfully\n", frame_to_write->frame_number);
            
            // Cleanup and free slot
            pthread_mutex_lock(&g.pipeline.mutex);
            if (frame_to_write->yuv_data) {
                free(frame_to_write->yuv_data->Y);
                free(frame_to_write->yuv_data->U);
                free(frame_to_write->yuv_data->V);
                free(frame_to_write->yuv_data);
                frame_to_write->yuv_data = NULL;
            }
            frame_to_write->ready_to_send = false;
            frame_to_write->result_received = false;
            frame_to_write->frame_number = -1;
            g.pipeline.frames_written++;
            g.pipeline.next_write_frame++;
            
            pthread_mutex_unlock(&g.pipeline.mutex);
            
            // Continue processing more frames if available
        }
    }
    
    printf("ReceiveWrite: Thread finished\n");
    return NULL;
}

// Send dimensions to server
bool send_dimensions() {
    struct dimensions_data dim_data = {g.width, g.height};
    
    if (!send_dma_data(&dim_data, sizeof(dim_data))) {
        printf("Failed to send dimensions\n");
        return false;
    }
    
    SCIFlush(NULL, NO_FLAGS);
    g.server_recv->packet.cmd = CMD_DIMENSIONS;
    SCIFlush(NULL, NO_FLAGS);
    
    if (!wait_for_command(CMD_DIMENSIONS_ACK, g.recv_seg, TIMEOUT_SECONDS)) {
        printf("Timeout waiting for dimensions ACK\n");
        return false;
    }
    
    g.recv_seg->packet.cmd = CMD_INVALID;
    printf("Dimensions acknowledged\n");
    return true;
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

// Initialize SISCI
bool init_sisci() {
    sci_error_t error;
    unsigned int localAdapterNo = 0;
    
    SCIInitialize(NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    sci_desc_t sd;
    SCIOpen(&sd, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    // Create segments
    SCICreateSegment(sd, &g.send_segment, SEGMENT_CLIENT_SEND, sizeof(struct send_segment),
                     NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    SCICreateSegment(sd, &g.recv_segment, SEGMENT_CLIENT_RECV, sizeof(struct recv_segment),
                     NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    // Prepare segments
    SCIPrepareSegment(g.send_segment, localAdapterNo, NO_FLAGS, &error);
    SCIPrepareSegment(g.recv_segment, localAdapterNo, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    // Create DMA queue
    SCICreateDMAQueue(sd, &g.dma_queue, localAdapterNo, 1, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    // Map local segments
    sci_map_t send_map, recv_map;
    g.send_seg = (volatile struct send_segment *)SCIMapLocalSegment(
        g.send_segment, &send_map, 0, sizeof(struct send_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    g.recv_seg = (volatile struct recv_segment *)SCIMapLocalSegment(
        g.recv_segment, &recv_map, 0, sizeof(struct recv_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    g.send_seg->packet.cmd = CMD_INVALID;
    g.recv_seg->packet.cmd = CMD_INVALID;
    
    // Make segments available
    SCISetSegmentAvailable(g.send_segment, localAdapterNo, NO_FLAGS, &error);
    SCISetSegmentAvailable(g.recv_segment, localAdapterNo, NO_FLAGS, &error);
    
    // Connect to server segments
    do {
        SCIConnectSegment(sd, &g.remote_server_recv, g.remote_node, SEGMENT_SERVER_RECV, 
                         localAdapterNo, NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    } while (error != SCI_ERR_OK);
    
    do {
        SCIConnectSegment(sd, &g.remote_server_send, g.remote_node, SEGMENT_SERVER_SEND,
                         localAdapterNo, NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    } while (error != SCI_ERR_OK);
    
    // Map server segments
    sci_map_t server_recv_map, server_send_map;
    g.server_recv = (volatile struct recv_segment *)SCIMapRemoteSegment(
        g.remote_server_recv, &server_recv_map, 0, sizeof(struct recv_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    g.server_send = (volatile struct send_segment *)SCIMapRemoteSegment(
        g.remote_server_send, &server_send_map, 0, sizeof(struct send_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    printf("SISCI initialized and connected\n");
    return true;
}

// Print help
static void print_help() {
    printf("Usage: ./c63client -r nodeid [options] input_file\n");
    printf("Options:\n");
    printf("  -r   Node id of server\n");
    printf("  -h   Height of images\n");
    printf("  -w   Width of images\n");
    printf("  -o   Output file (.c63)\n");
    printf("  -f   Limit number of frames\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
    int c;
    
    if (argc == 1) print_help();
    
    while ((c = getopt(argc, argv, "r:h:w:o:f:")) != -1) {
        switch (c) {
            case 'r': g.remote_node = atoi(optarg); break;
            case 'h': g.height = atoi(optarg); break;
            case 'w': g.width = atoi(optarg); break;
            case 'o': g.output_file = optarg; break;
            case 'f': g.limit_frames = atoi(optarg); break;
            default: print_help(); break;
        }
    }
    
    if (optind >= argc || g.remote_node == 0) {
        fprintf(stderr, "Missing input file or remote node\n");
        exit(EXIT_FAILURE);
    }
    
    g.input_file = argv[optind];
    
    // Open files
    g.outfile = fopen(g.output_file, "wb");
    if (!g.outfile) {
        perror("Output file");
        exit(EXIT_FAILURE);
    }
    
    g.infile = fopen(g.input_file, "rb");
    if (!g.infile) {
        perror("Input file");
        exit(EXIT_FAILURE);
    }
    
    // Initialize encoder and pipeline
    g.cm = init_encoder(g.width, g.height);
    g.cm->e_ctx.fp = g.outfile;
    g.cm->curframe = create_frame(g.cm, NULL);
    g.cm->refframe = create_frame(g.cm, NULL);
    
    pipeline_init(&g.pipeline);
    
    if (!init_sisci()) {
        fprintf(stderr, "SISCI initialization failed\n");
        exit(EXIT_FAILURE);
    }
    
    if (!send_dimensions()) {
        fprintf(stderr, "Failed to send dimensions\n");
        exit(EXIT_FAILURE);
    }
    
    // Start the two threads
    pthread_t read_send_tid, receive_write_tid;
    
    pthread_create(&read_send_tid, NULL, read_send_thread, NULL);
    pthread_create(&receive_write_tid, NULL, receive_write_thread, NULL);
    
    // Wait for both threads to complete
    pthread_join(read_send_tid, NULL);
    pthread_join(receive_write_tid, NULL);
    
    // Send quit command to server
    SCIFlush(NULL, NO_FLAGS);
    g.server_recv->packet.cmd = CMD_QUIT;
    SCIFlush(NULL, NO_FLAGS);
    
    // Print final statistics
    printf("Client finished:\n");
    printf("  Frames read: %d\n", g.pipeline.frames_read);
    printf("  Frames sent: %d\n", g.pipeline.frames_sent);
    printf("  Frames received: %d\n", g.pipeline.frames_received);
    printf("  Frames written: %d\n", g.pipeline.frames_written);
    
    // Cleanup
    pipeline_destroy(&g.pipeline);
    if (g.cm) {
        destroy_frame(g.cm->refframe);
        destroy_frame(g.cm->curframe);
        free(g.cm);
    }
    fclose(g.outfile);
    fclose(g.infile);
    
    printf("Client finished successfully\n");
    return 0;
}
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

/* Frame buffer for 3-frame pipeline */
typedef struct {
    yuv_t *yuv_data;
    char *encoded_data;
    size_t encoded_size;
    int frame_number;
    bool ready_to_send;
    bool result_received;
    bool keyframe;
    bool being_sent;
} frame_buffer_t;

/* Pipeline state for 3-frame architecture */
typedef struct {
    frame_buffer_t frames[MAX_FRAMES];
    int next_send_frame;
    int next_write_frame;
    
    int frames_read;
    int frames_sent;
    int frames_received;
    int frames_written;
    
    bool finished_reading;
    bool quit_requested;
    
    pthread_mutex_t mutex;
    pthread_cond_t slot_available;
} pipeline_t;

/* Global state */
static struct {
    char *input_file, *output_file;
    uint32_t remote_node, width, height;
    int limit_frames;
    FILE *infile, *outfile;
    pipeline_t pipeline;
    struct c63_common *cm;
    
    // SISCI
    volatile struct send_segment *send_seg;
    volatile struct recv_segment *recv_seg;
    volatile struct recv_segment *server_recv;
    volatile struct send_segment *server_send;
    sci_dma_queue_t dma_queue;
    sci_local_segment_t send_segment, recv_segment;
    sci_remote_segment_t remote_server_recv, remote_server_send;
} g;

int frames_in_flight = 0;

/* Initialize pipeline */
void pipeline_init(pipeline_t *p) {
    memset(p, 0, sizeof(pipeline_t));
    
    for (int i = 0; i < MAX_FRAMES; i++) {
        p->frames[i].encoded_data = (char*)malloc(MESSAGE_SIZE);
        p->frames[i].yuv_data = NULL;
        p->frames[i].ready_to_send = false;
        p->frames[i].result_received = false;
        p->frames[i].frame_number = -1;
        p->frames[i].being_sent = false;
    }
    
    pthread_mutex_init(&p->mutex, NULL);
    pthread_cond_init(&p->slot_available, NULL);
}

/* Cleanup pipeline and destroy the pipeline */
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
    pthread_cond_destroy(&p->slot_available);
}

/* Read YUV frame from file */
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

/* Timeout protection, waits for commands to arrive from the server */
bool wait_for_command(uint32_t expected_cmd, volatile struct recv_segment *seg, int timeout) {
    time_t start = time(NULL);
    while (seg->packet.cmd != expected_cmd) {
        if (time(NULL) - start > timeout) return false;
        usleep(1000);
    }
    return true;
}

/* Reusable function for transferring data to the server, from the client send segment to server receive segment */
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

/* Thread 1
    Handles the reading and sending to the server receive segment:
    - First checks the exit conditions
    - Finds available slot for the next frame / waits until the write thread makes one available
    - Reads frame from file and puts it in open slot in pipeline
    - Checks if frames in flight is < 3
    - Finds the frame that will be sent
    - Package the frame and send it with DMA to receiving server segment with ack
    When the sequence is finished the thread keeps reading and sending whenever there is a open slot-
    making it a 3 frame in flight pipeline
*/
void *read_send_thread(void *arg) {
    printf("ReadSend: Thread started\n");
    
    while (true) {
        pthread_mutex_lock(&g.pipeline.mutex);

        if (g.pipeline.quit_requested || 
            (g.limit_frames && g.pipeline.frames_read >= g.limit_frames)) {
            g.pipeline.finished_reading = true;
            pthread_mutex_unlock(&g.pipeline.mutex);
            break;
        }
        
        int read_slot = -1;
        for (int i = 0; i < MAX_FRAMES; i++) {
            if (g.pipeline.frames[i].frame_number == -1) {
                read_slot = i;
                break;
            }
        }
        
        if (read_slot == -1) {
            pthread_cond_wait(&g.pipeline.slot_available, &g.pipeline.mutex);
            pthread_mutex_unlock(&g.pipeline.mutex);
            continue;
        }
        
        pthread_mutex_unlock(&g.pipeline.mutex);
        
        yuv_t *image = read_yuv_frame();
        if (!image) {
            pthread_mutex_lock(&g.pipeline.mutex);
            g.pipeline.finished_reading = true;
            pthread_mutex_unlock(&g.pipeline.mutex);
            printf("ReadSend: End of file reached\n");
            break;
        }
        
        pthread_mutex_lock(&g.pipeline.mutex);
        
        frame_buffer_t *frame = &g.pipeline.frames[read_slot];
        frame->yuv_data = image;
        frame->frame_number = g.pipeline.frames_read++;
        frame->ready_to_send = true;
        frame->result_received = false;
        frame->being_sent = false;
        
        printf("ReadSend: Read frame %d into slot %d\n", frame->frame_number, read_slot);
        pthread_mutex_unlock(&g.pipeline.mutex);

        while (true) {
            pthread_mutex_lock(&g.pipeline.mutex);
            
            int frames_in_flight = g.pipeline.frames_sent - g.pipeline.frames_received;
            if (frames_in_flight >= MAX_FRAMES) {
                pthread_mutex_unlock(&g.pipeline.mutex);
                break;
            }
            
            frame_buffer_t *send_frame = NULL;
            for (int i = 0; i < MAX_FRAMES; i++) {
                if (g.pipeline.frames[i].ready_to_send && 
                    !g.pipeline.frames[i].result_received &&
                    !g.pipeline.frames[i].being_sent &&
                    g.pipeline.frames[i].frame_number == g.pipeline.next_send_frame) {
                    send_frame = &g.pipeline.frames[i];
                    send_frame->being_sent = true;
                    break;
                }
            }
            
            if (!send_frame) {
                if (g.pipeline.finished_reading && g.pipeline.frames_sent >= g.pipeline.frames_read) {
                    pthread_mutex_unlock(&g.pipeline.mutex);
                    printf("ReadSend: All frames sent\n");
                    return NULL;
                }
                pthread_mutex_unlock(&g.pipeline.mutex);
                break;
            }
            
            yuv_t *send_image = send_frame->yuv_data;
            int frame_number = send_frame->frame_number;

            if (!send_image || !send_image->Y || !send_image->U || !send_image->V) {
                printf("ReadSend: Invalid frame data for frame %d\n", frame_number);
                send_frame->being_sent = false;
                pthread_mutex_unlock(&g.pipeline.mutex);
                continue;
            }
            
            pthread_mutex_unlock(&g.pipeline.mutex);
            
            printf("ReadSend: Sending frame %d (frames in flight: %d)\n", frame_number, frames_in_flight);
            
            size_t y_size = g.width * g.height;
            size_t u_size = (g.width * g.height) / 4;
            size_t v_size = (g.width * g.height) / 4;
            size_t total_size = y_size + u_size + v_size;
            
            memcpy((void*)g.send_seg->message_buffer, send_image->Y, y_size);
            memcpy((void*)(g.send_seg->message_buffer + y_size), send_image->U, u_size);
            memcpy((void*)(g.send_seg->message_buffer + y_size + u_size), send_image->V, v_size);
            
            if (!send_dma_data((void*)g.send_seg->message_buffer, total_size)) {
                printf("ReadSend: DMA transfer failed for frame %d\n", frame_number);
                pthread_mutex_lock(&g.pipeline.mutex);
                send_frame->being_sent = false;
                pthread_mutex_unlock(&g.pipeline.mutex);
                continue;
            }

            SCIFlush(NULL, NO_FLAGS);
            g.server_recv->packet.cmd = CMD_YUV_DATA;
            g.server_recv->packet.data_size = total_size;
            SCIFlush(NULL, NO_FLAGS);

            if (!wait_for_command(CMD_YUV_DATA_ACK, g.recv_seg, TIMEOUT_SECONDS)) {
                printf("ReadSend: Timeout waiting for YUV ACK for frame %d\n", frame_number);
                pthread_mutex_lock(&g.pipeline.mutex);
                send_frame->being_sent = false;
                pthread_mutex_unlock(&g.pipeline.mutex);
                continue;
            }
            g.recv_seg->packet.cmd = CMD_INVALID;
            
            pthread_mutex_lock(&g.pipeline.mutex);
            g.pipeline.frames_sent++;
            g.pipeline.next_send_frame++;
            send_frame->being_sent = false;
            pthread_mutex_unlock(&g.pipeline.mutex);
            
            printf("ReadSend: Frame %d sent successfully\n", frame_number);
        }
    }
    
    printf("ReadSend: Thread finished\n");
    return NULL;
}

/* Thread 2 
    Receive encoded frame from server and write to output:
    - Waits for encoded frame from server
    - Find the correct encoded frame
    - Extract the encoded data
    - Send ack to server and update the pipeline (signal the client that open slot is available)
    - Reconstruct and write frame to output
    - Cleanup
    The general flow of the thread is receiving the encoded frame from the server, set an available pipeline slot, then write it to output.
    Once the frame has been written the thread waits for next fram.
*/
void *receive_write_thread(void *arg) {
    printf("ReceiveWrite: Thread started\n");
    
    while (true) {
        if (!wait_for_command(CMD_ENCODED_DATA, g.recv_seg, TIMEOUT_SECONDS * 4)) {
            pthread_mutex_lock(&g.pipeline.mutex);
            if (g.pipeline.finished_reading && 
                g.pipeline.frames_received >= g.pipeline.frames_sent) {
                pthread_mutex_unlock(&g.pipeline.mutex);
                printf("ReceiveWrite: All results received\n");
                break;
            }
            pthread_mutex_unlock(&g.pipeline.mutex);
            /* 10ms */
            usleep(10000);
            continue;
        }
        
        size_t data_size = g.recv_seg->packet.data_size;
        char *encoded_data = (char*)g.recv_seg->message_buffer;
        
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
        
        if (result_frame->frame_number != g.pipeline.next_write_frame) {
            printf("ReceiveWrite: Warning! Expected frame %d but got %d\n", 
                   g.pipeline.next_write_frame, result_frame->frame_number);
            pthread_mutex_unlock(&g.pipeline.mutex);
            g.recv_seg->packet.cmd = CMD_INVALID;
            continue;
        }
        
        while (result_frame->being_sent) {
            pthread_mutex_unlock(&g.pipeline.mutex);
            usleep(1000);
            pthread_mutex_lock(&g.pipeline.mutex);
        }
        
        pthread_mutex_unlock(&g.pipeline.mutex);
        
        result_frame->keyframe = *((int*)encoded_data);
        encoded_data += sizeof(int);
        
        memcpy(result_frame->encoded_data, encoded_data, data_size - sizeof(int));
        result_frame->encoded_size = data_size - sizeof(int);
        
        g.recv_seg->packet.cmd = CMD_INVALID;
        SCIFlush(NULL, NO_FLAGS);
        g.server_send->packet.cmd = CMD_ENCODED_DATA_ACK;
        SCIFlush(NULL, NO_FLAGS);
        
        printf("ReceiveWrite: Writing frame %d\n", result_frame->frame_number);
        
        char *ptr = result_frame->encoded_data;
        
        size_t ydct_size = g.cm->ypw * g.cm->yph * sizeof(int16_t);
        memcpy(g.cm->curframe->residuals->Ydct, ptr, ydct_size);
        ptr += ydct_size;
        
        size_t udct_size = g.cm->upw * g.cm->uph * sizeof(int16_t);
        memcpy(g.cm->curframe->residuals->Udct, ptr, udct_size);
        ptr += udct_size;
        
        size_t vdct_size = g.cm->vpw * g.cm->vph * sizeof(int16_t);
        memcpy(g.cm->curframe->residuals->Vdct, ptr, vdct_size);
        ptr += vdct_size;
        
        size_t mby_size = g.cm->mb_rows * g.cm->mb_cols * sizeof(struct macroblock);
        memcpy(g.cm->curframe->mbs[Y_COMPONENT], ptr, mby_size);
        ptr += mby_size;
        
        size_t mbu_size = (g.cm->mb_rows/2) * (g.cm->mb_cols/2) * sizeof(struct macroblock);
        memcpy(g.cm->curframe->mbs[U_COMPONENT], ptr, mbu_size);
        ptr += mbu_size;
        
        size_t mbv_size = (g.cm->mb_rows/2) * (g.cm->mb_cols/2) * sizeof(struct macroblock);
        memcpy(g.cm->curframe->mbs[V_COMPONENT], ptr, mbv_size);
        
        g.cm->curframe->keyframe = result_frame->keyframe;
        
        write_frame(g.cm);
        g.cm->framenum++;
        g.cm->frames_since_keyframe++;
        if (g.cm->curframe->keyframe) {
            g.cm->frames_since_keyframe = 0;
        }
        
        printf("ReceiveWrite: Frame %d written successfully\n", result_frame->frame_number);
        
        pthread_mutex_lock(&g.pipeline.mutex);
        
        while (result_frame->being_sent) {
            pthread_mutex_unlock(&g.pipeline.mutex);
            usleep(1000);
            pthread_mutex_lock(&g.pipeline.mutex);
        }
        
        if (result_frame->yuv_data) {
            free(result_frame->yuv_data->Y);
            free(result_frame->yuv_data->U);
            free(result_frame->yuv_data->V);
            free(result_frame->yuv_data);
            result_frame->yuv_data = NULL;
        }
        result_frame->ready_to_send = false;
        result_frame->result_received = false;
        result_frame->frame_number = -1;
        result_frame->being_sent = false;
        
        g.pipeline.frames_received++;
        g.pipeline.frames_written++;
        g.pipeline.next_write_frame++;
        
        pthread_cond_signal(&g.pipeline.slot_available);
        
        if (g.pipeline.finished_reading && 
            g.pipeline.frames_written >= g.pipeline.frames_read) {
            pthread_mutex_unlock(&g.pipeline.mutex);
            printf("ReceiveWrite: All frames written\n");
            break;
        }
        
        pthread_mutex_unlock(&g.pipeline.mutex);
    }
    
    printf("ReceiveWrite: Thread finished\n");
    return NULL;
}

/* Send dimensions to server and wait for ack */
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

/* Initialize encoder */
struct c63_common *init_cm(int width, int height) {
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

/* Initialize SISCI */
bool init_sisci() {
    sci_error_t error;
    unsigned int localAdapterNo = 0;
    
    SCIInitialize(NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    sci_desc_t sd;
    SCIOpen(&sd, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    /* Create segments */
    SCICreateSegment(sd, &g.send_segment, SEGMENT_CLIENT_SEND, sizeof(struct send_segment),
                     NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    SCICreateSegment(sd, &g.recv_segment, SEGMENT_CLIENT_RECV, sizeof(struct recv_segment),
                     NO_CALLBACK, NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    /* Prepare segments */
    SCIPrepareSegment(g.send_segment, localAdapterNo, NO_FLAGS, &error);
    SCIPrepareSegment(g.recv_segment, localAdapterNo, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    /* Create DMA queue */
    SCICreateDMAQueue(sd, &g.dma_queue, localAdapterNo, 1, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    /* Map local segments */
    sci_map_t send_map, recv_map;
    g.send_seg = (volatile struct send_segment *)SCIMapLocalSegment(
        g.send_segment, &send_map, 0, sizeof(struct send_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    g.recv_seg = (volatile struct recv_segment *)SCIMapLocalSegment(
        g.recv_segment, &recv_map, 0, sizeof(struct recv_segment), NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) return false;
    
    g.send_seg->packet.cmd = CMD_INVALID;
    g.recv_seg->packet.cmd = CMD_INVALID;
    
    /* Make segments available */
    SCISetSegmentAvailable(g.send_segment, localAdapterNo, NO_FLAGS, &error);
    SCISetSegmentAvailable(g.recv_segment, localAdapterNo, NO_FLAGS, &error);
    
    /* Connect to server segments */
    do {
        SCIConnectSegment(sd, &g.remote_server_recv, g.remote_node, SEGMENT_SERVER_RECV, 
                         localAdapterNo, NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    } while (error != SCI_ERR_OK);
    
    do {
        SCIConnectSegment(sd, &g.remote_server_send, g.remote_node, SEGMENT_SERVER_SEND,
                         localAdapterNo, NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    } while (error != SCI_ERR_OK);
    
    /* Map server segments */
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

/* Print help */
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
    
    /* Open files */
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
    
    /* Initialize encoder and pipeline */
    g.cm = init_cm(g.width, g.height);
    g.cm->e_ctx.fp = g.outfile;
    g.cm->curframe = create_frame(g.cm, NULL);
    g.cm->refframe = create_frame(g.cm, NULL);
    
    pipeline_init(&g.pipeline);
    
    if (!init_sisci()) {
        fprintf(stderr, "SISCI initialization failed\n");
        exit(EXIT_FAILURE);
    }
    
    /* sends dimensions at the start once */
    if (!send_dimensions()) {
        fprintf(stderr, "Failed to send dimensions\n");
        exit(EXIT_FAILURE);
    }
    
    /* Start the two threads */
    pthread_t read_send_tid, receive_write_tid;
    
    pthread_create(&read_send_tid, NULL, read_send_thread, NULL);
    pthread_create(&receive_write_tid, NULL, receive_write_thread, NULL);
    
    /* Wait for both threads to complete */
    pthread_join(read_send_tid, NULL);
    pthread_join(receive_write_tid, NULL);
    
    /* Send quit command to server */
    SCIFlush(NULL, NO_FLAGS);
    g.server_recv->packet.cmd = CMD_QUIT;
    SCIFlush(NULL, NO_FLAGS);
    
    /* print statistics */
    printf("Client finished:\n");
    printf("  Frames read: %d\n", g.pipeline.frames_read);
    printf("  Frames sent: %d\n", g.pipeline.frames_sent);
    printf("  Frames received: %d\n", g.pipeline.frames_received);
    printf("  Frames written: %d\n", g.pipeline.frames_written);
    
    /* Cleanup */
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
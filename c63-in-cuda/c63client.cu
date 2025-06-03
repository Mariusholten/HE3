#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sisci_error.h>
#include <sisci_api.h>

#include "c63.h"
#include "c63_write.h"
#include "common.h"
#include "tables.h"
#include <time.h>

static char *output_file, *input_file;
FILE *outfile;

static uint32_t remote_node = 0;
static int limit_numframes = 0;

static uint32_t width;
static uint32_t height;

/* getopt */
extern int optind;
extern char *optarg;

/* Read planar YUV frames with 4:2:0 chroma sub-sampling */
static yuv_t *read_yuv(FILE *file, struct c63_common *cm)
{
    size_t len = 0;
    yuv_t *image = (yuv_t *)malloc(sizeof(*image));

    /* Read Y. The size of Y is the same as the size of the image. */
    image->Y = (uint8_t *)calloc(1, cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT]);
    len += fread(image->Y, 1, width * height, file);

    /* Read U. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y */
    image->U = (uint8_t *)calloc(1, cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT]);
    len += fread(image->U, 1, (width * height) / 4, file);

    /* Read V. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y. */
    image->V = (uint8_t *)calloc(1, cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT]);
    len += fread(image->V, 1, (width * height) / 4, file);

    if (ferror(file))
    {
        perror("ferror");
        exit(EXIT_FAILURE);
    }

    if (feof(file))
    {
        free(image->Y);
        free(image->U);
        free(image->V);
        free(image);
        return NULL;
    }
    else if (len != width * height * 1.5)
    {
        fprintf(stderr, "Reached end of file, but incorrect bytes read.\n");
        fprintf(stderr, "Wrong input? (height: %d width: %d)\n", height, width);
        free(image->Y);
        free(image->U);
        free(image->V);
        free(image);
        return NULL;
    }

    return image;
}

struct c63_common *
init_c63_enc( int width, int height )
{
    int i;

    /* calloc() sets allocated memory to zero */
    c63_common *cm =
        ( c63_common * ) calloc( 1, sizeof( struct c63_common ) );

    cm->width = width;
    cm->height = height;

    cm->padw[Y_COMPONENT] = cm->ypw =( uint32_t ) ( ceil( width / 16.0f ) * 16 );
    cm->padh[Y_COMPONENT] = cm->yph =( uint32_t ) ( ceil( height / 16.0f ) * 16 );
    cm->padw[U_COMPONENT] = cm->upw =( uint32_t ) ( ceil( width * UX / ( YX * 8.0f ) ) * 8 );
    cm->padh[U_COMPONENT] = cm->uph =( uint32_t ) ( ceil( height * UY / ( YY * 8.0f ) ) * 8 );
    cm->padw[V_COMPONENT] = cm->vpw =( uint32_t ) ( ceil( width * VX / ( YX * 8.0f ) ) * 8 );
    cm->padh[V_COMPONENT] = cm->vph =( uint32_t ) ( ceil( height * VY / ( YY * 8.0f ) ) * 8 );

    cm->mb_cols = cm->ypw / 8;
    cm->mb_rows = cm->yph / 8;

    /* Quality parameters -- Home exam deliveries should have original values,
       i.e., quantization factor should be 25, search range should be 16, and the
       keyframe interval should be 100. */
    cm->qp = 25;                // Constant quantization factor. Range: [1..50]
    cm->me_search_range = 16;   // Pixels in every direction
    cm->keyframe_interval = 100;        // Distance between keyframes

    /* Initialize quantization tables */
    for ( i = 0; i < 64; ++i )
    {
        cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / ( cm->qp / 10.0 );
        cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / ( cm->qp / 10.0 );
        cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / ( cm->qp / 10.0 );
    }

    return cm;
}

void
free_c63_enc( struct c63_common *cm )
{
    destroy_frame( cm->curframe );
    free( cm );
}

// Main loop for client - Handles transfers and acknowledgements and writing
// Add these lines at the top after the existing includes in paste.txt

// Timing structures and functions for benchmarking
struct frame_timing {
    struct timespec frame_start, yuv_start, yuv_end, encode_start, encode_end, result_start, result_end, frame_end;
    long yuv_us, encode_us, result_us, total_us;
};

struct benchmark_stats {
    long frames, total_yuv_us, total_encode_us, total_result_us, total_roundtrip_us;
    long min_yuv_us, max_yuv_us, min_encode_us, max_encode_us, min_result_us, max_result_us, min_total_us, max_total_us;
};

static inline void get_time(struct timespec *ts) { clock_gettime(CLOCK_MONOTONIC, ts); }
static inline long time_diff_us(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000000L + (end->tv_nsec - start->tv_nsec) / 1000L;
}

// Helper function to format time with appropriate units
void format_time(long us, char *buffer, size_t size) {
    if (us < 1000) {
        snprintf(buffer, size, "%ld μs", us);
    } else if (us < 1000000) {
        snprintf(buffer, size, "%.1f ms", us / 1000.0);
    } else {
        snprintf(buffer, size, "%.2f s", us / 1000000.0);
    }
}

void update_benchmark_stats(struct benchmark_stats *stats, struct frame_timing *timing) {
    stats->frames++;
    stats->total_yuv_us += timing->yuv_us;
    stats->total_encode_us += timing->encode_us;
    stats->total_result_us += timing->result_us;
    stats->total_roundtrip_us += timing->total_us;
    
    if (stats->frames == 1) {
        stats->min_yuv_us = stats->max_yuv_us = timing->yuv_us;
        stats->min_encode_us = stats->max_encode_us = timing->encode_us;
        stats->min_result_us = stats->max_result_us = timing->result_us;
        stats->min_total_us = stats->max_total_us = timing->total_us;
    } else {
        if (timing->yuv_us < stats->min_yuv_us) stats->min_yuv_us = timing->yuv_us;
        if (timing->yuv_us > stats->max_yuv_us) stats->max_yuv_us = timing->yuv_us;
        if (timing->encode_us < stats->min_encode_us) stats->min_encode_us = timing->encode_us;
        if (timing->encode_us > stats->max_encode_us) stats->max_encode_us = timing->encode_us;
        if (timing->result_us < stats->min_result_us) stats->min_result_us = timing->result_us;
        if (timing->result_us > stats->max_result_us) stats->max_result_us = timing->result_us;
        if (timing->total_us < stats->min_total_us) stats->min_total_us = timing->total_us;
        if (timing->total_us > stats->max_total_us) stats->max_total_us = timing->total_us;
    }
}

void print_benchmark_results(struct benchmark_stats *stats, uint32_t width, uint32_t height) {
    if (stats->frames == 0) return;
    
    char avg_yuv[32], min_yuv[32], max_yuv[32];
    char avg_encode[32], min_encode[32], max_encode[32];
    char avg_result[32], min_result[32], max_result[32];
    char avg_total[32], min_total[32], max_total[32];
    
    format_time(stats->total_yuv_us / stats->frames, avg_yuv, sizeof(avg_yuv));
    format_time(stats->min_yuv_us, min_yuv, sizeof(min_yuv));
    format_time(stats->max_yuv_us, max_yuv, sizeof(max_yuv));
    
    format_time(stats->total_encode_us / stats->frames, avg_encode, sizeof(avg_encode));
    format_time(stats->min_encode_us, min_encode, sizeof(min_encode));
    format_time(stats->max_encode_us, max_encode, sizeof(max_encode));
    
    format_time(stats->total_result_us / stats->frames, avg_result, sizeof(avg_result));
    format_time(stats->min_result_us, min_result, sizeof(min_result));
    format_time(stats->max_result_us, max_result, sizeof(max_result));
    
    format_time(stats->total_roundtrip_us / stats->frames, avg_total, sizeof(avg_total));
    format_time(stats->min_total_us, min_total, sizeof(min_total));
    format_time(stats->max_total_us, max_total, sizeof(max_total));
    
    printf("\n═══ BENCHMARK RESULTS (%ld frames) ═══\n", stats->frames);
    printf("YUV Transfer:    Avg:%8s  Min:%8s  Max:%8s\n", avg_yuv, min_yuv, max_yuv);
    printf("Encoding:        Avg:%8s  Min:%8s  Max:%8s\n", avg_encode, min_encode, max_encode);
    printf("Result Transfer: Avg:%8s  Min:%8s  Max:%8s\n", avg_result, min_result, max_result);
    printf("Total Roundtrip: Avg:%8s  Min:%8s  Max:%8s\n", avg_total, min_total, max_total);
    
    double yuv_mbps = (width * height * 1.5 * 8.0) / (stats->total_yuv_us / stats->frames);
    double fps = 1000000.0 / (stats->total_roundtrip_us / stats->frames);
    printf("YUV Transfer Rate: %.1f Mbps\n", yuv_mbps);
    printf("Processing Rate:   %.1f FPS\n", fps);
    printf("═══════════════════════════════════════\n");
}

// REPLACE the entire main_client_loop function with this version:
int main_client_loop(struct c63_common *cm, FILE *infile, int limit_numframes,
                    volatile struct client_segment *local_seg,
                    volatile struct server_segment *remote_seg,
                    sci_dma_queue_t dma_queue,
                    sci_local_segment_t local_segment,
                    sci_remote_segment_t remote_segment) 
{
    yuv_t *image;
    int numframes = 0;
    sci_error_t error;
    struct benchmark_stats stats = {0};
    struct frame_timing timing;
    
    printf("Client: Starting video encoding with benchmarking\n");
    
    // Dimensions exchange (unchanged)
    struct dimensions_data dim_data;
    dim_data.width = width;
    dim_data.height = height;
    
    memcpy((void*)local_seg->message_buffer, &dim_data, sizeof(struct dimensions_data));
    
    SCIStartDmaTransfer(dma_queue, local_segment, remote_segment,
        offsetof(struct client_segment, message_buffer), sizeof(struct dimensions_data),
        offsetof(struct server_segment, message_buffer), NO_CALLBACK, NULL, NO_FLAGS, &error);
                       
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Client: SCIStartDmaTransfer for dimensions failed - Error code 0x%x\n", error);
        return -1;
    }
    
    SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_DIMENSIONS;
    SCIFlush(NULL, NO_FLAGS);
    
    printf("Client: Waiting for server to acknowledge dimensions\n");
    time_t dim_start = time(NULL);
    bool dim_timeout = false;
    
    while (local_seg->packet.cmd != CMD_DIMENSIONS_ACK && !dim_timeout) {
        if (time(NULL) - dim_start > 30) {
            dim_timeout = true;
            fprintf(stderr, "Client: Timeout waiting for dimensions acknowledgment\n");
        }
    }
    
    if (dim_timeout) {
        fprintf(stderr, "Client: Failed to receive dimensions acknowledgment, exiting\n");
        return -1;
    }
    
    printf("Client: Dimensions acknowledged by server\n");
    local_seg->packet.cmd = CMD_INVALID;
    
    // Main processing loop with benchmarking
    while (1) {
        get_time(&timing.frame_start);
        
        image = read_yuv(infile, cm);
        if (!image) {
            printf("Client: End of input file reached\n");
            break;
        }

        printf("Processing frame %d, ", numframes);

        size_t y_size = width * height;
        size_t u_size = (width * height) / 4;
        size_t v_size = (width * height) / 4;
        size_t total_yuv_size = y_size + u_size + v_size;

        if (total_yuv_size > MESSAGE_SIZE) {
            fprintf(stderr, "Client: ERROR - Total YUV frame size (%zu) exceeds message buffer size (%d)\n", total_yuv_size, MESSAGE_SIZE);
            free(image->Y); free(image->U); free(image->V); free(image);
            return -1;
        }

        // Pack YUV data
        memcpy((void*)local_seg->message_buffer, image->Y, y_size);
        memcpy((void*)(local_seg->message_buffer + y_size), image->U, u_size);
        memcpy((void*)(local_seg->message_buffer + y_size + u_size), image->V, v_size);

        // Time YUV transfer
        get_time(&timing.yuv_start);
        SCIStartDmaTransfer(dma_queue, local_segment, remote_segment,
                        offsetof(struct client_segment, message_buffer), total_yuv_size,
                        offsetof(struct server_segment, message_buffer), NO_CALLBACK, NULL, NO_FLAGS, &error);

        if (error != SCI_ERR_OK) {
            fprintf(stderr, "Client: YUV frame DMA transfer failed - Error code 0x%x\n", error);
            free(image->Y); free(image->U); free(image->V); free(image);
            continue;
        }

        SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
        get_time(&timing.yuv_end);

        SCIFlush(NULL, NO_FLAGS);
        remote_seg->packet.cmd = CMD_YUV_DATA;
        remote_seg->packet.data_size = total_yuv_size;
        remote_seg->packet.y_size = y_size;
        remote_seg->packet.u_size = u_size;
        remote_seg->packet.v_size = v_size;
        SCIFlush(NULL, NO_FLAGS);

        // Wait for frame acknowledgment
        time_t frame_start = time(NULL);
        bool frame_timeout = false;
        while (local_seg->packet.cmd != CMD_YUV_DATA_ACK && !frame_timeout) {
            if (time(NULL) - frame_start > 30) {
                frame_timeout = true;
                fprintf(stderr, "Client: Timeout waiting for YUV frame acknowledgment\n");
            }
        }

        if (frame_timeout) {
            free(image->Y); free(image->U); free(image->V); free(image);
            continue;
        }

        local_seg->packet.cmd = CMD_INVALID;
        free(image->Y); free(image->U); free(image->V); free(image);

        // Mark encoding start (server is now processing)
        get_time(&timing.encode_start);

        // Wait for encoded data
        time_t encode_start = time(NULL);
        bool encode_timeout = false;
        
        while (local_seg->packet.cmd != CMD_ENCODED_DATA && !encode_timeout) {
            if (time(NULL) - encode_start > 120) {
                encode_timeout = true;
                fprintf(stderr, "Client: Timeout waiting for encoded data\n");
            }
        }
        
        if (encode_timeout) continue;

        // Mark encoding end and result transfer start
        get_time(&timing.encode_end);
        get_time(&timing.result_start);

        // Process encoded data (existing logic)
        size_t data_size = local_seg->packet.data_size;
        int keyframe = *((int*)local_seg->message_buffer);
        cm->curframe->keyframe = keyframe;
        
        char* encoded_data = (char*)local_seg->message_buffer + sizeof(int);
        
        size_t ydct_size = cm->ypw * cm->yph * sizeof(int16_t);
        memcpy(cm->curframe->residuals->Ydct, encoded_data, ydct_size);
        encoded_data += ydct_size;
        
        size_t udct_size = cm->upw * cm->uph * sizeof(int16_t);
        memcpy(cm->curframe->residuals->Udct, encoded_data, udct_size);
        encoded_data += udct_size;
        
        size_t vdct_size = cm->vpw * cm->vph * sizeof(int16_t);
        memcpy(cm->curframe->residuals->Vdct, encoded_data, vdct_size);
        encoded_data += vdct_size;
        
        size_t mby_size = cm->mb_rows * cm->mb_cols * sizeof(struct macroblock);
        memcpy(cm->curframe->mbs[Y_COMPONENT], encoded_data, mby_size);
        encoded_data += mby_size;
        
        size_t mbu_size = (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(struct macroblock);
        memcpy(cm->curframe->mbs[U_COMPONENT], encoded_data, mbu_size);
        encoded_data += mbu_size;
        
        size_t mbv_size = (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(struct macroblock);
        memcpy(cm->curframe->mbs[V_COMPONENT], encoded_data, mbv_size);

        get_time(&timing.result_end);

        // Acknowledge encoded data
        local_seg->packet.cmd = CMD_INVALID;
        SCIFlush(NULL, NO_FLAGS);
        remote_seg->packet.cmd = CMD_ENCODED_DATA_ACK;
        SCIFlush(NULL, NO_FLAGS);
        
        write_frame(cm);
        get_time(&timing.frame_end);
        
        // Calculate timings
        timing.yuv_us = time_diff_us(&timing.yuv_start, &timing.yuv_end);
        timing.encode_us = time_diff_us(&timing.encode_start, &timing.encode_end);
        timing.result_us = time_diff_us(&timing.result_start, &timing.result_end);
        timing.total_us = time_diff_us(&timing.frame_start, &timing.frame_end);
        
        update_benchmark_stats(&stats, &timing);
        
        char yuv_str[16], encode_str[16], result_str[16], total_str[16];
        format_time(timing.yuv_us, yuv_str, sizeof(yuv_str));
        format_time(timing.encode_us, encode_str, sizeof(encode_str));
        format_time(timing.result_us, result_str, sizeof(result_str));
        format_time(timing.total_us, total_str, sizeof(total_str));
        
        printf("YUV:%s, Encode:%s, Result:%s, Total:%s\n", 
               yuv_str, encode_str, result_str, total_str);
        
        cm->framenum++;
        cm->frames_since_keyframe++;
        if (cm->curframe->keyframe) {
            cm->frames_since_keyframe = 0;
        }
        
        ++numframes;
        
        if (limit_numframes && numframes >= limit_numframes) {
            printf("Client: Reached frame limit (%d frames), stopping\n", limit_numframes);
            break;
        }
    }
    
    // Print final results
    print_benchmark_results(&stats, width, height);
    
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.cmd = CMD_QUIT;
    SCIFlush(NULL, NO_FLAGS);
    
    printf("Client: Finished processing %d frames\n", numframes);
    return numframes;
}
static void print_help()
{
    printf("Usage: ./c63client -r nodeid [options] input_file\n");
    printf("Commandline options:\n");
    printf("  -r                             Node id of server\n");
    printf("  -h                             Height of images to compress\n");
    printf("  -w                             Width of images to compress\n");
    printf("  -o                             Output file (.c63)\n");
    printf("  [-f]                           Limit number of frames to encode\n");
    printf("\n");

    exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    unsigned int localAdapterNo = 0;
    int c;
    sci_error_t error;
    
    if (argc == 1) {
        print_help();
    }

    while ((c = getopt(argc, argv, "r:h:w:o:f:i:")) != -1)
    {
        switch (c)
        {
            case 'r':
                remote_node = atoi(optarg);
                break;
            case 'h':
                height = atoi(optarg);
                break;
            case 'w':
                width = atoi(optarg);
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'f':
                limit_numframes = atoi(optarg);
                break;
            default:
                print_help();
                break;
        }
    }

    if (optind >= argc)
    {
        fprintf(stderr, "Error getting program options, try --help.\n");
        exit(EXIT_FAILURE);
    }

    input_file = argv[optind];

    if (remote_node == 0) {
        fprintf(stderr, "Remote node-id is not specified. Use -r <remote node-id>\n");
        exit(EXIT_FAILURE);
    }

    // Open output file
    outfile = fopen(output_file, "wb");
    if (outfile == NULL)
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    // Initialize encoder
    struct c63_common *cm = init_c63_enc(width, height);
    cm->e_ctx.fp = outfile;

    if (limit_numframes)
    {
        printf("Limited to %d frames.\n", limit_numframes);
    }
    cm->curframe = create_frame(cm, NULL);  // Create with NULL for a placeholder
    cm->refframe = create_frame(cm, NULL); 
    // Open input file
    FILE *infile = fopen(input_file, "rb");
    if (infile == NULL)
    {
        perror("fopen");
        exit(EXIT_FAILURE); 
    }

    // Initialize SISCI
    SCIInitialize(NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIInitialize failed: %s\n", SCIGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Set up SISCI resources
    sci_desc_t sd;
    sci_local_segment_t localSegment;
    sci_remote_segment_t remoteSegment;
    sci_map_t localMap, remoteMap;
    sci_dma_queue_t dmaQueue;
    volatile struct client_segment *client_segment;
    volatile struct server_segment *server_segment;

    // Open virtual device
    SCIOpen(&sd, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIOpen failed - Error code 0x%x\n", error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    // Create local segment
    SCICreateSegment(sd,
                     &localSegment,
                     SEGMENT_CLIENT,
                     sizeof(struct client_segment),
                     NO_CALLBACK,
                     NULL,
                     NO_FLAGS,
                     &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCICreateSegment failed - Error code 0x%x\n", error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    // Prepare segment
    SCIPrepareSegment(localSegment, localAdapterNo, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIPrepareSegment failed - Error code 0x%x\n", error);
        SCIRemoveSegment(localSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    // Create DMA queue
    SCICreateDMAQueue(sd, &dmaQueue, localAdapterNo, 1, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCICreateDMAQueue failed - Error code 0x%x\n", error);
        SCIRemoveSegment(localSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    // Map local segment
    client_segment = (volatile struct client_segment *)SCIMapLocalSegment(
        localSegment, 
        &localMap, 
        0, 
        sizeof(struct client_segment), 
        NULL, 
        NO_FLAGS, 
        &error);
    
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIMapLocalSegment failed - Error code 0x%x\n", error);
        SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
        SCIRemoveSegment(localSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    // Initialize control packet
    client_segment->packet.cmd = CMD_INVALID;
    
    // Make segment available
    SCISetSegmentAvailable(localSegment, localAdapterNo, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCISetSegmentAvailable failed - Error code 0x%x\n", error);
        SCIUnmapSegment(localMap, NO_FLAGS, &error);
        SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
        SCIRemoveSegment(localSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    printf("Client: Connecting to server segment...\n");
    
    // Connect to server segment
    do {
        SCIConnectSegment(sd,
                          &remoteSegment,
                          remote_node,
                          SEGMENT_SERVER,
                          localAdapterNo,
                          NO_CALLBACK,
                          NULL,
                          SCI_INFINITE_TIMEOUT,
                          NO_FLAGS,
                          &error);
    } while (error != SCI_ERR_OK);
    
    printf("Client: Connected to server segment\n");
    
    // Map remote segment
    server_segment = (volatile struct server_segment *)SCIMapRemoteSegment(
        remoteSegment, 
        &remoteMap, 
        0,
        sizeof(struct server_segment),
        NULL, 
        NO_FLAGS, 
        &error);
    
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIMapRemoteSegment failed - Error code 0x%x\n", error);
        printf("Segment handle: %p\n", (void*)remoteSegment);
        printf("Map handle: %p\n", (void*)remoteMap);
        SCIUnmapSegment(remoteMap, NO_FLAGS, &error);
        SCIDisconnectSegment(remoteSegment, NO_FLAGS, &error);
        SCISetSegmentUnavailable(localSegment, localAdapterNo, NO_FLAGS, &error);
        SCIUnmapSegment(localMap, NO_FLAGS, &error);
        SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
        SCIRemoveSegment(localSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }
    
    // Enter main processing loop
    main_client_loop(cm, infile, limit_numframes, client_segment, server_segment, 
                     dmaQueue, localSegment, remoteSegment);
    
    // Clean up resources
    destroy_frame(cm->refframe);
    fclose(outfile);
    fclose(infile);
    free_c63_enc(cm);
    
    SCIUnmapSegment(remoteMap, NO_FLAGS, &error);
    SCIDisconnectSegment(remoteSegment, NO_FLAGS, &error);
    SCISetSegmentUnavailable(localSegment, localAdapterNo, NO_FLAGS, &error);
    SCIUnmapSegment(localMap, NO_FLAGS, &error);
    SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
    SCIRemoveSegment(localSegment, NO_FLAGS, &error);
    SCIClose(sd, NO_FLAGS, &error);
    SCITerminate();
    
    return 0;
}
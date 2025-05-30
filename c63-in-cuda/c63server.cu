#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c63.h"
#include "quantdct.h"
#include "common.h"
#include "me.h"
#include "tables.h"

#include <time.h>
#include <stdbool.h>

#include <cuda_runtime.h>
#include <sisci_error.h>
#include <sisci_api.h>

static uint32_t width = 0;
static uint32_t height = 0;

/* getopt */
extern int optind;
extern char *optarg;

struct c63_common *
init_c63_enc( int width, int height )
{
    int i;

    /* calloc() sets allocated memory to zero */
    c63_common *cm =
        ( c63_common * ) calloc( 1, sizeof( struct c63_common ) );

    cm->width = width;
    cm->height = height;

    cm->padw[Y_COMPONENT] = cm->ypw =
        ( uint32_t ) ( ceil( width / 16.0f ) * 16 );
    cm->padh[Y_COMPONENT] = cm->yph =
        ( uint32_t ) ( ceil( height / 16.0f ) * 16 );
    cm->padw[U_COMPONENT] = cm->upw =
        ( uint32_t ) ( ceil( width * UX / ( YX * 8.0f ) ) * 8 );
    cm->padh[U_COMPONENT] = cm->uph =
        ( uint32_t ) ( ceil( height * UY / ( YY * 8.0f ) ) * 8 );
    cm->padw[V_COMPONENT] = cm->vpw =
        ( uint32_t ) ( ceil( width * VX / ( YX * 8.0f ) ) * 8 );
    cm->padh[V_COMPONENT] = cm->vph =
        ( uint32_t ) ( ceil( height * VY / ( YY * 8.0f ) ) * 8 );

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

void encode_frame(struct c63_common *cm, yuv_t *image, int frame_count,
                 volatile struct server_segment *local_seg,
                 volatile struct client_segment *remote_seg,
                 sci_dma_queue_t dma_queue,
                 sci_local_segment_t local_segment,
                 sci_remote_segment_t remote_segment) 
{
    sci_error_t error;
    
    // Now we have all YUV planes, encode the frame
    printf("Server: Encoding frame %d\n", cm->framenum);
    
    // Encode the frame
    // First, advance frame pointers
    destroy_frame(cm->refframe);
    cm->refframe = cm->curframe;
    cm->curframe = create_frame(cm, image);
    
    // Check if keyframe
    if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval) {
        cm->curframe->keyframe = 1;
        cm->frames_since_keyframe = 0;
        printf("Server: Frame %d is a keyframe\n", cm->framenum);
    } else {
        cm->curframe->keyframe = 0;
        printf("Server: Frame %d is not a keyframe\n", cm->framenum);
    }
    
    // Perform motion estimation if not keyframe
    if (!cm->curframe->keyframe) {
        printf("Server: Starting motion estimation\n");
        c63_motion_estimate(cm);
        printf("Server: Motion estimation complete\n");
        
        printf("Server: Starting motion compensation\n");
        c63_motion_compensate(cm);
        printf("Server: Motion compensation complete\n");
    }
    
    // DCT and Quantization
    printf("Server: Starting DCT/quantization for Y plane\n");
    dct_quantize(image->Y, cm->curframe->predicted->Y, cm->padw[Y_COMPONENT],
               cm->padh[Y_COMPONENT], cm->curframe->residuals->Ydct,
               cm->quanttbl[Y_COMPONENT]);
    printf("Server: Y plane DCT/quantization complete\n");
    
    printf("Server: Starting DCT/quantization for U plane\n");
    dct_quantize(image->U, cm->curframe->predicted->U, cm->padw[U_COMPONENT],
               cm->padh[U_COMPONENT], cm->curframe->residuals->Udct,
               cm->quanttbl[U_COMPONENT]);
    printf("Server: U plane DCT/quantization complete\n");
    
    printf("Server: Starting DCT/quantization for V plane\n");
    dct_quantize(image->V, cm->curframe->predicted->V, cm->padw[V_COMPONENT],
               cm->padh[V_COMPONENT], cm->curframe->residuals->Vdct,
               cm->quanttbl[V_COMPONENT]);
    printf("Server: V plane DCT/quantization complete\n");
    
    // Reconstruct frame for inter-prediction
    printf("Server: Starting dequantization/IDCT for reference frame\n");
    dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y, cm->ypw, cm->yph, cm->curframe->recons->Y, cm->quanttbl[Y_COMPONENT]);
    dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U, cm->upw, cm->uph, cm->curframe->recons->U, cm->quanttbl[U_COMPONENT]);
    dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V, cm->vpw, cm->vph, cm->curframe->recons->V, cm->quanttbl[V_COMPONENT]);
    printf("Server: Dequantization/IDCT complete\n");
    
    // Now send all the encoded data back to the client in one transfer
    printf("Server: Preparing encoded data for transfer\n");
    
    // First, place keyframe flag
    *((int*)local_seg->message_buffer) = cm->curframe->keyframe;
    
    // Calculate offsets for each piece of data
    char* ptr = (char*)local_seg->message_buffer + sizeof(int);
    
    // Copy all data into the message buffer
    // Ydct
    memcpy(ptr, cm->curframe->residuals->Ydct, cm->ypw * cm->yph * sizeof(int16_t));
    ptr += cm->ypw * cm->yph * sizeof(int16_t);
    
    // Udct
    memcpy(ptr, cm->curframe->residuals->Udct, cm->upw * cm->uph * sizeof(int16_t));
    ptr += cm->upw * cm->uph * sizeof(int16_t);
    
    // Vdct
    memcpy(ptr, cm->curframe->residuals->Vdct, cm->vpw * cm->vph * sizeof(int16_t));
    ptr += cm->vpw * cm->vph * sizeof(int16_t);
    
    // Macroblocks - Y component
    memcpy(ptr, cm->curframe->mbs[Y_COMPONENT], cm->mb_rows * cm->mb_cols * sizeof(struct macroblock));
    ptr += cm->mb_rows * cm->mb_cols * sizeof(struct macroblock);
    
    // Macroblocks - U component
    memcpy(ptr, cm->curframe->mbs[U_COMPONENT], (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(struct macroblock));
    ptr += (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(struct macroblock);
    
    // Macroblocks - V component
    memcpy(ptr, cm->curframe->mbs[V_COMPONENT], (cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(struct macroblock));
    
    // Calculate total size of encoded data
    size_t total_size = sizeof(int) + // keyframe flag
                       (cm->ypw * cm->yph * sizeof(int16_t)) + // Ydct
                       (cm->upw * cm->uph * sizeof(int16_t)) + // Udct
                       (cm->vpw * cm->vph * sizeof(int16_t)) + // Vdct
                       (cm->mb_rows * cm->mb_cols * sizeof(struct macroblock)) + // Y macroblocks
                       ((cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(struct macroblock)) + // U macroblocks
                       ((cm->mb_rows/2) * (cm->mb_cols/2) * sizeof(struct macroblock)); // V macroblocks
    
    printf("Server: Encoded data size: %zu bytes\n", total_size);
    
    // Check if encoded data exceeds buffer size
    if (total_size > MESSAGE_SIZE) {
        fprintf(stderr, "Server: ERROR - Encoded data size (%zu) exceeds message buffer size (%d)\n", 
               total_size, MESSAGE_SIZE);
        // Handle error condition
        return;
    }
    
    // Transfer encoded data to client
    printf("Server: Starting DMA transfer of encoded data\n");
    SCIStartDmaTransfer(dma_queue, 
                       local_segment,
                       remote_segment,
                       offsetof(struct server_segment, message_buffer),
                       total_size,
                       offsetof(struct client_segment, message_buffer),
                       NO_CALLBACK,
                       NULL,
                       NO_FLAGS,
                       &error);
    
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "Server: Error in encoded data DMA transfer: 0x%x\n", error);
        return;
    }
    
    // Wait for transfer to complete
    printf("Server: Waiting for encoded data DMA transfer to complete\n");
    SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    printf("Server: Encoded data DMA transfer complete\n");
    
    // Signal client that encoded data is ready
    printf("Server: Signaling client that encoded data is ready\n");
    SCIFlush(NULL, NO_FLAGS);
    remote_seg->packet.data_size = total_size;
    remote_seg->packet.cmd = CMD_ENCODED_DATA;
    SCIFlush(NULL, NO_FLAGS);
    printf("Server: Encoded data ready signal sent\n");
    
    // Wait for client to acknowledge receipt
    printf("Server: Waiting for client to acknowledge encoded data\n");
    time_t start_time = time(NULL);
    bool timeout = false;
    
    while (local_seg->packet.cmd != CMD_ENCODED_DATA_ACK && !timeout) {
        if (time(NULL) - start_time > 30) {  // 30 second timeout
            timeout = true;
            fprintf(stderr, "Server: Timeout waiting for encoded data acknowledgment\n");
        }
    }
    
    if (!timeout) {
        printf("Server: Client acknowledged receipt of encoded data\n");
        local_seg->packet.cmd = CMD_INVALID;
        
        // Advance frame counters
        printf("Server: Frame %d processing complete\n", cm->framenum);
        cm->framenum++;
        cm->frames_since_keyframe++;
    }
}

// Modified main loop to echo frame numbers
int main_loop(sci_desc_t sd, 
    volatile struct server_segment *local_seg,
    volatile struct client_segment *remote_seg,
    sci_dma_queue_t dma_queue,
    sci_local_segment_t local_segment,
    sci_remote_segment_t remote_segment)
{
    int running = 1;
    uint32_t cmd;
    sci_error_t error;
    int frame_count = 0;
    struct c63_common *cm = NULL;
    yuv_t image;
    
    printf("Server: Waiting for commands...\n");

    while(running)
    {
        // Wait for command from client
        while(local_seg->packet.cmd == CMD_INVALID) {
        }

        // Process command
        cmd = local_seg->packet.cmd;
        
        // Reset command field
        local_seg->packet.cmd = CMD_INVALID;
        
        switch(cmd) {
            case CMD_DIMENSIONS:
                // Extract the dimensions from the message buffer
                struct dimensions_data dim_data;
                memcpy(&dim_data, (const void*)local_seg->message_buffer, sizeof(struct dimensions_data));
                
                printf("Server: Received dimensions (width=%u, height=%u)\n", 
                       dim_data.width, dim_data.height);
                
                // Store dimensions for later use
                width = dim_data.width;
                height = dim_data.height;
                
                // Initialize cm structure
                cm = init_c63_enc(width, height);

                // Initialize thread pool and task pool
                thread_pool_init();
                task_pool_init(cm->padh[Y_COMPONENT]);
                
                // Allocate YUV structure for frames using CUDA unified memory
                image.Y = NULL;
                image.U = NULL;
                image.V = NULL;
                
                // Use cudaMallocManaged for Y component
                cudaMallocManaged((void**)&image.Y, cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT] * sizeof(uint8_t));
                
                // Use cudaMallocManaged for U component
                cudaMallocManaged((void**)&image.U, cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT] * sizeof(uint8_t));
                
                // Use cudaMallocManaged for V component
                cudaMallocManaged((void**)&image.V, cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT] * sizeof(uint8_t));
                
                // Transfer dimensions back to client as acknowledgment
                memcpy((void*)local_seg->message_buffer, &dim_data, sizeof(struct dimensions_data));
                
                SCIStartDmaTransfer(dma_queue, 
                                   local_segment,  // Source
                                   remote_segment, // Destination
                                   offsetof(struct server_segment, message_buffer),
                                   sizeof(struct dimensions_data),
                                   offsetof(struct client_segment, message_buffer),
                                   NO_CALLBACK, 
                                   NULL, 
                                   NO_FLAGS, 
                                   &error);
                                   
                // Wait for transfer to complete
                SCIWaitForDMAQueue(dma_queue, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
                
                // Signal client that dimensions are acknowledged
                SCIFlush(NULL, NO_FLAGS);
                remote_seg->packet.cmd = CMD_DIMENSIONS_ACK;
                SCIFlush(NULL, NO_FLAGS);
                
                printf("Server: Dimensions acknowledged\n");
                printf("Server: Waiting for frames...\n");
                break;
                
            // In server main_loop, modify the YUV_DATA case:
            case CMD_YUV_DATA:
            {
                if (cm == NULL) {
                    fprintf(stderr, "Server: ERROR - Received YUV data before dimensions!\n");
                    break;
                }
                size_t data_size = local_seg->packet.data_size;
                
                // Reset command immediately
                local_seg->packet.cmd = CMD_INVALID;

                size_t full_frame_size = (width * height) + ((width * height) / 2);
                size_t y_size = width * height;
                size_t u_size = (width * height) / 4;
                size_t v_size = (width * height) / 4;

                if (data_size == full_frame_size) {
                    printf("Server: Received YUV frame with size %zu bytes\n", data_size);
                    // Y plane
                    memcpy(image.Y, (const void*)local_seg->message_buffer, y_size);
                    // U plane follows Y plane
                    memcpy(image.U, (const void*)(local_seg->message_buffer + y_size), u_size);
                    // V plane follows U plane
                    memcpy(image.V, (const void*)(local_seg->message_buffer + y_size + u_size), v_size);
                    
                    // Send acknowledgment
                    printf("Server: Sending YUV frame acknowledgment to client\n");
                    SCIFlush(NULL, NO_FLAGS);
                    remote_seg->packet.cmd = CMD_YUV_DATA_ACK;
                    SCIFlush(NULL, NO_FLAGS);
                    printf("Server: YUV frame acknowledgment sent\n");
                }
                else {
                    fprintf(stderr, "Server: ERROR - Received unexpected data size %zu, expected %zu\n", 
                        data_size, full_frame_size);
                
                    // Still send ACK to prevent client from hanging
                    SCIFlush(NULL, NO_FLAGS);
                    remote_seg->packet.cmd = CMD_YUV_DATA_ACK;
                    SCIFlush(NULL, NO_FLAGS);
                }
                encode_frame(cm, &image, frame_count, local_seg, remote_seg, dma_queue, local_segment, remote_segment);
            }
            break;    
            case CMD_QUIT:
                printf("Server: Received quit command after processing %d frames\n", cm->framenum);
                running = 0;
                
                // Free allocated resources
                if (cm) {
                    if (image.Y) cudaFree(image.Y);
                    if (image.U) cudaFree(image.U);
                    if (image.V) cudaFree(image.V);
                    free_c63_enc(cm);
                    task_pool_destroy();
                    thread_pool_destroy();
                }
                break;
                
            default:
                printf("Server: Unknown command: %d\n", cmd);
                break;
        }
    }

    return frame_count;
}


int main(int argc, char **argv)
{
    unsigned int localAdapterNo = 0;
    unsigned int remoteNodeId = 0;

    sci_error_t error;
    sci_desc_t sd;
    sci_remote_segment_t remoteSegment;
    sci_local_segment_t localSegment;
    sci_dma_queue_t dmaQueue;
    sci_map_t localMap, remoteMap;

    volatile struct server_segment *server_segment;
    volatile struct client_segment *client_segment;

    int c;

    while ((c = getopt(argc, argv, "r:")) != -1)
    {
        switch (c)
        {
            case 'r':
                remoteNodeId = atoi(optarg);
                break;
            default:
                break;
        }
    }

    if (remoteNodeId == 0) {
        fprintf(stderr, "Remote node-id is not specified. Use -r <remote node-id>\n");
        exit(EXIT_FAILURE);
    }

    // Initialize SISCI
    SCIInitialize(NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIInitialize failed - Error code: 0x%x\n", error);
        exit(EXIT_FAILURE);
    }

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
                SEGMENT_SERVER,
                sizeof(struct server_segment),
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

    // Make segment available
    SCISetSegmentAvailable(localSegment, localAdapterNo, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCISetSegmentAvailable failed - Error code 0x%x\n", error);
        SCIRemoveSegment(localSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }

    // Create DMA queue
    SCICreateDMAQueue(sd, &dmaQueue, localAdapterNo, 1, NO_FLAGS, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCICreateDMAQueue failed - Error code 0x%x\n", error);
        SCISetSegmentUnavailable(localSegment, localAdapterNo, NO_FLAGS, &error);
        SCIRemoveSegment(localSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }

    // Map local segment
    server_segment = (volatile struct server_segment *)SCIMapLocalSegment(
        localSegment, 
        &localMap, 
        0, 
        sizeof(struct server_segment), 
        NULL, 
        NO_FLAGS, 
        &error);

    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIMapLocalSegment failed - Error code 0x%x\n", error);
        SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
        SCISetSegmentUnavailable(localSegment, localAdapterNo, NO_FLAGS, &error);
        SCIRemoveSegment(localSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }

    // Initialize control packet
    server_segment->packet.cmd = CMD_INVALID;

    printf("Server: Waiting for client connection...\n");

    // Connect to client segment
    do {
        SCIConnectSegment(sd,
                        &remoteSegment,
                        remoteNodeId,
                        SEGMENT_CLIENT,
                        localAdapterNo,
                        NO_CALLBACK,
                        NULL,
                        SCI_INFINITE_TIMEOUT,
                        NO_FLAGS,
                        &error);
    } while (error != SCI_ERR_OK);

    printf("Server: Connected to client segment\n");

    // Map remote segment
    client_segment = (volatile struct client_segment *)SCIMapRemoteSegment(
        remoteSegment, 
        &remoteMap, 
        0,
        sizeof(struct client_segment),
        NULL, 
        NO_FLAGS, 
        &error);

    if (error != SCI_ERR_OK) {
        fprintf(stderr, "SCIMapRemoteSegment failed - Error code 0x%x\n", error);
        SCIUnmapSegment(remoteMap, NO_FLAGS, &error);
        SCIDisconnectSegment(remoteSegment, NO_FLAGS, &error);
        SCIUnmapSegment(localMap, NO_FLAGS, &error);
        SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
        SCISetSegmentUnavailable(localSegment, localAdapterNo, NO_FLAGS, &error);
        SCIRemoveSegment(localSegment, NO_FLAGS, &error);
        SCIClose(sd, NO_FLAGS, &error);
        SCITerminate();
        exit(EXIT_FAILURE);
    }

    // Enter main loop
    main_loop(sd, server_segment, client_segment, dmaQueue, localSegment, remoteSegment);

    printf("Server: Exiting\n");
    
    // Clean up resources
    SCIUnmapSegment(remoteMap, NO_FLAGS, &error);
    SCIDisconnectSegment(remoteSegment, NO_FLAGS, &error);
    SCIUnmapSegment(localMap, NO_FLAGS, &error);
    SCIRemoveDMAQueue(dmaQueue, NO_FLAGS, &error);
    SCISetSegmentUnavailable(localSegment, localAdapterNo, NO_FLAGS, &error);
    SCIRemoveSegment(localSegment, NO_FLAGS, &error);
    SCIClose(sd, NO_FLAGS, &error);
    SCITerminate();

    return 0;
}
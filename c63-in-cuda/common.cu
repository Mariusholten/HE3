#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

#include <cuda_runtime.h>

void destroy_frame(struct frame *f)
{
    /* First frame doesn't have a reconstructed frame to destroy */
    if (!f) { return; }
    
    // Free memory using cudaFree instead of free for unified memory
    cudaFree(f->recons->Y);
    cudaFree(f->recons->U);
    cudaFree(f->recons->V);
    free(f->recons);
    
    cudaFree(f->residuals->Ydct);
    cudaFree(f->residuals->Udct);
    cudaFree(f->residuals->Vdct);
    free(f->residuals);
    
    cudaFree(f->predicted->Y);
    cudaFree(f->predicted->U);
    cudaFree(f->predicted->V);
    free(f->predicted);
    
    cudaFree(f->mbs[Y_COMPONENT]);
    cudaFree(f->mbs[U_COMPONENT]);
    cudaFree(f->mbs[V_COMPONENT]);
    
    free(f);
}

struct frame* create_frame(struct c63_common *cm, yuv_t *image)
{
    frame *f = (frame*)malloc(sizeof(struct frame));
    f->orig = image;
    
    f->recons = (yuv_t*)malloc(sizeof(yuv_t));
    
    // Use cudaMallocManaged for unified memory
    cudaMallocManaged((void**)&f->recons->Y, cm->ypw * cm->yph * sizeof(uint8_t));
    cudaMallocManaged((void**)&f->recons->U, cm->upw * cm->uph * sizeof(uint8_t));
    cudaMallocManaged((void**)&f->recons->V, cm->vpw * cm->vph * sizeof(uint8_t));
    
    f->predicted = (yuv_t*)malloc(sizeof(yuv_t));
    cudaMallocManaged((void**)&f->predicted->Y, cm->ypw * cm->yph * sizeof(uint8_t));
    cudaMallocManaged((void**)&f->predicted->U, cm->upw * cm->uph * sizeof(uint8_t));
    cudaMallocManaged((void**)&f->predicted->V, cm->vpw * cm->vph * sizeof(uint8_t));
    
    // Initialize to zero
    cudaMemset(f->predicted->Y, 0, cm->ypw * cm->yph * sizeof(uint8_t));
    cudaMemset(f->predicted->U, 0, cm->upw * cm->uph * sizeof(uint8_t));
    cudaMemset(f->predicted->V, 0, cm->vpw * cm->vph * sizeof(uint8_t));
    
    f->residuals = (dct_t*)malloc(sizeof(dct_t));
    cudaMallocManaged((void**)&f->residuals->Ydct, cm->ypw * cm->yph * sizeof(int16_t));
    cudaMallocManaged((void**)&f->residuals->Udct, cm->upw * cm->uph * sizeof(int16_t));
    cudaMallocManaged((void**)&f->residuals->Vdct, cm->vpw * cm->vph * sizeof(int16_t));
    
    // Initialize to zero
    cudaMemset(f->residuals->Ydct, 0, cm->ypw * cm->yph * sizeof(int16_t));
    cudaMemset(f->residuals->Udct, 0, cm->upw * cm->uph * sizeof(int16_t));
    cudaMemset(f->residuals->Vdct, 0, cm->vpw * cm->vph * sizeof(int16_t));
    
    cudaMallocManaged((void**)&f->mbs[Y_COMPONENT], cm->mb_rows * cm->mb_cols * sizeof(struct macroblock));
    cudaMallocManaged((void**)&f->mbs[U_COMPONENT], cm->mb_rows/2 * cm->mb_cols/2 * sizeof(struct macroblock));
    cudaMallocManaged((void**)&f->mbs[V_COMPONENT], cm->mb_rows/2 * cm->mb_cols/2 * sizeof(struct macroblock));
    
    // Initialize to zero
    cudaMemset(f->mbs[Y_COMPONENT], 0, cm->mb_rows * cm->mb_cols * sizeof(struct macroblock));
    cudaMemset(f->mbs[U_COMPONENT], 0, cm->mb_rows/2 * cm->mb_cols/2 * sizeof(struct macroblock));
    cudaMemset(f->mbs[V_COMPONENT], 0, cm->mb_rows/2 * cm->mb_cols/2 * sizeof(struct macroblock));
    
    return f;
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
  fwrite(image->Y, 1, w*h, fp);
  fwrite(image->U, 1, w*h/4, fp);
  fwrite(image->V, 1, w*h/4, fp);
}

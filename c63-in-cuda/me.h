#ifndef C63_ME_H_
#define C63_ME_H_

#include "c63.h"

// Declaration
void c63_motion_estimate(struct c63_common *cm);

void c63_motion_compensate(struct c63_common *cm);

#define SAD_THRESHOLD 2000


/*Function to update the macroblocks with the calculated motion vector information by iterating through each mb position, 
it also sets the use_mv (flag to determine whether to use motion vector) if best sad < SAD_THRESHOLD.
Parameters for this function are: best_mv_x, best_mv_y, best_sad which are pointers to host allocated arrays, 
mbs which is a pointer to an array of macroblock structures (host allocated) that will be updated,
mb_rows and mb_cols which are the number of macroblock rows and colums in the frame. */
void process_component_results(
    int *best_mv_x,
    int *best_mv_y,
    int *best_sad,
    struct macroblock *mbs,
    int mb_rows,
    int mb_cols
);



#endif  /* C63_ME_H_ */

#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sisci_error.h>
#include <sisci_api.h>

static uint32_t remote_node = 0;

/* getopt */
extern int optind;
extern char *optarg;

static void print_help()
{
    printf("Usage: ./c63server -r nodeid\n");
    printf("Commandline options:\n");
    printf("  -r                             Node id of client\n");
    printf("\n");

    exit(EXIT_FAILURE);
}


int main(int argc, char **argv)
{
    int c;
    sci_desc_t sd;  
    sci_error_t error;

    if (argc == 1) { print_help(); }

    while ((c = getopt(argc, argv, "h:w:o:f:i:r:")) != -1)
    {
        switch (c)
        {
            case 'r':
                remote_node = atoi(optarg);
                break;
            default:
                print_help();
                break;
        }
    }

    /* Initialize the SISCI library */
    SCIInitialize(0, &error);
    if (error != SCI_ERR_OK) {
        fprintf(stderr,"SCIInitialize failed: %s\n", SCIGetErrorString(error));
        exit(EXIT_FAILURE);
    }


    SCITerminate();

    return 0;

}

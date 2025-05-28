# Codec63 #

The repository contains code for both native C code that is meant for compilation with GCC,
and native CUDA code that is meant for compilation with NVCC. Choose the subdirectory 
c63-in-c for the first and c63-in-cuda for the second.

The code is actually the same at this point (C++ is pretty much a superset of C,
and CUDA a superset of C++), only the filenames are different for convenience.

## Dolphin SISCI

This branch contains a few changes compared to master to facilitate using SISCI
in a Dolphin NTB cluster. First of all, two new executables are build:
c63server and c63client. The client is a copy of c63enc with takes an
additional `-r` parameter for remote node and call `SCIInitialize`. The
c63server is a very bare bones executable that just calls `SCIInitialize`. The
intention is that c63client runs on the x86 node and the c63server on the
Xavier. Additionally there is a `run.sh` script that compiles and runs the code
on the target nodes. This script should be run on the login node, it
automatically copies over the source to both nodes, compiles and runs, passing
the correct remote node id with `-r` to the program. You may change the
`run.sh` script, but the delivery should contain a run.sh with the same usage.

Same as with the previous exam, select either c63-in-c or c63-in-cuda, but this
time, set it in `run.sh` so that it compiles the correct variant. You also need
to set your group number in `common.h`.

### Things you need to change

- Set cuda or c in `run.sh`
- Set your group number in `c63.h`

aa3pKLdWCf

## Usage

The `run.sh` script will compile and launch the encoder on both nodes. `c63client`
will be run on the PC and `c63server` will be launched
on the tegra. To specify the cluster to run on, specify the `--tegra` parameter.
The x86 node is automatically selected from the given tegra node. To pass arguments
to `c63client`, use `--args "arg1 arg2"`.

Example usage:
```
./run.sh --tegra tegra-3 --args "/opt/Media/foreman.yuv -o output -w 352 -h 288"
```

We strongly recommend that clone the github repository in5050-codec63 twice.
Make your change in one of them, but use the c53dec and c63pred command from your
_other_ checkout. The reason is that nearly all files are shared between encoding
and decoding.

mv /home/in5050-g09/in5050-codec63-build/c63-in-cuda/build/tractor /home/in5050-g09/in5050-codec63/c63-in-cuda/build

To decode the c63 file:
```
./c63dec foremanout.c63 output.yuv
```

To playback the raw YUV file before encoding or after decoding, you can use mplayer or ffplay.
```
mplayer -demuxer rawvideo -rawvideo w=352:h=288 output.yuv
ffplay -f rawvideo -pixel_format yuv420p -video_size 352x288 -i output.yuv
```
When you are remotely logged into a Tegra, you may have to force the use of X11:
```
mplayer -vo x11 -demuxer rawvideo -rawvideo w=352:h=288 output.yuv
```


### Description ###
This project is used in IN5050 (Programming Heterogeneous Multi-core Architectures) at the Department of Informatics, University of Oslo, Norway. For more information, see the [course page](https://www.uio.no/studier/emner/matnat/ifi/IN5050/).

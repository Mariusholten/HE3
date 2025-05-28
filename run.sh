#!/bin/bash
set -e

#
# USAGE: ./run.sh [--tegra hostname]
#

SRC_DIR="$(realpath $(dirname $0))"

TEGRA_CMD="c63server"
TEGRA_ARGS=""

# Set this to use either the c variant or the CUDA variant
SUBDIR="c63-in-cuda"

if [ ! -d $SRC_DIR/$SUBDIR ]; then
    echo "Please set SUBDIR variable to c63-in-c or c63-in-cuda in $(realpath $0)"
    exit 1
fi

PC_CMD="c63client"
#PC_ARGS="/opt/Media/foreman.yuv -f 120 -o output -w 352 -h 288"
PC_ARGS="/opt/Media/tractor.yuv -f 100 -o tractor -w 1920 -h 1080"

DATE=$(date -u +%Y%m%d-%H%M%S)
RSYNC_ARGS="-rt --exclude=logs/ --exclude=.*"
BUILD_DIR="in5050-codec63-build"

echo "Source dir: $SRC_DIR"

trap quit INT

function quit()
{
    echo "Cleaning up"
    ssh $TEGRA "pkill -u \$(whoami) $TEGRA_CMD" &> /dev/null || true
    ssh $PC "pkill -u \$(whoami) $PC_CMD" &> /dev/null || true
    echo "Logfiles:"
    ls -lh logs/$DATE-*.log
}

mkdir -p logs

#Argument parsing
while [ $# -gt 0 ] ; do
    arg=$1
    shift

    case $arg in

        --clean)
            CLEAN="clean"
            ;;
        --args)
            PC_ARGS=$1
            shift
            ;;
        --tegra)
            TEGRA=$1
            shift
            ;;
        --pc)
            PC=$1
            shift
            ;;
        --reverse) #For debugging only. Not required to work
            T="$TEGRA_CMD"
            TEGRA_CMD="$PC_CMD"
            PC_CMD="$T"
            T="$TEGRA_ARGS"
            TEGRA_ARGS="$PC_ARGS"
            PC_ARGS="$T"
            ;;
    esac
done

if [ -z "$PC" ]; then
    if [ "$TEGRA" == "tegra-1" ]; then
        PC="in5050-2016-10"
        TEGRA="tegra-1"
    elif [ "$TEGRA" == "tegra-2" ]; then
        PC="in5050-2014-11"
        TEGRA="tegra-2"
    elif [ "$TEGRA" == "tegra-3" ]; then
        PC="in5050-2014-11"
        TEGRA="tegra-3"
    elif [ "$TEGRA" == "tegra-4" ]; then
        PC="in5050-2014-11"
        TEGRA="tegra-4"
    else
        echo "unknown tegra $TEGRA"
        exit 1
    fi
fi

TEGRA_NODE=$(/opt/DIS/sbin/disinfo get-nodeid -hostname ${TEGRA})
PC_NODE=$(/opt/DIS/sbin/disinfo get-nodeid -hostname ${PC})

echo "Using $TEGRA ($TEGRA_NODE) and $PC ($PC_NODE)"

echo "Syncing source"
rsync ${RSYNC_ARGS} ${SRC_DIR}/ $TEGRA:${BUILD_DIR}/
rsync ${RSYNC_ARGS} ${SRC_DIR}/ $PC:${BUILD_DIR}/

#Compile on tegra and pc
echo
echo "### Compiling on Tegra ###"
echo
ssh -t $TEGRA "export PATH=/usr/local/cuda/bin:\$PATH && mkdir -p ${BUILD_DIR}/$SUBDIR/build && cd $BUILD_DIR/$SUBDIR/build && cmake .. && make ${CLEAN} $TEGRA_CMD" || exit $?

echo
echo "### Compiling on PC ###"
echo
ssh -t $PC "export PATH=/usr/local/cuda/bin:\$PATH && mkdir -p ${BUILD_DIR}/${SUBDIR}/build && cd $BUILD_DIR/$SUBDIR/build && cmake .. && make ${CLEAN} $PC_CMD" || exit $?

#Launch on both nodes
echo "Running:"
stdbuf -oL -eL ssh $TEGRA "cd $BUILD_DIR/$SUBDIR/build && time stdbuf -oL -eL ./$TEGRA_CMD -r $PC_NODE $TEGRA_ARGS; echo Tegra exit code: \$?" |& tee logs/$DATE-tegra.log &
stdbuf -oL -eL ssh $PC "cd $BUILD_DIR/$SUBDIR/build && time stdbuf -oL -eL ./$PC_CMD -r $TEGRA_NODE $PC_ARGS; echo PC exit code: \$?" |& tee logs/$DATE-pc.log &

wait 

echo "Done!"

quit

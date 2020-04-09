#!/bin/bash

#SBATCH --job-name="bmd"
#SBATCH --mail-type=end,fail
#SBATCH --mail-user="till.meyerzuwestram@artorg.unibe.ch"
#SBATCH --cpus-per-task=3
#SBATCH --mem=30G
#SBATCH --time=24:00:0
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --partition=gpu
#SBATCH --tmp=30G

# just a little safeguard...
declare -r TARGET_USER="tm19i462"
if [ "$(whoami)" != "$TARGET_USER" ]; then
	echo "You are not $TARGET_USER -- aborting to make sure you don't break stuff" >&2
	exit 1
fi

# CUDA library
module load CUDA/10.1.105-GCC-8.2.0-2.31.1
# optimized kernels for CUDA
module load cuDNN/7.6.0.64-gcccuda-2019a
module load Python/3.7.2-GCCcore-8.2.0


declare -r BASEDIR=$HOME/bmd
test -d $BASEDIR || exit 1
#declare -r WORKDIR=$(mktemp -d -p /data/users/$(whoami) "$(date +%Y_%m_%d-%H_%M_%S)-XXX")
declare -r WORKDIR=$(mktemp -d -p $BASEDIR "$(date +%Y_%m_%d-%H_%M_%S)-XXX")

mkdir -p $WORKDIR

echo -e "======================================================"
echo -e "WORKDIR:"
echo -e "\t$WORKDIR"
echo -e "TMPDIR:"
echo -e "\t$TMPDIR"
echo -e "======================================================"

(
	# copy data on-demand to local scratch fs, as defined in $TMPDIR
	echo "Copying data from $BASEDIR/data to $TMPDIR..."
	time rsync -a --stats $BASEDIR/data $TMPDIR | tee $WORKDIR/rsync-log.txt
	cd $BASEDIR || exit
	pip install --user --upgrade -r requirements.txt > /dev/null
	./run.py \
		--device cuda \
		--dir $BASEDIR \
		--data-dir $TMPDIR/data \
		--state $WORKDIR/model.pth \
		--batch 24 \
		--epochs 30 \
		--train \
		--validate \
		--log $WORKDIR/log.txt \
		--scratch $WORKDIR \
		"$@"
)

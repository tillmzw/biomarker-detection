#!/bin/bash

#SBATCH --job-name="bmd-debug"
#SBATCH --mail-type=end,fail
#SBATCH --mail-user="till.meyerzuwestram@artorg.unibe.ch"
#SBATCH --cpus-per-task=3
#SBATCH --mem=30G
#SBATCH --time=00:10:0
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --partition=gpu-debug

# CUDA library
module load CUDA/10.1.105-GCC-8.2.0-2.31.1
# optimized kernels for CUDA
module load cuDNN/7.6.0.64-gcccuda-2019a
module load Python/3.7.2-GCCcore-8.2.0


declare -r BASEDIR=$HOME/bmd
test -d $BASEDIR || exit 1
#declare -r WORKDIR=$(mktemp -d -p /data/users/$(whoami) "$(date +%Y_%m_%d-%H_%M_%S)-XXX")
declare -r WORKDIR=$(mktemp -d -p $BASEDIR "DEBUG_$(date +%Y_%m_%d-%H_%M_%S)-XXX")

mkdir -p $WORKDIR

echo -e "======================================================"
echo -e "WORKDIR:"
echo -e "\t$WORKDIR"
echo -e "======================================================"

(
	cd $BASEDIR
	pip install --user --upgrade -r requirements.txt > /dev/null
	./run.py \
		--device cuda \
		--dir $BASEDIR \
		--state $WORKDIR/model.pth \
		--limit 128 \
		--validation-limit 128 \
		--batch 24 \
		--epochs 5 \
		--train \
		--validate \
		--log $WORKDIR/log.txt \
		$@
)

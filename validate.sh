#!/bin/bash

#SBATCH --job-name="bmd-validate"
#SBATCH --mail-type=end,fail
#SBATCH --mail-user="till.meyerzuwestram@artorg.unibe.ch"
#SBATCH --cpus-per-task=3
#SBATCH --mem=30G
#SBATCH --time=01:00:0
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --partition=gpu

# CUDA library
module load CUDA/10.1.105-GCC-8.2.0-2.31.1
# optimized kernels for CUDA
module load cuDNN/7.6.0.64-gcccuda-2019a
module load Python/3.7.2-GCCcore-8.2.0


declare -r BASEDIR=$HOME/bmd
test -d $BASEDIR || exit 1
#declare -r WORKDIR=$(mktemp -d -p /data/users/$(whoami) "$(date +%Y_%m_%d-%H_%M_%S)-XXX")
declare -r WORKDIR=$1

test -n $WORKDIR || exit 1
test -d $WORKDIR || exit 1

echo -e "======================================================"
echo -e "WORKDIR:"
echo -e "\t$WORKDIR"
echo -e "======================================================"

(
	cd $BASEDIR
	pip install --user --upgrade -r requirements.txt > /dev/null
	./run.py \
		--dir $BASEDIR \
		--state $WORKDIR/model.pth \
		--batch 64 \
		--validate \
		--scratch $WORKDIR \
		--mismatch

	[ $? -eq 0 ] || exit 1
	echo "Here is a list of entries with the maximum mismatch:"
	# tabulate, then run awk to extract the first line and every line where the 4th col is 4 or -4
	cat $WORKDIR/mismatches.csv | tr ',' '\t' | awk '(NR == 1 || $4 == 4 || $4 == -4) {print $0}' | column -t

)


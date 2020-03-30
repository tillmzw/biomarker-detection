#!/bin/bash

#SBATCH --job-name="bmd-label-count"
#SBATCH --mail-type=end,fail
#SBATCH --mail-user="till.meyerzuwestram@artorg.unibe.ch"
#SBATCH --cpus-per-task=16
#SBATCH --mem=10G
#SBATCH --time=6:00:0
#SBATCH --partition=gpu

# just a little safeguard...
declare -r TARGET_USER="tm19i462"
if [ "$(whoami)" != "$TARGET_USER" ]; then
	echo "You are not $TARGET_USER -- aborting to make sure you don't break stuff" >&2
	exit 1
fi

module load Python/3.7.2-GCCcore-8.2.0


declare -r BASEDIR=$HOME/bmd
test -d $BASEDIR || exit 1
#declare -r WORKDIR=$(mktemp -d -p /data/users/$(whoami) "$(date +%Y_%m_%d-%H_%M_%S)-XXX")
declare -r WORKDIR=$(mktemp -d -p $BASEDIR "$(date +%Y_%m_%d-%H_%M_%S)-XXX")

mkdir -p $WORKDIR

echo -e "======================================================"
echo -e "WORKDIR:"
echo -e "\t$WORKDIR"
echo -e "======================================================"

(
	cd $BASEDIR || exit
	pip install --user --upgrade -r requirements.txt > /dev/null
  ./tools/label-count.py \
    "$@" | tee $WORKDIR/label-count.txt
)

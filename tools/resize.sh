#!/bin/bash

declare -r DIR="$1"
declare -r PREFIX="$2"

set -e

if [ -z $DIR ]; then
	echo "Need to provide a folder in which to resize."
	exit 1
fi

if [ ! -d $DIR ]; then
	echo "$DIR is not a directory."
	exit 2
fi

for image in $DIR/*.jpeg; do
	# resize the smaller size to 1024, maintain aspect ratio
	echo "$(dirname $image)"/"$PREFIX""$(basename $image)"
	convert $image -resize '1024^>' "$(dirname $image)/$PREFIX$(basename $image)"

done

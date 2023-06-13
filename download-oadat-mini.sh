#!/bin/bash

OUTDIR="data/OADAT-mini"
mkdir -p $OUTDIR
declare -a filenames=("OADAT-mini.h5" "MSFD_multisegment_RawBP-mini.h5" "SCD_RawBP-mini.h5" "SWFD_multisegment_RawBP-mini.h5" "SWFD_semicircle_RawBP-mini.h5")
for fname in "${filenames[@]}"
do
    echo "Downloading file $fname to $OUTDIR"
    curl -o "$OUTDIR/$fname" "https://libdrive.ethz.ch/index.php/s/i974KegIjhJJxkK/download?path=%2F&files=$fname"
done

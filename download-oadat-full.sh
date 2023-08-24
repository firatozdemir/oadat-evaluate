#!/bin/bash

OUTDIR="data/OADAT"
mkdir -p $OUTDIR
declare -a filenames=("OADAT_v2.h5" "MSFD_multisegment_RawBP.h5" "MSFD_multisegment_ss_RawBP.h5" "SCD_RawBP.h5" "SCD_multisegment_ss_RawBP.h5 "SWFD_multisegment_RawBP.h5" "SWFD_multisegment_ss_RawBP.h5 "SWFD_semicircle_RawBP.h5")
for fname in "${filenames[@]}"
do
    echo "Downloading file $fname to $OUTDIR"
    curl -o "$OUTDIR/$fname" "https://libdrive.ethz.ch/index.php/s/i974KegIjhJJxkK/download?path=%2F&files=$fname"
done

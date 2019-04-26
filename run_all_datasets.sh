#!/bin/bash

# This script is for timing how fast your machine will solve the example problems (they are all small)
# Run this file from [CPLEXHomeDir]/cplex/examples/src/python/

# Change to *.mps for mps files
# NOTE noswot.mps will run forever, just crtl-C it once and it will continue
for f in ../../data/*.lp
do
    echo Doing $f
    time python2 admipex1.py $f 
    echo Done with $f
    echo HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHh
    echo
    echo

done

# This notification might not work on all machines
canberra-gtk-play --file=/usr/share/sounds/ubuntu/notifications/Mallet.ogg


#!/bin/bash

if [ $# -ne 2 ]; then
  echo -e "usage:\tbash hw6_fgsm.sh [input_image_directory] [out_image_directory]"
  exit
fi

python3 hw6_fgsm.py $1 $2

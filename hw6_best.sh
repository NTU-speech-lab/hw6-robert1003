#!/bin/bash

if [ $# -ne 2 ]; then
  echo -e "usage:\tbash hw6_best.sh [input_image_directory] [out_image_directory]"
  exit
fi

python3 hw6_best.py $1 $2 0 99 &
python3 hw6_best.py $1 $2 100 199 &

wait
echo done


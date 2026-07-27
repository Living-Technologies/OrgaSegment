#!/bin/bash

echo "usage: ./train_cellpose.sh pretrained_model data_folder"

# data_folder should contain both images and corresponding masks.
export data_folder=$2
#pretrained model, either a file or a name accepted by cellpose.
export pm=$1

export train_dir="$data_folder/train"
export val_dir="$data_folder/val"

export img_filter="_img"
export mask_filter="_masks_organoid"

python -m cellpose --train \
  --dir "$(realpath $train_dir)" --use_gpu --test_dir "$(realpath $val_dir)" \
  --img_filter $img_filter --mask_filter $mask_filter \
  --pretrained_model $pm --verbose

cp ~/.cellpose/run.log ./"$$"-run.log

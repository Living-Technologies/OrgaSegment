#!/bin/bash

# This script will download SLA organoid growth data and traing cellpose on it.
# A python environment with cellpose must be activated to run this script.

export data_URL="https://zenodo.org/records/10278229/files/OrganoidBasic_v20211206.zip"
export data_file=${data_URL/*\//}
export data_folder=${data_file/.zip/}


if [[ -a $1 ]]
 then
  echo "using pretrained model"
  export pm=$1
 else
  echo "using no pretrained model"
  export pm=None
fi

if [[ -d $data_folder ]]
then
  echo "folder $data_folder exists, assuming it contains the relevant data"
  echo "to re-download data, rename or delete this folder"
else
    if [[ -a $data_file ]]
    then
        echo "data file $data_file exists"
    else
      echo "getting $data_URL"
      wget $data_URL
    fi
    unzip $data_file
fi

export img_filter="_img"
export mask_filter="_masks_organoid"

export train_dir="$data_folder/train"
export val_dir="$data_folder/val"

python -m cellpose --train \
  --dir "$(realpath $train_dir)" --use_gpu --test_dir "$(realpath $val_dir)" \
  --img_filter $img_filter --mask_filter $mask_filter \
  --pretrained_model $pm --verbose

cp ~/.cellpose/run.log ./"$$"-run.log


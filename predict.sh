#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=RTX6000:1
#SBATCH --time=4-00:00:00
#SBATCH --mem=96G
#SBATCH --error=log/JobName.%J.err
#SBATCH --output=log/JobName.%J.out

PREDICT=false
TRACK=false

while getopts :c:f:pt flag
do
    case "${flag}" in
        p) PREDICT=true;;
        t) TRACK=true;;
        c) CONFIG=${OPTARG};;
        f) FOLDER=${OPTARG};;
    esac
done
echo "PREDICT: $PREDICT";
echo "TRACK: $TRACK";
echo "CONFIG: $CONFIG";
echo "FOLDER: $FOLDER";

ENV=OrgaSegment

source ~/.bashrc

cd $SLURM_SUBMIT_DIR

echo $ENV

conda activate $ENV

conda info --envs

nvidia-smi

## Predict
if [ "$PREDICT" = true ] ; then
    python predict_mrcnn.py $SLURM_JOB_ID $CONFIG $FOLDER
fi
 
## Track
if [ "$TRACK" = true ] ; then
    python track.py $SLURM_JOB_ID $CONFIG $FOLDER
fi

conda deactivate
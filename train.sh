#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=RTX6000:4
#SBATCH --cpus-per-gpu=4
#SBATCH --time=8-00:00:00
#SBATCH --mem=320G
#SBATCH --error=log/JobName.%J.err
#SBATCH --output=log/JobName.%J.out

TRAIN=false
EVAL=false
EVALMODEL=None

while getopts :c:m:te flag
do
    case "${flag}" in
        t) TRAIN=true;;
        e) EVAL=true;;
        c) CONFIG=${OPTARG};;
        m) EVALMODEL=${OPTARG};;
    esac
done
echo "TRAIN: $TRAIN";
echo "EVAL: $EVAL";
echo "CONFIG: $CONFIG";
echo "EVALMODEL: $MODEL";

ENV=OrgaSegment

source ~/.bashrc

cd $SLURM_SUBMIT_DIR

echo $ENV

conda activate $ENV

conda info --envs

nvidia-smi

## Train model
if [ "$TRAIN" = true ] ; then
    python train_mrcnn.py $SLURM_JOB_ID $CONFIG
fi
 
##Eval model
if [ "$EVAL" = true ] ; then
    python eval_mrcnn.py $SLURM_JOB_ID $CONFIG $EVALMODEL
fi


conda deactivate
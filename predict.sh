#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=RTX6000:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --mail-user=s.vanbeuningen@umcutrecht.nl
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --error=log/JobName.%J.err
#SBATCH --output=log/JobName.%J.out

ENV=OrgaSegment
DATA_DIR='/hpc/umc_beekman/DIS/20210526/DIS_COPY'

source ~/.bashrc

cd $SLURM_SUBMIT_DIR

echo $ENV

conda activate $ENV

conda info --envs

nvidia-smi

## execute python script
python predict_mrcnn.py $SLURM_JOB_ID $DATA_DIR

conda deactivate
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=RTX6000:4
#SBATCH --cpus-per-gpu=12
#SBATCH --time=8-00:00:00
#SBATCH --mem=320G
#SBATCH --mail-user=s.vanbeuningen@umcutrecht.nl
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --error=log/JobName.%J.err
#SBATCH --output=log/JobName.%J.out

ENV=OrgaSegment

source ~/.bashrc

cd $SLURM_SUBMIT_DIR

echo $ENV

conda activate $ENV

conda info --envs

nvidia-smi

## Train model
python train_mrcnn.py $SLURM_JOB_ID

##Eval model
python eval_mrcnn.py $SLURM_JOB_ID None

conda deactivate
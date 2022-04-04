#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=RTX6000:1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=20:00:00
#SBATCH --mem=120G
#SBATCH --mail-user=s.vanbeuningen@umcutrecht.nl
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --error=log/JobName.%J.err
#SBATCH --output=log/JobName.%J.out

ENV=OrgaSegment

source ~/.bashrc

cd $SLURM_SUBMIT_DIR
pwd

echo $ENV
nvidia-smi -q

conda activate $ENV

conda info --envs

## execute cellpose
python eval-cellpose.py

conda deactivate
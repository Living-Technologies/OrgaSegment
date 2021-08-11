#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=RTX6000:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
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

for INPUTDIR in /hpc/umc_beekman/data_organoids/boekhout_20210806/
do
    python predict_mrcnn.py $SLURM_JOB_ID $INPUTDIR
    python track.py $SLURM_JOB_ID $INPUTDIR
done

conda deactivate
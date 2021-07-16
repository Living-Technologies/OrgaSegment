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

source ~/.bashrc

cd $SLURM_SUBMIT_DIR

echo $ENV

conda activate $ENV

conda info --envs

nvidia-smi

for INPUTDIR in /hpc/umc_beekman/orgaswell/data/Livia_20201110_HNEC0116 /hpc/umc_beekman/orgaswell/data/Lisa_LR-035B
do
    python predict_mrcnn.py $SLURM_JOB_ID $INPUTDIR
done

conda deactivate
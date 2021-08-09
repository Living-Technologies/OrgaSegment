#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=RTX6000:1
#SBATCH --time=08:00:00
#SBATCH --mem=128G
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

for INPUTDIR in /hpc/umc_beekman/data_organoids/boekhout_20210806/ #/hpc/umc_beekman/data_organoids/20210727_DIS_BF_VALIDATION/Bassay_01_Thunder_BF/ /hpc/umc_beekman/data_organoids/20210727_DIS_BF_VALIDATION/DIS_BF_ValidationDonors_01_Thunder_BF/ /hpc/umc_beekman/data_organoids/20210727_DIS_BF_VALIDATION/Bassay_01_Zeiss_ESID/ /hpc/umc_beekman/data_organoids/20210727_DIS_BF_VALIDATION/DIS_BF_ValidationDonors_01_Zeiss_ESID/
do
    python predict_mrcnn.py $SLURM_JOB_ID $INPUTDIR
done

conda deactivate
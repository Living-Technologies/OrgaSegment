#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --mail-user=s.vanbeuningen@umcutrecht.nl
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --error=log/JobName.%J.err
#SBATCH --output=log/JobName.%J.out

ENV=OrgaSegment
DATA_DIR='/hpc/umc_beekman/orgaswell/data/Lisa_LR-035B'
REGEX='(?P<WELL>[A-Z]{1}_[0-9]{1,2})_t(?P<T>[0-9]{1,2})_.*'

source ~/.bashrc

cd $SLURM_SUBMIT_DIR

echo $ENV

conda activate $ENV

conda info --envs

## execute python script
python track.py $SLURM_JOB_ID $DATA_DIR $REGEX

conda deactivate
#!/bin/bash
#SBATCH --job-name=activitynet
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=3:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --requeue

singularity exec --bind /scratch --nv --overlay /scratch/crg9968/overlay-25GB-500K.ext3:ro /scratch/crg9968/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
cd /scratch/crg9968/nlp_project

python3 main.py --num_epochs=25 --fian --num_heads=2 --experiment=tacos_2 --config_path=\"config/tacos.yml\" > ./logs/tacos_smin_train_2.txt
"

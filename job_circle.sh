#!/bin/bash
#$ -N circle_train
#$ -cwd
#$ -o circle_train.log
#$ -j y
#$ -q batch
source /etc/profile.d/modules.sh
module load python/3.9.10
source /work/isat/sa2740ba/pinn-env/bin/activate
export MPLCONFIGDIR=/work/isat/sa2740ba/.mpl
export TMPDIR=/work/isat/sa2740ba/tmp
python scenarios/circle/train_circle.py

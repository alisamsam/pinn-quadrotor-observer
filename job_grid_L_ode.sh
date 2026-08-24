#!/bin/bash
#$ -N grid_Lode
#$ -cwd
#$ -o /work/isat/sa2740ba/grid_train_L_ode_log.txt
#$ -j y
#$ -q gpu
## if the gpu queue needs an explicit device request, add:  #$ -l gpu=1
module load pytorch/2.0.0/gpu
export MPLCONFIGDIR=/work/isat/sa2740ba/.mpl
export TMPDIR=/work/isat/sa2740ba/tmp
python L_gain/L_gain_TrainODE/grid_train_L_ode.py

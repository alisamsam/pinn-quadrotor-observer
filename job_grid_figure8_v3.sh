#!/bin/bash
#$ -N grid_fig8
#$ -cwd
#$ -o /work/isat/sa2740ba/grid_figure8_v3_log.txt
#$ -j y
#$ -q gpu
## if the gpu queue needs an explicit device request, add:  #$ -l gpu=1
module load pytorch/2.0.0/gpu
export MPLCONFIGDIR=/work/isat/sa2740ba/.mpl
export TMPDIR=/work/isat/sa2740ba/tmp
python Layers_Neurons_Simulation/grid_study_figure8_v3.py

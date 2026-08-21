#!/bin/bash
#$ -N grid_L
#$ -cwd
#$ -o /work/isat/sa2740ba/grid_L.log
#$ -j y
#$ -q batch
## Same queue as the original grid_study_v2 run, for comparable timing.
## If a GPU queue is available and you want it, switch to:  #$ -q gpu
source /etc/profile.d/modules.sh
module load python/3.9.10
source /work/isat/sa2740ba/pinn-env/bin/activate
export MPLCONFIGDIR=/work/isat/sa2740ba/.mpl
export TMPDIR=/work/isat/sa2740ba/tmp
# submit from the project root (pinn-quadrotor/), same as job_circle.sh
python L_gain_ArchStudy/grid_study_L_v2.py

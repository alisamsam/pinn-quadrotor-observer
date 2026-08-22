#!/bin/bash
#$ -N grid_L_int
#$ -cwd
#$ -o /work/isat/sa2740ba/grid_L_integrated.log
#$ -j y
#$ -q batch
## Same queue as the original L-OFF grid, for comparable timing.
## Switch to  #$ -q gpu  if you want the trainings on a GPU node.
source /etc/profile.d/modules.sh
module load python/3.9.10
source /work/isat/sa2740ba/pinn-env/bin/activate
export MPLCONFIGDIR=/work/isat/sa2740ba/.mpl
export TMPDIR=/work/isat/sa2740ba/tmp
# submit from the repo root (pinn-quadrotor/)
python L_gain_ArchStudy/grid_study_L_integrated.py

#!/bin/bash
#$ -N eval_diag
#$ -cwd
#$ -o /work/isat/sa2740ba/eval_diag.log
#$ -j y
#$ -q batch
source /etc/profile.d/modules.sh
module load python/3.9.10
source /work/isat/sa2740ba/pinn-env/bin/activate
export MPLCONFIGDIR=/work/isat/sa2740ba/.mpl
export TMPDIR=/work/isat/sa2740ba/tmp
# submit from the repo root (pinn-quadrotor/), so datasets/ and phase1a/ resolve
python L_gain_ArchStudy/eval_ode_diag.py

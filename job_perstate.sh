#!/bin/bash
#$ -N perstate10
#$ -cwd
#$ -o perstate10.log
#$ -j y
#$ -q batch
source /etc/profile.d/modules.sh
module load python/3.9.10
source /work/isat/sa2740ba/pinn-env/bin/activate
export MPLCONFIGDIR=/work/isat/sa2740ba/.mpl
export TMPDIR=/work/isat/sa2740ba/tmp
python eval_perstate_allflights.py

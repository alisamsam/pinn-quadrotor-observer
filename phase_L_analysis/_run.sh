cd ~/pinn-quadrotor
conda activate pinn-obs
pip install openpyxl -q 2>&1 | tail -1
python phase_L_analysis/_make_xlsx.py

cd ~/pinn-quadrotor
rm -f _push.sh
git push origin main 2>&1
echo '---status---'
git status -sb | head -1

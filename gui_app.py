import os, sys, subprocess, threading
import tkinter as tk
from tkinter import scrolledtext

ROOT = os.path.dirname(os.path.abspath(__file__))

TASKS = [
    ("Plot L behavior (per state)",      ["python", "phase_L_analysis/plot_L_behavior.py"]),
    ("Eigenvalue analysis (A - LC)",     ["python", "phase_L_analysis/eig_analysis.py"]),
    ("Compare L effect (errors + RMSE)", ["python", "phase_multi/compare_L_effect.py"]),
    ("Evaluate Phase 4  ([t,x0])",       ["python", "phase1a/evaluate_phase4.py"]),
    ("Evaluate Phase 5  ([t,x0]+L)",     ["python", "phase1a/evaluate_phase5.py"]),
    ("Evaluate multi-trajectory",        ["python", "phase_multi/evaluate_multi.py"]),
    ("Plot multi trajectories",          ["python", "phase_multi/plot_multi.py"]),
    ("Git status",                       ["git", "status", "--short"]),
    ("Git log (last 5)",                 ["git", "log", "--oneline", "-5"]),
]

class App:
    def __init__(self, root):
        root.title("PINN Quadrotor Lab - Samsaam Ali Baig")
        root.geometry("1050x640")
        left = tk.Frame(root, padx=10, pady=10)
        left.pack(side="left", fill="y")
        tk.Label(left, text="Tasks", font=("Calibri", 14, "bold")).pack(anchor="w")
        self.buttons = []
        for label, cmd in TASKS:
            b = tk.Button(left, text=label, width=32, anchor="w",
                          command=lambda c=cmd, l=label: self.run(c, l))
            b.pack(pady=3)
            self.buttons.append(b)
        self.out = scrolledtext.ScrolledText(root, bg="#101827", fg="#d7e3ee",
                                             font=("Consolas", 10))
        self.out.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        self.log(f"Project: {ROOT}\nPython:  {sys.executable}\nReady.\n")

    def log(self, text):
        self.out.insert("end", text)
        self.out.see("end")

    def set_buttons(self, state):
        for b in self.buttons: b.config(state=state)

    def run(self, cmd, label):
        self.set_buttons("disabled")
        self.log(f"\n=== {label} ===\n$ {' '.join(cmd)}\n")
        if cmd[0] == "python": cmd = [sys.executable] + cmd[1:]
        threading.Thread(target=self._worker, args=(cmd,), daemon=True).start()

    def _worker(self, cmd):
        try:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                self.out.after(0, self.log, line)
            p.wait()
            self.out.after(0, self.log, f"[exit code {p.returncode}]\n")
        except Exception as e:
            self.out.after(0, self.log, f"[ERROR] {e}\n")
        self.out.after(0, self.set_buttons, "normal")

root = tk.Tk()
App(root)
root.mainloop()
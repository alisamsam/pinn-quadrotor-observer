import numpy as np

G = 9.81

def peak_accel(omega, n=2000):
    """Peak total acceleration demand of the figure-8 at slowdown omega."""
    t = np.linspace(0, 2*np.pi/omega, n)     # one full period at this omega
    w = omega
    # accelerations (each original term, now with w*t and ×w²)
    xddot = -10 * w**2 * np.sin(w*t)
    yddot = -20 * w**2 * np.sin(2*w*t)
    zddot = -1  * w**2 * np.cos(w*t)
    # total acceleration magnitude at each instant
    a_mag = np.sqrt(xddot**2 + yddot**2 + zddot**2)
    return a_mag.max()

print("omega | peak accel (m/s²) | feasible? (< g=9.81)")
print("-"*50)
for omega in [1.0, 0.7, 0.5, 0.4, 0.3, 0.2]:
    a = peak_accel(omega)
    flag = "✅" if a < G*0.6 else ("⚠️" if a < G else "🔴 CRASH")
    print(f"{omega:5.2f} | {a:8.2f}          | {flag}")
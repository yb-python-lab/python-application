import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import simpledialog

# ----------------------------
# Popup window for damping input
# ----------------------------
root = tk.Tk()
root.withdraw()

c = simpledialog.askfloat(
    "Damping Input",
    "Enter damping coefficient (unit: s⁻¹)\nExample: 0.2",
    minvalue=0.0,
    maxvalue=10.0
)

if c is None:
    c = 0.2

# ----------------------------
# Physical parameters
# ----------------------------
g = 9.81
L = 1.0

# ----------------------------
# Pendulum differential equation
# ----------------------------
def pendulum(t, y, c):
    theta, omega = y
    dtheta = omega
    domega = -c * omega - (g / L) * np.sin(theta)
    return [dtheta, domega]

# ----------------------------
# Simulation setup
# ----------------------------
t_span = (0, 15)
y0 = [1.0, 0.0]
t_eval = np.linspace(0, 15, 400)

solution = solve_ivp(pendulum, t_span, y0, t_eval=t_eval, args=(c,))

t = solution.t
theta = solution.y[0]
omega = solution.y[1]

# ----------------------------
# Estimate oscillation period
# ----------------------------
peaks = []
for i in range(1, len(theta)-1):
    if theta[i-1] < theta[i] and theta[i] > theta[i+1]:
        peaks.append(t[i])

if len(peaks) > 1:
    periods = np.diff(peaks)
    T_est = np.mean(periods)
else:
    T_est = 2*np.pi*np.sqrt(L/g)

# ----------------------------
# Pendulum coordinates
# ----------------------------
x = L*np.sin(theta)
y = -L*np.cos(theta)

# ----------------------------
# Create figure with two panels
# ----------------------------
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,9))

# ---- Animation panel ----
ax1.set_xlim(-1.2,1.2)
ax1.set_ylim(-1.2,1.2)
ax1.set_aspect("equal")
ax1.grid(True)
ax1.set_title("Pendulum Simulation")

line, = ax1.plot([], [], 'o-', lw=3)

info_text = ax1.text(
    0.02,0.95,"",
    transform=ax1.transAxes,
    verticalalignment='top',
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.8)
)

# ---- Static graph ----
ax2.plot(t, theta, label="θ(t)", linewidth=2)
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Angle [rad]")
ax2.set_title("Static Graph: Pendulum Angle vs Time")
ax2.grid(True)

# period boundary lines
n_periods = int(t[-1] // T_est) + 1
for n in range(1, n_periods+1):
    boundary = n*T_est
    if boundary <= t[-1]:
        ax2.axvline(boundary, linestyle='--', alpha=0.5)

# moving point on static graph
point, = ax2.plot([], [], 'ro')

# ----------------------------
# Animation update function
# ----------------------------
def update(frame):

    line.set_data([0, x[frame]], [0, y[frame]])

    point.set_data([t[frame]], [theta[frame]])

    current_time = t[frame]
    current_period = int(current_time // T_est) + 1

    info_text.set_text(
        f"damping = {c:.2f} s⁻¹\n"
        f"estimated period ≈ {T_est:.2f} s\n"
        f"time = {current_time:.2f} s\n"
        f"current cycle = Period #{current_period}"
    )

    return line, point, info_text

ani = FuncAnimation(fig, update, frames=len(t), interval=40)

plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# ============================================================
# Cart-Pole Simulation V3
# Angle convention:
#   theta = 0   -> pole is upright (unstable)
#   theta > 0   -> pole tilts to one side
#   theta < 0   -> pole tilts to the other side
#
# State vector:
#   y = [x, x_dot, theta, theta_dot]
# ============================================================

# ----------------------------
# Physical parameters
# ----------------------------
M = 1.0      # cart mass [kg]
m = 0.2      # pole mass [kg]
L = 1.0      # pole length [m]
g = 9.81     # gravity [m/s^2]

# ----------------------------
# Simulation settings
# ----------------------------
t_start = 0.0
t_end = 5.0
num_points = 400

# Initial condition:
# [x, x_dot, theta, theta_dot]
# Start near upright with a small perturbation
y0 = [0.0, 0.0, 0.05, 0.0]

# Output video file name
video_filename = "cartpole_simulation.mp4"

# ----------------------------
# Cart-pole dynamics
# No control input yet: F = 0
# Upright position should be unstable
# ----------------------------
def cartpole_dynamics(t, y):
    x, x_dot, theta, theta_dot = y

    F = 0.0  # no control input

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    denom = M + m * sin_t**2

    x_ddot = (F - m * sin_t * (L * theta_dot**2 - g * cos_t)) / denom

    theta_ddot = (
        -F * cos_t
        + m * L * theta_dot**2 * sin_t * cos_t
        + (M + m) * g * sin_t
    ) / (L * denom)

    return [x_dot, x_ddot, theta_dot, theta_ddot]

# ----------------------------
# Solve ODE
# ----------------------------
t_eval = np.linspace(t_start, t_end, num_points)

solution = solve_ivp(
    cartpole_dynamics,
    (t_start, t_end),
    y0,
    t_eval=t_eval
)

t = solution.t
x = solution.y[0]
x_dot = solution.y[1]
theta = solution.y[2]
theta_dot = solution.y[3]

# ----------------------------
# Static analysis plots
# ----------------------------
fig1, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

axes[0].plot(t, x, label="Cart position x [m]")
axes[0].set_ylabel("x [m]")
axes[0].set_title("Cart Position vs Time")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(t, x_dot, label="Cart velocity x_dot [m/s]", color="green")
axes[1].set_ylabel("x_dot [m/s]")
axes[1].set_title("Cart Velocity vs Time")
axes[1].grid(True)
axes[1].legend()

axes[2].plot(t, theta, label="Pole angle theta [rad]", color="red")
axes[2].set_ylabel("theta [rad]")
axes[2].set_title("Pole Angle vs Time")
axes[2].grid(True)
axes[2].legend()

axes[3].plot(t, theta_dot, label="Pole angular velocity theta_dot [rad/s]", color="purple")
axes[3].set_xlabel("Time [s]")
axes[3].set_ylabel("theta_dot [rad/s]")
axes[3].set_title("Pole Angular Velocity vs Time")
axes[3].grid(True)
axes[3].legend()

plt.tight_layout()

# ----------------------------
# Phase portrait
# ----------------------------
fig2, ax_phase = plt.subplots(figsize=(6, 5))
ax_phase.plot(theta, theta_dot, color="darkorange")
ax_phase.set_xlabel("theta [rad]")
ax_phase.set_ylabel("theta_dot [rad/s]")
ax_phase.set_title("Phase Portrait: theta vs theta_dot")
ax_phase.grid(True)

# ----------------------------
# Animation preparation
# ----------------------------
pole_x = x + L * np.sin(theta)
pole_y = L * np.cos(theta)

fig3, ax_anim = plt.subplots(figsize=(8, 4))
ax_anim.set_xlim(np.min(x) - 1.5, np.max(x) + 1.5)
ax_anim.set_ylim(-0.5, 1.8)
ax_anim.set_aspect("equal")
ax_anim.grid(True)
ax_anim.set_title("Cart-Pole Simulation V3 (No Control, Unstable Upright)")

cart_width = 0.4
cart_height = 0.2

cart = plt.Rectangle((0, 0), cart_width, cart_height, color="blue")
ax_anim.add_patch(cart)

line, = ax_anim.plot([], [], "o-", lw=3)
time_text = ax_anim.text(0.02, 0.95, "", transform=ax_anim.transAxes, va="top")
theta_text = ax_anim.text(0.02, 0.88, "", transform=ax_anim.transAxes, va="top")

def update(frame):
    cart_center_x = x[frame]
    cart.set_xy((cart_center_x - cart_width / 2, -cart_height / 2))

    line.set_data([cart_center_x, pole_x[frame]], [0, pole_y[frame]])

    time_text.set_text(f"time = {t[frame]:.2f} s")
    theta_text.set_text(f"theta = {theta[frame]:.3f} rad")

    return cart, line, time_text, theta_text

ani = FuncAnimation(fig3, update, frames=len(t), interval=30)

# ----------------------------
# Save animation to MP4
# ----------------------------
ani.save(video_filename, writer="ffmpeg", fps=30)

print(f"Video saved as: {video_filename}")

# ----------------------------
# Show all figures
# ----------------------------
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# ----------------------------
# Physical parameters
# ----------------------------
M = 1.0      # cart mass [kg]
m = 0.2      # pole mass [kg]
L = 1.0      # pole length [m]
g = 9.81     # gravity [m/s^2]

# ----------------------------
# Cart-pole dynamics
# State = [x, x_dot, theta, theta_dot]
# theta = 0 means pole is upright
# ----------------------------
def cartpole_dynamics(t, y):
    x, x_dot, theta, theta_dot = y

    F = 0.0  # no control input yet

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    denom = M + m * sin_t**2

    x_ddot = (F + m * sin_t * (L * theta_dot**2 + g * cos_t)) / denom
    theta_ddot = (
        -F * cos_t
        - m * L * theta_dot**2 * sin_t * cos_t
        - (M + m) * g * sin_t
    ) / (L * denom)

    return [x_dot, x_ddot, theta_dot, theta_ddot]

# ----------------------------
# Simulation setup
# ----------------------------
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 400)

# Initial condition:
# x, x_dot, theta, theta_dot
# small initial tilt from upright
y0 = [0.0, 0.0, 0.2, 0.0]

solution = solve_ivp(cartpole_dynamics, t_span, y0, t_eval=t_eval)

t = solution.t
x = solution.y[0]
x_dot = solution.y[1]
theta = solution.y[2]
theta_dot = solution.y[3]

# ----------------------------
# Static plots
# ----------------------------
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))

ax1.plot(t, x, label="Cart position x [m]")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("x [m]")
ax1.set_title("Cart Position vs Time")
ax1.grid(True)
ax1.legend()

ax2.plot(t, theta, label="Pole angle theta [rad]", color="red")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("theta [rad]")
ax2.set_title("Pole Angle vs Time")
ax2.grid(True)
ax2.legend()

plt.tight_layout()

# ----------------------------
# Animation data
# ----------------------------
pole_x = x + L * np.sin(theta)
pole_y = L * np.cos(theta)

fig2, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(np.min(x) - 1.5, np.max(x) + 1.5)
ax.set_ylim(-0.5, 1.8)
ax.set_aspect("equal")
ax.grid(True)
ax.set_title("Real Cart-Pole Simulation (No Control)")

cart_width = 0.4
cart_height = 0.2

cart = plt.Rectangle((0, 0), cart_width, cart_height, color="blue")
ax.add_patch(cart)

line, = ax.plot([], [], 'o-', lw=3)
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

def update(frame):
    cart_center_x = x[frame]
    cart.set_xy((cart_center_x - cart_width / 2, -cart_height / 2))

    line.set_data([cart_center_x, pole_x[frame]], [0, pole_y[frame]])
    time_text.set_text(f"time = {t[frame]:.2f} s")

    return cart, line, time_text

ani = FuncAnimation(fig2, update, frames=len(t), interval=30)

plt.show()
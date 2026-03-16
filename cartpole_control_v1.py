import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# ============================================================
# Simple Cart-Pole Control V1
#
# theta = 0 is upright equilibrium
# Lightweight version for quick testing
# ============================================================

# ----------------------------
# Physical parameters
# ----------------------------
M = 1.0
m = 0.2
L = 1.0
g = 9.81

# ----------------------------
# Simulation settings
# ----------------------------
t_start = 0.0
t_end = 8.0
num_points = 500
max_force = 50.0

# Initial state
y0 = np.array([0.20, 0.30, 0.50, 0.80])

# ----------------------------
# Angle wrapping
# ----------------------------
def wrap_angle(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

# ----------------------------
# Linearized model for LQR
# ----------------------------
A = np.array([
    [0, 1, 0, 0],
    [0, 0, -(m*g)/M, 0],
    [0, 0, 0, 1],
    [0, 0, (M+m)*g/(L*M), 0]
])

B = np.array([
    [0],
    [1/M],
    [0],
    [-1/(L*M)]
])

Q = np.diag([10, 2, 250, 20])
R = np.array([[1]])

P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

# ----------------------------
# Nonlinear dynamics
# ----------------------------
def cartpole_dynamics(t, y):

    x, x_dot, theta, theta_dot = y

    theta_error = wrap_angle(theta)

    state = np.array([x, x_dot, theta_error, theta_dot])
    F = float(-(K @ state.reshape(-1,1)).item())
    F = np.clip(F, -max_force, max_force)

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    denom = M + m*sin_t**2

    x_ddot = (F + m*sin_t*(L*theta_dot**2 - g*cos_t)) / denom

    theta_ddot = (
        -F*cos_t
        - m*L*theta_dot**2*sin_t*cos_t
        + (M+m)*g*sin_t
    )/(L*denom)

    return [x_dot, x_ddot, theta_dot, theta_ddot]

# ----------------------------
# Solve system
# ----------------------------
t_eval = np.linspace(t_start, t_end, num_points)

sol = solve_ivp(
    cartpole_dynamics,
    (t_start, t_end),
    y0,
    t_eval=t_eval,
    max_step=0.02
)

t = sol.t
x = sol.y[0]
theta = sol.y[2]
theta_dot = sol.y[3]

theta_error = wrap_angle(theta)

# ----------------------------
# Control force history
# ----------------------------
F_hist = np.zeros_like(t)

for i in range(len(t)):
    state = np.array([x[i], sol.y[1][i], theta_error[i], theta_dot[i]])
    F_hist[i] = np.clip(float(-(K @ state.reshape(-1,1)).item()),
                        -max_force, max_force)

# ============================================================
# TIME HISTORY + PHASE PLOT
# ============================================================

fig1, axes = plt.subplots(2,2, figsize=(10,7))

# Cart position
axes[0,0].plot(t, x)
axes[0,0].set_title("Cart Position")
axes[0,0].set_ylabel("x [m]")
axes[0,0].grid(True)

# Pole angle
axes[0,1].plot(t, theta_error, color="red")
axes[0,1].axhline(0, linestyle="--", color="gray")
axes[0,1].set_title("Pole Angle")
axes[0,1].set_ylabel("theta [rad]")
axes[0,1].grid(True)

# Control force
axes[1,0].plot(t, F_hist, color="black")
axes[1,0].set_title("Control Force")
axes[1,0].set_xlabel("time [s]")
axes[1,0].set_ylabel("F [N]")
axes[1,0].grid(True)

# Phase portrait
axes[1,1].plot(theta_error, theta_dot, color="orange")
axes[1,1].set_title("Phase Portrait")
axes[1,1].set_xlabel("theta [rad]")
axes[1,1].set_ylabel("theta_dot [rad/s]")
axes[1,1].grid(True)

plt.tight_layout()

# ============================================================
# SIMPLE ANIMATION
# ============================================================

cart_width = 0.4
cart_height = 0.2
pivot_y = cart_height/2

pole_x = x + L*np.sin(theta_error)
pole_y = pivot_y + L*np.cos(theta_error)

fig2, ax = plt.subplots(figsize=(8,4))

ax.set_xlim(np.min(x)-1.5, np.max(x)+1.5)
ax.set_ylim(-0.3, 1.5)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Cart-Pole Animation")

ax.axhline(0,color="gray")

cart = Rectangle(
    (-cart_width/2, -cart_height/2),
    cart_width,
    cart_height,
    color="blue"
)

ax.add_patch(cart)

pole_line, = ax.plot([],[], "o-", lw=3)

def update(frame):

    cart_x = x[frame]

    cart.set_xy((cart_x-cart_width/2, -cart_height/2))

    pivot_x = cart_x

    pole_line.set_data(
        [pivot_x, pole_x[frame]],
        [pivot_y, pole_y[frame]]
    )

    return cart, pole_line

ani = FuncAnimation(fig2, update, frames=len(t), interval=25)

plt.show()
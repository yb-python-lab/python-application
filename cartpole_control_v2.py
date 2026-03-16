import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle

# ============================================================
# Cart-Pole Control Dashboard
# Correct convention:
#   theta = 0 -> upright equilibrium
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
t_end = 10.0
num_points = 800
max_force = 50.0
video_filename = "cartpole_control_dashboard.mp4"

# Initial condition: [x, x_dot, theta, theta_dot]
y0 = np.array([0.0, 0.0, 0.12, 0.0], dtype=float)

# ----------------------------
# Helper
# ----------------------------
def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2 * np.pi) - np.pi

# ----------------------------
# Linearized model around theta = 0
# ----------------------------
A = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, -(m * g) / M, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, ((M + m) * g) / (L * M), 0.0]
])

B = np.array([
    [0.0],
    [1.0 / M],
    [0.0],
    [-1.0 / (L * M)]
])

Q = np.diag([10.0, 2.0, 250.0, 20.0])
R = np.array([[1.0]])

P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P   # shape (1, 4)

# ----------------------------
# Nonlinear dynamics with LQR control
# ----------------------------
def cartpole_dynamics(t: float, y: np.ndarray) -> list[float]:
    x, x_dot, theta, theta_dot = y

    theta_error = wrap_angle(theta)

    state = np.array([x, x_dot, theta_error, theta_dot], dtype=float)
    F = float(-(K @ state.reshape(-1, 1)).item())
    F = float(np.clip(F, -max_force, max_force))

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    denom = M + m * sin_t**2

    x_ddot = (F + m * sin_t * (L * theta_dot**2 - g * cos_t)) / denom

    theta_ddot = (
        -F * cos_t
        - m * L * theta_dot**2 * sin_t * cos_t
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
    t_eval=t_eval,
    max_step=0.02
)

# Convert solution arrays explicitly
t = np.asarray(solution.t, dtype=float)
y = np.asarray(solution.y, dtype=float)

x = y[0, :]
x_dot = y[1, :]
theta = y[2, :]
theta_dot = y[3, :]

theta_error = wrap_angle(theta)

# Recompute control force history
F_hist = np.zeros_like(t)
for i in range(len(t)):
    state_i = np.array([x[i], x_dot[i], theta_error[i], theta_dot[i]], dtype=float)
    Fi = float(-(K @ state_i.reshape(-1, 1)).item())
    F_hist[i] = np.clip(Fi, -max_force, max_force)

# ----------------------------
# Geometry for animation
# ----------------------------
cart_width = 0.4
cart_height = 0.2
wheel_radius = 0.06
wheel_offset_x = 0.12
pivot_y = cart_height / 2.0

pole_x = x + L * np.sin(theta_error)
pole_y = pivot_y + L * np.cos(theta_error)

# ----------------------------
# Dashboard figure
# ----------------------------
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 3)

ax_cart = fig.add_subplot(gs[0, 0:2])
ax_phase = fig.add_subplot(gs[0, 2])
ax_x = fig.add_subplot(gs[1, 0])
ax_xdot = fig.add_subplot(gs[1, 1])
ax_theta = fig.add_subplot(gs[1, 2])
ax_thetadot = fig.add_subplot(gs[2, 0])
ax_force = fig.add_subplot(gs[2, 1])
ax_blank = fig.add_subplot(gs[2, 2])
ax_blank.axis("off")

# ----------------------------
# Cart animation panel
# ----------------------------
ax_cart.set_xlim(np.min(x) - 1.5, np.max(x) + 1.5)
ax_cart.set_ylim(-0.35, 1.5)
ax_cart.set_aspect("equal")
ax_cart.grid(True)
ax_cart.set_title("Cart-Pole Animation")
ax_cart.axhline(0.0, color="gray", linewidth=1)

cart = Rectangle(
    (-cart_width / 2, -cart_height / 2),
    cart_width,
    cart_height,
    color="blue"
)
ax_cart.add_patch(cart)

left_wheel = Circle((0.0, 0.0), wheel_radius, facecolor="black", edgecolor="black")
right_wheel = Circle((0.0, 0.0), wheel_radius, facecolor="black", edgecolor="black")
ax_cart.add_patch(left_wheel)
ax_cart.add_patch(right_wheel)

left_spoke, = ax_cart.plot([], [], color="white", linewidth=2)
right_spoke, = ax_cart.plot([], [], color="white", linewidth=2)

pole_line, = ax_cart.plot([], [], "o-", lw=3)

time_text = ax_cart.text(0.02, 0.95, "", transform=ax_cart.transAxes, va="top")
theta_text = ax_cart.text(0.02, 0.88, "", transform=ax_cart.transAxes, va="top")
force_text = ax_cart.text(0.02, 0.81, "", transform=ax_cart.transAxes, va="top")

# ----------------------------
# Phase portrait
# ----------------------------
ax_phase.plot(theta_error, theta_dot, color="darkorange")
ax_phase.set_xlabel("theta [rad]")
ax_phase.set_ylabel("theta_dot [rad/s]")
ax_phase.set_title("Phase Portrait")
ax_phase.grid(True)
phase_point, = ax_phase.plot([], [], "ro", markersize=8)

# ----------------------------
# Time-history plots + markers
# ----------------------------
ax_x.plot(t, x, color="blue")
ax_x.set_title("Cart Position")
ax_x.set_xlabel("Time [s]")
ax_x.set_ylabel("x [m]")
ax_x.grid(True)
x_point, = ax_x.plot([], [], "ro", markersize=6)

ax_xdot.plot(t, x_dot, color="green")
ax_xdot.set_title("Cart Velocity")
ax_xdot.set_xlabel("Time [s]")
ax_xdot.set_ylabel("x_dot [m/s]")
ax_xdot.grid(True)
xdot_point, = ax_xdot.plot([], [], "ro", markersize=6)

ax_theta.plot(t, theta_error, color="red")
ax_theta.axhline(0.0, color="gray", linestyle="--")
ax_theta.set_title("Pole Angle")
ax_theta.set_xlabel("Time [s]")
ax_theta.set_ylabel("theta [rad]")
ax_theta.grid(True)
theta_point, = ax_theta.plot([], [], "ro", markersize=6)

ax_thetadot.plot(t, theta_dot, color="purple")
ax_thetadot.set_title("Pole Angular Velocity")
ax_thetadot.set_xlabel("Time [s]")
ax_thetadot.set_ylabel("theta_dot [rad/s]")
ax_thetadot.grid(True)
thetadot_point, = ax_thetadot.plot([], [], "ro", markersize=6)

ax_force.plot(t, F_hist, color="black")
ax_force.set_title("Control Force")
ax_force.set_xlabel("Time [s]")
ax_force.set_ylabel("F [N]")
ax_force.grid(True)
force_point, = ax_force.plot([], [], "ro", markersize=6)

plt.tight_layout()

# ----------------------------
# Update function
# ----------------------------
def update(frame: int):
    cart_center_x = x[frame]

    # Cart body
    cart.set_xy((cart_center_x - cart_width / 2, -cart_height / 2))

    # Wheels
    wheel_y = -cart_height / 2 - wheel_radius
    left_center = (cart_center_x - wheel_offset_x, wheel_y)
    right_center = (cart_center_x + wheel_offset_x, wheel_y)
    left_wheel.center = left_center
    right_wheel.center = right_center

    # Wheel rotation
    phi = cart_center_x / wheel_radius
    spoke_dx = wheel_radius * np.cos(phi)
    spoke_dy = wheel_radius * np.sin(phi)

    left_spoke.set_data(
        [left_center[0], left_center[0] + spoke_dx],
        [left_center[1], left_center[1] + spoke_dy]
    )
    right_spoke.set_data(
        [right_center[0], right_center[0] + spoke_dx],
        [right_center[1], right_center[1] + spoke_dy]
    )

    # Pole
    pivot_x = cart_center_x
    pole_line.set_data(
        [pivot_x, pole_x[frame]],
        [pivot_y, pole_y[frame]]
    )

    # Phase marker
    phase_point.set_data([theta_error[frame]], [theta_dot[frame]])

    # Time-history markers
    x_point.set_data([t[frame]], [x[frame]])
    xdot_point.set_data([t[frame]], [x_dot[frame]])
    theta_point.set_data([t[frame]], [theta_error[frame]])
    thetadot_point.set_data([t[frame]], [theta_dot[frame]])
    force_point.set_data([t[frame]], [F_hist[frame]])

    # Text
    time_text.set_text(f"time = {t[frame]:.2f} s")
    theta_text.set_text(f"theta = {theta_error[frame]:.3f} rad")
    force_text.set_text(f"F = {F_hist[frame]:.2f} N")

    return (
        cart,
        left_wheel,
        right_wheel,
        left_spoke,
        right_spoke,
        pole_line,
        phase_point,
        x_point,
        xdot_point,
        theta_point,
        thetadot_point,
        force_point,
        time_text,
        theta_text,
        force_text,
    )

ani = FuncAnimation(fig, update, frames=len(t), interval=25)

# ----------------------------
# Save video
# ----------------------------
ani.save(video_filename, writer="ffmpeg", fps=30)
print(f"Video saved as: {video_filename}")

plt.show()
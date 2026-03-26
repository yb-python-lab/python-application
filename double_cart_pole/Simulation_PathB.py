import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================================================
# Fast Path B: Double Cart-Pole with Mechanics-Based LQR
# ---------------------------------------------------------
# This is a FAST mechanics-based LINEARIZED model around the
# upright equilibrium, not the full symbolic nonlinear model.
#
# Generalized coordinates:
#   q = [x, th1, th2]
#
# State:
#   X = [x, x_dot, th1, th1_dot, th2, th2_dot]
#
# Angle convention:
#   th = 0  --> upright
#
# Link angles are ABSOLUTE angles measured from upright.
# =========================================================

# -----------------------------
# User options
# -----------------------------
SAVE_VIDEO = True
VIDEO_FILENAME = "double_cart_pole_fast_pathB_lqr.mp4"
PRINT_MATRICES = True

# -----------------------------
# Physical parameters
# -----------------------------
M = 1.0           # cart mass [kg]
m1 = 0.5          # link 1 mass [kg]
m2 = 0.5          # link 2 mass [kg]

l1 = 1.0          # link 1 full length [m]
l2 = 1.0          # link 2 full length [m]

lc1 = 0.5         # link 1 COM distance [m]
lc2 = 0.5         # link 2 COM distance [m]

I1 = (1 / 12) * m1 * l1**2
I2 = (1 / 12) * m2 * l2**2

g = 9.81

bx = 0.10         # cart damping
b1 = 0.03         # joint 1 damping
b2 = 0.03         # joint 2 damping

u_max = 150.0      # actuator saturation [N]

# animation
cart_width = 0.40
cart_height = 0.20

# -----------------------------
# Build mechanics-based linear model
# M0 qdd + D qd - Kg q = Bq u
# where q = [x, th1, th2]
# ---------------------------------------------------------
# Derived from small-angle linearization around upright:
#   sin(th) ~ th
#   cos(th) ~ 1
# ---------------------------------------------------------
M0 = np.array([
    [M + m1 + m2,        m1 * lc1 + m2 * l1,    m2 * lc2],
    [m1 * lc1 + m2 * l1, I1 + m1 * lc1**2 + m2 * l1**2, m2 * l1 * lc2],
    [m2 * lc2,           m2 * l1 * lc2,         I2 + m2 * lc2**2]
], dtype=float)

Dq = np.array([
    [bx, 0.0, 0.0],
    [0.0, b1, 0.0],
    [0.0, 0.0, b2]
], dtype=float)

Kg = np.array([
    [0.0, 0.0, 0.0],
    [0.0, (m1 * lc1 + m2 * l1) * g, 0.0],
    [0.0, 0.0, m2 * lc2 * g]
], dtype=float)

Bq = np.array([[1.0], [0.0], [0.0]], dtype=float)

M0_inv = np.linalg.inv(M0)

# state X = [x, x_dot, th1, th1_dot, th2, th2_dot]
A = np.zeros((6, 6), dtype=float)
B = np.zeros((6, 1), dtype=float)

A[0, 1] = 1.0
A[2, 3] = 1.0
A[4, 5] = 1.0

# qdd = M0^{-1} ( -D qd + Kg q + Bq u )
A_acc_q = M0_inv @ Kg
A_acc_qd = -M0_inv @ Dq
B_acc = M0_inv @ Bq

# rows for x_ddot, th1_ddot, th2_ddot
A[1, 0] = A_acc_q[0, 0]
A[1, 2] = A_acc_q[0, 1]
A[1, 4] = A_acc_q[0, 2]
A[1, 1] = A_acc_qd[0, 0]
A[1, 3] = A_acc_qd[0, 1]
A[1, 5] = A_acc_qd[0, 2]

A[3, 0] = A_acc_q[1, 0]
A[3, 2] = A_acc_q[1, 1]
A[3, 4] = A_acc_q[1, 2]
A[3, 1] = A_acc_qd[1, 0]
A[3, 3] = A_acc_qd[1, 1]
A[3, 5] = A_acc_qd[1, 2]

A[5, 0] = A_acc_q[2, 0]
A[5, 2] = A_acc_q[2, 1]
A[5, 4] = A_acc_q[2, 2]
A[5, 1] = A_acc_qd[2, 0]
A[5, 3] = A_acc_qd[2, 1]
A[5, 5] = A_acc_qd[2, 2]

B[1, 0] = B_acc[0, 0]
B[3, 0] = B_acc[1, 0]
B[5, 0] = B_acc[2, 0]

if PRINT_MATRICES:
    np.set_printoptions(precision=4, suppress=True)
    print("A matrix:")
    print(A)
    print("\nB matrix:")
    print(B)

# -----------------------------
# LQR design
# Tune Q and R if needed
# -----------------------------
Q = np.diag([20.0, 8.0, 250.0, 25.0, 250.0, 25.0])
R = np.array([[1.0]])

P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

print("\nLQR gain K:")
print(K)

# equilibrium
Xeq = np.zeros(6, dtype=float)

# -----------------------------
# Controller
# -----------------------------
def controller(state):
    state = np.asarray(state, dtype=float).reshape(6,)
    x_err = state - Xeq
    u = -K @ x_err
    u = float(np.asarray(u).squeeze())
    u = np.clip(u, -u_max, u_max)
    return u

# -----------------------------
# Closed-loop linear mechanics model
# with actuator saturation
# -----------------------------
last_print_time = -1.0

def dynamics(t, state):
    global last_print_time

    if t - last_print_time >= 0.5:
        print(f"Simulation time = {t:.2f} s")
        last_print_time = t

    state = np.asarray(state, dtype=float).reshape(6,)
    u = controller(state)

    # linear closed-loop dynamics with saturated input
    xdot = A @ state + B.flatten() * u
    return xdot

# -----------------------------
# Initial condition
# Start small because this is linearized around upright
# -----------------------------
X0 = np.array([
    0.0,                  # x
    0.0,                  # x_dot
    np.deg2rad(15.0),      # th1
    0.0,                  # th1_dot
    np.deg2rad(-12.0),     # th2
    0.0                   # th2_dot
], dtype=float)

# -----------------------------
# Simulation
# -----------------------------
print("\nStarting fast Path B simulation...")

t_start = 0.0
t_end = 10.0
num_points = 1200
t_eval = np.linspace(t_start, t_end, num_points)

sol = solve_ivp(
    dynamics,
    (t_start, t_end),
    X0,
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-8,
    method="RK45"
)

if not sol.success:
    raise RuntimeError("Simulation failed: " + sol.message)

print("Simulation finished.")

# -----------------------------
# Extract results
# -----------------------------
t = sol.t
x = sol.y[0]
xd = sol.y[1]
th1 = sol.y[2]
th1d = sol.y[3]
th2 = sol.y[4]
th2d = sol.y[5]

u_hist = np.array([controller(sol.y[:, i]) for i in range(sol.y.shape[1])])

print(f"max |x|      = {np.max(np.abs(x)):.4f} m")
print(f"max |theta1| = {np.max(np.abs(th1)):.4f} rad")
print(f"max |theta2| = {np.max(np.abs(th2)):.4f} rad")
print(f"max |u|      = {np.max(np.abs(u_hist)):.4f} N")

# -----------------------------
# Plots
# -----------------------------
plt.figure(figsize=(9, 5))
plt.plot(t, th1, label="theta1 (rad)")
plt.plot(t, th2, label="theta2 (rad)")
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.title("Fast Path B Controlled: Double Cart-Pole Angles")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(9, 5))
plt.plot(t, x, label="cart position x (m)")
plt.xlabel("Time [s]")
plt.ylabel("Cart Position [m]")
plt.title("Fast Path B Controlled: Cart Motion")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(9, 5))
plt.plot(t, u_hist, label="control input u (N)")
plt.xlabel("Time [s]")
plt.ylabel("Force [N]")
plt.title("Fast Path B Controlled: Input Force")
plt.grid(True)
plt.legend()
plt.tight_layout()

# -----------------------------
# Animation geometry
# Using absolute angles from upright
# -----------------------------
pivot_x = x
pivot_y = np.zeros_like(x) + cart_height / 2.0

x1 = pivot_x + l1 * np.sin(th1)
y1 = pivot_y + l1 * np.cos(th1)

x2 = x1 + l2 * np.sin(th2)
y2 = y1 + l2 * np.cos(th2)

# -----------------------------
# Animation
# -----------------------------
fig_anim, ax_anim = plt.subplots(figsize=(10, 5))
ax_anim.set_title("Fast Path B Controlled: Double Cart-Pole Animation")
ax_anim.set_xlabel("X [m]")
ax_anim.set_ylabel("Y [m]")
ax_anim.grid(True)
ax_anim.set_aspect("equal", adjustable="box")

x_center = 0.5 * (np.min(x) + np.max(x))
x_span = max(4.0, np.max(x) - np.min(x) + 4.0)

xmin = x_center - 0.5 * x_span
xmax = x_center + 0.5 * x_span
ymin = -0.5
ymax = cart_height + l1 + l2 + 0.5

ax_anim.set_xlim(xmin, xmax)
ax_anim.set_ylim(ymin, ymax)

ax_anim.plot([xmin, xmax], [0, 0], "k-", linewidth=1)

cart_line, = ax_anim.plot([], [], linewidth=3, label="cart")
link1_line, = ax_anim.plot([], [], "o-", linewidth=2, markersize=6, label="link 1")
link2_line, = ax_anim.plot([], [], "o-", linewidth=2, markersize=6, label="link 2")
time_text = ax_anim.text(0.02, 0.95, "", transform=ax_anim.transAxes)

def init_animation():
    cart_line.set_data([], [])
    link1_line.set_data([], [])
    link2_line.set_data([], [])
    time_text.set_text("")
    return cart_line, link1_line, link2_line, time_text

def update_animation(i):
    cart_left = x[i] - cart_width / 2.0
    cart_right = x[i] + cart_width / 2.0
    cart_bottom = 0.0
    cart_top = cart_height

    cart_xs = [cart_left, cart_right, cart_right, cart_left, cart_left]
    cart_ys = [cart_bottom, cart_bottom, cart_top, cart_top, cart_bottom]
    cart_line.set_data(cart_xs, cart_ys)

    link1_line.set_data([pivot_x[i], x1[i]], [pivot_y[i], y1[i]])
    link2_line.set_data([x1[i], x2[i]], [y1[i], y2[i]])

    time_text.set_text(f"t = {t[i]:.2f} s")
    return cart_line, link1_line, link2_line, time_text

frame_step = 4

ani = FuncAnimation(
    fig_anim,
    update_animation,
    frames=range(0, len(t), frame_step),
    init_func=init_animation,
    blit=True,
    interval=20,
    repeat=True
)

if SAVE_VIDEO:
    try:
        ani.save(VIDEO_FILENAME, writer="ffmpeg", fps=30)
        print(f"Video saved successfully: {VIDEO_FILENAME}")
    except Exception as e:
        print("MP4 save failed. Check whether ffmpeg is installed.")
        print("Error:", e)

plt.show()
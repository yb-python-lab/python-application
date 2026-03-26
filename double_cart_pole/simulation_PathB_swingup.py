import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================================================
# Double Cart-Pole: Direct Trajectory Optimization + LQR
# ---------------------------------------------------------
# State:
#   X = [x, x_dot, th1, th1_dot, th2, th2_dot]
#
# Angle convention:
#   th = 0   -> upright
#   th = pi  -> downward
#
# Link angles are ABSOLUTE angles measured from upright.
# =========================================================

# -----------------------------
# User options
# -----------------------------
SAVE_VIDEO = True
VIDEO_FILENAME = "double_cart_pole_trajopt_lqr.mp4"
PRINT_PROGRESS = True
PRINT_MATRICES = False

# -----------------------------
# Physical parameters
# -----------------------------
M = 1.0
m1 = 0.5
m2 = 0.5

l1 = 1.0
l2 = 1.0

lc1 = 0.5
lc2 = 0.5

I1 = (1.0 / 12.0) * m1 * l1**2
I2 = (1.0 / 12.0) * m2 * l2**2

g = 9.81

bx = 0.10
b1 = 0.03
b2 = 0.03

u_max = 200.0

# animation
cart_width = 0.40
cart_height = 0.20

# -----------------------------
# Simulation settings
# -----------------------------
t_start=0.0
dt = 0.01
t_end = 20.0

# Trajectory optimization settings
opt_dt = 0.5               # coarse step for optimization rollout
N_opt = 6               # 120 * 0.05 = 6.0 s swing-up horizon
u_hold_steps = 1            # each optimized control used for 1 coarse step

# -----------------------------
# Helper functions
# -----------------------------
def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def angle_error_to_upright(th):
    return wrap_angle(th)


def angle_error_to_downward(th):
    return wrap_angle(th - np.pi)


def state_error_to_upright(state):
    x, xd, th1, th1d, th2, th2d = state
    return np.array([
        x,
        xd,
        angle_error_to_upright(th1),
        th1d,
        angle_error_to_upright(th2),
        th2d
    ], dtype=float)


# =========================================================
# 1) Nonlinear mechanics model
# =========================================================
def mass_matrix(q):
    x, th1, th2 = q

    c1 = np.cos(th1)
    c2 = np.cos(th2)
    c12 = np.cos(th1 - th2)

    M11 = M + m1 + m2
    M12 = (m1 * lc1 + m2 * l1) * c1
    M13 = m2 * lc2 * c2

    M22 = I1 + m1 * lc1**2 + m2 * l1**2
    M23 = m2 * l1 * lc2 * c12

    M33 = I2 + m2 * lc2**2

    return np.array([
        [M11, M12, M13],
        [M12, M22, M23],
        [M13, M23, M33]
    ], dtype=float)


def coriolis_centrifugal(q, qd):
    x, th1, th2 = q
    xd, th1d, th2d = qd

    s1 = np.sin(th1)
    s2 = np.sin(th2)
    s12 = np.sin(th1 - th2)

    h1 = -(m1 * lc1 + m2 * l1) * s1 * th1d**2 - m2 * lc2 * s2 * th2d**2
    h2 = m2 * l1 * lc2 * s12 * th2d**2
    h3 = -m2 * l1 * lc2 * s12 * th1d**2

    return np.array([h1, h2, h3], dtype=float)


def gravity_vector(q):
    x, th1, th2 = q

    g1 = 0.0
    g2 = -(m1 * lc1 + m2 * l1) * g * np.sin(th1)
    g3 = -m2 * lc2 * g * np.sin(th2)

    return np.array([g1, g2, g3], dtype=float)


def damping_vector(qd):
    xd, th1d, th2d = qd
    return np.array([bx * xd, b1 * th1d, b2 * th2d], dtype=float)


def nonlinear_acceleration(state, u):
    x, xd, th1, th1d, th2, th2d = state

    q = np.array([x, th1, th2], dtype=float)
    qd = np.array([xd, th1d, th2d], dtype=float)

    Mq = mass_matrix(q)
    h = coriolis_centrifugal(q, qd)
    G = gravity_vector(q)
    D = damping_vector(qd)
    Bq = np.array([1.0, 0.0, 0.0], dtype=float)

    rhs = Bq * u - h - G - D
    qdd = np.linalg.solve(Mq, rhs)
    return qdd


def nonlinear_dynamics(state, u):
    x, xd, th1, th1d, th2, th2d = state
    xdd, th1dd, th2dd = nonlinear_acceleration(state, u)
    return np.array([xd, xdd, th1d, th1dd, th2d, th2dd], dtype=float)


def rk4_step(state, u, h):
    k1 = nonlinear_dynamics(state, u)
    k2 = nonlinear_dynamics(state + 0.5 * h * k1, u)
    k3 = nonlinear_dynamics(state + 0.5 * h * k2, u)
    k4 = nonlinear_dynamics(state + h * k3, u)
    nxt = state + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # wrap angles
    nxt[2] = wrap_angle(nxt[2])
    nxt[4] = wrap_angle(nxt[4])
    return nxt


# =========================================================
# 2) Upright linearization for LQR
# =========================================================
M0 = mass_matrix([0.0, 0.0, 0.0])

Kg = np.array([
    [0.0, 0.0, 0.0],
    [0.0, (m1 * lc1 + m2 * l1) * g, 0.0],
    [0.0, 0.0, m2 * lc2 * g]
], dtype=float)

Dq = np.array([
    [bx, 0.0, 0.0],
    [0.0, b1, 0.0],
    [0.0, 0.0, b2]
], dtype=float)

Bq = np.array([[1.0], [0.0], [0.0]], dtype=float)

M0_inv = np.linalg.inv(M0)

A = np.zeros((6, 6), dtype=float)
B = np.zeros((6, 1), dtype=float)

A[0, 1] = 1.0
A[2, 3] = 1.0
A[4, 5] = 1.0

A_acc_q = M0_inv @ Kg
A_acc_qd = -M0_inv @ Dq
B_acc = M0_inv @ Bq

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

Q_lqr = np.diag([8.0, 4.0, 600.0, 80.0, 600.0, 80.0])
R_lqr = np.array([[1.0]])

P = solve_continuous_are(A, B, Q_lqr, R_lqr)
K = np.linalg.inv(R_lqr) @ B.T @ P
Xeq = np.zeros(6, dtype=float)


def controller_lqr(state):
    err = state_error_to_upright(state)
    u = -K @ err
    u = float(np.asarray(u).squeeze())
    return np.clip(u, -u_max, u_max)


# =========================================================
# 3) Trajectory optimization (direct shooting)
# =========================================================
def rollout_coarse(x0, u_seq):
    """
    Roll out coarse dynamics using piecewise-constant controls.
    """
    x = x0.copy()
    traj = [x.copy()]
    for uk in u_seq:
        u = np.clip(uk, -u_max, u_max)
        for _ in range(u_hold_steps):
            x = rk4_step(x, u, opt_dt)
        traj.append(x.copy())
    return np.array(traj)


def objective(u_seq, x0):
    """
    Cost for direct shooting.
    Goal:
    - bring system near upright at final time
    - penalize cart drift and excessive force
    - encourage both links upright
    """
    traj = rollout_coarse(x0, u_seq)
    xf = traj[-1]
    ef = state_error_to_upright(xf)

    J = 0.0

    # running cost
    for k in range(len(u_seq)):
        xk = traj[k]
        ek = state_error_to_upright(xk)

        J += 0.02 * (u_seq[k] ** 2)
        J += 0.02 * (xk[0] ** 2)
        J += 0.01 * (xk[1] ** 2)

        # reward getting closer to upright during trajectory
        J += 0.2 * (ek[2] ** 2)
        J += 0.2 * (ek[4] ** 2)

    # terminal cost
    J += 800.0 * (ef[2] ** 2)
    J += 800.0 * (ef[4] ** 2)
    J += 120.0 * (ef[3] ** 2)
    J += 120.0 * (ef[5] ** 2)
    J += 40.0 * (ef[0] ** 2)
    J += 20.0 * (ef[1] ** 2)

    return float(J)


def optimize_swingup_controls(x0):
    """
    Find a coarse open-loop force sequence to reach upright neighborhood.
    """
    if PRINT_PROGRESS:
        print("Starting direct trajectory optimization...")

    # initial guess: mild alternating pulse
    u0 = np.zeros(N_opt)
    for k in range(N_opt):
        if k < 20:
            u0[k] = 40.0
        elif k < 40:
            u0[k] = -40.0
        elif k < 60:
            u0[k] = 50.0
        elif k < 80:
            u0[k] = -50.0
        else:
            u0[k] = 0.0

    bounds = [(-u_max, u_max)] * N_opt

    result = minimize(
        objective,
        u0,
        args=(x0,),
        method="SLSQP",
        bounds=bounds,
        options={
            "maxiter": 120,
            "ftol": 1e-3,
            "disp": PRINT_PROGRESS
        }
    )

    if PRINT_PROGRESS:
        print("Trajectory optimization success:", result.success)
        print("Final optimization cost:", result.fun)

    return np.clip(result.x, -u_max, u_max), result


# =========================================================
# 4) Capture region + hybrid logic
# =========================================================
def near_upright_enter_region(state):
    e = state_error_to_upright(state)
    return (
        abs(e[2]) < np.deg2rad(25.0)
        and abs(e[4]) < np.deg2rad(25.0)
        and abs(e[3]) < 4.0
        and abs(e[5]) < 4.0
    )


def outside_lqr_hold_region(state):
    e = state_error_to_upright(state)
    return (
        abs(e[2]) > np.deg2rad(40.0)
        or abs(e[4]) > np.deg2rad(40.0)
    )


# =========================================================
# 5) Main simulation:
#    optimized swing-up input, then LQR takeover
# =========================================================
print("Building double cart-pole trajectory-optimization + LQR simulation...")
print("LQR gain K:")
print(K)

X0 = np.array([
    0.0,
    0.0,
    np.pi - 0.18,
    0.0,
    np.pi + 0.12,
    0.0
], dtype=float)

u_open_loop, opt_result = optimize_swingup_controls(X0)

num_steps = int((t_end - t_start) / dt) + 1
t = np.linspace(t_start, t_end, num_steps)

X_hist = np.zeros((num_steps, 6), dtype=float)
u_hist = np.zeros(num_steps, dtype=float)
mode_hist = np.zeros(num_steps, dtype=float)   # 0=open-loop trajopt, 1=LQR

state = X0.copy()
X_hist[0] = state

mode_state = 0
progress_interval = max(1, int(0.5 / dt))

opt_segment_duration = opt_dt * u_hold_steps
opt_total_time = N_opt * opt_segment_duration

for i in range(1, num_steps):
    current_t = t[i - 1]

    # phase 1: use optimized open-loop control
    if mode_state == 0:
        if current_t < opt_total_time:
            idx = min(int(current_t / opt_segment_duration), N_opt - 1)
            u = float(u_open_loop[idx])

            # if already near upright earlier than planned, hand over to LQR
            if near_upright_enter_region(state):
                mode_state = 1
                u = controller_lqr(state)
        else:
            # after optimized horizon, try LQR if near upright, otherwise keep last optimized input gently
            if near_upright_enter_region(state):
                mode_state = 1
                u = controller_lqr(state)
            else:
                u = 0.5 * float(u_open_loop[-1])

    else:
        # phase 2: LQR stabilization with hysteresis
        if outside_lqr_hold_region(state):
            # fall back to zero or mild continuation if you leave the region
            mode_state = 0
            u = 0.5 * float(u_open_loop[-1])
        else:
            u = controller_lqr(state)

    u = np.clip(u, -u_max, u_max)

    if PRINT_PROGRESS and (i % progress_interval == 0):
        mode_text = "LQR" if mode_state == 1 else "TRAJOPT"
        print(f"Simulation time = {current_t:.2f} s | mode = {mode_text} | u = {u:.2f}")

    state = rk4_step(state, u, dt)

    X_hist[i] = state
    u_hist[i] = u
    mode_hist[i] = mode_state

print("Simulation finished.")

# -----------------------------
# Extract
# -----------------------------
x = X_hist[:, 0]
x_dot = X_hist[:, 1]
th1 = X_hist[:, 2]
th1_dot = X_hist[:, 3]
th2 = X_hist[:, 4]
th2_dot = X_hist[:, 5]

th1_err = np.array([angle_error_to_upright(v) for v in th1])
th2_err = np.array([angle_error_to_upright(v) for v in th2])

print(f"max |x|          = {np.max(np.abs(x)):.4f} m")
print(f"max |theta1 err| = {np.max(np.abs(th1_err)):.4f} rad")
print(f"max |theta2 err| = {np.max(np.abs(th2_err)):.4f} rad")
print(f"max |u|          = {np.max(np.abs(u_hist)):.4f} N")

# -----------------------------
# Plots
# -----------------------------
plt.figure(figsize=(9, 5))
plt.plot(t, th1_err, label="theta1 error to upright (rad)")
plt.plot(t, th2_err, label="theta2 error to upright (rad)")
plt.xlabel("Time [s]")
plt.ylabel("Angle error [rad]")
plt.title("Double Cart-Pole Trajectory Optimization + LQR: Angle Errors")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(9, 5))
plt.plot(t, x, label="cart position x (m)")
plt.xlabel("Time [s]")
plt.ylabel("Cart Position [m]")
plt.title("Double Cart-Pole Trajectory Optimization + LQR: Cart Motion")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(9, 5))
plt.plot(t, u_hist, label="control input u (N)")
plt.xlabel("Time [s]")
plt.ylabel("Force [N]")
plt.title("Double Cart-Pole Trajectory Optimization + LQR: Input Force")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(9, 4))
plt.plot(t, mode_hist, label="mode (0=trajopt, 1=LQR)")
plt.xlabel("Time [s]")
plt.ylabel("Mode")
plt.title("Controller Mode")
plt.grid(True)
plt.legend()
plt.tight_layout()

# -----------------------------
# Animation geometry
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
ax_anim.set_title("Double Cart-Pole Trajectory Optimization + LQR")
ax_anim.set_xlabel("X [m]")
ax_anim.set_ylabel("Y [m]")
ax_anim.grid(True)
ax_anim.set_aspect("equal", adjustable="box")

x_center = 0.5 * (np.min(x) + np.max(x))
x_span = max(8.0, np.max(x) - np.min(x) + 4.0)

xmin = x_center - 0.5 * x_span
xmax = x_center + 0.5 * x_span
ymin = -2.5
ymax = cart_height + l1 + l2 + 0.5

ax_anim.set_xlim(xmin, xmax)
ax_anim.set_ylim(ymin, ymax)

ax_anim.plot([xmin, xmax], [0, 0], "k-", linewidth=1)

cart_line, = ax_anim.plot([], [], linewidth=3)
link1_line, = ax_anim.plot([], [], "o-", linewidth=2, markersize=6)
link2_line, = ax_anim.plot([], [], "o-", linewidth=2, markersize=6)
time_text = ax_anim.text(0.02, 0.95, "", transform=ax_anim.transAxes)
mode_text_artist = ax_anim.text(0.02, 0.90, "", transform=ax_anim.transAxes)

def init_animation():
    cart_line.set_data([], [])
    link1_line.set_data([], [])
    link2_line.set_data([], [])
    time_text.set_text("")
    mode_text_artist.set_text("")
    return cart_line, link1_line, link2_line, time_text, mode_text_artist


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
    mode_text_artist.set_text("mode = LQR" if mode_hist[i] > 0.5 else "mode = TRAJOPT")

    return cart_line, link1_line, link2_line, time_text, mode_text_artist


frame_step = max(1, int(0.02 / dt))

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
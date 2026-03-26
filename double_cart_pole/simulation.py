import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================================================
# Double Cart-Pole Simulation - Path A with corrected control
# =========================================================

# -----------------------------
# User options
# -----------------------------
SAVE_VIDEO = True
VIDEO_FILENAME = "double_cart_pole_pathA_controlled_fixed.mp4"

# -----------------------------
# Parameters
# -----------------------------
g = 9.81

M = 1.0
m1 = 0.5
m2 = 0.5

l1 = 1.0
l2 = 1.0

cart_damping = 0.25
joint1_damping = 0.10
joint2_damping = 0.10

velocity_coupling = 0.08
angle_coupling = 0.20

u_max = 80.0

cart_width = 0.40
cart_height = 0.20

# -----------------------------
# Control gains
# Angles measured from upright: th = 0 is upright
# IMPORTANT:
# - cart position/velocity -> negative feedback
# - pendulum angle/angular velocity -> positive feedback
#   because cart acceleration must counter the unstable upright mode
# -----------------------------
Kx = 2.0
Kxd = 4.0

Kth1 = 35.0
Kth1d = 8.0

Kth2 = 25.0
Kth2d = 6.0


def controller(X):
    x, x_dot, th1, th1_dot, th2, th2_dot = X

    u = (
        -Kx * x
        -Kxd * x_dot
        +Kth1 * th1
        +Kth1d * th1_dot
        +Kth2 * th2
        +Kth2d * th2_dot
    )

    return np.clip(u, -u_max, u_max)


# -----------------------------
# Dynamics
# State X = [x, x_dot, th1, th1_dot, th2, th2_dot]
# -----------------------------
def dynamics(t, X):
    x, x_dot, th1, th1_dot, th2, th2_dot = X

    u = controller(X)

    pend_force = (
        m1 * l1 * (th1_dot**2) * np.sin(th1)
        + m2 * l2 * (th2_dot**2) * np.sin(th2)
        + m1 * g * np.sin(th1) * np.cos(th1)
        + m2 * g * np.sin(th2) * np.cos(th2)
    )

    effective_mass = (
        M
        + m1 * (1.0 - np.cos(th1)**2)
        + m2 * (1.0 - np.cos(th2)**2)
        + 1e-6
    )

    x_ddot = (
        u
        + pend_force
        - cart_damping * x_dot
    ) / effective_mass

    delta = th1 - th2
    coupling_12 = angle_coupling * np.sin(delta) + velocity_coupling * (th1_dot - th2_dot)
    coupling_21 = -angle_coupling * np.sin(delta) - velocity_coupling * (th2_dot - th1_dot)

    th1_ddot = (
        (g / l1) * np.sin(th1)
        - (x_ddot / l1) * np.cos(th1)
        - joint1_damping * th1_dot
        - coupling_12
    )

    th2_ddot = (
        (g / l2) * np.sin(th2)
        - (x_ddot / l2) * np.cos(th2)
        - joint2_damping * th2_dot
        - coupling_21
    )

    return [x_dot, x_ddot, th1_dot, th1_ddot, th2_dot, th2_ddot]


# -----------------------------
# Initial conditions
# Use smaller angles first
# -----------------------------
x0 = 0.0
x_dot0 = 0.0

th1_0 = np.deg2rad(2.0)
th1_dot0 = 0.0

th2_0 = np.deg2rad(-1.5)
th2_dot0 = 0.0

X0 = [x0, x_dot0, th1_0, th1_dot0, th2_0, th2_dot0]


# -----------------------------
# Simulation
# -----------------------------
t_start = 0.0
t_end = 10.0
num_points = 1500
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

# -----------------------------
# Extract
# -----------------------------
t = sol.t
x = sol.y[0]
x_dot = sol.y[1]
th1 = sol.y[2]
th1_dot = sol.y[3]
th2 = sol.y[4]
th2_dot = sol.y[5]

u_hist = np.array([controller(sol.y[:, i]) for i in range(sol.y.shape[1])])

print(f"max |x|      = {np.max(np.abs(x)):.3f} m")
print(f"max |theta1| = {np.max(np.abs(th1)):.3f} rad")
print(f"max |theta2| = {np.max(np.abs(th2)):.3f} rad")
print(f"max |u|      = {np.max(np.abs(u_hist)):.3f} N")

# -----------------------------
# Plots
# -----------------------------
plt.figure(figsize=(9, 5))
plt.plot(t, th1, label="theta1 (rad)")
plt.plot(t, th2, label="theta2 (rad)")
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.title("Path A Controlled (Fixed): Double Cart-Pole Angles")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(9, 5))
plt.plot(t, x, label="cart position x (m)")
plt.xlabel("Time [s]")
plt.ylabel("Cart Position [m]")
plt.title("Path A Controlled (Fixed): Cart Motion")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(9, 5))
plt.plot(t, u_hist, label="control force u (N)")
plt.xlabel("Time [s]")
plt.ylabel("Force [N]")
plt.title("Path A Controlled (Fixed): Input Force")
plt.grid(True)
plt.legend()
plt.tight_layout()

# -----------------------------
# Animation coordinates
# -----------------------------
pivot_x = x
pivot_y = np.zeros_like(x) + cart_height / 2.0

x1 = pivot_x + l1 * np.sin(th1)
y1 = pivot_y + l1 * np.cos(th1)

x2 = x1 + l2 * np.sin(th2)
y2 = y1 + l2 * np.cos(th2)

# -----------------------------
# Animation figure
# Focus axis around actual cart range
# -----------------------------
fig_anim, ax_anim = plt.subplots(figsize=(10, 5))
ax_anim.set_title("Path A Controlled (Fixed): Double Cart-Pole Animation")
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

# -----------------------------
# Save video
# -----------------------------
if SAVE_VIDEO:
    try:
        ani.save(VIDEO_FILENAME, writer="ffmpeg", fps=30)
        print(f"Video saved successfully: {VIDEO_FILENAME}")
    except Exception as e:
        print("MP4 save failed. Check whether ffmpeg is installed.")
        print("Error:", e)

plt.show()
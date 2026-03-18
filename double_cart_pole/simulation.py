import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================================================
# Double Cart-Pole Simulation
# Clean intermediate version with simple coupling + animation
# =========================================================

# -----------------------------
# Parameters
# -----------------------------
g = 9.81       # gravity [m/s^2]
M = 1.0        # cart mass [kg]
m1 = 0.5       # pendulum 1 mass [kg]
m2 = 0.5       # pendulum 2 mass [kg]
l1 = 1.0       # pendulum 1 length [m]
l2 = 1.0       # pendulum 2 length [m]

cart_damping = 0.15
joint1_damping = 0.05
joint2_damping = 0.05

coupling_gain = 0.8   # simple interaction strength
input_force = 0.0     # no controller yet

cart_width = 0.4
cart_height = 0.2

# -----------------------------
# Dynamics
# State X = [x, x_dot, th1, th1_dot, th2, th2_dot]
# Angles are measured from upright (0 = upright)
# -----------------------------
def dynamics(t, X):
    x, x_dot, th1, th1_dot, th2, th2_dot = X

    x_ddot = (
        input_force
        + coupling_gain * (m1 * np.sin(th1) + m2 * np.sin(th2))
        - cart_damping * x_dot
    ) / (M + m1 + m2)

    th1_ddot = (
        (g / l1) * np.sin(th1)
        - (x_ddot / l1) * np.cos(th1)
        - joint1_damping * th1_dot
    )

    th2_ddot = (
        (g / l2) * np.sin(th2)
        - (x_ddot / l2) * np.cos(th2)
        - joint2_damping * th2_dot
    )

    return [x_dot, x_ddot, th1_dot, th1_ddot, th2_dot, th2_ddot]


# -----------------------------
# Initial conditions
# -----------------------------
x0 = 0.0
x_dot0 = 0.0

th1_0 = np.deg2rad(5.0)
th1_dot0 = 0.0

th2_0 = np.deg2rad(-3.0)
th2_dot0 = 0.0

X0 = [x0, x_dot0, th1_0, th1_dot0, th2_0, th2_dot0]


# -----------------------------
# Simulation settings
# -----------------------------
t_start = 0.0
t_end = 10.0
num_points = 1000

t_eval = np.linspace(t_start, t_end, num_points)

sol = solve_ivp(
    dynamics,
    (t_start, t_end),
    X0,
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-8
)

if not sol.success:
    raise RuntimeError("Simulation failed: " + sol.message)


# -----------------------------
# Extract results
# -----------------------------
t = sol.t
x = sol.y[0]
x_dot = sol.y[1]
th1 = sol.y[2]
th1_dot = sol.y[3]
th2 = sol.y[4]
th2_dot = sol.y[5]


# -----------------------------
# Plot 1: Angles
# -----------------------------
plt.figure(figsize=(9, 5))
plt.plot(t, th1, label="theta1 (rad)")
plt.plot(t, th2, label="theta2 (rad)")
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.title("Double Cart-Pole Angles")
plt.grid(True)
plt.legend()
plt.tight_layout()


# -----------------------------
# Plot 2: Cart position
# -----------------------------
plt.figure(figsize=(9, 5))
plt.plot(t, x, label="cart position x (m)")
plt.xlabel("Time [s]")
plt.ylabel("Cart Position [m]")
plt.title("Cart Motion")
plt.grid(True)
plt.legend()
plt.tight_layout()


# -----------------------------
# Animation data
# -----------------------------
# Pivot point is the top center of the cart
pivot_x = x
pivot_y = np.zeros_like(x) + cart_height / 2.0

# First link end
x1 = pivot_x + l1 * np.sin(th1)
y1 = pivot_y + l1 * np.cos(th1)

# Second link end
x2 = x1 + l2 * np.sin(th2)
y2 = y1 + l2 * np.cos(th2)


# -----------------------------
# Animation figure
# -----------------------------
fig_anim, ax_anim = plt.subplots(figsize=(10, 5))
ax_anim.set_title("Double Cart-Pole Animation")
ax_anim.set_xlabel("X [m]")
ax_anim.set_ylabel("Y [m]")
ax_anim.grid(True)
ax_anim.set_aspect('equal', adjustable='box')

x_margin = 1.0
xmin = np.min(np.concatenate([x, x1, x2])) - x_margin
xmax = np.max(np.concatenate([x, x1, x2])) + x_margin
ymin = -0.5
ymax = cart_height + l1 + l2 + 0.5

ax_anim.set_xlim(xmin, xmax)
ax_anim.set_ylim(ymin, ymax)

# Ground line
ax_anim.plot([xmin, xmax], [0, 0], 'k-', linewidth=1)

# Cart body as a rectangle line
cart_line, = ax_anim.plot([], [], linewidth=3, label="cart")

# Links
link1_line, = ax_anim.plot([], [], 'o-', linewidth=2, markersize=6, label="link 1")
link2_line, = ax_anim.plot([], [], 'o-', linewidth=2, markersize=6, label="link 2")

# Time text
time_text = ax_anim.text(0.02, 0.95, "", transform=ax_anim.transAxes)


def init_animation():
    cart_line.set_data([], [])
    link1_line.set_data([], [])
    link2_line.set_data([], [])
    time_text.set_text("")
    return cart_line, link1_line, link2_line, time_text


def update_animation(i):
    # Cart rectangle corners
    cart_left = x[i] - cart_width / 2.0
    cart_right = x[i] + cart_width / 2.0
    cart_bottom = 0.0
    cart_top = cart_height

    cart_xs = [cart_left, cart_right, cart_right, cart_left, cart_left]
    cart_ys = [cart_bottom, cart_bottom, cart_top, cart_top, cart_bottom]
    cart_line.set_data(cart_xs, cart_ys)

    # Link 1: from cart pivot to first mass
    link1_line.set_data([pivot_x[i], x1[i]], [pivot_y[i], y1[i]])

    # Link 2: from first mass to second mass
    link2_line.set_data([x1[i], x2[i]], [y1[i], y2[i]])

    time_text.set_text(f"t = {t[i]:.2f} s")
    return cart_line, link1_line, link2_line, time_text


frame_step = 5  # skip frames for smoother animation speed
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
# Save animation as video file
# -----------------------------
video_filename = "double_cart_pole_animation.mp4"

try:
    ani.save(video_filename, writer="ffmpeg", fps=30)
    print(f"Video saved successfully: {video_filename}")
except Exception as e:
    print("MP4 save failed. Check whether ffmpeg is installed and available.")
    print("Error:", e)

plt.show()
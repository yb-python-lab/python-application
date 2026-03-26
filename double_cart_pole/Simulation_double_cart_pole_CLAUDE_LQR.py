"""
Double Cart Pole — LQR Controller
pip install numpy scipy matplotlib pillow
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

M  = 1.0;  m1 = 0.3;  m2 = 0.2
L1 = 0.6;  L2 = 0.4
g  = 9.81
l1 = L1 / 2;  l2 = L2 / 2
I1 = (1/12) * m1 * L1**2
I2 = (1/12) * m2 * L2**2

def get_AB():
    M11 = M + m1 + m2
    M12 = m1*l1 + m2*L1
    M13 = m2*l2
    M22 = m1*l1**2 + m2*L1**2 + I1
    M23 = m2*L1*l2
    M33 = m2*l2**2 + I2
    Mm  = np.array([[M11,M12,M13],[M12,M22,M23],[M13,M23,M33]])
    Kg  = np.array([[0,0,0],[0,-(m1*l1+m2*L1)*g,0],[0,0,-m2*l2*g]])
    Bi  = np.array([1.0, 0.0, 0.0])
    Mi  = np.linalg.inv(Mm)
    MK  = Mi @ (-Kg)
    MB  = Mi @ Bi
    A   = np.zeros((6,6))
    A[0,1]=A[2,3]=A[4,5]=1.0
    A[1,0]=MK[0,0]; A[1,2]=MK[0,1]; A[1,4]=MK[0,2]
    A[3,0]=MK[1,0]; A[3,2]=MK[1,1]; A[3,4]=MK[1,2]
    A[5,0]=MK[2,0]; A[5,2]=MK[2,1]; A[5,4]=MK[2,2]
    B   = np.zeros((6,1))
    B[1]=MB[0]; B[3]=MB[1]; B[5]=MB[2]
    return A, B

def lqr(A, B, Q, R):
    P = linalg.solve_continuous_are(A, B, Q, R)
    return np.linalg.inv(R) @ B.T @ P

def eom(s, u):
    _, xd, t1, t1d, t2, t2d = s
    s1, c1 = np.sin(t1), np.cos(t1)
    s2, c2 = np.sin(t2), np.cos(t2)
    s12 = np.sin(t1 - t2)
    c12 = np.cos(t1 - t2)
    Mm = np.array([
        [M+m1+m2,          (m1*l1+m2*L1)*c1,     m2*l2*c2     ],
        [(m1*l1+m2*L1)*c1, m1*l1**2+m2*L1**2+I1, m2*L1*l2*c12 ],
        [m2*l2*c2,         m2*L1*l2*c12,          m2*l2**2+I2  ]])
    rhs = np.array([
        u + (m1*l1+m2*L1)*s1*t1d**2 + m2*l2*s2*t2d**2,
        (m1*l1+m2*L1)*g*s1 - m2*L1*l2*s12*t2d**2,
         m2*l2*g*s2        + m2*L1*l2*s12*t1d**2])
    xdd, t1dd, t2dd = np.linalg.solve(Mm, rhs)
    return np.array([xd, xdd, t1d, t1dd, t2d, t2dd])

def simulate(K, s0, dt=0.005, T=10.0, umax=50.0):
    t = np.arange(0, T, dt)
    S = np.zeros((len(t), 6))
    S[0] = s0
    U = np.zeros(len(t))
    for i in range(1, len(t)):
        u = float((-K @ S[i-1]).flat[0])
        u = float(np.clip(u, -umax, umax))
        U[i-1] = u
        k1 = eom(S[i-1],          u)
        k2 = eom(S[i-1]+dt/2*k1,  u)
        k3 = eom(S[i-1]+dt/2*k2,  u)
        k4 = eom(S[i-1]+dt*k3,    u)
        S[i] = S[i-1] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return t, S, U

def animate(t, S, U):
    fig = plt.figure(figsize=(13, 7), facecolor='#1a1a2e')
    ax  = fig.add_axes([0.05, 0.10, 0.52, 0.82], facecolor='#16213e')
    a1  = fig.add_axes([0.63, 0.55, 0.34, 0.36], facecolor='#16213e')
    a2  = fig.add_axes([0.63, 0.08, 0.34, 0.36], facecolor='#16213e')

    for a in [ax, a1, a2]:
        a.tick_params(colors='#cccccc', labelsize=8)
        for sp in a.spines.values():
            sp.set_edgecolor('#445566')

    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-0.3, 1.4)
    ax.set_aspect('equal')
    ax.set_title('Double Cart Pole  --  LQR Controller', color='#a0c4ff', fontsize=11)
    ax.axhline(0, color='#556677', lw=1.5)
    ax.set_xlabel('Position (m)', color='#a0c4ff', fontsize=9)

    cw, ch = 0.30, 0.12
    cart = patches.Rectangle(
        (-cw/2, -ch/2), cw, ch,
        linewidth=1.5, edgecolor='#a0c4ff', facecolor='#2d2d6e')
    ax.add_patch(cart)

    w1 = plt.Circle((-cw/4, -ch/2 - 0.03), 0.03, color='#a0c4ff')
    w2 = plt.Circle(( cw/4, -ch/2 - 0.03), 0.03, color='#a0c4ff')
    ax.add_patch(w1)
    ax.add_patch(w2)

    p1_line, = ax.plot([], [], 'o-', color='#ff6b6b', lw=5, ms=8,
                       solid_capstyle='round', label=f'Pole 1  L={L1}m')
    p2_line, = ax.plot([], [], 'o-', color='#ffd93d', lw=4, ms=8,
                       solid_capstyle='round', label=f'Pole 2  L={L2}m')
    time_txt  = ax.text(0.02, 0.96, '', transform=ax.transAxes,
                        color='#a0c4ff', fontsize=10)
    force_txt = ax.text(0.02, 0.90, '', transform=ax.transAxes,
                        color='#ffd93d', fontsize=10)
    ax.legend(loc='upper right', facecolor='#1a1a2e',
              edgecolor='#445566', labelcolor='white', fontsize=8)

    a1.set_title('Pole Angles', color='#a0c4ff', fontsize=9)
    a1.set_ylabel('angle (rad)', color='#cccccc', fontsize=8)
    a1.axhline(0, color='#445566', ls='--', lw=0.8)
    l1_, = a1.plot([], [], color='#ff6b6b', lw=1.5, label='th1')
    l2_, = a1.plot([], [], color='#ffd93d', lw=1.5, label='th2')
    a1.legend(facecolor='#1a1a2e', edgecolor='#445566',
              labelcolor='white', fontsize=8)

    a2.set_title('Control Force', color='#a0c4ff', fontsize=9)
    a2.set_ylabel('force (N)', color='#cccccc', fontsize=8)
    a2.axhline(0, color='#445566', ls='--', lw=0.8)
    lc, = a2.plot([], [], color='#4cc9f0', lw=1.5)

    skip = 3

    def update(f):
        i = min(f * skip, len(t) - 1)
        x, _, th1, _, th2, _ = S[i]
        cart.set_x(x - cw/2)
        w1.center = (x - cw/4, -ch/2 - 0.03)
        w2.center = (x + cw/4, -ch/2 - 0.03)
        x1 = x  + L1 * np.sin(th1)
        y1 =      L1 * np.cos(th1)
        x2 = x1 + L2 * np.sin(th2)
        y2 = y1 + L2 * np.cos(th2)
        p1_line.set_data([x,  x1], [0,  y1])
        p2_line.set_data([x1, x2], [y1, y2])
        time_txt.set_text(f't = {t[i]:.2f} s')
        force_txt.set_text(f'F = {U[i]:.1f} N')
        sl = slice(0, i + 1)
        l1_.set_data(t[sl], S[sl, 2])
        l2_.set_data(t[sl], S[sl, 4])
        lc.set_data(t[sl], U[sl])
        for a in [a1, a2]:
            a.relim()
            a.autoscale_view()
        return cart, p1_line, p2_line, time_txt, force_txt

    anim = FuncAnimation(fig, update, frames=len(t) // skip,
                         interval=20, blit=False, repeat=False)
    plt.suptitle('Double Cart Pole  --  LQR Control',
                 color='white', fontsize=14, y=1.0)

    print("Saving video... please wait (may take 1-2 min)")
    anim.save('double_cartpole.gif', writer='pillow', fps=30)
    print("Saved: double_cartpole.gif")

    plt.show()

if __name__ == '__main__':
    A, B = get_AB()
    Q = np.diag([1.0, 0.1, 100.0, 10.0, 100.0, 10.0])
    R = np.array([[0.01]])
    K = lqr(A, B, Q, R)
    print("LQR gain K =", np.round(K[0], 3))
    s0 = np.array([0.1, 0.0, 0.20, 0.1, -0.25, -0.1])
    print(f"Initial:  th1 = {np.degrees(s0[2]):.1f} deg   th2 = {np.degrees(s0[4]):.1f} deg")
    t, S, U = simulate(K, s0)
    print(f"Final:    th1 = {np.degrees(S[-1,2]):.4f} deg   th2 = {np.degrees(S[-1,4]):.4f} deg")
    animate(t, S, U)
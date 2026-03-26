"""
Double CartPole — BVP Swing-Up  v2
====================================
Key fixes over v1:
  1. Cosine series parameterization — exactly Graichen 2007
     x(t) = Σ pₖ · sin²(kπt/T)  →  satisfies all 4 BCs automatically
  2. Scan T_SWING ∈ {1.5, 1.8, 2.0, 2.2, 2.5}s — solutions only exist
     at specific T values (Graichen found T∈{1.8, 2.5}s)
  3. More robust multi-start optimization per T value

Hardware: ACIN Hasomed  MC=5kg  M1=M2=0.327kg  L1=L2=0.323m
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize  import minimize, differential_evolution
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches  as patches
import time as _time

# ════════════════════════════════════════════════════════
#  PLANT — ACIN Hasomed hardware
# ════════════════════════════════════════════════════════
MC, M1, M2 = 5.00, 0.327, 0.327
L1, L2     = 0.323, 0.323
G          = 9.81
XLIM       = 0.70
XSOFT      = 0.58
VEL_LIMIT  = 2.20
ACC_LIMIT  = 20.0
F_MAX      = MC * ACC_LIMIT    # 100 N
X_TOL, XD_TOL = 0.02, 0.05

w_nat = np.sqrt(G / L1)        # 5.51 rad/s
T_nat = 2*np.pi / w_nat        # 1.14 s
N_PARAMS = 8                   # cosine series terms

print("=" * 60)
print(f"  ACIN BVP v2  (Graichen 2007 cosine parameterization)")
print(f"  MC={MC}  M1=M2={M1}  L1=L2={L1} m")
print(f"  ω={w_nat:.3f} rad/s   T_nat={T_nat:.3f} s")
print("=" * 60)


# ════════════════════════════════════════════════════════
#  COSINE SERIES PARAMETERIZATION  (Graichen 2007)
#
#  x(t)  = Σ_{k=1}^{N} pₖ · sin²(kπt/T)
#         = Σ_{k=1}^{N} pₖ · ½[1 - cos(2kπt/T)]
#
#  Automatically satisfies:
#    x(0)=0  x(T)=0  ẋ(0)=0  ẋ(T)=0
#  for any p — no windowing needed.
# ════════════════════════════════════════════════════════
def xref(t, p, T):
    t = np.asarray(t, float)
    x = np.zeros_like(t)
    for k, pk in enumerate(p, start=1):
        x += pk * np.sin(k*np.pi*t/T)**2
    return x

def xdref(t, p, T):
    t = np.asarray(t, float)
    xd = np.zeros_like(t)
    for k, pk in enumerate(p, start=1):
        xd += pk * (k*np.pi/T) * np.sin(2*k*np.pi*t/T)
    return xd

def xddref(t, p, T):
    t = np.asarray(t, float)
    xdd = np.zeros_like(t)
    for k, pk in enumerate(p, start=1):
        xdd += pk * 2*(k*np.pi/T)**2 * np.cos(2*k*np.pi*t/T)
    return xdd


# ════════════════════════════════════════════════════════
#  INTERNAL DYNAMICS — pole ODE given prescribed ẍ(t)
# ════════════════════════════════════════════════════════
def _pole_rhs(t, y, p, T):
    th1, th1d, th2, th2d = y
    s1,c1   = np.sin(th1), np.cos(th1)
    s2,c2   = np.sin(th2), np.cos(th2)
    s12,c12 = np.sin(th1-th2), np.cos(th1-th2)
    xdd     = float(xddref(t, p, T))
    M11 = (M1+M2)*L1**2
    M12 = M2*L1*L2*c12
    M22 = M2*L2**2
    b1  = ((M1+M2)*G*L1*s1 + M2*L1*L2*th2d**2*s12
           - (M1+M2)*L1*c1*xdd)
    b2  = (-M2*L1*L2*th1d**2*s12 + M2*G*L2*s2
           - M2*L2*c2*xdd)
    det   = M11*M22 - M12**2
    th1dd = (M22*b1 - M12*b2) / det
    th2dd = (M11*b2 - M12*b1) / det
    return [th1d, th1dd, th2d, th2dd]

def _wrap(a):
    return (np.asarray(a)+np.pi) % (2*np.pi) - np.pi

def _shoot(p, T):
    try:
        return solve_ivp(
            lambda t,y: _pole_rhs(t,y,p,T),
            [0, T], [np.pi, 0., np.pi, 0.],
            method='DOP853', rtol=1e-7, atol=1e-9,
            max_step=T/500, dense_output=False)
    except Exception:
        return None


# ════════════════════════════════════════════════════════
#  COST FUNCTION
#  Terminal error + constraint penalties
# ════════════════════════════════════════════════════════
def _cost(p, T, fast=False):
    rtol = 1e-6 if fast else 1e-8
    sol  = _shoot(p, T) if not fast else \
           solve_ivp(lambda t,y: _pole_rhs(t,y,p,T),
                     [0,T],[np.pi,0,np.pi,0],
                     method='RK45',rtol=1e-5,atol=1e-7,
                     max_step=T/200)
    if sol is None or not sol.success: return 1e6
    yT = sol.y[:,-1]
    e1 = _wrap(yT[0]); e2 = _wrap(yT[2])

    # Endpoint: poles at upright (0) with zero velocity
    c_end = (10*e1**2 + 5*yT[1]**2 +
             10*e2**2 + 5*yT[3]**2)

    # Constraint penalties
    tv   = np.linspace(1e-4, T*(1-1e-4), 400)
    xv   = xref(tv,p,T);  xdv = xdref(tv,p,T);  xddv = xddref(tv,p,T)
    c_x  = np.mean(np.maximum(0, np.abs(xv)  -XLIM*0.90)**2)*100
    c_xd = np.mean(np.maximum(0, np.abs(xdv) -VEL_LIMIT*0.90)**2)*40
    c_xdd= np.mean(np.maximum(0, np.abs(xddv)-ACC_LIMIT*0.90)**2)*10
    return float(c_end + c_x + c_xd + c_xdd)


# ════════════════════════════════════════════════════════
#  SCAN MULTIPLE T_SWING VALUES
#  BVP solutions only exist at specific T — must search
# ════════════════════════════════════════════════════════
T_CANDIDATES = [1.5, 1.8, 2.0, 2.2, 2.5, 3.0]

best_overall_cost = 1e7
best_overall_p    = None
best_overall_T    = None
all_results       = {}

print(f"\n[Step 1] Scanning T_SWING ∈ {T_CANDIDATES} s ...")
t0 = _time.time()

for T_try in T_CANDIDATES:
    print(f"\n  T={T_try:.1f}s:", end=" ", flush=True)
    best_p_T = None
    best_c_T = 1e7

    # --- Multi-start Nelder-Mead with physics guesses ---
    # Each guess is scaled by T to be physically reasonable
    scale = min(XLIM*0.7, VEL_LIMIT/(w_nat*2))  # rough amplitude
    guesses = [
        [ scale,  0,  0,  0,  0,  0,  0,  0],   # single hump right
        [-scale,  0,  0,  0,  0,  0,  0,  0],   # single hump left
        [ scale, -scale,  0,  0,  0,  0,  0,  0],   # 2 humps
        [-scale,  scale,  0,  0,  0,  0,  0,  0],
        [ scale,  0, -scale/2, 0,  0,  0,  0,  0],
        [ 0, scale,  0, -scale,  0,  0,  0,  0],
        [ scale, -scale/2,  scale/4, 0, 0, 0, 0, 0],
        [-scale,  scale/2, -scale/4, 0, 0, 0, 0, 0],
    ]

    for g in guesses:
        r = minimize(lambda p: _cost(p, T_try, fast=True), g,
                     method='Nelder-Mead',
                     options={'maxiter':2000,'xatol':1e-4,'fatol':1e-6})
        if r.fun < best_c_T:
            best_c_T = r.fun; best_p_T = r.x.copy()
        print(f"{r.fun:.3f}", end=" ", flush=True)

    # --- DE global search for this T ---
    bounds = [(-XLIM*3, XLIM*3)] * N_PARAMS
    de = differential_evolution(
        lambda p: _cost(p, T_try, fast=True),
        bounds, seed=int(T_try*10), maxiter=150,
        popsize=12, tol=1e-6, mutation=(0.5,1.5),
        recombination=0.8, polish=False, workers=1)
    if de.fun < best_c_T:
        best_c_T = de.fun; best_p_T = de.x.copy()

    # --- Local polish at full accuracy ---
    for method in ['Nelder-Mead', 'Powell']:
        r = minimize(lambda p: _cost(p, T_try), best_p_T,
                     method=method,
                     options={'maxiter':50000,'xatol':1e-10,'fatol':1e-12})
        if r.fun < best_c_T:
            best_c_T = r.fun; best_p_T = r.x.copy()

    all_results[T_try] = (best_c_T, best_p_T.copy())
    print(f"  → best={best_c_T:.5f}")

    if best_c_T < best_overall_cost:
        best_overall_cost = best_c_T
        best_overall_p    = best_p_T.copy()
        best_overall_T    = T_try

print(f"\n[Step 1 done] {_time.time()-t0:.0f}s")
print(f"  Best T={best_overall_T}s  cost={best_overall_cost:.5f}")

# Summary table
print("\n  T_SWING  cost")
for T_,( c_,_) in sorted(all_results.items()):
    marker = " ←best" if T_==best_overall_T else ""
    print(f"  {T_:.1f}s    {c_:.5f}{marker}")


# ════════════════════════════════════════════════════════
#  USE BEST SOLUTION
# ════════════════════════════════════════════════════════
T_SWING = best_overall_T
p_opt   = best_overall_p

sol_bvp = _shoot(p_opt, T_SWING)
yT_bvp  = sol_bvp.y[:,-1] if (sol_bvp and sol_bvp.success) \
          else [np.pi,0,np.pi,0]

tv    = np.linspace(0, T_SWING, 800)
x_bvp = xref(tv, p_opt, T_SWING)
xd_bvp= xdref(tv, p_opt, T_SWING)
xdd_bvp=xddref(tv, p_opt, T_SWING)

print(f"\n  BVP result at t={T_SWING:.1f}s:")
print(f"    θ₁={np.degrees(_wrap(yT_bvp[0])):+.2f}°  "
      f"θ̇₁={yT_bvp[1]:+.4f} rad/s")
print(f"    θ₂={np.degrees(_wrap(yT_bvp[2])):+.2f}°  "
      f"θ̇₂={yT_bvp[3]:+.4f} rad/s")
print(f"  max|x|  ={np.max(np.abs(x_bvp)):.4f}m  "
      f"({'✓' if np.max(np.abs(x_bvp))<=XLIM else '✗'})")
print(f"  max|ẋ|  ={np.max(np.abs(xd_bvp)):.4f}m/s  "
      f"({'✓' if np.max(np.abs(xd_bvp))<=VEL_LIMIT else '✗'})")
print(f"  max|ẍ|  ={np.max(np.abs(xdd_bvp)):.4f}m/s²  "
      f"({'✓' if np.max(np.abs(xdd_bvp))<=ACC_LIMIT else '✗'})")


# ════════════════════════════════════════════════════════
#  FULL SIMULATION
# ════════════════════════════════════════════════════════
K_TRACK_P = 200.0; K_TRACK_D = 40.0
K_T1,K_T1D = 70.0, 32.0
K_T2,K_T2D = 70.0, 32.0
F_LQR_LIM  = 90.0
K_X0,K_XD0 = -12.0,-18.0
K_X1,K_XD1 = -18.0,-25.0
K_X2,K_XD2 = -25.0,-35.0
LQR_ANG1=LQR_ANG2=30.0; LQR_ANGVEL=12.0
CATCH_DEG=40.0; CATCH_BLEND=8.0; CATCH_CAP=90.0
K_TIP_POS=35.0; K_TIP_VEL=22.0
S1_ANG=12.0; S1_ANGVEL=2.0; S1_OFFSET=0.05; S1_TIME=0.10
S2_ANG=6.0;  S2_ANGVEL=0.8; S2_OFFSET=0.02; S2_FORCE=5.0; S2_TIME=0.20
DT=0.0008; NSTEP=10; T_MAX=10.0


def eom(st,F):
    x,xd,t1,t1d,t2,t2d=st
    s1,c1=np.sin(t1),np.cos(t1)
    s2,c2=np.sin(t2),np.cos(t2)
    s12,c12=np.sin(t1-t2),np.cos(t1-t2)
    M=np.array([[MC+M1+M2,(M1+M2)*L1*c1,M2*L2*c2],
                [(M1+M2)*L1*c1,(M1+M2)*L1**2,M2*L1*L2*c12],
                [M2*L2*c2,M2*L1*L2*c12,M2*L2**2]])
    rhs=np.array([F-(M1+M2)*L1*t1d**2*s1-M2*L2*t2d**2*s2,
                  (M1+M2)*G*L1*s1+M2*L1*L2*t2d**2*s12,
                  -M2*L1*L2*t1d**2*s12+M2*G*L2*s2])
    acc=np.linalg.solve(M,rhs)
    return np.array([xd,acc[0],t1d,acc[1],t2d,acc[2]])

def rk4(st,F):
    k1=eom(st,F); k2=eom(st+DT/2*k1,F)
    k3=eom(st+DT/2*k2,F); k4=eom(st+DT*k3,F)
    return st+DT/6*(k1+2*k2+2*k3+k4)

def clamp(v,lim): return float(np.clip(v,-lim,lim))
def deadband(v,tol): return 0.0 if abs(v)<=tol else v-np.sign(v)*tol

def soft_wall(x,xd):
    if   x> XSOFT: p_=(x-XSOFT)/(XLIM-XSOFT); return -800*p_**2-80*xd
    elif x<-XSOFT: p_=(-x-XSOFT)/(XLIM-XSOFT);return +800*p_**2-80*xd
    return 0.0

def tip2_offset(st): return 2*L1*np.sin(st[2])+2*L2*np.sin(st[4])
def tip2_vel(st):
    x,xd,t1,t1d,t2,t2d=st
    return xd+2*L1*np.cos(t1)*t1d+2*L2*np.cos(t2)*t2d
def tip2_pos(st):
    x,xd,t1,t1d,t2,t2d=st
    return x+2*L1*np.sin(t1)+2*L2*np.sin(t2)
def joint_pos(st): return st[0]+2*L1*np.sin(st[2])

def catch_w(st):
    err=abs(_wrap(st[4]))*180/np.pi
    lo,hi=CATCH_DEG-CATCH_BLEND,CATCH_DEG
    if err<=lo: return 1.0
    if err>=hi: return 0.0
    t_=(hi-err)/CATCH_BLEND; return 3*t_**2-2*t_**3

def catch_force(st):
    cw=catch_w(st)
    if cw<0.01: return 0.0,cw
    off=tip2_offset(st); vel=tip2_vel(st)-st[1]
    return clamp((K_TIP_POS*off+K_TIP_VEL*vel)*cw,CATCH_CAP),cw

def lqr_met(st):
    x,xd,t1,t1d,t2,t2d=st
    return (abs(_wrap(t1))*180/np.pi<LQR_ANG1 and
            abs(_wrap(t2))*180/np.pi<LQR_ANG2 and
            abs(t1d)<LQR_ANGVEL and abs(t2d)<LQR_ANGVEL)

def s2_check(st,Fc):
    x,xd,t1,t1d,t2,t2d=st
    return (abs(_wrap(t1))*180/np.pi<S2_ANG and
            abs(_wrap(t2))*180/np.pi<S2_ANG and
            abs(t1d)<S2_ANGVEL and abs(t2d)<S2_ANGVEL and
            abs(tip2_offset(st))<S2_OFFSET and abs(Fc)<S2_FORCE)

def s2_detail(st,Fc):
    x,xd,t1,t1d,t2,t2d=st
    return dict(ang1=abs(_wrap(t1))*180/np.pi<S2_ANG,
                ang2=abs(_wrap(t2))*180/np.pi<S2_ANG,
                avel1=abs(t1d)<S2_ANGVEL,avel2=abs(t2d)<S2_ANGVEL,
                offset=abs(tip2_offset(st))<S2_OFFSET,
                force=abs(Fc)<S2_FORCE)


_lqr_on=False; _lqr_t=None; _stage=0
_s1_tm=0.0; _s2_tm=0.0; _s1_t=None; _s2_t=None

def control(st,t):
    global _lqr_on,_lqr_t,_stage,_s1_tm,_s2_tm,_s1_t,_s2_t
    x,xd,t1,t1d,t2,t2d=st

    # Switch to LQR at T_SWING or earlier if poles near upright
    if not _lqr_on and (t>=T_SWING or lqr_met(st)):
        _lqr_on=True; _lqr_t=t

    Fc,cw=catch_force(st)

    if _lqr_on:
        # Stage progression
        s1_now=(abs(_wrap(t1))*180/np.pi<S1_ANG and
                abs(_wrap(t2))*180/np.pi<S1_ANG and
                abs(t1d)<S1_ANGVEL and abs(t2d)<S1_ANGVEL and
                abs(tip2_offset(st))<S1_OFFSET)
        if s1_now: _s1_tm+=DT
        else:      _s1_tm=0.0
        if _stage>=1 and s2_check(st,Fc): _s2_tm+=DT
        else:                             _s2_tm=0.0
        if _stage==0 and _s1_tm>=S1_TIME: _stage=1; _s1_t=t
        if _stage==1 and _s2_tm>=S2_TIME: _stage=2; _s2_t=t

        pg=1.0-cw
        Fp=-(K_T1*_wrap(t1)+K_T1D*t1d
            +K_T2*_wrap(t2)+K_T2D*t2d)*pg
        if   _stage==0: kx,kxd=K_X0,K_XD0
        elif _stage==1: kx,kxd=K_X1,K_XD1
        else:           kx,kxd=K_X2,K_XD2
        Fx=-(kx*deadband(x,X_TOL)+kxd*deadband(xd,XD_TOL))
        F=clamp(Fp+Fx+Fc+soft_wall(x,xd),F_LQR_LIM)
        xr=0.0; phase='lqr'
    else:
        xr  = float(xref(t,p_opt,T_SWING))
        xdr = float(xdref(t,p_opt,T_SWING))
        xddr= float(xddref(t,p_opt,T_SWING))
        Fff = MC*xddr
        Fpd = K_TRACK_P*(xr-x)+K_TRACK_D*(xdr-xd)
        F   = clamp(Fff+Fpd+soft_wall(x,xd),F_MAX)
        phase='bvp'

    return (F,_lqr_on,cw,Fc,_stage,
            _s1_tm,_s2_tm,s2_detail(st,Fc),
            phase,xr)

def reset_ctrl():
    global _lqr_on,_lqr_t,_stage,_s1_tm,_s2_tm,_s1_t,_s2_t
    _lqr_on=False;_lqr_t=None;_stage=0
    _s1_tm=0.0;_s2_tm=0.0;_s1_t=None;_s2_t=None


print("\n[Step 2] Simulating ...")
reset_ctrl()
s0=np.array([0.005,0.0,np.pi,0.0,np.pi,0.0])
N=int(T_MAX/(DT*NSTEP))
states=np.zeros((N,6)); forces=np.zeros(N)
catches=np.zeros(N); catch_ws=np.zeros(N)
lqr_ons=np.zeros(N,bool); stg_arr=np.zeros(N,int)
s2tm_arr=np.zeros(N); tip_arr=np.zeros(N)
x_refs=np.zeros(N); phases=[]; cs2_arr=[]; times=np.zeros(N)

st=s0.copy(); t=0.0
for i in range(N):
    for _ in range(NSTEP):
        F,lon,cw,Fc,stg,s1t,s2t,cs2,ph,xr=control(st,t)
        st=rk4(st,F)
        if st[0]> XLIM: st[0]=XLIM;  st[1]=min(0,st[1])
        if st[0]<-XLIM: st[0]=-XLIM; st[1]=max(0,st[1])
        if abs(st[1])>VEL_LIMIT: st[1]=np.sign(st[1])*VEL_LIMIT
        t+=DT
    states[i]=st; forces[i]=F; catches[i]=Fc
    catch_ws[i]=cw; lqr_ons[i]=lon; stg_arr[i]=stg
    s2tm_arr[i]=s2t; tip_arr[i]=tip2_pos(st)
    x_refs[i]=xr; phases.append(ph); cs2_arr.append(cs2); times[i]=t

lqr_time=_lqr_t; s2_time=_s2_t
print(f"Done. LQR={lqr_time:.4f}s  "
      f"STABLE={'✓ '+str(round(s2_time,3))+'s' if s2_time else '✗'}")


# ════════════════════════════════════════════════════════
#  FIGURE
# ════════════════════════════════════════════════════════
DISP=XLIM+0.12
fig=plt.figure(figsize=(17,11),facecolor='#f4f6f8')
ss=f'✓ STABLE {s2_time:.3f}s' if s2_time else '✗ not stable'
fig.suptitle(
    f'Double CartPole — BVP Swing-Up v2  (Graichen 2007 cosine series)\n'
    f'MC={MC}kg  M1=M2={M1}kg  L1=L2={L1}m  '
    f'Best T={best_overall_T}s  cost={best_overall_cost:.4f}  |  {ss}',
    fontsize=11,fontweight='bold',
    color='darkgreen' if s2_time else 'black')

def vl(ax_,t_,col,ls='--',lw=1.5,lbl=None):
    if t_ is not None: ax_.axvline(t_,color=col,lw=lw,ls=ls,alpha=0.85,label=lbl)

# Cost-vs-T bar chart (top right inset)
ax_bar=fig.add_axes([0.34,0.87,0.20,0.09])
Ts=[T_ for T_ in T_CANDIDATES if T_ in all_results]
cs=[min(all_results[T_][0],20) for T_ in Ts]
cols=['#2e7d32' if T_==best_overall_T else '#1976d2' for T_ in Ts]
ax_bar.bar([f'{T_}' for T_ in Ts],cs,color=cols,alpha=0.8)
ax_bar.set_title('BVP cost per T (lower=better)',fontsize=8)
ax_bar.set_ylabel('cost',fontsize=7); ax_bar.tick_params(labelsize=7)
ax_bar.set_facecolor('white'); ax_bar.grid(True,alpha=0.2,axis='y')

# BVP trajectory preview
ax_bvp=fig.add_axes([0.57,0.87,0.41,0.09])
ax_bvp.set_xlim(0,T_SWING); ax_bvp.set_ylim(-XLIM-0.05,XLIM+0.05)
ax_bvp.axhline(0,color='gray',lw=0.8,ls='--',alpha=0.5)
ax_bvp.axhline( XLIM,color='#e53935',lw=1.5,alpha=0.7)
ax_bvp.axhline(-XLIM,color='#e53935',lw=1.5,alpha=0.7)
ax_bvp.plot(tv,x_bvp,  color='#2e7d32',lw=2,label=f'x*(t)  T={T_SWING}s')
ax_bvp.plot(tv,xd_bvp, color='#1565c0',lw=1.2,ls='--',label='ẋ*(t)')
ax_bvp.axvline(T_SWING,color='red',lw=1.5,ls=':')
ax_bvp.set_title(f'BVP reference: T={T_SWING}s  '
    f'max|x|={np.max(np.abs(x_bvp)):.3f}m  '
    f'max|ẋ|={np.max(np.abs(xd_bvp)):.3f}m/s  '
    f'cost={best_overall_cost:.4f}',fontsize=9)
ax_bvp.legend(fontsize=7,loc='upper right',ncol=2)
ax_bvp.set_facecolor('white'); ax_bvp.grid(True,alpha=0.2)
tc_bvp=ax_bvp.axvline(0,color='red',lw=1,alpha=0.6)

# Pendulum canvas
ax=fig.add_axes([0.01,0.10,0.30,0.74])
ax.set_xlim(-DISP,DISP); ax.set_ylim(-0.8,4*L1+0.25)
ax.set_aspect('equal'); ax.set_facecolor('white'); ax.grid(True,alpha=0.2)
ax.axhline(0,color='#90a4ae',lw=1.5,ls='--')
for xl in [-XLIM,XLIM]:
    ax.axvline(xl,color='#e53935',lw=2.5,alpha=0.8)
    ax.text(xl,-0.7,f'{xl:+.2f}m',ha='center',fontsize=7,
            color='#e53935',fontweight='bold')
ax.axvspan(-X_TOL,X_TOL,color='#c8e6c9',alpha=0.4,zorder=0)
ax.axhline(4*L1,color='#1565c0',lw=0.8,ls=':',alpha=0.5)
ax.axhline(2*L1,color='#7b1fa2',lw=0.8,ls=':',alpha=0.3)
ax.text(DISP-0.01,4*L1+0.02,f'upright {4*L1:.3f}m',
        ha='right',fontsize=7,color='#1565c0')
ax.plot([],[],'-',color='#185FA5',lw=5,label='Pole 1')
ax.plot([],[],'-',color='#BA7517',lw=4,label='Pole 2')
ax.legend(loc='upper right',fontsize=9)
ttl_=f'BVP(T={T_SWING}s cost={best_overall_cost:.3f})'
if lqr_time: ttl_+=f'  LQR={lqr_time:.3f}s'
if s2_time:  ttl_+=f'  ✓={s2_time:.3f}s'
ax.set_title(ttl_,fontsize=8,color='green' if s2_time else 'black')
ax.set_xlabel('Position (m)'); ax.set_ylabel('Height (m)')

# Pole angles
ax_a=fig.add_axes([0.34,0.73,0.64,0.12])
ax_a.set_xlim(0,T_MAX); ax_a.set_ylim(-np.pi-0.2,np.pi+0.2)
ax_a.axhline(0,color='green',lw=1.2,ls='--',alpha=0.8)
ax_a.axhspan(-S2_ANG*np.pi/180,S2_ANG*np.pi/180,
             color='#c8e6c9',alpha=0.4,label=f'S2±{S2_ANG}°')
ax_a.axhspan(-LQR_ANG1*np.pi/180,LQR_ANG1*np.pi/180,
             color='#fff9c4',alpha=0.4,label=f'LQR±{LQR_ANG1}°')
ax_a.axvline(T_SWING,color='red',lw=2,ls=':',alpha=0.8,label=f'T={T_SWING}s')
vl(ax_a,lqr_time,'#e65100',lw=2,lbl=f'LQR {lqr_time:.3f}s' if lqr_time else None)
vl(ax_a,s2_time,'blue',lbl=f'✓{s2_time:.3f}s' if s2_time else None)
ax_a.set_yticks([-np.pi,0,np.pi],['-π','0','π'])
ax_a.set_ylabel('Angle (rad)')
ax_a.set_title('Pole angles  (π→0 = upright)',fontsize=10)
ax_a.legend(fontsize=7,loc='upper right',ncol=6)
ax_a.set_facecolor('white'); ax_a.grid(True,alpha=0.2)

# Angular velocities
ax_av=fig.add_axes([0.34,0.62,0.64,0.09])
ax_av.set_xlim(0,T_MAX); ax_av.set_ylim(-25,25)
ax_av.axhline(0,color='gray',lw=0.8,ls='--',alpha=0.5)
ax_av.axhspan(-LQR_ANGVEL,LQR_ANGVEL,color='#fff9c4',alpha=0.4)
ax_av.axvline(T_SWING,color='red',lw=2,ls=':',alpha=0.8)
vl(ax_av,lqr_time,'#e65100',lw=2); vl(ax_av,s2_time,'blue')
ax_av.set_ylabel('θ̇ (rad/s)')
ax_av.set_title(f'Angular velocities  [ω={w_nat:.2f} rad/s  LQR gate=±{LQR_ANGVEL} rad/s]',fontsize=9)
ax_av.set_facecolor('white'); ax_av.grid(True,alpha=0.2)

# Stage-2 criteria
ax_c=fig.add_axes([0.34,0.44,0.64,0.16])
ax_c.set_xlim(0,T_MAX); ax_c.set_ylim(-0.2,6.2)
ax_c.set_yticks(range(6),
    [f'Fc<{S2_FORCE}N',f'off<{S2_OFFSET}m',
     f'|θ̇₂|<{S2_ANGVEL}',f'|θ̇₁|<{S2_ANGVEL}',
     f'|θ₂|<{S2_ANG}°',f'|θ₁|<{S2_ANG}°'],fontsize=7)
ax_c.set_title('Stage-2 criteria',fontsize=9)
ax_c.set_facecolor('#f8f8f8')
ax_c.axvline(T_SWING,color='red',lw=2,ls=':',alpha=0.8)
vl(ax_c,lqr_time,'#e65100'); vl(ax_c,s2_time,'blue')
for row,key in enumerate(['ang1','ang2','avel1','avel2','offset','force']):
    for i in range(len(times)-1):
        met=cs2_arr[i].get(key,False)
        ax_c.barh(row,times[i+1]-times[i],left=times[i],height=0.75,
                  color='#43a047' if met else '#e53935',alpha=0.7)

# Cart position
ax_x=fig.add_axes([0.34,0.31,0.64,0.11])
ax_x.set_xlim(0,T_MAX); ax_x.set_ylim(-XLIM-0.05,XLIM+0.05)
ax_x.axhline(0,color='green',lw=1,ls='--',alpha=0.6)
ax_x.axhline( XLIM,color='#e53935',lw=2,alpha=0.8,label='±0.70m wall')
ax_x.axhline(-XLIM,color='#e53935',lw=2,alpha=0.8)
ax_x.axvline(T_SWING,color='red',lw=2,ls=':',alpha=0.8)
vl(ax_x,lqr_time,'#e65100',lw=2); vl(ax_x,s2_time,'blue')
ax_x.set_ylabel('x (m)')
ax_x.set_title('Cart (blue)  x*(t) BVP ref (green)  tip2 (orange)',fontsize=9)
ax_x.legend(fontsize=7,loc='upper right')
ax_x.set_facecolor('white'); ax_x.grid(True,alpha=0.2)

# Phase bar
ax_ph=fig.add_axes([0.34,0.22,0.64,0.07])
ax_ph.set_xlim(0,T_MAX); ax_ph.set_ylim(0,1); ax_ph.set_yticks([])
for i in range(len(times)-1):
    if   stg_arr[i]==2: c='#0d2b6b'
    elif stg_arr[i]==1: c='#1976d2'
    elif lqr_ons[i]:    c='#90caf9'
    else:               c='#e8a020'
    ax_ph.axvspan(times[i],times[i+1],facecolor=c,alpha=0.85)
ax_ph.axvline(T_SWING,color='white',lw=3,alpha=0.9)
if lqr_time: ax_ph.axvline(lqr_time,color='#e65100',lw=2.5,alpha=0.9)
if s2_time:  ax_ph.axvline(s2_time,color='blue',lw=2.5,alpha=0.9)
ax_ph.set_title('gold=BVP tracking  lt.blue=LQR-L0  blue=L1  dk.blue=STABLE',fontsize=9)
ax_ph.set_xlabel('Time (s)')

# Stage timer
ax_t=fig.add_axes([0.34,0.12,0.64,0.08])
ax_t.set_xlim(0,T_MAX); ax_t.set_ylim(-0.01,max(S1_TIME,S2_TIME)*1.3)
ax_t.axhline(S1_TIME,color='orange',lw=1.2,ls='--',label=f'S1 {S1_TIME}s')
ax_t.axhline(S2_TIME,color='blue',lw=1.2,ls='--',label=f'S2 {S2_TIME}s')
ax_t.axvline(T_SWING,color='red',lw=2,ls=':',alpha=0.8)
vl(ax_t,lqr_time,'#e65100'); vl(ax_t,s2_time,'blue')
ax_t.set_xlabel('Time (s)'); ax_t.set_ylabel('Timer(s)')
ax_t.set_title('Stage timers',fontsize=9)
ax_t.legend(fontsize=7,loc='upper right')
ax_t.set_facecolor('white'); ax_t.grid(True,alpha=0.2)


# ════════════════════════════════════════════════════════
#  ANIMATION
# ════════════════════════════════════════════════════════
CW,CH=0.08,0.04
cr=patches.Rectangle((-CW/2,-CH),CW,CH,lw=2,
    edgecolor='#4682b4',facecolor='#37474f',zorder=4)
ax.add_patch(cr)
wL=plt.Circle((-0.025,-CH-0.015),0.013,color='#546e7a',zorder=5)
wR=plt.Circle(( 0.025,-CH-0.015),0.013,color='#546e7a',zorder=5)
ax.add_patch(wL); ax.add_patch(wR)
jc=plt.Circle((0,0),0.025,fill=True,facecolor='#ce93d8',
    alpha=0.2,edgecolor='#7b1fa2',lw=1.5,zorder=7)
ax.add_patch(jc)
ref_dot,=ax.plot([],[],'g^',ms=9,alpha=0.0,zorder=9)
p1l,=ax.plot([],[],'-',color='#185FA5',lw=6,solid_capstyle='round',zorder=3)
p2l,=ax.plot([],[],'-',color='#BA7517',lw=5,solid_capstyle='round',zorder=3)
j0,=ax.plot([],[],'o',color='#212121',ms=7,zorder=6)
j1,=ax.plot([],[],'o',color='#7b1fa2',ms=7,zorder=8)
j2,=ax.plot([],[],'o',color='#BA7517',ms=6,zorder=6)
tr,=ax.plot([],[],'-',color='#BA7517',lw=1,alpha=0.3,zorder=2)
fa,=ax.plot([],[],'-',lw=2.5,zorder=7)
TXS,TYS=[],[]
info=ax.text(-DISP+0.02,4*L1-0.10,'',fontsize=8,fontfamily='monospace',
    bbox=dict(boxstyle='round,pad=0.3',fc='white',ec='#cfd8dc',alpha=0.93),zorder=9)
ln_t1,=ax_a.plot([],[],color='#185FA5',lw=1.5,label='θ₁')
ln_t2,=ax_a.plot([],[],color='#BA7517',lw=1.5,label='θ₂')
ax_a.legend(fontsize=8,loc='upper right')
ln_td1,=ax_av.plot([],[],color='#185FA5',lw=1.5)
ln_td2,=ax_av.plot([],[],color='#BA7517',lw=1.5)
ln_xc,=ax_x.plot([],[],color='#1a237e',lw=1.5,label='cart')
ln_xr,=ax_x.plot([],[],color='#2e7d32',lw=1.5,ls='--',alpha=0.9,label='x*')
ln_xt,=ax_x.plot([],[],color='#f57c00',lw=1.2,alpha=0.7,label='tip2')
ax_x.legend(fontsize=7,loc='upper right')
ln_s2,=ax_t.plot([],[],color='#1565c0',lw=2)
tc_a =ax_a.axvline(0,color='red',lw=1,alpha=0.5)
tc_av=ax_av.axvline(0,color='red',lw=1,alpha=0.5)
tc_c =ax_c.axvline(0,color='red',lw=1,alpha=0.5)
tc_x =ax_x.axvline(0,color='red',lw=1,alpha=0.5)
tc_ph=ax_ph.axvline(0,color='white',lw=2,alpha=0.8)
tc_t =ax_t.axvline(0,color='red',lw=1,alpha=0.5)

def update(frame):
    st=states[frame]; F=forces[frame]; Fc=catches[frame]
    lon=lqr_ons[frame]; stg=stg_arr[frame]
    ph=phases[frame]; xr=x_refs[frame]; t_=times[frame]
    x,xd,t1,t1d,t2,t2d=st
    p1x=x+2*L1*np.sin(t1); p1y=2*L1*np.cos(t1)
    p2x=p1x+2*L2*np.sin(t2); p2y=p1y+2*L2*np.cos(t2)
    w1=_wrap(t1); w2=_wrap(t2)

    if   stg==2: fc,ec='#0d2b6b','#3949ab'
    elif stg==1: fc,ec='#1565c0','#42a5f5'
    elif lon:    fc,ec='#90caf9','#1976d2'
    else:        fc,ec='#5d4a1e','#f9a825'
    cr.set_facecolor(fc); cr.set_edgecolor(ec)
    cr.set_xy((x-CW/2,-CH)); wL.center=(x-0.025,-CH-0.015)
    wR.center=(x+0.025,-CH-0.015)
    if ph=='bvp': ref_dot.set_data([xr],[-0.07]); ref_dot.set_alpha(0.9)
    else:         ref_dot.set_alpha(0.0)
    jc.center=(p1x,p1y)
    p1l.set_data([x,p1x],[0,p1y]); p2l.set_data([p1x,p2x],[p1y,p2y])
    j0.set_data([x],[0]); j1.set_data([p1x],[p1y]); j2.set_data([p2x],[p2y])
    TXS.append(p2x); TYS.append(p2y)
    if len(TXS)>400: TXS.pop(0); TYS.pop(0)
    tr.set_data(TXS,TYS)
    if abs(F)>1:
        d=np.sign(F); mag=min(0.25,abs(F)/F_MAX*0.32)
        fa.set_data([x,x+d*mag],[-CH*0.2,-CH*0.2])
        fa.set_color('#f9a825' if ph=='bvp' else '#185FA5' if F>0 else '#c62828')
    else: fa.set_data([],[])
    if   stg==2:  ms='✓ STABLE!'
    elif stg==1:  ms='LQR-L1'
    elif lon:     ms='LQR-L0'
    else:         ms=f'BVP {min(100,t_/T_SWING*100):.0f}%'
    info.set_text(f't  ={t_:.3f}s\nx  ={x:+.4f}m\nx* ={xr:+.4f}m\n'
                  f'xd ={xd:+.3f}m/s\nθ₁ ={np.degrees(w1):+.1f}°\n'
                  f'θ₂ ={np.degrees(w2):+.1f}°\n'
                  f'θ̇₁={t1d:+.2f}\nθ̇₂={t2d:+.2f}\n'
                  f'F  ={F:+.1f}N\n{ms}')
    end=frame+1
    ln_t1.set_data(times[:end],[_wrap(states[k,2]) for k in range(end)])
    ln_t2.set_data(times[:end],[_wrap(states[k,4]) for k in range(end)])
    ln_td1.set_data(times[:end],states[:end,3])
    ln_td2.set_data(times[:end],states[:end,5])
    ln_xc.set_data(times[:end],states[:end,0])
    ln_xr.set_data(times[:end],x_refs[:end])
    ln_xt.set_data(times[:end],tip_arr[:end])
    ln_s2.set_data(times[:end],s2tm_arr[:end])
    for tc in [tc_a,tc_av,tc_c,tc_x,tc_ph,tc_t,tc_bvp]:
        tc.set_xdata([t_,t_])
    return (cr,wL,wR,ref_dot,jc,p1l,p2l,j0,j1,j2,tr,fa,info,
            ln_t1,ln_t2,ln_td1,ln_td2,ln_xc,ln_xr,ln_xt,ln_s2,
            tc_a,tc_av,tc_c,tc_x,tc_ph,tc_t,tc_bvp)

ani=animation.FuncAnimation(fig,update,frames=N,interval=20,blit=False,repeat=False)
plt.show()
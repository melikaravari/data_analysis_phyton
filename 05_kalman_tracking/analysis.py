"""
Project 5 - Kalman filter and smoother for tracking a moving object from noisy position measurements

Data (simulated)
  x_sim.csv : the "true" trajectory, 69 time steps. Columns (my reading of the file, see README):
              [vx, x, vy, y, ax, ay] = velocity and position in x and y, plus constant accelerations.
  y_sim.csv : noisy measurements of the position, columns [x_measured, y_measured].

Questions
  1. How much better is a Kalman filter than using the raw noisy measurements?
  2. Can the noise parameters be estimated from the measurements alone (maximum likelihood)?
  3. What do we gain from smoothing (using future data), and what is lost with a wrong model?

Run:  python analysis.py      (needs numpy scipy matplotlib pandas)
Outputs: results.txt and figures/*.png in this folder.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize

HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


# ----------------------------------------------------------------------------
# 1. Load data and check the assumed structure
# ----------------------------------------------------------------------------
xs = np.loadtxt(HERE.parent / "data" / "x_sim.csv")  # (69, 6), no header
ys = np.loadtxt(HERE.parent / "data" / "y_sim.csv")  # (69, 2), no header
T = len(xs)
# Re-order the truth to [x, y, vx, vy, ax, ay]
truth = xs[:, [1, 3, 0, 2, 4, 5]]

# Time step: position(t+1) - position(t) should equal velocity * dt
dt = float(np.median(np.diff(truth[:, 0]) / truth[:-1, 2]))
log(f"Estimated time step dt = {dt:.4f} (from position increments / velocity)")
log(f"Acceleration columns constant? ax std = {truth[:, 4].std():.1e}, ay std = {truth[:, 5].std():.1e}")
noise = ys - truth[:, :2]
log(f"Measurement errors (y_sim - true position): mean {noise.mean(axis=0).round(2)}, std {noise.std(axis=0).round(2)}")
log(f"Shapiro-Wilk normality p-values of the errors: {[round(stats.shapiro(noise[:, i])[1], 3) for i in range(2)]}\n")

# ----------------------------------------------------------------------------
# 2. State-space models
#    Model A (constant acceleration): state per axis [pos, vel, acc]; acceleration follows a random walk.
#    Model B (constant velocity)    : state per axis [pos, vel]; misspecified, because the truth accelerates.
# ----------------------------------------------------------------------------
def model(kind):
    if kind == "CA":
        F1 = np.array([[1, dt, dt**2 / 2], [0, 1, dt], [0, 0, 1]])
        Q1 = np.array([[dt**5 / 20, dt**4 / 8, dt**3 / 6], [dt**4 / 8, dt**3 / 3, dt**2 / 2], [dt**3 / 6, dt**2 / 2, dt]])
        H1 = np.array([[1.0, 0, 0]])
    else:
        F1 = np.array([[1, dt], [0, 1]])
        Q1 = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        H1 = np.array([[1.0, 0]])
    k = F1.shape[0]
    # Stack the two independent axes: state = [x-block, y-block]
    F = np.kron(np.eye(2), F1)
    Q0 = np.kron(np.eye(2), Q1)
    H = np.kron(np.eye(2), H1)
    return F, Q0, H, k


def kalman(y, kind, q, r, smooth=False):
    """Kalman filter (and optional Rauch-Tung-Striebel smoother). q: process-noise intensity, r: measurement variance."""
    F, Q0, H, k = model(kind)
    n = F.shape[0]
    Q, R = q * Q0, r * np.eye(2)
    m = np.zeros(n)
    m[[0, k]] = y[0]  # start at the first measurement, velocities zero
    P = np.eye(n) * 1e4  # very uncertain about velocity/acceleration at the start
    P[0, 0] = P[k, k] = r
    ms, Ps, mp, Pp, ll, innov, S_list = [], [], [], [], 0.0, [], []
    for t in range(T):
        if t > 0:
            m, P = F @ m, F @ P @ F.T + Q
        mp.append(m.copy())
        Pp.append(P.copy())
        S = H @ P @ H.T + R
        v = y[t] - H @ m
        K = P @ H.T @ np.linalg.inv(S)
        m, P = m + K @ v, (np.eye(n) - K @ H) @ P
        ll += -0.5 * (np.log(np.linalg.det(2 * np.pi * S)) + v @ np.linalg.solve(S, v))
        ms.append(m.copy())
        Ps.append(P.copy())
        innov.append(v)
        S_list.append(S)
    ms, Ps = np.array(ms), np.array(Ps)
    if smooth:
        sm = ms.copy()
        for t in range(T - 2, -1, -1):
            G = Ps[t] @ F.T @ np.linalg.inv(Pp[t + 1])
            sm[t] = ms[t] + G @ (sm[t + 1] - mp[t + 1])
        ms = sm
    return ms, ll, np.array(innov), np.array(S_list), k


def fit(y, kind):
    """Maximum likelihood estimate of (q, r) from the innovations - uses only the measurements."""
    f = lambda p: -kalman(y, kind, np.exp(p[0]), np.exp(p[1]))[1]
    best = min((minimize(f, x0, method="Nelder-Mead") for x0 in ([0, 0], [3, 0], [-3, 2])), key=lambda r: r.fun)
    return np.exp(best.x), -best.fun


WARM = 10  # the filter starts with no idea of the velocity; exclude the first 10 steps from the "after warm-up" columns


def pos_rmse(est, k, start=0):
    e = est[start:, [0, k]] - truth[start:, :2]
    return float(np.sqrt((e ** 2).mean()))


def vel_rmse(est, k, start=0):
    e = est[start:, [1, k + 1]] - truth[start:, 2:4]
    return float(np.sqrt((e ** 2).mean()))


# ----------------------------------------------------------------------------
# 3. Estimate and compare
# ----------------------------------------------------------------------------
rows = []
raw = float(np.sqrt(((ys - truth[:, :2]) ** 2).mean()))
raw_w = float(np.sqrt(((ys[WARM:] - truth[WARM:, :2]) ** 2).mean()))
rows.append(("Raw measurements", raw, raw_w, np.nan, np.nan))
results = {}
for kind, label in [("CA", "Constant-acceleration"), ("CV", "Constant-velocity (misspecified)")]:
    (q, r), ll = fit(ys, kind)
    flt, _, innov, S, k = kalman(ys, kind, q, r)
    smo, *_ = kalman(ys, kind, q, r, smooth=True)
    results[kind] = (q, r, ll, flt, smo, innov, S, k)
    log(f"{label}: ML estimates q = {q:.3g}, measurement std = {np.sqrt(r):.2f}, log-likelihood = {ll:.1f}")
    rows.append((f"Kalman filter, {label}", pos_rmse(flt, k), pos_rmse(flt, k, WARM), vel_rmse(flt, k), vel_rmse(flt, k, WARM)))
    rows.append((f"Kalman smoother, {label}", pos_rmse(smo, k), pos_rmse(smo, k, WARM), vel_rmse(smo, k), vel_rmse(smo, k, WARM)))

log("\nTrue measurement noise std (from data): %.2f / %.2f" % tuple(noise.std(axis=0)))
log("\nPosition RMSE and velocity RMSE against the true trajectory (position units / velocity units)")
log(f"{'model':50s} {'pos all':>8s} {'pos >=10':>9s} {'vel all':>8s} {'vel >=10':>9s}")
for n, p, pw, v, vw in rows:
    fmt = lambda z: "       -" if np.isnan(z) else f"{z:8.2f}"
    log(f"{n:50s} {p:8.2f} {pw:9.2f} {fmt(v)} {fmt(vw)}")

# Diagnostics for the better model: standardised innovations should be white noise with unit variance
q, r, ll, flt, smo, innov, S, k = results["CA"]
std_innov = np.array([innov[t] / np.sqrt(np.diag(S[t])) for t in range(T)])
nis = np.array([innov[t] @ np.linalg.solve(S[t], innov[t]) for t in range(T)])
log("\nDiagnostics, constant-acceleration model (innovations after the first 5 steps)")
log(f"  standardised innovation std (x, y): {std_innov[5:].std(axis=0).round(2)}  (should be about 1)")
log(f"  mean normalised innovation squared: {nis[5:].mean():.2f}  (should be about 2 = number of measurements)")
ac = [np.corrcoef(std_innov[6:, i], std_innov[5:-1, i])[0, 1] for i in range(2)]
log(f"  lag-1 autocorrelation of innovations: {np.round(ac, 2)}  (should be about 0)")

# ----------------------------------------------------------------------------
# 4. Figures
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(truth[:, 0], truth[:, 1], "k-", lw=2, label="true path")
axes[0].plot(ys[:, 0], ys[:, 1], "o", ms=3, alpha=0.5, label="noisy measurements")
axes[0].plot(flt[:, 0], flt[:, k], "-", label="Kalman filter")
axes[0].plot(smo[:, 0], smo[:, k], "--", label="Kalman smoother")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend(fontsize=8)
axes[0].set_title("Trajectory")
axes[1].plot(truth[:, 2], "k-", lw=2, label="true vx")
axes[1].plot(flt[:, 1], label="filter estimate of vx")
axes[1].plot(smo[:, 1], "--", label="smoother estimate of vx")
axes[1].set_title("Velocity in x is never observed, only inferred")
axes[1].set_xlabel("time step")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "figures" / "kalman_tracking.png", dpi=150)
plt.close(fig)
out.close()
print("\nDone. See results.txt and figures/.")

"""Diagram: how each inference-time execution strategy turns predicted
action chunks into executed actions. Synthetic 1-D action trajectories."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)
T = 160  # timesteps shown
CHUNK = 100  # prediction length


def base_traj(t):
    return 0.55 * np.sin(t / 22.0) + 0.25 * np.sin(t / 9.0 + 1.2)


def chunk_pred(t0, drift):
    """Chunk predicted at time t0: true trajectory + its own systematic error."""
    t = np.arange(t0, t0 + CHUNK)
    err = drift * (0.35 + 0.1 * np.sin(t / 15 + drift * 7))
    return t, base_traj(t) + err


fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
fig.suptitle(
    "Inference-time execution strategies for chunked policies\n"
    "(grey dashed = predicted chunks; colored = actions actually executed)",
    fontsize=13,
)

PRED_KW = dict(color="#999999", linestyle="--", linewidth=1.3, alpha=0.8)
JERK_KW = dict(color="#d7191c", s=70, zorder=5, marker="o", facecolors="none", linewidths=2)


def draw_replan_lines(ax, points):
    for p in points:
        ax.axvline(p, color="#bbbbbb", linewidth=0.8, linestyle=":")


# ── 1. default: execute the full chunk ──
ax = axes[0]
t1, c1 = chunk_pred(0, drift=0.5)
t2, c2 = chunk_pred(100, drift=-0.4)
ax.plot(t1, c1, **PRED_KW)
ax.plot(t2, c2, **PRED_KW)
ax.plot(t1, c1, color="#2c7fb8", linewidth=2.5)
ax.plot(t2[: T - 100], c2[: T - 100], color="#2c7fb8", linewidth=2.5)
ax.scatter([100], [c2[0]], **JERK_KW)
ax.annotate("jump: stale 100-step-old plan meets new plan", xy=(100, c2[0]), xytext=(108, c2[0] + 0.45), fontsize=9, arrowprops=dict(arrowstyle="->"))
draw_replan_lines(ax, [0, 100])
ax.set_title("1) Default: predict 100, execute all 100  —  66%", loc="left", fontsize=11, fontweight="bold")

# ── 2. replan-50 ──
ax = axes[1]
segs = [(0, 0.5), (50, -0.35), (100, 0.4)]
prev = None
for t0, d in segs:
    t, c = chunk_pred(t0, d)
    ax.plot(t, c, **PRED_KW)
    n = min(50, T - t0)
    ax.plot(t[:n], c[:n], color="#1a9641", linewidth=2.5)
    if prev is not None:
        ax.scatter([t0], [c[0]], **JERK_KW)
    prev = (t, c)
ax.annotate("smaller, more frequent jumps;\nfresher plans between them", xy=(50, chunk_pred(50, -0.35)[1][0]), xytext=(58, 0.75), fontsize=9, arrowprops=dict(arrowstyle="->"))
draw_replan_lines(ax, [0, 50, 100, 150])
ax.set_title("2) Replan-50: predict 100, execute 50  —  80%  (25 is worse: jumps dominate)", loc="left", fontsize=11, fontweight="bold")

# ── 3. temporal ensembling ──
ax = axes[2]
ens_preds = []
for t0 in range(0, T, 10):  # show a subset of the per-step predictions
    t, c = chunk_pred(t0, rng.normal(0.0, 0.45))
    ens_preds.append((t, c))
    ax.plot(t[:60], c[:60], color="#999999", linestyle="--", linewidth=0.8, alpha=0.5)
# executed = exponentially weighted average over all chunks covering each t (older obs get weight)
tt = np.arange(T)
ex = []
for ti in tt:
    vals, ws = [], []
    for t0 in range(0, ti + 1, 2):
        if ti - t0 < CHUNK:
            _, c = chunk_pred(t0, 0.45 * np.sin(t0))
            vals.append(c[ti - t0])
            ws.append(np.exp(-0.01 * (ti - t0)))  # coeff 0.01: old predictions keep weight
    ex.append(np.average(vals, weights=ws))
ax.plot(tt, ex, color="#e66101", linewidth=2.5)
ax.plot(tt, base_traj(tt), color="#444444", linewidth=1.0, alpha=0.5)
ax.annotate("smooth but lagged: average is dominated\nby predictions from outdated observations", xy=(90, ex[90]), xytext=(98, ex[90] - 0.75), fontsize=9, arrowprops=dict(arrowstyle="->"))
ax.set_title("3) Temporal ensembling: predict every step, average all overlaps  —  64%", loc="left", fontsize=11, fontweight="bold")

# ── 4. crossfade (ours) ──
ax = axes[3]
XF = 15
t1, c1 = chunk_pred(0, 0.5)
t2, c2 = chunk_pred(50, -0.35)
t3, c3 = chunk_pred(100, 0.4)
for t, c in [(t1, c1), (t2, c2), (t3, c3)]:
    ax.plot(t, c, **PRED_KW)
ax.plot(t1[:50], c1[:50], color="#5e3c99", linewidth=2.5)


def xfade(old_t, old_c, new_t, new_c, t0):
    alpha = np.linspace(1.0 / XF, 1.0, XF)
    old_tail = old_c[t0 - old_t[0] : t0 - old_t[0] + XF]
    blend = alpha * new_c[:XF] + (1 - alpha) * old_tail
    return blend


b1 = xfade(t1, c1, t2, c2, 50)
ax.plot(np.arange(50, 50 + XF), b1, color="#e66101", linewidth=3.2)
ax.plot(t2[XF:50], c2[XF:50], color="#5e3c99", linewidth=2.5)
b2 = xfade(t2, c2, t3, c3, 100)
ax.plot(np.arange(100, 100 + XF), b2, color="#e66101", linewidth=3.2)
n3 = T - 100 - XF
ax.plot(t3[XF : XF + n3], c3[XF : XF + n3], color="#5e3c99", linewidth=2.5)
ax.annotate("15-step blend: old chunk's tail covers the same\ntimesteps -> ramp from old plan onto new plan", xy=(107, b2[7]), xytext=(112, b2[7] + 0.55), fontsize=9, arrowprops=dict(arrowstyle="->"))
draw_replan_lines(ax, [0, 50, 100, 150])
ax.set_title("4) Crossfade (ours): replan-50 + blend at boundaries — no jumps, no lag  —  82%", loc="left", fontsize=11, fontweight="bold")

# ── 5. conditioning (generative policies) ──
ax = axes[4]
t1, c1 = chunk_pred(0, 0.5)
ax.plot(t1, c1, **PRED_KW)
ax.plot(t1[:50], c1[:50], color="#0571b0", linewidth=2.5)
# new chunk: overlap region frozen to old plan, remainder generated consistently
t2 = np.arange(50, 150)
frozen = c1[50:65]
gen = base_traj(t2[15:]) - 0.32
gen = gen + (frozen[-1] - gen[0]) * np.exp(-(np.arange(len(gen))) / 18)
ax.plot(t2[:15], frozen, color="#fdae61", linewidth=4, solid_capstyle="butt")
ax.plot(t2[15:], gen, color="#0571b0", linewidth=2.5)
ax.axvspan(50, 65, color="#fdae61", alpha=0.15)
ax.annotate("overlap FROZEN (inpainting): sampler must\ngenerate a future consistent with it\n(Real-Time Chunking; for DP / pi0 / flow VLAs)", xy=(57, frozen[7]), xytext=(72, frozen[7] - 0.85), fontsize=9, arrowprops=dict(arrowstyle="->"))
draw_replan_lines(ax, [0, 50])
ax.set_title("5) Conditioning (for generative policies): freeze overlap, model inpaints the rest — averaging samples would blend modes", loc="left", fontsize=11, fontweight="bold")

for ax in axes:
    ax.set_xlim(-2, T + 2)
    ax.set_yticks([])
    ax.set_ylabel("action", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
axes[-1].set_xlabel("timestep", fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("media/inference_methods.png", dpi=160)
print("saved media/inference_methods.png")

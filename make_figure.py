"""Generate the methods-comparison figure from recorded results."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
fig.suptitle(
    "VLA AutoResearch: closed-loop sim eval on ALOHA TransferCube (50 episodes/point, RTX 5090)",
    fontsize=13,
)

# ── (a) replan horizon curve (frozen ACT 20k, dev seeds) ──
ax = axes[0]
horizon = [25, 40, 50, 75, 100]
succ = [52, 72, 80, 64, 66]
ax.plot(horizon, succ, "o-", color="#1f77b4", linewidth=2, markersize=7)
ax.annotate("peak: replan every 50", xy=(50, 80), xytext=(48, 86), fontsize=9, ha="center")
ax.axhline(66, color="gray", linestyle="--", linewidth=1)
ax.text(95, 67.2, "default (100)", fontsize=8, color="gray", ha="right")
ax.set_xlabel("actions executed per 100-step chunk (replan horizon)")
ax.set_ylabel("success rate (%)")
ax.set_title("(a) Replan horizon is non-monotonic")
ax.set_ylim(40, 95)

# ── (b) chunk-boundary crossfade (nas=50, dev seeds) ──
ax = axes[1]
xf = ["none", "8", "15", "30"]
xf_succ = [80, 76, 82, 80]
xf_rew = [215.0, 227.2, 230.7, 228.7]
x = range(len(xf))
b = ax.bar(x, xf_rew, color=["#999999", "#74add1", "#2c7fb8", "#74add1"], width=0.6)
for i, (r, s) in enumerate(zip(xf_rew, xf_succ)):
    ax.text(i, r + 1.5, f"{s}%", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(list(x))
ax.set_xticklabels(xf)
ax.set_xlabel("crossfade length (steps) at chunk boundary")
ax.set_ylabel("avg episode reward (max 400)")
ax.set_title("(b) Crossfade smooths chunk transitions\n(bars: reward; labels: success)")
ax.set_ylim(200, 240)

# ── (c) method comparison, dev vs held-out seeds ──
ax = axes[2]
methods = [
    ("VQ-BeT 20k\n(default cfg)", 2, None),
    ("ACT 20k\ndefault", 66, 74),
    ("ACT 20k\nchampion", 82, 76),
    ("ACT 80k\ndefault", 68, 76),
    ("ACT 80k\nchampion", 82, 88),
]
xpos = range(len(methods))
w = 0.38
dev = [m[1] for m in methods]
held = [m[2] for m in methods]
ax.bar([i - w / 2 for i in xpos], dev, w, label="dev seeds", color="#2c7fb8")
ax.bar(
    [i + w / 2 for i in xpos],
    [h if h is not None else 0 for h in held],
    w,
    label="held-out seeds",
    color="#fdae61",
)
for i, (d, h) in enumerate(zip(dev, held)):
    ax.text(i - w / 2, d + 1, str(d), ha="center", fontsize=9)
    if h is not None:
        ax.text(i + w / 2, h + 1, str(h), ha="center", fontsize=9)
ax.set_xticks(list(xpos))
ax.set_xticklabels([m[0] for m in methods], fontsize=8.5)
ax.set_ylabel("success rate (%)")
ax.set_title("(c) Methods: champion = replan-50 + crossfade-15")
ax.legend(loc="upper left", fontsize=9)
ax.set_ylim(0, 100)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("media/comparison.png", dpi=160)
print("saved media/comparison.png")

"""Block diagram comparing the three policy architectures in the bake-off."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(15, 8.5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.axis("off")
fig.suptitle("Same inputs, three ways to generate actions", fontsize=15, y=0.97)

COLS = {"act": 2.5, "dp": 7.5, "vqbet": 12.5}
W = 3.9  # box width


def box(x, y, text, color, h=0.85, fs=9.5, bold=False):
    b = FancyBboxPatch(
        (x - W / 2, y - h / 2),
        W,
        h,
        boxstyle="round,pad=0.08",
        facecolor=color,
        edgecolor="#444444",
        linewidth=1.2,
    )
    ax.add_patch(b)
    ax.text(
        x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal"
    )


def arrow(x, y1, y2, label=None):
    a = FancyArrowPatch(
        (x, y1), (x, y2), arrowstyle="-|>", mutation_scale=16, color="#444444", linewidth=1.4
    )
    ax.add_patch(a)
    if label:
        ax.text(x + 0.12, (y1 + y2) / 2, label, fontsize=8, va="center", color="#333333")


# ── column titles ──
ax.text(COLS["act"], 9.45, "ACT  (52M)", ha="center", fontsize=13, fontweight="bold", color="#1a5276")
ax.text(COLS["act"], 9.05, "action chunking transformer — regression", ha="center", fontsize=9, style="italic")
ax.text(COLS["dp"], 9.45, "Diffusion Policy  (260M)", ha="center", fontsize=13, fontweight="bold", color="#6c3483")
ax.text(COLS["dp"], 9.05, "iterative denoising — generative", ha="center", fontsize=9, style="italic")
ax.text(COLS["vqbet"], 9.45, "VQ-BeT  (40M)", ha="center", fontsize=13, fontweight="bold", color="#943126")
ax.text(COLS["vqbet"], 9.05, "behavior transformer — discrete tokens", ha="center", fontsize=9, style="italic")

# ── shared input row ──
for c in COLS.values():
    box(c, 8.2, "camera RGB 480x640  +  14-D joint state", "#eaecee")
    arrow(c, 7.75, 7.3)
    box(c, 6.85, "ResNet-18 vision encoder", "#d6dbdf")

# ── ACT column ──
c = COLS["act"]
arrow(c, 6.4, 5.95)
box(c, 5.3, "Transformer encoder\n(vision feats + joints + latent z)\nCVAE: z~posterior at train, z=0 at test", "#d4e6f1", h=1.5)
arrow(c, 4.55, 4.1)
box(c, 3.45, "Transformer decoder\ncross-attends, outputs all 100\nactions in ONE forward pass", "#d4e6f1", h=1.5)
arrow(c, 2.7, 2.25, "L1 regression loss")
box(c, 1.8, "100-step action chunk", "#aed6f1", bold=True)
ax.text(c, 0.95, "deterministic: multimodal demos\nget averaged | fastest inference\nbake-off: 66% -> 82% w/ champion cfg", ha="center", fontsize=8.5, color="#1a5276")

# ── DP column ──
c = COLS["dp"]
arrow(c, 6.4, 5.95)
box(c, 5.3, "observation embedding\n(2 obs steps, FiLM conditioning)", "#e8daef", h=1.2)
arrow(c, 4.7, 4.25)
box(c, 3.45, "1D conditional UNet\nstarts from pure NOISE, denoises the\naction sequence over ~100 steps\n(repeated forward passes)", "#e8daef", h=1.9)
arrow(c, 2.5, 2.25, "denoising (DDPM) loss")
box(c, 1.8, "64-step trajectory (execute 32)", "#d2b4de", bold=True)
ax.text(c, 0.95, "generative: keeps multimodal demos\ndistinct | slowest inference (iterative)\nbake-off: 46% (2x ACT train time)", ha="center", fontsize=8.5, color="#6c3483")

# ── VQ-BeT column ──
c = COLS["vqbet"]
arrow(c, 6.4, 5.95)
box(c, 5.3, "residual VQ-VAE\ncompresses action snippets into a\nDISCRETE codebook of motion primitives\n(pretrained, stage 1)", "#fadbd8", h=1.7)
arrow(c, 4.45, 4.0)
box(c, 3.45, "minGPT predicts next code token\n+ small offset head (categorical,\nlike a language model over motions)", "#fadbd8", h=1.4)
arrow(c, 2.7, 2.25, "cross-entropy + offset loss")
box(c, 1.8, "decoded action chunk", "#f5b7b1", bold=True)
ax.text(c, 0.95, "discrete: multimodality via token\ndistribution | tuned for 2D tasks\nbake-off: 2% (default cfg, 14-DoF)", ha="center", fontsize=8.5, color="#943126")

# separators
for x in (5.0, 10.0):
    ax.plot([x, x], [0.4, 8.8], color="#cccccc", linewidth=1, linestyle=":")

plt.savefig("media/architectures.png", dpi=160, bbox_inches="tight")
print("saved media/architectures.png")

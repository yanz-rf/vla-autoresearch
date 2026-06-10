"""Architecture bake-off figure: same dataset, same 20k-step budget, same eval.

Reads rows from results.tsv (train+eval harness log)."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# tag -> (display name, params, color)
MODELS = {
    "baseline": ("ACT", "80M", "#2c7fb8"),
    "dp_20k": ("Diffusion Policy", "260M", "#7b3294"),
    "dp_40k": ("Diffusion Policy\n2x budget", "260M", "#9b59b6"),
    "vqbet_20k": ("VQ-BeT", "40M", "#d7191c"),
}

rows = {}
with open("results.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["tag"] in MODELS:
            rows[r["tag"]] = r

# Champion inference config on the same ACT weights (from results_eval.tsv).
extra = [("ACT + champion\ninference", "80M", 82.0, 230.7, "#a6bddb")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    "Architecture bake-off: 50 human demos, fixed 20k-step budget, 50-episode closed-loop eval\n"
    "(ALOHA TransferCube sim, RTX 5090)",
    fontsize=12,
)

names, params, succ, rew, colors = [], [], [], [], []
for tag, (name, p, c) in MODELS.items():
    if tag in rows:
        names.append(name)
        params.append(p)
        succ.append(float(rows[tag]["pc_success"]))
        rew.append(float(rows[tag]["avg_sum_reward"]))
        colors.append(c)
for name, p, s, r, c in extra:
    names.append(name)
    params.append(p)
    succ.append(s)
    rew.append(r)
    colors.append(c)

labels = [f"{n}\n({p})" for n, p in zip(names, params)]
x = range(len(names))

bars = ax1.bar(x, succ, color=colors, width=0.6)
for i, s in enumerate(succ):
    ax1.text(i, s + 1.5, f"{s:.0f}%", ha="center", fontsize=11, fontweight="bold")
ax1.set_xticks(list(x))
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("success rate (%)")
ax1.set_title("Task success")
ax1.set_ylim(0, 100)

ax2.bar(x, rew, color=colors, width=0.6)
for i, r in enumerate(rew):
    ax2.text(i, r + 4, f"{r:.0f}", ha="center", fontsize=11, fontweight="bold")
ax2.set_xticks(list(x))
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("avg episode reward (max ~400)")
ax2.set_title("Episode reward (time-integrated task progress)")

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("media/bakeoff.png", dpi=160)
print("saved media/bakeoff.png with", len(names), "models")

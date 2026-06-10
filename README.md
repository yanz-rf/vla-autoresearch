# vla-autoresearch

[karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted to
robot learning: an autonomous research loop for VLA / imitation-learning
policies where the scored metric is **closed-loop simulation success rate**
(not open-loop action error). Built in a 3-hour hackathon on a single RTX 5090.

**Headline result:** starting from a frozen ACT checkpoint at 66% success on
ALOHA TransferCube (gym-aloha, MuJoCo), the loop found an inference-time
configuration — replan every 50 of 100 predicted steps + a novel 15-step
chunk-boundary crossfade — reaching **82% on dev seeds (~+10 pooled across
seed sets)** in 11 experiments / ~30 min of compute. On a fully-trained 80k-step
model the same config holds up: **85% pooled vs 72% default (88% on held-out
seeds)**. See [FINDINGS.md](FINDINGS.md),
including the held-out-seed validation showing why eval noise is the central
methodological issue for autoresearch-style loops in robotics.

## How it works

Same shape as karpathy's loop — the coding agent is the orchestrator:

| autoresearch (LM) | this repo (robotics) |
|---|---|
| mutable `train.py` | LeRobot policy code (editable install) + CLI overrides |
| read-only `prepare.py` (val_bpb) | `run_eval_experiment.sh` / `run_experiment.sh` (sim success, fixed seeds) |
| `program.md` protocol | `program_hackathon.md` (eval-only) / `program.md` (train+eval) |
| `results.tsv` | `results_eval.tsv` / `results.tsv` |

Two loop variants:

- **Eval-only** (~2.5 min/experiment): freeze a checkpoint, research
  inference-time ideas (replan horizon, chunk crossfade, ensembling, ...).
  `EXP_DESC="idea" ./run_eval_experiment.sh <tag> [overrides...]`
- **Train+eval** (~12 min/experiment on a 5090): fixed 20k-step ACT training
  budget, then 50-episode eval. `EXP_DESC="idea" ./run_experiment.sh <tag> [overrides...]`

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
git clone https://github.com/huggingface/lerobot && cd lerobot
git am ../lerobot-patches/*.patch        # eval fix + experiment code (env-var gated)
uv pip install -e ".[training,aloha,pusht]"
```

Notes for RTX 5090 (Blackwell/sm_120): LeRobot's pyproject already pins the
cu128 torch index. Use `--dataset.video_backend=pyav` (torchcodec needs newer
system FFmpeg) and `--eval.use_async_envs=false` (forkserver workers don't
import gym_aloha). `MUJOCO_GL=egl` for headless rendering.

## Repo contents

- `run_eval_experiment.sh`, `run_experiment.sh` — read-only scoring harnesses
- `program_hackathon.md`, `program.md` — agent protocols (point Claude Code at one)
- `results_eval.tsv`, `FINDINGS.md` — experiment log and writeup
- `results_80k.tsv`, `eval_runs_80k/` — 80k-model validation (default vs champion, dev + held-out seeds)
- `lerobot-patches/` — the code-level experiments (crossfade, deviation-replan)
- `eval_runs/*/videos/` — closed-loop rollout videos for every experiment

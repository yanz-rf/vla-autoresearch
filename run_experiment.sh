#!/usr/bin/env bash
# READ-ONLY EVAL HARNESS — the research agent must never modify this file.
# Trains a policy under a fixed step budget, then scores it by closed-loop
# simulation success rate with fixed seeds. Analogous to prepare.py in
# karpathy/autoresearch: the scoring function lives here, outside the
# mutable surface, so experiments can't game the eval.
#
# Usage: ./run_experiment.sh <tag> [extra lerobot-train overrides...]
# Output: appends one row to results.tsv; full logs in runs/<tag>/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(dirname "$ROOT")"
source "$WS/.venv/bin/activate"
export MUJOCO_GL=egl

# ── Fixed experiment budget & eval protocol (do not change per-experiment) ──
DATASET=lerobot/aloha_sim_transfer_cube_human
ENV_TYPE=aloha
ENV_TASK=AlohaTransferCube-v0
TRAIN_STEPS=20000          # fixed compute budget for comparability
EVAL_EPISODES=50           # enough to resolve ~10% differences in success
EVAL_BATCH=10              # parallel sim envs
EVAL_START_SEED=100000     # fixed seed -> same initial conditions every run
TRAIN_SEED=1000

TAG="${1:?usage: run_experiment.sh <tag> [train overrides...]}"
shift || true
RUN_DIR="$ROOT/runs/$TAG"
mkdir -p "$ROOT/runs"

if [[ -e "$RUN_DIR" ]]; then
  echo "run dir $RUN_DIR already exists; pick a new tag" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

GIT_SHA=$(git -C "$WS/lerobot" rev-parse --short HEAD 2>/dev/null || echo "nogit")

echo "[harness] tag=$TAG sha=$GIT_SHA steps=$TRAIN_STEPS eval_eps=$EVAL_EPISODES"
t0=$SECONDS

lerobot-train \
  --dataset.repo_id=$DATASET \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$RUN_DIR/train" \
  --job_name="$TAG" \
  --batch_size=8 \
  --steps=$TRAIN_STEPS \
  --log_freq=500 \
  --save_freq=$TRAIN_STEPS \
  --eval_freq=0 \
  --wandb.enable=false \
  --seed=$TRAIN_SEED \
  "$@" > "$RUN_DIR/train.log" 2>&1
train_s=$((SECONDS - t0))

CKPT="$RUN_DIR/train/checkpoints/last/pretrained_model"
[[ -d "$CKPT" ]] || CKPT=$(ls -d "$RUN_DIR"/train/checkpoints/*/pretrained_model | tail -1)

t1=$SECONDS
lerobot-eval \
  --policy.path="$CKPT" \
  --policy.device=cuda \
  --env.type=$ENV_TYPE \
  --env.task=$ENV_TASK \
  --eval.n_episodes=$EVAL_EPISODES \
  --eval.batch_size=$EVAL_BATCH \
  --eval.use_async_envs=false \
  --seed=$EVAL_START_SEED \
  --output_dir="$RUN_DIR/eval" > "$RUN_DIR/eval.log" 2>&1
eval_s=$((SECONDS - t1))

EVAL_JSON="$RUN_DIR/eval/eval_info.json"
SUCCESS=$(python -c "import json; print(json.load(open('$EVAL_JSON'))['overall']['pc_success'])")
REWARD=$(python -c "import json; print(round(json.load(open('$EVAL_JSON'))['overall']['avg_sum_reward'],2))")

DESC="${EXP_DESC:-}"
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "$TAG" "$GIT_SHA" "$SUCCESS" "$REWARD" "$train_s" "$eval_s" "$DESC" >> "$ROOT/results.tsv"

echo "[harness] DONE tag=$TAG pc_success=$SUCCESS avg_sum_reward=$REWARD train=${train_s}s eval=${eval_s}s"

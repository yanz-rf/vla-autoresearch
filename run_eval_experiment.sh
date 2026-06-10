#!/usr/bin/env bash
# READ-ONLY EVAL HARNESS (hackathon edition) — never modified by the agent.
# Scores a FROZEN checkpoint + current inference-time code/config by
# closed-loop sim success over fixed-seed episodes. No training.
#
# Usage: EXP_DESC="..." ./run_eval_experiment.sh <tag> [lerobot-eval overrides...]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(dirname "$ROOT")"
source "$WS/.venv/bin/activate"
export MUJOCO_GL=egl

# ── Fixed eval protocol ──
CKPT="$ROOT/runs/baseline/train/checkpoints/last/pretrained_model"
ENV_TYPE=aloha
ENV_TASK=AlohaTransferCube-v0
EVAL_EPISODES=50
EVAL_BATCH=10
EVAL_START_SEED=100000

TAG="${1:?usage: run_eval_experiment.sh <tag> [overrides...]}"
shift || true
RUN_DIR="$ROOT/eval_runs/$TAG"
[[ -e "$RUN_DIR" ]] && { echo "tag exists" >&2; exit 1; }
mkdir -p "$RUN_DIR"

GIT_SHA=$(git -C "$WS/lerobot" rev-parse --short HEAD 2>/dev/null || echo "nogit")
t0=$SECONDS

lerobot-eval \
  --policy.path="$CKPT" \
  --policy.device=cuda \
  --env.type=$ENV_TYPE \
  --env.task=$ENV_TASK \
  --eval.n_episodes=$EVAL_EPISODES \
  --eval.batch_size=$EVAL_BATCH \
  --eval.use_async_envs=false \
  --seed=$EVAL_START_SEED \
  --output_dir="$RUN_DIR" \
  "$@" > "$RUN_DIR/eval.log" 2>&1
eval_s=$((SECONDS - t0))

EVAL_JSON="$RUN_DIR/eval_info.json"
SUCCESS=$(python -c "import json; print(json.load(open('$EVAL_JSON'))['overall']['pc_success'])")
REWARD=$(python -c "import json; print(round(json.load(open('$EVAL_JSON'))['overall']['avg_sum_reward'],2))")

printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
  "$TAG" "$GIT_SHA" "$SUCCESS" "$REWARD" "$eval_s" "${EXP_DESC:-}" >> "$ROOT/results_eval.tsv"
echo "[harness] DONE tag=$TAG pc_success=$SUCCESS avg_sum_reward=$REWARD eval=${eval_s}s"

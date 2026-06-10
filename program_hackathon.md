# VLA AutoResearch — 3-hour hackathon edition

Goal: maximize closed-loop sim success of a FROZEN ACT checkpoint on ALOHA
TransferCube by researching **inference-time** ideas. No training — every
experiment is a ~2.5 min eval. Budget: ~40 experiments.

## Substrate

- Frozen checkpoint: `runs/baseline/train/checkpoints/last/pretrained_model`
  (ACT, 20k steps, 66.0% success at default inference settings).
- Mutable surface:
  1. CLI overrides passed to `run_eval_experiment.sh` (e.g.
     `--policy.n_action_steps=50`, `--policy.temporal_ensemble_coeff=0.01`).
  2. Inference code in `../lerobot/src/lerobot/policies/act/modeling_act.py`
     (select_action path, temporal ensembling, any test-time logic) and the
     env/obs preprocessing in the policy processor — commit each change.
- Read-only: this protocol, `run_eval_experiment.sh`, checkpoint weights,
  env, seeds, episode count. Never touch the scoring.

## Idea space (research, not grid search)

Action-chunk execution horizon; temporal ensembling variants (coeff,
windowing, schemes beyond exponential); action interpolation/smoothing;
test-time ensembling over augmented observations; latent sampling (z != 0);
replan-on-uncertainty; early-episode vs late-episode strategies. Propose
ideas from reading the code and the failure videos in eval_runs/<tag>/videos/.

## Loop

1. ONE idea per experiment. Commit code changes in ../lerobot first.
2. `EXP_DESC="<idea>" ./run_eval_experiment.sh <tag> [overrides...]`
3. Read appended row in results_eval.tsv.
4. Keep/revert (50 eps => ~7% std at p=0.5; be conservative):
   - ≥ best+6: KEEP. Within ±6: rerun ONCE with seed-shifted protocol is NOT
     allowed (fixed seeds) — instead mark ambiguous and only build on it if a
     second related experiment confirms the direction.
   - else REVERT (git -C ../lerobot reset --hard <last-good>).
5. Log everything; never ask permission; stop at hackathon end and write
   a short findings summary.

## results_eval.tsv columns

tag, git_sha, pc_success, avg_sum_reward, eval_s, description

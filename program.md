# VLA AutoResearch — program.md

Adaptation of karpathy/autoresearch to robot imitation learning: instead of
optimizing val_bpb on LM pretraining, you optimize **closed-loop simulation
success rate** of a manipulation policy under a fixed training budget.

## Substrate

- Mutable surface: the LeRobot clone at `../lerobot` (installed editable).
  You may modify policy code (`src/lerobot/policies/act/**`) and pass
  hyperparameter overrides as extra CLI args to `run_experiment.sh`.
- Read-only: `run_experiment.sh` (the harness), this file's Protocol section,
  the eval environment, seeds, episode count, and the train step budget.
  Never change the scoring function or eval protocol to make a number go up.

## Task

Dataset: `lerobot/aloha_sim_transfer_cube_human` (50 human demos, bimanual
ALOHA, MuJoCo). Policy: ACT (~80M params). Score: `pc_success` over 50 eval
episodes with fixed seeds in gym-aloha `AlohaTransferCube-v0`.

Budget: 20,000 train steps (~10 min) + eval (~5 min) ≈ 15 min/experiment.

## Loop protocol

1. Work on a branch `autoresearch/<date>` in `../lerobot`.
2. Pick ONE idea: architecture tweak, hyperparameter, loss change, data
   augmentation, chunk size, etc. Everything inside the policy is fair game.
3. Commit the change in `../lerobot` (even pure-CLI-override experiments:
   note the overrides in the commit message).
4. Run: `EXP_DESC="<one line>" ./run_experiment.sh <tag> [overrides...]`
5. Read the appended row in `results.tsv`.
6. Keep/revert rule (eval is stochastic — be conservative):
   - pc_success ≥ best + 4 (absolute %): KEEP (commit stays).
   - within ±4 of best: AMBIGUOUS — rerun once with a different tag;
     keep only if the mean of both runs beats best by ≥ 2.
   - otherwise: REVERT (`git -C ../lerobot reset --hard <last-good>`).
7. Crashes: mark status crash in results.tsv (description column), revert,
   move on. Never ask permission; loop until stopped.

## results.tsv columns

tag, git_sha, pc_success, avg_sum_reward, train_s, eval_s, description

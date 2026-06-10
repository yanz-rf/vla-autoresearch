# VLA AutoResearch — hackathon findings (2026-06-10)

Autoresearch-style loop (karpathy/autoresearch adapted to robotics) where the
scored metric is **closed-loop simulation success rate**, not open-loop loss.
Frozen ACT checkpoint (20k steps, ALOHA TransferCube, MuJoCo via gym-aloha);
each experiment mutates **inference-time** code/config only, then runs a
fixed-seed 50-episode sim eval (~2.5 min on an RTX 5090).

## Results (dev seeds 100000+, 50 episodes)

| tag            | idea                                   | success | reward |
|----------------|----------------------------------------|--------:|-------:|
| baseline_infer | default: execute full 100-step chunk   | 66.0    | 199.5  |
| nas25          | replan every 25 steps                  | 52.0    | 170.3  |
| nas40          | replan every 40 steps                  | 72.0    | 204.3  |
| **nas50**      | replan every 50 steps                  | **80.0**| 215.0  |
| nas75          | replan every 75 steps                  | 64.0    | 191.5  |
| tens001        | ACT temporal ensembling (coeff 0.01)   | 64.0    | 193.9  |
| xfade8_nas50   | 8-step chunk-boundary crossfade        | 76.0    | 227.2  |
| **xfade15_nas50** | 15-step crossfade (champion)        | **82.0**| 230.7  |
| xfade30_nas50  | 30-step crossfade                      | 80.0    | 228.7  |
| replan_dbg     | champion rerun (instrumented)          | 82.0    | 230.1  |
| replan2_xf15   | deviation-triggered replanning         | 68.0    | 204.6  |

## Held-out seed validation (seed 200000, 50 episodes)

- champion (nas50 + xfade15): **76%**
- default inference:          **74%**

## Post-hackathon: does it transfer to a fully-trained model? (80k steps)

Retrained the same ACT config to 80k steps (~40 min on the 5090) and re-ran
the comparison, 50 episodes per cell:

| config                        | dev seeds (100000) | held-out (200000) | reward    |
|-------------------------------|-------------------:|------------------:|-----------|
| default inference             | 68%                | 76%               | 197 / 215 |
| champion (replan-50 + xfade15)| **82%**            | **88%**           | 221 / 243 |

- Training 4x longer barely moved default inference (~70% -> ~72% pooled):
  the model converges early on 50 demos.
- The inference-time gains fully transfer: pooled +13 (72% -> 85%), consistent
  across both seed sets and in reward. The discoveries are genuine inference
  improvements, not compensation for undertraining.

## Conclusions

1. **Replan horizon is the dominant inference knob** and is non-monotonic:
   peak at executing 50 of 100 predicted steps (too-frequent replanning adds
   chunk-boundary jitter; too-rare leaves the policy open-loop too long).
2. **Chunk-boundary crossfade** (blend new chunk's first k actions with the
   previous chunk's overlapping tail) adds a small consistent gain, clearest
   in episode reward (215 → ~230).
3. **Temporal ensembling did not help** this 20k-step checkpoint (64%).
4. **Deviation-triggered replanning failed** (68%): with batched envs the
   shared action queue makes the trigger global, and large tracking errors
   correlate with contact phases where mid-grasp replanning is harmful.
5. **Eval noise is the central methodological issue**: dev-set gap (+16) vs
   held-out gap (+2). Pooled estimate: champion ~80% vs default ~70%, i.e.
   a real but ~+10 gain. 50-episode evals carry ~7% std — autoresearch-style
   loops on robotics MUST use held-out seed validation and conservative
   keep/revert thresholds, or they will overfit the eval.

## Artifacts

- Loop protocol: `program_hackathon.md`; scorer: `run_eval_experiment.sh`
  (read-only); log: `results_eval.tsv`; per-run videos in `eval_runs/*/videos/`.
- Code experiments are commits in `../lerobot` (crossfade + replan-trigger,
  both env-var-gated, default-off).
- Full-training loop variant (20k-step budget, ~12 min/exp): `run_experiment.sh`
  + `program.md` — used to create the frozen baseline (66% @ 20k steps).

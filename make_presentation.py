"""Build a self-contained HTML slide deck (images/videos base64-embedded)."""

import base64
from pathlib import Path


def b64(path, mime):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:{mime};base64,{data}"


comparison = b64("media/comparison.png", "image/png")
bakeoff = b64("media/bakeoff.png", "image/png")
arch = b64("media/architectures.png", "image/png")
vid_success = b64("eval_runs_80k/champ_200000/videos/aloha_0/eval_episode_5.mp4", "video/mp4")
vid_fail = b64("eval_runs_80k/champ_200000/videos/aloha_0/eval_episode_1.mp4", "video/mp4")

slides = [
    # 1 ── title
    """
    <h1>VLA AutoResearch</h1>
    <h2 style="color:#5dade2">An autonomous research loop for robot policies,<br>scored by closed-loop simulation</h2>
    <p class="big">karpathy/autoresearch, ported from language models to robot manipulation</p>
    <p>3-hour hackathon &middot; single RTX 5090 &middot; June 2026</p>
    <p><code>github.com/yanz-rf/vla-autoresearch</code></p>
    """,
    # 2 ── motivation
    """
    <h2>Motivation</h2>
    <ul>
      <li><b>karpathy/autoresearch</b> (Mar 2026): a coding agent runs LM-pretraining experiments
          autonomously &mdash; one idea per run, fixed budget, keep/revert by val_bpb, ~100 experiments overnight.</li>
      <li><b>The robotics gap:</b> VLA / imitation policies are usually iterated on with <i>open-loop</i> metrics
          (action prediction error) &mdash; but low prediction error &ne; task success.</li>
      <li><b>Our port:</b> the experiment score is <i>closed-loop simulation success rate</i> &mdash;
          the policy actually controls the robot, mistakes compound, and only finished tasks count.</li>
      <li>No public robotics port of autoresearch existed. Now one does.</li>
    </ul>
    """,
    # 3 ── system design
    """
    <h2>System design: same loop, new scoring</h2>
    <table>
      <tr><th></th><th>autoresearch (LM)</th><th>ours (robotics)</th></tr>
      <tr><td>mutable surface</td><td>train.py</td><td>policy code (LeRobot, editable) + CLI overrides</td></tr>
      <tr><td>read-only scorer</td><td>val bits-per-byte</td><td>50 sim episodes, fixed seeds &rarr; % success</td></tr>
      <tr><td>protocol</td><td>program.md</td><td>program.md (keep/revert thresholds tuned for eval noise)</td></tr>
      <tr><td>audit trail</td><td>results.tsv + git</td><td>results.tsv + git + rollout videos</td></tr>
    </table>
    <ul>
      <li><b>Eval isolation:</b> scoring script and seeds are outside the mutable surface &mdash; the agent can't game the metric.</li>
      <li><b>Loop economics on one 5090:</b> train+eval experiment &asymp; 12 min; inference-only experiment &asymp; 2.5 min.</li>
    </ul>
    """,
    # 4 ── setup
    """
    <h2>Testbed</h2>
    <ul>
      <li><b>Task:</b> ALOHA TransferCube (MuJoCo, gym-aloha) &mdash; bimanual: right arm picks a cube, hands to left arm.</li>
      <li><b>Data:</b> 50 public human teleop demos (<code>lerobot/aloha_sim_transfer_cube_human</code>).</li>
      <li><b>Policy:</b> ACT, 80M params &mdash; predicts 100-action chunks from camera + joint state.</li>
      <li><b>Score:</b> % of 50 fixed-seed sim episodes fully completed; reward (0-4 staged progress) as dense secondary signal.</li>
      <li><b>Hackathon constraint:</b> trained once (20k steps, 10 min, 66%), then froze weights &mdash;
          all research on <i>inference-time</i> behavior at 2.5 min/experiment.</li>
    </ul>
    """,
    # 5 ── discoveries
    f"""
    <h2>What the loop found (11 experiments, ~30 min compute)</h2>
    <img src="{comparison}" style="width:96%">
    <ul style="font-size:0.85em">
      <li><b>(a)</b> Execute 50 of 100 predicted actions: 66% &rarr; 80%. Non-monotonic &mdash; replanning too often (25) is <i>worse</i> than default.</li>
      <li><b>(b)</b> Novel trick: <b>crossfade chunk boundaries</b> (blend new chunk with old chunk's overlapping tail) &rarr; 82%, reward 215&rarr;231.</li>
      <li><b>(c)</b> Champion config validated on held-out seeds and on a 4&times;-longer-trained model.</li>
    </ul>
    """,
    # 6 ── videos
    f"""
    <h2>Closed-loop rollouts (champion config, held-out seeds)</h2>
    <div style="display:flex;gap:2em;justify-content:center">
      <div><video src="{vid_success}" controls loop muted autoplay style="height:360px"></video>
        <p>success: grasp &rarr; lift &rarr; handoff (reward 313)</p></div>
      <div><video src="{vid_fail}" controls loop muted style="height:360px"></video>
        <p>the one failure in 10: fumbles before handoff (reward 72)</p></div>
    </div>
    <p style="font-size:0.8em">every experiment's eval saves 10 rollout videos &mdash; failures are debuggable, not just countable</p>
    """,
    # 7 ── validation
    """
    <h2>Does it hold up? (the part most hackathon demos skip)</h2>
    <table>
      <tr><th>config</th><th>dev seeds</th><th>held-out seeds</th></tr>
      <tr><td>ACT 20k, default</td><td>66%</td><td>74%</td></tr>
      <tr><td>ACT 20k, champion</td><td>82%</td><td>76%</td></tr>
      <tr><td>ACT 80k, default</td><td>68%</td><td>76%</td></tr>
      <tr><td><b>ACT 80k, champion</b></td><td><b>82%</b></td><td><b>88%</b></td></tr>
    </table>
    <ul>
      <li>4&times; more training barely moved default inference (~70&rarr;72% pooled) &mdash; the <i>inference research</i> was worth more (+13).</li>
      <li>Dev-set gain (+16) shrank on held-out seeds at 20k: <b>50-episode evals carry ~7% std</b>.
          Autoresearch loops on robotics must validate on held-out seeds or they overfit the eval.</li>
    </ul>
    """,
    # 8 ── architectures
    f"""
    <h2>Architecture bake-off: same data, same budget, same eval</h2>
    <img src="{arch}" style="width:80%">
    """,
    # 9 ── bakeoff results
    f"""
    <h2>Bake-off results</h2>
    <img src="{bakeoff}" style="width:88%">
    <p style="font-size:0.85em">DP/VQ-BeT use library defaults tuned on simpler benchmarks (floor, not ceiling) &mdash;
    tuning them is the loop's natural next assignment. DP at 2&times; budget (40k steps): running now.</p>
    """,
    # 10 ── learnings
    """
    <h2>Learnings</h2>
    <ol>
      <li><b>Closed-loop eval is the right loop signal</b> &mdash; the wins came from execution-time behavior
          that open-loop loss literally cannot see.</li>
      <li><b>At a fixed budget, inference-time research beat architecture search</b> (+16 dev / +13 pooled vs
          &minus;20 for swapping to Diffusion Policy).</li>
      <li><b>Eval noise is the central methodological issue</b> &mdash; conservative keep/revert thresholds,
          dense reward as tiebreaker, held-out seed validation.</li>
      <li><b>Defaults don't transfer across embodiments</b> (VQ-BeT: 2% on bimanual with PushT-tuned config).</li>
      <li><b>Agent-as-orchestrator works for robotics</b>: ~25 scored runs today, every claim traceable to
          a git commit + results row + videos.</li>
    </ol>
    <p style="font-size:0.9em;color:#9fb2c8"><b>Appendix</b> &mdash;
       A1: open-loop loss vs closed-loop success (mis)alignment data &middot;
       A2: which inference smoothing fits which architecture &middot;
       A3: why regression averages modes and diffusion doesn't</p>
    """,
    # 11 ── next
    """
    <h2>What's next</h2>
    <ul>
      <li><b>Harder target:</b> RoboTwin 2.0 &mdash; 50 bimanual ALOHA tasks, public leaderboard, baselines at
          10-40% under domain randomization (integrated in LeRobot already).</li>
      <li><b>Small VLAs in the loop:</b> SmolVLA (450M) fine-tunes on this GPU; pi0-LoRA fits in 32GB.</li>
      <li><b>Cross-architecture inference research:</b> does crossfade help every chunked policy?</li>
      <li><b>Overnight run:</b> the loop is built for it &mdash; ~100 train+eval experiments/night at 20k budget.</li>
    </ul>
    <p class="big" style="margin-top:1em"><code>github.com/yanz-rf/vla-autoresearch</code></p>
    """,
    # A1 ── open-loop vs closed-loop alignment
    """
    <h2>Appendix A1 &mdash; Open-loop loss vs closed-loop success: poor alignment</h2>
    <table>
      <tr><th>train step</th><th>open-loop L1 loss</th><th>closed-loop success (dev / held-out)</th></tr>
      <tr><td>20k</td><td>~0.11</td><td>66% / 74%</td></tr>
      <tr><td>80k</td><td>0.052 (2.3&times; better)</td><td>68% / 76% (statistical tie)</td></tr>
    </table>
    <ul style="font-size:1.05em">
      <li><b>Evidence 1:</b> loss fell smoothly for 60k more steps; task success didn't move.
          Past ~20k, gradients improved imitation of demo frames, not task completion.</li>
      <li><b>Evidence 2 (sharper):</b> all our inference configs share frozen weights &rarr;
          <i>identical</i> open-loop loss, yet closed-loop success ranged <b>52% &rarr; 82%</b>.
          The whole dimension we mined for gains is invisible to open-loop eval by construction.</li>
      <li><b>Why they decouple:</b> open-loop asks "predict the human's action in states the human visited";
          closed-loop asks "recover in states your own mistakes created". Compounding error and execution
          strategy live only in the second. Low loss is ~necessary, far from sufficient.</li>
    </ul>
    """,
    # A2 ── smoothing × architecture
    """
    <h2>Appendix A2 &mdash; Which inference smoothing fits which architecture</h2>
    <table style="font-size:0.95em">
      <tr><th>policy type</th><th>output is...</th><th>safe smoothing</th><th>examples</th></tr>
      <tr><td>regression (ACT)</td><td>point estimate (conditional mean)</td>
          <td><b>averaging</b> in action space</td>
          <td>temporal ensembling (ACT paper), our chunk crossfade</td></tr>
      <tr><td>generative (DP, &pi;0, flow VLAs)</td><td>one sample from a multimodal distribution</td>
          <td><b>conditioning or selection</b>, never sample-averaging</td>
          <td>warm-start / inpainting: Real-Time Chunking (arXiv 2506.07339, in LeRobot);
              selection: Bidirectional Decoding (arXiv 2408.17355)</td></tr>
    </table>
    <ul style="font-size:1.0em">
      <li>Averaging two <i>samples</i> can blend two modes &rarr; the invalid in-between action generative models exist to avoid (BID shows ensembling degrades stochastic chunked policies).</li>
      <li>Even within "averaging is safe": our ACT data shows heavy ensembling lags and hurts (64%) while a
          short boundary-only crossfade helps (82%) &mdash; smooth the transition, not the whole plan.</li>
      <li>Caveat: our task is near-unimodal, so crossfade <i>might</i> also help DP here &mdash; a 3-min loop experiment.</li>
    </ul>
    """,
    # A3 ── mean vs mode
    """
    <h2>Appendix A3 &mdash; Why regression averages modes and diffusion doesn't</h2>
    <ul style="font-size:1.05em">
      <li><b>Regression:</b> L2/L1 training is minimized by the conditional mean/median. If demos contain
          "grasp left" and "grasp right" for the same observation, the optimal single output is
          <i>between them</i> &mdash; and the average of two valid actions is generally not a valid action.
          (ACT's CVAE latent z mitigates this in training, but z=0 at inference collapses to one blend.)</li>
      <li><b>Diffusion:</b> the denoising objective learns the <i>shape</i> of p(action | obs) (its score).
          Sampling starts from noise that lands by chance nearer one mode; each denoising step is conditioned
          on the current iterate, so it commits to that mode's basin &mdash; like a ball rolling into one of two
          valleys. Output: a clean sample from one strategy, mush from none.</li>
      <li><b>Why DP still lost our bake-off (46% vs 66%):</b> near-unimodal demos (its strength never engaged),
          fixed 20k budget penalizes slow-converging diffusion training, PushT-tuned defaults,
          and stochastic decoding noise at the precision handoff. Floor, not ceiling &mdash; dp_40k tests
          the budget hypothesis.</li>
    </ul>
    """,
]

slide_html = "\n".join(
    f'<section class="slide" id="s{i}">{s}<div class="num">{i + 1} / {len(slides)}</div></section>'
    for i, s in enumerate(slides)
)

html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>VLA AutoResearch</title>
<style>
  body {{ margin:0; font-family: 'Segoe UI', Helvetica, Arial, sans-serif; background:#111; }}
  .slide {{ display:none; box-sizing:border-box; width:100vw; height:100vh; padding:4vh 6vw;
           background:#1a1f2b; color:#eceff4; flex-direction:column; justify-content:center; }}
  .slide.active {{ display:flex; }}
  h1 {{ font-size:3.2em; margin:0.2em 0; color:#5dade2; }}
  h2 {{ font-size:1.9em; color:#85c1e9; margin-top:0; }}
  ul, ol {{ font-size:1.25em; line-height:1.55; }}
  p {{ font-size:1.15em; }}
  .big {{ font-size:1.4em; }}
  code {{ background:#2c3447; padding:0.1em 0.4em; border-radius:4px; }}
  table {{ border-collapse:collapse; font-size:1.1em; margin:0.6em 0; }}
  th, td {{ border:1px solid #3b4252; padding:0.4em 0.9em; text-align:left; }}
  th {{ background:#2c3447; }}
  img {{ align-self:center; border-radius:6px; background:#fff; }}
  .num {{ position:absolute; bottom:14px; right:22px; color:#666; font-size:0.8em; }}
</style></head>
<body>
{slide_html}
<script>
  let cur = 0;
  const slides = document.querySelectorAll('.slide');
  function show(i) {{
    cur = Math.max(0, Math.min(slides.length - 1, i));
    slides.forEach((s, j) => s.classList.toggle('active', j === cur));
  }}
  document.addEventListener('keydown', e => {{
    if (['ArrowRight',' ','PageDown'].includes(e.key)) show(cur + 1);
    if (['ArrowLeft','PageUp'].includes(e.key)) show(cur - 1);
  }});
  document.addEventListener('click', e => {{ if (e.target.tagName !== 'VIDEO') show(cur + 1); }});
  show(0);
</script>
</body></html>"""

Path("presentation.html").write_text(html)
print(f"saved presentation.html ({len(html) / 1e6:.1f} MB, {len(slides)} slides)")

"""Visually upgraded deck (presentation_v2.html). Content matches v1;
presentation.html is left untouched."""

import base64
from pathlib import Path


def b64(path, mime):
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"


comparison = b64("media/comparison.png", "image/png")
inference = b64("media/inference_methods.png", "image/png")
bakeoff = b64("media/bakeoff.png", "image/png")
arch = b64("media/architectures.png", "image/png")
vid_success = b64("eval_runs_80k/champ_200000/videos/aloha_0/eval_episode_5.mp4", "video/mp4")
vid_fail = b64("eval_runs_80k/champ_200000/videos/aloha_0/eval_episode_1.mp4", "video/mp4")

slides = [
    # 1 ── title
    """
    <div class="center">
      <div class="kicker">3-HOUR HACKATHON &middot; 2&times; RTX 5090 &middot; JUNE 2026</div>
      <h1 class="hero">VLA <span class="grad">AutoResearch</span></h1>
      <p class="sub">An autonomous research loop for robot policies,<br>scored by <b>closed-loop simulation</b> &mdash; not open-loop loss</p>
      <div class="statrow">
        <div class="stat"><div class="n">66&rarr;82<span class="pct">%</span></div><div class="l">task success, discovered<br>by the loop in 30 min</div></div>
        <div class="stat"><div class="n">25</div><div class="l">scored experiments<br>across 2 GPUs</div></div>
        <div class="stat"><div class="n">2.5<span class="pct">min</span></div><div class="l">per inference<br>experiment</div></div>
      </div>
      <p class="repo">github.com/yanz-rf/vla-autoresearch</p>
    </div>
    """,
    # 2 ── motivation
    """
    <h2>Motivation</h2>
    <div class="cards">
      <div class="card"><div class="ct">The inspiration</div>
        <b>karpathy/autoresearch</b> (Mar 2026): a coding agent runs LM-pretraining experiments
        autonomously &mdash; one idea per run, fixed budget, keep/revert by val_bpb, ~100 experiments overnight.</div>
      <div class="card"><div class="ct">The robotics gap</div>
        VLA / imitation policies are mostly iterated with <i>open-loop</i> metrics (action prediction error)
        &mdash; but low prediction error &ne; task success.</div>
      <div class="card hl"><div class="ct">Our port</div>
        The experiment score is <b>closed-loop sim success rate</b>: the policy controls the robot,
        mistakes compound, only finished tasks count. No public robotics port existed. Now one does.</div>
    </div>
    """,
    # 3 ── system design
    """
    <h2>System design <span class="dim">&mdash; same loop, new scoring</span></h2>
    <table>
      <tr><th></th><th>autoresearch (LM)</th><th>ours (robotics)</th></tr>
      <tr><td>mutable surface</td><td>train.py</td><td>policy code (LeRobot, editable) + CLI overrides</td></tr>
      <tr><td>read-only scorer</td><td>val bits-per-byte</td><td>50 sim episodes, fixed seeds &rarr; % success</td></tr>
      <tr><td>protocol</td><td>program.md</td><td>program.md, thresholds tuned for eval noise</td></tr>
      <tr><td>audit trail</td><td>results.tsv + git</td><td>results.tsv + git + rollout videos</td></tr>
    </table>
    <div class="callout"><b>Eval isolation:</b> scorer + seeds live outside the mutable surface &mdash; the agent can't game the metric.
    &nbsp;&nbsp;<b>Economics:</b> train+eval &asymp; 12 min &middot; inference-only &asymp; 2.5 min per experiment.</div>
    """,
    # 4 ── testbed
    """
    <h2>Testbed</h2>
    <ul class="roomy">
      <li><b>Task</b> &mdash; ALOHA TransferCube (MuJoCo): right arm picks a cube, hands it to the left arm.</li>
      <li><b>Data</b> &mdash; 50 public human teleop demos (<code>lerobot/aloha_sim_transfer_cube_human</code>).</li>
      <li><b>Policy</b> &mdash; ACT, 80M params: predicts 100-action chunks from camera + joint state
          (from scratch; ImageNet-initialized ResNet-18 encoder).</li>
      <li><b>Score</b> &mdash; % of 50 fixed-seed episodes completed; staged reward (0-4) as dense secondary signal.</li>
      <li><b>Hackathon move</b> &mdash; train once (20k steps &asymp; 10 min, 66%), freeze weights, research
          <i>inference-time</i> behavior at 2.5 min/experiment.</li>
    </ul>
    """,
    # 5 ── discoveries
    f"""
    <h2>What the loop found <span class="dim">&mdash; 11 experiments, ~30 min compute</span></h2>
    <img src="{comparison}" class="wide">
    <div class="trio">
      <div><b>(a)</b> Execute 50 of 100 predicted actions: 66&rarr;80%. Non-monotonic &mdash; replanning too often is <i>worse</i>.</div>
      <div><b>(b)</b> Novel: <b>crossfade chunk boundaries</b> &rarr; 82%, reward 215&rarr;231.</div>
      <div><b>(c)</b> Validated on held-out seeds and a 4&times;-longer-trained model.</div>
    </div>
    """,
    # 6 ── inference mechanisms
    f"""
    <h2>How the execution strategies differ</h2>
    <img src="{inference}" style="max-height:78vh" class="tall">
    """,
    # 7 ── videos
    f"""
    <h2>Closed-loop rollouts <span class="dim">&mdash; champion config, held-out seeds</span></h2>
    <div class="vids">
      <figure><video src="{vid_success}" controls loop muted autoplay></video>
        <figcaption><span class="ok">SUCCESS</span> grasp &rarr; lift &rarr; handoff &middot; reward 313</figcaption></figure>
      <figure><video src="{vid_fail}" controls loop muted></video>
        <figcaption><span class="bad">FAIL</span> the 1-in-10: fumbles before handoff &middot; reward 72</figcaption></figure>
    </div>
    <p class="foot">every experiment saves 10 rollout videos &mdash; failures are debuggable, not just countable</p>
    """,
    # 8 ── validation
    """
    <h2>Does it hold up? <span class="dim">&mdash; the slide most demos skip</span></h2>
    <table>
      <tr><th>config</th><th>dev seeds</th><th>held-out seeds</th></tr>
      <tr><td>ACT 20k, default</td><td>66%</td><td>74%</td></tr>
      <tr><td>ACT 20k, champion</td><td>82%</td><td>76%</td></tr>
      <tr><td>ACT 80k, default</td><td>68%</td><td>76%</td></tr>
      <tr class="win"><td>ACT 80k, champion</td><td>82%</td><td>88%</td></tr>
    </table>
    <ul>
      <li>4&times; more training barely moved default inference (~70&rarr;72%) &mdash; the <b>inference research was worth more</b> (+13).</li>
      <li>Dev gain (+16) shrank on held-out seeds at 20k: <b>50-episode evals carry ~7% std</b> &mdash;
          loops must validate on held-out seeds or they overfit the eval.</li>
    </ul>
    """,
    # 9 ── architectures
    f"""
    <h2>Architecture bake-off <span class="dim">&mdash; same data, same budget, same eval</span></h2>
    <img src="{arch}" class="wide" style="max-height:72vh">
    """,
    # 10 ── bakeoff results
    f"""
    <h2>Bake-off results</h2>
    <img src="{bakeoff}" class="wide" style="max-height:64vh">
    <p class="foot">DP/VQ-BeT ran library defaults tuned on simpler benchmarks (floor, not ceiling). Follow-ups the loop already chased:
    <b>DP 2&times; budget &rarr; 62%</b> (+16: it was underfit &mdash; ACT wins <i>per unit compute</i>, DP scales with budget) &middot; diagnosed VQ-BeT config fix running.</p>
    """,
    # 11 ── scatter
    """
    <h2>Scaling out <span class="dim">&mdash; 2nd GPU over Tailscale + a harder task</span></h2>
    <div class="cards">
      <div class="card"><div class="ct">Horizontal scaling</div>
        Second RTX 5090 bootstrapped from the public repo in ~10 min (clone &rarr; patches &rarr; uv install &rarr; run);
        results merge back as git rows.</div>
      <div class="card hl"><div class="ct">New benchmark row (remote)</div>
        ACT on <b>ALOHA Insertion</b> (peg-in-socket, both arms), same 20k budget: <b>18%</b> &mdash;
        independently reproduces the ACT paper's ~20%; confirms Insertion as the high-headroom target.</div>
      <div class="card"><div class="ct">Diagnosed, fixed, still negative (local)</div>
        VQ-BeT's 2% traced to an 84&times;84 crop (2.3% of the frame &mdash; nearly blind) and a two-phase schedule
        where the whole 20k budget trained only the codebook. Fix: reward 0.5&rarr;20 (now engages the task)
        but <b>0% success</b> &mdash; residual failure is structural (15-step horizon, 16-code vocabulary).
        A negative result with a sharpened hypothesis.</div>
    </div>
    """,
    # 12 ── learnings
    """
    <h2>Learnings</h2>
    <ol class="roomy">
      <li><b>Closed-loop eval is the right loop signal</b> &mdash; the wins came from execution-time behavior open-loop loss cannot see.</li>
      <li><b>At fixed budget, inference-time research beat architecture search</b> (+16 dev / +13 pooled vs &minus;20 for swapping to DP).</li>
      <li><b>Eval noise is the central methodological issue</b> &mdash; conservative keep/revert, dense reward tiebreaker, held-out seeds.</li>
      <li><b>Defaults don't transfer across embodiments</b> (VQ-BeT: 2% on bimanual with PushT-tuned config).</li>
      <li><b>Agent-as-orchestrator works for robotics</b>: every claim traceable to a git commit + results row + videos.</li>
    </ol>
    <p class="foot"><b>Appendix</b> &mdash; A1 open-loop vs closed-loop alignment data &middot; A2 smoothing &times; architecture &middot; A3 mean-vs-mode</p>
    """,
    # 13 ── next
    """
    <h2>What's next</h2>
    <ul class="roomy">
      <li><b>Harder target:</b> RoboTwin 2.0 &mdash; 50 bimanual ALOHA tasks, public leaderboard, baselines at 10-40% under domain randomization.</li>
      <li><b>Small VLAs in the loop:</b> SmolVLA (450M) fine-tunes on one 5090; pi0-LoRA fits in 32GB.</li>
      <li><b>Cross-architecture inference research:</b> does crossfade help every chunked policy?</li>
      <li><b>Overnight run:</b> ~100 train+eval experiments/night at the 20k budget.</li>
    </ul>
    <p class="repo" style="margin-top:1.2em">github.com/yanz-rf/vla-autoresearch</p>
    """,
    # A1
    """
    <h2><span class="appx">A1</span> Open-loop loss vs closed-loop success: poor alignment</h2>
    <table>
      <tr><th>train step</th><th>open-loop L1 loss</th><th>closed-loop success (dev / held-out)</th></tr>
      <tr><td>20k</td><td>~0.11</td><td>66% / 74%</td></tr>
      <tr><td>80k</td><td>0.052 <span class="dim">(2.3&times; better)</span></td><td>68% / 76% <span class="dim">(statistical tie)</span></td></tr>
    </table>
    <ul>
      <li><b>Evidence 1:</b> loss fell for 60k more steps; success didn't move &mdash; gradients improved frame imitation, not task completion.</li>
      <li><b>Evidence 2:</b> all inference configs share frozen weights &rarr; <i>identical</i> open-loop loss, yet closed-loop ranged <b>52&rarr;82%</b>. The dimension we mined is invisible to open-loop eval by construction.</li>
      <li><b>Why:</b> open-loop = "predict the human's action in states the human visited"; closed-loop = "recover in states your own mistakes created". Low loss is ~necessary, far from sufficient.</li>
    </ul>
    """,
    # A2
    """
    <h2><span class="appx">A2</span> Which inference smoothing fits which architecture</h2>
    <table>
      <tr><th>policy type</th><th>output is...</th><th>safe smoothing</th><th>examples</th></tr>
      <tr><td>regression (ACT)</td><td>point estimate (cond. mean)</td><td><b>averaging</b> in action space</td>
          <td>temporal ensembling, our crossfade</td></tr>
      <tr><td>generative (DP, &pi;0, flow VLAs)</td><td>one sample of a multimodal dist.</td>
          <td><b>conditioning or selection</b> &mdash; never sample-averaging</td>
          <td>warm-start / inpainting: RTC (2506.07339); selection: BID (2408.17355)</td></tr>
    </table>
    <ul>
      <li>Averaging two <i>samples</i> can blend two modes &rarr; the invalid in-between action generative models exist to avoid.</li>
      <li>Within "averaging is safe": heavy ensembling lags and hurt (64%); short boundary-only crossfade helped (82%).</li>
      <li>Caveat: our task is near-unimodal &mdash; crossfade <i>might</i> also help DP here; a 3-min loop experiment.</li>
    </ul>
    """,
    # A3
    """
    <h2><span class="appx">A3</span> Why regression averages modes and diffusion doesn't</h2>
    <ul class="roomy">
      <li><b>Regression:</b> L1/L2 training is minimized by the conditional mean/median. With "grasp left" and
          "grasp right" in the demos, the optimal single output is <i>between them</i> &mdash; and the average of two
          valid actions is generally not valid. (ACT's CVAE z mitigates in training; z=0 at inference collapses it.)</li>
      <li><b>Diffusion:</b> the denoising objective learns the <i>shape</i> of p(action|obs). Sampling starts from
          noise that lands nearer one mode; each step conditions on the current iterate and commits to that basin
          &mdash; a ball rolling into one of two valleys. One clean strategy out, mush from none.</li>
      <li><b>Why DP still lost (46% vs 66%):</b> near-unimodal demos, slow-converging training under a fixed budget,
          PushT-tuned defaults, decoding noise at the precision handoff. Budget hypothesis <b>confirmed</b>:
          2&times; steps &rarr; 62% (+16) &mdash; most of the gap was underfitting.</li>
    </ul>
    """,
]

slide_html = "\n".join(
    f'<section class="slide" id="s{i}">{s}</section>' for i, s in enumerate(slides)
)

html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>VLA AutoResearch</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family:'Avenir Next','Segoe UI',system-ui,Helvetica,Arial,sans-serif;
         background:#0b0e14; color:#e8edf4; }}
  .slide {{ display:none; width:100vw; height:100vh; padding:5vh 7vw;
           background:radial-gradient(1200px 700px at 80% -10%, #16263d 0%, #0b0e14 60%);
           flex-direction:column; justify-content:center; animation:fade .35s ease; }}
  .slide.active {{ display:flex; }}
  @keyframes fade {{ from {{ opacity:0; transform:translateY(8px); }} to {{ opacity:1; transform:none; }} }}

  h1.hero {{ font-size:4.6em; margin:0.1em 0; letter-spacing:-0.02em; }}
  .grad {{ background:linear-gradient(90deg,#4fc3f7,#9c6bff); -webkit-background-clip:text; background-clip:text; color:transparent; }}
  .kicker {{ letter-spacing:0.25em; font-size:0.85em; color:#7a8aa0; margin-bottom:1em; }}
  .sub {{ font-size:1.5em; color:#b9c6d6; line-height:1.5; }}
  .center {{ text-align:center; }}
  .statrow {{ display:flex; gap:2.5em; justify-content:center; margin:1.6em 0 1em; }}
  .stat .n {{ font-size:3em; font-weight:700; color:#4fc3f7; }}
  .stat .pct {{ font-size:0.5em; color:#7a8aa0; }}
  .stat .l {{ color:#8fa1b5; font-size:0.95em; line-height:1.35; margin-top:0.2em; }}
  .repo {{ font-family:ui-monospace,Consolas,monospace; color:#9c6bff; font-size:1.15em; }}

  h2 {{ font-size:2.1em; margin:0 0 0.7em; padding-left:0.55em; border-left:6px solid #4fc3f7; }}
  .dim {{ color:#7a8aa0; font-weight:400; font-size:0.72em; }}
  .appx {{ background:#9c6bff; color:#0b0e14; border-radius:6px; padding:0.05em 0.4em; font-size:0.8em; margin-right:0.35em; }}

  ul, ol {{ font-size:1.28em; line-height:1.6; }}
  ul.roomy li, ol.roomy li {{ margin-bottom:0.55em; }}
  p {{ font-size:1.15em; }}
  b {{ color:#fff; }}
  code {{ background:#1b2433; padding:0.12em 0.45em; border-radius:6px; font-size:0.9em; color:#8be9fd; }}

  table {{ border-collapse:separate; border-spacing:0; font-size:1.15em; margin:0.7em 0 1em;
          border-radius:12px; overflow:hidden; box-shadow:0 6px 24px rgba(0,0,0,.35); }}
  th, td {{ padding:0.55em 1.1em; text-align:left; }}
  th {{ background:linear-gradient(90deg,#1d3a5f,#232a3f); color:#bfe3ff; }}
  td {{ background:#141a26; border-top:1px solid #232c3d; }}
  tr.win td {{ background:#15303a; color:#aef3c1; font-weight:600; }}

  .cards {{ display:flex; gap:1.4em; }}
  .card {{ flex:1; background:#141a26; border:1px solid #232c3d; border-radius:14px;
          padding:1.2em 1.3em; font-size:1.12em; line-height:1.55; box-shadow:0 6px 24px rgba(0,0,0,.3); }}
  .card.hl {{ border-color:#4fc3f7; background:linear-gradient(180deg,#142433,#141a26); }}
  .ct {{ color:#4fc3f7; font-weight:700; margin-bottom:0.45em; letter-spacing:0.03em; }}

  .callout {{ background:#141f2e; border-left:5px solid #9c6bff; border-radius:0 10px 10px 0;
             padding:0.8em 1.1em; font-size:1.12em; }}
  .trio {{ display:flex; gap:1.5em; font-size:1.0em; color:#c6d2e0; margin-top:0.6em; }}
  .trio div {{ flex:1; }}

  img.wide {{ align-self:center; max-width:94%; border-radius:10px; background:#fff; padding:6px;
             box-shadow:0 10px 36px rgba(0,0,0,.45); }}
  img.tall {{ align-self:center; border-radius:10px; background:#fff; padding:6px; }}

  .vids {{ display:flex; gap:2.5em; justify-content:center; }}
  figure {{ margin:0; text-align:center; }}
  video {{ height:380px; border-radius:12px; box-shadow:0 10px 36px rgba(0,0,0,.5); }}
  figcaption {{ margin-top:0.6em; color:#b9c6d6; }}
  .ok {{ background:#1d4d33; color:#aef3c1; padding:0.1em 0.55em; border-radius:6px; font-weight:700; }}
  .bad {{ background:#4d2222; color:#f3b0ae; padding:0.1em 0.55em; border-radius:6px; font-weight:700; }}
  .foot {{ color:#8fa1b5; font-size:1.0em; }}

  .progress {{ position:fixed; bottom:0; left:0; height:4px; background:linear-gradient(90deg,#4fc3f7,#9c6bff);
              transition:width .25s ease; }}
  .pagenum {{ position:fixed; bottom:14px; right:22px; color:#5d6b7e; font-size:0.85em; }}
</style></head>
<body>
{slide_html}
<div class="progress" id="bar"></div>
<div class="pagenum" id="pg"></div>
<script>
  let cur = 0;
  const slides = document.querySelectorAll('.slide');
  function show(i) {{
    cur = Math.max(0, Math.min(slides.length - 1, i));
    slides.forEach((s, j) => s.classList.toggle('active', j === cur));
    document.getElementById('bar').style.width = ((cur + 1) / slides.length * 100) + '%';
    document.getElementById('pg').textContent = (cur + 1) + ' / ' + slides.length;
  }}
  document.addEventListener('keydown', e => {{
    if (['ArrowRight',' ','PageDown'].includes(e.key)) show(cur + 1);
    if (['ArrowLeft','PageUp'].includes(e.key)) show(cur - 1);
    if (e.key === 'Home') show(0);
  }});
  document.addEventListener('click', e => {{ if (e.target.tagName !== 'VIDEO') show(cur + 1); }});
  show(0);
</script>
</body></html>"""

Path("presentation_v2.html").write_text(html)
print(f"saved presentation_v2.html ({len(html) / 1e6:.1f} MB, {len(slides)} slides)")

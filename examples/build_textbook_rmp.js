// Assemble the interactive-textbook chapter on RMPflow: inline the wasm-bindgen glue + base64-embed the
// wasm, so the page runs the real ferromotion RmpArm on-device. The reader drags an obstacle into an
// arm's path and watches the whole arm bend around it while the hand still reaches the goal — reactive
// behavior composition, no planner. Same self-contained pattern as chapters 1–9.
const fs = require("fs");
const path = require("path");

const pkg = path.join(__dirname, "..", "crates", "ferromotion-wasm", "pkg");
const glue = fs.readFileSync(path.join(pkg, "ferromotion_wasm.js"), "utf8");
const wasmB64 = fs.readFileSync(path.join(pkg, "ferromotion_wasm_bg.wasm")).toString("base64");

const APP = String.raw`
function b64ToBytes(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
const fmt=(x,n=2)=>Number(x).toFixed(n);
const TAU=Math.PI*2, D0=0.5;

let lab, drag=null, interacted=false, sweepT=0; // 'goal' | 'obs' | null
const cv=document.getElementById("stage");
function tf(){ const r=cv.getBoundingClientRect(); const s=Math.min(r.width,r.height)/4.6; // workspace ~ radius 2
  return {r,s,ox:r.width*0.5, oy:r.height*0.52}; }
function P(x,y){ const t=tf(); return [t.ox+x*t.s, t.oy - y*t.s]; }
function inv(px,py){ const t=tf(); return [(px-t.ox)/t.s, -(py-t.oy)/t.s]; }

function draw(){
  const ctx=cv.getContext("2d"); const t=tf();
  const dpr=Math.min(devicePixelRatio||1,2); cv.width=t.r.width*dpr; cv.height=t.r.height*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,t.r.width,t.r.height);
  // reach limit
  const b=P(0,0); ctx.beginPath(); ctx.arc(b[0],b[1],2*t.s,0,TAU); ctx.strokeStyle="rgba(120,140,180,.1)"; ctx.lineWidth=1; ctx.setLineDash([3,5]); ctx.stroke(); ctx.setLineDash([]);
  const active=lab.avoidance_active();
  // obstacle + influence ring
  const oc=P(lab.obstacle_x(),lab.obstacle_y()), orr=lab.obstacle_radius()*t.s;
  ctx.beginPath(); ctx.arc(oc[0],oc[1],orr+D0*t.s,0,TAU); ctx.strokeStyle="rgba(232,131,111,.28)"; ctx.lineWidth=1.5; ctx.setLineDash([4,4]); ctx.stroke(); ctx.setLineDash([]);
  ctx.beginPath(); ctx.arc(oc[0],oc[1],orr,0,TAU); ctx.fillStyle= active?"rgba(232,131,111,.28)":"rgba(232,131,111,.16)"; ctx.fill(); ctx.strokeStyle="#e8836f"; ctx.lineWidth=2; ctx.stroke();
  ctx.fillStyle="rgba(232,131,111,.8)"; ctx.font="500 10px ui-monospace,monospace"; ctx.fillText("obstacle · drag", oc[0]-30, oc[1]+orr+16);
  // goal
  const g=P(lab.goal_x(),lab.goal_y());
  ctx.strokeStyle="#f0cf82"; ctx.lineWidth=2.5; ctx.beginPath(); ctx.moveTo(g[0]-8,g[1]); ctx.lineTo(g[0]+8,g[1]); ctx.moveTo(g[0],g[1]-8); ctx.lineTo(g[0],g[1]+8); ctx.stroke();
  ctx.beginPath(); ctx.arc(g[0],g[1],10,0,TAU); ctx.setLineDash([2,3]); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle="rgba(240,207,130,.85)"; ctx.fillText("goal · drag", g[0]+12, g[1]-10);
  // arm
  const j=Array.from(lab.joints_xy());
  const acol = active ? "#7dd3a0" : "#d9b45e";
  ctx.strokeStyle=acol; ctx.lineWidth=7; ctx.lineCap="round"; ctx.lineJoin="round";
  ctx.beginPath(); for(let i=0;i<j.length;i+=2){ const p=P(j[i],j[i+1]); i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]); } ctx.stroke();
  // control points (the watchers)
  const cps=Array.from(lab.control_points_xy());
  for(let i=0;i<cps.length;i+=2){ const p=P(cps[i],cps[i+1]); ctx.beginPath(); ctx.arc(p[0],p[1],3,0,TAU); ctx.fillStyle="rgba(174,182,204,.6)"; ctx.fill(); }
  // joints
  for(let i=0;i<j.length;i+=2){ const p=P(j[i],j[i+1]); ctx.beginPath(); ctx.arc(p[0],p[1],i===j.length-2?6:5,0,TAU); ctx.fillStyle="#161f3a"; ctx.strokeStyle=i===j.length-2?"#f0cf82":acol; ctx.lineWidth=2.5; ctx.fill(); ctx.stroke(); }
  ctx.fillStyle=active?"rgba(125,211,160,.9)":"rgba(217,180,94,.85)"; ctx.font="500 10px ui-monospace,monospace";
  ctx.fillText(active?"avoidance engaged — the whole arm bends":"reaching freely", 12, t.r.height-12);
  // readouts
  const cl=lab.min_clearance();
  document.getElementById("clear").textContent=fmt(cl,3);
  document.getElementById("clear").style.color = cl<0.05?"#e8836f":(active?"#f0cf82":"#7dd3a0");
  document.getElementById("gerr").textContent=fmt(lab.goal_error(),3);
  const st=document.getElementById("avoidState"); st.textContent=active?"engaged":"dormant"; st.style.color=active?"#f0cf82":"#7dd3a0";
}

function pick(px,py){ const g=P(lab.goal_x(),lab.goal_y()), o=P(lab.obstacle_x(),lab.obstacle_y());
  if(Math.hypot(px-g[0],py-g[1])<20) return 'goal';
  if(Math.hypot(px-o[0],py-o[1])<lab.obstacle_radius()*tf().s+12) return 'obs';
  return null; }
cv.addEventListener("pointerdown",e=>{ const r=cv.getBoundingClientRect(); drag=pick(e.clientX-r.left,e.clientY-r.top); if(drag){ interacted=true; cv.setPointerCapture(e.pointerId);} });
cv.addEventListener("pointermove",e=>{ if(!drag) return; const r=cv.getBoundingClientRect(); const w=inv(e.clientX-r.left,e.clientY-r.top);
  if(drag==='goal'){ const n=Math.hypot(w[0],w[1]); const c=n>1.9?[w[0]/n*1.9,w[1]/n*1.9]:w; lab.set_goal(c[0],c[1]); }
  else lab.set_obstacle(w[0],w[1]); });
cv.addEventListener("pointerup",()=>{ drag=null; });
document.getElementById("rad").oninput=e=>{ lab.set_obstacle_r(+e.target.value); document.getElementById("radVal").textContent=fmt(+e.target.value,2); };
document.getElementById("resetBtn").onclick=()=>{ lab.reset(0.9,0.6); lab.set_obstacle(1.5,0.0); lab.set_obstacle_r(0.4); document.getElementById("rad").value=0.4; document.getElementById("radVal").textContent="0.40"; interacted=false; sweepT=0; };

function frame(){
  if(!interacted){ sweepT+=0.010; const a=-0.15+Math.sin(sweepT)*1.15; lab.set_goal(1.45*Math.cos(a), 1.45*Math.sin(a)); } // arc sweeping past the obstacle
  for(let i=0;i<8;i++) lab.step(2e-3); draw(); requestAnimationFrame(frame); }

/* self-check on load: reaches the goal while never colliding, from the canonical setup */
function selfCheck(){
  const t=new RmpLab(); t.reset(0.9,0.6); t.set_goal(0.6,-1.4); t.set_obstacle(1.5,0.0);
  let worst=t.min_clearance();
  for(let i=0;i<6000;i++){ t.step(2e-3); worst=Math.min(worst,t.min_clearance()); }
  document.getElementById("scClear").textContent=fmt(worst,3);
  document.getElementById("scErr").textContent=fmt(t.goal_error(),3);
  document.getElementById("scVerdict").textContent = (worst>0 && t.goal_error()<0.1) ? "reached the goal, never touched the obstacle" : "unexpected";
}

async function main(){
  await __wbg_init(b64ToBytes(WASM_B64));
  lab=new RmpLab(); selfCheck(); lab.reset(0.9,0.6);
  window.__rmp={lab:()=>lab, setObstacle:(x,y)=>lab.set_obstacle(x,y), setGoal:(x,y)=>lab.set_goal(x,y), clearance:()=>lab.min_clearance(), goalErr:()=>lab.goal_error(), active:()=>lab.avoidance_active()};
  window.__textbook_ready=true;
  addEventListener("resize",draw); frame();
}
main();
`;

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Do everything at once — ferromotion textbook</title>
<meta name="description" content="An interactive chapter on RMPflow: reactive motion that fuses reaching a goal with avoiding an obstacle by a metric that weights each behavior by how much it matters. Runs the real Rust RmpArm on-device."/>
<style>
:root{--ground:#0a0f1e;--panel:#111830;--line:#26324c;--ink:#eef1f8;--soft:#aeb6cc;--dim:#727d99;--gold:#d9b45e;--goldb:#f0cf82;--green:#7dd3a0;--red:#e8836f;--mono:ui-monospace,"SF Mono",Menlo,monospace;--sans:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
*{box-sizing:border-box}html,body{margin:0}
body{background:radial-gradient(1200px 700px at 50% -10%,#12203f,var(--ground));color:var(--ink);font-family:var(--sans);line-height:1.65}
.wrap{max-width:768px;margin:0 auto;padding:clamp(24px,5vw,56px) 20px 72px}
.kicker{font-family:var(--mono);font-size:.66rem;letter-spacing:.26em;text-transform:uppercase;color:var(--gold);display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:0 0 14px;max-width:none}
.kicker>span{white-space:nowrap}
h1{font-size:clamp(1.8rem,4.4vw,2.7rem);margin:0 0 .3em;font-weight:800;letter-spacing:-.025em;line-height:1.12}
h2{font-size:1.16rem;margin:56px 0 4px;font-weight:700;letter-spacing:-.01em}
h2 .n{font-family:var(--mono);font-size:.7rem;color:var(--gold);letter-spacing:.2em;display:block;margin-bottom:7px;font-weight:500}
.lede{color:var(--soft);font-size:1.1rem;margin:0 0 8px;max-width:62ch}
p{max-width:64ch;color:#c9d1e4}
.fig{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;margin:22px 0}
canvas{display:block;touch-action:none}
#stage{width:100%;height:420px;cursor:grab;border-radius:10px;background:radial-gradient(700px 420px at 50% 48%,#0e1730,#0b1122)}
.read{font-family:var(--mono);font-size:.76rem;color:var(--soft);margin:10px 0 0;text-align:center}.read b{color:var(--ink)}.dim{color:var(--dim)}
.ctl{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:14px;justify-content:center}
button{background:linear-gradient(180deg,#d9b45e,#a9832f);color:#161200;border:0;border-radius:9px;padding:8px 16px;font:700 .82rem var(--sans);cursor:pointer}
button.ghost{background:transparent;color:var(--soft);border:1px solid var(--line)}
label{font-family:var(--mono);font-size:.72rem;color:var(--soft);display:flex;align-items:center;gap:8px}
input[type=range]{accent-color:var(--gold);width:120px}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}
.stat{background:#0d1428;border:1px solid var(--line);border-radius:10px;padding:9px 8px;text-align:center}
.stat .v{font-family:var(--mono);font-size:1rem;font-weight:700;color:var(--goldb)}
.stat .k{font-family:var(--mono);font-size:.58rem;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);margin-top:2px}
.callout{border-left:2px solid var(--gold);padding:2px 0 2px 18px;margin:26px 0;color:var(--soft)}.callout b{color:var(--goldb)}
.verdict{background:linear-gradient(180deg,rgba(125,211,160,.08),rgba(125,211,160,.02));border:1px solid rgba(125,211,160,.3);border-radius:14px;padding:22px 24px;margin:30px 0}
.verdict .big{font-size:2.1rem;font-weight:800;color:var(--green);letter-spacing:-.02em;line-height:1.1}
.verdict p{margin:.5em 0 0;color:var(--soft);max-width:58ch}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.75rem;margin:14px 0 0}
td{padding:7px 0;border-bottom:1px solid var(--line);color:var(--soft)}td:last-child{text-align:right;color:var(--ink);font-weight:600}
.note{color:var(--dim);font-family:var(--mono);font-size:.7rem;margin-top:56px;border-top:1px solid var(--line);padding-top:16px;line-height:1.7}
.note b{color:var(--gold)}.note a{color:var(--soft)}
.badge{font-family:var(--mono);font-size:.62rem;white-space:nowrap;color:var(--dim);border:1px solid var(--line);border-radius:999px;padding:4px 10px}
</style></head><body>
<div class="wrap">
  <p class="kicker"><span style="color:var(--goldb);font-size:1.05rem">&#934;</span><span>ferromotion · textbook · chapter 10</span>
    <span class="badge">rust → wasm · on-device</span></p>
  <h1>Do everything at once</h1>
  <p class="lede">A robot rarely has just one job. It has to reach the target <i>and</i> keep its whole body clear of things <i>and</i> move smoothly — all at the same time, and reacting to a world that moves. This chapter fuses those into one motion, geometrically, with the same Rust code the native tools use.</p>

  <h2><span class="n">01 — the problem</span>Many jobs, one motion</h2>
  <p>Give an arm a goal for its hand and it is easy to servo toward it. Add an obstacle and it gets awkward: now the elbow, the forearm, the whole body has to stay clear too, and the importance of that flips from irrelevant when the obstacle is far to overriding when it is close. Stitching such behaviors together with hand-tuned weights is brittle, and re-planning every time the world moves is slow. What you want is for the behaviors to combine themselves, correctly, moment to moment.</p>

  <h2><span class="n">02 — the policies</span>Each behavior brings its own metric</h2>
  <p>RMPflow makes each behavior a <b>Riemannian Motion Policy</b>: a desired acceleration paired with a <b>metric</b> — a state-dependent weight that says how much this behavior should matter right now. Reach-the-goal has a modest, steady metric. Avoid-the-obstacle has a metric that <b>blows up as the arm nears the surface</b>. The behaviors are fused by a metric-weighted combination pulled back onto the joints, so near the obstacle its policy dominates and far away it simply disappears — no mode switch, no planner.</p>
  <p>Below, a two-link arm reaches for the goal. <b>Drag the obstacle</b> into its path and watch the whole arm bend around it; <b>drag the goal</b> anywhere in reach.</p>
  <div class="fig">
    <canvas id="stage"></canvas>
    <div class="ctl">
      <button id="resetBtn" class="ghost">Reset</button>
      <label>obstacle size <input type="range" id="rad" min="0.2" max="0.8" step="0.02" value="0.4"/> <span id="radVal">0.40</span></label>
    </div>
    <div class="stats">
      <div class="stat"><div class="v" id="avoidState" style="font-size:.82rem">—</div><div class="k">avoidance</div></div>
      <div class="stat"><div class="v" id="clear">—</div><div class="k">clearance</div></div>
      <div class="stat"><div class="v" id="gerr">—</div><div class="k">hand → goal</div></div>
    </div>
    <p class="read dim" style="margin-top:8px">the dashed ring is the obstacle's influence; push it onto the arm and the whole body flows around while the hand keeps reaching — clearance never goes negative</p>
  </div>

  <h2><span class="n">03 — the metric decides</span>Close means loud, far means silent</h2>
  <p>The obstacle's dashed ring is where its metric starts to bite. Drag the obstacle outside everything and the arm reaches in a clean line — the avoidance policy is present but weighted to nothing, so it contributes nothing. Slide it onto the path and, as the nearest arm point crosses the ring, that policy's weight climbs steeply and the arm yields exactly as much as it must, no more.</p>
  <div class="callout">This is the whole idea, and why it composes. You never wrote a rule for "if near obstacle, stop reaching." Each behavior states what it wants and how much it cares, in its own coordinates; the geometry does the arbitration. Add a third behavior — a joint limit, a second obstacle, a preferred posture — and it drops in the same way, its metric deciding when it speaks. The response is <b>graded</b>, not a switch: a closer obstacle simply forces a tighter, later pass.</p>

  <h2><span class="n">04 — the check</span>Reaches, and never touches</h2>
  <p>On load, this page put the obstacle squarely on the straight path between the arm's start and its goal, and ran the policy to rest:</p>
  <table>
    <tr><td>smallest clearance to the obstacle over the run</td><td id="scClear">…</td></tr>
    <tr><td>final hand-to-goal error</td><td id="scErr">…</td></tr>
    <tr><td>verdict</td><td id="scVerdict">…</td></tr>
  </table>
  <p>The clearance stays positive the whole way — the metric's blow-up makes contact impossible — and the hand still lands on the goal. Both jobs, done at once, with nothing planned ahead.</p>

  <h2><span class="n">05 — the point</span>Composition without a planner</h2>
  <div class="verdict">
    <div class="big">State what you want, and how much you mean it.</div>
    <p>Every behavior carries its own metric, and the geometry fuses them into one reactive motion — reaching where reaching matters, yielding where safety matters — so new behaviors compose in without re-planning and without a tangle of hand-tuned weights.</p>
  </div>
  <p>This is why reactive geometric methods took hold for whole-body control: a real robot faces a moving, cluttered world where a plan is stale the moment it is made, and RMPflow answers it by making arbitration a property of the geometry rather than a script. It draws on the same kinematics — the real Jacobians of the arm — as the model-based controllers, but spends them on reacting rather than planning. It is the composition member of this series: where other chapters found the one quantity that governs a system, this one lets many behaviors each name their own, and trusts the metric to referee.</p>

  <p class="note"><b>What you just drove:</b> <span style="color:var(--soft)">RmpArm</span> from <span style="color:var(--soft)">ferromotion-control</span> on a two-link arm built from URDF through the real <span style="color:var(--soft)">ferromotion-core</span> kinematics, compiled to WebAssembly — the same code the native tools link against. Every frame pulls back the attractor and per-control-point obstacle policies to a joint acceleration and integrates it; the Jacobians are the arm's real ones. Nothing precomputed — the motion is solved live as you drag.<br/><br/>
  <b>Verified in the library:</b> the arm reaches the goal while every control point clears an obstacle on the direct path · a far obstacle does not perturb the reach at all · the pass tightens continuously as the obstacle nears the path · clearance never goes negative. Each is a test in <span style="color:var(--soft)">cargo test</span>, not a claim in prose. See also <a href="/assets/sims/morphological-computation">ch.1</a> · <a href="/assets/sims/algebraic-connectivity">ch.2</a> · <a href="/assets/sims/safety-filter">ch.3</a> · <a href="/assets/sims/force-closure">ch.4</a> · <a href="/assets/sims/movement-primitives">ch.5</a> · <a href="/assets/sims/invariant-estimation">ch.6</a> · <a href="/assets/sims/time-optimal">ch.7</a> · <a href="/assets/sims/capture-point">ch.8</a> · <a href="/assets/sims/koopman">ch.9</a>.<br/><br/>
  <b>Institute for Physical AI</b> · <a href="https://github.com/dcharlot-physicalai-bmi/ferromotion">the Rust library</a> · <a href="https://crates.io/crates/ferromotion">crates.io</a></p>
</div>
<script type="module">
${glue}
const WASM_B64="${wasmB64}";
${APP}
</script></body></html>`;

const outFile = path.join(__dirname, "..", "..", "v2", "public", "assets", "sims", "reactive-motion.html");
fs.writeFileSync(outFile, html);
console.log(`wrote ${outFile} (${(html.length / 1024).toFixed(0)} KB, wasm ${(wasmB64.length / 1024).toFixed(0)} KB b64)`);

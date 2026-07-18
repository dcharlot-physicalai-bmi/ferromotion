// Assemble the interactive-textbook chapter on Successive Convexification (rocket powered-descent). Inline
// the wasm-bindgen glue + base64-embed the wasm so the page runs the real ferromotion ScvxProblem
// on-device. The reader scrubs through the SCvx iterations and watches a dynamically-infeasible
// straight-line guess bend into a feasible, fuel-efficient landing, while the dynamics defect falls
// superlinearly on a log plot. Same self-contained pattern as the other chapters.
const fs = require("fs");
const path = require("path");

const pkg = path.join(__dirname, "..", "crates", "ferromotion-wasm", "pkg");
const glue = fs.readFileSync(path.join(pkg, "ferromotion_wasm.js"), "utf8");
const wasmB64 = fs.readFileSync(path.join(pkg, "ferromotion_wasm_bg.wasm")).toString("base64");

const APP = String.raw`
function b64ToBytes(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
const fmt=(x,n=2)=>Number(x).toFixed(n);
const TAU=Math.PI*2;
let lab, frame=0, playing=false, timer=null;

// ---- the landing scene ----
const cv=document.getElementById("stage");
const WX0=-3.6, WX1=0.9, WY0=-0.3, WY1=4.7;
function SX(x,w){ const pl=44,pr=20; return pl+(x-WX0)/(WX1-WX0)*(w-pl-pr); }
function SY(y,h){ const pt=18,pb=34; return h-pb-(y-WY0)/(WY1-WY0)*(h-pt-pb); }
function drawScene(){
  const r=cv.getBoundingClientRect(); const dpr=Math.min(devicePixelRatio||1,2);
  cv.width=r.width*dpr; cv.height=r.height*dpr; const ctx=cv.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
  const w=r.width,h=r.height; ctx.clearRect(0,0,w,h);
  const np=lab.waypoints(), last=lab.frames()-1;
  // ground + pad
  ctx.strokeStyle="rgba(120,140,180,.35)"; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(SX(WX0,w),SY(0,h)); ctx.lineTo(SX(WX1,w),SY(0,h)); ctx.stroke();
  ctx.fillStyle="#7dd3a0"; ctx.fillRect(SX(-0.18,w),SY(0,h)-2,SX(0.18,w)-SX(-0.18,w),4);
  ctx.fillStyle="rgba(125,211,160,.7)"; ctx.font="10px ui-monospace,monospace"; ctx.fillText("landing pad",SX(0.24,w),SY(0,h)+4);
  ctx.fillStyle="#727d99"; ctx.fillText("downrange →",SX(WX1,w)-78,SY(0,h)+20);
  ctx.save(); ctx.translate(SX(WX0,w)-6,SY(2.3,h)); ctx.rotate(-Math.PI/2); ctx.fillText("altitude",0,0); ctx.restore();
  // faint initial straight-line guess for reference
  ctx.strokeStyle="rgba(174,182,204,.28)"; ctx.setLineDash([5,4]); ctx.lineWidth=1.5;
  ctx.beginPath(); for(let i=0;i<np;i++){ const p=[SX(lab.px(0,i),w),SY(lab.py(0,i),h)]; i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]); } ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle="rgba(174,182,204,.55)"; ctx.fillText("initial guess (a straight line — not a real trajectory)",SX(-3.4,w),SY(3.9,h));
  // current-frame trajectory
  const feasible=lab.defect(frame)<1e-2;
  ctx.strokeStyle=feasible?"#7dd3a0":"#d9b45e"; ctx.lineWidth=2.6;
  ctx.beginPath(); for(let i=0;i<np;i++){ const p=[SX(lab.px(frame,i),w),SY(lab.py(frame,i),h)]; i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]); } ctx.stroke();
  for(let i=0;i<np;i++){ const p=[SX(lab.px(frame,i),w),SY(lab.py(frame,i),h)]; ctx.beginPath(); ctx.arc(p[0],p[1],2.1,0,TAU); ctx.fillStyle=feasible?"rgba(125,211,160,.9)":"rgba(240,207,130,.9)"; ctx.fill(); }
  // rocket at start
  const s=[SX(lab.start_x(),w),SY(lab.start_y(),h)]; ctx.beginPath(); ctx.arc(s[0],s[1],5,0,TAU); ctx.fillStyle="#f0cf82"; ctx.fill();
  // readouts
  document.getElementById("iterv").textContent=frame+" / "+last;
  document.getElementById("defv").textContent=lab.defect(frame).toExponential(2);
  document.getElementById("radv").textContent=fmt(lab.radius(frame),2);
  document.getElementById("feasv").textContent=feasible?"feasible":"infeasible";
  document.getElementById("feasv").style.color=feasible?"#7dd3a0":"#e8836f";
  drawDefect();
}

// ---- defect convergence (log scale) ----
const df=document.getElementById("defectfig");
function drawDefect(){
  const r=df.getBoundingClientRect(); const dpr=Math.min(devicePixelRatio||1,2);
  df.width=r.width*dpr; df.height=r.height*dpr; const ctx=df.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
  const w=r.width,h=r.height,pl=48,pr=14,pt=14,pb=26; ctx.clearRect(0,0,w,h);
  const nf=lab.frames(); const dmax=lab.initial_defect()*2, lo=-10, hi=Math.log10(dmax);
  const LX=k=>pl+(nf<2?0:k/(nf-1))*(w-pl-pr);
  const LY=d=>pt+(hi-Math.log10(Math.max(d,1e-10)))/(hi-lo)*(h-pt-pb);
  // gridlines at decades
  ctx.strokeStyle="rgba(120,140,180,.14)"; ctx.fillStyle="#5c6885"; ctx.font="9px ui-monospace,monospace"; ctx.lineWidth=1;
  for(let e=Math.ceil(lo);e<=Math.floor(hi);e+=2){ const y=LY(Math.pow(10,e)); ctx.beginPath(); ctx.moveTo(pl,y); ctx.lineTo(w-pr,y); ctx.stroke(); ctx.fillText("1e"+e,6,y+3); }
  ctx.fillStyle="#727d99"; ctx.fillText("dynamics defect vs SCvx iteration",pl+4,h-8);
  // curve
  ctx.strokeStyle="#d9b45e"; ctx.lineWidth=2; ctx.beginPath();
  for(let k=0;k<nf;k++){ const p=[LX(k),LY(lab.defect(k))]; k?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]); } ctx.stroke();
  for(let k=0;k<nf;k++){ ctx.beginPath(); ctx.arc(LX(k),LY(lab.defect(k)),2,0,TAU); ctx.fillStyle="rgba(240,207,130,.8)"; ctx.fill(); }
  // current frame marker
  ctx.strokeStyle="rgba(125,211,160,.8)"; ctx.setLineDash([3,3]); ctx.beginPath(); ctx.moveTo(LX(frame),pt); ctx.lineTo(LX(frame),h-pb); ctx.stroke(); ctx.setLineDash([]);
  ctx.beginPath(); ctx.arc(LX(frame),LY(lab.defect(frame)),4,0,TAU); ctx.fillStyle="#7dd3a0"; ctx.fill();
}

function setFrame(k){ frame=Math.max(0,Math.min(lab.frames()-1,k)); document.getElementById("iter").value=frame; drawScene(); }
function play(){ if(playing){ stop(); return; } playing=true; document.getElementById("play").textContent="Pause";
  if(frame>=lab.frames()-1) frame=0;
  timer=setInterval(()=>{ if(frame>=lab.frames()-1){ stop(); return; } setFrame(frame+1); },420); }
function stop(){ playing=false; if(timer) clearInterval(timer); timer=null; document.getElementById("play").textContent="Play the iterations"; }
document.getElementById("iter").oninput=e=>{ stop(); setFrame(+e.target.value); };
document.getElementById("play").onclick=play;
document.getElementById("reset").onclick=()=>{ stop(); setFrame(0); };

function selfCheck(){
  document.getElementById("scInit").textContent=fmt(lab.initial_defect(),2);
  document.getElementById("scFinal").textContent=lab.final_defect().toExponential(2);
  const last=lab.frames()-1, tip=lab.waypoints()-1;
  document.getElementById("scLand").textContent="("+fmt(lab.px(last,tip),3)+", "+fmt(lab.py(last,tip),3)+")";
  document.getElementById("scIters").textContent=(lab.frames()-1)+" iterations";
  document.getElementById("scVerdict").textContent = (lab.final_defect()<1e-3) ? "a feasible landing from an infeasible guess" : "unexpected";
}

async function main(){
  await __wbg_init(b64ToBytes(WASM_B64));
  lab=new ScvxLab(); selfCheck();
  document.getElementById("iter").max=lab.frames()-1;
  frame=lab.frames()-1; setFrame(frame); // show the converged landing first
  window.__scvx={lab:()=>lab, frames:()=>lab.frames(), initDefect:()=>lab.initial_defect(), finalDefect:()=>lab.final_defect(), landX:()=>lab.px(lab.frames()-1,lab.waypoints()-1), landY:()=>lab.py(lab.frames()-1,lab.waypoints()-1)};
  window.__textbook_ready=true;
  addEventListener("resize",drawScene);
}
main();
`;

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Landing a rocket — ferromotion textbook</title>
<meta name="description" content="An interactive chapter on successive convexification: a non-convex landing problem solved as a sequence of convex subproblems. Scrub the iterations and watch a straight-line guess bend into a feasible, fuel-efficient rocket landing while the dynamics defect falls superlinearly. Runs the real Rust SCvx solver on-device."/>
<style>
:root{--ground:#0a0f1e;--panel:#111830;--line:#26324c;--ink:#eef1f8;--soft:#aeb6cc;--dim:#727d99;--gold:#d9b45e;--goldb:#f0cf82;--green:#7dd3a0;--red:#e8836f;--mono:ui-monospace,"SF Mono",Menlo,monospace;--sans:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
*{box-sizing:border-box}html,body{margin:0}
body{background:radial-gradient(1200px 700px at 50% -10%,#12203f,var(--ground));color:var(--ink);font-family:var(--sans);line-height:1.65}
.wrap{max-width:768px;margin:0 auto;padding:clamp(24px,5vw,56px) 20px 72px}
.kicker{font-family:var(--mono);font-size:.66rem;letter-spacing:.26em;text-transform:uppercase;color:var(--gold);display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:0 0 14px}
.kicker>span{white-space:nowrap}
h1{font-size:clamp(1.8rem,4.4vw,2.7rem);margin:0 0 .3em;font-weight:800;letter-spacing:-.025em;line-height:1.12}
h2{font-size:1.16rem;margin:56px 0 4px;font-weight:700;letter-spacing:-.01em}
h2 .n{font-family:var(--mono);font-size:.7rem;color:var(--gold);letter-spacing:.2em;display:block;margin-bottom:7px;font-weight:500}
.lede{color:var(--soft);font-size:1.1rem;margin:0 0 8px;max-width:62ch}
p{max-width:64ch;color:#c9d1e4}
.fig{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;margin:22px 0}
canvas{display:block;touch-action:none}
#stage{width:100%;height:380px;border-radius:10px;background:radial-gradient(700px 380px at 50% 20%,#0e1730,#0b1122)}
#defectfig{width:100%;height:180px;border-radius:10px;background:radial-gradient(700px 180px at 50% 50%,#0e1730,#0b1122);margin-top:12px}
.read{font-family:var(--mono);font-size:.76rem;color:var(--soft);margin:10px 0 0;text-align:center}.read b{color:var(--ink)}.dim{color:var(--dim)}
.ctl{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:14px;justify-content:center}
button{background:linear-gradient(180deg,#d9b45e,#a9832f);color:#161200;border:0;border-radius:9px;padding:8px 16px;font:700 .82rem var(--sans);cursor:pointer}
button.ghost{background:transparent;color:var(--soft);border:1px solid var(--line)}
label{font-family:var(--mono);font-size:.72rem;color:var(--soft);display:flex;align-items:center;gap:8px}
input[type=range]{accent-color:var(--gold);width:200px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}
.stat{background:#0d1428;border:1px solid var(--line);border-radius:10px;padding:9px 8px;text-align:center}
.stat .v{font-family:var(--mono);font-size:.95rem;font-weight:700;color:var(--goldb)}
.stat .k{font-family:var(--mono);font-size:.56rem;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);margin-top:2px}
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
  <p class="kicker"><span style="color:var(--goldb);font-size:1.05rem">&#934;</span><span>ferromotion · textbook · chapter 15</span>
    <span class="badge">rust → wasm · on-device</span></p>
  <h1>Landing a rocket</h1>
  <p class="lede">A powered descent is a hard optimization: the dynamics are nonlinear — thrust divided by a mass that burns away as you fire — and the trajectory must end, exactly, on the pad at zero velocity. Successive convexification cracks it by solving a sequence of easy convex problems, each a better linear picture of the last. This page runs that solver, on your device, and lets you watch it converge.</p>

  <h2><span class="n">01 — the problem</span>A guess that isn't a trajectory</h2>
  <p>Start with the simplest possible plan: a straight line from where the vehicle is to the pad. It has the right endpoints and is completely, physically impossible — it ignores gravity, ignores that thrust acts through a shrinking mass, ignores that you cannot pull downward. It is not a trajectory at all; it is a shape with the correct ends. The job is to bend that shape until every step obeys the real dynamics, without ever leaving the pad constraint or the thrust limits.</p>

  <h2><span class="n">02 — convexify, repeatedly</span></h2>
  <p>The dynamics are non-convex, so we cannot optimize them directly. But we <i>can</i> linearize them about the current guess — and a linear model with a quadratic cost is a convex problem, which solves fast and exactly. The catch is that a linearization is only trustworthy near where it was taken. So SCvx does not solve once; it solves, re-linearizes about the new answer, and solves again — each convex subproblem a sharper local picture, the sequence marching toward a trajectory that satisfies the true nonlinear dynamics.</p>
  <p><b>Scrub the iterations below</b>, or play them. The dashed line is the impossible initial guess; the solid curve is where SCvx has bent it. Watch it turn from a straight line into a real descent.</p>
  <div class="fig">
    <canvas id="stage"></canvas>
    <canvas id="defectfig"></canvas>
    <div class="ctl">
      <button id="play">Play the iterations</button>
      <button id="reset" class="ghost">Back to the guess</button>
      <label>iteration <input type="range" id="iter" min="0" max="1" step="1" value="0"/></label>
    </div>
    <div class="stats">
      <div class="stat"><div class="v" id="iterv">—</div><div class="k">iteration</div></div>
      <div class="stat"><div class="v" id="defv">—</div><div class="k">dynamics defect</div></div>
      <div class="stat"><div class="v" id="radv">—</div><div class="k">trust radius</div></div>
      <div class="stat"><div class="v" id="feasv">—</div><div class="k">trajectory</div></div>
    </div>
    <p class="read dim" style="margin-top:8px">the defect is how badly the path violates the true dynamics; feasible when it reaches ~zero</p>
  </div>

  <h2><span class="n">03 — three safeguards</span>Why the sequence converges</h2>
  <p>Naively iterating linearizations diverges. SCvx adds three safeguards, and they are the whole reason it works:</p>
  <p><b>Virtual controls.</b> A slack term is added to the linearized dynamics and then punished by a steep penalty. It guarantees every subproblem has a solution — the method can never stall because a linearization momentarily has none — and at convergence the penalty has driven that slack to zero, which is exactly dynamic feasibility.</p>
  <p><b>A trust region.</b> Each step is bounded so the solution cannot wander past where the linearization is believable. Without it the convex model, trusted too far, would fly off to a meaningless "optimum."</p>
  <p><b>A ratio test.</b> After each solve, SCvx compares the <i>actual</i> reduction in nonlinear cost to the reduction the convex model <i>predicted</i>. Agree well? The step is good — accept it and enlarge the trust region. Disagree? The linearization was overtrusted — reject and shrink. That single adaptive number is what turns the loop from fragile into superlinearly convergent.</p>
  <div class="callout">Look at the defect plot as you scrub. For the first iterations it falls steadily; near the end it drops by orders of magnitude per step — the hallmark of <b>superlinear convergence</b>. Once the guess is close enough that the linearization is nearly exact, each convex solve almost lands it, and the error collapses.</div>

  <h2><span class="n">04 — the check</span>From impossible line to real landing</h2>
  <p>On load, this page ran SCvx from the straight-line guess to convergence:</p>
  <table>
    <tr><td>defect of the initial straight-line guess</td><td id="scInit">…</td></tr>
    <tr><td>defect after convergence</td><td id="scFinal">…</td></tr>
    <tr><td>touchdown point (pad is the origin)</td><td id="scLand">…</td></tr>
    <tr><td>iterations to converge</td><td id="scIters">…</td></tr>
    <tr><td>verdict</td><td id="scVerdict">…</td></tr>
  </table>
  <p>The defect falls from a wholly-infeasible guess to machine-negligible, and the final trajectory arrives at the pad at rest, inside the thrust limits, having burned finite fuel. Nothing about the landing was designed by hand — it was <i>found</i>, as the fixed point of a sequence of convex problems.</p>

  <h2><span class="n">05 — the point</span>Solve the hard problem as a sequence of easy ones</h2>
  <div class="verdict">
    <div class="big">Don't solve the non-convex problem — solve a sequence of convex ones.</div>
    <p>Linearize, bound the step, add slack you then punish away, and judge each move by whether reality agreed with the model. Repeat. A problem no convex solver can touch becomes a short list of problems every convex solver eats for breakfast — and the fixed point they converge to is your trajectory.</p>
  </div>
  <p>This is the guidance idea behind autonomous rocket landing, and the same machinery replans robot arms, drones, and legged gaits through non-convex constraints. It closes the book's planning arc: where the contact chapter smoothed a kink and bounded the step, this one linearizes the whole trajectory and bounds the step — the same instinct, believe the model only where it holds, applied to an entire flight.</p>

  <p class="note"><b>What you just drove:</b> the <span style="color:var(--soft)">ScvxProblem</span> solver from <span style="color:var(--soft)">ferromotion-control</span>, compiled to WebAssembly — the same code the native tools link against. Every iteration's trajectory, defect, and trust radius was computed live from a straight-line guess; the convex subproblems are QPs solved on-device.<br/><br/>
  <b>Verified in the library:</b> the analytic dynamics Jacobian matches finite differences; a linear system reaches zero defect in one solve (the linearization is exact); from an infeasible guess the rocket lands with defect → ~1e-9 (superlinear); the virtual controls vanish at convergence; and the defect falls &gt;100× while the trust radius adapts. Each is a test in <span style="color:var(--soft)">cargo test</span>, not a claim in prose. See also <a href="/assets/sims/contact-planning">ch.14 — planning through contact</a> · <a href="/assets/sims/textbook">the full textbook</a>.<br/><br/>
  <b>Institute for Physical AI</b> · <a href="https://github.com/dcharlot-physicalai-bmi/ferromotion">the Rust library</a> · <a href="https://crates.io/crates/ferromotion">crates.io</a></p>
</div>
<script type="module">
${glue}
const WASM_B64="${wasmB64}";
${APP}
</script></body></html>`;

const outFile = path.join(__dirname, "..", "..", "v2", "public", "assets", "sims", "rocket-landing.html");
fs.writeFileSync(outFile, html);
console.log(`wrote ${outFile} (${(html.length / 1024).toFixed(0)} KB, wasm ${(wasmB64.length / 1024).toFixed(0)} KB b64)`);

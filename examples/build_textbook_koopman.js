// Assemble the interactive-textbook chapter on the Koopman operator: inline the wasm-bindgen glue +
// base64-embed the wasm, so the page runs the real ferromotion edmd/Koopman on-device. The reader
// toggles a single observable (x1²) and watches a linear model's prediction of a nonlinear system snap
// from drifting to exact. Same self-contained pattern as chapters 1–8.
const fs = require("fs");
const path = require("path");

const pkg = path.join(__dirname, "..", "crates", "ferromotion-wasm", "pkg");
const glue = fs.readFileSync(path.join(pkg, "ferromotion_wasm.js"), "utf8");
const wasmB64 = fs.readFileSync(path.join(pkg, "ferromotion_wasm_bg.wasm")).toString("base64");

const APP = String.raw`
function b64ToBytes(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
const fmt=(x,n=2)=>Number(x).toFixed(n);
const TAU=Math.PI*2;
const STEPS=60;

let lab, start=[1.3,-0.9], lifted=true, dragging=false, anim=0;
let truth=[], pred=[], manifold=[];

function recompute(){
  truth=Array.from(lab.true_traj(start[0],start[1],STEPS));
  pred=Array.from(lifted ? lab.koopman_traj(start[0],start[1],STEPS) : lab.naive_traj(start[0],start[1],STEPS));
  manifold=Array.from(lab.manifold(-2.0,2.0,80));
  syncReadouts();
}

const cv=document.getElementById("stage");
function tf(){ const r=cv.getBoundingClientRect(); const pad=34;
  const xlo=-2,xhi=2, ylo=-1.6,yhi=2.6;
  return {r,pad,xlo,xhi,ylo,yhi,
    sx:(r.width-2*pad)/(xhi-xlo), sy:(r.height-2*pad)/(yhi-ylo)}; }
function P(x1,x2){ const t=tf(); return [t.pad+(x1-t.xlo)*t.sx, t.r.height-t.pad-(x2-t.ylo)*t.sy]; }
function inv(px,py){ const t=tf(); return [t.xlo+(px-t.pad)/t.sx, t.ylo+(t.r.height-t.pad-py)/t.sy]; }

function draw(){
  const ctx=cv.getContext("2d"); const t=tf();
  const dpr=Math.min(devicePixelRatio||1,2); cv.width=t.r.width*dpr; cv.height=t.r.height*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,t.r.width,t.r.height);
  // axes
  ctx.strokeStyle="rgba(120,140,180,.18)"; ctx.lineWidth=1;
  const o=P(0,0); ctx.beginPath(); ctx.moveTo(t.pad,o[1]); ctx.lineTo(t.r.width-t.pad,o[1]); ctx.moveTo(o[0],t.pad); ctx.lineTo(o[0],t.r.height-t.pad); ctx.stroke();
  ctx.fillStyle="#5b6680"; ctx.font="500 10px ui-monospace,monospace"; ctx.fillText("x₁",t.r.width-t.pad+2,o[1]+4); ctx.fillText("x₂",o[0]+5,t.pad+2);
  // slow manifold (the parabola the nonlinearity lives on)
  ctx.beginPath(); for(let i=0;i<manifold.length;i+=2){ const p=P(manifold[i],manifold[i+1]); i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]); }
  ctx.strokeStyle="rgba(120,140,180,.3)"; ctx.lineWidth=1.5; ctx.setLineDash([3,4]); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle="rgba(138,160,200,.6)"; ctx.fillText("slow manifold  x₂ = b·x₁²", P(0.15,2.35)[0], P(0.15,2.35)[1]);
  const nT=truth.length/2, k=Math.min(nT-1, Math.floor(anim));
  // true trajectory (gold)
  const line=(arr,col,w,dash,upto)=>{ ctx.beginPath(); const m=upto??(arr.length/2-1);
    for(let i=0;i<=m;i++){ const p=P(arr[2*i],arr[2*i+1]); i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]); }
    ctx.strokeStyle=col; ctx.lineWidth=w; ctx.setLineDash(dash||[]); ctx.stroke(); ctx.setLineDash([]); };
  line(truth,"rgba(217,180,94,.85)",3.5,null,k);
  // prediction (green if lifted/exact, red if naive/drifting)
  const pcol = lifted ? "#7dd3a0" : "#e8836f";
  line(pred,pcol,2,[5,4],k);
  // moving heads
  const th=P(truth[2*k],truth[2*k+1]), ph=P(pred[2*k]||0,pred[2*k+1]||0);
  ctx.beginPath(); ctx.arc(th[0],th[1],6,0,TAU); ctx.fillStyle="#161f3a"; ctx.strokeStyle="#d9b45e"; ctx.lineWidth=2.5; ctx.fill(); ctx.stroke();
  if(Math.abs(pred[2*k])<6 && Math.abs(pred[2*k+1])<6){ ctx.beginPath(); ctx.arc(ph[0],ph[1],5,0,TAU); ctx.fillStyle=pcol; ctx.fill(); }
  // start marker (draggable)
  const s0=P(start[0],start[1]); ctx.beginPath(); ctx.arc(s0[0],s0[1],8,0,TAU); ctx.strokeStyle="#f0cf82"; ctx.lineWidth=2; ctx.setLineDash([2,2]); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle="rgba(240,207,130,.8)"; ctx.fillText("drag start", s0[0]+11, s0[1]-7);
  // legend
  ctx.fillStyle="rgba(217,180,94,.9)"; ctx.fillText("— true nonlinear trajectory", 12, t.r.height-24);
  ctx.fillStyle=pcol; ctx.fillText(lifted?"— linear model + x₁²  (exact)":"— linear model, raw state  (drifts)", 12, t.r.height-10);
}

function syncReadouts(){
  const ke=lab.koopman_peak_error(start[0],start[1],STEPS), ne=lab.naive_peak_error(start[0],start[1],STEPS);
  document.getElementById("predErr").textContent=(lifted?ke:ne).toExponential(1);
  document.getElementById("predErr").style.color=lifted?"#7dd3a0":"#e8836f";
  document.getElementById("opErr").textContent=lab.operator_error().toExponential(1);
  const st=document.getElementById("modelState");
  st.textContent = lifted ? "linear & exact" : "linear & wrong"; st.style.color = lifted ? "#7dd3a0" : "#e8836f";
}

cv.addEventListener("pointerdown",e=>{ const r=cv.getBoundingClientRect(); const s0=P(start[0],start[1]);
  if(Math.hypot(e.clientX-r.left-s0[0], e.clientY-r.top-s0[1])<20){ dragging=true; cv.setPointerCapture(e.pointerId); } });
cv.addEventListener("pointermove",e=>{ if(!dragging) return; const r=cv.getBoundingClientRect();
  const w=inv(e.clientX-r.left, e.clientY-r.top); start=[Math.max(-1.9,Math.min(1.9,w[0])), Math.max(-1.5,Math.min(2.5,w[1]))]; anim=0; recompute(); });
cv.addEventListener("pointerup",()=>{ dragging=false; });
document.getElementById("liftToggle").onchange=e=>{ lifted=e.target.checked; anim=0; recompute(); };

/* self-check on load: operator recovered exactly + lifted predicts, naive drifts */
function selfCheck(){
  const t=new KoopmanLab();
  document.getElementById("scOp").textContent=t.operator_error().toExponential(1);
  document.getElementById("scKoop").textContent=t.koopman_peak_error(1.3,-0.9,60).toExponential(1);
  document.getElementById("scNaive").textContent=fmt(t.naive_peak_error(1.3,-0.9,60),2);
  const ok=t.operator_error()<1e-9 && t.koopman_peak_error(1.3,-0.9,60)<1e-7;
  document.getElementById("scVerdict").textContent = ok ? "the lifted operator is the exact Koopman operator" : "unexpected";
}

function frame(){ anim+=0.5; if(anim>truth.length/2-1+18) anim=0; draw(); requestAnimationFrame(frame); }

async function main(){
  await __wbg_init(b64ToBytes(WASM_B64));
  lab=new KoopmanLab(); selfCheck(); recompute();
  window.__koop={lab:()=>lab, setStart:(a,b)=>{start=[a,b];anim=0;recompute();}, setLifted:(v)=>{lifted=v;document.getElementById('liftToggle').checked=v;anim=0;recompute();}, koopErr:()=>lab.koopman_peak_error(start[0],start[1],STEPS), naiveErr:()=>lab.naive_peak_error(start[0],start[1],STEPS), opErr:()=>lab.operator_error()};
  window.__textbook_ready=true;
  addEventListener("resize",draw); frame();
}
main();
`;

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Make it linear — ferromotion textbook</title>
<meta name="description" content="An interactive chapter on the Koopman operator: a nonlinear system becomes exactly linear in the right lifted coordinates, so a linear model predicts it perfectly. Runs the real Rust EDMD on-device."/>
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
#stage{width:100%;height:400px;cursor:crosshair;border-radius:10px;background:radial-gradient(700px 400px at 50% 45%,#0e1730,#0b1122)}
.read{font-family:var(--mono);font-size:.76rem;color:var(--soft);margin:10px 0 0;text-align:center}.read b{color:var(--ink)}.dim{color:var(--dim)}
.ctl{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-top:14px;justify-content:center}
label{font-family:var(--mono);font-size:.78rem;color:var(--soft);display:flex;align-items:center;gap:9px;cursor:pointer}
input[type=checkbox]{accent-color:var(--gold);width:17px;height:17px}
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
  <p class="kicker"><span style="color:var(--goldb);font-size:1.05rem">&#934;</span><span>ferromotion · textbook · chapter 9</span>
    <span class="badge">rust → wasm · on-device</span></p>
  <h1>Make it linear</h1>
  <p class="lede">Linear systems are the ones we can actually predict and control — but almost nothing is linear. This chapter is about a trick that gets the good behaviour anyway: look at a nonlinear system through the right variables, and it moves in a straight line after all. It runs the same Rust operator-learning code the native tools use.</p>

  <h2><span class="n">01 — the problem</span>Linear is the easy case, and rare</h2>
  <p>If a system is linear, you own it: you can predict it arbitrarily far ahead by multiplying by a matrix, and a century of control theory applies directly. Real dynamics — a pendulum, a fluid, a leg — are nonlinear, and the standard response is to linearize around a point and accept that the model is only good nearby. But there is another option that gives up nothing.</p>

  <h2><span class="n">02 — the lift</span>The nonlinearity was in your coordinates</h2>
  <p>Koopman's insight: instead of tracking the state <span style="font-family:var(--mono);color:var(--goldb)">x</span>, track some <i>observables</i> of it — functions like <span style="font-family:var(--mono);color:var(--goldb)">x²</span>. In that lifted space the dynamics can be <b>exactly linear</b>, <span style="font-family:var(--mono);color:var(--goldb)">ψ(x_{k+1}) = A ψ(x_k)</span>, even though the system is not. The curvature you were fighting was an artifact of watching only <span style="font-family:var(--mono);color:var(--goldb)">x</span>.</p>
  <p>Below is a genuinely nonlinear system in its state plane. The gold curve is the truth. The other line is a <i>linear</i> model's prediction. Toggle the one extra observable <span style="font-family:var(--mono);color:var(--goldb)">x₁²</span> and watch what the linear model can do.</p>
  <div class="fig">
    <canvas id="stage"></canvas>
    <div class="ctl">
      <label><input type="checkbox" id="liftToggle" checked/> include the x₁² observable</label>
    </div>
    <div class="stats">
      <div class="stat"><div class="v" id="modelState" style="font-size:.82rem">—</div><div class="k">the linear model</div></div>
      <div class="stat"><div class="v" id="predErr">—</div><div class="k">prediction error</div></div>
      <div class="stat"><div class="v" id="opErr">—</div><div class="k">operator recovery</div></div>
    </div>
    <p class="read dim" style="margin-top:8px">with x₁² the linear prediction lies exactly on the true curve; without it, the same linear machinery drifts off — <b>drag the start</b> anywhere and it holds</p>
  </div>

  <h2><span class="n">03 — exact, not approximate</span>A linear model that never drifts</h2>
  <p>This is not a linearization that is good near a point and bad elsewhere. With the right observables the linear operator is <i>exact</i>, so its prediction tracks the true nonlinear trajectory to machine precision no matter how far you roll it out or where you start. And the operator itself is learned purely from <b>data</b> — snapshot pairs of the lifted state — by least squares (Extended Dynamic Mode Decomposition). On load, this page fit it from data and checked it against the analytic answer:</p>
  <table>
    <tr><td>learned operator vs the exact Koopman operator</td><td id="scOp">…</td></tr>
    <tr><td>lifted model — peak prediction error over 60 steps</td><td id="scKoop">…</td></tr>
    <tr><td>naive linear model (no x₁²) — peak error, same run</td><td id="scNaive">…</td></tr>
    <tr><td>verdict</td><td id="scVerdict">…</td></tr>
  </table>

  <h2><span class="n">04 — where it went</span>One well-chosen observable</h2>
  <div class="callout">The whole difference is the single function <b>x₁²</b>. The <span style="font-family:var(--mono);color:var(--goldb)">x₁</span> channel was linear all along, so the naive model nails it; all the error lives in <span style="font-family:var(--mono);color:var(--goldb)">x₂</span>, which couples to <span style="font-family:var(--mono);color:var(--goldb)">x₁²</span>. Add that one observable to your dictionary and the coupling becomes just another linear coordinate. Choosing the dictionary is the real work — but when a finite one closes, as here, the payoff is a globally exact linear model of a nonlinear system, fit from data.</p>

  <h2><span class="n">05 — the point</span>Borrow the linear toolbox</h2>
  <div class="verdict">
    <div class="big">Change coordinates, keep the theory.</div>
    <p>Lift a nonlinear system into observables where it moves linearly, and every linear method — long-horizon prediction, optimal control, spectral analysis — applies to it unchanged, learned from measurements rather than derived from a model.</p>
  </div>
  <p>This is why Koopman methods spread so fast across robotics and fluids: they are a bridge from messy real dynamics, which you can measure but not cleanly write down, to the linear tools that actually work. Fit the operator from data, and a soft robot or a gust-buffeted drone gets a predictor you can drop an optimal controller straight onto. It is the data-driven member of this series — where the other chapters found the one quantity that governs a system, this one finds the coordinates in which the system is simple, and then the simplicity is exact.</p>

  <p class="note"><b>What you just drove:</b> <span style="color:var(--soft)">edmd</span> and <span style="color:var(--soft)">Koopman</span> from <span style="color:var(--soft)">ferromotion-control</span>, compiled to WebAssembly — the same code the native tools link against. The operator is fit by least squares from lifted snapshot pairs of Brunton's slow-manifold system, which has an exact finite Koopman invariant subspace; predictions roll the learned linear operator forward. Nothing precomputed — every drag re-fits nothing but re-rolls live.<br/><br/>
  <b>Verified in the library:</b> EDMD recovers the exact Koopman operator (≈1e-9) · the lifted model predicts the nonlinear trajectory to machine precision while the naive linear model drifts &gt;50× more · the drift lives entirely in the x₂ channel that couples to the missing x₁². Each is a test in <span style="color:var(--soft)">cargo test</span>, not a claim in prose. See also <a href="/assets/sims/morphological-computation">ch.1</a> · <a href="/assets/sims/algebraic-connectivity">ch.2</a> · <a href="/assets/sims/safety-filter">ch.3</a> · <a href="/assets/sims/force-closure">ch.4</a> · <a href="/assets/sims/movement-primitives">ch.5</a> · <a href="/assets/sims/invariant-estimation">ch.6</a> · <a href="/assets/sims/time-optimal">ch.7</a> · <a href="/assets/sims/capture-point">ch.8</a>.<br/><br/>
  <b>Institute for Physical AI</b> · <a href="https://github.com/dcharlot-physicalai-bmi/ferromotion">the Rust library</a> · <a href="https://crates.io/crates/ferromotion">crates.io</a></p>
</div>
<script type="module">
${glue}
const WASM_B64="${wasmB64}";
${APP}
</script></body></html>`;

const outFile = path.join(__dirname, "..", "..", "v2", "public", "assets", "sims", "koopman.html");
fs.writeFileSync(outFile, html);
console.log(`wrote ${outFile} (${(html.length / 1024).toFixed(0)} KB, wasm ${(wasmB64.length / 1024).toFixed(0)} KB b64)`);

// Assemble the interactive-textbook chapter on invariant estimation: inline the wasm-bindgen glue +
// base64-embed the wasm, so the page runs the real ferromotion InEKF machinery on-device. The reader
// scales up a robot's initial estimate error and watches the InEKF's linear error model stay EXACT
// while a standard EKF's drifts. Same self-contained pattern as chapters 1–5.
const fs = require("fs");
const path = require("path");

const pkg = path.join(__dirname, "..", "crates", "ferromotion-wasm", "pkg");
const glue = fs.readFileSync(path.join(pkg, "ferromotion_wasm.js"), "utf8");
const wasmB64 = fs.readFileSync(path.join(pkg, "ferromotion_wasm_bg.wasm")).toString("base64");

const APP = String.raw`
function b64ToBytes(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
const fmt=(x,n=2)=>Number(x).toFixed(n);

let lab, scale=3.0;
let T=[], inek=[], ekf=[], tx=[], ex=[];
function run(){
  lab.simulate(scale);
  T=Array.from(lab.t()); inek=Array.from(lab.inekf_residual()); ekf=Array.from(lab.ekf_residual());
  tx=Array.from(lab.true_xy()); ex=Array.from(lab.est_xy());
  syncReadouts();
}

/* --- residual plot: true-error-model error over time, InEKF vs EKF --- */
const pc=document.getElementById("plot");
function drawPlot(){
  const ctx=pc.getContext("2d"); const r=pc.getBoundingClientRect();
  const dpr=Math.min(devicePixelRatio||1,2); pc.width=r.width*dpr; pc.height=r.height*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  const W=r.width,H=r.height,pad={l:52,r:16,t:16,b:28};
  ctx.clearRect(0,0,W,H);
  const tmax=T.length?T[T.length-1]:1;
  const ymax=Math.max(0.5, ekf.reduce((a,b)=>Math.max(a,b),0)*1.1);
  const x2p=t=>pad.l+(t/tmax)*(W-pad.l-pad.r);
  const y2p=v=>H-pad.b-(v/ymax)*(H-pad.t-pad.b);
  // grid + y labels
  ctx.font="500 10px ui-monospace,monospace";
  for(let i=0;i<=4;i++){ const v=ymax*i/4; const y=y2p(v);
    ctx.strokeStyle="rgba(120,140,180,.12)"; ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();
    ctx.fillStyle="#5b6680"; ctx.fillText(v.toFixed(1),8,y+3); }
  ctx.fillStyle="#727d99"; ctx.fillText("time →", W-pad.r-42, H-8);
  ctx.save(); ctx.translate(12,pad.t+6); ctx.rotate(-Math.PI/2);
  ctx.fillText("‖ error − model ‖", -70, 0); ctx.restore();
  // EKF residual (red, climbs)
  const line=(arr,col,w,dash)=>{ ctx.beginPath(); arr.forEach((v,i)=>{const X=x2p(T[i]),Y=y2p(v); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});
    ctx.strokeStyle=col; ctx.lineWidth=w; ctx.setLineDash(dash||[]); ctx.stroke(); ctx.setLineDash([]); };
  line(ekf,"#e8836f",2.5);
  // InEKF residual (green, pinned to zero)
  line(inek,"#7dd3a0",2.5);
  // legend
  ctx.fillStyle="#7dd3a0"; ctx.fillText("InEKF model — exact", pad.l+8, pad.t+12);
  ctx.fillStyle="#e8836f"; ctx.fillText("standard EKF model — drifts", pad.l+8, pad.t+28);
}

/* --- trajectory inset: true vs estimated path diverging --- */
const tc=document.getElementById("traj");
function drawTraj(){
  const ctx=tc.getContext("2d"); const r=tc.getBoundingClientRect();
  const dpr=Math.min(devicePixelRatio||1,2); tc.width=r.width*dpr; tc.height=r.height*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  const W=r.width,H=r.height; ctx.clearRect(0,0,W,H);
  const all=tx.concat(ex); let minx=1e9,maxx=-1e9,miny=1e9,maxy=-1e9;
  for(let i=0;i<all.length;i+=2){ minx=Math.min(minx,all[i]);maxx=Math.max(maxx,all[i]);miny=Math.min(miny,all[i+1]);maxy=Math.max(maxy,all[i+1]); }
  const sx=(W-40)/Math.max(1e-6,maxx-minx), sy=(H-40)/Math.max(1e-6,maxy-miny), s=Math.min(sx,sy);
  const ox=(W-(maxx-minx)*s)/2 - minx*s, oy=(H+(maxy-miny)*s)/2 + miny*s;
  const P=(x,y)=>[ox+x*s, oy-y*s];
  const path=(arr,col,w)=>{ ctx.beginPath(); for(let i=0;i<arr.length;i+=2){const p=P(arr[i],arr[i+1]); i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]);}
    ctx.strokeStyle=col; ctx.lineWidth=w; ctx.stroke(); };
  path(tx,"rgba(217,180,94,.85)",2.5);
  path(ex,"rgba(232,131,111,.85)",2.5);
  // start dots
  const dot=(arr,col)=>{ if(arr.length<2) return; const p=P(arr[0],arr[1]); ctx.beginPath();ctx.arc(p[0],p[1],4,0,7);ctx.fillStyle=col;ctx.fill(); };
  dot(tx,"#f0cf82"); dot(ex,"#e8836f");
  ctx.font="500 10px ui-monospace,monospace";
  ctx.fillStyle="rgba(217,180,94,.9)"; ctx.fillText("— true path", 10, H-22);
  ctx.fillStyle="#e8836f"; ctx.fillText("— the robot's guess", 10, H-9);
}

function syncReadouts(){
  document.getElementById("inekfPk").textContent=lab.inekf_peak().toExponential(1);
  document.getElementById("ekfPk").textContent=fmt(lab.ekf_peak(),2);
  const en=Array.from(lab.err_norm());
  document.getElementById("errMag").textContent=fmt(en[en.length-1],2);
}

document.getElementById("scale").oninput=e=>{ scale=+e.target.value; document.getElementById("scaleVal").textContent=fmt(scale,1)+"×"; run(); drawPlot(); drawTraj(); };

/* self-check on load: InEKF exact across a huge sweep of error sizes */
function selfCheck(){
  const t=new InekfLab(); let worst=0, big=0;
  for(const s of [0.5,2,5,10]){ t.simulate(s); worst=Math.max(worst,t.inekf_peak()); big=Math.max(big,t.ekf_peak()); }
  document.getElementById("scInekf").textContent=worst.toExponential(1);
  document.getElementById("scEkf").textContent=fmt(big,1);
  document.getElementById("scVerdict").textContent = worst<1e-4 ? "the invariant model is exact at every error size" : "unexpected";
}

async function main(){
  await __wbg_init(b64ToBytes(WASM_B64));
  lab=new InekfLab(); selfCheck(); run();
  window.__inekf={lab:()=>lab, setScale:(s)=>{scale=s;run();}, inekfPeak:()=>lab.inekf_peak(), ekfPeak:()=>lab.ekf_peak()};
  window.__textbook_ready=true;
  addEventListener("resize",()=>{drawPlot();drawTraj();});
  drawPlot(); drawTraj();
}
main();
`;

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>The estimator that stays honest — ferromotion textbook</title>
<meta name="description" content="An interactive chapter on the invariant EKF: why its error model is exact for any error size while a standard EKF's degrades. Runs the real Rust InEKF machinery on-device."/>
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
#plot{width:100%;height:300px}
#traj{width:100%;height:190px;border-radius:8px;background:#0c1326;margin-bottom:12px}
.read{font-family:var(--mono);font-size:.76rem;color:var(--soft);margin:10px 0 0;text-align:center}.read b{color:var(--ink)}.dim{color:var(--dim)}
.ctl{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:14px;justify-content:center}
label{font-family:var(--mono);font-size:.72rem;color:var(--soft);display:flex;align-items:center;gap:8px}
input[type=range]{accent-color:var(--gold);width:180px}
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
  <p class="kicker"><span style="color:var(--goldb);font-size:1.05rem">&#934;</span><span>ferromotion · textbook · chapter 6</span>
    <span class="badge">rust → wasm · on-device</span></p>
  <h1>The estimator that stays honest</h1>
  <p class="lede">A robot's sense of where it is drifts, and the filter that tracks that drift has to model how its own error grows. The invariant EKF models it <i>exactly</i> — for any error, however large — where a standard filter's model quietly falls apart. This page runs both, on your device, with the same Rust estimator the native tools use.</p>

  <h2><span class="n">01 — the problem</span>A model of your own error</h2>
  <p>An estimator carries two things: a best guess of the state, and a sense of how uncertain that guess is. The uncertainty is propagated through a <i>linear model of how the error evolves</i> — and that model is what keeps the filter honest about its own confidence. If the error-model is wrong, the filter's uncertainty is wrong: it grows overconfident, trusts a bad estimate, and diverges.</p>
  <p>The catch is that the true error dynamics are nonlinear, so the model has to be a linearization — and a standard EKF linearizes around its <b>current estimate</b>. Exactly when the estimate is far off, the linearization point is far off, and the error-model is least trustworthy right when it matters most.</p>

  <h2><span class="n">02 — the setup</span>A wrong guess, dead-reckoned</h2>
  <p>Below, a robot moves under IMU dead-reckoning. Its filter started with a wrong initial guess, so the true path (gold) and the robot's belief (red) pull apart. The question is not whether the guess is wrong — it is — but whether the filter <i>knows how wrong</i>, i.e. whether its error-model tracks the real error.</p>
  <div class="fig">
    <canvas id="traj"></canvas>
    <canvas id="plot"></canvas>
    <div class="ctl">
      <label>initial error size <input type="range" id="scale" min="0.5" max="10" step="0.5" value="3"/> <span id="scaleVal">3.0×</span></label>
    </div>
    <div class="stats">
      <div class="stat"><div class="v" id="inekfPk">—</div><div class="k">InEKF model error</div></div>
      <div class="stat"><div class="v" id="ekfPk">—</div><div class="k">EKF model error</div></div>
      <div class="stat"><div class="v" id="errMag">—</div><div class="k">actual error size</div></div>
    </div>
    <p class="read dim" style="margin-top:8px">the plot is how far each filter's <b>error-model</b> is from the true error — drag the initial error up and watch the red curve balloon while the green stays on zero</p>
  </div>

  <h2><span class="n">03 — the invariant trick</span>An error that doesn't care where you are</h2>
  <p>Barrau & Bonnabel's move is to measure the error not as a subtraction but as a ratio on the group the state lives on — attitude, velocity and position together as one element of <span style="font-family:var(--mono);color:var(--goldb)">SE₂(3)</span>. In those coordinates the error obeys <span style="font-family:var(--mono);color:var(--goldb)">ξ̇ = A ξ</span> with a matrix <b>A that depends only on gravity</b> — not on the estimate, not on the measurements, not on how large the error is.</p>
  <div class="callout">That is the whole difference. The InEKF's <b>A</b> is a constant, so its linear error-model is not an approximation around a point — it is <b>exact everywhere</b>. The standard EKF's transition <b>F</b> carries the estimate's own attitude inside it, so its model is only tangent at the current guess and drifts away as the guess does. Same robot, same measurements, same initial error; one filter's error-model is right to machine precision and the other's is off by as much as the error itself.</p>

  <h2><span class="n">04 — the guarantee</span>Exact, at any error</h2>
  <p>On load, this page ran both filters across error sizes from small to enormous and recorded how far each error-model strayed from the truth:</p>
  <table>
    <tr><td>worst InEKF model error, over error sizes ×0.5 – ×10</td><td id="scInekf">…</td></tr>
    <tr><td>worst standard-EKF model error, same sweep</td><td id="scEkf">…</td></tr>
    <tr><td>verdict</td><td id="scVerdict">…</td></tr>
  </table>
  <p>The InEKF number is machine zero and stays there no matter how wrong the guess starts — the theorem is not "small-error accurate," it is exact. The EKF number climbs with the error, because its model is only ever tangent to the truth at a point it has already left.</p>

  <h2><span class="n">05 — the point</span>Consistency you can prove</h2>
  <div class="verdict">
    <div class="big">Pick the error so its model is exact.</div>
    <p>An estimator is only as honest as its error-model. By measuring error on the group instead of by subtraction, the invariant EKF gets a model that is exact for any error — so its uncertainty stays truthful, and it does not talk itself into a confident wrong answer.</p>
  </div>
  <p>This is why invariant estimation runs under so many legged robots and drones now: the same filter machinery you already know, but consistent by construction rather than by luck and hand-tuning. It is the estimation counterpart to the guarantees in the earlier chapters — the barrier that cannot be crossed, the goal that must be reached — a hard property secured by choosing the right structure, not by hoping the linearization holds.</p>

  <p class="note"><b>What you just drove:</b> the <span style="color:var(--soft)">Se₂(3)</span> group, <span style="color:var(--soft)">riekf_a_matrix</span> and <span style="color:var(--soft)">standard_ekf_f</span> from <span style="color:var(--soft)">ferromotion-control</span>, compiled to WebAssembly — the same code the native tools link against. Truth and a wrong estimate are dead-reckoned through one IMU stream; the true right-invariant error is compared against the InEKF's estimate-independent prediction and a standard EKF's estimate-dependent one. Nothing precomputed — every slider move re-runs the simulation.<br/><br/>
  <b>Verified in the library:</b> the invariant error-model is exact (≈1e-13) at every error size from ×0.2 to ×8 · the standard-EKF model is &gt;100× worse and degrades as the error grows · the invariant attitude error is conserved · A is state-independent while F is not. Each is a test in <span style="color:var(--soft)">cargo test</span>, not a claim in prose. See also <a href="/assets/sims/morphological-computation">ch.1</a> · <a href="/assets/sims/algebraic-connectivity">ch.2</a> · <a href="/assets/sims/safety-filter">ch.3</a> · <a href="/assets/sims/force-closure">ch.4</a> · <a href="/assets/sims/movement-primitives">ch.5</a>.<br/><br/>
  <b>Institute for Physical AI</b> · <a href="https://github.com/dcharlot-physicalai-bmi/ferromotion">the Rust library</a> · <a href="https://crates.io/crates/ferromotion">crates.io</a></p>
</div>
<script type="module">
${glue}
const WASM_B64="${wasmB64}";
${APP}
</script></body></html>`;

const outFile = path.join(__dirname, "..", "..", "v2", "public", "assets", "sims", "invariant-estimation.html");
fs.writeFileSync(outFile, html);
console.log(`wrote ${outFile} (${(html.length / 1024).toFixed(0)} KB, wasm ${(wasmB64.length / 1024).toFixed(0)} KB b64)`);

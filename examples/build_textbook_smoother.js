// Assemble the interactive-textbook chapter on smoothing / SLAM loop closure: inline the wasm-bindgen
// glue + base64-embed the wasm, so the page runs the real ferromotion LegSmoother on-device. The reader
// drives a drifting odometry loop and flips on a loop closure to watch the whole trajectory snap into
// alignment — a smoother revising the past. Same self-contained pattern as the other chapters.
const fs = require("fs");
const path = require("path");

const pkg = path.join(__dirname, "..", "crates", "ferromotion-wasm", "pkg");
const glue = fs.readFileSync(path.join(pkg, "ferromotion_wasm.js"), "utf8");
const wasmB64 = fs.readFileSync(path.join(pkg, "ferromotion_wasm_bg.wasm")).toString("base64");

const APP = String.raw`
function b64ToBytes(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
const fmt=(x,n=2)=>Number(x).toFixed(n);
const TAU=Math.PI*2;
let lab, drift=0.04, closure=false, anim=0;

const cv=document.getElementById("stage");
function bounds(){ // fit all trajectories
  const all=[Array.from(lab.gt_xy()), Array.from(lab.estimate_xy(false)), Array.from(lab.estimate_xy(true))];
  let mnx=1e9,mxx=-1e9,mny=1e9,mxy=-1e9;
  for(const a of all) for(let i=0;i<a.length;i+=2){ mnx=Math.min(mnx,a[i]);mxx=Math.max(mxx,a[i]);mny=Math.min(mny,a[i+1]);mxy=Math.max(mxy,a[i+1]); }
  return {mnx,mxx,mny,mxy};
}
let B=null;
function P(t,x,y){ return [t.ox+x*t.s, t.oy - y*t.s]; }
function tf(){ const r=cv.getBoundingClientRect(); if(!B) B=bounds();
  const s=Math.min((r.width-80)/Math.max(0.5,B.mxx-B.mnx),(r.height-80)/Math.max(0.5,B.mxy-B.mny));
  return {r,s,ox:(r.width-(B.mxx-B.mnx)*s)/2-B.mnx*s, oy:(r.height+(B.mxy-B.mny)*s)/2+B.mny*s}; }

function poly(ctx,t,arr,col,w,dash){ ctx.beginPath(); for(let i=0;i<arr.length;i+=2){ const p=P(t,arr[i],arr[i+1]); i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]); }
  ctx.strokeStyle=col; ctx.lineWidth=w; ctx.setLineDash(dash||[]); ctx.stroke(); ctx.setLineDash([]); }

function draw(){
  const t=tf(); const ctx=cv.getContext("2d");
  const dpr=Math.min(devicePixelRatio||1,2); cv.width=t.r.width*dpr; cv.height=t.r.height*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,t.r.width,t.r.height);
  const gt=Array.from(lab.gt_xy());
  const open=Array.from(lab.estimate_xy(false));
  const closed=Array.from(lab.estimate_xy(true));
  // ground truth (faint dashed)
  poly(ctx,t,gt,"rgba(120,140,180,.35)",2,[5,5]);
  // the estimate: interpolate open→closed by the snap animation
  const a=anim;
  const est=open.map((v,i)=> v + a*(closed[i]-v));
  const col = closure ? "#7dd3a0" : "#e8836f";
  poly(ctx,t,est,col,3);
  // start marker + the loop-closure edge (last → first)
  const s0=P(t,est[0],est[1]);
  ctx.beginPath(); ctx.arc(s0[0],s0[1],6,0,TAU); ctx.fillStyle="#f0cf82"; ctx.fill();
  ctx.fillStyle="rgba(240,207,130,.85)"; ctx.font="500 10px ui-monospace,monospace"; ctx.fillText("start", s0[0]+10, s0[1]-6);
  const n=est.length/2; const eN=P(t,est[2*(n-1)],est[2*(n-1)+1]);
  ctx.beginPath(); ctx.arc(eN[0],eN[1],5,0,TAU); ctx.fillStyle=col; ctx.fill();
  if(closure){ // draw the closure constraint tying end to start
    ctx.beginPath(); ctx.moveTo(eN[0],eN[1]); ctx.lineTo(s0[0],s0[1]);
    ctx.strokeStyle="rgba(125,211,160,.6)"; ctx.lineWidth=1.5; ctx.setLineDash([3,3]); ctx.stroke(); ctx.setLineDash([]);
  } else {
    ctx.fillStyle="#e8836f"; ctx.fillText("← the loop never closes", eN[0]+9, eN[1]+4);
  }
  // legend
  ctx.font="500 10px ui-monospace,monospace";
  ctx.fillStyle="rgba(120,140,180,.6)"; ctx.fillText("- - true path", 12, t.r.height-24);
  ctx.fillStyle=col; ctx.fillText(closure?"— smoothed estimate (loop closed)":"— odometry estimate (drifting)", 12, t.r.height-10);
  // readouts
  document.getElementById("rms").textContent=fmt(lab.rms_error(closure),3);
  document.getElementById("gap").textContent=fmt(lab.loop_gap(closure),3);
  const st=document.getElementById("state"); st.textContent=closure?"loop closed":"open (drifting)"; st.style.color=closure?"#7dd3a0":"#e8836f";
}

function frame(){ // animate the snap when closure toggles
  const target=closure?1:0;
  if(Math.abs(anim-target)>1e-3){ anim += (target-anim)*0.12; draw(); }
  requestAnimationFrame(frame);
}

document.getElementById("closeBtn").onclick=()=>{ closure=!closure; document.getElementById("closeBtn").textContent=closure?"Open the loop":"Close the loop ⟲"; };
document.getElementById("drift").oninput=e=>{ drift=+e.target.value; lab.set_drift(drift); B=null; document.getElementById("driftVal").textContent=fmt(drift,3); draw(); };

function selfCheck(){
  const t=new SmootherLab(); t.set_drift(0.04);
  const open=t.rms_error(false), closed=t.rms_error(true);
  document.getElementById("scOpen").textContent=fmt(open,3);
  document.getElementById("scClosed").textContent=fmt(closed,3);
  document.getElementById("scCut").textContent=fmt((1-closed/open)*100,0)+"%";
  document.getElementById("scVerdict").textContent = closed<0.4*open ? "one late factor corrected the whole past" : "unexpected";
}

async function main(){
  await __wbg_init(b64ToBytes(WASM_B64));
  lab=new SmootherLab(); lab.set_drift(drift); selfCheck();
  window.__sm={lab:()=>lab, setDrift:(d)=>{lab.set_drift(d);}, rms:(c)=>lab.rms_error(c), gap:(c)=>lab.loop_gap(c), close:()=>{closure=true;}, open:()=>{closure=false;}, isClosed:()=>closure};
  window.__textbook_ready=true;
  addEventListener("resize",()=>{B=null;draw();}); draw(); frame();
}
main();
`;

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Revising the past — ferromotion textbook</title>
<meta name="description" content="An interactive chapter on smoothing and loop closure: a drifting odometry loop snaps shut when a single late factor re-optimizes the whole trajectory — a smoother revises the past, a filter cannot. Runs the real Rust pose-graph smoother on-device."/>
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
#stage{width:100%;height:420px;border-radius:10px;background:radial-gradient(700px 420px at 50% 48%,#0e1730,#0b1122)}
.read{font-family:var(--mono);font-size:.76rem;color:var(--soft);margin:10px 0 0;text-align:center}.read b{color:var(--ink)}.dim{color:var(--dim)}
.ctl{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:14px;justify-content:center}
button{background:linear-gradient(180deg,#d9b45e,#a9832f);color:#161200;border:0;border-radius:9px;padding:8px 16px;font:700 .82rem var(--sans);cursor:pointer}
label{font-family:var(--mono);font-size:.72rem;color:var(--soft);display:flex;align-items:center;gap:8px}
input[type=range]{accent-color:var(--gold);width:150px}
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
  <p class="kicker"><span style="color:var(--goldb);font-size:1.05rem">&#934;</span><span>ferromotion · textbook · chapter 12</span>
    <span class="badge">rust → wasm · on-device</span></p>
  <h1>Revising the past</h1>
  <p class="lede">A robot's dead-reckoned map of where it has been slowly drifts, and by the time it returns to the start the loop no longer closes. The fix is not a better guess about the present — it is the willingness to go back and correct the whole past at once. This page does it live, with the same Rust pose-graph smoother the native tools use.</p>

  <h2><span class="n">01 — the problem</span>Small errors, compounded</h2>
  <p>Every step, the robot estimates how far it moved from its wheels or its legs — and every estimate is a little wrong. Those little errors compound: over a long loop the belief spirals away from the truth, so when the robot physically returns to where it began, its map says it is somewhere else entirely. A filter, which keeps only the present estimate and throws the past away, can never repair this — the drift is baked into history it no longer holds.</p>

  <h2><span class="n">02 — the loop closure</span>"I have been here before"</h2>
  <p>A <b>smoother</b> keeps the whole trajectory as a graph of poses tied by constraints, and re-optimizes all of it whenever a new constraint arrives. The powerful one is a <b>loop closure</b>: the robot recognizes a place it has already visited and adds a single edge — "this pose equals that one." That one late fact is inconsistent with the drifted chain, and resolving the inconsistency pushes a correction backward through every pose in the loop.</p>
  <p>Below, the robot has driven a loop with drifting odometry. <b>Close the loop</b> and watch the entire path snap into alignment with the truth.</p>
  <div class="fig">
    <canvas id="stage"></canvas>
    <div class="ctl">
      <button id="closeBtn">Close the loop ⟲</button>
      <label>odometry drift <input type="range" id="drift" min="0" max="0.08" step="0.005" value="0.04"/> <span id="driftVal">0.040</span></label>
    </div>
    <div class="stats">
      <div class="stat"><div class="v" id="state" style="font-size:.82rem">—</div><div class="k">the loop</div></div>
      <div class="stat"><div class="v" id="rms">—</div><div class="k">trajectory error</div></div>
      <div class="stat"><div class="v" id="gap">—</div><div class="k">loop gap</div></div>
    </div>
    <p class="read dim" style="margin-top:8px">the dashed line is the true path; the drifting estimate never returns to <b>start</b> — until the closure edge ties the ends together</p>
  </div>

  <h2><span class="n">03 — the correction flows backward</span>Not just the endpoint</h2>
  <div class="callout">Notice what moves when you close the loop: not only the final pose snapping onto the start, but <b>every pose along the way</b> shifting to share the correction. The smoother does not clamp the end and leave the rest; it finds the trajectory that best satisfies <i>all</i> the constraints at once — the odometry chain and the closure together — so the error is distributed smoothly around the whole loop. That backward flow of information is exactly what a filter gives up.</div>

  <h2><span class="n">04 — the number</span>How much it helps</h2>
  <p>On load, this page drove the drifting loop and measured the whole-trajectory error with and without the single closure edge:</p>
  <table>
    <tr><td>trajectory RMS error — odometry only</td><td id="scOpen">…</td></tr>
    <tr><td>trajectory RMS error — with one loop closure</td><td id="scClosed">…</td></tr>
    <tr><td>error removed by one late factor</td><td id="scCut">…</td></tr>
    <tr><td>verdict</td><td id="scVerdict">…</td></tr>
  </table>
  <p>One constraint, added at the end, cuts the error across the entire history — because the estimate was never committed. Turn the <b>drift</b> up and the open loop yawns wider, yet the same single closure still pulls it shut.</p>

  <h2><span class="n">05 — the point</span>Keep the past editable</h2>
  <div class="verdict">
    <div class="big">A map is a hypothesis, not a record.</div>
    <p>By holding the whole trajectory as a graph of constraints rather than a fixed log, a smoother can accept a fact learned late and let it rewrite everything that came before — which is how a robot builds a map that actually closes.</p>
  </div>
  <p>This is the backbone of modern SLAM: pose-graph back-ends that keep every keyframe live and re-solve on each loop closure. It is the sibling of the estimation chapter — the invariant filter keeps the present consistent moment to moment; the smoother keeps the past consistent in hindsight — and together they are how an embodied system knows where it is and where it has been. The same sparse least-squares solver drives both a legged robot's foothold history and a drone's flight around a building.</p>

  <p class="note"><b>What you just drove:</b> the <span style="color:var(--soft)">LegSmoother</span> from <span style="color:var(--soft)">ferromotion-core</span> — an SE(2) pose-graph fixed-lag smoother over a sparse factor graph (faer sparse Cholesky), compiled to WebAssembly, the same code the native tools link against. The estimate is re-solved from the constraints on every change; nothing is precomputed.<br/><br/>
  <b>Verified in the library:</b> it recovers a ground-truth trajectory from perfect factors to 1e-6; a late long-baseline factor retro-corrects interior poses (a smoother, not a filter); loop closure slashes the trajectory error &gt;60% and shuts the loop; more drift opens a wider loop. Each is a test in <span style="color:var(--soft)">cargo test</span>, not a claim in prose. See also <a href="/assets/sims/invariant-estimation">ch.6 — the estimator that stays honest</a> · <a href="/assets/sims/textbook">the full textbook</a>.<br/><br/>
  <b>Institute for Physical AI</b> · <a href="https://github.com/dcharlot-physicalai-bmi/ferromotion">the Rust library</a> · <a href="https://crates.io/crates/ferromotion">crates.io</a></p>
</div>
<script type="module">
${glue}
const WASM_B64="${wasmB64}";
${APP}
</script></body></html>`;

const outFile = path.join(__dirname, "..", "..", "v2", "public", "assets", "sims", "loop-closure.html");
fs.writeFileSync(outFile, html);
console.log(`wrote ${outFile} (${(html.length / 1024).toFixed(0)} KB, wasm ${(wasmB64.length / 1024).toFixed(0)} KB b64)`);

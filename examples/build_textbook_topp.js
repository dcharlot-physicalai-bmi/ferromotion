// Assemble the interactive-textbook chapter on time-optimal path parameterization: inline the
// wasm-bindgen glue + base64-embed the wasm, so the page runs the real ferromotion topp() on-device.
// The reader watches the time-optimal traversal race a constant-speed one on the same path, and sees
// the speed profile ride the velocity ceiling with bang-bang structure. Same pattern as chapters 1–6.
const fs = require("fs");
const path = require("path");

const pkg = path.join(__dirname, "..", "crates", "ferromotion-wasm", "pkg");
const glue = fs.readFileSync(path.join(pkg, "ferromotion_wasm.js"), "utf8");
const wasmB64 = fs.readFileSync(path.join(pkg, "ferromotion_wasm_bg.wasm")).toString("base64");

const APP = String.raw`
function b64ToBytes(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
const fmt=(x,n=2)=>Number(x).toFixed(n);
const TAU=Math.PI*2;

let lab, vlim=1.2, alim=2.5, pathXY=[], clock=0;
let dur=1, naive=1, feasible=true;

function genPath(kind){
  const N=140, xs=[], ys=[];
  for(let i=0;i<=N;i++){ const s=i/N;
    if(kind==='corner'){ if(s<0.5){ xs.push(s*2*2); ys.push(0);} else { const a=(s-0.5)*2*Math.PI/2; xs.push(2+0.55*Math.sin(a)); ys.push(0.55*(1-Math.cos(a))); } }
    else if(kind==='hairpin'){ if(s<0.38){ xs.push(-1.6+s/0.38*1.6); ys.push(-0.9);} else if(s<0.62){ const a=(s-0.38)/0.24*Math.PI; xs.push(0.5*Math.sin(a)); ys.push(-0.9+0.9*(1-Math.cos(a))); } else { xs.push(-(s-0.62)/0.38*1.6); ys.push(0.9);} }
    else if(kind==='wave'){ xs.push(-2+4*s); ys.push(Math.sin(s*Math.PI*2)*0.9); }
  }
  loadPath(xs,ys);
  document.querySelectorAll('.seg button').forEach(b=>b.classList.toggle('on', b.dataset.k===kind));
}
function loadPath(xs,ys){
  lab.set_path(new Float64Array(xs), new Float64Array(ys));
  pathXY=Array.from(lab.path_xy());
  resolve();
}
function resolve(){
  feasible=lab.solve(vlim,vlim,alim,alim);
  dur=lab.duration(); naive=lab.naive_duration(); clock=0; syncReadouts(); drawProfile();
}

const cv=document.getElementById("stage");
function bounds(){ let mnx=1e9,mxx=-1e9,mny=1e9,mxy=-1e9; for(let i=0;i<pathXY.length;i+=2){mnx=Math.min(mnx,pathXY[i]);mxx=Math.max(mxx,pathXY[i]);mny=Math.min(mny,pathXY[i+1]);mxy=Math.max(mxy,pathXY[i+1]);} return {mnx,mxx,mny,mxy}; }
function draw(){
  const ctx=cv.getContext("2d"); const r=cv.getBoundingClientRect();
  const dpr=Math.min(devicePixelRatio||1,2); cv.width=r.width*dpr; cv.height=r.height*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  const W=r.width,H=r.height; ctx.clearRect(0,0,W,H);
  if(pathXY.length<2) return;
  const b=bounds(); const s=Math.min((W-70)/Math.max(1e-6,b.mxx-b.mnx),(H-70)/Math.max(1e-6,b.mxy-b.mny));
  const ox=(W-(b.mxx-b.mnx)*s)/2-b.mnx*s, oy=(H+(b.mxy-b.mny)*s)/2+b.mny*s;
  const P=(x,y)=>[ox+x*s, oy-y*s];
  // path
  ctx.beginPath(); for(let i=0;i<pathXY.length;i+=2){const p=P(pathXY[i],pathXY[i+1]); i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]);}
  ctx.strokeStyle="rgba(174,182,204,.35)"; ctx.lineWidth=8; ctx.lineCap="round"; ctx.stroke();
  // color the path by TOPP speed (fast=bright)
  const sp=Array.from(lab.speed()), mx=Math.max(...sp,1e-6);
  for(let i=0;i<pathXY.length/2-1;i++){ const p0=P(pathXY[2*i],pathXY[2*i+1]),p1=P(pathXY[2*i+2],pathXY[2*i+3]);
    const f=sp[i]/mx; ctx.beginPath(); ctx.moveTo(p0[0],p0[1]); ctx.lineTo(p1[0],p1[1]);
    ctx.strokeStyle="rgba("+Math.round(120+135*f)+","+Math.round(180+30*f)+","+Math.round(110+40*(1-f))+",.9)"; ctx.lineWidth=4; ctx.stroke(); }
  // constant-speed dot (gray) — traverses s uniformly over naive time
  if(feasible){
    const sc=Math.min(1, clock/naive); const ic=Math.min(pathXY.length/2-1, Math.floor(sc*(pathXY.length/2-1)));
    const pc=P(pathXY[2*ic],pathXY[2*ic+1]);
    ctx.beginPath(); ctx.arc(pc[0],pc[1],8,0,TAU); ctx.fillStyle="#5b6680"; ctx.strokeStyle="#8290a8"; ctx.lineWidth=2; ctx.fill(); ctx.stroke();
    // TOPP dot (green)
    const pt=Array.from(lab.pos_at_time(Math.min(clock,dur))); const pp=P(pt[0],pt[1]);
    ctx.beginPath(); ctx.arc(pp[0],pp[1],9,0,TAU); ctx.fillStyle="#161f3a"; ctx.strokeStyle="#7dd3a0"; ctx.lineWidth=3; ctx.fill(); ctx.stroke();
    // finish flags
    if(clock>=dur){ ctx.fillStyle="#7dd3a0"; ctx.font="600 11px ui-sans-serif,system-ui"; ctx.fillText("TOPP ✓", pp[0]+12, pp[1]-8); }
  } else {
    ctx.fillStyle="#e8836f"; ctx.font="600 13px ui-sans-serif,system-ui"; ctx.textAlign="center"; ctx.fillText("infeasible at these limits", W/2, H/2); ctx.textAlign="left";
  }
  ctx.font="500 10px ui-monospace,monospace";
  ctx.fillStyle="#7dd3a0"; ctx.fillText("● time-optimal", 12, H-24);
  ctx.fillStyle="#8290a8"; ctx.fillText("● one safe speed", 12, H-10);
}

const prof=document.getElementById("profile");
function drawProfile(){
  const ctx=prof.getContext("2d"); const r=prof.getBoundingClientRect();
  const dpr=Math.min(devicePixelRatio||1,2); prof.width=r.width*dpr; prof.height=r.height*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  const W=r.width,H=r.height,pad={l:20,r:12,t:14,b:22}; ctx.clearRect(0,0,W,H);
  if(!feasible){ return; }
  const mvc=Array.from(lab.mvc_speed()), sp=Array.from(lab.speed()); const n=sp.length;
  const ymax=Math.max(...mvc.filter(v=>isFinite(v)),1e-6)*1.1;
  const x2p=i=>pad.l+(i/(n-1))*(W-pad.l-pad.r), y2p=v=>H-pad.b-(Math.min(v,ymax)/ymax)*(H-pad.t-pad.b);
  ctx.strokeStyle="rgba(120,140,180,.12)"; for(let k=0;k<=3;k++){const y=pad.t+k/3*(H-pad.t-pad.b);ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();}
  // MVC ceiling (gold dashed)
  ctx.beginPath(); mvc.forEach((v,i)=>{const X=x2p(i),Y=y2p(v); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});
  ctx.strokeStyle="rgba(240,207,130,.8)"; ctx.lineWidth=1.5; ctx.setLineDash([5,4]); ctx.stroke(); ctx.setLineDash([]);
  // TOPP speed (green) — rides the ceiling
  ctx.beginPath(); sp.forEach((v,i)=>{const X=x2p(i),Y=y2p(v); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});
  ctx.strokeStyle="#7dd3a0"; ctx.lineWidth=2.5; ctx.stroke();
  ctx.fillStyle="rgba(240,207,130,.85)"; ctx.font="500 10px ui-monospace,monospace"; ctx.fillText("velocity ceiling (max possible)", pad.l+6, pad.t+10);
  ctx.fillStyle="#7dd3a0"; ctx.fillText("time-optimal speed", pad.l+6, pad.t+24);
  ctx.fillStyle="#727d99"; ctx.fillText("path position s →", W-pad.r-96, H-7);
}

function syncReadouts(){
  document.getElementById("toppT").textContent = feasible ? fmt(dur,2)+" s" : "—";
  document.getElementById("naiveT").textContent = feasible ? fmt(naive,2)+" s" : "—";
  document.getElementById("speedup").textContent = feasible ? fmt(naive/dur,2)+"×" : "—";
  document.getElementById("bang").textContent = feasible ? fmt(lab.saturated_fraction()*100,0)+"%" : "—";
}

document.getElementById("vlim").oninput=e=>{ vlim=+e.target.value; document.getElementById("vlimVal").textContent=fmt(vlim,1); resolve(); };
document.getElementById("alim").oninput=e=>{ alim=+e.target.value; document.getElementById("alimVal").textContent=fmt(alim,1); resolve(); };
document.querySelectorAll('.seg button').forEach(b=> b.onclick=()=>genPath(b.dataset.k));

function frame(){
  if(feasible){ clock+=0.02; if(clock > Math.max(dur,naive)+0.6) clock=0; }
  draw();
  requestAnimationFrame(frame);
}

/* self-check on load: straight-line duration matches the analytic trapezoid */
function selfCheck(){
  const t=new ToppLab(); const N=2000, xs=[], ys=[];
  for(let i=0;i<=N;i++){ xs.push(2*i/N); ys.push(0); }
  t.set_path(new Float64Array(xs), new Float64Array(ys));
  t.solve(1,1,2,2);
  const analytic=2/1+1/2; // L/v + v/a
  document.getElementById("scTopp").textContent=fmt(t.duration(),4)+" s";
  document.getElementById("scAna").textContent=fmt(analytic,4)+" s";
  document.getElementById("scErr").textContent=fmt(Math.abs(t.duration()-analytic)/analytic*100,3)+"%";
}

async function main(){
  await __wbg_init(b64ToBytes(WASM_B64));
  lab=new ToppLab(); selfCheck(); genPath('corner');
  window.__topp={lab:()=>lab, setLimits:(v,a)=>{vlim=v;alim=a;resolve();}, preset:genPath, dur:()=>dur, naive:()=>naive, feasible:()=>feasible, bang:()=>lab.saturated_fraction()};
  window.__textbook_ready=true;
  addEventListener("resize",()=>{draw();drawProfile();});
  frame();
}
main();
`;

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>As fast as the motors allow — ferromotion textbook</title>
<meta name="description" content="An interactive chapter on time-optimal path parameterization (TOPP): the fastest way to follow a path is bang-bang, always pinned to a limit. Runs the real Rust TOPP solver on-device."/>
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
#stage{width:100%;height:360px;border-radius:10px;background:radial-gradient(700px 360px at 50% 45%,#0e1730,#0b1122)}
#profile{width:100%;height:150px;margin-top:10px;border-radius:8px;background:#0c1326}
.read{font-family:var(--mono);font-size:.76rem;color:var(--soft);margin:10px 0 0;text-align:center}.read b{color:var(--ink)}.dim{color:var(--dim)}
.ctl{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:14px;justify-content:center}
.seg{display:inline-flex;background:#0d1428;border:1px solid var(--line);border-radius:9px;overflow:hidden;flex-wrap:wrap}
.seg button{background:transparent;color:var(--soft);border:0;border-radius:0;padding:8px 13px;font:600 .76rem var(--sans);cursor:pointer}
.seg button.on{background:linear-gradient(180deg,rgba(217,180,94,.22),rgba(217,180,94,.06));color:var(--goldb)}
label{font-family:var(--mono);font-size:.72rem;color:var(--soft);display:flex;align-items:center;gap:8px}
input[type=range]{accent-color:var(--gold);width:120px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}
.stat{background:#0d1428;border:1px solid var(--line);border-radius:10px;padding:9px 6px;text-align:center}
.stat .v{font-family:var(--mono);font-size:1rem;font-weight:700;color:var(--goldb)}
.stat .k{font-family:var(--mono);font-size:.56rem;letter-spacing:.06em;text-transform:uppercase;color:var(--dim);margin-top:2px}
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
  <p class="kicker"><span style="color:var(--goldb);font-size:1.05rem">&#934;</span><span>ferromotion · textbook · chapter 7</span>
    <span class="badge">rust → wasm · on-device</span></p>
  <h1>As fast as the motors allow</h1>
  <p class="lede">Give a robot a path to follow and ask it to do so in the least possible time, and the answer is never a single speed — it must crawl through the corners and floor it on the straights. This page computes that time-optimal schedule with the same Rust solver the native tools use.</p>

  <h2><span class="n">01 — the problem</span>The path is fixed; the timing is not</h2>
  <p>Suppose the <i>shape</i> of a motion is already decided — a planner handed you a path around the obstacles, or it traces a weld seam or a brush stroke. What is left is the timing: how fast to move along it at each point. Go too fast into a corner and you exceed what the motors can do; go slow enough for the corner everywhere and you crawl the straights for no reason. Somewhere between is the fastest traversal the hardware permits.</p>

  <h2><span class="n">02 — the race</span>One speed is never optimal</h2>
  <p>Below, two dots run the same path. The <b style="color:var(--green)">green</b> one uses the time-optimal schedule; the <b style="color:var(--soft)">grey</b> one uses the single fastest speed that is safe everywhere — which is set by the tightest corner, so it is stuck crawling the whole way. Watch the green pull ahead on the straights and ease through the turn. The path itself is tinted by speed.</p>
  <div class="fig">
    <canvas id="stage"></canvas>
    <canvas id="profile"></canvas>
    <div class="ctl" style="margin-top:12px">
      <span class="seg"><button data-k="corner" class="on">Corner</button><button data-k="hairpin">Hairpin</button><button data-k="wave">Wave</button></span>
    </div>
    <div class="ctl">
      <label>velocity limit <input type="range" id="vlim" min="0.4" max="3" step="0.1" value="1.2"/> <span id="vlimVal">1.2</span></label>
      <label>accel limit <input type="range" id="alim" min="0.5" max="6" step="0.1" value="2.5"/> <span id="alimVal">2.5</span></label>
    </div>
    <div class="stats">
      <div class="stat"><div class="v" id="toppT">—</div><div class="k">time-optimal</div></div>
      <div class="stat"><div class="v" id="naiveT">—</div><div class="k">one safe speed</div></div>
      <div class="stat"><div class="v" id="speedup">—</div><div class="k">speed-up</div></div>
      <div class="stat"><div class="v" id="bang">—</div><div class="k">pinned to a limit</div></div>
    </div>
    <p class="read dim" style="margin-top:8px">the lower plot is speed along the path: the <b>time-optimal</b> curve rides right under the <b>velocity ceiling</b>, dipping only where the path bends</p>
  </div>

  <h2><span class="n">03 — the trick</span>Square the speed and it goes linear</h2>
  <p>What makes this solvable exactly is a change of variable. Write everything in terms of <span style="font-family:var(--mono);color:var(--goldb)">x = ṡ²</span>, the squared path speed. Then the joint velocities are linear in <span style="font-family:var(--mono);color:var(--goldb)">√x</span> and the joint accelerations are linear in <span style="font-family:var(--mono);color:var(--goldb)">(s̈, x)</span> — so every velocity and torque limit becomes a straight line in this space. Velocity limits put a ceiling on <span style="font-family:var(--mono);color:var(--goldb)">x</span> — the maximum-velocity curve you see in the plot — and acceleration limits give, at each point, an interval of allowed path acceleration.</p>
  <div class="callout">With the limits linear, the optimum has a clean shape: sweep <b>backward</b> from the goal to find, at every point, the fastest you could be going and still stop in time; then sweep <b>forward</b> accelerating as hard as allowed. The result is <b>bang-bang</b> — at every instant you are flat-out accelerating, flat-out braking, or riding the ceiling. Never coasting in the middle. The "pinned to a limit" figure above is that fraction, and it stays near 100%.</p>

  <h2><span class="n">04 — the check</span>Against a known answer</h2>
  <p>For a straight move the optimum is the textbook trapezoid — accelerate at the limit, cruise at top speed, brake at the limit — with a duration you can write down by hand. On load, this page solved that case and compared:</p>
  <table>
    <tr><td>TOPP duration (straight move, L=2, v=1, a=2)</td><td id="scTopp">…</td></tr>
    <tr><td>analytic optimum L/v + v/a</td><td id="scAna">…</td></tr>
    <tr><td>agreement</td><td id="scErr">…</td></tr>
  </table>

  <h2><span class="n">05 — the point</span>The other half of motion planning</h2>
  <div class="verdict">
    <div class="big">Planning finds the path; this finds the clock.</div>
    <p>A geometric planner decides where to go; time-optimal parameterization decides how fast to go along it — squeezing every bit of speed the motors can give while never asking for more than they have.</p>
  </div>
  <p>The two halves compose cleanly, which is why they are usually separate. A sampling planner routes around the obstacles without worrying about dynamics; then this pass lays the fastest feasible timing onto that route. Loosen the acceleration limit above and the whole schedule speeds up; tighten it and the corners bite harder — but at every setting the traversal is provably the fastest the limits allow, not a hand-tuned guess. It is the same theme as the rest of this series: turn a hard problem into a place where the answer is forced, and then just read it off.</p>

  <p class="note"><b>What you just drove:</b> <span style="color:var(--soft)">topp</span> and <span style="color:var(--soft)">ToppPath</span> from <span style="color:var(--soft)">ferromotion-control</span>, compiled to WebAssembly — the same code the native tools link against. The path's derivatives are taken by finite differences; the solver runs a backward reachability pass and a greedy forward sweep in the <span style="font-family:var(--mono)">x = ṡ²</span> variable. Nothing precomputed — every slider move re-solves it.<br/><br/>
  <b>Verified in the library:</b> a straight move matches the analytic trapezoidal optimum, a short one the triangular optimum · the optimum is bang-bang — pinned to a limit &gt;95% of the way · varying the speed beats any single safe speed · tighter limits take longer. Each is a test in <span style="color:var(--soft)">cargo test</span>, not a claim in prose. See also <a href="/assets/sims/morphological-computation">ch.1</a> · <a href="/assets/sims/algebraic-connectivity">ch.2</a> · <a href="/assets/sims/safety-filter">ch.3</a> · <a href="/assets/sims/force-closure">ch.4</a> · <a href="/assets/sims/movement-primitives">ch.5</a> · <a href="/assets/sims/invariant-estimation">ch.6</a>.<br/><br/>
  <b>Institute for Physical AI</b> · <a href="https://github.com/dcharlot-physicalai-bmi/ferromotion">the Rust library</a> · <a href="https://crates.io/crates/ferromotion">crates.io</a></p>
</div>
<script type="module">
${glue}
const WASM_B64="${wasmB64}";
${APP}
</script></body></html>`;

const outFile = path.join(__dirname, "..", "..", "v2", "public", "assets", "sims", "time-optimal.html");
fs.writeFileSync(outFile, html);
console.log(`wrote ${outFile} (${(html.length / 1024).toFixed(0)} KB, wasm ${(wasmB64.length / 1024).toFixed(0)} KB b64)`);

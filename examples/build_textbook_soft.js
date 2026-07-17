// Assemble the interactive-textbook chapter on soft/continuum robots: inline the wasm-bindgen glue +
// base64-embed the wasm, so the page runs the real ferromotion CosseratRod on-device. The reader hangs
// a load on a compliant arm, watches the whole body curve, and sees the tip deflection match
// Euler-Bernoulli beam theory. Same self-contained pattern as the other chapters.
const fs = require("fs");
const path = require("path");

const pkg = path.join(__dirname, "..", "crates", "ferromotion-wasm", "pkg");
const glue = fs.readFileSync(path.join(pkg, "ferromotion_wasm.js"), "utf8");
const wasmB64 = fs.readFileSync(path.join(pkg, "ferromotion_wasm_bg.wasm")).toString("base64");

const APP = String.raw`
function b64ToBytes(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
const fmt=(x,n=2)=>Number(x).toFixed(n);
const TAU=Math.PI*2, L=1.0;

let rod, ei=0.8, load=[0.0,-0.3], dragging=false;
const cv=document.getElementById("stage");
function tf(){ const r=cv.getBoundingClientRect(); const s=(r.width-120)/1.35; return {r,s,ox:70,oy:r.height*0.42}; }
function P(x,y){ const t=tf(); return [t.ox+x*t.s, t.oy - y*t.s]; }
function inv(px,py){ const t=tf(); return [(px-t.ox)/t.s, -(py-t.oy)/t.s]; }

function recompute(){ rod.set_stiffness(ei); rod.set_load(load[0],load[1]); rod.solve(); }

function draw(){
  const ctx=cv.getContext("2d"); const t=tf();
  const dpr=Math.min(devicePixelRatio||1,2); cv.width=t.r.width*dpr; cv.height=t.r.height*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,t.r.width,t.r.height);
  // clamp/wall at the base
  const base=P(0,0);
  ctx.fillStyle="#1a2440"; ctx.fillRect(base[0]-16,base[1]-34,16,68);
  ctx.strokeStyle="#39456a"; ctx.strokeRect(base[0]-16,base[1]-34,16,68);
  // undeformed reference (faint)
  const refTip=P(L,0);
  ctx.strokeStyle="rgba(120,140,180,.18)"; ctx.lineWidth=2; ctx.setLineDash([4,5]);
  ctx.beginPath(); ctx.moveTo(base[0],base[1]); ctx.lineTo(refTip[0],refTip[1]); ctx.stroke(); ctx.setLineDash([]);
  // Euler-Bernoulli predicted tip (transverse), for the current downward load
  const eb=rod.euler_bernoulli();
  const ebValid = Math.abs(eb) > 1e-4 && Math.abs(eb) < 0.18; // small-deflection regime
  if(ebValid){ const ebp=P(L,-eb);
    ctx.beginPath(); ctx.arc(ebp[0],ebp[1],9,0,TAU); ctx.strokeStyle="rgba(138,180,240,.8)"; ctx.lineWidth=1.5; ctx.setLineDash([3,3]); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle="rgba(138,180,240,.85)"; ctx.font="500 10px ui-monospace,monospace"; ctx.fillText("beam theory", ebp[0]+13, ebp[1]+4);
  } else if(Math.abs(eb)>=0.18){
    ctx.fillStyle="rgba(138,180,240,.6)"; ctx.font="500 10px ui-monospace,monospace"; ctx.fillText("beam theory no longer applies (large deflection)", 12, 22); }
  // the soft arm — a thick tapered rubbery band along the backbone
  const b=Array.from(rod.backbone_xy()); const np=b.length/2;
  ctx.lineCap="round"; ctx.lineJoin="round";
  ctx.beginPath(); for(let i=0;i<np;i++){ const p=P(b[2*i],b[2*i+1]); i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]); }
  ctx.strokeStyle="rgba(217,180,94,.25)"; ctx.lineWidth=16; ctx.stroke();
  ctx.strokeStyle="#d9b45e"; ctx.lineWidth=7; ctx.stroke();
  // tip + hanging load handle
  const tip=P(rod.tip_x(), rod.tip_y());
  ctx.beginPath(); ctx.arc(tip[0],tip[1],8,0,TAU); ctx.fillStyle="#161f3a"; ctx.strokeStyle="#f0cf82"; ctx.lineWidth=2.5; ctx.fill(); ctx.stroke();
  // load arrow from the tip
  const sc=70;
  const lx=tip[0]+load[0]*sc, ly=tip[1]-load[1]*sc;
  ctx.strokeStyle="#e8836f"; ctx.fillStyle="#e8836f"; ctx.lineWidth=2.5;
  ctx.beginPath(); ctx.moveTo(tip[0],tip[1]); ctx.lineTo(lx,ly); ctx.stroke();
  const a=Math.atan2(ly-tip[1],lx-tip[0]);
  ctx.beginPath(); ctx.moveTo(lx,ly); ctx.lineTo(lx-9*Math.cos(a-0.4),ly-9*Math.sin(a-0.4)); ctx.lineTo(lx-9*Math.cos(a+0.4),ly-9*Math.sin(a+0.4)); ctx.closePath(); ctx.fill();
  ctx.beginPath(); ctx.arc(lx,ly,10,0,TAU); ctx.strokeStyle=dragging?"#f0cf82":"rgba(232,131,111,.7)"; ctx.lineWidth=2; ctx.stroke();
  ctx.fillStyle="rgba(232,131,111,.85)"; ctx.font="500 10px ui-monospace,monospace"; ctx.fillText("load · drag", lx+13, ly+4);
  // readouts
  document.getElementById("defl").textContent=fmt(rod.deflection(),4);
  document.getElementById("eb").textContent=fmt(eb,4);
  const rel=Math.abs(eb)>1e-4 ? Math.abs(rod.deflection()-eb)/Math.abs(eb)*100 : 0;
  const ag=document.getElementById("agree"); ag.textContent=Math.abs(eb)>1e-4?fmt(rel,1)+"%":"—";
  ag.style.color = rel<5 ? "#7dd3a0" : "#f0cf82";
  document.getElementById("eiVal").textContent=fmt(ei,1);
}

cv.addEventListener("pointerdown",e=>{ const r=cv.getBoundingClientRect(); const tip=P(rod.tip_x(),rod.tip_y());
  const sc=70, lx=tip[0]+load[0]*sc, ly=tip[1]-load[1]*sc;
  if(Math.hypot(e.clientX-r.left-lx, e.clientY-r.top-ly)<22){ dragging=true; cv.setPointerCapture(e.pointerId); } });
cv.addEventListener("pointermove",e=>{ if(!dragging) return; const r=cv.getBoundingClientRect(); const tip=P(rod.tip_x(),rod.tip_y());
  const sc=70; load=[(e.clientX-r.left-tip[0])/sc, -(e.clientY-r.top-tip[1])/sc];
  // clamp magnitude so the arm stays on screen (large loads curl it well past beam theory)
  const m=Math.hypot(load[0],load[1]), mx=1.2; if(m>mx){ load=[load[0]/m*mx, load[1]/m*mx]; }
  recompute(); draw(); });
cv.addEventListener("pointerup",()=>{ dragging=false; });
document.getElementById("ei").oninput=e=>{ ei=+e.target.value; recompute(); draw(); };
document.getElementById("reset").onclick=()=>{ load=[0.0,-0.3]; ei=0.8; document.getElementById("ei").value=0.8; recompute(); draw(); };

function selfCheck(){
  const t=new SoftRod(60,1.0,5.0); t.set_load(0.0,-0.02); t.solve();
  document.getElementById("scDefl").textContent=fmt(t.deflection(),4);
  document.getElementById("scEb").textContent=fmt(t.euler_bernoulli(),4);
  const rel=Math.abs(t.deflection()-t.euler_bernoulli())/t.euler_bernoulli()*100;
  document.getElementById("scErr").textContent=fmt(rel,2)+"%";
  document.getElementById("scVerdict").textContent = rel<2 ? "the strain model reproduces beam theory" : "unexpected";
}

async function main(){
  await __wbg_init(b64ToBytes(WASM_B64));
  rod=new SoftRod(60,1.0,5.0); selfCheck(); recompute();
  window.__soft={rod:()=>rod, setLoad:(x,y)=>{load=[x,y];recompute();}, setEi:(v)=>{ei=v;recompute();}, deflection:()=>rod.deflection(), eb:()=>rod.euler_bernoulli()};
  window.__textbook_ready=true;
  addEventListener("resize",draw); draw();
}
main();
`;

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>The robot that bends — ferromotion textbook</title>
<meta name="description" content="An interactive chapter on soft and continuum robots: describe a compliant arm by its strain field, and its bending under load matches Euler-Bernoulli beam theory. Runs the real Rust Cosserat model on-device."/>
<style>
:root{--ground:#0a0f1e;--panel:#111830;--line:#26324c;--ink:#eef1f8;--soft:#aeb6cc;--dim:#727d99;--gold:#d9b45e;--goldb:#f0cf82;--green:#7dd3a0;--red:#e8836f;--blue:#8ab4f0;--mono:ui-monospace,"SF Mono",Menlo,monospace;--sans:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
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
#stage{width:100%;height:380px;cursor:grab;border-radius:10px;background:radial-gradient(700px 380px at 45% 42%,#0e1730,#0b1122)}
.read{font-family:var(--mono);font-size:.76rem;color:var(--soft);margin:10px 0 0;text-align:center}.read b{color:var(--ink)}.dim{color:var(--dim)}
.ctl{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:14px;justify-content:center}
button{background:linear-gradient(180deg,#d9b45e,#a9832f);color:#161200;border:0;border-radius:9px;padding:8px 16px;font:700 .82rem var(--sans);cursor:pointer}
button.ghost{background:transparent;color:var(--soft);border:1px solid var(--line)}
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
  <p class="kicker"><span style="color:var(--goldb);font-size:1.05rem">&#934;</span><span>ferromotion · textbook · chapter 11</span>
    <span class="badge">rust → wasm · on-device</span></p>
  <h1>The robot that bends</h1>
  <p class="lede">Not every robot is rigid links and hinges. A soft arm has no joints at all — it curves continuously along its whole length, and to control it you first have to describe that curve. This page bends a real compliant rod on your device and checks its shape against beam theory.</p>

  <h2><span class="n">01 — the problem</span>A robot with no joints</h2>
  <p>A rigid arm has a handful of joint angles, and its pose is those numbers. A soft, continuum arm — a tentacle, a growing vine-robot, a silicone finger — bends everywhere. In principle it has infinitely many degrees of freedom. So the first question is not how to control it but how to even <i>write down</i> its configuration with a finite handful of numbers.</p>

  <h2><span class="n">02 — the strain field</span>Describe the bend, not the points</h2>
  <p>The trick is to describe the rod by its <b>strain along the arc length</b> — here, its curvature — rather than by where each point sits. Split the rod into short sections of constant curvature; a few curvatures then determine the whole shape, recovered by integrating along the body. It is the soft-robot analogue of joint angles: a compact handle on a continuous thing.</p>
  <p>Below is a compliant arm clamped at the wall. <b>Drag the load</b> at its tip and watch the entire body curve to balance it; adjust its <b>stiffness</b>.</p>
  <div class="fig">
    <canvas id="stage"></canvas>
    <div class="ctl">
      <button id="reset" class="ghost">Reset</button>
      <label>stiffness EI <input type="range" id="ei" min="0.2" max="6" step="0.1" value="0.8"/> <span id="eiVal">0.8</span></label>
    </div>
    <div class="stats">
      <div class="stat"><div class="v" id="defl">—</div><div class="k">tip deflection</div></div>
      <div class="stat"><div class="v" id="eb">—</div><div class="k">beam-theory δ</div></div>
      <div class="stat"><div class="v" id="agree">—</div><div class="k">agreement</div></div>
    </div>
    <p class="read dim" style="margin-top:8px">the blue ring is where <b>Euler-Bernoulli beam theory</b> predicts the tip lands — the bent arm meets it for small loads</p>
  </div>

  <h2><span class="n">03 — it is a beam</span>The same law as a diving board</h2>
  <p>For a gentle load the soft arm is nothing exotic: it is a cantilever beam, and it obeys the oldest result in the book — the tip deflects by <span style="font-family:var(--mono);color:var(--goldb)">δ = F L³ / (3 EI)</span>. The blue marker is that prediction; the arm's tip sits on it. On load, this page hung a small weight and compared the strain model to the formula:</p>
  <table>
    <tr><td>tip deflection — strain model, on device</td><td id="scDefl">…</td></tr>
    <tr><td>Euler-Bernoulli δ = F L³ / (3 EI)</td><td id="scEb">…</td></tr>
    <tr><td>agreement</td><td id="scErr">…</td></tr>
    <tr><td>verdict</td><td id="scVerdict">…</td></tr>
  </table>
  <p>The strain description is not a fudge — in the small-deflection limit it reproduces beam theory exactly, and past that limit it keeps working where the linear formula (cube of the length, small angles) breaks down. Pull the load hard and the arm curls well beyond what <span style="font-family:var(--mono)">δ = FL³/3EI</span> would say; the strain model still holds because it never assumed the deflection was small.</p>

  <h2><span class="n">04 — stiffness</span>The dial between soft and stiff</h2>
  <div class="callout">Slide <b>EI</b> down and the same load bends the arm far more — deflection scales as <span style="font-family:var(--mono);color:var(--goldb)">1/EI</span>. That dial is the whole design tension of soft robotics: a compliant arm is <b>safe</b> (it yields on contact instead of injuring), <b>adaptive</b> (it conforms to what it grasps), and <b>robust</b> (it survives impacts) — but the softer it is, the harder it is to place its tip precisely, because the body itself deflects under every load. Choosing EI is choosing where on that spectrum the robot lives.</div>

  <h2><span class="n">05 — the point</span>A robot you compute like a spring</h2>
  <div class="verdict">
    <div class="big">Curvature is the configuration.</div>
    <p>Trade joint angles for a strain field and a continuous, infinite-DOF body becomes a finite object you can simulate, load, and control — reducing, when it must, to the beam theory engineers have trusted for two centuries.</p>
  </div>
  <p>This is what makes soft robots tractable rather than mystifying. The strain-based model unifies the compliant and the rigid — a stiff link is just a section that barely strains — so the same machinery describes a steel arm and a silicone tentacle. It is a distinct member of this series: where the rigid-body chapters found the one quantity that governs a jointed machine, this one finds it for a machine with no joints at all.</p>

  <p class="note"><b>What you just drove:</b> the <span style="color:var(--soft)">CosseratRod</span> from <span style="color:var(--soft)">ferromotion-core</span> — a planar piecewise-constant-strain rod — compiled to WebAssembly, the same code the native tools link against. Equilibrium under load is the minimizer of bending energy minus load work, found by the analytic strain gradient; nothing precomputed, every drag re-solves it.<br/><br/>
  <b>Verified in the library:</b> the cantilever tip deflection matches Euler-Bernoulli δ=FL³/(3EI) to under 2%; a pure moment bends it to a uniform circular arc of radius EI/M; deflection scales as 1/EI; arc length is preserved; the analytic strain gradient matches finite differences. Each is a test in <span style="color:var(--soft)">cargo test</span>, not a claim in prose. See also <a href="/assets/sims/morphological-computation">ch.1</a> · <a href="/assets/sims/safety-filter">ch.3</a> · <a href="/assets/sims/capture-point">ch.8</a> · <a href="/assets/sims/reactive-motion">ch.10</a> · <a href="/assets/sims/textbook">the full textbook</a>.<br/><br/>
  <b>Institute for Physical AI</b> · <a href="https://github.com/dcharlot-physicalai-bmi/ferromotion">the Rust library</a> · <a href="https://crates.io/crates/ferromotion">crates.io</a></p>
</div>
<script type="module">
${glue}
const WASM_B64="${wasmB64}";
${APP}
</script></body></html>`;

const outFile = path.join(__dirname, "..", "..", "v2", "public", "assets", "sims", "soft-robots.html");
fs.writeFileSync(outFile, html);
console.log(`wrote ${outFile} (${(html.length / 1024).toFixed(0)} KB, wasm ${(wasmB64.length / 1024).toFixed(0)} KB b64)`);

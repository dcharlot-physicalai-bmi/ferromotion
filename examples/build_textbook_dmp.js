// Assemble the interactive-textbook chapter on movement primitives: inline the wasm-bindgen glue +
// base64-embed the wasm, so the page runs the real ferromotion Dmp on-device. The reader draws a motion
// once, then drags the goal and watches the learned SHAPE reproduce to the new target — always
// arriving, whatever was drawn. Same self-contained pattern as chapters 1–4.
const fs = require("fs");
const path = require("path");

const pkg = path.join(__dirname, "..", "crates", "ferromotion-wasm", "pkg");
const glue = fs.readFileSync(path.join(pkg, "ferromotion_wasm.js"), "utf8");
const wasmB64 = fs.readFileSync(path.join(pkg, "ferromotion_wasm_bg.wasm")).toString("base64");

const APP = String.raw`
function b64ToBytes(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
const fmt=(x,n=2)=>Number(x).toFixed(n);
const TAU=Math.PI*2;

let lab, demoX=[], demoY=[], goal=[2,1.2], rollout=[], drawing=false, draggingGoal=false, tauScale=1.0;
let execT=0;

const cv=document.getElementById("stage");
function tf(){ const r=cv.getBoundingClientRect(); return {ox:r.width*0.5, oy:r.height*0.62, s:Math.min(r.width,r.height)*0.20, W:r.width, H:r.height}; }
function w2s(x,y){ const t=tf(); return [t.ox+x*t.s, t.oy - y*t.s]; }
function s2w(px,py){ const t=tf(); return [(px-t.ox)/t.s, -(py-t.oy)/t.s]; }

/* preset demonstrations (distinct endpoints on both axes so the DMP can scale them) */
function presetDemo(kind){
  const N=90, xs=[], ys=[];
  for(let i=0;i<N;i++){ const s=i/(N-1);
    if(kind==='arc'){ xs.push(-1.6+3.2*s); ys.push(-0.8+2.0*s + Math.sin(Math.PI*s)*1.1); }
    else if(kind==='scoop'){ xs.push(-1.8+3.6*s); ys.push(1.0 - Math.sin(Math.PI*s)*1.9 + 0.2*s); }
    else if(kind==='hook'){ const a=Math.PI*1.3*s; xs.push(-1.7+2.6*s + Math.sin(a*1.4)*0.5); ys.push(-1.2+2.2*s + Math.cos(a)*0.3); }
  }
  loadDemo(xs,ys);
  document.querySelectorAll('.seg button').forEach(b=>b.classList.toggle('on', b.dataset.k===kind));
}
function loadDemo(xs,ys){
  demoX=xs.slice(); demoY=ys.slice();
  lab.fit(new Float64Array(demoX), new Float64Array(demoY), 0.02);
  goal=[lab.demo_goal_x(), lab.demo_goal_y()];
  recompute();
}
function recompute(){
  if(!lab.is_fitted()) return;
  rollout=Array.from(lab.rollout(demoX[0], demoY[0], goal[0], goal[1], tauScale));
  execT=0; syncReadouts();
}

function draw(){
  const ctx=cv.getContext("2d"); const t=tf();
  const dpr=Math.min(devicePixelRatio||1,2); cv.width=t.W*dpr; cv.height=t.H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,t.W,t.H);
  // grid
  ctx.strokeStyle="rgba(120,140,180,.07)"; ctx.lineWidth=1;
  for(let g=-3;g<=3;g++){ let a=w2s(g,-2.2),b=w2s(g,2.6); ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
    a=w2s(-3.4,g);b=w2s(3.4,g); ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke(); }
  // demonstration (faint)
  if(demoX.length>1){ ctx.beginPath(); demoX.forEach((x,i)=>{const p=w2s(x,demoY[i]); i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]);});
    ctx.strokeStyle="rgba(217,180,94,.32)"; ctx.lineWidth=3; ctx.setLineDash([5,5]); ctx.stroke(); ctx.setLineDash([]);
    const s0=w2s(demoX[0],demoY[0]); ctx.beginPath();ctx.arc(s0[0],s0[1],4,0,TAU); ctx.fillStyle="rgba(217,180,94,.6)"; ctx.fill();
  }
  // rollout (bright)
  const n=rollout.length/2;
  if(n>1){ ctx.beginPath(); for(let i=0;i<n;i++){const p=w2s(rollout[2*i],rollout[2*i+1]); i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]);}
    ctx.strokeStyle="#7dd3a0"; ctx.lineWidth=2.5; ctx.stroke();
    // executor dot
    const k=Math.min(n-1, Math.floor(execT));
    const ep=w2s(rollout[2*k],rollout[2*k+1]);
    ctx.beginPath(); ctx.arc(ep[0],ep[1],7,0,TAU); ctx.fillStyle="#161f3a"; ctx.strokeStyle="#7dd3a0"; ctx.lineWidth=2.5; ctx.fill(); ctx.stroke();
  }
  // goal marker (draggable)
  const gp=w2s(goal[0],goal[1]);
  ctx.beginPath(); ctx.arc(gp[0],gp[1],10,0,TAU); ctx.strokeStyle= draggingGoal ? "#f0cf82":"#f0cf82"; ctx.lineWidth=2.5; ctx.stroke();
  ctx.beginPath(); ctx.arc(gp[0],gp[1],2.5,0,TAU); ctx.fillStyle="#f0cf82"; ctx.fill();
  ctx.fillStyle="rgba(240,207,130,.85)"; ctx.font="500 10px ui-monospace,monospace"; ctx.fillText("goal · drag me", gp[0]+14, gp[1]-8);
  // legend
  ctx.font="500 10px ui-monospace,monospace";
  ctx.fillStyle="rgba(217,180,94,.7)"; ctx.fillText("- - demonstration (shown once)", 12, t.H-24);
  ctx.fillStyle="#7dd3a0"; ctx.fillText("— what it does now", 12, t.H-10);
}

function syncReadouts(){
  const n=rollout.length/2; if(n<2) return;
  const ex=rollout[2*(n-1)], ey=rollout[2*(n-1)+1];
  const err=Math.hypot(ex-goal[0], ey-goal[1]);
  const a=document.getElementById("arrives");
  a.textContent = err<0.03 ? "✓ arrives" : fmt(err,2)+" off";
  a.style.color = err<0.03 ? "#7dd3a0" : "#e8836f";
  document.getElementById("goalErr").textContent=err.toExponential(1);
  document.getElementById("basis").textContent="20";
}

/* interaction: draw a demo, or drag the goal */
function hitGoal(px,py){ const g=w2s(goal[0],goal[1]); return Math.hypot(px-g[0],py-g[1])<16; }
cv.addEventListener("pointerdown",e=>{ const r=cv.getBoundingClientRect(); const px=e.clientX-r.left,py=e.clientY-r.top;
  if(hitGoal(px,py)){ draggingGoal=true; cv.setPointerCapture(e.pointerId); return; }
  // start drawing a new demo
  drawing=true; demoX=[]; demoY=[]; const w=s2w(px,py); demoX.push(w[0]); demoY.push(w[1]);
  document.querySelectorAll('.seg button').forEach(b=>b.classList.remove('on'));
  cv.setPointerCapture(e.pointerId);
});
cv.addEventListener("pointermove",e=>{ const r=cv.getBoundingClientRect(); const px=e.clientX-r.left,py=e.clientY-r.top;
  if(draggingGoal){ goal=s2w(px,py); recompute(); draw(); return; }
  if(drawing){ const w=s2w(px,py); const lx=demoX[demoX.length-1],ly=demoY[demoY.length-1];
    if(Math.hypot(w[0]-lx,w[1]-ly)>0.06){ demoX.push(w[0]); demoY.push(w[1]); draw(); } }
});
cv.addEventListener("pointerup",()=>{
  if(draggingGoal){ draggingGoal=false; return; }
  if(drawing){ drawing=false;
    if(demoX.length>12){ loadDemo(demoX,demoY); } // fit the freshly-drawn motion
  }
});

document.querySelectorAll('.seg button').forEach(b=> b.onclick=()=>presetDemo(b.dataset.k));
document.getElementById("tau").oninput=e=>{ tauScale=+e.target.value; document.getElementById("tauVal").textContent=fmt(tauScale,1)+"×"; recompute(); };

/* the page proves structural convergence on load: a far, never-shown goal is still reached */
function selfCheck(){
  const t=new DmpLab(20);
  const N=90, xs=[], ys=[];
  for(let i=0;i<N;i++){ const s=i/(N-1); xs.push(2*s); ys.push(1.2*s+Math.sin(Math.PI*s)*0.7); }
  t.fit(new Float64Array(xs), new Float64Array(ys), 0.02);
  let worst=0;
  for(const [gx,gy] of [[3.5,-2.0],[-1.5,2.8],[0.3,-1.7]]){
    const p=Array.from(t.rollout(0,0,gx,gy,1.0)); const n=p.length/2;
    worst=Math.max(worst, Math.hypot(p[2*(n-1)]-gx, p[2*(n-1)+1]-gy));
  }
  document.getElementById("scWorst").textContent=worst.toExponential(2);
  document.getElementById("scVerdict").textContent = worst<0.03 ? "every goal reached — convergence is structural" : "unexpected";
}

function frame(){
  const n=rollout.length/2;
  if(n>1){ execT+=0.9; if(execT>=n-1+30) execT=0; } // trace, pause, loop
  draw();
  requestAnimationFrame(frame);
}

async function main(){
  await __wbg_init(b64ToBytes(WASM_B64));
  lab=new DmpLab(20);
  selfCheck(); presetDemo('arc');
  window.__dmp={lab:()=>lab, setGoal:(x,y)=>{goal=[x,y];recompute();}, preset:presetDemo, rollout:()=>rollout, goalOf:()=>goal};
  window.__textbook_ready=true;
  addEventListener("resize",draw); frame();
}
main();
`;

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Show it once — ferromotion textbook</title>
<meta name="description" content="An interactive chapter on Dynamic Movement Primitives: learn a motion from a single demonstration and replay its shape to any new goal, always arriving. Runs the real Rust DMP on-device."/>
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
#stage{width:100%;height:440px;cursor:crosshair;border-radius:10px;background:radial-gradient(700px 440px at 50% 45%,#0e1730,#0b1122)}
.read{font-family:var(--mono);font-size:.76rem;color:var(--soft);margin:10px 0 0;text-align:center}.read b{color:var(--ink)}.dim{color:var(--dim)}
.ctl{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:14px;justify-content:center}
button{background:linear-gradient(180deg,#d9b45e,#a9832f);color:#161200;border:0;border-radius:9px;padding:8px 16px;font:700 .82rem var(--sans);cursor:pointer}
.seg{display:inline-flex;background:#0d1428;border:1px solid var(--line);border-radius:9px;overflow:hidden;flex-wrap:wrap}
.seg button{background:transparent;color:var(--soft);border:0;border-radius:0;padding:8px 13px;font:600 .76rem var(--sans)}
.seg button.on{background:linear-gradient(180deg,rgba(217,180,94,.22),rgba(217,180,94,.06));color:var(--goldb)}
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
  <p class="kicker"><span style="color:var(--goldb);font-size:1.05rem">&#934;</span><span>ferromotion · textbook · chapter 5</span>
    <span class="badge">rust → wasm · on-device</span></p>
  <h1>Show it once</h1>
  <p class="lede">Teach a robot a motion by demonstrating it a single time, and have it reproduce not just that motion but its <i>shape</i> — to targets you never showed it, always arriving. This page learns and replays with the same Rust movement primitive the native tools use.</p>

  <h2><span class="n">01 — the problem</span>One demonstration, many goals</h2>
  <p>The most natural way to give a robot a skill is to show it once — guide its hand through a pour, a reach, a wipe. But a recording is brittle: replay the exact positions and the first time the cup sits somewhere new, the motion misses. What you want is for the robot to keep the <i>character</i> of the motion — the arc of the pour, the curl of the wipe — while retargeting it to wherever the cup actually is.</p>

  <h2><span class="n">02 — the primitive</span>A spring, bent into shape</h2>
  <p>A Dynamic Movement Primitive is a stable spring-damper pulling the hand toward the goal, plus a learned <b>forcing term</b> that bends the straight pull into the demonstrated shape. The trick is that the forcing term is driven by a <b>phase</b> — a clock that runs once from 1 to 0 — and is multiplied by that phase, so it fades out on its own before the motion ends.</p>
  <p>Draw a motion in the box below with your cursor — any squiggle — then release. The dashed gold curve is what you drew; the green curve is the primitive reproducing it. Or pick a preset.</p>
  <div class="fig">
    <canvas id="stage"></canvas>
    <div class="ctl" style="margin-top:12px">
      <span class="seg"><button data-k="arc" class="on">Arc</button><button data-k="scoop">Scoop</button><button data-k="hook">Hook</button></span>
      <span class="dim mono" style="font-size:.7rem">— or draw your own —</span>
    </div>
    <div class="ctl">
      <label>speed τ <input type="range" id="tau" min="0.5" max="3" step="0.1" value="1"/> <span id="tauVal">1.0×</span></label>
    </div>
    <div class="stats">
      <div class="stat"><div class="v" id="arrives" style="font-size:.82rem">—</div><div class="k">at the goal</div></div>
      <div class="stat"><div class="v" id="goalErr">—</div><div class="k">distance to goal</div></div>
      <div class="stat"><div class="v" id="basis">—</div><div class="k">basis functions</div></div>
    </div>
    <p class="read dim" style="margin-top:8px"><b>drag the gold goal marker</b> anywhere — the learned shape follows it and always arrives</p>
  </div>

  <h2><span class="n">03 — the guarantee</span>It cannot fail to arrive</h2>
  <p>Drag the goal to the far corner, somewhere the demonstration never went. The motion still lands on it. This is not luck and not the learning working well — it is structural. Because the forcing term is switched off by the phase clock, by the end of the motion only the spring-damper is left, and a spring-damper has exactly one destination: the goal.</p>
  <div class="callout">This is the quiet virtue of the design. You can learn the forcing term with any method, fit it to a messy human demonstration, get the weights <b>wrong</b> — and the motion still arrives, because convergence was never the learner's job. The shape is learned; the arrival is guaranteed by construction. On load, this page fired the primitive at three far-flung goals it was never taught:</p>
  <table>
    <tr><td>largest miss over three never-shown goals</td><td id="scWorst">…</td></tr>
    <tr><td>verdict</td><td id="scVerdict">…</td></tr>
  </table>

  <h2><span class="n">04 — generalization</span>The same motion, somewhere new</h2>
  <p>Drag the goal outward along the direction of the demonstration and the whole shape scales with it — twice as far, twice as big, same character. Drag it off to the side and the shape shears, because the primitive scales each axis by its own displacement; that shear is a real property, not a bug, and it is why a DMP is retargeted, not merely translated. The <b>speed</b> slider stretches the motion in time without touching its path at all — the phase clock, not the wall clock, drives the shape.</p>

  <h2><span class="n">05 — the point</span>A skill you can keep</h2>
  <div class="verdict">
    <div class="big">Learn the shape; the arrival is free.</div>
    <p>A single demonstration becomes a reusable, retargetable skill — one that generalizes to new goals and rescales in time, and whose safety-critical property, actually reaching the target, is guaranteed by the structure rather than trusted to the learning.</p>
  </div>
  <p>That split is why movement primitives endure. The learned part carries the style of the motion and can be as rough as a one-shot human demo; the guaranteed part carries the thing you cannot afford to get wrong. It is the same bargain as the safety filter two chapters back — put the hard guarantee in simple, provable structure and let the learned component be creative inside it — and it is a good bargain to make whenever a robot has to learn from people and still be relied on.</p>

  <p class="note"><b>What you just drove:</b> the <span style="color:var(--soft)">Dmp</span> from <span style="color:var(--soft)">ferromotion-control</span> — two of them, one per axis — compiled to WebAssembly, the same code the native tools link against. Weights are fit from your demonstration by locally weighted regression; every drag re-rolls the primitive live. Nothing is precomputed.<br/><br/>
  <b>Verified in the library:</b> it reproduces the demonstration it was shown · the goal is a structural attractor for any goal and even for scrambled weights · a rescaled goal reproduces the same shape · τ rescales duration without changing the path. Each is a test in <span style="color:var(--soft)">cargo test</span>, not a claim in prose. See also <a href="/assets/sims/morphological-computation">ch.1</a> · <a href="/assets/sims/algebraic-connectivity">ch.2</a> · <a href="/assets/sims/safety-filter">ch.3</a> · <a href="/assets/sims/force-closure">ch.4</a>.<br/><br/>
  <b>Institute for Physical AI</b> · <a href="https://github.com/dcharlot-physicalai-bmi/ferromotion">the Rust library</a> · <a href="https://crates.io/crates/ferromotion">crates.io</a></p>
</div>
<script type="module">
${glue}
const WASM_B64="${wasmB64}";
${APP}
</script></body></html>`;

const outFile = path.join(__dirname, "..", "..", "v2", "public", "assets", "sims", "movement-primitives.html");
fs.writeFileSync(outFile, html);
console.log(`wrote ${outFile} (${(html.length / 1024).toFixed(0)} KB, wasm ${(wasmB64.length / 1024).toFixed(0)} KB b64)`);

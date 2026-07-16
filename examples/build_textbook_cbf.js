// Assemble the interactive-textbook chapter on control barrier functions: inline the wasm-bindgen glue
// + base64-embed the wasm, so the page runs the real ferromotion CbfFilter on-device. The reader
// steers a robot and finds that a safety filter physically will not let them drive it into a hazard —
// and alters the command as little as possible to do so. Same self-contained pattern as chapters 1–2.
const fs = require("fs");
const path = require("path");

const pkg = path.join(__dirname, "..", "crates", "ferromotion-wasm", "pkg");
const glue = fs.readFileSync(path.join(pkg, "ferromotion_wasm.js"), "utf8");
const wasmB64 = fs.readFileSync(path.join(pkg, "ferromotion_wasm_bg.wasm")).toString("base64");

const APP = String.raw`
function b64ToBytes(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
const fmt=(x,n=2)=>Number(x).toFixed(n);
const TAU=Math.PI*2;

let lab, forcing=false, trail=[];
const HAZ=[[0,0,1.0],[2.1,1.1,0.7],[-1.7,1.4,0.6]];
function build(){
  lab=new CbfLab();
  lab.set_pos(-3.2,-2.2); lab.set_gain(2.2); lab.set_alpha(4);
  HAZ.forEach(h=>lab.add_obstacle(h[0],h[1],h[2]));
  lab.set_target(-3.2,-2.2); trail=[];
}

const cv=document.getElementById("stage");
function tf(){ const r=cv.getBoundingClientRect(); return {ox:r.width/2, oy:r.height/2, s:Math.min(r.width,r.height)/9, W:r.width, H:r.height}; }
function w2s(x,y){ const t=tf(); return [t.ox+x*t.s, t.oy+y*t.s]; }
function s2w(px,py){ const t=tf(); return [(px-t.ox)/t.s, (py-t.oy)/t.s]; }

function draw(){
  const ctx=cv.getContext("2d"); const t=tf();
  const dpr=Math.min(devicePixelRatio||1,2); cv.width=t.W*dpr; cv.height=t.H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,t.W,t.H);
  // hazards
  HAZ.forEach(h=>{ const [sx,sy]=w2s(h[0],h[1]); const r=h[2]*t.s;
    ctx.beginPath(); ctx.arc(sx,sy,r,0,TAU); ctx.fillStyle="rgba(232,131,111,.14)"; ctx.fill();
    ctx.strokeStyle="rgba(232,131,111,.75)"; ctx.lineWidth=2; ctx.stroke();
    ctx.beginPath(); ctx.arc(sx,sy,4,0,TAU); ctx.fillStyle="rgba(232,131,111,.55)"; ctx.fill();
  });
  // target
  const [tx,ty]=w2s(lab_target[0],lab_target[1]);
  ctx.strokeStyle="rgba(174,182,204,.6)"; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(tx-7,ty);ctx.lineTo(tx+7,ty);ctx.moveTo(tx,ty-7);ctx.lineTo(tx,ty+7); ctx.stroke();
  // trail
  const [rx,ry]=w2s(lab.x(),lab.y());
  trail.push([rx,ry]); if(trail.length>140) trail.shift();
  ctx.beginPath(); trail.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));
  ctx.strokeStyle="rgba(217,180,94,.28)"; ctx.lineWidth=2; ctx.stroke();
  // command arrows (nominal = what you asked, filtered = what it does)
  const nx=lab.nominal_x(), ny=lab.nominal_y(), fx=lab.filtered_x(), fy=lab.filtered_y();
  const sc=0.42*t.s;
  arrow(ctx, rx,ry, rx+nx*sc, ry+ny*sc, "rgba(240,207,130,.85)", true);   // nominal, dashed gold
  arrow(ctx, rx,ry, rx+fx*sc, ry+fy*sc, "#7dd3a0", false);                 // filtered, solid green
  // robot
  ctx.beginPath(); ctx.arc(rx,ry,8,0,TAU); ctx.fillStyle="#161f3a";
  ctx.strokeStyle= lab.correction()>1e-3 ? "#7dd3a0" : "#d9b45e"; ctx.lineWidth=2.5; ctx.fill(); ctx.stroke();
  // legend
  ctx.font="500 10px ui-monospace,monospace";
  ctx.fillStyle="rgba(240,207,130,.9)"; ctx.fillText("— command you gave", 12, t.H-24);
  ctx.fillStyle="#7dd3a0"; ctx.fillText("— what the filter allows", 12, t.H-10);
}
function arrow(ctx,x0,y0,x1,y1,color,dashed){
  const a=Math.atan2(y1-y0,x1-x0), L=Math.hypot(x1-x0,y1-y0); if(L<3) return;
  ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=2.2;
  ctx.setLineDash(dashed?[5,4]:[]); ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y1); ctx.stroke(); ctx.setLineDash([]);
  ctx.beginPath(); ctx.moveTo(x1,y1);
  ctx.lineTo(x1-8*Math.cos(a-0.4), y1-8*Math.sin(a-0.4));
  ctx.lineTo(x1-8*Math.cos(a+0.4), y1-8*Math.sin(a+0.4));
  ctx.closePath(); ctx.fill();
}

let lab_target=[-3.2,-2.2];
function setTargetFromEvent(e){ const r=cv.getBoundingClientRect(); lab_target=s2w(e.clientX-r.left, e.clientY-r.top); lab.set_target(lab_target[0],lab_target[1]); }
cv.addEventListener("pointermove",e=>{ if(!forcing) setTargetFromEvent(e); });
cv.addEventListener("pointerdown",e=>{ forcing=false; setTargetFromEvent(e); });

document.getElementById("force").onclick=()=>{
  // aim through the nearest hazard at a point on its FAR side, so the robot has to slide around the
  // rim to chase it — showing refusal, sliding, and minimal intervention at once
  forcing=true;
  const rx=lab.x(),ry=lab.y();
  let best=HAZ[0],bd=1e9; HAZ.forEach(h=>{const d=Math.hypot(h[0]-rx,h[1]-ry); if(d<bd){bd=d;best=h;}});
  const dx=best[0]-rx, dy=best[1]-ry, L=Math.hypot(dx,dy)||1;
  lab_target=[best[0]+dx/L*best[2]*1.7, best[1]+dy/L*best[2]*1.7];
  lab.set_target(lab_target[0],lab_target[1]);
};
document.getElementById("alpha").oninput=e=>{ lab.set_alpha(+e.target.value); document.getElementById("alphaVal").textContent=fmt(+e.target.value,1); };
document.getElementById("reset").onclick=()=>{ build(); lab_target=[-3.2,-2.2]; };

function syncReadouts(){
  const mh=lab.min_h(), corr=lab.correction();
  document.getElementById("margin").textContent=fmt(mh,3);
  const mb=document.getElementById("marginBar");
  mb.style.width=Math.max(0,Math.min(100,mh/2.5*100))+"%";
  mb.style.background = mh<0 ? "#e8836f" : (corr>1e-3 ? "#f0cf82" : "#7dd3a0");
  document.getElementById("corr").textContent=fmt(corr,3);
  const st=document.getElementById("filterState");
  if(corr>1e-3){ st.textContent="filter active — deflecting"; st.style.color="#f0cf82"; }
  else { st.textContent="command passing through untouched"; st.style.color="#7dd3a0"; }
}

function frame(){
  for(let i=0;i<6;i++) lab.step(4e-3);
  draw(); syncReadouts();
  requestAnimationFrame(frame);
}

/* ---------- the page proves its own guarantee, on load ---------- */
function selfCheck(){
  const t=new CbfLab();
  t.set_pos(-2.0,0.0); t.add_obstacle(0,0,1.0); t.set_alpha(4); t.set_gain(2.5);
  t.set_target(0,0); // aim straight into the hazard centre
  let worst=Infinity;
  for(let i=0;i<20000;i++){ t.step(1e-3); worst=Math.min(worst,t.min_h()); }
  document.getElementById("scWorst").textContent=worst.toExponential(2);
  document.getElementById("scFinal").textContent=fmt(t.min_h(),4);
  document.getElementById("scVerdict").textContent = worst>-1e-6 ? "NEVER — the barrier held every step" : "VIOLATED";
}

async function main(){
  await __wbg_init(b64ToBytes(WASM_B64));
  build(); selfCheck();
  window.__cbf={lab:()=>lab, force:()=>document.getElementById("force").click()};
  window.__textbook_ready=true;
  addEventListener("resize",draw);
  frame();
}
main();
`;

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>The command it will not obey — ferromotion textbook</title>
<meta name="description" content="An interactive chapter on control barrier functions: a safety filter that minimally corrects any command so a robot physically cannot enter a hazard. Runs the real Rust CBF filter on-device."/>
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
#stage{width:100%;height:420px;cursor:crosshair;border-radius:10px;background:radial-gradient(700px 420px at 50% 42%,#0e1730,#0b1122)}
.read{font-family:var(--mono);font-size:.76rem;color:var(--soft);margin:10px 0 0;text-align:center}
.read b{color:var(--ink);font-weight:600}.dim{color:var(--dim)}
.ctl{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:14px;justify-content:center}
button{background:linear-gradient(180deg,#d9b45e,#a9832f);color:#161200;border:0;border-radius:9px;padding:8px 16px;font:700 .82rem var(--sans);cursor:pointer}
button.ghost{background:transparent;color:var(--soft);border:1px solid var(--line)}
button:active{transform:translateY(1px)}
label{font-family:var(--mono);font-size:.72rem;color:var(--soft);display:flex;align-items:center;gap:8px}
input[type=range]{accent-color:var(--gold);width:130px}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}
.stat{background:#0d1428;border:1px solid var(--line);border-radius:10px;padding:9px 8px;text-align:center}
.stat .v{font-family:var(--mono);font-size:1rem;font-weight:700;color:var(--goldb)}
.stat .k{font-family:var(--mono);font-size:.58rem;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);margin-top:2px}
.bar{height:6px;border-radius:3px;background:#0d1428;overflow:hidden;margin-top:8px;border:1px solid var(--line)}
.bar>div{height:100%;width:60%;background:var(--green);transition:width .1s linear,background .1s linear}
.callout{border-left:2px solid var(--gold);padding:2px 0 2px 18px;margin:26px 0;color:var(--soft)}
.callout b{color:var(--goldb)}
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
  <p class="kicker"><span style="color:var(--goldb);font-size:1.05rem">&#934;</span><span>ferromotion · textbook · chapter 3</span>
    <span class="badge">rust → wasm · on-device</span></p>
  <h1>The command it will not obey</h1>
  <p class="lede">A safety layer that sits between any controller and the robot, and changes the command only when it must — so the robot physically cannot be driven into a hazard, whatever it is told. Everything below runs on your device, on the same Rust safety filter the native tools use.</p>

  <h2><span class="n">01 — the problem</span>A command you can't fully trust</h2>
  <p>Whatever is driving a robot — a hand controller, a planner, a learned policy — will occasionally command something unsafe. The usual fix is to make the controller more careful, and hope. But hope does not compose: a policy that is safe in training can be commanded into a wall the first time the world differs. We want a guarantee that does not depend on trusting the thing giving the orders.</p>
  <p>The idea is a <b>filter</b>. Let the controller ask for whatever it wants; between it and the motors, a thin layer corrects the command to the nearest one that keeps the robot safe — and, when the command is already safe, does nothing at all.</p>

  <h2><span class="n">02 — the barrier</span>One function that means "safe"</h2>
  <p>Pick a function <span style="font-family:var(--mono);color:var(--goldb)">h(x)</span> that is positive where the robot is safe and zero at the edge of danger — here, distance to a hazard minus its radius. The filter enforces one inequality, <span style="font-family:var(--mono);color:var(--goldb)">ḣ + α·h ≥ 0</span>: near the boundary (small h) the robot's approach speed must shrink to zero, so it can never cross. That single condition is linear in the command, so enforcing it is cheap.</p>
  <p>Drive the robot below — <b>move your cursor</b> and it follows. The gold arrow is the command you gave; the green arrow is what the filter allows. Steer straight at a hazard.</p>
  <div class="fig">
    <canvas id="stage"></canvas>
    <div class="ctl">
      <button id="force">Aim into the hazard</button>
      <button id="reset" class="ghost">Reset</button>
      <label>berth α <input type="range" id="alpha" min="1" max="14" step="0.5" value="4"/> <span id="alphaVal">4.0</span></label>
    </div>
    <div class="stats">
      <div class="stat"><div class="v" id="margin">—</div><div class="k">safety margin h</div></div>
      <div class="stat"><div class="v" id="corr">—</div><div class="k">command changed by</div></div>
      <div class="stat"><div class="v" id="filterState" style="font-size:.72rem;color:var(--green)">—</div><div class="k">filter</div></div>
    </div>
    <div class="bar"><div id="marginBar"></div></div>
    <p class="read dim" style="margin-top:8px">the safety margin bar never reaches zero — press <b>Aim into the hazard</b> and watch the robot refuse</p>
  </div>
  <p>It slides along the edge. The gold arrow keeps pointing into the hazard — you are still commanding it inward — but the green arrow bends to run along the boundary. The robot obeys you exactly as far as it safely can, and no further.</p>

  <h2><span class="n">03 — minimal intervention</span>As much of your command as it safely can</h2>
  <p>The filter solves <span style="font-family:var(--mono);color:var(--goldb)">min ½‖u − u<sub>nom</sub>‖²</span> subject to the barrier — the closest safe command to the one you asked for. Away from every hazard the two arrows are identical: the filter is invisible. Only when your command has an inward component does it remove exactly that component and nothing else, which is why the robot slides rather than stops.</p>
  <div class="callout">This is what makes a safety filter composable: it does not replace your controller, it defers to it. Any policy at all can drive — a hand controller, a planner, a black-box neural policy — and the guarantee holds regardless, because it is enforced on the command that actually reaches the motors, not on the thing that produced it. Turn <b>berth α</b> down for a wider, more cautious margin; up to let it commit later and graze closer.</p>

  <h2><span class="n">04 — the guarantee</span>Not "usually." Never.</h2>
  <p>On load, this page took a robot, aimed it dead-centre at a hazard, and integrated twenty thousand steps of a command pointing straight in — then recorded the smallest safety margin over the whole run:</p>
  <table>
    <tr><td>smallest safety margin h over 20,000 steps</td><td id="scWorst">…</td></tr>
    <tr><td>margin where it came to rest (on the boundary)</td><td id="scFinal">…</td></tr>
    <tr><td>did it ever enter the hazard?</td><td id="scVerdict">…</td></tr>
  </table>
  <p>The margin never goes negative — not as an average, not with high probability, but on every single step. Because the barrier condition is convex here, each discrete step lands on the safe side of its own linear prediction, so the discrete robot is at least as safe as the continuous guarantee promises.</p>

  <h2><span class="n">05 — the point</span>Safety as a filter, not a hope</h2>
  <div class="verdict">
    <div class="big">The guarantee lives on the command, not the controller.</div>
    <p>Whatever asks for the motion — teleop, a planner, a learned policy you did not write and cannot fully trust — the last thing it passes through is one small convex program that will not sign off on a command that crosses the barrier.</p>
  </div>
  <p>For physical AI this is the shape of the safety story worth betting on. You are not going to verify a large neural policy line by line, and you do not have to. You wrap it in a certified filter whose only job is to keep one inequality true, and you get a guarantee that holds no matter how the policy behaves. The controller can be as clever, as learned, as opaque as you like; the barrier is simple enough to prove correct, and it has the last word.</p>

  <p class="note"><b>What you just drove:</b> the <span style="color:var(--soft)">CbfFilter</span> and <span style="color:var(--soft)">CbfConstraint</span> from <span style="color:var(--soft)">ferromotion-control</span>, compiled to WebAssembly — the same code the native tools link against, not a reimplementation. Each hazard becomes a relative-degree-1 barrier; the filter is a small quadratic program (closed-form for a single active hazard, <span style="color:var(--soft)">clarabel</span> otherwise). Nothing is precomputed — the correction is solved every frame, and the guarantee above was integrated live on load.<br/><br/>
  <b>Verified in the library:</b> the robot cannot be driven into a hazard over 20,000 steps aimed straight in (min h ≥ 0) · a safe command passes through bit-for-bit unchanged · the correction is the exact orthogonal projection — only the inward component is removed, so tangential motion is untouched · a whole field of hazards is respected at once · larger α rides provably closer. Each is a test in <span style="color:var(--soft)">cargo test</span>, not a claim in prose. See also <a href="/assets/sims/morphological-computation">chapter 1</a> and <a href="/assets/sims/algebraic-connectivity">chapter 2</a>.<br/><br/>
  <b>Institute for Physical AI</b> · <a href="https://github.com/dcharlot-physicalai-bmi/ferromotion">the Rust library</a> · <a href="https://crates.io/crates/ferromotion">crates.io</a></p>
</div>
<script type="module">
${glue}
const WASM_B64="${wasmB64}";
${APP}
</script></body></html>`;

const outFile = path.join(__dirname, "..", "..", "v2", "public", "assets", "sims", "safety-filter.html");
fs.writeFileSync(outFile, html);
console.log(`wrote ${outFile} (${(html.length / 1024).toFixed(0)} KB, wasm ${(wasmB64.length / 1024).toFixed(0)} KB b64)`);

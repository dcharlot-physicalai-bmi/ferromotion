// Assemble the interactive-textbook chapter on capture-point walking: inline the wasm-bindgen glue +
// base64-embed the wasm, so the page runs the real ferromotion capture_point/DCM on-device. The reader
// pushes a falling body and drops the foot on the capture point to catch it, then watches the same
// idea chained into a walk. Same self-contained pattern as chapters 1–7.
const fs = require("fs");
const path = require("path");

const pkg = path.join(__dirname, "..", "crates", "ferromotion-wasm", "pkg");
const glue = fs.readFileSync(path.join(pkg, "ferromotion_wasm.js"), "utf8");
const wasmB64 = fs.readFileSync(path.join(pkg, "ferromotion_wasm_bg.wasm")).toString("base64");

const APP = String.raw`
function b64ToBytes(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
const fmt=(x,n=2)=>Number(x).toFixed(n);
const TAU=Math.PI*2;
const Z=0.9, G=9.81;

let lab, caught=false, fallen=false, running=true;
function reset(){ lab.set_state(0,0); lab.set_foot(0); caught=true; fallen=false; }

/* ---------- the catch ---------- */
const cv=document.getElementById("stage");
function map(){ const r=cv.getBoundingClientRect(); const s=(r.width-80)/3.0; // 3 m across
  return {s, ox:r.width*0.32, gy:r.height-42, W:r.width, H:r.height}; }
function X(x){ const m=map(); return m.ox + x*m.s; }
function Yg(){ return map().gy; }
function Yc(){ const m=map(); return m.gy - Z*m.s; } // CoM height on screen (leg length z)

function drawCatch(){
  const ctx=cv.getContext("2d"); const m=map();
  const dpr=Math.min(devicePixelRatio||1,2); cv.width=m.W*dpr; cv.height=m.H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,m.W,m.H);
  // ground
  ctx.strokeStyle="rgba(174,182,204,.4)"; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(10,m.gy); ctx.lineTo(m.W-10,m.gy); ctx.stroke();
  const cx=lab.com_x(), v=lab.com_vx(), foot=lab.foot_x(), cp=lab.capture_point();
  // capture-point marker (glowing target on the ground)
  const cpx=X(cp);
  ctx.setLineDash([4,4]); ctx.strokeStyle="rgba(240,207,130,.55)"; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(cpx,m.gy-Z*m.s-30); ctx.lineTo(cpx,m.gy); ctx.stroke(); ctx.setLineDash([]);
  ctx.beginPath(); ctx.arc(cpx,m.gy,7,0,TAU); ctx.strokeStyle="#f0cf82"; ctx.lineWidth=2.5; ctx.stroke();
  ctx.beginPath(); ctx.arc(cpx,m.gy,2.5,0,TAU); ctx.fillStyle="#f0cf82"; ctx.fill();
  ctx.fillStyle="rgba(240,207,130,.9)"; ctx.font="600 12px ui-monospace,monospace"; ctx.fillText("ξ  capture point", cpx-30, m.gy+20);
  // foot
  const fx=X(foot);
  ctx.fillStyle=caught?"#7dd3a0":(fallen?"#e8836f":"#8aa0c8");
  ctx.fillRect(fx-12,m.gy-4,24,7);
  // leg + CoM
  const comX=X(cx), comY=Yc();
  ctx.strokeStyle=caught?"rgba(125,211,160,.8)":(fallen?"rgba(232,131,111,.8)":"rgba(217,180,94,.8)"); ctx.lineWidth=4;
  ctx.beginPath(); ctx.moveTo(fx,m.gy-2); ctx.lineTo(comX,comY); ctx.stroke();
  ctx.beginPath(); ctx.arc(comX,comY,13,0,TAU); ctx.fillStyle="#161f3a"; ctx.strokeStyle=caught?"#7dd3a0":(fallen?"#e8836f":"#d9b45e"); ctx.lineWidth=3; ctx.fill(); ctx.stroke();
  // velocity arrow
  if(Math.abs(v)>0.02){ const ax=comX+v*m.s*0.5;
    ctx.strokeStyle="#f0cf82"; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(comX,comY-20); ctx.lineTo(ax,comY-20); ctx.stroke();
    const dir=Math.sign(v); ctx.beginPath(); ctx.moveTo(ax,comY-20); ctx.lineTo(ax-7*dir,comY-24); ctx.lineTo(ax-7*dir,comY-16); ctx.closePath(); ctx.fillStyle="#f0cf82"; ctx.fill(); }
  ctx.fillStyle="#727d99"; ctx.font="500 10px ui-monospace,monospace"; ctx.fillText("click the ground to plant the foot", 12, 20);
  if(caught){ ctx.fillStyle="#7dd3a0"; ctx.font="600 13px ui-sans-serif,system-ui"; ctx.fillText("balanced ✓", comX-30, comY-32); }
  if(fallen){ ctx.fillStyle="#e8836f"; ctx.font="600 13px ui-sans-serif,system-ui"; ctx.fillText("fell", comX-14, comY-32); }
  document.getElementById("cpVal").textContent=fmt(cp,3);
  document.getElementById("comV").textContent=fmt(v,2);
  const st=document.getElementById("catchState");
  st.textContent = caught?"balanced":(fallen?"fell over":"falling"); st.style.color=caught?"#7dd3a0":(fallen?"#e8836f":"#f0cf82");
}
cv.addEventListener("pointerdown",e=>{ const r=cv.getBoundingClientRect(); const m=map();
  const xw=(e.clientX-r.left-m.ox)/m.s;
  lab.set_foot(xw); caught=false; fallen=false; running=true;
});
document.getElementById("push").onclick=()=>{ lab.push(0.5); caught=false; fallen=false; running=true; };
document.getElementById("catchBtn").onclick=()=>{ lab.step_to_capture(); caught=false; fallen=false; running=true; };
document.getElementById("resetBtn").onclick=reset;

/* ---------- the walk ---------- */
let walk, wcom=[], wt=0, wtot=0, feet=[];
function planWalk(){
  walk=new WalkLab(Z,G);
  feet=[0.0,0.3,0.6,0.9,1.2,1.5,1.5];
  walk.plan(new Float64Array(feet),0.55);
  wcom=Array.from(walk.walk_com(0.02)); wtot=walk.total_time(); wt=0;
}
const wc=document.getElementById("walk");
function drawWalk(){
  const ctx=wc.getContext("2d"); const r=wc.getBoundingClientRect();
  const dpr=Math.min(devicePixelRatio||1,2); wc.width=r.width*dpr; wc.height=r.height*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  const W=r.width,H=r.height,gy=H-34; ctx.clearRect(0,0,W,H);
  const minx=-0.2,maxx=1.8, s=(W-60)/(maxx-minx), ox=30-minx*s;
  const PX=x=>ox+x*s, zc=Z*s*0.7;
  ctx.strokeStyle="rgba(174,182,204,.35)"; ctx.lineWidth=2; ctx.beginPath();ctx.moveTo(10,gy);ctx.lineTo(W-10,gy);ctx.stroke();
  // footstep marks
  feet.forEach((f,i)=>{ const fx=PX(f); ctx.fillStyle="rgba(138,160,200,.5)"; ctx.fillRect(fx-9,gy-3,18,6);
    ctx.fillStyle="#5b6680"; ctx.font="500 9px ui-monospace,monospace"; if(i<feet.length-1) ctx.fillText((i+1),fx-3,gy+15); });
  // current support foot
  const sf=walk.support_foot_at(wt), sfx=PX(sf);
  ctx.fillStyle="#8aa0c8"; ctx.fillRect(sfx-10,gy-4,20,7);
  // DCM reference (leading marker)
  const xi=walk.dcm_ref_at(wt), xix=PX(xi);
  ctx.beginPath(); ctx.arc(xix,gy-6,6,0,TAU); ctx.strokeStyle="#f0cf82"; ctx.lineWidth=2; ctx.stroke();
  ctx.fillStyle="rgba(240,207,130,.85)"; ctx.font="600 10px ui-monospace,monospace"; ctx.fillText("ξ leads", xix-16, gy-16);
  // CoM (following), arcing over the support foot
  const k=Math.min(wcom.length-1, Math.floor(wt/0.02)); const cxw=wcom[k];
  const cxx=PX(cxw), cyy=gy-zc;
  ctx.strokeStyle="rgba(125,211,160,.8)"; ctx.lineWidth=4; ctx.beginPath(); ctx.moveTo(sfx,gy-2); ctx.lineTo(cxx,cyy); ctx.stroke();
  ctx.beginPath(); ctx.arc(cxx,cyy,11,0,TAU); ctx.fillStyle="#161f3a"; ctx.strokeStyle="#7dd3a0"; ctx.lineWidth=3; ctx.fill(); ctx.stroke();
  // CoM trail
  ctx.beginPath(); for(let i=0;i<=k;i++){ const p=PX(wcom[i]); i?ctx.lineTo(p,gy-zc):ctx.moveTo(p,gy-zc); }
  ctx.strokeStyle="rgba(125,211,160,.2)"; ctx.lineWidth=2; ctx.stroke();
  ctx.font="500 10px ui-monospace,monospace"; ctx.fillStyle="#7dd3a0"; ctx.fillText("● centre of mass follows", 12, 16);
  ctx.fillStyle="rgba(240,207,130,.85)"; ctx.fillText("○ capture point / DCM leads", 12, 30);
}

function syncCatchReadouts(){}

/* self-check on load: stepping to the capture point brings the body to rest */
function selfCheck(){
  const t=new WalkLab(Z,G); t.set_state(0,0.6); t.step_to_capture(); const foot=t.foot_x();
  let ok=false;
  for(let i=0;i<30000;i++){ t.advance(1e-4); if(t.com_speed()<5e-3 && Math.abs(t.com_x()-foot)<5e-3){ok=true;break;} }
  document.getElementById("scFoot").textContent=fmt(foot,3)+" m";
  document.getElementById("scRest").textContent=ok?"rest ✓":"—";
  document.getElementById("scVerdict").textContent = ok ? "the foot on ξ brings the fall to a stop" : "unexpected";
}

function frame(){
  // catch sim
  if(running && !caught && !fallen){
    for(let i=0;i<40;i++){ lab.advance(1e-3); }
    const dx=lab.com_x()-lab.foot_x();
    if(lab.com_speed()<0.03 && Math.abs(dx)<0.05){ lab.set_state(lab.foot_x(),0); caught=true; }
    if(Math.abs(dx)>1.1){ fallen=true; }
  }
  drawCatch();
  // walk anim
  wt+=0.014; if(wt>wtot+0.5) wt=0;
  drawWalk();
  requestAnimationFrame(frame);
}

async function main(){
  await __wbg_init(b64ToBytes(WASM_B64));
  lab=new WalkLab(Z,G); reset(); selfCheck(); planWalk();
  window.__walk={lab:()=>lab, push:()=>lab.push(0.5), stepCapture:()=>{lab.step_to_capture();caught=false;fallen=false;running=true;}, setFoot:(f)=>{lab.set_foot(f);caught=false;fallen=false;running=true;}, caught:()=>caught, fallen:()=>fallen, cp:()=>lab.capture_point()};
  window.__textbook_ready=true;
  addEventListener("resize",()=>{drawCatch();drawWalk();});
  frame();
}
main();
`;

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Where to put your foot — ferromotion textbook</title>
<meta name="description" content="An interactive chapter on capture-point walking: the one point on the ground where a step brings a falling body to rest, and how chaining it makes a walk. Runs the real Rust DCM code on-device."/>
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
#stage{width:100%;height:300px;cursor:crosshair;border-radius:10px;background:radial-gradient(700px 300px at 40% 60%,#0e1730,#0b1122)}
#walk{width:100%;height:220px;border-radius:10px;background:radial-gradient(700px 220px at 50% 60%,#0e1730,#0b1122)}
.read{font-family:var(--mono);font-size:.76rem;color:var(--soft);margin:10px 0 0;text-align:center}.read b{color:var(--ink)}.dim{color:var(--dim)}
.ctl{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:14px;justify-content:center}
button{background:linear-gradient(180deg,#d9b45e,#a9832f);color:#161200;border:0;border-radius:9px;padding:8px 16px;font:700 .82rem var(--sans);cursor:pointer}
button.ghost{background:transparent;color:var(--soft);border:1px solid var(--line)}
button:active{transform:translateY(1px)}
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
  <p class="kicker"><span style="color:var(--goldb);font-size:1.05rem">&#934;</span><span>ferromotion · textbook · chapter 8</span>
    <span class="badge">rust → wasm · on-device</span></p>
  <h1>Where to put your foot</h1>
  <p class="lede">Walking is a fall, caught over and over. A body topples forward and a foot is thrown out to stop it — and there is exactly one place on the ground to put that foot. This page finds it, and turns it into a walk, with the same Rust code the native tools use.</p>

  <h2><span class="n">01 — the problem</span>A body that has to fall to move</h2>
  <p>Stand a broomstick on your palm and it topples; a walking robot is the same, a tall mass balanced over a small foot. To move, it lets itself fall in the direction it wants to go — and then it must catch that fall by stepping, or it hits the ground. Standing still and stepping are not different modes; standing still is just the special case where the fall to catch is zero.</p>

  <h2><span class="n">02 — the one point</span>The capture point</h2>
  <p>Model the body as a point mass on a springless leg — the linear inverted pendulum. Its balance is governed by one combination of position and speed, the <b>capture point</b> <span style="font-family:var(--mono);color:var(--goldb)">ξ = x + ẋ/ω</span>, where <span style="font-family:var(--mono);color:var(--goldb)">ω = √(g/z)</span> is set by the body's height. Plant the foot exactly on ξ and the body coasts to a perfect stop over it. Plant it short and the body keeps toppling past; plant it beyond and the body rocks back.</p>
  <p>Below, <b>push</b> the body to set it moving, then <b>click the ground</b> to plant the foot. The gold marker is the capture point — try to hit it.</p>
  <div class="fig">
    <canvas id="stage"></canvas>
    <div class="ctl">
      <button id="push">Push →</button>
      <button id="catchBtn">Step to ξ</button>
      <button id="resetBtn" class="ghost">Reset</button>
    </div>
    <div class="stats">
      <div class="stat"><div class="v" id="catchState" style="font-size:.82rem">—</div><div class="k">body</div></div>
      <div class="stat"><div class="v" id="cpVal">—</div><div class="k">capture point ξ (m)</div></div>
      <div class="stat"><div class="v" id="comV">—</div><div class="k">forward speed (m/s)</div></div>
    </div>
    <p class="read dim" style="margin-top:8px">press <b>Step to ξ</b> to drop the foot exactly on the capture point and watch the fall coast to a balanced stop</p>
  </div>

  <h2><span class="n">03 — why it works</span>Split the fall in two</h2>
  <p>The inverted pendulum looks unstable, and half of it is. Written in terms of the capture point the dynamics split cleanly into two first-order parts: <span style="font-family:var(--mono);color:var(--goldb)">ξ̇ = ω(ξ − foot)</span>, which runs <i>away</i> from the foot and is the unstable half, and <span style="font-family:var(--mono);color:var(--goldb)">ẋ = −ω(x − ξ)</span>, by which the centre of mass always chases the capture point and is stable for free.</p>
  <div class="callout">So all of balance lives in one scalar. The centre of mass takes care of itself — it converges to the capture point no matter what. The only thing you actually steer is ξ, and the only handle you have on it is <b>where you put the foot</b>. Put the foot on ξ and ξ stops moving; then the body glides to rest beneath it. Balance is not a whole-body problem, it is a one-number problem, and the number tells you exactly where to step.</p>

  <h2><span class="n">04 — the walk</span>Catch, and catch again</h2>
  <p>A walk is this catch, chained. Lay down the footsteps you want, and plan the capture-point path <i>backward</i> from a final resting stance — because ξ is unstable forward in time, it is stable backward — threading it through each footstep. Then just play it forward: the capture point leads from foot to foot, and the centre of mass follows it into a walk.</p>
  <div class="fig">
    <canvas id="walk"></canvas>
    <p class="read dim" style="margin-top:8px">the capture point (gold) hops ahead to each planned footstep; the centre of mass (green) chases it — that chase <i>is</i> the walk</p>
  </div>
  <p>On load, this page took a body moving forward at 0.6 m/s, dropped the foot on its capture point, and integrated the pendulum until it settled:</p>
  <table>
    <tr><td>capture point the foot was planted on</td><td id="scFoot">...</td></tr>
    <tr><td>did the body coast to a stop over it?</td><td id="scRest">...</td></tr>
    <tr><td>verdict</td><td id="scVerdict">...</td></tr>
  </table>

  <h2><span class="n">05 — the point</span>Balance is a scalar</h2>
  <div class="verdict">
    <div class="big">Steer one point on the ground.</div>
    <p>A tall, toppling, many-jointed machine reduces to a single unstable number and a single lever — where the next foot lands. Get that placement right and the whole body follows into balance, or into a stride.</p>
  </div>
  <p>This is why capture-point and DCM control run on so many walking robots: they turn the frightening problem of balancing a tall mass into the tractable one of placing a foot. The planner picks footsteps; the capture point says how to time and place them so the fall is always caught; the body, which converges to the capture point on its own, needs no further persuading. It is the locomotion member of the same family as the rest of this series — find the one quantity that governs the system, and control that.</p>

  <p class="note"><b>What you just drove:</b> <span style="color:var(--soft)">capture_point</span>, <span style="color:var(--soft)">dcm</span> and <span style="color:var(--soft)">plan_dcm</span> from <span style="color:var(--soft)">ferromotion-control</span>, compiled to WebAssembly — the same code the native tools link against. The catch integrates the linear inverted pendulum <span style="font-family:var(--mono)">ẍ = ω²(x − foot)</span>; the walk plans the DCM backward through the footsteps and lets the CoM converge to it. Nothing precomputed.<br/><br/>
  <b>Verified in the library:</b> the capture point equals x + ẋ/ω (and the DCM) · stepping onto it brings the body to rest over the foot · stepping short keeps it toppling · stepping past rocks it back · a backward-planned walk ends at rest over the final foot and leads the CoM forward. Each is a test in <span style="color:var(--soft)">cargo test</span>, not a claim in prose. See also <a href="/assets/sims/morphological-computation">ch.1</a> · <a href="/assets/sims/algebraic-connectivity">ch.2</a> · <a href="/assets/sims/safety-filter">ch.3</a> · <a href="/assets/sims/force-closure">ch.4</a> · <a href="/assets/sims/movement-primitives">ch.5</a> · <a href="/assets/sims/invariant-estimation">ch.6</a> · <a href="/assets/sims/time-optimal">ch.7</a>.<br/><br/>
  <b>Institute for Physical AI</b> · <a href="https://github.com/dcharlot-physicalai-bmi/ferromotion">the Rust library</a> · <a href="https://crates.io/crates/ferromotion">crates.io</a></p>
</div>
<script type="module">
${glue}
const WASM_B64="${wasmB64}";
${APP}
</script></body></html>`;

const outFile = path.join(__dirname, "..", "..", "v2", "public", "assets", "sims", "capture-point.html");
fs.writeFileSync(outFile, html);
console.log(`wrote ${outFile} (${(html.length / 1024).toFixed(0)} KB, wasm ${(wasmB64.length / 1024).toFixed(0)} KB b64)`);

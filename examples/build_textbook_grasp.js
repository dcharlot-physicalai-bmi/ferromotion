// Assemble the interactive-textbook chapter on grasp force closure: inline the wasm-bindgen glue +
// base64-embed the wasm, so the page runs the real ferromotion Ferrari–Canny Q1 metric on-device. The
// reader drags fingers around an object and watches whether the grip can resist a push from ANY
// direction — force closure — with the weakest direction shown live. Same self-contained pattern as
// chapters 1–3.
const fs = require("fs");
const path = require("path");

const pkg = path.join(__dirname, "..", "crates", "ferromotion-wasm", "pkg");
const glue = fs.readFileSync(path.join(pkg, "ferromotion_wasm.js"), "utf8");
const wasmB64 = fs.readFileSync(path.join(pkg, "ferromotion_wasm_bg.wasm")).toString("base64");

const APP = String.raw`
function b64ToBytes(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
const fmt=(x,n=2)=>Number(x).toFixed(n);
const TAU=Math.PI*2;

let lab, dragging=-1;
function preset(kind){
  lab=new GraspLab(); lab.set_mu(0.5);
  if(kind==='pinch'){ lab.add_contact(0); lab.add_contact(Math.PI); }
  else if(kind==='tripod'){ for(let k=0;k<3;k++) lab.add_contact(k*TAU/3 + 0.35); }
  else if(kind==='slip'){ lab.add_contact(0.32); lab.add_contact(-0.32); } // same side
  else if(kind==='quad'){ for(let k=0;k<4;k++) lab.add_contact(k*TAU/4 + 0.2); }
  document.getElementById('muRange').value=0.5; document.getElementById('muVal').textContent='0.50';
  document.querySelectorAll('.seg button').forEach(b=>b.classList.toggle('on', b.dataset.k===kind));
  syncReadouts(); draw();
}

const cv=document.getElementById("stage");
function tf(){ const r=cv.getBoundingClientRect(); return {ox:r.width/2, oy:r.height/2, s:Math.min(r.width,r.height)*0.30, W:r.width, H:r.height}; }
function w2s(x,y){ const t=tf(); return [t.ox+x*t.s, t.oy - y*t.s]; } // flip y so up is +y
function angleAt(px,py){ const t=tf(); return Math.atan2(-(py-t.oy), px-t.ox); }

function draw(){
  const ctx=cv.getContext("2d"); const t=tf();
  const dpr=Math.min(devicePixelRatio||1,2); cv.width=t.W*dpr; cv.height=t.H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,t.W,t.H);
  const R=lab.radius(), mu=lab.mu(), fc=lab.is_force_closure(), n=lab.n();
  // object
  const [ox,oy]=w2s(0,0); const rs=R*t.s;
  ctx.beginPath(); ctx.arc(ox,oy,rs,0,TAU);
  ctx.fillStyle= fc ? "rgba(125,211,160,.08)" : "rgba(232,131,111,.07)";
  ctx.fill(); ctx.strokeStyle="rgba(174,182,204,.4)"; ctx.lineWidth=1.5; ctx.stroke();
  ctx.beginPath(); ctx.arc(ox,oy,3,0,TAU); ctx.fillStyle="rgba(174,182,204,.5)"; ctx.fill();
  // friction cones + fingers
  const phi=Math.atan(mu);
  for(let i=0;i<n;i++){
    const th=lab.theta(i); const [px,py]=w2s(R*Math.cos(th), R*Math.sin(th));
    const nin=[-Math.cos(th),-Math.sin(th)]; // inward normal (world)
    const clen=rs*1.15;
    const e1=[nin[0]*Math.cos(phi)-nin[1]*Math.sin(phi), nin[0]*Math.sin(phi)+nin[1]*Math.cos(phi)];
    const e2=[nin[0]*Math.cos(-phi)-nin[1]*Math.sin(-phi), nin[0]*Math.sin(-phi)+nin[1]*Math.cos(-phi)];
    // screen y is flipped: convert world dir to screen dir (negate y)
    const sd=(v)=>[v[0], -v[1]];
    const a=sd(e1), b=sd(e2);
    ctx.beginPath(); ctx.moveTo(px,py); ctx.lineTo(px+a[0]*clen,py+a[1]*clen); ctx.lineTo(px+b[0]*clen,py+b[1]*clen); ctx.closePath();
    ctx.fillStyle="rgba(217,180,94,.13)"; ctx.fill();
    ctx.strokeStyle="rgba(217,180,94,.4)"; ctx.lineWidth=1; ctx.stroke();
    // finger pad
    ctx.beginPath(); ctx.arc(px,py,8,0,TAU);
    ctx.fillStyle= i===dragging ? "#f0cf82" : "#161f3a"; ctx.strokeStyle="#d9b45e"; ctx.lineWidth=2.5; ctx.fill(); ctx.stroke();
  }
  // weakest wrench direction — the disturbance the grip is closest to failing against
  if(n>=2){
    const d=lab.weakest_dir(); const f=[d[0],d[1]], tq=d[2];
    const col = fc ? "rgba(125,211,160,.9)" : "#e8836f";
    const fl=Math.hypot(f[0],f[1]);
    if(fl>0.08){ // linear-push part
      const L=rs*0.9, a=Math.atan2(-f[1],f[0]);
      const ex=ox+Math.cos(a)*L, ey=oy+Math.sin(a)*L;
      ctx.strokeStyle=col; ctx.fillStyle=col; ctx.lineWidth=2.5;
      ctx.beginPath(); ctx.moveTo(ox,oy); ctx.lineTo(ex,ey); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(ex,ey);
      ctx.lineTo(ex-9*Math.cos(a-0.4),ey-9*Math.sin(a-0.4));
      ctx.lineTo(ex-9*Math.cos(a+0.4),ey-9*Math.sin(a+0.4)); ctx.closePath(); ctx.fill();
    }
    if(Math.abs(tq)>0.25){ // twist part
      const rr=rs*0.5, s0=-0.9, s1=0.9*Math.sign(tq)+ (tq>0?0:0);
      ctx.strokeStyle=col; ctx.lineWidth=2.5;
      ctx.beginPath(); ctx.arc(ox,oy,rr, -0.6, -0.6 + 2.4*Math.sign(tq), tq<0); ctx.stroke();
    }
  }
  ctx.font="500 10px ui-monospace,monospace"; ctx.fillStyle=(fc?"rgba(125,211,160,.85)":"#e8836f");
  ctx.fillText(fc?"— hardest push it still resists":"— the push that breaks the grip", 12, t.H-12);
}

function syncReadouts(){
  const q=lab.q1(), fc=lab.is_force_closure();
  const v=document.getElementById("verdict");
  v.textContent = fc ? "FORCE CLOSURE" : "NOT SECURE";
  v.style.color = fc ? "#7dd3a0" : "#e8836f";
  document.getElementById("q1").textContent = fmt(q,3);
  document.getElementById("q1").style.color = fc ? "#7dd3a0" : "#e8836f";
  const mb=document.getElementById("q1Bar");
  mb.style.width=Math.max(0,Math.min(100,q/0.6*100))+"%";
  mb.style.background = fc ? "#7dd3a0" : "#e8836f";
  document.getElementById("fingers").textContent=lab.n();
}

/* interaction: drag a finger around the rim */
cv.addEventListener("pointerdown",e=>{ const r=cv.getBoundingClientRect(); const t=tf();
  const px=e.clientX-r.left, py=e.clientY-r.top;
  // pick nearest finger
  let best=-1,bd=20; for(let i=0;i<lab.n();i++){ const [fx,fy]=w2s(lab.radius()*Math.cos(lab.theta(i)),lab.radius()*Math.sin(lab.theta(i))); const dd=Math.hypot(px-fx,py-fy); if(dd<bd){bd=dd;best=i;} }
  dragging=best; if(best>=0){ cv.setPointerCapture(e.pointerId); }
});
cv.addEventListener("pointermove",e=>{ if(dragging<0) return; const r=cv.getBoundingClientRect();
  lab.set_theta(dragging, angleAt(e.clientX-r.left, e.clientY-r.top)); syncReadouts(); draw(); });
cv.addEventListener("pointerup",()=>{ dragging=-1; draw(); });

document.getElementById("muRange").oninput=e=>{ lab.set_mu(+e.target.value); document.getElementById("muVal").textContent=fmt(+e.target.value,2); syncReadouts(); draw(); };
document.querySelectorAll('.seg button').forEach(b=> b.onclick=()=>preset(b.dataset.k));
document.getElementById("addf").onclick=()=>{ lab.add_contact(Math.random()*TAU); syncReadouts(); draw(); };

/* the page verifies the force-closure test on load: antipodal holds, same-side fails */
function selfCheck(){
  const a=new GraspLab(); a.set_mu(0.5); a.add_contact(0); a.add_contact(Math.PI);
  const s=new GraspLab(); s.set_mu(0.5); s.add_contact(0.3); s.add_contact(-0.3);
  document.getElementById("scAnti").textContent=fmt(a.q1(),3);
  document.getElementById("scSame").textContent=fmt(s.q1(),3);
  document.getElementById("scVerdict").textContent =
    (a.q1()>0 && s.q1()<0) ? "the pinch closes, the same-side grip does not" : "unexpected";
}

async function main(){
  await __wbg_init(b64ToBytes(WASM_B64));
  selfCheck(); preset('pinch');
  window.__grasp={lab:()=>lab, preset, setTheta:(i,t)=>{lab.set_theta(i,t);syncReadouts();}, setMu:(m)=>{lab.set_mu(m);syncReadouts();}};
  window.__textbook_ready=true;
  addEventListener("resize",draw); draw();
}
main();
`;

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>When touching becomes holding — ferromotion textbook</title>
<meta name="description" content="An interactive chapter on grasp force closure: why a grip holds against a push from any direction only when its friction cones span wrench space. Runs the real Rust Ferrari–Canny metric on-device."/>
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
#stage{width:100%;height:420px;cursor:grab;border-radius:10px;background:radial-gradient(700px 420px at 50% 45%,#0e1730,#0b1122)}
.read{font-family:var(--mono);font-size:.76rem;color:var(--soft);margin:10px 0 0;text-align:center}.read b{color:var(--ink)}.dim{color:var(--dim)}
.ctl{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:14px;justify-content:center}
button{background:linear-gradient(180deg,#d9b45e,#a9832f);color:#161200;border:0;border-radius:9px;padding:8px 16px;font:700 .82rem var(--sans);cursor:pointer}
button.ghost{background:transparent;color:var(--soft);border:1px solid var(--line)}
button:active{transform:translateY(1px)}
label{font-family:var(--mono);font-size:.72rem;color:var(--soft);display:flex;align-items:center;gap:8px}
input[type=range]{accent-color:var(--gold);width:120px}
.seg{display:inline-flex;background:#0d1428;border:1px solid var(--line);border-radius:9px;overflow:hidden;flex-wrap:wrap}
.seg button{background:transparent;color:var(--soft);border:0;border-radius:0;padding:8px 13px;font:600 .76rem var(--sans)}
.seg button.on{background:linear-gradient(180deg,rgba(217,180,94,.22),rgba(217,180,94,.06));color:var(--goldb)}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}
.stat{background:#0d1428;border:1px solid var(--line);border-radius:10px;padding:9px 8px;text-align:center}
.stat .v{font-family:var(--mono);font-size:1rem;font-weight:700;color:var(--goldb)}
.stat .k{font-family:var(--mono);font-size:.58rem;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);margin-top:2px}
.bar{height:6px;border-radius:3px;background:#0d1428;overflow:hidden;margin-top:8px;border:1px solid var(--line)}
.bar>div{height:100%;width:0;background:var(--green);transition:width .1s linear,background .1s linear}
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
  <p class="kicker"><span style="color:var(--goldb);font-size:1.05rem">&#934;</span><span>ferromotion · textbook · chapter 4</span>
    <span class="badge">rust → wasm · on-device</span></p>
  <h1>When touching becomes holding</h1>
  <p class="lede">A hand can be pressed against an object and still not hold it. What separates a grip from mere contact is one geometric property — force closure — and this page computes it on your device with the same Rust metric a grasp planner optimizes.</p>

  <h2><span class="n">01 — the problem</span>Contact is not a grip</h2>
  <p>Put two fingers on the same side of a mug and squeeze: it slips away the instant anything nudges it sideways. Put them on opposite sides and it is locked. Both are "touching." Only one is a grasp. The difference is whether the fingers, together, can push back against a disturbance <b>from every direction at once</b> — a shove, a pull, a twist, any combination.</p>
  <p>That is a hard-sounding requirement — infinitely many directions — but it collapses to a single geometric test, and once you see it you can read a grip by eye.</p>

  <h2><span class="n">02 — the friction cone</span>What one finger can do</h2>
  <p>A finger cannot pull, and it cannot push purely sideways — friction only lets it push within a <b>cone</b> around the surface normal, half-angle <span style="font-family:var(--mono);color:var(--goldb)">arctan μ</span>. Every force it can apply, and the twist that force makes about the object's centre, is one point in a three-dimensional <i>wrench space</i> of <span style="font-family:var(--mono);color:var(--goldb)">(force, torque)</span>. A grip's whole repertoire is the cone of combinations its fingers can jointly produce.</p>
  <p>Below, drag the fingers around the object and widen the friction with <b>μ</b>. Each gold wedge is one finger's cone.</p>
  <div class="fig">
    <canvas id="stage"></canvas>
    <div class="ctl" style="margin-top:12px">
      <span class="seg"><button data-k="pinch" class="on">Two-finger pinch</button><button data-k="tripod">Tripod</button><button data-k="quad">Four fingers</button><button data-k="slip">Same-side (slips)</button></span>
    </div>
    <div class="ctl">
      <label>friction μ <input type="range" id="muRange" min="0" max="1.5" step="0.02" value="0.5"/> <span id="muVal">0.50</span></label>
      <button id="addf" class="ghost">+ finger</button>
    </div>
    <div class="stats">
      <div class="stat"><div class="v" id="verdict" style="font-size:.82rem">—</div><div class="k">grasp</div></div>
      <div class="stat"><div class="v" id="q1">—</div><div class="k">Q1 margin</div></div>
      <div class="stat"><div class="v" id="fingers">—</div><div class="k">fingers</div></div>
    </div>
    <div class="bar"><div id="q1Bar"></div></div>
    <p class="read dim" style="margin-top:8px">the arrow is the <b>weakest</b> direction — the push the grip resists least; drag the fingers until it turns green</p>
  </div>

  <h2><span class="n">03 — force closure</span>The origin inside the hull</h2>
  <p>Here is the whole test. Collect the extreme wrenches at the edges of every finger's cone. The grasp can resist a disturbance in a given direction exactly when some combination of those wrenches points back against it. It can resist <i>every</i> direction — force closure — precisely when those wrenches surround the origin: when <span style="font-family:var(--mono);color:var(--goldb)">0</span> lies strictly inside their convex hull.</p>
  <div class="callout">Watch the weakest-direction arrow as you drag. It always points at the disturbance the grip resists least. When the fingers surround the object, even that weakest direction is covered and the arrow is <b style="color:var(--green)">green</b> — the grip holds against anything. Slide the fingers to the same side and a gap opens in the hull: the arrow turns <b style="color:var(--red)">red</b> and points straight out through it — the exact push that peels the object away.</p>

  <h2><span class="n">04 — the number</span>How firm, not just whether</h2>
  <p>Force closure is yes-or-no, but grips have degrees. The <b>Ferrari–Canny Q1</b> metric measures the margin: the radius of the largest wrench-ball the grasp resists in <i>every</i> direction — the strength of its weakest direction. <span style="font-family:var(--mono);color:var(--goldb)">Q1 &gt; 0</span> is force closure; larger is firmer. On load, this page ran the metric on two grips:</p>
  <table>
    <tr><td>Q1 — two fingers, opposite sides (a pinch)</td><td id="scAnti">…</td></tr>
    <tr><td>Q1 — two fingers, same side</td><td id="scSame">…</td></tr>
    <tr><td>verdict</td><td id="scVerdict">…</td></tr>
  </table>
  <p>The pinch scores positive and the same-side grip scores negative, on your device — the metric agrees with the intuition that one holds and one slips, and puts a number on how much.</p>

  <h2><span class="n">05 — the point</span>A grip you can optimize</h2>
  <div class="verdict">
    <div class="big">Q1 is a grip you can do gradient ascent on.</div>
    <p>Because a smoothed Q1 is differentiable in where the fingers land, a grasp synthesizer does not search blindly — it slides the contacts uphill on exactly this margin until the object is locked, then keeps climbing for robustness.</p>
  </div>
  <p>This is how a robot decides where to put its fingers. Not by matching a library of remembered grasps, but by holding a differentiable measure of "how securely am I holding this" and improving it — the same number you were just dragging fingers to maximize by hand. It generalizes to any object whose surface you can sample, and it degrades gracefully: a grip with a small positive margin is one a careful controller can still use, and the number tells you which.</p>

  <p class="note"><b>What you just drove:</b> <span style="color:var(--soft)">force_closure_q1</span>, <span style="color:var(--soft)">primitive_wrenches</span> and <span style="color:var(--soft)">GraspContact</span> from <span style="color:var(--soft)">ferromotion-core</span>, compiled to WebAssembly — the same code the native tools link against, not a reimplementation. Each contact's friction cone is linearized to primitive wrenches; Q1 is the Ferrari–Canny support-function minimum over sampled directions; the weakest-direction arrow is the argmin of that minimum. Nothing precomputed — every drag re-solves it.<br/><br/>
  <b>Verified in the library:</b> an antipodal pinch is force closure and a same-side grip is not · a narrow pinch needs enough friction to close · a symmetric tripod grips more firmly than a two-finger pinch · more friction never lowers Q1 · the reported weakest direction realises Q1 exactly. Each is a test in <span style="color:var(--soft)">cargo test</span>, not a claim in prose. See also <a href="/assets/sims/morphological-computation">ch.1</a> · <a href="/assets/sims/algebraic-connectivity">ch.2</a> · <a href="/assets/sims/safety-filter">ch.3</a>.<br/><br/>
  <b>Institute for Physical AI</b> · <a href="https://github.com/dcharlot-physicalai-bmi/ferromotion">the Rust library</a> · <a href="https://crates.io/crates/ferromotion">crates.io</a></p>
</div>
<script type="module">
${glue}
const WASM_B64="${wasmB64}";
${APP}
</script></body></html>`;

const outFile = path.join(__dirname, "..", "..", "v2", "public", "assets", "sims", "force-closure.html");
fs.writeFileSync(outFile, html);
console.log(`wrote ${outFile} (${(html.length / 1024).toFixed(0)} KB, wasm ${(wasmB64.length / 1024).toFixed(0)} KB b64)`);

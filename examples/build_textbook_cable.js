// Assemble the interactive-textbook chapter on cable-driven parallel robots: inline the wasm-bindgen
// glue + base64-embed the wasm, so the page runs the real ferromotion Cdpr tension distribution on-
// device. The reader drags a platform hung from four cables and watches the pull redistribute, going
// slack/over-tensioned at the edges of the workspace. Same self-contained pattern as the other chapters.
const fs = require("fs");
const path = require("path");

const pkg = path.join(__dirname, "..", "crates", "ferromotion-wasm", "pkg");
const glue = fs.readFileSync(path.join(pkg, "ferromotion_wasm.js"), "utf8");
const wasmB64 = fs.readFileSync(path.join(pkg, "ferromotion_wasm_bg.wasm")).toString("base64");

const APP = String.raw`
function b64ToBytes(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
const fmt=(x,n=2)=>Number(x).toFixed(n);
const TAU=Math.PI*2;
let lab, drag=false;

const cv=document.getElementById("stage");
function tf(){ const r=cv.getBoundingClientRect(); const s=Math.min(r.width,r.height)/4.6; return {r,s,ox:r.width/2,oy:r.height/2}; }
function P(x,y){ const t=tf(); return [t.ox+x*t.s, t.oy - y*t.s]; }
function inv(px,py){ const t=tf(); return [(px-t.ox)/t.s, -(py-t.oy)/t.s]; }

function draw(){
  const t=tf(); const ctx=cv.getContext("2d");
  const dpr=Math.min(devicePixelRatio||1,2); cv.width=t.r.width*dpr; cv.height=t.r.height*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,t.r.width,t.r.height);
  const n=lab.n(), tmax=lab.t_max(), tmin=lab.t_min(), feasible=lab.feasible();
  // frame + anchors
  ctx.strokeStyle="rgba(120,140,180,.25)"; ctx.lineWidth=2;
  ctx.beginPath(); for(let i=0;i<=n;i++){ const a=P(lab.anchor_x(i%n),lab.anchor_y(i%n)); i?ctx.lineTo(a[0],a[1]):ctx.moveTo(a[0],a[1]); } ctx.stroke();
  for(let i=0;i<n;i++){ const a=P(lab.anchor_x(i),lab.anchor_y(i)); ctx.beginPath(); ctx.arc(a[0],a[1],6,0,TAU); ctx.fillStyle="#8aa0c8"; ctx.fill(); }
  // cables — thickness & colour by tension
  for(let i=0;i<n;i++){
    const a=P(lab.anchor_x(i),lab.anchor_y(i)), b=P(lab.attach_x(i),lab.attach_y(i));
    const ti=lab.tension(i); const frac=(ti-tmin)/(tmax-tmin);
    const slack = ti<=tmin+1e-6, over = ti>=tmax-1e-6;
    ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]);
    ctx.lineWidth=1.5+7*Math.max(0,Math.min(1,frac));
    ctx.strokeStyle = (slack||over) ? "#e8836f" : "rgba(217,180,94,"+(0.5+0.5*frac)+")";
    ctx.stroke();
  }
  // platform
  const pc=P(lab.pose_x(),lab.pose_y());
  ctx.beginPath();
  for(let i=0;i<n;i++){ const b=P(lab.attach_x(i),lab.attach_y(i)); i?ctx.lineTo(b[0],b[1]):ctx.moveTo(b[0],b[1]); }
  ctx.closePath(); ctx.fillStyle="#161f3a"; ctx.strokeStyle=feasible?"#7dd3a0":"#e8836f"; ctx.lineWidth=2.5; ctx.fill(); ctx.stroke();
  ctx.beginPath(); ctx.arc(pc[0],pc[1],5,0,TAU); ctx.fillStyle=drag?"#f0cf82":(feasible?"#7dd3a0":"#e8836f"); ctx.fill();
  // weight arrow
  const w=P(lab.pose_x(),lab.pose_y()); const al=34;
  ctx.strokeStyle="rgba(174,182,204,.55)"; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(w[0],w[1]+8); ctx.lineTo(w[0],w[1]+8+al); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(w[0]-5,w[1]+8+al-6); ctx.lineTo(w[0],w[1]+8+al); ctx.lineTo(w[0]+5,w[1]+8+al-6); ctx.fillStyle="rgba(174,182,204,.55)"; ctx.fill();
  ctx.fillStyle="rgba(174,182,204,.6)"; ctx.font="500 10px ui-monospace,monospace"; ctx.fillText("weight", w[0]+9, w[1]+8+al);
  ctx.fillStyle="#727d99"; ctx.fillText("drag the platform", 12, 20);
  if(!feasible){ ctx.fillStyle="#e8836f"; ctx.font="600 12px ui-sans-serif,system-ui"; ctx.fillText("outside the workspace — a cable goes slack", 12, t.r.height-14); }
  // readouts
  document.getElementById("fstate").textContent=feasible?"holding":"cannot hold"; document.getElementById("fstate").style.color=feasible?"#7dd3a0":"#e8836f";
  document.getElementById("tmaxv").textContent=fmt(lab.max_tension(),1);
  document.getElementById("tminv").textContent=fmt(lab.min_tension(),1);
}

cv.addEventListener("pointerdown",e=>{ const r=cv.getBoundingClientRect(); const pc=P(lab.pose_x(),lab.pose_y());
  if(Math.hypot(e.clientX-r.left-pc[0], e.clientY-r.top-pc[1])<40){ drag=true; cv.setPointerCapture(e.pointerId); } });
cv.addEventListener("pointermove",e=>{ if(!drag) return; const r=cv.getBoundingClientRect(); const w=inv(e.clientX-r.left,e.clientY-r.top);
  lab.set_pose(Math.max(-1.95,Math.min(1.95,w[0])), Math.max(-1.95,Math.min(1.95,w[1])), 0); lab.solve(); draw(); });
cv.addEventListener("pointerup",()=>{ drag=false; });
document.getElementById("wt").oninput=e=>{ lab.set_weight(+e.target.value); lab.solve(); document.getElementById("wtVal").textContent=fmt(+e.target.value,0); draw(); };
document.getElementById("reset").onclick=()=>{ lab.set_pose(0,0,0); lab.set_weight(10); document.getElementById("wt").value=10; document.getElementById("wtVal").textContent="10"; lab.solve(); draw(); };

function selfCheck(){
  const t=new CableLab(); t.set_pose(0,0,0); t.solve();
  document.getElementById("scFeas").textContent=t.feasible()?"feasible ✓":"—";
  document.getElementById("scTmax").textContent=fmt(t.max_tension(),1);
  document.getElementById("scVerdict").textContent = t.feasible() ? "positive, in-range tensions balance the load exactly" : "unexpected";
}

async function main(){
  await __wbg_init(b64ToBytes(WASM_B64));
  lab=new CableLab(); selfCheck(); lab.solve();
  window.__cable={lab:()=>lab, setPose:(x,y)=>{lab.set_pose(x,y,0);lab.solve();}, feasible:()=>lab.feasible(), maxT:()=>lab.max_tension(), minT:()=>lab.min_tension()};
  window.__textbook_ready=true;
  addEventListener("resize",draw); draw();
}
main();
`;

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Held by cables — ferromotion textbook</title>
<meta name="description" content="An interactive chapter on cable-driven parallel robots: cables can only pull, so holding a platform means finding positive tensions that balance the load — and the workspace ends where a cable would go slack. Runs the real Rust CDPR tension distribution on-device."/>
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
#stage{width:100%;height:420px;cursor:grab;border-radius:10px;background:radial-gradient(700px 420px at 50% 50%,#0e1730,#0b1122)}
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
  <p class="kicker"><span style="color:var(--goldb);font-size:1.05rem">&#934;</span><span>ferromotion · textbook · chapter 13</span>
    <span class="badge">rust → wasm · on-device</span></p>
  <h1>Held by cables</h1>
  <p class="lede">A platform hung from a few cables can move faster and reach farther than any jointed arm — but a cable can only pull, never push. Holding it steady means finding a set of tensions that are all positive and all within limits, and this page finds them on your device with the same Rust code the native tools use.</p>

  <h2><span class="n">01 — the problem</span>A cable only pulls</h2>
  <p>Cable-driven parallel robots — camera rigs over a stadium, giant 3D printers, rehabilitation supports — trade rigid links for lightweight cables spooled from fixed anchors. That makes them fast and huge, but it comes with a hard constraint: a cable cannot push. Every cable must stay <b>taut</b>, its tension between a floor (so it never goes slack) and a ceiling (so the motor and cable survive). Holding the platform is a question of whether such tensions exist at all.</p>

  <h2><span class="n">02 — redundancy</span>More cables than freedoms</h2>
  <p>To keep every cable able to pull, these robots use <i>more</i> cables than the platform has degrees of freedom. That makes the tensions <b>non-unique</b> — many combinations produce the same supporting wrench — so the controller must choose one that keeps them all comfortably in range. The clean choice is the tension nearest the middle of each cable's limits that still balances the load exactly.</p>
  <p>Below, a platform hangs from four cables against its own weight. <b>Drag it</b> around the frame; each cable's thickness shows its tension.</p>
  <div class="fig">
    <canvas id="stage"></canvas>
    <div class="ctl">
      <button id="reset" class="ghost">Reset</button>
      <label>weight <input type="range" id="wt" min="2" max="30" step="1" value="10"/> <span id="wtVal">10</span></label>
    </div>
    <div class="stats">
      <div class="stat"><div class="v" id="fstate" style="font-size:.82rem">—</div><div class="k">platform</div></div>
      <div class="stat"><div class="v" id="tmaxv">—</div><div class="k">max tension</div></div>
      <div class="stat"><div class="v" id="tminv">—</div><div class="k">min tension</div></div>
    </div>
    <p class="read dim" style="margin-top:8px">drag it toward a corner and watch a cable's tension collapse toward slack — that edge is the boundary of the workspace</p>
  </div>

  <h2><span class="n">03 — the workspace</span>Where a valid grip exists</h2>
  <p>Drag the platform toward a corner and one of the far cables loses tension — it would go slack, and the moment it does the platform is no longer controlled. The set of poses where a fully-taut, in-range distribution <i>exists</i> is the robot's <b>wrench-feasible workspace</b>, and its boundary is exactly where you feel a cable give out. Raise the weight and the workspace shrinks; the heavier the load, the smaller the region the cables can hold it in.</p>
  <div class="callout">This is the trade at the heart of cable robots. The tension distribution is a tiny computation — project the mid-range tension onto the set that balances the load — but whether it lands <b>inside the limits</b> is what decides if the pose is usable at all. The controller does not just pick tensions; it is continuously checking that a valid grip still exists, and steering to stay inside the region where it does.</p>

  <h2><span class="n">04 — the check</span>Positive, balanced, in range</h2>
  <p>On load, this page hung the platform in the centre and solved for the holding tensions:</p>
  <table>
    <tr><td>a valid (all-taut, in-range) distribution exists?</td><td id="scFeas">…</td></tr>
    <tr><td>largest cable tension</td><td id="scTmax">…</td></tr>
    <tr><td>verdict</td><td id="scVerdict">…</td></tr>
  </table>
  <p>The four tensions come out positive, within limits, and balance the weight exactly — the equilibrium residual is machine-zero by construction. There is one subtlety the model is honest about: at the perfectly centred, unrotated pose the cables pull straight along their own moment arms, so the platform has plenty of force authority but <i>no torque authority at all</i> — a wrench that twists it cannot be balanced there until it rotates even slightly.</p>

  <h2><span class="n">05 — the point</span>Reach is bounded by the pull</h2>
  <div class="verdict">
    <div class="big">The workspace is where the cables can still pull.</div>
    <p>A pull-only mechanism can be fast, light, and enormous — and its reachable, controllable region is precisely the set of poses where some all-positive, in-range tension distribution balances the load. Find that distribution, and check it exists.</p>
  </div>
  <p>Cable robots close out this series on a different note than the rigid and soft chapters: here the governing question is not a single stability margin but the <i>feasibility of a pull</i>, a small least-squares projection wrapped in a bounds check. It is the same shape of answer, though — turn a control problem into a place where the solution is forced and then simply read off whether it lands in the allowed set. Fast, light, and reaching far, cable robots are physical AI built from tension alone.</p>

  <p class="note"><b>What you just drove:</b> the <span style="color:var(--soft)">Cdpr</span> tension distribution from <span style="color:var(--soft)">ferromotion-control</span>, compiled to WebAssembly — the same code the native tools link against. Each drag rebuilds the structure matrix at the new pose and projects the mid-range tension onto the wrench-equilibrium set; nothing precomputed.<br/><br/>
  <b>Verified in the library:</b> the distribution balances the wrench exactly at a generic pose (‖Wt−w‖&lt;1e-9); a centred platform holds a load with positive, symmetric tensions; the correction lies in row(W) — the nearest-to-mid property; an over-range wrench is flagged infeasible; and the centred symmetric config is torque-singular. Each is a test in <span style="color:var(--soft)">cargo test</span>, not a claim in prose. See also <a href="/assets/sims/soft-robots">ch.11 — the robot that bends</a> · <a href="/assets/sims/textbook">the full textbook</a>.<br/><br/>
  <b>Institute for Physical AI</b> · <a href="https://github.com/dcharlot-physicalai-bmi/ferromotion">the Rust library</a> · <a href="https://crates.io/crates/ferromotion">crates.io</a></p>
</div>
<script type="module">
${glue}
const WASM_B64="${wasmB64}";
${APP}
</script></body></html>`;

const outFile = path.join(__dirname, "..", "..", "v2", "public", "assets", "sims", "cable-robots.html");
fs.writeFileSync(outFile, html);
console.log(`wrote ${outFile} (${(html.length / 1024).toFixed(0)} KB, wasm ${(wasmB64.length / 1024).toFixed(0)} KB b64)`);

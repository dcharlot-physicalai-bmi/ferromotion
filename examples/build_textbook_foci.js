// Assemble the interactive-textbook chapter on FOCI — collision on Gaussian-splat maps. Inline the
// wasm-bindgen glue + base64-embed the wasm so the page runs the real ferromotion overlap-integral
// collision on-device. The reader slides an elongated robot toward a slot between two obstacle Gaussians
// and rotates it: head-on it jams; turned to align with the gap it slips through. Same self-contained
// pattern as the other chapters.
const fs = require("fs");
const path = require("path");

const pkg = path.join(__dirname, "..", "crates", "ferromotion-wasm", "pkg");
const glue = fs.readFileSync(path.join(pkg, "ferromotion_wasm.js"), "utf8");
const wasmB64 = fs.readFileSync(path.join(pkg, "ferromotion_wasm_bg.wasm")).toString("base64");

const APP = String.raw`
function b64ToBytes(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
const fmt=(x,n=3)=>Number(x).toFixed(n);
const TAU=Math.PI*2;
let lab, drag=false;

const cv=document.getElementById("stage");
function tf(){ const r=cv.getBoundingClientRect(); const s=Math.min(r.width/6.0, r.height/3.4); return {r,s,ox:r.width*0.5,oy:r.height*0.5}; }
function P(x,y){ const t=tf(); return [t.ox+x*t.s, t.oy - y*t.s]; }
function inv(px,py){ const t=tf(); return [(px-t.ox)/t.s, -(py-t.oy)/t.s]; }

// draw an ellipse at (cx,cy) with per-axis radii (rx,ry) rotated by ang, in world coords
function ellipse(ctx,cx,cy,rx,ry,ang,fill,stroke){
  const t=tf(); const c=P(cx,cy);
  ctx.save(); ctx.translate(c[0],c[1]); ctx.rotate(-ang);
  ctx.beginPath(); ctx.ellipse(0,0,rx*t.s,ry*t.s,0,0,TAU);
  if(fill){ctx.fillStyle=fill;ctx.fill();} if(stroke){ctx.lineWidth=2;ctx.strokeStyle=stroke;ctx.stroke();}
  ctx.restore();
}

function costColor(c){ // green (clear) → red (collision)
  const t=Math.max(0,Math.min(1,c/1.2));
  const r=Math.round(125+(232-125)*t), g=Math.round(211+(131-211)*t), b=Math.round(160+(111-160)*t);
  return "rgb("+r+","+g+","+b+")";
}

function draw(){
  const r=cv.getBoundingClientRect(); const dpr=Math.min(devicePixelRatio||1,2);
  cv.width=r.width*dpr; cv.height=r.height*dpr; const ctx=cv.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,r.width,r.height);
  // obstacles (2σ ellipses)
  for(let i=0;i<lab.n_obstacles();i++){
    ellipse(ctx,lab.obs_x(i),lab.obs_y(i),2*lab.obs_sx(i),2*lab.obs_sy(i),0,"rgba(120,140,180,.16)","rgba(138,160,200,.6)");
    ellipse(ctx,lab.obs_x(i),lab.obs_y(i),lab.obs_sx(i),lab.obs_sy(i),0,"rgba(120,140,180,.10)",null);
  }
  // the slot hint
  ctx.fillStyle="rgba(125,211,160,.55)"; ctx.font="11px ui-monospace,monospace"; ctx.fillText("the slot",P(0,0)[0]-22,P(0,-0.02)[1]);
  // robot (elongated ellipse), colored by collision cost
  const c=lab.cost();
  ellipse(ctx,lab.pose_x(),lab.pose_y(),lab.robot_sx()*2,lab.robot_sy()*2,lab.pose_yaw(),costColor(c),"#f0cf82");
  // orientation tick
  const pc=P(lab.pose_x(),lab.pose_y());
  ctx.beginPath(); ctx.arc(pc[0],pc[1],4,0,TAU); ctx.fillStyle="#161f3a"; ctx.fill();
  ctx.fillStyle="#727d99"; ctx.font="11px ui-monospace,monospace"; ctx.fillText("drag the robot · rotate with the slider",12,20);
  // readouts
  document.getElementById("costv").textContent=fmt(c,3);
  document.getElementById("yawv").textContent=fmt(lab.pose_yaw()*180/Math.PI,0)+"°";
  const passable = c<0.5;
  document.getElementById("statev").textContent=passable?"clear":"blocked";
  document.getElementById("statev").style.color=passable?"#7dd3a0":"#e8836f";
}

cv.addEventListener("pointerdown",e=>{ const r=cv.getBoundingClientRect(); const pc=P(lab.pose_x(),lab.pose_y());
  if(Math.hypot(e.clientX-r.left-pc[0],e.clientY-r.top-pc[1])<60){ drag=true; cv.setPointerCapture(e.pointerId);} });
cv.addEventListener("pointermove",e=>{ if(!drag)return; const r=cv.getBoundingClientRect(); const w=inv(e.clientX-r.left,e.clientY-r.top);
  lab.set_pose(Math.max(-2.8,Math.min(2.8,w[0])),Math.max(-2.0,Math.min(2.0,w[1])),lab.pose_yaw()); draw(); });
cv.addEventListener("pointerup",()=>{drag=false;});
document.getElementById("yaw").oninput=e=>{ lab.set_pose(lab.pose_x(),lab.pose_y(),+e.target.value*Math.PI/180); draw(); };
document.getElementById("reset").onclick=()=>{ lab.set_pose(-2.2,0,0); document.getElementById("yaw").value=0; draw(); };
document.getElementById("thread").onclick=()=>{ lab.set_pose(0,0,90*Math.PI/180); document.getElementById("yaw").value=90; draw(); };

function selfCheck(){
  document.getElementById("scHead").textContent=fmt(lab.head_on_cost(),3);
  document.getElementById("scTurn").textContent=fmt(lab.turned_cost(),3);
  const ratio=lab.head_on_cost()/Math.max(lab.turned_cost(),1e-9);
  document.getElementById("scRatio").textContent="×"+fmt(ratio,0);
  document.getElementById("scVerdict").textContent=(lab.turned_cost()<0.25*lab.head_on_cost())?"turning to align clears the slot":"unexpected";
}

async function main(){
  await __wbg_init(b64ToBytes(WASM_B64));
  lab=new FociLab(); selfCheck();
  window.__foci={lab:()=>lab, cost:()=>lab.cost(), headOn:()=>lab.head_on_cost(), turned:()=>lab.turned_cost()};
  window.__textbook_ready=true;
  addEventListener("resize",draw); draw();
}
main();
`;

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Turning to fit — ferromotion textbook</title>
<meta name="description" content="An interactive chapter on FOCI: collision checked directly on a 3D Gaussian-splat map via the overlap integral between Gaussians. Because the robot's Gaussians rotate with it, an elongated body can turn to slip through a slot it would hit head-on — orientation-aware collision. Runs the real Rust overlap-integral collision on-device."/>
<style>
:root{--ground:#0a0f1e;--panel:#111830;--line:#26324c;--ink:#eef1f8;--soft:#aeb6cc;--dim:#727d99;--gold:#d9b45e;--goldb:#f0cf82;--green:#7dd3a0;--red:#e8836f;--mono:ui-monospace,"SF Mono",Menlo,monospace;--sans:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
*{box-sizing:border-box}html,body{margin:0}
body{background:radial-gradient(1200px 700px at 50% -10%,#12203f,var(--ground));color:var(--ink);font-family:var(--sans);line-height:1.65}
.wrap{max-width:768px;margin:0 auto;padding:clamp(24px,5vw,56px) 20px 72px}
.kicker{font-family:var(--mono);font-size:.66rem;letter-spacing:.26em;text-transform:uppercase;color:var(--gold);display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:0 0 14px}
.kicker>span{white-space:nowrap}
h1{font-size:clamp(1.8rem,4.4vw,2.7rem);margin:0 0 .3em;font-weight:800;letter-spacing:-.025em;line-height:1.12}
h2{font-size:1.16rem;margin:56px 0 4px;font-weight:700;letter-spacing:-.01em}
h2 .n{font-family:var(--mono);font-size:.7rem;color:var(--gold);letter-spacing:.2em;display:block;margin-bottom:7px;font-weight:500}
.lede{color:var(--soft);font-size:1.1rem;margin:0 0 8px;max-width:62ch}
p{max-width:64ch;color:#c9d1e4}
.fig{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;margin:22px 0}
canvas{display:block;touch-action:none}
#stage{width:100%;height:420px;cursor:grab;border-radius:10px;background:radial-gradient(720px 420px at 50% 50%,#0e1730,#0b1122)}
.read{font-family:var(--mono);font-size:.76rem;color:var(--soft);margin:10px 0 0;text-align:center}.read b{color:var(--ink)}.dim{color:var(--dim)}
.ctl{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:14px;justify-content:center}
button{background:linear-gradient(180deg,#d9b45e,#a9832f);color:#161200;border:0;border-radius:9px;padding:8px 16px;font:700 .82rem var(--sans);cursor:pointer}
button.ghost{background:transparent;color:var(--soft);border:1px solid var(--line)}
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
  <p class="kicker"><span style="color:var(--goldb);font-size:1.05rem">&#934;</span><span>ferromotion · textbook · chapter 16</span>
    <span class="badge">rust → wasm · on-device</span></p>
  <h1>Turning to fit</h1>
  <p class="lede">Modern robots increasingly see the world as a cloud of little 3D Gaussians — the native output of splat reconstruction. FOCI checks collision directly on that map: the overlap between two Gaussians has a closed form, and because the robot's own Gaussians rotate with it, an elongated body can <i>turn</i> to slip through a gap it would smash into head-on. This page runs that overlap-integral collision on your device.</p>

  <h2><span class="n">01 — the map is Gaussians</span></h2>
  <p>A 3D Gaussian-splat reconstruction represents a scene as thousands of small anisotropic blobs. Rather than meshing that into boxes and spheres — throwing away the very shape information the splats encode — FOCI keeps the Gaussians and asks a cleaner question: how much do two Gaussian density fields <b>overlap</b>? That overlap integral has an exact closed form, a single Gaussian in the separation of their means under their <i>summed</i> covariance. It is smooth, cheap, and — the key property — it knows about <i>shape and orientation</i>, not just distance between centers.</p>

  <h2><span class="n">02 — orientation is a control input</span></h2>
  <p>Below, two obstacle Gaussians wall off a corridor, leaving a narrow <b>slot</b> in the middle. The robot is a single Gaussian, deliberately <i>long and thin</i>. <b>Drag it</b> toward the slot and <b>rotate it</b> with the slider. Its colour is its collision cost — green is clear, red is jammed. Pushed in broadside, the long axis spans the walls and the cost flares red; rotate it to line up with the corridor and it threads through, green.</p>
  <div class="fig">
    <canvas id="stage"></canvas>
    <div class="ctl">
      <button id="thread">Turn &amp; thread the slot</button>
      <button id="reset" class="ghost">Reset</button>
      <label>yaw <input type="range" id="yaw" min="0" max="180" step="1" value="0"/></label>
    </div>
    <div class="stats">
      <div class="stat"><div class="v" id="costv">—</div><div class="k">overlap collision cost</div></div>
      <div class="stat"><div class="v" id="yawv">—</div><div class="k">robot yaw</div></div>
      <div class="stat"><div class="v" id="statev">—</div><div class="k">slot</div></div>
    </div>
    <p class="read dim" style="margin-top:8px">a conservative box or sphere around this robot could never fit — only its true orientation-aware shape does</p>
  </div>

  <h2><span class="n">03 — why a box would fail here</span></h2>
  <p>The usual shortcut is to wrap the robot in a bounding box or a sphere and keep that clear of obstacles. But a bounding box of a long thin robot is nearly as wide as it is long — it can <i>never</i> fit through a slot narrower than the robot's length, no matter how you turn it. The robot physically fits; the conservative model says it doesn't. FOCI avoids that by scoring the actual Gaussian overlap, so the planner is free to <i>use</i> orientation as a way through — exactly what lets a legged robot slip sideways between two rocks.</p>
  <div class="callout">This is the whole point. Collision is not a property of a position alone; for a non-round body it is a property of a <b>pose</b>. A representation that forgets orientation forecloses solutions that exist. FOCI keeps orientation in the cost — analytically, differentiably — so turning-to-fit becomes just another direction the optimizer can descend.</div>

  <h2><span class="n">04 — the check</span>Head-on vs turned</h2>
  <p>On load, this page measured the collision cost of entering the slot broadside (yaw 0°) versus turned to align with it (yaw 90°), at the slot's center:</p>
  <table>
    <tr><td>collision cost head-on (yaw 0°)</td><td id="scHead">…</td></tr>
    <tr><td>collision cost turned to align (yaw 90°)</td><td id="scTurn">…</td></tr>
    <tr><td>turning cuts the cost by</td><td id="scRatio">…</td></tr>
    <tr><td>verdict</td><td id="scVerdict">…</td></tr>
  </table>
  <p>Same position, same robot, same obstacles — only the orientation changed, and the collision cost collapsed. A model that scored collision from position alone would report the same value for both and never find the way through.</p>

  <h2><span class="n">05 — the point</span>Collision belongs to the pose</h2>
  <div class="verdict">
    <div class="big">Keep the shape; let the body turn to fit.</div>
    <p>Score collision as the overlap of the actual Gaussians — the map's and the robot's — and orientation stays in the cost where a planner can exploit it. A tight slot is not a wall; it is an invitation to rotate.</p>
  </div>
  <p>FOCI closes the geometry side of the book the way the planning chapters closed the optimization side: keep the real structure of the problem instead of a conservative surrogate, and the solutions hiding in that structure become reachable. On a real Gaussian-splat map of a room, the same overlap integral scores a whole robot against hundreds of thousands of splats — and the robot turns to fit.</p>

  <p class="note"><b>What you just drove:</b> the <span style="color:var(--soft)">foci</span> overlap-integral collision from <span style="color:var(--soft)">ferromotion-core</span>, compiled to WebAssembly — the same code the native tools link against. Every frame sums the closed-form overlap of the robot's Gaussian against each obstacle Gaussian at the current pose; nothing precomputed.<br/><br/>
  <b>Verified in the library:</b> the closed-form overlap integral matches Monte-Carlo integration of the product of the two Gaussian densities (rel err &lt;1%); the kernel is 1 at coincident means and decays monotonically to ~0; its gradient matches finite differences; and turning an elongated robot to align with a slot cuts its collision cost to under a quarter of the head-on value. Each is a test in <span style="color:var(--soft)">cargo test</span>, not a claim in prose. See also <a href="/assets/sims/rocket-landing">ch.15 — landing a rocket</a> · <a href="/assets/sims/textbook">the full textbook</a>.<br/><br/>
  <b>Institute for Physical AI</b> · <a href="https://github.com/dcharlot-physicalai-bmi/ferromotion">the Rust library</a> · <a href="https://crates.io/crates/ferromotion">crates.io</a></p>
</div>
<script type="module">
${glue}
const WASM_B64="${wasmB64}";
${APP}
</script></body></html>`;

const outFile = path.join(__dirname, "..", "..", "v2", "public", "assets", "sims", "gaussian-collision.html");
fs.writeFileSync(outFile, html);
console.log(`wrote ${outFile} (${(html.length / 1024).toFixed(0)} KB, wasm ${(wasmB64.length / 1024).toFixed(0)} KB b64)`);

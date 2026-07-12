// Assemble a fully self-contained demo page: inline the wasm-bindgen glue + base64-embed the wasm,
// so the page runs the ferromotion Rust→WASM kinematics core with no server/fetch/install.
const fs = require("fs");
const path = require("path");

const pkg = path.join(__dirname, "..", "crates", "ferromotion-wasm", "pkg");
const glue = fs.readFileSync(path.join(pkg, "ferromotion_wasm.js"), "utf8");
const wasmB64 = fs.readFileSync(path.join(pkg, "ferromotion_wasm_bg.wasm")).toString("base64");

const APP = `
function b64ToBytes(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}

const LINK=0.8, N=4, TOOL=0.5, ORIGINX=[0,LINK,LINK,LINK];
let chain, q=[0.35,0.5,-0.6,0.35];
let mode="follow";
let target=[1.7,0.7];
let goal=[-1.5,1.3];
let obstacle={x:0.2,y:1.55,r:0.32};
let dragging=null, playing=false;

const cv=document.getElementById("c"), ctx=cv.getContext("2d");
let W,H,scale,ox,oy;
function resize(){const r=cv.getBoundingClientRect();const dpr=Math.min(devicePixelRatio||1,2);cv.width=r.width*dpr;cv.height=r.height*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);W=r.width;H=r.height;scale=Math.min(W,H)/6.6;ox=W*0.5;oy=H*0.66;draw();}
function w2s(x,y){return [ox+x*scale, oy-y*scale];}
function s2w(px,py){return [(px-ox)/scale,(oy-py)/scale];}

function fkPoints(qq){let x=0,y=0,a=0;const pts=[[x,y]];for(let i=0;i<N;i++){x+=ORIGINX[i]*Math.cos(a);y+=ORIGINX[i]*Math.sin(a);a+=qq[i];pts.push([x,y]);}x+=TOOL*Math.cos(a);y+=TOOL*Math.sin(a);pts.push([x,y]);return pts;}

function draw(){
  ctx.clearRect(0,0,W,H);
  // ground grid
  ctx.strokeStyle="rgba(120,140,180,0.10)";ctx.lineWidth=1;
  for(let gx=-3;gx<=3;gx++){const a=w2s(gx,-3),b=w2s(gx,3);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();}
  for(let gy=-3;gy<=3;gy++){const a=w2s(-3,gy),b=w2s(3,gy);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();}

  if(mode==="reach"){
    // obstacle
    const oc=w2s(obstacle.x,obstacle.y);
    ctx.beginPath();ctx.arc(oc[0],oc[1],obstacle.r*scale,0,7);ctx.fillStyle="rgba(220,90,90,0.16)";ctx.fill();
    ctx.strokeStyle="rgba(230,110,110,0.85)";ctx.lineWidth=2;ctx.stroke();
    ctx.fillStyle="rgba(230,140,140,0.9)";ctx.font="600 12px ui-sans-serif,system-ui";ctx.fillText("obstacle",oc[0]-24,oc[1]+4);
    // goal
    const gc=w2s(goal[0],goal[1]);
    ctx.beginPath();ctx.arc(gc[0],gc[1],9,0,7);ctx.strokeStyle="#f0cf82";ctx.lineWidth=2.5;ctx.stroke();
    ctx.beginPath();ctx.arc(gc[0],gc[1],2.5,0,7);ctx.fillStyle="#f0cf82";ctx.fill();
    ctx.fillStyle="#d9b45e";ctx.fillText("goal",gc[0]+12,gc[1]+4);
  } else {
    const tc=w2s(target[0],target[1]);
    ctx.strokeStyle="#f0cf82";ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(tc[0]-9,tc[1]);ctx.lineTo(tc[0]+9,tc[1]);ctx.moveTo(tc[0],tc[1]-9);ctx.lineTo(tc[0],tc[1]+9);ctx.stroke();
  }

  // arm
  const pts=fkPoints(q).map(p=>w2s(p[0],p[1]));
  ctx.lineCap="round";ctx.lineJoin="round";
  ctx.strokeStyle="rgba(217,180,94,0.35)";ctx.lineWidth=13;
  ctx.beginPath();ctx.moveTo(pts[0][0],pts[0][1]);for(let i=1;i<pts.length;i++)ctx.lineTo(pts[i][0],pts[i][1]);ctx.stroke();
  ctx.strokeStyle="#d9b45e";ctx.lineWidth=5;
  ctx.beginPath();ctx.moveTo(pts[0][0],pts[0][1]);for(let i=1;i<pts.length;i++)ctx.lineTo(pts[i][0],pts[i][1]);ctx.stroke();
  for(let i=0;i<pts.length-1;i++){ctx.beginPath();ctx.arc(pts[i][0],pts[i][1],5.5,0,7);ctx.fillStyle="#0a0f1e";ctx.fill();ctx.strokeStyle="#f0cf82";ctx.lineWidth=2;ctx.stroke();}
  // base
  ctx.beginPath();ctx.arc(pts[0][0],pts[0][1],9,0,7);ctx.fillStyle="#161f3a";ctx.fill();ctx.strokeStyle="#8aa0c8";ctx.lineWidth=2;ctx.stroke();
  // tool tip
  const tip=pts[pts.length-1];
  ctx.beginPath();ctx.arc(tip[0],tip[1],6,0,7);ctx.fillStyle="#f0cf82";ctx.fill();
}

function stepIK(){
  const out=chain.retarget_step(new Uint32Array([N]),new Float64Array([TOOL,0,0]),new Float64Array([target[0],target[1],0]),new Float64Array(q),0.01,0.0);
  q=Array.from(out.slice(0,N));
}

function playReach(){
  if(playing)return;
  const flat=chain.plan_reach(new Float64Array(q),new Float64Array([goal[0],goal[1],0]),new Float64Array([obstacle.x,obstacle.y,0,obstacle.r]),44);
  const steps=flat.length/N;
  playing=true;let k=0;const t0=performance.now();
  function frame(now){
    const kk=Math.min(steps-1,Math.floor((now-t0)/1400*(steps-1)));
    q=Array.from(flat.slice(kk*N,kk*N+N));draw();
    if(kk<steps-1){requestAnimationFrame(frame);}else{playing=false;}
  }
  requestAnimationFrame(frame);
}

function pick(px,py){
  if(mode!=="reach")return null;
  const gc=w2s(goal[0],goal[1]);if(Math.hypot(px-gc[0],py-gc[1])<16)return "goal";
  const oc=w2s(obstacle.x,obstacle.y);if(Math.hypot(px-oc[0],py-oc[1])<obstacle.r*scale+8)return "obs";
  return null;
}
function pointer(e){
  const r=cv.getBoundingClientRect();const px=e.clientX-r.left,py=e.clientY-r.top;const w=s2w(px,py);
  if(mode==="follow"){target=w;stepIK();draw();}
  else if(dragging==="goal"){goal=w;draw();}
  else if(dragging==="obs"){obstacle.x=w[0];obstacle.y=w[1];draw();}
}
cv.addEventListener("pointerdown",e=>{const r=cv.getBoundingClientRect();dragging=pick(e.clientX-r.left,e.clientY-r.top);cv.setPointerCapture(e.pointerId);pointer(e);});
cv.addEventListener("pointermove",e=>{if(mode==="follow"||dragging)pointer(e);});
cv.addEventListener("pointerup",e=>{dragging=null;});

function setMode(m){mode=m;document.getElementById("mFollow").classList.toggle("on",m==="follow");document.getElementById("mReach").classList.toggle("on",m==="reach");document.getElementById("hint").textContent=m==="follow"?"Move your cursor — the arm solves inverse kinematics to follow the tip, live.":"Drag the goal and the obstacle, then Plan & play — it optimizes a smooth trajectory that routes around the obstacle.";document.getElementById("play").style.display=m==="reach"?"inline-block":"none";draw();}
document.getElementById("mFollow").onclick=()=>setMode("follow");
document.getElementById("mReach").onclick=()=>setMode("reach");
document.getElementById("play").onclick=playReach;

async function main(){
  await __wbg_init(b64ToBytes(WASM_B64));
  chain=new Chain();
  chain.add_revolute(0,0,0, 1,0,0,0, 0,0,1);
  chain.add_revolute(LINK,0,0, 1,0,0,0, 0,0,1);
  chain.add_revolute(LINK,0,0, 1,0,0,0, 0,0,1);
  chain.add_revolute(LINK,0,0, 1,0,0,0, 0,0,1);
  chain.set_tool(TOOL,0,0, 1,0,0,0);
  window.__ferromotion_ready=true;
  addEventListener("resize",resize);resize();setMode("follow");
}
main();
`;

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ferromotion — kinematics in your browser</title>
<style>
:root{--ground:#0a0f1e;--panel:#111830;--line:#26324c;--ink:#eef1f8;--soft:#aeb6cc;--gold:#d9b45e;--goldb:#f0cf82;--mono:ui-monospace,"SF Mono",Menlo,monospace;--sans:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
*{box-sizing:border-box}html,body{margin:0;height:100%}body{background:radial-gradient(1200px 700px at 50% -10%,#12203f,var(--ground));color:var(--ink);font-family:var(--sans);display:flex;flex-direction:column;min-height:100%}
.wrap{max-width:920px;margin:0 auto;padding:clamp(20px,4vw,40px) 20px 40px;width:100%}
.kicker{font-family:var(--mono);font-size:.68rem;letter-spacing:.32em;text-transform:uppercase;color:var(--gold);display:flex;gap:10px;align-items:center;margin:0 0 8px}
h1{font-size:clamp(1.5rem,3.4vw,2.1rem);margin:0 0 .25em;font-weight:800;letter-spacing:-.02em}
.lede{color:var(--soft);margin:0 0 18px;max-width:64ch;font-size:1rem}
.stage{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:16px;overflow:hidden;aspect-ratio:16/10}
canvas{width:100%;height:100%;display:block;touch-action:none;cursor:crosshair}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:16px 0 6px}
.seg{display:inline-flex;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.seg button{background:transparent;color:var(--soft);border:0;padding:9px 15px;font:600 .86rem var(--sans);cursor:pointer}
.seg button.on{background:linear-gradient(180deg,rgba(217,180,94,.22),rgba(217,180,94,.08));color:var(--goldb)}
#play{background:linear-gradient(180deg,#d9b45e,#a9832f);color:#161200;border:0;border-radius:10px;padding:9px 16px;font:700 .86rem var(--sans);cursor:pointer}
.hint{color:var(--soft);font-size:.88rem;margin:4px 0 0}
.badge{margin-left:auto;font-family:var(--mono);font-size:.66rem;color:var(--soft);border:1px solid var(--line);border-radius:999px;padding:5px 11px}
.note{color:#727d99;font-family:var(--mono);font-size:.7rem;margin-top:22px;border-top:1px solid var(--line);padding-top:14px}
.note b{color:var(--gold)}
</style></head><body>
<div class="wrap">
  <p class="kicker"><span style="color:var(--goldb);font-size:1.05rem">Φ</span> ferromotion · Rust → WebAssembly</p>
  <h1>Robot kinematics, running in your browser</h1>
  <p class="lede">A live inverse-kinematics and trajectory-optimization core — ported to Rust, compiled to WebAssembly. Everything below is computed on-device by the same solver the native tools use. No server, no install.</p>
  <div class="stage"><canvas id="c"></canvas><span class="badge" id="badge">wasm · on-device</span></div>
  <div class="controls">
    <span class="seg"><button id="mFollow" class="on">Follow (IK)</button><button id="mReach">Reach around obstacle</button></span>
    <button id="play" style="display:none">▶ Plan &amp; play</button>
  </div>
  <p class="hint" id="hint"></p>
  <p class="note"><b>What you're driving:</b> a 4-joint arm. <b>Follow</b> solves inverse kinematics every frame (Levenberg–Marquardt over composable costs). <b>Reach</b> optimizes a whole smooth trajectory that routes the tool around the obstacle (block-tridiagonal trajectory solver + sphere-collision costs). Same Rust crate, native and in the browser.</p>
</div>
<script type="module">
${glue}
const WASM_B64="${wasmB64}";
${APP}
</script></body></html>`;

const outDir = path.join(__dirname, "..", "demo");
fs.mkdirSync(outDir, { recursive: true });
const outFile = path.join(outDir, "index.html");
fs.writeFileSync(outFile, html);
console.log(`wrote ${outFile} (${(html.length / 1024).toFixed(0)} KB, wasm ${(wasmB64.length / 1024).toFixed(0)} KB b64)`);

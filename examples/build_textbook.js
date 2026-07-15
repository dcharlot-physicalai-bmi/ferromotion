// Assemble the interactive-textbook chapter: inline the wasm-bindgen glue + base64-embed the wasm,
// so the page runs the real ferromotion Hill muscle on-device with no server/fetch/install.
// The reader isn't watching a reimplementation of the library — they are driving the library.
const fs = require("fs");
const path = require("path");

const pkg = path.join(__dirname, "..", "crates", "ferromotion-wasm", "pkg");
const glue = fs.readFileSync(path.join(pkg, "ferromotion_wasm.js"), "utf8");
const wasmB64 = fs.readFileSync(path.join(pkg, "ferromotion_wasm_bg.wasm")).toString("base64");

const APP = String.raw`
function b64ToBytes(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
const MASS=2.0, LOAD=300.0;
const fmt=(x,n=2)=>Number(x).toFixed(n);

/* ---------- generic canvas helper ---------- */
function setup(cv){
  const r=cv.getBoundingClientRect(), dpr=Math.min(devicePixelRatio||1,2);
  cv.width=r.width*dpr; cv.height=r.height*dpr;
  const ctx=cv.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx,W:r.width,H:r.height};
}

/* ---------- 1. the force-velocity curve ---------- */
const fvCv=document.getElementById("fv");
let fvRig, fvHover=0.0;
function drawFv(){
  const {ctx,W,H}=setup(fvCv);
  const vmax=fvRig.v_max(), pts=Array.from(fvRig.fv_curve(161));
  const pad={l:52,r:16,t:18,b:34};
  const x2p=v=>pad.l+((v+vmax)/(2*vmax))*(W-pad.l-pad.r);
  const y2p=f=>H-pad.b-(f/1.7)*(H-pad.t-pad.b);
  ctx.clearRect(0,0,W,H);
  // axes
  ctx.strokeStyle="rgba(120,140,180,.18)"; ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const f=i*0.425;const y=y2p(f);ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();
    ctx.fillStyle="#727d99";ctx.font="500 10px ui-monospace,monospace";ctx.fillText(f.toFixed(1),8,y+3);}
  // isometric + zero-velocity reference
  ctx.setLineDash([3,4]); ctx.strokeStyle="rgba(174,182,204,.4)";
  ctx.beginPath();ctx.moveTo(x2p(0),pad.t);ctx.lineTo(x2p(0),H-pad.b);ctx.stroke(); ctx.setLineDash([]);
  // shading: shortening vs lengthening
  ctx.fillStyle="rgba(217,180,94,.05)"; ctx.fillRect(x2p(0),pad.t,W-pad.r-x2p(0),H-pad.b-pad.t);
  // the curve
  ctx.beginPath();
  pts.forEach((f,i)=>{const v=-vmax+2*vmax*i/(pts.length-1);const X=x2p(v),Y=y2p(f);i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});
  ctx.strokeStyle="#d9b45e"; ctx.lineWidth=2.5; ctx.stroke();
  // hover readout: the slope IS the damping
  const fh=fvRig.fv_at(fvHover), e=1e-5;
  const slope=(fvRig.fv_at(fvHover+e)-fvRig.fv_at(fvHover-e))/(2*e);
  const hx=x2p(fvHover), hy=y2p(fh);
  // tangent line — the thing that matters
  const dx=52, dfp=slope*(dx/((W-pad.l-pad.r)/(2*vmax)));
  ctx.save(); ctx.beginPath(); ctx.rect(pad.l,pad.t,W-pad.l-pad.r,H-pad.t-pad.b); ctx.clip();
  ctx.beginPath();
  ctx.moveTo(hx-dx, y2p(fh-dfp)); ctx.lineTo(hx+dx, y2p(fh+dfp));
  ctx.strokeStyle="rgba(240,207,130,.85)"; ctx.lineWidth=1.5; ctx.setLineDash([5,3]); ctx.stroke(); ctx.setLineDash([]);
  ctx.restore();
  ctx.beginPath();ctx.arc(hx,hy,5,0,7);ctx.fillStyle="#f0cf82";ctx.fill();
  ctx.fillStyle="#727d99";ctx.font="500 10px ui-monospace,monospace";
  ctx.fillText("← shortening", pad.l+4, H-12);
  ctx.textAlign="right"; ctx.fillText("lengthening →", W-pad.r-4, H-12); ctx.textAlign="left";
  document.getElementById("fvRead").innerHTML =
    "v = <b>"+fmt(fvHover,3)+"</b> m/s &nbsp;·&nbsp; f<sub>v</sub> = <b>"+fmt(fh,3)+"</b> &nbsp;·&nbsp; slope ∂f<sub>v</sub>/∂v = <b style='color:#f0cf82'>+"+fmt(slope,3)+"</b>";
}
function fvPointer(e){
  const r=fvCv.getBoundingClientRect(); const vmax=fvRig.v_max();
  const t=(e.clientX-r.left-52)/(r.width-68);
  fvHover=Math.max(-vmax*0.98,Math.min(vmax*0.98,-vmax+2*vmax*t)); drawFv();
}
fvCv.addEventListener("pointermove",fvPointer);
fvCv.addEventListener("pointerdown",fvPointer);

/* ---------- 2. the main explorable: a mass on a muscle ---------- */
const mCv=document.getElementById("mus");
let rig, dragging=false, trail=[];
function drawMuscle(ctx,W,H,x,anchorY,massY,glow){
  const cx=W*0.5, halfW=13+Math.max(-0.03,Math.min(0.05,-x))*260;
  ctx.beginPath();
  ctx.moveTo(cx,anchorY);
  ctx.bezierCurveTo(cx-halfW,anchorY+22, cx-halfW,massY-22, cx,massY);
  ctx.bezierCurveTo(cx+halfW,massY-22, cx+halfW,anchorY+22, cx,anchorY);
  ctx.closePath();
  const g=ctx.createLinearGradient(cx-halfW,0,cx+halfW,0);
  g.addColorStop(0,"rgba(217,180,94,.10)"); g.addColorStop(.5,"rgba(240,207,130,"+(0.30+glow*0.42)+")"); g.addColorStop(1,"rgba(217,180,94,.10)");
  ctx.fillStyle=g; ctx.fill();
  ctx.strokeStyle="rgba(240,207,130,"+(0.55+glow*0.4)+")"; ctx.lineWidth=1.5; ctx.stroke();
  // fibers
  ctx.strokeStyle="rgba(240,207,130,.18)"; ctx.lineWidth=1;
  for(let i=-2;i<=2;i++){ if(!i) continue; const o=i*halfW*0.3;
    ctx.beginPath(); ctx.moveTo(cx,anchorY);
    ctx.bezierCurveTo(cx+o,anchorY+22, cx+o,massY-22, cx,massY); ctx.stroke(); }
}
function drawRig(){
  const {ctx,W,H}=setup(mCv);
  ctx.clearRect(0,0,W,H);
  const S=1200, l0=rig.l0(), x=rig.x(), eq=rig.equilibrium();
  const anchorY=40, massY=anchorY+(l0+x)*S;
  // ceiling
  ctx.fillStyle="#1a2440"; ctx.fillRect(W*0.5-70,anchorY-14,140,14);
  ctx.strokeStyle="#39456a"; ctx.lineWidth=1; ctx.strokeRect(W*0.5-70,anchorY-14,140,14);
  // equilibrium marker
  const eqY=anchorY+(l0+eq)*S;
  ctx.setLineDash([4,4]); ctx.strokeStyle="rgba(120,200,160,.5)"; ctx.beginPath();
  ctx.moveTo(W*0.5-118,eqY); ctx.lineTo(W*0.5+118,eqY); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle="rgba(120,200,160,.75)"; ctx.font="500 10px ui-monospace,monospace";
  ctx.fillText("equilibrium", W*0.5+64, eqY-6);
  // trail of recent motion — makes ringing legible
  trail.push(massY); if(trail.length>90) trail.shift();
  trail.forEach((y,i)=>{ const a=i/trail.length*0.16;
    ctx.fillStyle="rgba(240,207,130,"+a+")"; ctx.fillRect(W*0.5-30,y-1,60,2); });
  const glow=Math.min(1,Math.abs(rig.v())*1.6);
  drawMuscle(ctx,W,H,x,anchorY,massY,glow);
  // the mass
  ctx.fillStyle="#161f3a"; ctx.strokeStyle=dragging?"#f0cf82":"#8aa0c8"; ctx.lineWidth=2;
  ctx.beginPath(); ctx.roundRect(W*0.5-30,massY,60,34,7); ctx.fill(); ctx.stroke();
  ctx.fillStyle="#aeb6cc"; ctx.font="600 11px ui-sans-serif,system-ui"; ctx.textAlign="center";
  ctx.fillText("2 kg",W*0.5,massY+22); ctx.textAlign="left";
  // load arrow
  ctx.strokeStyle="rgba(174,182,204,.45)"; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(W*0.5,massY+34); ctx.lineTo(W*0.5,massY+62); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(W*0.5-5,massY+56); ctx.lineTo(W*0.5,massY+62); ctx.lineTo(W*0.5+5,massY+56); ctx.fillStyle="rgba(174,182,204,.45)"; ctx.fill();
  ctx.fillStyle="#727d99"; ctx.font="500 10px ui-monospace,monospace"; ctx.fillText("300 N load",W*0.5+9,massY+54);
  document.getElementById("rigRead").innerHTML =
    "stretch <b>"+fmt(x*100,2)+"</b> cm &nbsp;·&nbsp; velocity <b>"+fmt(rig.v(),3)+"</b> m/s &nbsp;·&nbsp; force <b>"+fmt(rig.force(),1)+"</b> N"+
    "<br><span class='dim'>measured K = <b style='color:#f0cf82'>"+fmt(rig.stiffness(),0)+"</b> N/m &nbsp;·&nbsp; measured B = <b style='color:#f0cf82'>"+fmt(rig.damping(),0)+"</b> N·s/m</span>";
}
function rigPointer(e){
  if(!dragging) return;
  const r=mCv.getBoundingClientRect();
  const y=(e.clientY-r.top-40)/1200 - rig.l0();
  rig.displace(Math.max(-0.05,Math.min(0.04,y)));
}
mCv.addEventListener("pointerdown",e=>{dragging=true;mCv.setPointerCapture(e.pointerId);rigPointer(e);});
mCv.addEventListener("pointermove",rigPointer);
mCv.addEventListener("pointerup",()=>{dragging=false;});
document.getElementById("kick").onclick=()=>{trail=[];rig.kick(0.55);};
document.getElementById("act").oninput=e=>{
  rig.set_activation(+e.target.value);
  document.getElementById("actVal").textContent=fmt(+e.target.value,2);
};
const fvToggle=document.getElementById("fvOn");
fvToggle.onchange=()=>{
  rig.set_fv(fvToggle.checked); trail=[];
  document.getElementById("fvState").textContent = fvToggle.checked
    ? "force-velocity ON — a muscle" : "force-velocity OFF — a plain spring";
  document.getElementById("fvState").style.color = fvToggle.checked ? "#7dd3a0" : "#e8836f";
};

/* ---------- 3. the controlled experiment ---------- */
const cCv=document.getElementById("cmp");
let mRig, nRig, tau=0.030, diverged=false, cmpTrailM=[], cmpTrailN=[];
const LOST=0.40; // metres of runaway that count as "lost control" for the visual
const DT=1e-4;
function rebuildNeural(){
  const k=mRig.stiffness(), b=mRig.damping();
  nRig=new DelayedRig(MASS,k,b,tau,DT);
  nRig.displace(0.0); diverged=false; cmpTrailN=[];
}
function drawCmp(){
  const {ctx,W,H}=setup(cCv);
  ctx.clearRect(0,0,W,H);
  const half=W/2, S=1100, l0=mRig.l0();
  ctx.strokeStyle="rgba(120,140,180,.14)"; ctx.beginPath(); ctx.moveTo(half,10); ctx.lineTo(half,H-10); ctx.stroke();
  // labels
  ctx.font="600 11px ui-sans-serif,system-ui"; ctx.textAlign="center";
  ctx.fillStyle="#d9b45e"; ctx.fillText("MUSCLE · physics, zero delay", half*0.5, 20);
  ctx.fillStyle=diverged?"#e8836f":"#8ab4f0"; ctx.fillText("NEURAL LOOP · same K, same B, delayed", half*1.5, 20);
  ctx.textAlign="left";
  const eq=mRig.equilibrium();
  // muscle side
  const my=44+(mRig.x()-eq)*S+70;
  cmpTrailM.push(my); if(cmpTrailM.length>70) cmpTrailM.shift();
  cmpTrailM.forEach((y,i)=>{ctx.fillStyle="rgba(240,207,130,"+(i/cmpTrailM.length*0.14)+")";ctx.fillRect(half*0.5-26,y-1,52,2);});
  ctx.setLineDash([4,4]); ctx.strokeStyle="rgba(120,200,160,.4)";
  ctx.beginPath();ctx.moveTo(half*0.5-70,114);ctx.lineTo(half*0.5+70,114);ctx.stroke();
  ctx.beginPath();ctx.moveTo(half*1.5-70,114);ctx.lineTo(half*1.5+70,114);ctx.stroke();ctx.setLineDash([]);
  drawMuscle(ctx,half,H,mRig.x(),40,my,Math.min(1,Math.abs(mRig.v())*1.6));
  ctx.fillStyle="#161f3a";ctx.strokeStyle="#d9b45e";ctx.lineWidth=2;
  ctx.beginPath();ctx.roundRect(half*0.5-26,my,52,28,6);ctx.fill();ctx.stroke();
  // neural side
  const nx=Math.max(-0.075,Math.min(0.075,nRig.x()));
  const ny=44+nx*S+70;
  cmpTrailN.push(ny); if(cmpTrailN.length>70) cmpTrailN.shift();
  cmpTrailN.forEach((y,i)=>{ctx.fillStyle=(diverged?"rgba(232,131,111,":"rgba(138,180,240,")+(i/cmpTrailN.length*0.14)+")";ctx.fillRect(half*1.5-26,y-1,52,2);});
  // actuator drawn as a rigid strut — it is a motor, not a tissue
  ctx.strokeStyle=diverged?"rgba(232,131,111,.6)":"rgba(138,180,240,.55)"; ctx.lineWidth=3;
  ctx.beginPath(); ctx.moveTo(half*1.5,40); ctx.lineTo(half*1.5,ny); ctx.stroke();
  for(let i=0;i<5;i++){const yy=40+(ny-40)*(i+0.5)/5;
    ctx.beginPath();ctx.moveTo(half*1.5-7,yy);ctx.lineTo(half*1.5+7,yy);ctx.lineWidth=1;ctx.stroke();}
  ctx.fillStyle="#161f3a";ctx.strokeStyle=diverged?"#e8836f":"#8ab4f0";ctx.lineWidth=2;
  ctx.beginPath();ctx.roundRect(half*1.5-26,ny,52,28,6);ctx.fill();ctx.stroke();
  // ceilings
  ctx.fillStyle="#1a2440";ctx.fillRect(half*0.5-56,26,112,14);ctx.fillRect(half*1.5-56,26,112,14);
  if(diverged){
    ctx.fillStyle="rgba(232,131,111,.95)"; ctx.font="700 13px ui-sans-serif,system-ui"; ctx.textAlign="center";
    ctx.fillText("LOST CONTROL", half*1.5, H-16); ctx.textAlign="left";
  }
  document.getElementById("cmpRead").innerHTML =
    "muscle offset <b>"+fmt((mRig.x()-eq)*100,2)+"</b> cm &nbsp;·&nbsp; neural offset <b style='color:"+(diverged?"#e8836f":"#8ab4f0")+"'>"+
    (diverged?"diverged":fmt(nRig.x()*100,2)+" cm")+"</b>";
}
document.getElementById("tau").oninput=e=>{
  tau=+e.target.value/1000;
  document.getElementById("tauVal").textContent=fmt(+e.target.value,2)+" ms";
  rebuildNeural();
};
document.getElementById("perturb").onclick=()=>{
  cmpTrailM=[];cmpTrailN=[];
  mRig.displace(mRig.equilibrium()); mRig.kick(0.35);
  rebuildNeural(); nRig.kick(0.35);
};

/* ---------- the page measures its own delay margin, on-device ---------- */
function measureCriticalDelay(k,b){
  const dt=1e-5;
  const stable=t=>{
    const d=new DelayedRig(MASS,k,b,t,dt); d.displace(0.0); d.kick(0.2);
    let peak=0;
    for(let i=0;i<60000;i++){
      d.step(dt);
      if(i>30000) peak=Math.max(peak,Math.abs(d.x()));
      if(!isFinite(d.x())||Math.abs(d.x())>1e3) return false;
    }
    return peak<0.01;
  };
  let lo=1e-6, hi=5e-3;
  if(!stable(lo)||stable(hi)) return null;
  for(let i=0;i<26;i++){const mid=0.5*(lo+hi); if(stable(mid)) lo=mid; else hi=mid;}
  return lo;
}

/* ---------- loop ---------- */
function frame(){
  if(!dragging){ for(let i=0;i<160;i++) rig.step(1e-4); }
  else { trail=[]; }
  drawRig();
  for(let i=0;i<24;i++){
    mRig.step(DT);
    if(!diverged){ nRig.step(DT); if(!isFinite(nRig.x())||Math.abs(nRig.x())>LOST) diverged=true; }
  }
  drawCmp();
  requestAnimationFrame(frame);
}

async function main(){
  await __wbg_init(b64ToBytes(WASM_B64));
  fvRig=new MuscleRig(MASS,LOAD,0.5);
  rig=new MuscleRig(MASS,LOAD,0.5);
  mRig=new MuscleRig(MASS,LOAD,0.5);
  mRig.displace(mRig.equilibrium());
  rebuildNeural();

  // live verification, computed here, now, on this device
  const K=mRig.stiffness(), B=mRig.damping();
  document.getElementById("hillRes").textContent=rig.hill_residual().toExponential(1);
  const tc=measureCriticalDelay(K,B);
  const u=(B*B+Math.sqrt(Math.pow(B,4)+4*MASS*MASS*K*K))/(2*MASS*MASS), w=Math.sqrt(u);
  const tAnalytic=Math.atan2(B*w,K)/w;
  document.getElementById("tcMeas").textContent=fmt(tc*1000,3)+" ms";
  document.getElementById("tcAna").textContent=fmt(tAnalytic*1000,3)+" ms";
  document.getElementById("tcErr").textContent=fmt(Math.abs(tc-tAnalytic)/tAnalytic*100,2)+"%";
  document.getElementById("tcRatio").textContent=Math.round(0.030/tAnalytic)+"×";
  document.getElementById("kVal").textContent=fmt(K,0);
  document.getElementById("bVal").textContent=fmt(B,0);
  // mark the critical delay on the slider
  const s=document.getElementById("tau");
  document.getElementById("tcMark").style.left="calc("+(tc*1000/30*100)+"% - 1px)";

  addEventListener("resize",()=>{drawFv();});
  drawFv();
  // test hook — lets the verification harness drive the same objects the reader drives
  window.__lab={rig:()=>rig, mRig:()=>mRig, nRig:()=>nRig, diverged:()=>diverged, tau:()=>tau};
  window.__textbook_ready=true;
  frame();
}
main();
`;

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>The body is the controller — ferromotion textbook</title>
<meta name="description" content="An interactive chapter on morphological computation: why a muscle's force-velocity curve is damping no neural loop could deliver. Runs the real Rust library on-device."/>
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
canvas{width:100%;display:block;touch-action:none}
#fv{height:210px;cursor:ew-resize}
#mus{height:310px;cursor:grab}
#cmp{height:250px}
.read{font-family:var(--mono);font-size:.76rem;color:var(--soft);margin:10px 0 0;text-align:center}
.read b{color:var(--ink);font-weight:600}
.dim{color:var(--dim)}
.ctl{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:14px;justify-content:center}
button{background:linear-gradient(180deg,#d9b45e,#a9832f);color:#161200;border:0;border-radius:9px;padding:8px 16px;font:700 .82rem var(--sans);cursor:pointer}
button:active{transform:translateY(1px)}
label{font-family:var(--mono);font-size:.72rem;color:var(--soft);display:flex;align-items:center;gap:8px}
input[type=range]{accent-color:var(--gold);width:140px}
input[type=checkbox]{accent-color:var(--gold);width:16px;height:16px}
.slider-wrap{position:relative}
#tcMark{position:absolute;top:-4px;width:2px;height:22px;background:var(--red);pointer-events:none}
#tcMark::after{content:"τc";position:absolute;top:-13px;left:-6px;font:600 9px var(--mono);color:var(--red)}
.callout{border-left:2px solid var(--gold);padding:2px 0 2px 18px;margin:26px 0;color:var(--soft)}
.callout b{color:var(--goldb)}
.verdict{background:linear-gradient(180deg,rgba(217,180,94,.09),rgba(217,180,94,.02));border:1px solid rgba(217,180,94,.3);border-radius:14px;padding:22px 24px;margin:30px 0}
.verdict .big{font-size:2.5rem;font-weight:800;color:var(--goldb);letter-spacing:-.03em;line-height:1}
.verdict p{margin:.5em 0 0;color:var(--soft);max-width:58ch}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.75rem;margin:14px 0 0}
td{padding:7px 0;border-bottom:1px solid var(--line);color:var(--soft)}
td:last-child{text-align:right;color:var(--ink);font-weight:600}
.note{color:var(--dim);font-family:var(--mono);font-size:.7rem;margin-top:56px;border-top:1px solid var(--line);padding-top:16px;line-height:1.7}
.note b{color:var(--gold)}
.note a{color:var(--soft)}
.badge{font-family:var(--mono);font-size:.62rem;white-space:nowrap;color:var(--dim);border:1px solid var(--line);border-radius:999px;padding:4px 10px}
</style></head><body>
<div class="wrap">
  <p class="kicker"><span style="color:var(--goldb);font-size:1.05rem">&#934;</span><span>ferromotion · textbook · chapter 1</span>
    <span class="badge">rust → wasm · on-device</span></p>
  <h1>The body is the controller</h1>
  <p class="lede">A muscle rejects a disturbance before a nerve could carry the news. This chapter is about what that means for machines — and every number in it is computed on your device, right now, by the same Rust library the native tools use.</p>

  <h2><span class="n">01 — the problem</span>Feedback arrives late</h2>
  <p>Perturb a robot joint and the usual story begins: a sensor reports, a controller computes, a motor responds. Every stage costs time. Animals are worse off — a spinal reflex takes roughly <b>30 ms</b> to close, and anything routed through cortex is slower still. Yet a guinea fowl hits a hidden pothole mid-stride and simply keeps running. The recovery is over before the reflex signal has arrived.</p>
  <p>So the leg cannot be waiting for instructions. Something in the tissue is already doing the work — and we can be precise about what.</p>

  <h2><span class="n">02 — the mechanism</span>A muscle is not a motor</h2>
  <p>A motor takes a command and produces a torque. A muscle produces force as a product of three things: how hard it is activated, how long it currently is, and <b>how fast it is changing length</b>. That last factor is the one that matters here.</p>
  <div class="fig">
    <canvas id="fv"></canvas>
    <p class="read" id="fvRead"></p>
    <p class="read dim" style="margin-top:6px">drag across the curve — the dashed line is the slope</p>
  </div>
  <p>Drag along it and notice the slope is <b>positive everywhere</b>. Stretch the muscle faster and it pulls back harder; let it shorten and it gives way. Written down, that is <span style="font-family:var(--mono);color:var(--goldb)">∂F/∂v &gt; 0</span> — the definition of a damper.</p>
  <div class="callout">The muscle is not <i>computing</i> a damping force. It <b>is</b> a damper. The response is not fast; it is <b>immediate</b>, because it is the same event as the disturbance.</div>

  <h2><span class="n">03 — the explorable</span>Fixed activation, no controller</h2>
  <p>Below, a 2 kg mass hangs on a Hill muscle pulling against a 300 N load. The activation is <b>constant</b>. There is no controller in this simulation at all — no feedback, no reflex, nothing reading the state and deciding anything. Drag the mass anywhere and let go. Kick it.</p>
  <div class="fig">
    <canvas id="mus"></canvas>
    <p class="read" id="rigRead"></p>
    <div class="ctl">
      <button id="kick">Kick it</button>
      <label>activation <input type="range" id="act" min="0.25" max="1" step="0.01" value="0.5"/> <span id="actVal">0.50</span></label>
      <label><input type="checkbox" id="fvOn" checked/> force-velocity</label>
    </div>
    <p class="read" style="margin-top:8px"><span id="fvState" style="color:var(--green)">force-velocity ON — a muscle</span></p>
  </div>
  <p>It comes home and stays there. Now switch the force-velocity curve off — everything else identical, same stiffness, same activation, same load — and the mass rings like a bell and never stops. That single curve is the entire difference between a spring and a limb.</p>

  <h2><span class="n">04 — the experiment</span>Could a nervous system just do this?</h2>
  <p>The obvious objection: fine, but a fast enough controller could deliver the same stiffness and damping. So let's grant it exactly that. We measure the muscle's real impedance at its operating point — stiffness <b id="kVal">…</b> N/m and damping <b id="bVal">…</b> N·s/m, taken by finite differences off the live model — and hand those very numbers to a neural controller driving the same mass against the same load.</p>
  <p>Identical mechanical impedance. One difference: the controller acts on state from <b>τ</b> milliseconds ago.</p>
  <div class="fig">
    <canvas id="cmp"></canvas>
    <p class="read" id="cmpRead"></p>
    <div class="ctl">
      <button id="perturb">Perturb both</button>
      <label>delay τ <span class="slider-wrap"><input type="range" id="tau" min="0" max="30" step="0.05" value="30"/><span id="tcMark"></span></span> <span id="tauVal">30.00 ms</span></label>
    </div>
  </div>
  <p>Drag τ down. The neural loop holds only in a sliver near zero, marked <span style="color:var(--red);font-family:var(--mono)">τc</span> on the slider. Past it the same impedance that stabilizes the muscle tears the mass apart, because a correction computed from stale state arrives pointing the wrong way and pumps energy in.</p>
  <p>This is not a quirk of the simulation. It is the classical delay margin, and the page checks itself against it — bisecting for the critical delay in WebAssembly on load, then comparing with the analytic crossover of <span style="font-family:var(--mono);color:var(--goldb)">m·s²</span> under <span style="font-family:var(--mono);color:var(--goldb)">(K + B·s)·e<sup>−sτ</sup></span>:</p>
  <table>
    <tr><td>critical delay — measured here, by bisection</td><td id="tcMeas">…</td></tr>
    <tr><td>critical delay — analytic delay margin</td><td id="tcAna">…</td></tr>
    <tr><td>agreement</td><td id="tcErr">…</td></tr>
    <tr><td>Hill's hyperbola residual, live</td><td id="hillRes">…</td></tr>
  </table>

  <h2><span class="n">05 — the point</span>Not an optimization. A requirement.</h2>
  <div class="verdict">
    <div class="big" id="tcRatio">…</div>
    <p>A spinal reflex is that many times slower than the slowest delay this loop survives. The muscle's impedance is not merely easier to get from tissue than from a controller — <b>it is not available to a neural loop at all.</b></p>
  </div>
  <p>This inverts the usual reading of morphological computation. The body is not helping the controller by taking some load off it. The body is doing something the controller <i>cannot do</i>, at any gain, with any tuning, because the delay forbids it. What the nervous system sends is not a correction but a <b>setpoint</b> — the activation level — and the mechanics resolve everything faster than the message could have returned.</p>
  <p>For a machine the lesson is a design constraint rather than an inspiration. Rejecting disturbances at the millisecond scale is not a control problem you can solve with a faster loop or a better estimator. It is a decision that gets made when you choose the actuator. Series elasticity, tuned compliance, and the shape of a force-velocity curve are not padding around the real controller; at these timescales, they <i>are</i> the controller.</p>

  <p class="note"><b>What you just drove:</b> the <span style="color:var(--soft)">HillMuscle</span> from <span style="color:var(--soft)">ferromotion-control</span>, compiled to WebAssembly — the same code the native tools link against, not a reimplementation. Force-velocity follows Hill (1938); the model form follows Zajac (1989). Nothing here is precomputed: the impedance is measured off the model by finite differences, the critical delay is bisected in your browser, and the residual above is Hill's hyperbola checked live.<br/><br/>
  <b>Verified in the library:</b> Hill's hyperbola holds to 1e-12 · ∂F/∂v &gt; 0 across the range · the force-velocity branches are C¹ at v=0 by construction · fixed activation reaches the same equilibrium from both position and velocity perturbations · the measured critical delay matches the analytic margin to under 2%. Each is a test in <span style="color:var(--soft)">cargo test</span>, not a claim in prose.<br/><br/>
  <b>Institute for Physical AI</b> · <a href="https://github.com/dcharlot-physicalai-bmi/ferromotion">the Rust library</a> · <a href="https://crates.io/crates/ferromotion">crates.io</a></p>
</div>
<script type="module">
${glue}
const WASM_B64="${wasmB64}";
${APP}
</script></body></html>`;

const outFile = path.join(__dirname, "..", "..", "v2", "public", "assets", "sims", "morphological-computation.html");
fs.writeFileSync(outFile, html);
console.log(`wrote ${outFile} (${(html.length / 1024).toFixed(0)} KB, wasm ${(wasmB64.length / 1024).toFixed(0)} KB b64)`);

// Assemble the interactive-textbook chapter on algebraic connectivity: inline the wasm-bindgen glue +
// base64-embed the wasm, so the page runs the real ferromotion consensus protocol on-device. The
// reader edits a graph and watches λ₂ — computed from the Laplacian spectrum — turn out to be the
// exact rate at which the swarm agrees. Same self-contained pattern as build_textbook.js (chapter 1).
const fs = require("fs");
const path = require("path");

const pkg = path.join(__dirname, "..", "crates", "ferromotion-wasm", "pkg");
const glue = fs.readFileSync(path.join(pkg, "ferromotion_wasm.js"), "utf8");
const wasmB64 = fs.readFileSync(path.join(pkg, "ferromotion_wasm_bg.wasm")).toString("base64");

const APP = String.raw`
function b64ToBytes(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
const fmt=(x,n=2)=>Number(x).toFixed(n);
const TAU=Math.PI*2;

let lab, N=7, running=false, dragging=-1, pendingEdge=-1;
let spreadHist=[], tHist=[], simT=0;

/* ---------- graph presets (deterministic layouts, no RNG) ---------- */
function layoutRing(n){ const o=[]; for(let i=0;i<n;i++){const a=i/n*TAU - Math.PI/2; o.push([Math.cos(a),Math.sin(a)]);} return o; }
function preset(kind){
  const rings=layoutRing(N);
  lab = new ConsensusLab(N);
  // scatter agents around their ring slots so the collapse is visible
  for(let i=0;i<N;i++){
    const jx=Math.sin(i*1.7)*0.28, jy=Math.cos(i*2.3)*0.28;
    lab.set_agent(i, rings[i][0]*0.82+jx, rings[i][1]*0.82+jy);
  }
  const E=(i,j)=>lab.set_edge(((i%N)+N)%N, ((j%N)+N)%N, true);
  if(kind==='path'){ for(let i=0;i<N-1;i++) E(i,i+1); }
  else if(kind==='ring'){ for(let i=0;i<N;i++) E(i,i+1); }
  else if(kind==='star'){ for(let i=1;i<N;i++) E(0,i); }
  else if(kind==='complete'){ for(let i=0;i<N;i++) for(let j=i+1;j<N;j++) E(i,j); }
  else if(kind==='clusters'){ // two tight groups {0,1,2} and {3,4,5,6} joined by one fragile bridge
    // lay the two groups out as separated blobs so the split reads at a glance
    const A=[[-0.72,-0.34],[-0.86,0.22],[-0.44,0.26]];              // left triangle
    const B=[[0.5,-0.44],[0.86,-0.06],[0.82,0.42],[0.46,0.34]];     // right quad
    A.forEach((q,i)=>lab.set_agent(i,q[0],q[1]));
    B.forEach((q,i)=>lab.set_agent(3+i,q[0],q[1]));
    E(0,1);E(1,2);E(2,0);                 // triangle
    E(3,4);E(4,5);E(5,6);E(6,3);E(3,5);   // well-linked quad
    E(2,3);                               // the lone bridge — cut it to disconnect
  }
  spreadHist=[]; tHist=[]; simT=0; running=false;
  document.querySelectorAll('.seg button').forEach(b=>b.classList.toggle('on', b.dataset.k===kind));
  syncReadouts(); draw(); drawPlot();
}

/* ---------- geometry ---------- */
const gCv=document.getElementById("graph"), pCv=document.getElementById("plot");
function tf(cv){ const r=cv.getBoundingClientRect(); const s=Math.min(r.width,r.height)*0.42;
  return {ox:r.width/2, oy:r.height/2, s, W:r.width, H:r.height}; }
function w2s(cv,x,y){ const t=tf(cv); return [t.ox+x*t.s, t.oy+y*t.s]; }
function nodeAt(cv,px,py){ const t=tf(cv);
  for(let i=0;i<N;i++){ const [sx,sy]=w2s(cv,lab.x(i),lab.y(i)); if(Math.hypot(px-sx,py-sy)<16) return i; } return -1; }

/* ---------- draw the graph ---------- */
function draw(){
  const cv=gCv, ctx=cv.getContext("2d"); const t=tf(cv);
  const dpr=Math.min(devicePixelRatio||1,2);
  cv.width=t.W*dpr; cv.height=t.H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,t.W,t.H);
  const conn=lab.is_connected();
  // edges
  for(let i=0;i<N;i++) for(let j=i+1;j<N;j++){ if(lab.has_edge(i,j)){
    const [ax,ay]=w2s(cv,lab.x(i),lab.y(i)), [bx,by]=w2s(cv,lab.x(j),lab.y(j));
    ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by);
    ctx.strokeStyle=conn?"rgba(217,180,94,.34)":"rgba(232,131,111,.4)"; ctx.lineWidth=2; ctx.stroke();
  }}
  // centroid (the only point they can agree on) — draw before nodes
  const [cx,cy]=w2s(cv,lab.centroid_x(),lab.centroid_y());
  ctx.setLineDash([3,4]); ctx.strokeStyle="rgba(125,211,160,.7)"; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.arc(cx,cy,11,0,TAU); ctx.stroke(); ctx.setLineDash([]);
  ctx.beginPath(); ctx.arc(cx,cy,2.5,0,TAU); ctx.fillStyle="rgba(125,211,160,.9)"; ctx.fill();
  ctx.fillStyle="rgba(125,211,160,.8)"; ctx.font="500 10px ui-monospace,monospace"; ctx.fillText("centroid",cx+15,cy-9);
  // pending edge (reader picked a first node)
  if(pendingEdge>=0){ const [ax,ay]=w2s(cv,lab.x(pendingEdge),lab.y(pendingEdge));
    ctx.beginPath(); ctx.arc(ax,ay,15,0,TAU); ctx.strokeStyle="#f0cf82"; ctx.lineWidth=1.5; ctx.setLineDash([3,3]); ctx.stroke(); ctx.setLineDash([]); }
  // nodes
  for(let i=0;i<N;i++){ const [sx,sy]=w2s(cv,lab.x(i),lab.y(i)); const deg=lab.degree(i);
    ctx.beginPath(); ctx.arc(sx,sy,9,0,TAU);
    ctx.fillStyle= deg<0.5 ? "#3a2530" : "#161f3a"; // isolated node reads red-ish
    ctx.fill(); ctx.strokeStyle= i===dragging ? "#f0cf82" : (conn?"#d9b45e":"#e8836f"); ctx.lineWidth=2; ctx.stroke();
  }
}

/* ---------- draw the decay plot (log disagreement vs time) ---------- */
function drawPlot(){
  const cv=pCv, ctx=cv.getContext("2d"); const r=cv.getBoundingClientRect();
  const dpr=Math.min(devicePixelRatio||1,2); cv.width=r.width*dpr; cv.height=r.height*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  const W=r.width,H=r.height,pad={l:38,r:12,t:14,b:22};
  ctx.clearRect(0,0,W,H);
  const lam=lab.fiedler();
  const Tmax=Math.max(6, tHist.length?tHist[tHist.length-1]:6);
  const yTop=0.5, yBot=1e-4; // log range for spread
  const x2p=t=>pad.l+(t/Tmax)*(W-pad.l-pad.r);
  const y2p=s=>{ const c=Math.log10(Math.max(yBot,Math.min(yTop,s))); const c0=Math.log10(yTop),c1=Math.log10(yBot);
    return pad.t+(c0-c)/(c0-c1)*(H-pad.t-pad.b); };
  // gridlines (decades)
  ctx.font="500 9px ui-monospace,monospace";
  for(let e=0;e>=-4;e--){ const y=y2p(Math.pow(10,e)); ctx.strokeStyle="rgba(120,140,180,.13)"; ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();
    ctx.fillStyle="#5b6680"; ctx.fillText("1e"+e,4,y+3); }
  // theoretical slope −λ₂ from the first measured point
  if(spreadHist.length>1 && lam>1e-6){
    const t0=tHist[0], s0=spreadHist[0];
    ctx.beginPath();
    for(let k=0;k<=60;k++){ const t=t0+(Tmax-t0)*k/60; const s=s0*Math.exp(-lam*(t-t0));
      const X=x2p(t),Y=y2p(s); k?ctx.lineTo(X,Y):ctx.moveTo(X,Y); }
    ctx.strokeStyle="rgba(125,211,160,.85)"; ctx.lineWidth=1.5; ctx.setLineDash([5,3]); ctx.stroke(); ctx.setLineDash([]);
  }
  // measured trace
  ctx.beginPath();
  spreadHist.forEach((s,k)=>{ const X=x2p(tHist[k]),Y=y2p(s); k?ctx.lineTo(X,Y):ctx.moveTo(X,Y); });
  ctx.strokeStyle="#d9b45e"; ctx.lineWidth=2; ctx.stroke();
  ctx.fillStyle="#7dd3a0"; ctx.font="500 10px ui-monospace,monospace"; ctx.fillText("slope = −λ₂ (theory)", W-pad.r-138, pad.t+10);
  ctx.fillStyle="#d9b45e"; ctx.fillText("measured disagreement", W-pad.r-138, pad.t+24);
}

/* ---------- live decay-rate estimate (self-verification) ---------- */
function measuredRate(){
  // slope of ln(spread) over the recent half of the trace, once it is decaying cleanly
  const m=spreadHist.length; if(m<12) return NaN;
  const a=Math.floor(m*0.4), b=m-1;
  if(spreadHist[a]<1e-4 || spreadHist[b]<1e-5) return NaN;
  return -(Math.log(spreadHist[b])-Math.log(spreadHist[a]))/(tHist[b]-tHist[a]);
}
function syncReadouts(){
  const lam=lab.fiedler(), conn=lab.is_connected();
  document.getElementById("lam").textContent=fmt(lam,4);
  const cs=document.getElementById("connState");
  cs.textContent=conn?"connected":"DISCONNECTED"; cs.style.color=conn?"#7dd3a0":"#e8836f";
  document.getElementById("edges").textContent=lab.num_edges();
  const rate=measuredRate();
  const rr=document.getElementById("rate"), ag=document.getElementById("agree"), ag2=document.getElementById("agree2");
  document.getElementById("lam2disp").textContent=fmt(lam,4);
  if(isFinite(rate)&&lam>1e-6){ const pct=fmt(Math.abs(rate-lam)/lam*100,2)+"%";
    rr.textContent=fmt(rate,4); ag.textContent=pct; ag2.textContent=pct; }
  else { rr.textContent="—"; ag.textContent="—"; ag2.textContent="—"; }
  drawSpectrum(lam);
}
function drawSpectrum(lam){
  const cv=document.getElementById("spectrum"), ctx=cv.getContext("2d"); const r=cv.getBoundingClientRect();
  const dpr=Math.min(devicePixelRatio||1,2); cv.width=r.width*dpr; cv.height=r.height*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  const W=r.width,H=r.height; ctx.clearRect(0,0,W,H);
  const ev=Array.from(lab.eigenvalues()); const max=Math.max(1e-6,ev[ev.length-1]);
  const pad=10, bw=(W-2*pad)/ev.length;
  ev.forEach((e,i)=>{ const h=(e/max)*(H-20); const x=pad+i*bw;
    const isL2 = i===1;
    ctx.fillStyle= isL2 ? "#7dd3a0" : (i===0?"#5b6680":"rgba(217,180,94,.5)");
    ctx.fillRect(x+1,H-14-h,bw-2,h);
  });
  ctx.fillStyle="#7dd3a0"; ctx.font="600 10px ui-monospace,monospace"; ctx.textAlign="center";
  ctx.fillText("λ₂", pad+bw*1.5, H-2); ctx.textAlign="left";
  ctx.fillStyle="#5b6680"; ctx.fillText("λ₁=0", pad, H-2);
}

/* ---------- interaction ---------- */
gCv.addEventListener("pointerdown",e=>{ const r=gCv.getBoundingClientRect(); const px=e.clientX-r.left,py=e.clientY-r.top;
  const i=nodeAt(gCv,px,py);
  if(i>=0){
    if(e.shiftKey || document.getElementById("edgeMode").checked){
      if(pendingEdge<0){ pendingEdge=i; } else { if(pendingEdge!==i){ lab.toggle_edge(pendingEdge,i); spreadHist=[];tHist=[];simT=0; } pendingEdge=-1; syncReadouts(); }
    } else { dragging=i; gCv.setPointerCapture(e.pointerId); }
    draw();
  }
});
gCv.addEventListener("pointermove",e=>{ if(dragging<0) return; const r=gCv.getBoundingClientRect(); const t=tf(gCv);
  lab.set_agent(dragging, (e.clientX-r.left-t.ox)/t.s, (e.clientY-r.top-t.oy)/t.s);
  spreadHist=[];tHist=[];simT=0; syncReadouts(); draw(); drawPlot(); });
gCv.addEventListener("pointerup",()=>{ dragging=-1; });

document.getElementById("play").onclick=()=>{ running=!running; document.getElementById("play").textContent=running?"❚❚ Pause":"▶ Run consensus"; };
document.getElementById("reset").onclick=()=>{ const on=document.querySelector('.seg button.on'); preset(on?on.dataset.k:'ring'); };
document.querySelectorAll('.seg button').forEach(b=> b.onclick=()=>preset(b.dataset.k));

/* ---------- loop ---------- */
let acc=0;
function frame(){
  if(running){
    const dt=2e-3;
    for(let i=0;i<8;i++){ lab.step(dt); simT+=dt; }
    // sample the decay trace ~ every 0.04s of sim time
    acc+=8*dt;
    if(acc>=0.03){ acc=0; spreadHist.push(lab.spread()); tHist.push(simT);
      if(spreadHist.length>400){ spreadHist.shift(); tHist.shift(); } }
    if(lab.spread()<2e-5) running=false, document.getElementById("play").textContent="▶ Run consensus";
    syncReadouts();
  }
  draw(); drawPlot();
  requestAnimationFrame(frame);
}

/* ---------- the page verifies λ₂-is-the-rate for a canonical graph, on load ---------- */
function selfCheck(){
  const t=new ConsensusLab(6);
  const ring=layoutRing(6);
  for(let i=0;i<6;i++){ t.set_agent(i, ring[i][0]+Math.sin(i*1.7)*0.3, ring[i][1]+Math.cos(i*2.3)*0.3); }
  for(let i=0;i<5;i++) t.set_edge(i,i+1,true); // path
  const lam=t.fiedler();
  const dt=1e-4;
  for(let i=0;i<40000;i++) t.step(dt);       // warm-up: kill fast modes
  const s1=t.spread();
  for(let i=0;i<40000;i++) t.step(dt);
  const s2=t.spread();
  const measured=-(Math.log(s2/s1))/(40000*dt);
  document.getElementById("scLam").textContent=fmt(lam,5);
  document.getElementById("scMeas").textContent=fmt(measured,5);
  document.getElementById("scErr").textContent=fmt(Math.abs(measured-lam)/lam*100,3)+"%";
}

async function main(){
  await __wbg_init(b64ToBytes(WASM_B64));
  selfCheck();
  preset('ring');
  window.__consensus={lab:()=>lab, measuredRate, run:()=>{running=true;}, preset};
  window.__textbook_ready=true;
  addEventListener("resize",()=>{draw();drawPlot();syncReadouts();});
  frame();
}
main();
`;

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Agreement, and the one number behind it — ferromotion textbook</title>
<meta name="description" content="An interactive chapter on algebraic connectivity: why the Fiedler value λ₂ of a graph is not a bound on how fast a swarm agrees — it is the rate. Runs the real Rust consensus protocol on-device."/>
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
#graph{width:100%;height:380px;cursor:grab;border-radius:10px;background:radial-gradient(600px 380px at 50% 40%,#0e1730,#0b1122)}
#plot{width:100%;height:180px}
#spectrum{width:100%;height:66px}
.read{font-family:var(--mono);font-size:.76rem;color:var(--soft);margin:10px 0 0;text-align:center}
.read b{color:var(--ink);font-weight:600}
.dim{color:var(--dim)}
.ctl{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:14px;justify-content:center}
button{background:linear-gradient(180deg,#d9b45e,#a9832f);color:#161200;border:0;border-radius:9px;padding:8px 16px;font:700 .82rem var(--sans);cursor:pointer}
button.ghost{background:transparent;color:var(--soft);border:1px solid var(--line)}
button:active{transform:translateY(1px)}
label{font-family:var(--mono);font-size:.72rem;color:var(--soft);display:flex;align-items:center;gap:7px}
input[type=checkbox]{accent-color:var(--gold);width:15px;height:15px}
.seg{display:inline-flex;background:#0d1428;border:1px solid var(--line);border-radius:9px;overflow:hidden;flex-wrap:wrap}
.seg button{background:transparent;color:var(--soft);border:0;border-radius:0;padding:8px 13px;font:600 .76rem var(--sans)}
.seg button.on{background:linear-gradient(180deg,rgba(217,180,94,.22),rgba(217,180,94,.06));color:var(--goldb)}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}
.stat{background:#0d1428;border:1px solid var(--line);border-radius:10px;padding:9px 6px;text-align:center}
.stat .v{font-family:var(--mono);font-size:1rem;font-weight:700;color:var(--goldb)}
.stat .k{font-family:var(--mono);font-size:.58rem;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);margin-top:2px}
.callout{border-left:2px solid var(--gold);padding:2px 0 2px 18px;margin:26px 0;color:var(--soft)}
.callout b{color:var(--goldb)}
.verdict{background:linear-gradient(180deg,rgba(125,211,160,.08),rgba(125,211,160,.02));border:1px solid rgba(125,211,160,.3);border-radius:14px;padding:22px 24px;margin:30px 0}
.verdict .big{font-size:2.3rem;font-weight:800;color:var(--green);letter-spacing:-.03em;line-height:1.05}
.verdict p{margin:.5em 0 0;color:var(--soft);max-width:58ch}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.75rem;margin:14px 0 0}
td{padding:7px 0;border-bottom:1px solid var(--line);color:var(--soft)}
td:last-child{text-align:right;color:var(--ink);font-weight:600}
.note{color:var(--dim);font-family:var(--mono);font-size:.7rem;margin-top:56px;border-top:1px solid var(--line);padding-top:16px;line-height:1.7}
.note b{color:var(--gold)}.note a{color:var(--soft)}
.badge{font-family:var(--mono);font-size:.62rem;white-space:nowrap;color:var(--dim);border:1px solid var(--line);border-radius:999px;padding:4px 10px}
</style></head><body>
<div class="wrap">
  <p class="kicker"><span style="color:var(--goldb);font-size:1.05rem">&#934;</span><span>ferromotion · textbook · chapter 2</span>
    <span class="badge">rust → wasm · on-device</span></p>
  <h1>Agreement, and the one number behind it</h1>
  <p class="lede">A swarm in which every agent can see only its neighbours still comes to agree — and how fast it does is governed entirely by a single number of the network, the Fiedler value λ₂. Everything below is computed on your device by the same Rust consensus code the native tools run.</p>

  <h2><span class="n">01 — the setup</span>Local rules, global agreement</h2>
  <p>Give each agent one instruction: <i>drift toward the average of your neighbours</i>. No agent sees the whole network, no one is in charge, no one holds the target. Written for the whole swarm at once, that rule is exactly <span style="font-family:var(--mono);color:var(--goldb)">ẋ = −L·x</span>, where <span style="font-family:var(--mono);color:var(--goldb)">L = D − A</span> is the graph Laplacian — the degree of each node minus who it links to.</p>
  <p>Below is a live swarm. Drag the agents around; toggle <b>edge mode</b> and click two of them to add or cut a link; pick a network shape. Then run consensus and watch them collapse.</p>

  <div class="fig">
    <canvas id="graph"></canvas>
    <div class="ctl" style="margin-top:12px">
      <span class="seg">
        <button data-k="ring" class="on">Ring</button><button data-k="path">Path</button><button data-k="star">Star</button><button data-k="complete">Complete</button><button data-k="clusters">Two clusters</button>
      </span>
    </div>
    <div class="ctl">
      <button id="play">▶ Run consensus</button>
      <button id="reset" class="ghost">Reset</button>
      <label><input type="checkbox" id="edgeMode"/> edge mode (click two agents)</label>
    </div>
    <div class="stats">
      <div class="stat"><div class="v" id="lam">—</div><div class="k">λ₂ (Fiedler)</div></div>
      <div class="stat"><div class="v" id="connState" style="font-size:.8rem">—</div><div class="k">connectivity</div></div>
      <div class="stat"><div class="v" id="edges">—</div><div class="k">edges</div></div>
      <div class="stat"><div class="v" id="agree">—</div><div class="k">rate vs λ₂</div></div>
    </div>
  </div>

  <h2><span class="n">02 — the one number</span>λ₂ is the rate, not a bound on it</h2>
  <p>The Laplacian has a whole spectrum of eigenvalues. The smallest is always zero. The <b>second-smallest</b>, λ₂ — the Fiedler value — is the one that matters: the swarm's disagreement decays as <span style="font-family:var(--mono);color:var(--goldb)">e<sup>−λ₂ t</sup></span>. Not slower than that, not a worst case — that rate, exactly.</p>
  <div class="fig">
    <canvas id="spectrum"></canvas>
    <canvas id="plot"></canvas>
    <p class="read dim" style="margin-top:8px">run consensus above · the gold trace is the measured disagreement, the green dashed line is <span style="color:var(--green)">e<sup>−λ₂ t</sup></span> — they lie on top of each other</p>
    <div class="stats" style="grid-template-columns:repeat(3,1fr)">
      <div class="stat"><div class="v" id="lam2disp" style="font-size:.8rem">see above</div><div class="k">λ₂ from spectrum</div></div>
      <div class="stat"><div class="v" id="rate">—</div><div class="k">measured decay rate</div></div>
      <div class="stat"><div class="v" id="agree2" style="font-size:.8rem">see λ₂ tile</div><div class="k">agreement</div></div>
    </div>
  </div>
  <p>Pick <b>Complete</b> and the whole spectrum lifts — everyone talks to everyone, λ₂ is large, agreement is near-instant. Pick <b>Path</b> and λ₂ is tiny; the same rule, but news crawls end to end. You are not changing the agents. You are changing one eigenvalue of the wiring, and it sets the clock.</p>

  <h2><span class="n">03 — connectivity</span>Cut one link and agreement becomes impossible</h2>
  <p>Choose <b>Two clusters</b>: two tight groups joined by a single bridge. Run it — they still agree, but slowly, because that lone edge is the whole conversation between the halves. Now switch on edge mode and cut the bridge.</p>
  <div class="callout">The moment the graph falls into two pieces, <b>λ₂ drops to exactly zero</b>. Each half now collapses to its <i>own</i> average and the two averages never meet. Connectivity is not a nicety of consensus — it is the precondition. λ₂ &gt; 0 <b>is</b> the statement that the network is connected.</div>

  <h2><span class="n">04 — the invariant</span>They can only agree on where they already were</h2>
  <p>Watch the green centroid marker as the swarm collapses: it never moves. Because every row of L sums to zero, the average position is conserved to machine precision through the entire run. Consensus cannot invent a destination — it can only discover the one the swarm already averaged to. A leaderless network has no way to agree on anything but its own centre of mass.</p>

  <h2><span class="n">05 — the point</span>The topology is the algorithm</h2>
  <div class="verdict">
    <div class="big">λ₂ is the rate.</div>
    <p>Not an estimate of it, not an upper bound — the exact exponential rate at which a leaderless swarm reaches agreement. Verified on your device against the live simulation:</p>
  </div>
  <table>
    <tr><td>Fiedler value λ₂ — from the Laplacian spectrum</td><td id="scLam">…</td></tr>
    <tr><td>decay rate — measured from a running swarm</td><td id="scMeas">…</td></tr>
    <tr><td>agreement</td><td id="scErr">…</td></tr>
  </table>
  <p>For a fleet of robots, a sensor mesh, or a formation of drones, this is the design lever. You do not tune agreement by making each agent cleverer; you tune it by choosing who talks to whom. Add a link and λ₂ rises and the fleet syncs faster; the same protocol on a better-connected graph is simply a faster algorithm. Formation control is this exact protocol run on the <i>offsets</i> from a target shape — which is why a well-connected formation holds together, and a stringy one wobbles.</p>

  <p class="note"><b>What you just drove:</b> the <span style="color:var(--soft)">Graph</span> and <span style="color:var(--soft)">consensus_step</span> from <span style="color:var(--soft)">ferromotion-control</span>, compiled to WebAssembly — the same code the native tools link against, not a reimplementation. λ₂ and the whole spectrum come from the Laplacian's eigendecomposition; the decay rate is fit from the live trace; nothing is precomputed.<br/><br/>
  <b>Verified in the library:</b> the measured decay rate matches λ₂ to under 2% · the centroid is conserved to 1e-12 · a disconnected graph has λ₂ = 0 and each component agrees only within itself · adding an edge never lowers λ₂. Each is a test in <span style="color:var(--soft)">cargo test</span>, not a claim in prose. See also <a href="/assets/sims/morphological-computation">chapter 1 — the body is the controller</a>.<br/><br/>
  <b>Institute for Physical AI</b> · <a href="https://github.com/dcharlot-physicalai-bmi/ferromotion">the Rust library</a> · <a href="https://crates.io/crates/ferromotion">crates.io</a></p>
</div>
<script type="module">
${glue}
const WASM_B64="${wasmB64}";
${APP}
</script></body></html>`;

const outFile = path.join(__dirname, "..", "..", "v2", "public", "assets", "sims", "algebraic-connectivity.html");
fs.writeFileSync(outFile, html);
console.log(`wrote ${outFile} (${(html.length / 1024).toFixed(0)} KB, wasm ${(wasmB64.length / 1024).toFixed(0)} KB b64)`);

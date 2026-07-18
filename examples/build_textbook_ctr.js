// Assemble the interactive-textbook chapter on planning THROUGH contact (Contact Trust Region). Inline the
// wasm-bindgen glue + base64-embed the wasm so the page runs the real ferromotion PusherSlider /
// SmoothedContact on-device. Two figures: (A) the smoothed contact force morphing toward rigid as κ grows;
// (B) the headline — at a contact transition, the linear model of the slider's next position stays honest
// inside the contact trust region and diverges outside it. Same self-contained pattern as the other chapters.
const fs = require("fs");
const path = require("path");

const pkg = path.join(__dirname, "..", "crates", "ferromotion-wasm", "pkg");
const glue = fs.readFileSync(path.join(pkg, "ferromotion_wasm.js"), "utf8");
const wasmB64 = fs.readFileSync(path.join(pkg, "ferromotion_wasm_bg.wasm")).toString("base64");

const APP = String.raw`
function b64ToBytes(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
const fmt=(x,n=3)=>Number(x).toFixed(n);
let lab, du=0.5;

// ---- figure A: the smoothed contact force ----
const fc=document.getElementById("forcefig");
function drawForce(){
  const r=fc.getBoundingClientRect(); const dpr=Math.min(devicePixelRatio||1,2);
  fc.width=r.width*dpr; fc.height=r.height*dpr; const ctx=fc.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
  const w=r.width,h=r.height,pad=34; ctx.clearRect(0,0,w,h);
  const d0=-0.15,d1=0.35; const fmax=lab.rigid_force(d1)*1.05;
  const X=d=>pad+(d-d0)/(d1-d0)*(w-2*pad); const Y=f=>h-pad-(f/fmax)*(h-2*pad);
  // axes
  ctx.strokeStyle="rgba(120,140,180,.3)"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(X(0),pad-6); ctx.lineTo(X(0),h-pad); ctx.moveTo(pad,Y(0)); ctx.lineTo(w-pad,Y(0)); ctx.stroke();
  ctx.fillStyle="#727d99"; ctx.font="10px ui-monospace,monospace";
  ctx.fillText("penetration d →",w-pad-92,h-pad+18); ctx.fillText("contact force λ",X(0)+6,pad+2);
  ctx.fillText("separated",X(-0.14),h-pad+18); ctx.fillText("touching",X(0.16),h-pad+18);
  // rigid reference k·max(0,d)
  ctx.strokeStyle="rgba(174,182,204,.5)"; ctx.setLineDash([5,4]); ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(X(d0),Y(0)); ctx.lineTo(X(0),Y(0)); ctx.lineTo(X(d1),Y(lab.rigid_force(d1))); ctx.stroke();
  ctx.setLineDash([]);
  // smoothed force
  ctx.strokeStyle="#d9b45e"; ctx.lineWidth=2.5; ctx.beginPath();
  for(let i=0;i<=180;i++){ const d=d0+(d1-d0)*i/180; const p=[X(d),Y(lab.contact_force(d))]; i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]); } ctx.stroke();
  ctx.fillStyle="#f0cf82"; ctx.fillText("smoothed (κ="+fmt(lab.kappa(),0)+")",X(0.02),Y(lab.contact_force(0.32))-6);
  ctx.fillStyle="rgba(174,182,204,.7)"; ctx.fillText("rigid k·max(0,d)",X(0.14),Y(lab.rigid_force(0.30)));
}

// ---- figure B: the contact trust region ----
const cv=document.getElementById("stage");
const DUMAX=3.5;
function drawCtr(){
  const r=cv.getBoundingClientRect(); const dpr=Math.min(devicePixelRatio||1,2);
  cv.width=r.width*dpr; cv.height=r.height*dpr; const ctx=cv.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
  const w=r.width,h=r.height,pad=40; ctx.clearRect(0,0,w,h);
  const rho=lab.trust_radius();
  // y range from the actual curve
  let smax=0; for(let i=0;i<=60;i++){ smax=Math.max(smax,lab.actual_s(DUMAX*i/60)); }
  smax=Math.max(smax,lab.predict_s(DUMAX))*1.1;
  const X=u=>pad+(u/DUMAX)*(w-2*pad); const Y=s=>h-pad-(s/smax)*(h-2*pad);
  // trust-region band [0, rho]
  ctx.fillStyle="rgba(125,211,160,.10)"; ctx.fillRect(X(0),pad-6,X(rho)-X(0),h-pad-(pad-6));
  ctx.strokeStyle="rgba(125,211,160,.5)"; ctx.setLineDash([4,4]); ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(X(rho),pad-6); ctx.lineTo(X(rho),h-pad); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle="#7dd3a0"; ctx.font="10px ui-monospace,monospace"; ctx.fillText("contact trust region  Δu ≤ 1/(κh)",X(0)+8,pad+8);
  // axes
  ctx.strokeStyle="rgba(120,140,180,.3)"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,pad-6); ctx.lineTo(pad,h-pad); ctx.lineTo(w-pad,h-pad); ctx.stroke();
  ctx.fillStyle="#727d99"; ctx.fillText("control step Δu →",w-pad-108,h-pad+18); ctx.fillText("next slider position s⁺",pad+6,pad-10);
  // linear model (tangent at Δu=0)
  ctx.strokeStyle="rgba(120,150,220,.9)"; ctx.setLineDash([6,4]); ctx.lineWidth=1.8;
  ctx.beginPath(); ctx.moveTo(X(0),Y(lab.predict_s(0))); ctx.lineTo(X(DUMAX),Y(lab.predict_s(DUMAX))); ctx.stroke(); ctx.setLineDash([]);
  // true nonlinear response
  ctx.strokeStyle="#d9b45e"; ctx.lineWidth=2.6; ctx.beginPath();
  for(let i=0;i<=200;i++){ const u=DUMAX*i/200; const p=[X(u),Y(lab.actual_s(u))]; i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]); } ctx.stroke();
  ctx.fillStyle="#f0cf82"; ctx.fillText("true dynamics",X(2.4),Y(lab.actual_s(2.4))-8);
  ctx.fillStyle="rgba(120,150,220,.95)"; ctx.fillText("linear model",X(2.4),Y(lab.predict_s(2.4))+16);
  // current Δu marker + error bar
  const pa=lab.actual_s(du), pp=lab.predict_s(du);
  ctx.strokeStyle="rgba(232,131,111,.9)"; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(X(du),Y(pa)); ctx.lineTo(X(du),Y(pp)); ctx.stroke();
  ctx.beginPath(); ctx.arc(X(du),Y(pa),5,0,7); ctx.fillStyle="#d9b45e"; ctx.fill();
  ctx.beginPath(); ctx.arc(X(du),Y(pp),4,0,7); ctx.fillStyle="#7c96dc"; ctx.fill();
  const outside=du>rho;
  ctx.fillStyle=outside?"#e8836f":"#7dd3a0"; ctx.font="600 11px ui-sans-serif,system-ui";
  ctx.fillText(outside?"outside the trust region — the model is wrong here":"inside the trust region — the model is trustworthy", X(du)+8>w-230?w-232:X(du)+8, Y(Math.max(pa,pp))-10);
  // readouts
  document.getElementById("duv").textContent=fmt(du,2);
  document.getElementById("rhov").textContent=fmt(rho,2);
  document.getElementById("errv").textContent=fmt(lab.lin_error(du),4);
  document.getElementById("errv").style.color=outside?"#e8836f":"#7dd3a0";
}

function drawAll(){ drawForce(); drawCtr(); }

// dragging Δu on the trust-region canvas
function pick(e){ const r=cv.getBoundingClientRect(); const w=r.width,pad=40; let u=(e.clientX-r.left-pad)/(w-2*pad)*DUMAX; du=Math.max(0,Math.min(DUMAX,u)); document.getElementById("du").value=du; drawCtr(); }
cv.addEventListener("pointerdown",e=>{ cv.setPointerCapture(e.pointerId); pick(e); });
cv.addEventListener("pointermove",e=>{ if(e.buttons) pick(e); });
document.getElementById("du").oninput=e=>{ du=+e.target.value; drawCtr(); };
document.getElementById("kap").oninput=e=>{ lab.set_kappa(+e.target.value); drawAll(); refreshRatio(); };
document.getElementById("reset").onclick=()=>{ lab.set_kappa(40); du=0.5; document.getElementById("kap").value=40; document.getElementById("du").value=0.5; drawAll(); refreshRatio(); };

function refreshRatio(){
  const rho=lab.trust_radius(); const ec=lab.lin_error(rho), eb=lab.lin_error(6*rho);
  document.getElementById("scRatio").textContent = "×"+fmt(eb/Math.max(ec,1e-12),0);
  document.getElementById("scVerdict").textContent = ec < 0.2*eb ? "the trust-region step is far more valid" : "unexpected";
}

function selfCheck(){
  // headline: at default κ the CTR-sized step's linearization error is a small fraction of a 6× step's.
  const rho=lab.trust_radius(); const ec=lab.lin_error(rho), eb=lab.lin_error(6*rho);
  document.getElementById("scEc").textContent=fmt(ec,4);
  document.getElementById("scEb").textContent=fmt(eb,4);
  refreshRatio();
  // planner reaches a contact-only target
  document.getElementById("scPlan").textContent=fmt(lab.plan_final(0.3),3)+" (target 0.300)";
}

async function main(){
  await __wbg_init(b64ToBytes(WASM_B64));
  lab=new CtrLab(); selfCheck();
  window.__ctr={lab:()=>lab, rho:()=>lab.trust_radius(), err:(u)=>lab.lin_error(u), plan:(t)=>lab.plan_final(t)};
  window.__textbook_ready=true;
  addEventListener("resize",drawAll); drawAll();
}
main();
`;

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Planning through contact — ferromotion textbook</title>
<meta name="description" content="An interactive chapter on the contact trust region: contact makes a robot's dynamics nondifferentiable, so we smooth it and then bound the plan's step to where the smooth model is still valid — a trust region shaped by the contact, not a ball. Runs the real Rust pusher-slider on-device."/>
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
#stage{width:100%;height:400px;cursor:ew-resize;border-radius:10px;background:radial-gradient(700px 400px at 50% 50%,#0e1730,#0b1122)}
#forcefig{width:100%;height:250px;border-radius:10px;background:radial-gradient(700px 250px at 50% 50%,#0e1730,#0b1122)}
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
  <p class="kicker"><span style="color:var(--goldb);font-size:1.05rem">&#934;</span><span>ferromotion · textbook · chapter 14</span>
    <span class="badge">rust → wasm · on-device</span></p>
  <h1>Planning through contact</h1>
  <p class="lede">The moment a robot touches something, its equations of motion develop a kink: nothing happens as the gap closes, then a force switches on. Gradient-based planners need a slope to follow, and a kink has none. This page shows the fix — smooth the contact, then step only as far as the smooth model stays honest — running the real Rust pusher-slider on your device.</p>

  <h2><span class="n">01 — the kink</span>Contact is nondifferentiable</h2>
  <p>Rigid contact is a switch: while a gap remains the contact force is exactly zero; the instant the gap closes the force turns on. That corner — <span class="mono">λ = k·max(0, d)</span> — is where planning through contact breaks. An optimizer linearizes the dynamics to decide its next move, but a linearization needs a derivative, and at the corner there isn't one. Worse, on the flat "no contact yet" side the gradient is <i>zero</i>: the plan has no signal that a push is even available.</p>

  <h2><span class="n">02 — smooth it</span>A force with a slope everywhere</h2>
  <p>The remedy is to replace the hard corner with a smooth surrogate — a softplus of penetration, <span class="mono">λ = (k/κ)·log(1 + e^{κd})</span>. It is positive and differentiable <i>everywhere</i>, so a planner always has a gradient to follow; and as the sharpness <span class="mono">κ</span> grows it converges back to true rigid contact. Drag <span class="mono">κ</span> and watch the smooth curve sharpen toward the corner.</p>
  <div class="fig">
    <canvas id="forcefig"></canvas>
    <div class="ctl">
      <label>sharpness κ <input type="range" id="kap" min="10" max="200" step="2" value="40"/></label>
    </div>
    <p class="read dim" style="margin-top:8px">low κ: soft and easy to optimize · high κ: nearly rigid but the gradient collapses back into a corner</p>
  </div>

  <h2><span class="n">03 — the trap</span>A step that jumps the boundary</h2>
  <p>Smoothing alone is not enough. The smooth model is only accurate <i>near</i> where it was linearized — and a plain trust region (a ball, <span class="mono">‖Δu‖ ≤ ρ</span>) knows nothing about where the contact turns on. A large ball-shaped step can leap clean across the contact boundary, into a region where the linearization it was based on is meaningless. The plan then trusts a prediction that is simply wrong.</p>
  <p>Below, at a contact transition, the gold curve is the true next slider position as a function of the control step <span class="mono">Δu</span>; the dashed line is the linear model the planner uses. The green band is the <b>contact trust region</b>. <b>Drag left–right</b> to move the step.</p>
  <div class="fig">
    <canvas id="stage"></canvas>
    <div class="ctl">
      <button id="reset" class="ghost">Reset</button>
      <label>step Δu <input type="range" id="du" min="0" max="3.5" step="0.02" value="0.5"/></label>
    </div>
    <div class="stats">
      <div class="stat"><div class="v" id="duv">—</div><div class="k">step Δu</div></div>
      <div class="stat"><div class="v" id="rhov">—</div><div class="k">trust radius 1/(κh)</div></div>
      <div class="stat"><div class="v" id="errv">—</div><div class="k">linear-model error</div></div>
    </div>
    <p class="read dim" style="margin-top:8px">inside the band the red error bar is tiny; drag past it and the linear model peels away from the truth</p>
  </div>

  <h2><span class="n">04 — the trust region, shaped by contact</span></h2>
  <p>The contact trust region sizes the step so the <i>contact configuration</i> changes by no more than the model's smoothing bandwidth — <span class="mono">Δd ≤ 1/κ</span>, i.e. <span class="mono">Δu ≤ 1/(κh)</span>. Inside it the penetration never moves far enough to leave the region the linearization describes, so the model's prediction stays trustworthy; outside it, all bets are off. It is a trust region whose shape comes from the physics of the contact rather than from a generic ball.</p>
  <div class="callout">This is the whole idea. Smoothing gives you a gradient; the contact trust region tells you <b>how far you are allowed to believe it</b>. Together they turn a nondifferentiable, gradient-free contact problem into one an ordinary optimizer can march through — which is what lets a plan discover a push, a grasp, or a foothold that only exists <i>through</i> contact.</p>

  <h2><span class="n">05 — the check</span>The step you can trust vs the one you can't</h2>
  <p>On load, this page linearized the smoothed pusher–slider right at contact onset and measured the model's error at a trust-region-sized step against a step six times larger:</p>
  <table>
    <tr><td>linear-model error at the trust-region step</td><td id="scEc">…</td></tr>
    <tr><td>linear-model error at a 6× step</td><td id="scEb">…</td></tr>
    <tr><td>the big step is this much worse</td><td id="scRatio">…</td></tr>
    <tr><td>trust-region planner reaches a contact-only target</td><td id="scPlan">…</td></tr>
    <tr><td>verdict</td><td id="scVerdict">…</td></tr>
  </table>
  <p>The trust-region step's prediction is off by a small fraction of the oversized step's — and a planner that caps every move to that region discovers, from a standoff, exactly the push that drives the unactuated slider to its target. The slider has no motor of its own; it moves only through contact, and the plan found the contact.</p>

  <h2><span class="n">06 — the point</span>Believe the model only where it holds</h2>
  <div class="verdict">
    <div class="big">Smooth the contact; step only as far as the smoothing lets you.</div>
    <p>Contact-rich planning is not defeated by the kink but by trusting a linearization past where it is valid. Smooth the force so a gradient exists, then bound the step to the contact's own bandwidth — a trust region shaped by the physics — and the optimizer can plan straight through the touch.</p>
  </div>
  <p>This chapter is the companion to the consensus-complementarity method a chapter earlier: both make contact plannable, one by consensus over the complementarity constraints, this one by smoothing plus a physically-shaped trust region. The pattern is the same the book keeps returning to — reshape the problem until the answer is one an ordinary solver can reach.</p>

  <p class="note"><b>What you just drove:</b> the <span style="color:var(--soft)">PusherSlider</span> and <span style="color:var(--soft)">SmoothedContact</span> from <span style="color:var(--soft)">ferromotion-control</span>, compiled to WebAssembly — the same code the native tools link against. The force curve, the linearization, the trust radius, and the planner are all evaluated live; nothing precomputed.<br/><br/>
  <b>Verified in the library:</b> the smoothed force converges to rigid <span style="color:var(--soft)">k·max(0,d)</span> as κ→∞; the analytic contact-force gradient matches finite differences; a trust-region-sized step keeps the linearization more than 5× more valid than an oversized step; and the trust-region planner drives the unactuated slider to a contact-only target. Each is a test in <span style="color:var(--soft)">cargo test</span>, not a claim in prose. See also <a href="/assets/sims/loop-closure">ch.12 — closing the loop</a> · <a href="/assets/sims/textbook">the full textbook</a>.<br/><br/>
  <b>Institute for Physical AI</b> · <a href="https://github.com/dcharlot-physicalai-bmi/ferromotion">the Rust library</a> · <a href="https://crates.io/crates/ferromotion">crates.io</a></p>
</div>
<script type="module">
${glue}
const WASM_B64="${wasmB64}";
${APP}
</script></body></html>`;

const outFile = path.join(__dirname, "..", "..", "v2", "public", "assets", "sims", "contact-planning.html");
fs.writeFileSync(outFile, html);
console.log(`wrote ${outFile} (${(html.length / 1024).toFixed(0)} KB, wasm ${(wasmB64.length / 1024).toFixed(0)} KB b64)`);

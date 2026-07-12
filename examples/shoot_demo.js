// Serve the self-contained demo and drive it in real Chrome to prove it renders + runs the WASM.
const http = require("http");
const fs = require("fs");
const path = require("path");
const PW = require("/Users/dcharlot/vibe-coding/bmi-concept/ipai-tv/node_modules/playwright-core");

const file = path.join(__dirname, "..", "demo", "index.html");
const html = fs.readFileSync(file);
const srv = http.createServer((_req, res) => {
  res.setHeader("Content-Type", "text/html");
  res.end(html);
});

(async () => {
  await new Promise((r) => srv.listen(8099, r));
  const browser = await PW.chromium.launch({
    executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: true,
  });
  const page = await browser.newPage({ viewport: { width: 1000, height: 820 }, deviceScaleFactor: 2 });
  const errs = [];
  page.on("pageerror", (e) => errs.push(String(e)));
  await page.goto("http://localhost:8099/");
  await page.waitForFunction(() => window.__ferromotion_ready === true, { timeout: 20000 });
  await page.waitForTimeout(400);

  const box = await page.$eval("#c", (el) => {
    const r = el.getBoundingClientRect();
    return { x: r.x, y: r.y, w: r.width, h: r.height };
  });
  // Drive IK: move the cursor over the canvas so the arm solves to follow it.
  await page.mouse.move(box.x + box.w * 0.74, box.y + box.h * 0.32, { steps: 12 });
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(__dirname, "..", "demo", "shot-follow.png") });

  // Reach mode: plan and play a trajectory around the obstacle.
  await page.click("#mReach");
  await page.waitForTimeout(200);
  await page.click("#play");
  await page.waitForTimeout(1700);
  await page.screenshot({ path: path.join(__dirname, "..", "demo", "shot-reach.png") });

  await browser.close();
  srv.close();
  console.log(errs.length ? "PAGE ERRORS:\n" + errs.join("\n") : "OK — no page errors; shots saved");
})().catch((e) => {
  console.error(e);
  process.exit(1);
});

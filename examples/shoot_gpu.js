// Run the batched-FK WebGPU compute in real Chrome (Metal on this Mac) and report the result.
const http = require("http");
const fs = require("fs");
const path = require("path");
const PW = require("/Users/dcharlot/vibe-coding/bmi-concept/ipai-tv/node_modules/playwright-core");

const html = fs.readFileSync(path.join(__dirname, "..", "gpu", "fk.html"));
const srv = http.createServer((_q, res) => { res.setHeader("Content-Type", "text/html"); res.end(html); });

(async () => {
  await new Promise((r) => srv.listen(8102, r));
  const browser = await PW.chromium.launch({
    executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: false, // WebGPU needs a real GPU context
    args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan,Metal", "--use-angle=metal"],
  });
  const page = await browser.newPage();
  await page.goto("http://localhost:8102/");
  await page.waitForFunction(() => window.__result !== undefined, { timeout: 30000 });
  const result = await page.evaluate(() => window.__result);
  console.log(JSON.stringify(result, null, 2));
  await browser.close();
  srv.close();
  process.exit(result && !result.error && result.maxErr < 1e-3 ? 0 : 1);
})().catch((e) => { console.error(e); process.exit(1); });

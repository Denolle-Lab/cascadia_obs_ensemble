// atlas_check.js: headless check of a built atlas page (build_v4_atlas.py output).
//
// Loads the page in headless Chromium (Playwright), then for every catalog view and Compare reports:
//   - the number of events shown at Min M 2.0 (each view uses its own catalog's magnitudes, D-B102);
//   - the Mc layer's set, cell count and legend source (D-B104, D-B105);
//   - the velocity-model legend of every model variable: range, end labels, colour map (D-B104).
// Exits 1 on any page error, console error, or count that differs from EXPECT.
//
// Usage (the Playwright env of this machine; any env with playwright + Chromium works):
//   NE=~/.cache/rattler/cache/cached-envs-v0/55304966f11fb1c1
//   LD_LIBRARY_PATH=$NE/lib PW=$NE/lib/node_modules/playwright PAGE=/path/atlas.html \
//     [EXPECT=expected.json] $NE/bin/node workflow/06_analysis/atlas_check.js
// EXPECT: optional JSON {"<catalog id>": <events with M >= 2.0 drawn in its view>, ...}. Count from the
// registry catalog files: round(M * 100) >= 200 and loc_source != "none" (the reference catalog: every event
// with a magnitude).
const { chromium } = require(process.env.PW || "playwright");
const fs = require("fs");

(async () => {
  const expect = process.env.EXPECT ? JSON.parse(fs.readFileSync(process.env.EXPECT, "utf8")) : {};
  const b = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
  const p = await b.newPage({ viewport: { width: 1600, height: 1000 } });
  const errs = [];
  p.on("pageerror", e => errs.push(String(e)));
  p.on("console", m => { if (m.type() === "error") errs.push(m.text()); });
  await p.goto("file://" + require("path").resolve(process.env.PAGE));
  await p.waitForTimeout(8000);
  let bad = 0;

  // magnitudes and Mc per view
  await p.$eval("#mmin", el => { el.value = "2"; el.dispatchEvent(new Event("input")); });
  const hasMc = await p.$eval("#mcSel", el => el.options.length > 0).catch(() => false);
  if (hasMc) await p.$eval("#mcSel", el => { el.value = "best"; el.dispatchEvent(new Event("change")); });
  const views = await p.$$eval("#segCat button", bs => bs.map(x => [x.dataset.v, x.textContent]));
  for (const [v, label] of views) {
    await p.click(`#segCat button[data-v="${v}"]`);
    await p.waitForTimeout(1500);
    const n = await p.$eval("#nShown", e => e.textContent);
    const id = label.replace(/prelim\.$/, "");
    let line = `${label} | shown M>=2: ${n}`;
    if (id in expect) {
      const ok = Number(n.replace(/,/g, "")) === expect[id];
      line += ok ? " (= expected)" : ` (EXPECTED ${expect[id]})`;
      if (!ok) bad++;
    }
    if (hasMc) {
      const opt = await p.$eval("#mcSel", e => e.options[e.selectedIndex].textContent);
      const src = await p.$$eval("#legend div", ds => (ds.map(d => d.textContent).find(x => x.startsWith("Nearest 250")) || "")
        .replace(/.*deployed\./, "").trim());
      line += ` | Mc: ${opt} | ${src}`;
    }
    const mag = await p.$$eval(".attribution", es => (es.map(e => e.textContent).find(x => x.includes("Magnitudes (")) || "")
      .split("Bathymetry")[0].trim());
    console.log(line + (mag ? ` | ${mag}` : ""));
  }

  // velocity-model legend per variable
  await p.click(`#segCat button[data-v="0"]`);
  const lb = await p.$$("#layers button");
  for (const x of lb) if ((await x.textContent()).includes("Velocity model")) { await x.click(); break; }
  await p.waitForTimeout(1000);
  const vars = await p.$$eval("#tomoVar option", os => os.map(o => o.value)).catch(() => []);
  for (const v of vars) {
    await p.$eval("#tomoVar", (el, v) => { el.value = v; el.dispatchEvent(new Event("change")); }, v);
    await p.waitForTimeout(1200);
    const leg = await p.$$eval("#legend div", ds => {
      const d = ds.find(d => { const h = d.querySelector("h4"); return h && h.textContent.startsWith("Velocity model"); });
      if (!d) return "no model legend";
      const m = d.textContent.match(/Crameri[^)]*\)/);
      return [d.querySelector("h4").textContent, d.querySelector(".ramp-labels").textContent, m ? m[0] : "no colour-map line"].join(" | ");
    });
    console.log(`model ${v} | ${leg}`);
  }

  console.log(`page errors: ${errs.length}` + (errs.length ? ` ${JSON.stringify(errs.slice(0, 5))}` : ""));
  await b.close();
  process.exit(errs.length || bad ? 1 : 0);
})();

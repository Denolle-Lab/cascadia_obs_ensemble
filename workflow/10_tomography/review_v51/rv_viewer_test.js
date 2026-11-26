// Tests of review_v51_viewer.html without a browser (node >= 10):
//  1. the three inline scripts parse; 2. CSV round trip of review_v51_events.csv through the viewer's parser and
//  serializer (byte-equal); 3. edit -> export -> re-import -> merge gives the same CSV; 4. merge of two reviewers lists
//  conflicts, report-page CSVs are mapped; 5. the page script runs against a stub DOM: initial render, an edit through
//  the table's change handler, a download through the button (the CSV written equals the edited records).
// Usage: node rv_viewer_test.js /wd1/mdenolle_data/catalog_v5/v5.1_review
"use strict";
var fs = require("fs"), vm = require("vm"), path = require("path");
var OUT = process.argv[2] || "/wd1/mdenolle_data/catalog_v5/v5.1_review";
var html = fs.readFileSync(path.join(OUT, "review_v51_viewer.html"), "utf8");
var csv = fs.readFileSync(path.join(OUT, "review_v51_events.csv"), "utf8");
var fails = 0;
function ok(cond, what) { console.log((cond ? "PASS " : "FAIL ") + what); if (!cond) { fails++; } }

// 1. scripts
var scripts = [], re = /<script>([\s\S]*?)<\/script>/g, m;
while ((m = re.exec(html)) !== null) { scripts.push(m[1]); }
ok(scripts.length === 3, "three inline scripts (" + scripts.length + ")");
scripts.forEach(function (s, i) { try { new vm.Script(s); ok(true, "script " + i + " parses (" + s.length + " chars)"); }
  catch (e) { ok(false, "script " + i + " parses: " + e.message); } });

// core from the page itself
var ctx0 = {module: undefined};
vm.createContext(ctx0);
vm.runInContext(scripts[0] + "\n" + scripts[1] + "\nthis.VCORE = VCORE; this.DATA_CSV = DATA_CSV;", ctx0);
var V = ctx0.VCORE;
ok(ctx0.DATA_CSV === csv, "embedded CSV equals review_v51_events.csv");

// 2. round trip
var d = V.toObjects(V.parseCSV(csv));
ok(d.records.length === 1512, "1512 records parsed (" + d.records.length + ")");
ok(V.serialize(d.header, d.records) === csv, "parse -> serialize is byte-equal to the file");

// 3. edit, export, re-import
var e = V.toObjects(V.parseCSV(csv)), n = 0;
e.records.forEach(function (r, i) {
  if (i % 50 === 0) {
    r.reviewer = "tester A"; r.is_earthquake = ["yes", "no", "unsure"][n % 3]; r.depth_verdict = ["plausible", "implausible", "cannot_tell"][n % 3];
    r.action = ["keep", "flag_not_earthquake", "flag_depth_unreliable", "relocate", "other"][n % 5]; r.corrected_depth_km = n % 4 === 0 ? "12.5" : "";
    r.comment = n % 2 ? 'quote " and, comma\nnew line' : "plain"; r.reviewed_utc = "2026-11-25T16:00:0" + (n % 10) + "Z"; n++;
  }
});
var exported = V.serialize(e.header, e.records);
var back = V.toObjects(V.parseCSV(exported));
var fresh = V.toObjects(V.parseCSV(csv));
var st = V.merge(fresh.records, V.normalize(back.header, back.records));
ok(st.applied === n && st.conflicts.length === 0, "re-import applied " + st.applied + " of " + n + " edited rows, 0 conflicts");
ok(V.serialize(fresh.header, fresh.records) === exported, "export -> re-import -> merge -> export is byte-equal");

// 4. second reviewer: half identical, half different -> conflicts; invalid value rejected; report-page format mapped
var b2 = V.toObjects(V.parseCSV(exported)), k = 0, nconf = 0;
b2.records.forEach(function (r) { if (V.hasReview(r)) { r.reviewer = "tester B"; if (k % 2) { r.is_earthquake = r.is_earthquake === "yes" ? "no" : "yes"; nconf++; } k++; } });
b2.records[1].is_earthquake = "maybe"; b2.records[1].reviewer = "tester B";
var st2 = V.merge(fresh.records, b2.records);
ok(st2.conflicts.length === nconf && st2.identical === k - nconf, "second reviewer: " + st2.conflicts.length + " conflicts (expected " + nconf + "), " + st2.identical + " identical");
ok(st2.invalid.length === 1, "invalid value rejected (" + st2.invalid.join(";") + ")");
var page = "report,problem,event_id,view_group,origin_time_utc,v51_depth_km,reviewer,earthquake,v51_depth,comment,answered_at_utc,exported_at_utc\n" +
  "P1_msh_deep,P1,29839,Wednesday,2012-06-13T19:17:19.412732,57.60,tester C,not_earthquake,implausible,step on analog,2026-11-25T16:10:00Z,2026-11-25T16:11:00Z\n";
var pg = V.toObjects(V.parseCSV(page)), f3 = V.toObjects(V.parseCSV(csv)), st3 = V.merge(f3.records, V.normalize(pg.header, pg.records));
var r3 = f3.records.filter(function (r) { return r.ID === "29839" && r.problem === "P1"; })[0];
ok(st3.applied === 1 && r3.is_earthquake === "no" && r3.depth_verdict === "implausible" && r3.reviewer === "tester C", "report-page CSV mapped and merged");

// 5. page script against a stub DOM
function el(id) {
  var o = {id: id, value: "", checked: false, innerHTML: "", textContent: "", files: [], style: {}, _h: {},
           classList: {toggle: function () {}}, addEventListener: function (t, f) { (o._h[t] = o._h[t] || []).push(f); },
           getAttribute: function () { return null; }, appendChild: function () {}, click: function () { (o._h.click || []).forEach(function (f) { f({target: o}); }); },
           remove: function () {}};
  return o;
}
var els = {}, store = {}, downloaded = null;
var doc = {getElementById: function (id) { return els[id] || (els[id] = el(id)); }, querySelectorAll: function () { return []; },
           createElement: function () { var a = el("a"); a.click = function () {}; return a; }, body: el("body")};
var sandbox = {document: doc, console: console, setTimeout: function () {}, confirm: function () { return true; },
  localStorage: {getItem: function (k) { return store.hasOwnProperty(k) ? store[k] : null; }, setItem: function (k, v) { store[k] = String(v); }, removeItem: function (k) { delete store[k]; }},
  Blob: function (parts) { downloaded = parts.join(""); }, URL: {createObjectURL: function () { return "blob:x"; }, revokeObjectURL: function () {}},
  Date: Date, JSON: JSON, Object: Object, Array: Array, String: String, parseFloat: parseFloat, isNaN: isNaN};
vm.createContext(sandbox);
els.fp = el("fp"); els.fi = el("fi"); els.ft = el("ft"); els.fs = el("fs"); els.fv = el("fv"); els.fx = el("fx"); els.who = el("who");
els.who.value = "stub reviewer";
try { vm.runInContext(scripts.join("\n;\n"), sandbox); ok(true, "page script runs against the stub DOM"); }
catch (err) { ok(false, "page script runs: " + err.stack); }
els.who.value = "stub reviewer";      // typed after the page loaded (the page restores the stored name at start)
var rows = (els.tb.innerHTML.match(/<tr /g) || []).length;
ok(rows === 1512, "initial render: " + rows + " table rows");
ok(/1512 of 1512 rows shown/.test(els.count.textContent), "count line: " + els.count.textContent);
// edit through the delegated change handler
var key = "31475|P2", cells = [];
for (var i = 0; i < 30; i++) { cells.push({textContent: ""}); }
var tr = {getAttribute: function () { return key; }, className: "", cells: cells, querySelector: function () { return {textContent: ""}; }};
var target = {getAttribute: function (a) { return a === "data-k" ? "action" : null; }, value: "flag_depth_unreliable", closest: function () { return tr; }};
els.tb._h.change[0]({target: target});
target.getAttribute = function (a) { return a === "data-k" ? "is_earthquake" : null; }; target.value = "yes";
els.tb._h.change[0]({target: target});
ok(store["v51review:viewer:v1"] && JSON.parse(store["v51review:viewer:v1"])[key].action === "flag_depth_unreliable", "edit autosaved to localStorage");
els.dl._h.click[0]({});
var dd = V.toObjects(V.parseCSV(downloaded || ""));
var r5 = dd.records.filter(function (r) { return r.ID === "31475" && r.problem === "P2"; })[0];
ok(dd.records.length === 1512 && dd.header.join() === d.header.join(), "download: 1512 rows, same columns as review_v51_events.csv");
ok(r5 && r5.action === "flag_depth_unreliable" && r5.is_earthquake === "yes" && r5.reviewer === "stub reviewer" && r5.reviewed_utc !== "",
   "download carries the edit (" + (r5 ? [r5.action, r5.is_earthquake, r5.reviewer].join("/") : "missing") + ")");
var others = dd.records.filter(function (r) { return !(r.ID === "31475" && r.problem === "P2"); });
ok(others.every(function (r) { return !V.hasReview(r); }), "no other row changed");
// filter: problem P2 only
els.fp.value = "P2"; (els.fp._h.change || [])[0]();
ok((els.tb.innerHTML.match(/<tr /g) || []).length === 39, "filter problem P2: 39 rows");
console.log(fails ? fails + " FAILED" : "ALL PASS");
process.exit(fails ? 1 : 0);

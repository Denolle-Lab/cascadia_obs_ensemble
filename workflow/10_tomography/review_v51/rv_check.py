"""Checks of the built review pages: HTML nesting, unique ids, local links and images exist, script syntax (node --check),
folder sizes. Usage: python rv_check.py [page.html ...] (default: the four index pages and the top index)."""
import re
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path

import rv_common as C

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


class P(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.errors, self.ids, self.refs, self.scripts, self._in_script = [], [], [], [], [], False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.append(a["id"])
        for k in ("src", "href"):
            if k in a and a[k] and not a[k].startswith(("#", "http", "mailto", "data:")):
                self.refs.append(a[k])
        if tag == "script":
            self._in_script = True
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> (open: {self.stack[-3:]})")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
            return
        self.stack.pop()

    def handle_data(self, data):
        if self._in_script:
            self.scripts.append(data)


def check(page):
    page = Path(page)
    p = P()
    p.feed(page.read_text())
    out = []
    if p.errors:
        out.append(f"nesting: {len(p.errors)} errors, e.g. {p.errors[:3]}")
    if p.stack:
        out.append(f"unclosed: {p.stack[:5]}")
    dup = {i for i in p.ids if p.ids.count(i) > 1}
    if dup:
        out.append(f"duplicate ids: {sorted(dup)[:5]}")
    miss = [r for r in set(p.refs) if not (page.parent / r).exists()]
    if miss:
        out.append(f"{len(miss)} missing local files, e.g. {sorted(miss)[:5]}")
    for s in p.scripts:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, dir=C.WORK) as f:
            f.write(s)
        r = subprocess.run(["node", "--check", f.name], capture_output=True, text=True)
        Path(f.name).unlink()
        if r.returncode:
            out.append("script: " + r.stderr.strip()[:300])
    n_img = sum(1 for r in p.refs if r.endswith(".webp"))
    n_ev = len([i for i in p.ids if i.startswith("ev-")])
    return out, dict(images=n_img, event_views=n_ev, refs=len(set(p.refs)), kB=page.stat().st_size // 1024)


if __name__ == "__main__":
    pages = sys.argv[1:] or [str(C.OUT / "index.html")] + [str(C.OUT / d / "index.html") for d in
                                                             ("P1_msh_deep", "P2_brookings", "P3_tensor", "P4_deep_far")]
    for pg in pages:
        errs, info = check(pg)
        size = sum(f.stat().st_size for f in Path(pg).parent.rglob("*") if f.is_file()) // (1 << 20)
        print(pg, info, f"folder {size} MB", "OK" if not errs else "PROBLEMS: " + " | ".join(errs))

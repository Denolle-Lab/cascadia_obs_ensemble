"""Track B decision map: inject data/tomo_v1/decision_map/decisions_graph.json into decision_map_template.html and write a
standalone HTML page (data/tomo_v1/decision_map/decision_map.html). The graph (records, relations, alternatives, threads) was
extracted from DECISIONS.md on 2026-11-25; edit the JSON when new decisions are added, then rerun.

    python workflow/10_tomography/decision_map_build.py
"""
import json
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "data" / "tomo_v1" / "decision_map"


def main():
    data = json.loads((D / "decisions_graph.json").read_text())
    ids = {r["id"] for r in data["decisions"]}
    bad = [(r["id"], x["target"]) for r in data["decisions"] for x in r.get("relations", []) if x["target"] not in ids]
    bad += [(f["name"], i) for f in data["forks"] for i in f["chain"] if i not in ids]
    if bad:
        raise SystemExit(f"unknown ids: {bad}")
    t = (D / "decision_map_template.html").read_text()
    assert "/*DATA*/null" in t
    page = t.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    head, body = page.split('<div class="wrap">', 1)
    html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + head + '<style>html,body{margin:0}</style>\n</head>\n<body>\n<div class="wrap">' + body + '\n</body>\n</html>\n')
    (D / "decision_map.html").write_text(html)
    print(D / "decision_map.html", len(html), "bytes,", len(data["decisions"]), "decisions,", len(data["forks"]), "threads")


if __name__ == "__main__":
    main()

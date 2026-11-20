"""Adapt CHM13 complex144 to the Plan B metrics package input format.

- graph: graph_with_rc.graphml (both read orientations, needed by build_mate_map),
  each edge enriched with the PAF overlap coordinates it was built from
  (paf_line is the 1-based line number in overlaps.exact.paf.gz).
- candidates: certified top paths (weighted_certificate.json) and the distinct
  SQA best paths from Actions run 39137698403 / 39141763797 (normalized read ids,
  mapped to signed nodes via nodes.tsv).
No reference data is read here.
"""
import csv, gzip, hashlib, json, sys
from pathlib import Path
import networkx as nx

data, logs, out = map(Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=False)

g = nx.read_graphml(data / "graph_with_rc.graphml")
need = {int(d["paf_line"]) for _, _, d in g.edges(data=True)}
paf = {}
with gzip.open(data / "overlaps.exact.paf.gz", "rt") as f:
    for i, line in enumerate(f, 1):
        if i in need:
            paf[i] = line.rstrip("\n").split("\t")
assert set(paf) == need
for u, v, d in g.edges(data=True):
    row = paf[d["paf_line"]]
    q, t = row[0], row[5]
    assert {q, t} == {u[:-1], v[:-1]}, (u, v, q, t)
    d.update(q=q, qlen=int(row[1]), qstart=int(row[2]), qend=int(row[3]),
             strand=row[4], t=t, tlen=int(row[6]), tstart=int(row[7]), tend=int(row[8]),
             paf_row=d["paf_line"],
             edge_id=f"paf{d['paf_line']}-" + hashlib.sha1(f"{u}>{v}".encode()).hexdigest()[:8])
for n, d in g.nodes(data=True):
    d["physical_read"] = d["read_id"]
nx.write_graphml(g, out / "graph_planb.graphml")

sign = {r["read_id"]: r["selected_sign"] for r in csv.DictReader((data / "nodes.tsv").open(), delimiter="\t")}
cert = json.loads((data / "weighted_certificate.json").read_text())
cands = {}
for rank, p in enumerate(cert["top_paths"][:3], 1):
    cands[f"cert_rank{rank}_{p['score']}"] = p["nodes"]
seen = set()
for p in json.loads((logs / "original_best_paths.json").read_text()):
    key = tuple(p["order"])
    if key in seen:
        continue
    seen.add(key)
    cands[f"sqa_{p['score']}_{len(seen)}"] = [r + sign[r] for r in p["order"]]

summary = {}
for name, nodes in cands.items():
    assert len(nodes) == 144 and all(g.has_edge(a, b) for a, b in zip(nodes, nodes[1:])), name
    score = sum(int(g[a][b]["weight"]) for a, b in zip(nodes, nodes[1:]))
    d = out / name
    d.mkdir()
    (d / "chain.txt").write_text("\n".join(nodes) + "\n")
    (d / "edges.txt").write_text("\n".join(g[a][b]["edge_id"] for a, b in zip(nodes, nodes[1:])) + "\n")
    summary[name] = score
(out / "candidates.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

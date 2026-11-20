"""CHM13 complex144: evaluate every Hamilton path at solver, order, evidence and sequence level.

Subcommands
  enumerate   exact SMT enumeration of all free-endpoint Hamilton paths (dataset verification code)
  graph       graph_with_rc.graphml + PAF coordinates per edge -> Plan B input graph
  candidates  write chain/edge-ID files for one shard of the enumerated paths
  aggregate   join Plan B / QUAST results with order- and evidence-level metrics

The reference is not read by enumerate/graph/candidates. Plan B evaluation itself
(freeze, audit, spell, QUAST) runs through external/planb_metrics_r262/run_metrics.py unmodified.
"""
import argparse, csv, gzip, hashlib, json, sys
from pathlib import Path

REF_LEN = 1647338


def cmd_enumerate(a):
    sys.path[:0] = [str(a.data / "verification/deps"), str(a.data / "verification")]
    import networkx as nx
    from phase_solver import enumerate_paths
    g = nx.read_graphml(a.data / "graph.graphml")
    e = enumerate_paths(g, 2000, 600000)
    assert e["end"] == "unsat", e["end"]
    rows = sorted(({"score": s, "nodes": p} for s, p in zip(e["scores"], e["paths"])), key=lambda r: -r["score"])
    a.out.write_text(json.dumps(rows))
    print("paths", len(rows), "best", rows[0]["score"], "worst", rows[-1]["score"])


def cmd_graph(a):
    import networkx as nx
    g = nx.read_graphml(a.data / "graph_with_rc.graphml")
    need = {int(d["paf_line"]) for _, _, d in g.edges(data=True)}
    paf = {}
    with gzip.open(a.data / "overlaps.exact.paf.gz", "rt") as f:
        for i, line in enumerate(f, 1):
            if i in need:
                paf[i] = line.rstrip("\n").split("\t")
    assert set(paf) == need
    for u, v, d in g.edges(data=True):
        row = paf[d["paf_line"]]
        assert {row[0], row[5]} == {u[:-1], v[:-1]}, (u, v)
        d.update(q=row[0], qlen=int(row[1]), qstart=int(row[2]), qend=int(row[3]), strand=row[4],
                 t=row[5], tlen=int(row[6]), tstart=int(row[7]), tend=int(row[8]), paf_row=d["paf_line"],
                 edge_id=f"paf{d['paf_line']}-" + hashlib.sha1(f"{u}>{v}".encode()).hexdigest()[:8])
    for _, d in g.nodes(data=True):
        d["physical_read"] = d["read_id"]
    nx.write_graphml(g, a.out)
    print("graph", g.number_of_nodes(), g.number_of_edges())


def cmd_candidates(a):
    import networkx as nx
    g = nx.read_graphml(a.graph)
    paths = json.loads(a.paths.read_text())
    for k, p in enumerate(paths, 1):
        if (k - 1) % a.nshards != a.shard:
            continue
        n = p["nodes"]
        assert sum(int(g[x][y]["weight"]) for x, y in zip(n, n[1:])) == p["score"]
        d = a.out / f"r{k:03d}_{p['score']}"
        d.mkdir(parents=True)
        (d / "chain.txt").write_text("\n".join(n) + "\n")
        (d / "edges.txt").write_text("\n".join(g[x][y]["edge_id"] for x, y in zip(n, n[1:])) + "\n")


def _inversions(a):
    if len(a) < 2:
        return a, 0
    m = len(a) // 2
    l, x = _inversions(a[:m]); r, y = _inversions(a[m:])
    res, i, j, c = [], 0, 0, x + y
    while i < len(l) and j < len(r):
        if l[i] <= r[j]: res.append(l[i]); i += 1
        else: res.append(r[j]); j += 1; c += len(l) - i
    return res + l[i:] + r[j:], c


def cmd_aggregate(a):
    import networkx as nx
    g = nx.read_graphml(a.data / "graph.graphml")
    nodes = {r["node"]: r for r in csv.DictReader((a.data / "nodes.tsv").open(), delimiter="\t")}
    truth = json.loads((a.data / "reference_path.json").read_text())["nodes"]
    pos = {v: i for i, v in enumerate(truth)}
    start = {v: int(nodes[v]["start"]) for v in g}
    end = {v: int(nodes[v]["end"]) for v in g}
    true_adj = set(zip(truth, truth[1:]))

    # evidence level: does the selected PAF row place the two reads at their reference offset?
    rows_by_pair = {}
    with gzip.open(a.data / "overlaps.exact.paf.gz", "rt") as f:
        for i, line in enumerate(f, 1):
            x = line.split("\t")
            if x[0] != x[5]:
                rows_by_pair.setdefault(frozenset((x[0], x[5])), []).append((i, x))
    def oriented(node, s, e, L):
        return (s, e) if node[-1] == "+" else (L - e, L - s)
    TOL = 200
    edges = []
    for u, v, d in g.edges(data=True):
        true_off = start[v] - start[u]
        cands = []
        for i, x in rows_by_pair[frozenset((u[:-1], v[:-1]))]:
            side = {x[0]: (int(x[2]), int(x[3]), int(x[1])), x[5]: (int(x[7]), int(x[8]), int(x[6]))}
            cands.append((i, oriented(u, *side[u[:-1]])[0] - oriented(v, *side[v[:-1]])[0]))
        off = dict(cands)[int(d["paf_line"])]
        best = min(cands, key=lambda c: abs(c[1] - true_off))
        edges.append(dict(source=u, target=v, weight=int(d["weight"]), paf_line=int(d["paf_line"]),
                          true_adjacent=(u, v) in true_adj, true_offset=true_off, evidence_offset=off,
                          offset_error=off - true_off, shifted=abs(off - true_off) > TOL,
                          consistent_row=best[0] if abs(best[1] - true_off) <= TOL else ""))
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "edge_evidence.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(edges[0])); w.writeheader(); w.writerows(edges)
    shifted = {(e["source"], e["target"]) for e in edges if e["shifted"]}

    paths = json.loads(a.paths.read_text())
    N = len(truth); maxinv = N * (N - 1) // 2
    keys = ["Total length", "# misassemblies", "# local misassemblies", "Duplication ratio",
            "# mismatches per 100 kbp", "# indels per 100 kbp", "NGA50"]
    out, missing = [], []
    for k, p in enumerate(paths, 1):
        cid = f"r{k:03d}_{p['score']}"
        n = p["nodes"]; pairs = list(zip(n, n[1:]))
        ok = [pr in true_adj for pr in pairs]
        blocks, cur = [], [n[0]]
        for (x, y), t in zip(pairs, ok):
            if t: cur.append(y)
            else: blocks.append(cur); cur = [y]
        blocks.append(cur)
        spans = sorted((end[b[-1]] - start[b[0]] for b in blocks), reverse=True)
        acc = ng50 = 0
        for s in spans:
            acc += s
            if acc >= REF_LEN / 2: ng50 = s; break
        idx = [pos[v] for v in n]; inv = _inversions(idx)[1]
        disp = [abs(i - j) for i, j in enumerate(idx)]
        row = dict(candidate=cid, rank=k, score=p["score"], gap=paths[0]["score"] - p["score"],
                   true_adjacencies=sum(ok), breakpoints=len(pairs) - sum(ok), correct_blocks=len(blocks),
                   longest_block_reads=max(map(len, blocks)), longest_block_bp=spans[0], layout_NG50_bp=ng50,
                   kendall_tau=round(1 - 2 * inv / maxinv, 6), reads_displaced=sum(x > 0 for x in disp),
                   max_displacement=max(disp), shifted_evidence_edges=sum(pr in shifted for pr in pairs))
        gf = a.results / cid / "GENOME_FRACTION.json"
        if gf.exists():
            r = json.loads(gf.read_text())
            res = r["results"][0]
            row.update(planb_status=r["status"], genome_fraction=res["genome_fraction_percent"],
                       **{k2: res["secondary_metrics"].get(k2) for k2 in keys})
        else:
            missing.append(cid); row.update(planb_status="MISSING")
        out.append(row)
    fields = list(dict.fromkeys(k2 for r in out for k2 in r))
    with (a.out / "all_paths_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)
    print(json.dumps(dict(paths=len(out), evaluated=len(out) - len(missing), missing=missing[:20],
                          edges=len(edges), shifted_edges=len(shifted),
                          true_adjacent_shifted=sum(e["shifted"] and e["true_adjacent"] for e in edges))))


p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
s = p.add_subparsers(dest="cmd", required=True)
x = s.add_parser("enumerate"); x.add_argument("--data", type=Path, required=True); x.add_argument("--out", type=Path, required=True)
x = s.add_parser("graph"); x.add_argument("--data", type=Path, required=True); x.add_argument("--out", type=Path, required=True)
x = s.add_parser("candidates"); x.add_argument("--graph", type=Path, required=True); x.add_argument("--paths", type=Path, required=True)
x.add_argument("--out", type=Path, required=True); x.add_argument("--shard", type=int, default=0); x.add_argument("--nshards", type=int, default=1)
x = s.add_parser("aggregate"); x.add_argument("--data", type=Path, required=True); x.add_argument("--paths", type=Path, required=True)
x.add_argument("--results", type=Path, required=True); x.add_argument("--out", type=Path, required=True)
a = p.parse_args()
{"enumerate": cmd_enumerate, "graph": cmd_graph, "candidates": cmd_candidates, "aggregate": cmd_aggregate}[a.cmd](a)

"""Verify extracted CHM13_complex144 package from exact PAF."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from audit_graph import readfa,rcnode,pafedges
from phase_solver import quality_graph,better_than,enumerate_paths
import networkx as nx
TABLE=str.maketrans('ACGTNacgtn','TGCANtgcan')
p=Path(sys.argv[1] if len(sys.argv)>1 else '.')
for name,digest in json.loads((p/'sha256.json').read_text()).items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest,name
raw=readfa(p/'reads.fasta');norm=readfa(p/'reads.normalized.fasta');truth_meta=json.loads((p/'truth.json').read_text());truth=[r['id']+r['strand'] for r in truth_meta]
for r in truth_meta:
    expected=raw[r['id']] if r['strand']=='+' else raw[r['id']].translate(TABLE)[::-1]
    assert norm[r['id']]==expected,r['id']
threshold=float(json.loads((p/'selection.json').read_text())['identity_threshold'])
spec=json.loads((p/'weight_spec.json').read_text())
assert threshold==.98 and spec['primitive_formula']=='1*M - 49*d' and spec['divisor']==20
evidence,_,_=pafedges(p/'overlaps.exact.paf.gz');rebuilt=quality_graph(evidence,list(raw),threshold,300)
full=nx.read_graphml(p/'graph_full_oriented.graphml');layout=nx.read_graphml(p/'graph.graphml');mirrored=nx.read_graphml(p/'graph_with_rc.graphml');normal=nx.read_graphml(p/'graph_normalized.graphml')
assert set(full)==set(rebuilt) and set(full.edges)==set(rebuilt.edges)
assert all(int(full[u][v]['weight'])==rebuilt[u][v]['weight'] for u,v in full.edges)
for graph in [full,layout,mirrored,normal]:
    assert graph.graph['weight_scheme']=='quality_margin_gcd_normalized_v2'
    assert graph.graph['weight_formula']=='1*M - 49*d'
    assert int(graph.graph['weight_scale_divisor'])==20
for _,_,a in full.edges(data=True):
    M=int(a['matching_bases']);d=int(a['errors']);w=int(a['weight'])
    assert w==M-49*d
    assert int(a['raw_quality_margin'])==20*w
    assert int(a['weight_divisor'])==20
assert set(layout)==set(truth) and set(layout.edges)==set(rebuilt.subgraph(truth).edges)
mirror=[rcnode(v) for v in truth];assert set(mirrored)==set(truth+mirror)
assert all(mirrored.has_edge(rcnode(v),rcnode(u)) and int(mirrored[u][v]['weight'])==int(mirrored[rcnode(v)][rcnode(u)]['weight']) for u,v in layout.edges)
assert not any((u in truth)!=(v in truth) for u,v in mirrored.edges)
assert set(normal)==set(raw) and normal.number_of_edges()==layout.number_of_edges()
b=better_than(layout,truth,60000);assert b['status']=='unsat',b
e=enumerate_paths(layout,2000,60000);assert e['end']=='unsat',e
expected=json.loads((p/'weighted_certificate.json').read_text())
assert e['n_found']==expected['n_paths']==192
assert max(e['scores'])==expected['reference_score']
refscore=sum(int(layout[u][v]['weight']) for u,v in zip(truth,truth[1:]))
runner=max(score for score,path in zip(e['scores'],e['paths']) if path!=truth)
assert refscore==expected['reference_score']==spec['reference_path_score']
assert runner==expected['runner_up_score']==spec['runner_up_score']
assert refscore-runner==expected['margin']==spec['reference_margin']
print('VERIFIED',json.dumps(dict(nodes=len(layout),edges=layout.number_of_edges(),scc_sizes=sorted([len(s) for s in nx.strongly_connected_components(layout) if len(s)>1],reverse=True),hamilton_paths=e['n_found'],reference_score=expected['reference_score'],margin=expected['margin'])),flush=True)

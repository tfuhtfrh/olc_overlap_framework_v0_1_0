"""Exact signed Hamilton optimization checks with no fixed endpoint/order.

Symmetry break selects '+' for the lexicographically first physical read;
every reverse-complement pair has exactly one representative satisfying it.
"""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'deps'))
import z3,networkx as nx
def check(g,truth,timeout=60000,query='better'):
    ids=sorted({v[:-1] for v in g});n=len(ids);t=time.monotonic()
    if any(not g.has_edge(u,v) for u,v in zip(truth,truth[1:])):return dict(status='reference_missing_edges')
    target=sum(g[u][v]['weight'] for u,v in zip(truth,truth[1:]))
    p={r:z3.Int(f'p{i}') for i,r in enumerate(ids)};plus={r:z3.Bool(f'o{i}') for i,r in enumerate(ids)}
    arcs={e:z3.Bool(f'e{i}') for i,e in enumerate(g.edges)}
    sl=z3.Solver();sl.set(timeout=timeout)
    sl.add(z3.Distinct(list(p.values())),plus[ids[0]])
    for r in ids:sl.add(p[r]>=0,p[r]<n)
    for r in ids:
        incoming=[(b,1) for (u,v),b in arcs.items() if v[:-1]==r]
        outgoing=[(b,1) for (u,v),b in arcs.items() if u[:-1]==r]
        sl.add(z3.If(p[r]==0,z3.PbEq(incoming,0),z3.PbEq(incoming,1)))
        sl.add(z3.If(p[r]==n-1,z3.PbEq(outgoing,0),z3.PbEq(outgoing,1)))
    for (u,v),b in arcs.items():
        sl.add(z3.Implies(b,z3.And(plus[u[:-1]] if u[-1]=='+' else z3.Not(plus[u[:-1]]),plus[v[:-1]] if v[-1]=='+' else z3.Not(plus[v[:-1]]),p[v[:-1]]==p[u[:-1]]+1)))
    # Existence of a strictly better layout would refute optimality.
    if query=='better':
        sl.add(z3.PbGe([(b,int(g[u][v]['weight'])) for (u,v),b in arcs.items()],target+1))
    elif query=='other_orientation':
        signs={v[:-1]:v[-1]=='+' for v in truth};flip=not signs[ids[0]]
        sl.add(z3.Or([plus[r] != (signs[r] != flip) for r in ids]))
    else:raise ValueError(query)
    result=sl.check();out=dict(status=str(result),query=query,reference_score=target,reference_is_optimal=(result==z3.unsat if query=='better' and result!=z3.unknown else None),seconds=time.monotonic()-t,
        method='Signed Hamilton path SMT, all physical reads once, free endpoints, position constraints exclude subtours; no layout scores depend on reference. For better query, UNSAT excludes a strictly better path; for other_orientation query, UNSAT excludes any non-mirror alternative orientation assignment.')
    if result==z3.sat:
        m=sl.model();path=sorted((r+('+' if z3.is_true(m.eval(plus[r])) else '-') for r in ids),key=lambda v:m.eval(p[v[:-1]]).as_long())
        assert len(set(v[:-1] for v in path))==n and all(g.has_edge(u,v) for u,v in zip(path,path[1:]))
        out['better_path']=path;out['better_score']=sum(g[u][v]['weight'] for u,v in zip(path,path[1:]))
    if result==z3.unknown:out['reason']=sl.reason_unknown()
    return out
if __name__=='__main__':
    folder=Path(sys.argv[1]);g=nx.read_graphml(folder/'graph_with_rc.graphml');truth=[r['id']+r['strand'] for r in json.loads((folder/'truth.json').read_text())]
    r=check(g,truth);(folder/'signed_certificate.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)

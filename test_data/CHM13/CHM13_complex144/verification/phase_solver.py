"""Exact checks for an already orientation-normalized directed OLC graph."""
import sys,time,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'deps'))
import z3,networkx as nx
from audit_graph import rcnode,exact_cigar

def model_graph(g, timeout_ms=60000):
    nodes=sorted(g);n=len(nodes)
    pos={v:z3.Int(f'p{i}') for i,v in enumerate(nodes)}
    edge={e:z3.Bool(f'e{i}') for i,e in enumerate(g.edges)}
    sl=z3.Solver();sl.set(timeout=timeout_ms);sl.add(z3.Distinct(list(pos.values())))
    def eq(items,value):
        return z3.PbEq(items,value) if items else z3.BoolVal(value==0)
    for v in nodes:
        sl.add(pos[v]>=0,pos[v]<n)
        inc=[(b,1) for (u,w),b in edge.items() if w==v]
        out=[(b,1) for (u,w),b in edge.items() if u==v]
        sl.add(z3.If(pos[v]==0,eq(inc,0),eq(inc,1)))
        sl.add(z3.If(pos[v]==n-1,eq(out,0),eq(out,1)))
    for (u,v),b in edge.items():sl.add(z3.Implies(b,pos[v]==pos[u]+1))
    return sl,pos,edge

def quality_weight_parameters(threshold):
    """Return the primitive-integer form of the threshold-margin score.

    Starting from 1000*M-q*B with B=M+d, divide every coefficient by
    gcd(1000-q, q).  This preserves every path ordering while minimizing the
    integer coefficient scale.  For q=980 the result is M-49*d.
    """
    q=int(round(threshold*1000));divisor=math.gcd(1000-q,q)
    return dict(threshold=threshold,q=q,divisor=divisor,
        match_coefficient=(1000-q)//divisor,error_coefficient=q//divisor)

def quality_margin_weight(matches, block, threshold):
    p=quality_weight_parameters(threshold);errors=block-matches
    raw=1000*matches-p['q']*block
    assert raw%p['divisor']==0
    return raw//p['divisor'],raw,p

def weight_specification(threshold):
    p=quality_weight_parameters(threshold)
    return dict(
        scheme='quality_margin_gcd_normalized_v2',
        identity_threshold=threshold,
        symbols=dict(M='exact matching bases',B='alignment block length',d='B-M; mismatch and gap bases'),
        source_formula=f"1000*M - {p['q']}*B",
        primitive_formula=f"{p['match_coefficient']}*M - {p['error_coefficient']}*d",
        divisor=p['divisor'],
        energy_convention='E_layout = -sum(weight) over selected path edges',
        reference_used_in_weight=False)

def annotate_weight_scheme(g, threshold):
    p=quality_weight_parameters(threshold)
    g.graph.update(weight_scheme='quality_margin_gcd_normalized_v2',
        weight_formula=f"{p['match_coefficient']}*M - {p['error_coefficient']}*d",
        weight_source_formula=f"1000*M - {p['q']}*B",
        weight_scale_divisor=p['divisor'],weight_identity_threshold=float(threshold),
        weight_reference_used=False)
    return g

def set_quality_margin_weights(g, threshold):
    """Primitive integer support above the predeclared identity threshold."""
    h=g.copy();annotate_weight_scheme(h,threshold)
    for u,v,d in h.edges(data=True):
        matches=int(d.get('matches',round(float(d['identity'])*int(d['block']))))
        weight,raw,p=quality_margin_weight(matches,int(d['block']),threshold)
        d['matching_bases']=matches;d['errors']=int(d['block'])-matches
        d['weight']=weight;d['raw_quality_margin']=raw;d['weight_divisor']=p['divisor']
        assert d['weight']>=0,(u,v,d)
    return h

def quality_graph(evidence, ids, threshold=.985, min_overlap=300):
    """Select each arc's evidence by quality-margin weight itself."""
    g=nx.DiGraph();g.add_nodes_from(r+s for r in ids for s in '+-');annotate_weight_scheme(g,threshold)
    for e in evidence:
        span=min(e['qspan'],e['tspan'])
        if e['identity']<threshold or span<min_overlap:continue
        matches=exact_cigar(e['cigar'])[2].get('=',0);weight,raw,p=quality_margin_weight(matches,e['block'],threshold)
        d=dict(weight=weight,raw_quality_margin=raw,weight_divisor=p['divisor'],matching_bases=matches,errors=e['block']-matches,identity=e['identity'],block=e['block'],overlap_span=span,paf_line=e['line'])
        for u,v in [(e['source'],e['target']),(rcnode(e['target']),rcnode(e['source']))]:
            if not g.has_edge(u,v) or (weight,matches,span)>(g[u][v]['weight'],g[u][v]['matching_bases'],g[u][v]['overlap_span']):g.add_edge(u,v,**d)
    return g

def better_than(g,truth,timeout_ms=60000):
    if any(not g.has_edge(u,v) for u,v in zip(truth,truth[1:])):return dict(status='reference_missing_edges')
    target=sum(int(g[u][v]['weight']) for u,v in zip(truth,truth[1:]));sl,pos,edge=model_graph(g,timeout_ms);begin=time.monotonic()
    sl.add(z3.PbGe([(b,int(g[u][v]['weight'])) for (u,v),b in edge.items()],target+1));r=sl.check()
    out=dict(status=str(r),reference_score=target,reference_is_optimal=(r==z3.unsat if r!=z3.unknown else None),seconds=time.monotonic()-begin)
    if r==z3.sat:
        m=sl.model();path=sorted(g,key=lambda v:m.eval(pos[v]).as_long());out.update(better_score=sum(int(g[u][v]['weight']) for u,v in zip(path,path[1:])),better_path=path)
    if r==z3.unknown:out['reason']=sl.reason_unknown()
    return out

def enumerate_paths(g,limit=1000,timeout_ms=60000):
    sl,pos,edge=model_graph(g,timeout_ms);paths=[];scores=[];begin=time.monotonic();end='cap'
    for _ in range(limit):
        r=sl.check()
        if r!=z3.sat:end=str(r);break
        m=sl.model();path=sorted(g,key=lambda v:m.eval(pos[v]).as_long());used=[edge[(u,v)] for u,v in zip(path,path[1:])]
        assert all((u,v) in edge and z3.is_true(m.eval(edge[(u,v)])) for u,v in zip(path,path[1:]))
        paths.append(path);scores.append(sum(int(g[u][v]['weight']) for u,v in zip(path,path[1:])))
        sl.add(z3.Or([z3.Not(b) for b in used]))
    return dict(n_found=len(paths),end=end,seconds=time.monotonic()-begin,score_min=min(scores) if scores else None,score_max=max(scores) if scores else None,paths=paths,scores=scores)

if __name__=='__main__':
    folder=Path(sys.argv[1]);g=nx.read_graphml(folder/'graph.graphml');truth=json.loads((folder/'reference_path.json').read_text())['nodes']
    b=better_than(g,truth);e=enumerate_paths(g);print(json.dumps({**b,'enumeration':{k:v for k,v in e.items() if k!='paths'}},indent=2))

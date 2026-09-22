"""Read-only graph construction from exact PAF; truth is evaluation only.
No transitive reduction, tip clipping, bubble popping or reference-added edges.
"""
import sys,json,re,gzip,collections,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'deps'))
import networkx as nx
import z3
ROOT=Path(__file__).parent
def rcnode(v): return v[:-1]+('-' if v[-1]=='+' else '+')
def readfa(p):
    seqs={}; name=None
    for line in open(p):
        if line.startswith('>'): name=line[1:].split()[0]; seqs[name]=''
        else: seqs[name]+=line.strip()
    return seqs
def exact_cigar(s):
    count=collections.Counter()
    for n,op in re.findall(r'(\d+)([=XMID])',s): count[op]+=int(n)
    if count['M']: raise ValueError('--eqx required')
    block=sum(count.values())
    return count['=']/block,block,dict(count)
def pafedges(paf):
    edges=[]; contained=[]; lines=0
    op=gzip.open if str(paf).endswith('.gz') else open
    with op(paf,'rt') as f:
        for line in f:
            lines+=1; a=line.rstrip().split('\t'); q,ql,qs,qe,sg,t,tl,ts,te=a[:9]
            if q==t: continue
            ql,qs,qe,tl,ts,te=map(int,(ql,qs,qe,tl,ts,te))
            tags={v[:2]:v[5:] for v in a[12:]}
            ident,block,cig=exact_cigar(tags['cg'])
            u,v=q+'+',t+sg
            if sg=='-': ts,te=tl-te,tl-ts
            tol=min(100,max(5,int(.02*min(qe-qs,te-ts))))
            common=dict(identity=ident,block=block,qspan=qe-qs,tspan=te-ts,cigar=tags['cg'],line=lines)
            if qs<=tol and ql-qe<=tol:
                contained.append(dict(query=q,target=t,identity=ident,block=block,query_covered=qe-qs))
            if ts<=tol and tl-te<=tol:
                contained.append(dict(query=t,target=q,identity=ident,block=block,query_covered=te-ts))
            # Strict outward extensions on both reads; aligned endpoints may differ by tol.
            if ql-qe<=tol and ts<=tol and qs>tol and tl-te>tol:
                edges.append(dict(source=u,target=v,source_tail=ql-qe,target_head=ts,**common))
            if tl-te<=tol and qs<=tol and ts>tol and ql-qe>tol:
                edges.append(dict(source=v,target=u,source_tail=tl-te,target_head=qs,**common))
    return edges,contained,lines
def solve(ids,arcs,limit=12,fixed=None):
    ns=[r+s for r in ids for s in '+-']; n=len(ids)
    used={v:z3.Bool('u_'+v) for v in ns}; pos={v:z3.Int('p_'+v) for v in ns}
    starts={v:z3.Bool('s_'+v) for v in ns}; ends={v:z3.Bool('t_'+v) for v in ns}
    es={e:z3.Bool('e_'+str(i)) for i,e in enumerate(sorted(arcs))}
    sl=z3.Solver(); sl.set(timeout=15000)
    sl.add(z3.PbEq([(starts[v],1) for v in ns],1),z3.PbEq([(ends[v],1) for v in ns],1))
    for r in ids: sl.add(z3.Xor(used[r+'+'],used[r+'-']))
    for v in ns:
        sl.add(z3.Implies(starts[v],used[v]),z3.Implies(ends[v],used[v]))
        sl.add(z3.If(used[v],z3.And(pos[v]>=0,pos[v]<n),pos[v]==-1))
        sl.add(z3.Implies(starts[v],pos[v]==0),z3.Implies(ends[v],pos[v]==n-1))
        incoming=[z3.If(b,1,0) for (a,c),b in es.items() if c==v]
        outgoing=[z3.If(b,1,0) for (a,c),b in es.items() if a==v]
        sl.add(z3.Sum(incoming)==z3.If(used[v],1,0)-z3.If(starts[v],1,0))
        sl.add(z3.Sum(outgoing)==z3.If(used[v],1,0)-z3.If(ends[v],1,0))
    for (u,v),b in es.items(): sl.add(z3.Implies(b,z3.And(used[u],used[v],pos[v]==pos[u]+1)))
    if fixed: sl.add(starts[fixed[0]],ends[fixed[1]])
    paths=[]; status='cap'; begin=time.monotonic()
    for _ in range(limit):
        result=sl.check()
        if result!=z3.sat: status=str(result); break
        m=sl.model(); path=sorted((v for v in ns if z3.is_true(m.eval(used[v]))),key=lambda v:m.eval(pos[v]).as_long())
        assert len(path)==n and len({v[:-1] for v in path})==n
        assert all((a,b) in arcs for a,b in zip(path,path[1:]))
        paths.append(path)
        sl.add(z3.Or([z3.Not(es[(a,b)]) for a,b in zip(path,path[1:])]))
    classes={min(tuple(p),tuple(rcnode(v) for v in p[::-1])) for p in paths}
    return dict(paths=paths,n_solutions_found=len(paths),rc_classes_found=len(classes),enumeration_end=status,seconds=time.monotonic()-begin)
def build(folder,paf):
    seqs=readfa(folder/'reads.fasta'); ids=sorted(seqs)
    evidence,contains,lines=pafedges(paf)
    (folder/'edge_evidence.json').write_text(json.dumps(evidence,indent=2))
    (folder/'paf_containment_candidates.json').write_text(json.dumps(contains,indent=2))
    truth=json.loads((folder/'truth.json').read_text()); truthnodes=[r['id']+r['strand'] for r in truth]
    trutharcs=set(zip(truthnodes,truthnodes[1:])); truthpos={v:i for i,v in enumerate(truthnodes)}
    summaries=[]
    for threshold in (.98,.985,.99,.995):
        for minol in (300,500,1000):
            chosen={}
            for e in evidence:
                if e['identity']<threshold or min(e['qspan'],e['tspan'])<minol: continue
                u,v=e['source'],e['target']
                for arc in [(u,v),(rcnode(v),rcnode(u))]:
                    if arc not in chosen or e['identity']>chosen[arc]['identity']: chosen[arc]=e
            g=nx.DiGraph();g.add_nodes_from(r+s for r in ids for s in '+-');g.add_edges_from(chosen)
            phase=g.subgraph(truthnodes)
            result=dict(identity=threshold,min_overlap=minol,physical_reads=len(ids),oriented_nodes=len(g),arcs=len(chosen),
                phase_arcs=phase.number_of_edges(),phase_splits=sum(d>1 for _,d in phase.out_degree()),
                phase_merges=sum(d>1 for _,d in phase.in_degree()),phase_nontrivial_sccs=[len(s) for s in nx.strongly_connected_components(phase) if len(s)>1],
                phase_back_edges=[(truthpos[u],truthpos[v]) for u,v in phase.edges if truthpos[v]<truthpos[u]],
                truth_edges_present=len(trutharcs & chosen.keys()),truth_edges_total=len(trutharcs),
                missing_truth_edges=[(truthpos[u],truthpos[v]) for u,v in trutharcs-chosen.keys()],
                phase_flip_arcs=sum((u in truthpos)!=(v in truthpos) for u,v in chosen),
                mirror_closed=all((rcnode(v),rcnode(u)) in chosen for u,v in chosen))
            # Graph construction above does not consult truth; below only assesses known reference layout.
            sol=solve(ids,chosen)
            result['solver']=sol
            result['solver']['contains_reference_order']=truthnodes in sol['paths'] or [rcnode(v) for v in truthnodes[::-1]] in sol['paths']
            name=f'graph_i{threshold}_o{minol}'
            (folder/(name+'.json')).write_text(json.dumps([dict(source=u,target=v,identity=e['identity'],block=e['block']) for (u,v),e in sorted(chosen.items())],indent=2))
            nx.write_graphml(g,folder/(name+'.graphml'))
            print(json.dumps({k:v for k,v in result.items() if k!='solver'}),{k:v for k,v in sol.items() if k!='paths'},flush=True)
            summaries.append(result)
    (folder/'audit_summary.json').write_text(json.dumps(dict(paf_lines=lines,summary=summaries),indent=2))
if __name__=='__main__':
    folder=Path(sys.argv[1]); paf=Path(sys.argv[2]); build(folder,paf)

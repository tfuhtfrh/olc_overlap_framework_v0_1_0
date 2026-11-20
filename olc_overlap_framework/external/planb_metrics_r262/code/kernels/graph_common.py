from __future__ import annotations
import copy, pickle
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, Tuple
import networkx as nx


def _normalize_directed_graph(g: nx.DiGraph | nx.MultiDiGraph, *, owned_storage=False) -> nx.MultiDiGraph:
    if not isinstance(g, (nx.DiGraph, nx.MultiDiGraph)) or not g.is_directed():
        raise TypeError(f"Expected directed graph, got {type(g)!r}")
    if owned_storage and _canonical_scalar_multigraph(g):
        # The file loader owns this object. Match the legacy insertion order
        # without a second full topology, attribute copy, or global edge list.
        g.graph=copy.deepcopy(g.graph)
        for u in g:
            neighbors=g._succ[u]
            for keys in neighbors.values():
                if len(keys)>1:
                    rows=sorted(keys.items(),key=lambda item:str(item[0]))
                    keys.clear();keys.update(rows)
                for data in keys.values():data.setdefault('edge_kind','ORIGINAL')
            rows=sorted(neighbors.items(),key=lambda item:str(item[0]))
            neighbors.clear();neighbors.update(rows)
        for neighbors in g._pred.values():
            rows=sorted(neighbors.items(),key=lambda item:str(item[0]))
            neighbors.clear();neighbors.update(rows)
        return g
    out=nx.MultiDiGraph()
    out.graph.update(copy.deepcopy(g.graph))
    out.add_nodes_from((n,copy.deepcopy(d)) for n,d in g.nodes(data=True))
    rows=[]
    if g.is_multigraph():
        for i,(u,v,k,d) in enumerate(g.edges(keys=True,data=True)):
            rows.append((str(u),str(v),str(k),i,u,v,k,d))
    else:
        for i,(u,v,d) in enumerate(g.edges(data=True)):
            rows.append((str(u),str(v),"",i,u,v,None,d))
    rows.sort(key=lambda x:x[:4])
    used=set()
    for j,(_su,_sv,_sk,_i,u,v,k,attrs) in enumerate(rows):
        d=copy.deepcopy(dict(attrs))
        eid=str(d.get('edge_id') or (k if k not in (None,'') else f'E::{j:012d}'))
        if eid in used:
            base=eid; z=1
            while f'{base}::{z}' in used: z+=1
            eid=f'{base}::{z}'
        used.add(eid)
        d['edge_id']=eid
        d.setdefault('edge_kind','ORIGINAL')
        out.add_edge(u,v,key=eid,**d)
    return out


def _canonical_scalar_multigraph(g):
    """A fast path only; every unsupported input retains legacy normalization."""
    scalar=(str,bytes,int,float,complex,bool,type(None))
    if not isinstance(g,nx.MultiDiGraph) or any(type(n)is not str for n in g):return False
    if any(type(k)is not str or type(v)not in scalar for _,d in g.nodes(data=True) for k,v in d.items()):return False
    seen=set()
    for _,_,key,data in g.edges(keys=True,data=True):
        if type(key)is not str or not key or data.get('edge_id')!=key or key in seen:return False
        if any(type(k)is not str or type(v)not in scalar for k,v in data.items()):return False
        seen.add(key)
    return True


def load_graph(path: str | Path, *, owned_storage=False) -> nx.MultiDiGraph:
    """Load contained-read-removed authoritative OLC without losing parallel edges."""
    path=Path(path).expanduser().resolve()
    if not path.exists(): raise FileNotFoundError(path)
    suf=path.suffix.lower()
    if suf in {'.graphml','.xml'}:
        g=nx.read_graphml(path)
    elif suf in {'.pkl','.pickle','.gpickle'}:
        with path.open('rb') as f: obj=pickle.load(f)
        if isinstance(obj,(nx.DiGraph,nx.MultiDiGraph)): g=obj
        elif isinstance(obj,dict):
            g=None
            for key in ('graph','G','contained_removed','contained_removed_graph'):
                if isinstance(obj.get(key),(nx.DiGraph,nx.MultiDiGraph)):
                    g=obj[key]; break
            if g is None: raise TypeError(f'pickle dict has no recognized directed graph; keys={list(obj)[:20]}')
        else: raise TypeError(f'unsupported pickle payload: {type(obj)!r}')
    else: raise ValueError(f'Unsupported graph format: {path}')
    return _normalize_directed_graph(g,owned_storage=owned_storage)


def topology_view(g: nx.DiGraph | nx.MultiDiGraph) -> nx.DiGraph:
    h=nx.DiGraph()
    h.add_nodes_from(g.nodes(data=True))
    if g.is_multigraph():
        h.add_edges_from((u,v) for u,v,_k in g.edges(keys=True))
    else:
        h.add_edges_from(g.edges())
    return h


def physical_and_orientation(data: Dict[str, Any], node: Hashable | None = None) -> Tuple[str, str]:
    r=data.get('physical_read')
    if r in (None,''):
        for k in ('physical_id','read_id','resource_id'):
            if data.get(k) not in (None,''): r=data[k]; break
    o=data.get('orientation'); s=str(node) if node is not None else ''
    if r in (None,'') and s.endswith(('+','-')): r=s[:-1]
    if o in (None,'') and s.endswith(('+','-')): o=s[-1]
    if r in (None,'') or o in (None,''): raise KeyError('Each oriented node must provide physical_read/orientation attributes or node name ending +/-')
    r,o=str(r),str(o)
    if o not in {'+','-'}: raise ValueError(f'Invalid orientation {o!r}')
    return r,o


def build_mate_map(g: nx.Graph) -> Dict[Hashable,Hashable]:
    by_read: Dict[str,Dict[str,Hashable]]={}
    for v,data in g.nodes(data=True):
        r,o=physical_and_orientation(data,v); d=by_read.setdefault(r,{})
        if o in d: raise ValueError(f'Duplicate orientation {o} for physical read {r}')
        d[o]=v
    mate={}; bad=[]
    for r,d in by_read.items():
        if set(d)!={'+','-'}: bad.append((r,sorted(d))); continue
        mate[d['+']]=d['-']; mate[d['-']]=d['+']
    if bad: raise ValueError(f'Incomplete +/- physical-read pairs: {bad[:8]}')
    return mate


def path_physical_unique(g: nx.Graph,path:Iterable[Hashable])->bool:
    seen=set()
    for v in path:
        r,_=physical_and_orientation(g.nodes[v],v)
        if r in seen:return False
        seen.add(r)
    return True

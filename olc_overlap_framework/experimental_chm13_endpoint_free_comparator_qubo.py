"""Endpoint-free latent-rank comparator QUBO for CHM13.

This variant removes source/sink variables entirely.

Selected edges must satisfy:
- at most one incoming edge per vertex,
- at most one outgoing edge per vertex,
- strict latent rank increase on every selected edge.

The strict-rank condition excludes directed cycles.  Therefore the selected
graph is a disjoint union of directed paths.  A dominant linear reward for
selecting edges maximizes cardinality.  If a Hamilton path exists, the maximum
is N-1 edges, which implies exactly one path component covering all N vertices.

Among maximum-cardinality paths, the normalized overlap cost breaks ties.
"""
from __future__ import annotations
import json,math
from dataclasses import dataclass

from demo_chm13_edge_ordered_path_qubo import DATASET_DIR,load_chm13_graph
from olc_pipeline.layout_solver import QUBOModel
from experimental_chm13_latent_rank_comparator_qubo import (
    add_first_borrow_penalty,add_general_borrow_penalty,
)


@dataclass(frozen=True)
class EndpointFreeComparatorConfig:
    cardinality_reward: float=144.0
    degree_conflict_penalty: float=288.0
    comparator_penalty: float=576.0
    order_gate_penalty: float=288.0
    edge_cost_scale: float=1.0


class EndpointFreeComparatorModel(QUBOModel):
    def __init__(self,read_ids,edge_pairs,k):
        self.edge_pairs=list(edge_pairs);self.k=k
        labels=[];self.edge_index={};self.rank_index={};self.borrow_index={}
        for e in edge_pairs:
            self.edge_index[e]=len(labels);labels.append(f"x[{e[0]},{e[1]}]")
        for r in read_ids:
            for bit in range(k):
                self.rank_index[(r,bit)]=len(labels);labels.append(f"p[{r},{bit}]")
        for e in edge_pairs:
            for j in range(1,k+1):
                self.borrow_index[(e,j)]=len(labels);labels.append(f"b[{e[0]},{e[1]},{j}]")
        super().__init__(read_ids=list(read_ids),linear=[0.0]*len(labels))
        self._labels=labels
    def variable_label(self,i):return self._labels[i]


def build_model(cfg=EndpointFreeComparatorConfig()):
    reads,edges,reward=load_chm13_graph(DATASET_DIR)
    rids=[r.rid for r in reads];rank0={r:i for i,r in enumerate(rids)}
    ep=sorted(reward,key=lambda e:(rank0[e[0]],rank0[e[1]]))
    n=len(rids);k=max(1,math.ceil(math.log2(max(2,n))))
    m=EndpointFreeComparatorModel(rids,ep,k)
    shift=max(reward.values(),default=0.0)
    raw={e:shift-reward[e] for e in ep};norm=max(max(raw.values(),default=1.0),1.0)

    incoming={r:[] for r in rids};outgoing={r:[] for r in rids}
    for e in ep:
        u,v=e;incoming[v].append(e);outgoing[u].append(e)
        x=m.edge_index[e]
        m.add_linear(x,-cfg.cardinality_reward+cfg.edge_cost_scale*raw[e]/norm)

    # At-most-one incoming / outgoing.  Pairwise form adds no auxiliaries.
    for r in rids:
        for group in (incoming[r],outgoing[r]):
            for i,e1 in enumerate(group):
                for e2 in group[i+1:]:
                    m.add_quadratic(
                        m.edge_index[e1],m.edge_index[e2],
                        cfg.degree_conflict_penalty,
                    )

    # Latent comparator exactly as in the source/sink version.
    for e in ep:
        u,v=e
        for bit in range(k):
            ub=m.rank_index[(u,bit)];vb=m.rank_index[(v,bit)]
            bout=m.borrow_index[(e,bit+1)]
            if bit==0:
                add_first_borrow_penalty(m,ub,vb,bout,cfg.comparator_penalty)
            else:
                bin_=m.borrow_index[(e,bit)]
                add_general_borrow_penalty(m,ub,vb,bin_,bout,cfg.comparator_penalty)
        m.add_quadratic(
            m.edge_index[e],m.borrow_index[(e,k)],cfg.order_gate_penalty
        )

    m.quadratic={ij:v for ij,v in m.quadratic.items() if abs(v)>1e-12}
    return m,reward,cfg


def encode_order(m,order,ranks=None):
    if ranks is None:ranks={r:i for i,r in enumerate(order)}
    s=[0]*m.num_variables
    for r,val in ranks.items():
        for bit in range(m.k):
            if (val>>bit)&1:s[m.rank_index[(r,bit)]]=1
    for e in zip(order,order[1:]):s[m.edge_index[e]]=1
    for e in m.edge_pairs:
        u,v=e;U=ranks[u];V=ranks[v];borrow=1
        for bit in range(m.k):
            borrow=int(((U>>bit)&1)+borrow>((V>>bit)&1))
            if borrow:s[m.borrow_index[(e,bit+1)]]=1
    return s


def audit(s,m,reward,cfg):
    selected=[e for e in m.edge_pairs if s[m.edge_index[e]]]
    indeg={r:0 for r in m.read_ids};outdeg={r:0 for r in m.read_ids};nxt={}
    for u,v in selected:
        indeg[v]+=1;outdeg[u]+=1
        if u not in nxt:nxt[u]=v
    conflicts=sum(max(0,d*(d-1)//2) for d in indeg.values())+sum(max(0,d*(d-1)//2) for d in outdeg.values())
    ranks={r:sum((1<<b)*int(s[m.rank_index[(r,b)]]) for b in range(m.k)) for r in m.read_ids}
    cmpv=0;ordv=0
    for e in m.edge_pairs:
        u,v=e;borrow=1
        for bit in range(m.k):
            expected=int(((ranks[u]>>bit)&1)+borrow>((ranks[v]>>bit)&1))
            actual=int(s[m.borrow_index[(e,bit+1)]])
            cmpv+=int(expected!=actual);borrow=expected
        if s[m.edge_index[e]] and ranks[v]<=ranks[u]:ordv+=1

    sources=[r for r in m.read_ids if indeg[r]==0 and outdeg[r]<=1]
    sinks=[r for r in m.read_ids if outdeg[r]==0 and indeg[r]<=1]
    structural=False;order=[]
    if conflicts==0 and len(selected)==len(m.read_ids)-1 and len(sources)==1 and len(sinks)==1:
        seen=set();cur=sources[0]
        while cur not in seen:
            seen.add(cur);order.append(cur)
            if cur==sinks[0]:break
            if cur not in nxt:break
            cur=nxt[cur]
        structural=len(order)==len(m.read_ids) and order[-1]==sinks[0]
    valid=structural and cmpv==0 and ordv==0
    score=int(sum(reward[e] for e in zip(order,order[1:]))) if valid else None
    return {"energy":float(m.energy(s)),"valid_path":valid,"path_score":score,
      "selected_edges":len(selected),"degree_conflicts":int(conflicts),
      "derived_sources":len(sources),"derived_sinks":len(sinks),
      "comparator_violations":int(cmpv),"selected_order_violations":int(ordv),
      "distinct_ranks":len(set(ranks.values())),"hamming_weight":sum(s)}


if __name__=="__main__":
    m,reward,cfg=build_model()
    ref=json.loads((DATASET_DIR/"reference_path.json").read_text())["normalized_nodes"]
    ranksets={
      "consecutive":{r:i for i,r in enumerate(ref)},
      "gapped":{r:int(round(i*255/(len(ref)-1))) for i,r in enumerate(ref)},
    }
    for name,ranks in ranksets.items():
        s=encode_order(m,ref,ranks)
        print(json.dumps({"scheme":name,"variables":m.num_variables,
          "quadratic_terms":len(m.quadratic),
          "max_abs_linear":max(abs(x) for x in m.linear),
          "max_abs_quadratic":max(abs(x) for x in m.quadratic.values()),
          "audit":audit(s,m,reward,cfg)},indent=2))

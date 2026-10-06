"""Experimental latent vertex-order QUBO with independent weighted edge bits.

Key design goals:
- x_e is the only edge-selection / edge-weight variable.
- Vertex position bits p[v,k] exist independently of edge selection and may take
  arbitrary values when a candidate edge is not selected.
- A selected edge u->v enforces P_v = P_u + 1.
- The conditional increment is quadratized with ONE product auxiliary per bit
  plus ripple carries.  Local coefficients are bounded (no 2^k coefficients
  inside a squared residual).

Compared with experimental_chm13_vertex_carry_qubo.py, this eliminates the
second masked copy b[e,k] = x_e*p[v,k].  Only a[e,k] = x_e*p[u,k] is needed.
"""
from __future__ import annotations
import json, math
from dataclasses import dataclass
from pathlib import Path

from demo_chm13_edge_ordered_path_qubo import DATASET_DIR,load_chm13_graph
from olc_pipeline.layout_solver import QUBOModel


@dataclass(frozen=True)
class LatentVertexOrderConfig:
    degree_penalty: float=144.0
    void_penalty: float=144.0
    order_penalty: float=144.0
    product_penalty: float=144.0
    carry_gate_penalty: float=144.0
    edge_cost_scale: float=1.0


class LatentVertexOrderModel(QUBOModel):
    void_id="__void__"
    def __init__(self,read_ids,edge_pairs,k):
        self.edge_pairs=list(edge_pairs); self.k=k
        labels=[]; self.edge_index={}; self.pos_index={}; self.prod_index={}; self.carry_index={}
        for e in edge_pairs:
            self.edge_index[e]=len(labels); labels.append(f"x[{e[0]},{e[1]}]")
        self.source_index={}; self.sink_index={}
        for r in read_ids:
            self.source_index[r]=len(labels); labels.append(f"s[{r}]")
        for r in read_ids:
            self.sink_index[r]=len(labels); labels.append(f"t[{r}]")
        for r in read_ids:
            for b in range(k):
                self.pos_index[(r,b)]=len(labels); labels.append(f"p[{r},{b}]")
        # a[e,k] = x_e * p[u,k]
        for e in edge_pairs:
            for b in range(k):
                self.prod_index[(e,b)]=len(labels); labels.append(f"a[{e[0]},{e[1]},{b}]")
        # carry into bit k, k=1..K-1. carry into bit 0 is x_e; top carry out fixed 0.
        for e in edge_pairs:
            for b in range(1,k):
                self.carry_index[(e,b)]=len(labels); labels.append(f"c[{e[0]},{e[1]},{b}]")
        super().__init__(read_ids=list(read_ids),linear=[0.0]*len(labels))
        self._labels=labels
    def variable_label(self,i): return self._labels[i]


def add_square(model,coeffs,constant,penalty):
    items=[(i,float(a)) for i,a in coeffs.items() if a]
    model.add_constant(penalty*constant*constant)
    for i,a in items:model.add_linear(i,penalty*(a*a+2*constant*a))
    for oi,(i,a) in enumerate(items):
        for j,b in items[oi+1:]:model.add_quadratic(i,j,2*penalty*a*b)


def add_rosenberg(model,x,u,a,penalty):
    # a = x*u : xu - 2xa - 2ua + 3a
    model.add_quadratic(x,u,penalty)
    model.add_quadratic(x,a,-2*penalty)
    model.add_quadratic(u,a,-2*penalty)
    model.add_linear(a,3*penalty)


def build_model(cfg=LatentVertexOrderConfig()):
    reads,edges,reward=load_chm13_graph(DATASET_DIR)
    rids=[r.rid for r in reads]; rank={r:i for i,r in enumerate(rids)}
    ep=sorted(reward,key=lambda e:(rank[e[0]],rank[e[1]]))
    n=len(rids); k=max(1,math.ceil(math.log2(max(2,n))))
    m=LatentVertexOrderModel(rids,ep,k)
    shift=max(reward.values(),default=0); raw={e:shift-reward[e] for e in ep}
    norm=max(max(raw.values(),default=1),1)
    incoming={r:[] for r in rids}; outgoing={r:[] for r in rids}
    for e in ep:
        u,v=e; incoming[v].append(e); outgoing[u].append(e)
        m.add_linear(m.edge_index[e],cfg.edge_cost_scale*raw[e]/norm)

    for r in rids:
        add_square(m,{m.source_index[r]:-1,**{m.edge_index[e]:-1 for e in incoming[r]}},1,cfg.degree_penalty)
        add_square(m,{m.sink_index[r]:-1,**{m.edge_index[e]:-1 for e in outgoing[r]}},1,cfg.degree_penalty)
    add_square(m,{m.source_index[r]:-1 for r in rids},1,cfg.void_penalty)
    add_square(m,{m.sink_index[r]:-1 for r in rids},1,cfg.void_penalty)

    for e in ep:
        u,v=e; x=m.edge_index[e]
        # Product masks and carry gates.
        for b in range(k):
            pu=m.pos_index[(u,b)]; a=m.prod_index[(e,b)]
            add_rosenberg(m,x,pu,a,cfg.product_penalty)
        for b in range(1,k):
            c=m.carry_index[(e,b)]
            # c <= x
            m.add_linear(c,cfg.carry_gate_penalty)
            m.add_quadratic(c,x,-cfg.carry_gate_penalty)

        # Conditional ripple increment.
        # When x=1: P_v = P_u + 1.
        # When x=0: a=0 and carries=0, and every local penalty below is exactly 0
        # regardless of p_u,p_v. Thus latent vertex positions remain free.
        for b in range(k):
            pu=m.pos_index[(u,b)]; pv=m.pos_index[(v,b)]
            a=m.prod_index[(e,b)]
            if b==0:
                # x*(pu + x - pv - 2*c1)^2 after using a=x*pu and c1<=x:
                # x + 3a - x*pv - 2a*pv - 4pu*c1 + 4pv*c1
                m.add_linear(x,cfg.order_penalty)
                m.add_linear(a,3*cfg.order_penalty)
                m.add_quadratic(x,pv,-cfg.order_penalty)
                m.add_quadratic(a,pv,-2*cfg.order_penalty)
                if k>1:
                    co=m.carry_index[(e,1)]
                    m.add_quadratic(pu,co,-4*cfg.order_penalty)
                    m.add_quadratic(pv,co,4*cfg.order_penalty)
            elif b==k-1:
                ci=m.carry_index[(e,b)]
                # x*(pu+ci-pv)^2
                m.add_linear(a,cfg.order_penalty)
                m.add_linear(ci,cfg.order_penalty)
                m.add_quadratic(x,pv,cfg.order_penalty)
                m.add_quadratic(pu,ci,2*cfg.order_penalty)
                m.add_quadratic(a,pv,-2*cfg.order_penalty)
                m.add_quadratic(ci,pv,-2*cfg.order_penalty)
            else:
                ci=m.carry_index[(e,b)]; co=m.carry_index[(e,b+1)]
                # x*(pu+ci-pv-2co)^2 with ci,co <= x and a=x*pu.
                m.add_linear(a,cfg.order_penalty)
                m.add_linear(ci,cfg.order_penalty)
                m.add_quadratic(x,pv,cfg.order_penalty)
                m.add_linear(co,4*cfg.order_penalty)
                m.add_quadratic(pu,ci,2*cfg.order_penalty)
                m.add_quadratic(a,pv,-2*cfg.order_penalty)
                m.add_quadratic(pu,co,-4*cfg.order_penalty)
                m.add_quadratic(ci,pv,-2*cfg.order_penalty)
                m.add_quadratic(ci,co,-4*cfg.order_penalty)
                m.add_quadratic(pv,co,4*cfg.order_penalty)

    m.quadratic={ij:v for ij,v in m.quadratic.items() if abs(v)>1e-12}
    return m,reward,cfg


def encode_order(m:LatentVertexOrderModel,order:list[str],offset:int=0):
    if offset<0 or offset+len(order)-1 >= (1<<m.k): raise ValueError("offset does not fit")
    s=[0]*m.num_variables
    pos={r:offset+i for i,r in enumerate(order)}
    s[m.source_index[order[0]]]=1; s[m.sink_index[order[-1]]]=1
    for r,val in pos.items():
        for b in range(m.k):
            if (val>>b)&1:s[m.pos_index[(r,b)]]=1
    for u,v in zip(order,order[1:]):
        e=(u,v); s[m.edge_index[e]]=1
        pu=pos[u]
        for b in range(m.k):
            if (pu>>b)&1:s[m.prod_index[(e,b)]]=1
        carry=1
        for b in range(1,m.k):
            carry &= (pu>>(b-1))&1
            if carry:s[m.carry_index[(e,b)]]=1
    return s


def audit(s,m,reward,cfg):
    selected=[e for e in m.edge_pairs if s[m.edge_index[e]]]
    sources=[r for r in m.read_ids if s[m.source_index[r]]]
    sinks=[r for r in m.read_ids if s[m.sink_index[r]]]
    indeg={r:0 for r in m.read_ids};outdeg={r:0 for r in m.read_ids};nxt={}
    for u,v in selected:
        indeg[v]+=1;outdeg[u]+=1
        if u not in nxt:nxt[u]=v
    dsq=sum((1-int(r in sources)-indeg[r])**2+(1-int(r in sinks)-outdeg[r])**2 for r in m.read_ids)
    vsq=(1-len(sources))**2+(1-len(sinks))**2
    prod=0;carry_gate=0;order_sq=0
    positions={}
    for r in m.read_ids:
        positions[r]=sum((1<<b)*int(s[m.pos_index[(r,b)]]) for b in range(m.k))
    for e in m.edge_pairs:
        u,v=e;x=int(s[m.edge_index[e]])
        for b in range(m.k):
            a=int(s[m.prod_index[(e,b)]])
            pu=int(s[m.pos_index[(u,b)]])
            prod+=int(a!=x*pu)
        for b in range(1,m.k):
            c=int(s[m.carry_index[(e,b)]])
            carry_gate+=int(c and not x)
        if x:
            order_sq+=(positions[v]-positions[u]-1)**2
    structural=False;order=[]
    if dsq==0 and vsq==0 and len(sources)==1 and len(sinks)==1:
        seen=set();cur=sources[0]
        while cur not in seen:
            seen.add(cur);order.append(cur)
            if cur==sinks[0]:break
            if cur not in nxt:break
            cur=nxt[cur]
        structural=len(order)==len(m.read_ids) and order[-1]==sinks[0]
    valid=structural and prod==0 and carry_gate==0 and order_sq==0
    score=int(sum(reward[e] for e in zip(order,order[1:]))) if valid else None
    return {"energy":float(m.energy(s)),"valid_path":valid,"path_score":score,
      "selected_edges":len(selected),"sources":len(sources),"sinks":len(sinks),
      "degree_residual_sq":int(dsq),"void_residual_sq":int(vsq),
      "product_violations":int(prod),"carry_gate_violations":int(carry_gate),
      "selected_edge_order_residual_sq":int(order_sq),
      "distinct_positions":len(set(positions.values())),"hamming_weight":sum(s)}


if __name__=="__main__":
    m,reward,cfg=build_model()
    ref=json.loads((DATASET_DIR/"reference_path.json").read_text())["normalized_nodes"]
    for off in (0,1,10,100):
        if off+len(ref)-1 >= (1<<m.k):continue
        s=encode_order(m,ref,off)
        print(json.dumps({"offset":off,"variables":m.num_variables,
            "quadratic_terms":len(m.quadratic),
            "max_abs_linear":max(abs(x) for x in m.linear),
            "max_abs_quadratic":max(abs(x) for x in m.quadratic.values()),
            "audit":audit(s,m,reward,cfg)},indent=2))

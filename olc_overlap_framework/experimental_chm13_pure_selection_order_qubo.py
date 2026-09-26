"""Experimental weighted Nusslein-style path QUBO with pure edge-selection variables.

This is a middle formulation between:
1) the current edge-position model P_e = x_e + sum 2^k z_e,k, where x_e is
   both selection and part of the numeric position, and
2) the bounded vertex/carry model, which greatly expands the auxiliary space.

Here:
- x_e is PURELY the edge-selection / weighted-objective variable;
- P_e = sum 2^k z_e,k is a separate edge position;
- exact activation x_e <-> P_e>0 is enforced with quadratic penalties plus
  ceil(log2 K) slack bits per edge;
- source/sink bits close the path through one virtual node but do not carry
  edge reward.

The order arithmetic remains binary-integer and therefore still has the
large-coefficient issue.  This experiment isolates whether separating weight
selection from position semantics improves search behavior while retaining a
single static QUBO.
"""
from __future__ import annotations
import json, math
from dataclasses import dataclass
from pathlib import Path

from demo_chm13_edge_ordered_path_qubo import DATASET_DIR,load_chm13_graph
from olc_pipeline.layout_solver import QUBOModel


@dataclass(frozen=True)
class PureSelectionOrderConfig:
    degree_penalty: float=144.0
    void_penalty: float=144.0
    order_penalty: float=144.0
    activation_penalty: float=144.0
    edge_cost_scale: float=1.0


class PureSelectionOrderModel(QUBOModel):
    void_id="__void__"
    def __init__(self,read_ids,edge_pairs,k):
        self.edge_pairs=list(edge_pairs); self.k=k
        self._ri={r:i for i,r in enumerate(read_ids)}
        labels=[]; self.edge_index={}; self.pos_index={}; self.slack_index={}
        for e in edge_pairs:
            self.edge_index[e]=len(labels); labels.append(f"x[{e[0]},{e[1]}]")
        for e in edge_pairs:
            for b in range(k):
                self.pos_index[(e,b)]=len(labels); labels.append(f"z[{e[0]},{e[1]},{b}]")
        self.slack_bits=max(1,math.ceil(math.log2(max(2,k))))
        for e in edge_pairs:
            for b in range(self.slack_bits):
                self.slack_index[(e,b)]=len(labels); labels.append(f"q[{e[0]},{e[1]},{b}]")
        self.source_index={}; self.sink_index={}
        for r in read_ids:
            self.source_index[r]=len(labels); labels.append(f"s[{r}]")
        for r in read_ids:
            self.sink_index[r]=len(labels); labels.append(f"t[{r}]")
        super().__init__(read_ids=list(read_ids),linear=[0.0]*len(labels))
        self._labels=labels
    def variable_label(self,i): return self._labels[i]


def add_square(model,coeffs,constant,penalty):
    items=[(i,float(a)) for i,a in coeffs.items() if a]
    model.add_constant(penalty*constant*constant)
    for i,a in items:model.add_linear(i,penalty*(a*a+2*constant*a))
    for oi,(i,a) in enumerate(items):
        for j,b in items[oi+1:]: model.add_quadratic(i,j,2*penalty*a*b)


def build_model(cfg=PureSelectionOrderConfig()):
    reads,edges,reward=load_chm13_graph(DATASET_DIR)
    rids=[r.rid for r in reads]; rank={r:i for i,r in enumerate(rids)}
    ep=sorted(reward,key=lambda e:(rank[e[0]],rank[e[1]]))
    n=len(rids); k=max(1,math.ceil(math.log2(max(2,n))))
    m=PureSelectionOrderModel(rids,ep,k)
    shift=max(reward.values(),default=0); raw={e:shift-reward[e] for e in ep}
    norm=max(max(raw.values(),default=1),1)
    incoming={r:[] for r in rids}; outgoing={r:[] for r in rids}
    for e in ep:
        u,v=e; x=m.edge_index[e]; incoming[v].append(e); outgoing[u].append(e)
        m.add_linear(x,cfg.edge_cost_scale*raw[e]/norm)

    # Degree/path-cover with virtual source/sink.
    for r in rids:
        add_square(m,{m.source_index[r]:-1,**{m.edge_index[e]:-1 for e in incoming[r]}},1,cfg.degree_penalty)
        add_square(m,{m.sink_index[r]:-1,**{m.edge_index[e]:-1 for e in outgoing[r]}},1,cfg.degree_penalty)
    # Exactly one path component through the virtual node.
    add_square(m,{m.source_index[r]:-1 for r in rids},1,cfg.void_penalty)
    add_square(m,{m.sink_index[r]:-1 for r in rids},1,cfg.void_penalty)

    # Exact x_e <-> P_e>0 gate.
    # z_k <= x_e.
    # m_e - x_e - slack_e = 0, slack in [0,K-1].
    for e in ep:
        x=m.edge_index[e]
        count_coeff={x:-1}
        for b in range(k):
            z=m.pos_index[(e,b)]
            m.add_linear(z,cfg.activation_penalty)
            m.add_quadratic(z,x,-cfg.activation_penalty)
            count_coeff[z]=1
        for b in range(m.slack_bits):
            count_coeff[m.slack_index[(e,b)]]=-(1<<b)
        add_square(m,count_coeff,0,cfg.activation_penalty)

    # Direct positions P_e = sum 2^k z_e,k.
    # Valid path convention: first real edge has position 2, last has position N.
    # For each read:
    #   O_v - I_v - 1 - s_v + (N+1)t_v = 0
    # Internal: P_out=P_in+1
    # Source: P_out=2
    # Sink: P_in=N
    for r in rids:
        coeff={m.source_index[r]:-1,m.sink_index[r]:float(n+1)}
        for e in incoming[r]:
            for b in range(k): coeff[m.pos_index[(e,b)]]=coeff.get(m.pos_index[(e,b)],0)-float(1<<b)
        for e in outgoing[r]:
            for b in range(k): coeff[m.pos_index[(e,b)]]=coeff.get(m.pos_index[(e,b)],0)+float(1<<b)
        add_square(m,coeff,-1,cfg.order_penalty)

    m.quadratic={ij:v for ij,v in m.quadratic.items() if abs(v)>1e-12}
    return m,reward,cfg


def encode_order(model:PureSelectionOrderModel,order:list[str]):
    sample=[0]*model.num_variables
    sample[model.source_index[order[0]]]=1; sample[model.sink_index[order[-1]]]=1
    for pos,e in enumerate(zip(order,order[1:]),start=2):
        x=model.edge_index[e]; sample[x]=1
        ones=0
        for b in range(model.k):
            if (pos>>b)&1:
                sample[model.pos_index[(e,b)]]=1; ones+=1
        slack=ones-1
        for b in range(model.slack_bits):
            if (slack>>b)&1: sample[model.slack_index[(e,b)]]=1
    return sample


def audit(sample,model,reward,cfg):
    selected=[e for e in model.edge_pairs if sample[model.edge_index[e]]]
    sources=[r for r in model.read_ids if sample[model.source_index[r]]]
    sinks=[r for r in model.read_ids if sample[model.sink_index[r]]]
    indeg={r:0 for r in model.read_ids}; outdeg={r:0 for r in model.read_ids}; nxt={}
    for u,v in selected:
        indeg[v]+=1; outdeg[u]+=1
        if u not in nxt:nxt[u]=v
    dsq=sum((1-int(r in sources)-indeg[r])**2+(1-int(r in sinks)-outdeg[r])**2 for r in model.read_ids)
    vsq=(1-len(sources))**2+(1-len(sinks))**2
    act=0; ordsq=0
    for e in model.edge_pairs:
        x=int(sample[model.edge_index[e]])
        z=[int(sample[model.pos_index[(e,b)]]) for b in range(model.k)]
        slack=sum((1<<b)*int(sample[model.slack_index[(e,b)]]) for b in range(model.slack_bits))
        act+=sum(int(zz and not x) for zz in z)
        act+=(sum(z)-x-slack)**2
    for r in model.read_ids:
        I=sum(sum((1<<b)*int(sample[model.pos_index[(e,b)]]) for b in range(model.k)) for e in [ee for ee in model.edge_pairs if ee[1]==r])
        O=sum(sum((1<<b)*int(sample[model.pos_index[(e,b)]]) for b in range(model.k)) for e in [ee for ee in model.edge_pairs if ee[0]==r])
        rr=O-I-1-int(r in sources)+(len(model.read_ids)+1)*int(r in sinks)
        ordsq+=rr*rr
    structural=False; order=[]
    if dsq==0 and vsq==0 and len(sources)==1 and len(sinks)==1:
        seen=set(); cur=sources[0]
        while cur not in seen:
            seen.add(cur); order.append(cur)
            if cur==sinks[0]: break
            if cur not in nxt: break
            cur=nxt[cur]
        structural=len(order)==len(model.read_ids) and order[-1]==sinks[0]
    valid=structural and act==0 and ordsq==0
    score=int(sum(reward[e] for e in zip(order,order[1:]))) if valid else None
    return {"energy":float(model.energy(sample)),"valid_path":valid,"path_score":score,
            "selected_edges":len(selected),"sources":len(sources),"sinks":len(sinks),
            "degree_residual_sq":int(dsq),"void_residual_sq":int(vsq),
            "activation_residual":int(act),"order_residual_sq":int(ordsq),
            "hamming_weight":int(sum(sample))}


if __name__=="__main__":
    m,reward,cfg=build_model()
    ref=json.loads((DATASET_DIR/"reference_path.json").read_text())["normalized_nodes"]
    s=encode_order(m,ref)
    out={"variables":m.num_variables,"quadratic_terms":len(m.quadratic),
         "max_abs_linear":max(abs(x) for x in m.linear),
         "max_abs_quadratic":max(abs(x) for x in m.quadratic.values()),
         "audit":audit(s,m,reward,cfg)}
    print(json.dumps(out,indent=2))

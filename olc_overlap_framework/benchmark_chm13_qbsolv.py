"""QBSolv comparison on CHM13 QUBOs.

Historical/decomposition baseline:
- dwave-qbsolv 0.3.4
- internal classical tabu subproblem solver
- energy-impact decomposition
- several subproblem sizes

Standalone by design because archived QBSolv pins dimod<0.11 and is
incompatible with the current dwave-samplers package used by other benchmarks.
"""
from __future__ import annotations
import json,time
from pathlib import Path
from dwave_qbsolv import QBSolv, ENERGY_IMPACT

from demo_chm13_edge_ordered_path_qubo import DATASET_DIR,load_chm13_graph
from experimental_chm13_vertex_carry_qubo import (
    BoundedVertexOrderConfig,BoundedVertexOrderModel,
    build_model as build_bounded_model,encode_order as encode_bounded_order,
)
from olc_pipeline.layout_solver import (
    EdgeOrderedPathHamiltonianConfig,EdgeOrderedPathQUBOHamiltonian,
    QUBOLayoutSolver,qubo_sample_for_order,
)

SEED=20260925
OUT=Path("debug/qubo/chm13_qbsolv_comparison_20260925.json")
CERT=json.loads((DATASET_DIR/"weighted_certificate.json").read_text())
REFERENCE=json.loads((DATASET_DIR/"reference_path.json").read_text())
REF_ORDER=list(REFERENCE["normalized_nodes"])

def build_current():
    reads,edges,reward=load_chm13_graph(DATASET_DIR)
    h=EdgeOrderedPathQUBOHamiltonian(EdgeOrderedPathHamiltonianConfig(
        degree_penalty=None,order_penalty=None,activation_penalty=None,
        edge_cost_scale=1.0,cost_mode="shifted_reward",score_mode="dp",
        normalize_costs=True,include_source_length=False))
    m=h.build(reads,edges)
    rs=qubo_sample_for_order(m,REF_ORDER)
    return {"name":"current","model":m,"reward":reward,
            "reference_sample":rs,"reference_energy":float(m.energy(rs))}

def build_bounded(name,d,v,s,p):
    cfg=BoundedVertexOrderConfig(degree_penalty=d,void_degree_penalty=v,
        successor_penalty=s,product_penalty=p,edge_cost_scale=1.0)
    m,reward,_,_=build_bounded_model(cfg)
    rs=encode_bounded_order(m,REF_ORDER)
    return {"name":name,"model":m,"reward":reward,"config":cfg,
            "reference_sample":rs,"reference_energy":float(m.energy(rs))}

def current_eval(problem,sample):
    m=problem["model"]
    order,meta=QUBOLayoutSolver._decode_edge_ordered_path(m,sample)
    energy=float(m.energy(sample))
    score=None
    if meta["valid_edge_path"]:
        score=int(sum(problem["reward"][pair] for pair in zip(order,order[1:])))
    return {
      "energy":energy,"energy_above_reference":energy-problem["reference_energy"],
      "valid_path":bool(meta["valid_edge_path"]),
      "path_score":score,
      "selected_edges":int(meta["selected_edge_count"]),
      "sources":int(meta["selected_source_count"]),
      "sinks":int(meta["selected_sink_count"]),
      "read_in_violations":int(meta["read_in_constraint_violations"]),
      "read_out_violations":int(meta["read_out_constraint_violations"]),
      "activation_violations":int(meta["activation_violations"]),
      "order_residual_l1":int(meta["order_constraint_violations"]),
      "hamming_from_reference":sum(a!=b for a,b in zip(sample,problem["reference_sample"])),
    }

def bounded_eval(problem,sample):
    m:BoundedVertexOrderModel=problem["model"]
    selected=[e for e in m.edge_pairs if sample[m.edge_index[e]]]
    sources=[r for r in m.read_ids if sample[m.source_index[r]]]
    sinks=[r for r in m.read_ids if sample[m.sink_index[r]]]
    indeg={r:0 for r in m.read_ids}; outdeg={r:0 for r in m.read_ids}; nxt={}
    for u,v in selected:
        outdeg[u]+=1; indeg[v]+=1
        if u not in nxt:nxt[u]=v
    degree_sq=sum((1-int(r in sources)-indeg[r])**2+
                  (1-int(r in sinks)-outdeg[r])**2 for r in m.read_ids)
    void_sq=(1-len(sources))**2+(1-len(sinks))**2
    prod=0; succ=0
    for e in m.edge_pairs:
      u,v=e; x=int(sample[m.edge_index[e]])
      for k in range(m.position_bits):
        pu=int(sample[m.position_index[(u,k)]]); pv=int(sample[m.position_index[(v,k)]])
        a=int(sample[m.left_copy_index[(e,k)]]); b=int(sample[m.right_copy_index[(e,k)]])
        prod+=int(a!=x*pu)+int(b!=x*pv)
        cin=x if k==0 else int(sample[m.carry_index[(e,k)]])
        cout=0 if k==m.position_bits-1 else int(sample[m.carry_index[(e,k+1)]])
        rr=a+cin-b-2*cout; succ+=rr*rr
    structural=False; order=[]
    if degree_sq==0 and void_sq==0 and len(sources)==1 and len(sinks)==1:
      seen=set(); cur=sources[0]
      while cur not in seen:
        seen.add(cur); order.append(cur)
        if cur==sinks[0]:break
        if cur not in nxt:break
        cur=nxt[cur]
      structural=len(order)==len(m.read_ids) and order[-1]==sinks[0]
    valid=structural and prod==0 and succ==0
    score=int(sum(problem["reward"][e] for e in zip(order,order[1:]))) if valid else None
    pos=[]
    for r in m.read_ids:
      val=sum(int(sample[m.position_index[(r,k)]])<<k for k in range(m.position_bits))
      pos.append(val)
    energy=float(m.energy(sample))
    return {
      "energy":energy,"energy_above_reference":energy-problem["reference_energy"],
      "valid_path":valid,"path_score":score,
      "selected_edges":len(selected),"sources":len(sources),"sinks":len(sinks),
      "read_degree_residual_sq":int(degree_sq),"void_residual_sq":int(void_sq),
      "product_violations":int(prod),"successor_residual_sq":int(succ),
      "distinct_positions":len(set(pos)),
      "hamming_from_reference":sum(a!=b for a,b in zip(sample,problem["reference_sample"])),
    }

CASES=[
 ("current",build_current,current_eval),
 ("bounded_weak16",lambda:build_bounded("bounded_weak16",16,32,0.25,0.5),bounded_eval),
 ("bounded_weak24",lambda:build_bounded("bounded_weak24",24,48,0.125,0.25),bounded_eval),
]
SUBSIZES=[47,100,200]

def run_case(name,builder,evaluator,subsize,seed):
    problem=builder(); bqm=problem["model"].to_dimod_bqm()
    t0=time.perf_counter()
    ss=QBSolv().sample(bqm,num_repeats=20,seed=seed,algorithm=ENERGY_IMPACT,
        verbosity=-1,timeout=20,solver_limit=subsize,solver="tabu")
    sec=time.perf_counter()-t0
    datum=next(ss.data(fields=["sample","energy"],sorted_by="energy"))
    sample=[int(datum.sample.get(i,0)) for i in range(problem["model"].num_variables)]
    return {"model":name,"subproblem_size":subsize,"seconds":sec,
      "variables":problem["model"].num_variables,
      "quadratic_terms":len(problem["model"].quadratic),**evaluator(problem,sample)}

def main():
    rows=[]
    for ci,(name,builder,evaluator) in enumerate(CASES):
      for si,size in enumerate(SUBSIZES):
        print("RUN",name,"S",size,flush=True)
        row=run_case(name,builder,evaluator,size,SEED+100*ci+si)
        rows.append(row); print(json.dumps(row),flush=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"solver":"dwave-qbsolv 0.3.4","rows":rows},indent=2)+"\n")
    print("\nSUMMARY"); print(json.dumps(rows,indent=2),flush=True)
if __name__=="__main__":main()

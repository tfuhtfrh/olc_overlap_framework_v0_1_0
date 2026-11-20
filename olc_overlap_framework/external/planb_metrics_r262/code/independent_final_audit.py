"""Independent post-run audit for a Plan B final chain.

Design contract:
- evaluation only; never imported by solver/search code
- accepts authoritative graph + run/output directory only
- accepts no reference assembly
- re-computes validity independently of the solver's validate_original()
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter
from graph_common import load_graph, physical_and_orientation, build_mate_map
from runtime_utils import sha256_file

SUCCESS_STATUSES={"SQA_VALIDATED_GLOBAL_CHAIN","PASS_EXACT_DAG","SQA_VALIDATED","PASS","HEURISTIC_VALIDATED"}

def _read_chain(run_dir:Path, chain_file:Path|None=None):
    if chain_file is not None:
        p=Path(chain_file)
    else:
        candidates=[run_dir/"FINAL_CHAIN.txt",run_dir/"final_path.txt"]
        p=next((x for x in candidates if x.exists()),None)
        if p is None:
            return None,()
    rows=tuple(x.strip() for x in p.read_text().splitlines() if x.strip())
    return p,rows

def _read_status(run_dir:Path):
    for name in ("FINAL_STATUS.json","final_summary.json"):
        p=run_dir/name
        if p.exists():
            try:return p,json.loads(p.read_text())
            except Exception as exc:return p,{"_parse_error":str(exc)}
    return None,None

def audit_final_chain(graph_path,run_dir,*,chain_file=None):
    graph_path=Path(graph_path).expanduser().resolve()
    run_dir=Path(run_dir).expanduser().resolve()
    g=load_graph(graph_path)
    mate=build_mate_map(g)

    chain_path,path=_read_chain(run_dir,Path(chain_file) if chain_file else None)
    status_path,status=_read_status(run_dir)

    missing_nodes=[v for v in path if v not in g]
    edge_failures=[]
    edge_evidence=[]
    if not missing_nodes:
        for i,(u,v) in enumerate(zip(path,path[1:])):
            if not g.has_edge(u,v):
                edge_failures.append({"index":i,"u":u,"v":v})
                continue
            ids=[]
            if g.is_multigraph():
                for k,d in g[u][v].items():
                    ids.append(str(d.get("edge_id",k)))
            else:
                ids.append(str(g[u][v].get("edge_id","")))
            edge_evidence.append({"index":i,"u":u,"v":v,"edge_ids":sorted(ids)})

    physical=[]
    orientations=[]
    mapping_errors=[]
    if not missing_nodes:
        for i,v in enumerate(path):
            try:
                r,o=physical_and_orientation(g.nodes[v],v)
                physical.append(r); orientations.append(o)
            except Exception as exc:
                mapping_errors.append({"index":i,"node":v,"error":str(exc)})

    counts=Counter(physical)
    repeated_physical=sorted((r,c) for r,c in counts.items() if c>1)
    mate_pair_collisions=[]
    selected=set(path)
    for v in path:
        m=mate.get(v)
        if m in selected and str(v)<str(m):
            mate_pair_collisions.append([str(v),str(m)])

    # A final expanded chain may contain only authoritative graph nodes.
    macro_or_non_authoritative_nodes=list(missing_nodes)

    status_name=(status or {}).get("status") if isinstance(status,dict) else None
    solver_claims_success=status_name in SUCCESS_STATUSES
    status_path_nodes=(status or {}).get("path_nodes") if isinstance(status,dict) else None
    real_sqa_calls=(status or {}).get("real_sqa_calls") if isinstance(status,dict) else None
    status_validation=(status or {}).get("validation") if isinstance(status,dict) else None

    status_consistency={
        "status_present":status is not None,
        "status_parse_ok":not (isinstance(status,dict) and "_parse_error" in status),
        "solver_status":status_name,
        "solver_claims_success":solver_claims_success,
        "path_nodes_match":(status_path_nodes is None or int(status_path_nodes)==len(path)),
        "reported_validation_agrees":None,
        "sqa_success_has_real_sqa_call":None,
    }
    if isinstance(status_validation,dict):
        status_consistency["reported_validation_agrees"]=(
            bool(status_validation.get("physical_unique"))==(not repeated_physical and not mapping_errors)
            and bool(status_validation.get("authoritative_edges_valid"))==(not edge_failures and not missing_nodes)
        )
    if status_name in {"SQA_VALIDATED_GLOBAL_CHAIN","SQA_VALIDATED"}:
        status_consistency["sqa_success_has_real_sqa_call"]=(real_sqa_calls is None or int(real_sqa_calls)>0)

    checks={
        "chain_file_present":chain_path is not None,
        "chain_nonempty":bool(path),
        "all_nodes_authoritative":not missing_nodes,
        "all_consecutive_edges_authoritative":not edge_failures,
        "physical_orientation_mapping_valid":not mapping_errors,
        "physical_reads_unique":not repeated_physical,
        "no_mate_pair_collision":not mate_pair_collisions,
        "fully_expanded_no_macro_nodes":not macro_or_non_authoritative_nodes,
        "status_parse_ok":status_consistency["status_parse_ok"],
        "status_path_nodes_match":status_consistency["path_nodes_match"],
        "reported_validation_agrees":status_consistency["reported_validation_agrees"] is not False,
        "sqa_success_has_real_sqa_call":status_consistency["sqa_success_has_real_sqa_call"] is not False,
    }
    passed=all(checks.values())

    return {
        "schema":"PLANB_INDEPENDENT_FINAL_AUDIT_V1",
        "status":"PASS" if passed else "FAIL",
        "reference_used":False,
        "reference_input_supported":False,
        "graph":{"path":str(graph_path),"sha256":sha256_file(graph_path),
                 "nodes":g.number_of_nodes(),"edges":g.number_of_edges()},
        "run_dir":str(run_dir),
        "chain_file":str(chain_path) if chain_path else None,
        "chain_sha256":sha256_file(chain_path) if chain_path and chain_path.exists() else None,
        "path_nodes":len(path),
        "physical_reads":len(set(physical)),
        "orientations":{"plus":orientations.count("+"),"minus":orientations.count("-")},
        "checks":checks,
        "missing_nodes":missing_nodes[:50],
        "bad_edges":edge_failures[:50],
        "repeated_physical_reads":[{"physical_read":r,"count":c} for r,c in repeated_physical[:50]],
        "mate_pair_collisions":mate_pair_collisions[:50],
        "mapping_errors":mapping_errors[:50],
        "status_file":str(status_path) if status_path else None,
        "status_consistency":status_consistency,
        "edge_evidence_sample":edge_evidence[:20],
        "interpretation":"independent authoritative-graph feasibility/provenance audit; not an optimality proof",
    }

def write_audit(graph_path,run_dir,*,chain_file=None,out=None):
    run_dir=Path(run_dir)
    result=audit_final_chain(graph_path,run_dir,chain_file=chain_file)
    target=Path(out) if out else run_dir/"INDEPENDENT_AUDIT.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,default=str)+"\n")
    return result,target

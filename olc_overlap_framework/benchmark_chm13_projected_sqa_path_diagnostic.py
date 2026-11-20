"""Replay the best extended projected-SQA path and compare it to reference order.

Also build a simple overlap-spliced contig for the SQA path and the certified
reference path using the same retained overlap spans, then map both contigs back
to the packaged CHM13 reference with mappy/minimap2.  This is an evaluation
only; reference coordinates are not used in the SQA search.
"""
from __future__ import annotations
import csv,json,random,statistics
from pathlib import Path
import mappy as mp

from benchmark_chm13_projected_edge_sqa import sample_edges_sqa
from benchmark_chm13_projected_edge_hybrid import (
    load_problem,project_rank,graph_metrics,build_bqm_count,
)
from demo_chm13_edge_ordered_path_qubo import DATASET_DIR,read_fasta

BASE_SEED=20260926
RUN=7
STOP_ITER=14
SCHEDULE=[0.0,0.25,0.5,1.0,2.0,4.0]
EXPECTED_SCORE=1327035
OUT=Path("debug/qubo/chm13_projected_sqa_best_path_diagnostic_20260926.json")


def derive_order(rids,selected):
    indeg={r:0 for r in rids};outdeg={r:0 for r in rids};nxt={}
    for u,v in selected:
        indeg[v]+=1;outdeg[u]+=1;nxt[u]=v
    sources=[r for r in rids if indeg[r]==0]
    if len(sources)!=1:
        raise RuntimeError(f"expected one source, got {len(sources)}")
    order=[];seen=set();cur=sources[0]
    while cur not in seen:
        seen.add(cur);order.append(cur)
        if cur not in nxt:break
        cur=nxt[cur]
    if len(order)!=len(rids):
        raise RuntimeError(f"path visits {len(order)} / {len(rids)}")
    return order


def overlap_map():
    reads,edges,reward=__import__("demo_chm13_edge_ordered_path_qubo").load_chm13_graph(DATASET_DIR)
    return {(e.left_id,e.right_id):int(e.overlap_len) for e in edges}


def build_spliced_contig(order,seqs,ov):
    chunks=[seqs[order[0]]]
    shifts=[0]
    pos=0
    for u,v in zip(order,order[1:]):
        olen=ov[(u,v)]
        shift=len(seqs[u])-olen
        pos+=shift
        shifts.append(pos)
        chunks.append(seqs[v][olen:])
    return "".join(chunks),shifts


def interval_union_length(intervals):
    if not intervals:return 0
    a=sorted((int(s),int(e)) for s,e in intervals if e>s)
    total=0;cs,ce=a[0]
    for s,e in a[1:]:
        if s<=ce:ce=max(ce,e)
        else:
            total+=ce-cs;cs,ce=s,e
    return total+ce-cs


def map_contig(seq,ref_path):
    aln=mp.Aligner(str(ref_path),preset="asm5",best_n=20)
    hits=list(aln.map(seq))
    hit_rows=[]
    for h in hits:
        hit_rows.append({
            "ctg":h.ctg,"q_st":int(h.q_st),"q_en":int(h.q_en),
            "r_st":int(h.r_st),"r_en":int(h.r_en),
            "strand":int(h.strand),"mapq":int(h.mapq),
            "mlen":int(h.mlen),"blen":int(h.blen),
            "identity":float(h.mlen/h.blen) if h.blen else 0.0,
            "is_primary":bool(getattr(h,"is_primary",False)),
        })
    usable=[x for x in hit_rows if x["mapq"]>0]
    qcov=interval_union_length((x["q_st"],x["q_en"]) for x in usable)/len(seq)
    refseq=read_fasta(ref_path)
    ref_len=len(next(iter(refseq.values())))
    rcov=interval_union_length((x["r_st"],x["r_en"]) for x in usable)/ref_len
    total_b=sum(x["blen"] for x in usable)
    weighted_identity=sum(x["mlen"] for x in usable)/total_b if total_b else 0.0
    return {
        "contig_length":len(seq),
        "hit_count":len(hit_rows),
        "usable_hit_count":len(usable),
        "query_coverage_union":qcov,
        "reference_coverage_union":rcov,
        "weighted_identity_over_hits":weighted_identity,
        "hits":sorted(hit_rows,key=lambda x:(-x["mapq"],-x["blen"]))[:20],
    }


def true_start_map():
    d={}
    with (DATASET_DIR/"nodes.tsv").open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f,delimiter="\t"):
            d[row["read_id"]]=int(row["start"])
    return d


def positional_geometry(order,pred_starts):
    truth=true_start_map()
    # Best global translation by median true-predicted offset.
    offsets=[truth[r]-pred_starts[i] for i,r in enumerate(order)]
    translation=int(round(statistics.median(offsets)))
    errors=[pred_starts[i]+translation-truth[r] for i,r in enumerate(order)]
    abs_errors=[abs(x) for x in errors]
    return {
      "best_translation":translation,
      "start_error_median_abs_bp":float(statistics.median(abs_errors)),
      "start_error_mean_abs_bp":float(statistics.mean(abs_errors)),
      "start_error_max_abs_bp":max(abs_errors),
      "start_error_max_read":order[abs_errors.index(max(abs_errors))],
    }


def main():
    rids,ep,reward,cost,incoming,outgoing=load_problem()
    ref=json.loads((DATASET_DIR/"reference_path.json").read_text())["normalized_nodes"]

    run=RUN
    rng=random.Random(BASE_SEED+10000*run)
    ranks={r:i for i,r in enumerate(rng.sample(rids,len(rids)))}
    selected=set()
    trace=[]
    for it in range(STOP_ITER+1):
        op=SCHEDULE[it%len(SCHEDULE)]
        bqm=build_bqm_count(
            ep,cost,incoming,outgoing,ranks,len(rids),
            degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=op
        )
        selected,sec,energy=sample_edges_sqa(
            bqm,selected if selected else None,BASE_SEED+10000*run+it
        )
        ranks=project_rank(rids,selected)
        metrics=graph_metrics(rids,selected,ranks,reward)
        trace.append({"iteration":it,"order_penalty":op,"energy":energy,**metrics})

    order=derive_order(rids,selected)
    score=int(sum(reward[e] for e in zip(order,order[1:])))
    if score!=EXPECTED_SCORE:
        raise RuntimeError(f"replay score {score} != expected {EXPECTED_SCORE}")

    pref={r:i for i,r in enumerate(ref)}
    psol={r:i for i,r in enumerate(order)}
    disp={r:psol[r]-pref[r] for r in rids}
    absdisp={r:abs(v) for r,v in disp.items()}
    inversions=0
    arr=[pref[r] for r in order]
    for i in range(len(arr)):
        inversions+=sum(arr[i]>arr[j] for j in range(i+1,len(arr)))

    ov=overlap_map();seqs=read_fasta(DATASET_DIR/"reads.normalized.fasta")
    sqa_contig,sqa_starts=build_spliced_contig(order,seqs,ov)
    ref_contig,ref_starts=build_spliced_contig(ref,seqs,ov)

    result={
      "run":RUN,"iteration":STOP_ITER,"score":score,"certified_optimum_score":1343093,
      "score_gap":1343093-score,
      "order_metrics":{
        "max_absolute_displacement":max(absdisp.values()),
        "max_displacement_reads":[
            {"read":r,"reference_index":pref[r],"sqa_index":psol[r],"displacement":disp[r]}
            for r in sorted(rids,key=lambda x:(-absdisp[x],x))[:20]
        ],
        "exact_position_reads":sum(d==0 for d in disp.values()),
        "mean_absolute_displacement":float(statistics.mean(absdisp.values())),
        "median_absolute_displacement":float(statistics.median(absdisp.values())),
        "inversion_count":inversions,
        "max_possible_inversions":len(rids)*(len(rids)-1)//2,
      },
      "sqa_order":order,
      "reference_order":ref,
      "sqa_layout_geometry":positional_geometry(order,sqa_starts),
      "reference_layout_geometry":positional_geometry(ref,ref_starts),
      "sqa_spliced_contig_map":map_contig(sqa_contig,DATASET_DIR/"reference.fasta"),
      "reference_spliced_contig_map":map_contig(ref_contig,DATASET_DIR/"reference.fasta"),
      "trace":trace,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({
      "score":score,
      "order_metrics":result["order_metrics"],
      "sqa_layout_geometry":result["sqa_layout_geometry"],
      "reference_layout_geometry":result["reference_layout_geometry"],
      "sqa_spliced_contig_map":result["sqa_spliced_contig_map"],
      "reference_spliced_contig_map":result["reference_spliced_contig_map"],
    },indent=2))


if __name__=="__main__":
    main()

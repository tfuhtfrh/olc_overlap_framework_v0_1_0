"""Reference evaluation of saved SQA layout and certified optimum; no search."""
import argparse, csv, gzip, hashlib, json, re, subprocess
from pathlib import Path
import edlib

def fasta(p):
    seqs={}; name=None
    for line in Path(p).read_text().splitlines():
        if line.startswith(">"):
            name=line[1:].split()[0]; seqs[name]=""
        elif line.strip(): seqs[name]+=line.strip()
    return seqs

def union_len(intervals):
    out=0; end=-1
    for a,b in sorted(intervals):
        if a>=end: out+=b-a
        elif b>end: out+=b-end
        end=max(end,b)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",type=Path,required=True)
    ap.add_argument("--trials",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    seq=fasta(a.dataset/"reads.normalized.fasta"); refseq=fasta(a.dataset/"reference.fasta")
    assert len(seq)==144 and len(refseq)==1
    ref=next(iter(refseq.values()))
    nodes=list(csv.DictReader((a.dataset/"nodes.tsv").open(),delimiter="\t"))
    bynode={r["node"]:r for r in nodes}; byid={r["read_id"]:r for r in nodes}
    edge_rows=list(csv.DictReader((a.dataset/"edges.tsv").open(),delimiter="\t"))
    edges={(bynode[r["source"]]["read_id"],bynode[r["target"]]["read_id"]):r for r in edge_rows}
    candidates=[]
    for p in a.trials.rglob("best_path.json"):
        d=json.loads(p.read_text())
        if d["score"]==1320804: candidates.append((str(p),d))
    assert candidates, "Saved score 1320804 not found"
    origin,best=sorted(candidates)[0]
    gold=json.loads((a.dataset/"reference_path.json").read_text())["normalized_nodes"]
    orders={"sqa_1320804":best["order"],"optimum_1343093":gold}
    paf_needed={int(edges[(u,v)]["paf_line"]) for order in orders.values() for u,v in zip(order,order[1:])}
    evidence={}
    with gzip.open(a.dataset/"overlaps.exact.paf.gz","rt") as f:
        for lineno,line in enumerate(f,1):
            if lineno in paf_needed: evidence[lineno]=line.rstrip().split("\t")
    assert set(evidence)==paf_needed
    goldedges=set(zip(gold,gold[1:]))
    report={"provenance":{"source_trial":origin,"source_run":39141763797,
            "reference_length":len(ref),"read_count":len(seq),
            "reference_sha256":hashlib.sha256(ref.encode()).hexdigest(),
            "method":"overlap-evidence end splice; no consensus or reference-guided polishing",
            "minimap2_version":subprocess.check_output(["minimap2","--version"],text=True).strip(),
            "edlib_version":__import__("importlib.metadata",fromlist=["version"]).version("edlib")},"results":{}}
    for label,order in orders.items():
        assert len(order)==len(set(order))==144 and set(order)==set(seq)
        selected=list(zip(order,order[1:])); score=sum(int(edges[p]["weight"]) for p in selected)
        assert score==int(label.rsplit("_",1)[1])
        assembly=seq[order[0]]; joins=[]
        for u,v in selected:
            row=edges[(u,v)]; p=evidence[int(row["paf_line"])]
            q,ql,qs,qe,strand,t,tl,ts,te=p[:9]
            ql,qs,qe,tl,ts,te=map(int,(ql,qs,qe,tl,ts,te))
            assert {q,t}=={u,v}
            assert (byid[q]["selected_sign"]==byid[t]["selected_sign"])==(strand=="+")
            intervals={}
            for rid,length,start,end in ((q,ql,qs,qe),(t,tl,ts,te)):
                assert len(seq[rid])==length
                intervals[rid]=(start,end) if byid[rid]["selected_sign"]=="+" else (length-end,length-start)
            us,ue=intervals[u]; vs,ve=intervals[v]
            assert us>0 and ve<len(seq[v])
            assert abs(int(p[9])-int(row["matching_bases"]))==0
            tail=len(seq[u])-ue
            assert tail<max(101,int(0.021*int(row["overlap_span"])))
            assembly=(assembly[:-tail] if tail else assembly)+seq[v][ve:]
            joins.append({"source":u,"target":v,"source_end":ue,"target_end":ve,"trimmed_source_tail":tail,
                          "matches":int(row["matching_bases"]),"errors":int(row["errors"]),
                          "reference_start_delta":int(byid[v]["start"])-int(byid[u]["start"]),
                          "reference_adjacency":(u,v) in goldedges})
        outfa=a.out/(label+".fasta"); outfa.write_text(">"+label+"\n"+"\n".join(assembly[i:i+80] for i in range(0,len(assembly),80))+"\n")
        paffile=a.out/(label+".paf")
        cmd=["minimap2","-x","asm5","-c","--eqx","--cs=long","-N","50","-t","2",str(a.dataset/"reference.fasta"),str(outfa)]
        with paffile.open("w") as f, (a.out/(label+".minimap2.log")).open("w") as log:
            subprocess.run(cmd,stdout=f,stderr=log,check=True)
        records=[]
        for line in paffile.read_text().splitlines():
            p=line.split("\t"); tags={x[:2]:x[5:] for x in p[12:]}
            if tags.get("tp")!="P": continue
            records.append({"qstart":int(p[2]),"qend":int(p[3]),"strand":p[4],"rstart":int(p[7]),"rend":int(p[8]),
                            "matches":int(p[9]),"block":int(p[10]),"mapq":int(p[11]),"NM":int(tags.get("NM","0"))})
        distance=edlib.align(assembly,ref,mode="NW",task="distance")["editDistance"]
        common=sum(j["reference_adjacency"] for j in joins)
        res={"score":score,"assembly_length":len(assembly),"length_delta":len(assembly)-len(ref),
             "global_edit_distance":distance,
             "global_edit_accuracy":1-distance/max(len(assembly),len(ref)),
             "primary_alignment_identity":sum(r["matches"] for r in records)/sum(r["block"] for r in records) if records else None,
             "reference_coverage":union_len([(r["rstart"],r["rend"]) for r in records])/len(ref),
             "query_coverage":union_len([(r["qstart"],r["qend"]) for r in records])/len(assembly),
             "primary_alignment_count":len(records),"primary_alignments":records,
             "reference_adjacencies":common,"total_adjacencies":143,"reference_adjacency_fraction":common/143,
             "backward_reference_jumps":sum(j["reference_start_delta"]<0 for j in joins),
             "changed_adjacencies":[j for j in joins if not j["reference_adjacency"]]}
        report["results"][label]=res
        (a.out/(label+".joins.json")).write_text(json.dumps(joins,indent=2))
        print("RESULT",json.dumps({"label":label,**res}),flush=True)
    x=report["results"]["sqa_1320804"]; y=report["results"]["optimum_1343093"]
    report["comparison"]={"extra_global_edit_distance":x["global_edit_distance"]-y["global_edit_distance"],
      "global_edit_accuracy_difference_percentage_points":100*(x["global_edit_accuracy"]-y["global_edit_accuracy"]),
      "reference_coverage_difference_percentage_points":100*(x["reference_coverage"]-y["reference_coverage"]),
      "score_gap":1343093-1320804}
    (a.out/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    print("COMPARISON",json.dumps(report["comparison"]),flush=True)
if __name__=="__main__": main()

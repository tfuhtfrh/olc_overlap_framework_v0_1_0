"""Post-hoc reference evaluation gate.

Reference use is evaluation-only.  Every public entry first verifies a PASS
from PLANB_INDEPENDENT_FINAL_AUDIT_V1 before opening any reference mapping or
reference FASTA.

Existing ordering/orientation logic is reused from upstream/hinge_layout_audit.py.
"""
from __future__ import annotations
import importlib.util, json, shutil, subprocess, sys, tarfile, gzip
from pathlib import Path
from collections import Counter
from runtime_utils import sha256_file

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/"upstream"/"hinge_layout_audit.py"

def _load_hinge_audit():
    spec=importlib.util.spec_from_file_location("planb_hinge_layout_audit",UPSTREAM)
    mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Python 3.13 dataclasses resolves cls.__module__ through sys.modules.
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

def require_independent_audit_pass(audit_path):
    p=Path(audit_path).expanduser().resolve()
    z=json.loads(p.read_text())
    if z.get("schema")!="PLANB_INDEPENDENT_FINAL_AUDIT_V1":
        raise RuntimeError("REFERENCE_LOCKED: independent-audit schema mismatch")
    if z.get("status")!="PASS":
        raise RuntimeError("REFERENCE_LOCKED: independent audit is not PASS")
    if z.get("reference_used") is not False:
        raise RuntimeError("REFERENCE_LOCKED: audit itself was not reference-free")
    checks=z.get("checks",{})
    if not checks or not all(bool(v) for v in checks.values()):
        raise RuntimeError("REFERENCE_LOCKED: one or more independent-audit checks are not PASS")
    return z

def load_oriented_chain(run_dir):
    run_dir=Path(run_dir).expanduser().resolve()
    p=run_dir/"FINAL_CHAIN.txt"
    if not p.exists():
        p=run_dir/"final_path.txt"
    if not p.exists():
        raise FileNotFoundError("final chain not found")
    rows=[]
    for raw in p.read_text().splitlines():
        node=raw.strip()
        if not node: continue
        if node[-1:] not in {"+","-"}:
            raise ValueError(f"final node lacks +/- orientation: {node}")
        rows.append((node[:-1],0 if node[-1]=="+" else 1,node))
    if not rows:
        raise ValueError("final chain is empty")
    return p,tuple(rows)

def _fasta_records(path):
    name=None; chunks=[]
    with Path(path).open() as f:
        for line in f:
            line=line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    yield name,"".join(chunks)
                name=line[1:].split()[0]; chunks=[]
            else:
                chunks.append(line.strip())
        if name is not None:
            yield name,"".join(chunks)

def extract_chain_reads(run_dir,source_reads_fasta,out_fasta):
    chain_path,rows=load_oriented_chain(run_dir)
    wanted={physical for physical,_,_ in rows}
    seqs={}
    for name,seq in _fasta_records(source_reads_fasta):
        if name in wanted:
            seqs[name]=seq
    missing=sorted(wanted-set(seqs))
    if missing:
        raise RuntimeError(f"missing {len(missing)} chain reads from FASTA; first={missing[:5]}")
    out=Path(out_fasta); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w") as f:
        for physical,orient,node in rows:
            # Keep the original read sequence. Orientation is evaluated separately
            # against the PAF strand using the existing HINGE audit convention.
            f.write(f">{physical}\n{seqs[physical]}\n")
    return {
        "chain_file":str(chain_path),
        "chain_nodes":len(rows),
        "physical_reads":len(wanted),
        "extracted_records":len(rows),
        "missing_reads":missing,
        "out_fasta":str(out),
        "out_sha256":sha256_file(out),
    }

def evaluate_chain_paf(run_dir,evaluation_paf,*,overlap_diagnostic=None):
    """Evaluate chain order/orientation from a precomputed mapping PAF.

    The PAF may target a reference assembly or, for plumbing tests, another
    assembly.  This function itself does not decide scientific truth status.
    """
    h=_load_hinge_audit()
    chain_path,rows=load_oriented_chain(run_dir)
    best=h.load_reference_best(Path(evaluation_paf))
    edge_counts=Counter()
    edge_rows=[]
    mapped=0
    mapq60=0
    intervals=[]
    targets=Counter()
    for physical,_,_ in rows:
        a=best.get(physical)
        if a is not None:
            mapped+=1; targets[a["target"]]+=1
            intervals.append((a["target"],a["start"],a["end"]))
            if a["rank"][0]==60: mapq60+=1
    for i,((a,ao,_),(b,bo,_)) in enumerate(zip(rows,rows[1:])):
        verdict=h.evaluate_edge(a,b,ao,bo,best)
        key="consistent" if verdict is True else "inconsistent" if verdict is False else "unmapped_or_cross_target"
        edge_counts[key]+=1
        edge_rows.append({"index":i,"source":a,"source_orientation":ao,
                          "target":b,"target_orientation":bo,"verdict":key})

    covered=0; fragments=0
    by_target={}
    for target,s,e in intervals:
        by_target.setdefault(target,[]).append((s,e))
    for its in by_target.values():
        c,n=h.interval_union_length(its); covered+=c; fragments+=n

    denom=edge_counts["consistent"]+edge_counts["inconsistent"]
    result={
        "schema":"PLANB_POSTHOC_CHAIN_PAF_EVAL_V1",
        "chain_file":str(chain_path),
        "evaluation_paf":str(Path(evaluation_paf).resolve()),
        "evaluation_paf_sha256":sha256_file(evaluation_paf),
        "chain_nodes":len(rows),
        "mapped_chain_reads":mapped,
        "mapq60_chain_reads":mapq60,
        "reference_targets":dict(targets),
        "mapped_union_bases":covered,
        "mapped_interval_fragments":fragments,
        "edges":{
            "metric":"ORDER_AND_ORIENTATION_ONLY",
            "total_chain_adjacencies":len(rows)-1,
            "comparable_adjacencies":denom,
            "comparable_fraction":denom/max(1,len(rows)-1),
            "consistent":edge_counts["consistent"],
            "inconsistent":edge_counts["inconsistent"],
            "unmapped_or_cross_target":edge_counts["unmapped_or_cross_target"],
            "consistent_fraction_among_comparable":edge_counts["consistent"]/max(1,denom),
        },
        "edge_rows":edge_rows,
        "assembly_correctness_certified":False,
        "distance_evaluation":{"status":"NOT_ASSESSED","unassessed_adjacencies":len(rows)-1},
        "interpretation":"order/orientation only; no overlap-distance, repeat-copy, assembly correctness or solver optimality certificate; mapped union uses whole reads, not emitted assembly bases",
    }
    if overlap_diagnostic is not None:
        result["distance_evaluation"]=evaluate_overlap_diagnostic(
            overlap_diagnostic,chain_path,evaluation_paf,rows)
    return result

def evaluate_overlap_diagnostic(path,chain_path,evaluation_paf,rows):
    """Consume a frozen CIGAR-based diagnostic, never a production constraint.

    Best reference placements can select different repeat copies. Even strong
    coordinate disagreements require review; they do not certify misjoins.
    """
    p=Path(path)
    raw=json.loads(p.read_text())
    if raw.get("schema")!="PLANB_BOUND_OVERLAP_DIAGNOSTIC_V1":
        raise ValueError("overlap diagnostic schema mismatch")
    for key,source in (("chain_sha256",chain_path),("evaluation_paf_sha256",evaluation_paf)):
        if raw.get(key)!=sha256_file(source):
            raise ValueError("overlap diagnostic identity mismatch: "+key)
    observed=raw.get("rows",[])
    if len(observed)!=len(rows)-1:
        raise ValueError("overlap diagnostic must account for every chain adjacency")
    counts=Counter(); strong=Counter()
    for i,(r,a,b) in enumerate(zip(observed,rows,rows[1:])):
        if (r.get("index"),r.get("source"),r.get("target"))!=(i,a[2],b[2]):
            raise ValueError("overlap diagnostic adjacency mismatch")
        n=r.get("jointly_mapped_overlap_bases")
        d=r.get("median_reference_coordinate_disagreement_bp")
        if not isinstance(n,int) or n<0:
            raise ValueError("invalid jointly mapped base count")
        if n<100:
            key="UNASSESSED_FEWER_THAN_100_BASES"
        else:
            import math
            if not isinstance(d,(int,float)) or not math.isfinite(d) or d<0:
                raise ValueError("invalid overlap coordinate distance")
            key="COPY_COORDINATE_DISAGREEMENT_REQUIRES_REVIEW" if d>1000 else "MEDIAN_WITHIN_1KB"
        counts[key]+=1
        if r.get("both_reads_strong_in_region") is True:
            strong[key]+=1
    return {"status":"POSTHOC_BEST_PLACEMENT_DIAGNOSTIC",
            "diagnostic_sha256":sha256_file(p),"counts":dict(counts),
            "strong_mapping_counts":dict(strong),
            "unassessed_adjacencies":counts["UNASSESSED_FEWER_THAN_100_BASES"],
            "assembly_correctness_certified":False,
            "interpretation":"median aligned-overlap base coordinate distance; ambiguous repeat placements and strong disagreements require review; no edge deletion or correctness promotion"}

def gated_evaluate_paf(audit_path,run_dir,evaluation_paf,out,*,overlap_diagnostic=None):
    # CRITICAL: gate executes before evaluation_paf is opened.
    audit=require_independent_audit_pass(audit_path)
    result=evaluate_chain_paf(run_dir,evaluation_paf,overlap_diagnostic=overlap_diagnostic)
    result["gate"]={
        "status":"UNLOCKED_AFTER_INDEPENDENT_AUDIT_PASS",
        "audit_file":str(Path(audit_path).resolve()),
        "audit_sha256":sha256_file(audit_path),
        "audit_graph_sha256":audit.get("graph",{}).get("sha256"),
    }
    p=Path(out); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,default=str)+"\n")
    return result

def gated_map_reference(audit_path,run_dir,source_reads_fasta,reference_fasta,out_dir,
                        *,preset="map-hifi",threads=4,minimap2="minimap2"):
    # CRITICAL: no reference_fasta stat/open/hash before this gate passes.
    audit=require_independent_audit_pass(audit_path)

    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    chain_fa=out_dir/"chain_reads.fasta"
    extract=extract_chain_reads(run_dir,source_reads_fasta,chain_fa)

    exe=shutil.which(minimap2) if "/" not in minimap2 else minimap2
    if not exe:
        raise RuntimeError(f"minimap2 not found: {minimap2}")

    # Reference path is first touched only after audit PASS.
    reference_fasta=Path(reference_fasta).expanduser().resolve()
    if not reference_fasta.exists():
        raise FileNotFoundError(reference_fasta)
    paf=out_dir/"chain_reads_vs_reference.paf"
    cmd=[str(exe),"-x",str(preset),"-t",str(int(threads)),str(reference_fasta),str(chain_fa)]
    with paf.open("w") as fo:
        cp=subprocess.run(cmd,stdout=fo,stderr=subprocess.PIPE,text=True)
    if cp.returncode!=0:
        raise RuntimeError(f"minimap2 failed rc={cp.returncode}: {(cp.stderr or '')[-3000:]}")

    result=evaluate_chain_paf(run_dir,paf)
    result.update({
        "gate":{
            "status":"UNLOCKED_AFTER_INDEPENDENT_AUDIT_PASS",
            "audit_file":str(Path(audit_path).resolve()),
            "audit_sha256":sha256_file(audit_path),
            "audit_graph_sha256":audit.get("graph",{}).get("sha256"),
        },
        "mapping":{
            "preset":preset,"threads":int(threads),"minimap2":str(exe),
            "reference_fasta":str(reference_fasta),
            "reference_sha256":sha256_file(reference_fasta),
            "chain_extract":extract,
        }
    })
    out_json=out_dir/"POSTHOC_REFERENCE_EVALUATION.json"
    out_json.write_text(json.dumps(result,indent=2,default=str)+"\n")
    return result,out_json

def gated_extract_reference_archive(audit_path,archive_path,out_dir,*,strain_token="CBS1190",member=None):
    """Open/extract a strain assembly archive only after independent audit PASS."""
    audit=require_independent_audit_pass(audit_path)

    # Archive is first touched only after the gate above.
    archive=Path(archive_path).expanduser().resolve()
    if not archive.exists():
        raise FileNotFoundError(archive)
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)

    fasta_suffixes=(".fa",".fasta",".fna",".fa.gz",".fasta.gz",".fna.gz")
    with tarfile.open(archive,"r:*") as tf:
        all_members=[m for m in tf.getmembers() if m.isfile()]
        if member is None:
            token=str(strain_token).lower()
            candidates=[m for m in all_members
                        if token in Path(m.name).name.lower()
                        and Path(m.name).name.lower().endswith(fasta_suffixes)]
            if len(candidates)!=1:
                return {
                    "schema":"PLANB_REFERENCE_ARCHIVE_GATE_V1",
                    "status":"AMBIGUOUS_REFERENCE_MEMBERS" if candidates else "NO_REFERENCE_MEMBER",
                    "gate":"UNLOCKED_AFTER_INDEPENDENT_AUDIT_PASS",
                    "strain_token":strain_token,
                    "candidate_count":len(candidates),
                    "candidates":[m.name for m in candidates[:100]],
                    "archive":str(archive),
                    "archive_sha256":sha256_file(archive),
                    "reference_used":False,
                },None
            chosen=candidates[0]
        else:
            chosen=next((m for m in all_members if m.name==member),None)
            if chosen is None:
                raise RuntimeError(f"requested archive member not found: {member}")

        src=tf.extractfile(chosen)
        if src is None:
            raise RuntimeError(f"cannot read archive member: {chosen.name}")
        raw_name=Path(chosen.name).name
        compressed=raw_name.lower().endswith(".gz")
        if compressed:
            base=raw_name[:-3]
            dst=out_dir/base
            with gzip.GzipFile(fileobj=src,mode="rb") as gz, dst.open("wb") as fo:
                shutil.copyfileobj(gz,fo)
        else:
            dst=out_dir/raw_name
            with dst.open("wb") as fo:
                shutil.copyfileobj(src,fo)

    result={
        "schema":"PLANB_REFERENCE_ARCHIVE_GATE_V1",
        "status":"REFERENCE_EXTRACTED",
        "gate":"UNLOCKED_AFTER_INDEPENDENT_AUDIT_PASS",
        "audit_file":str(Path(audit_path).resolve()),
        "archive":str(archive),
        "archive_sha256":sha256_file(archive),
        "member":chosen.name,
        "reference_fasta":str(dst.resolve()),
        "reference_sha256":sha256_file(dst),
        "reference_used":False,
    }
    (out_dir/"REFERENCE_EXTRACTION.json").write_text(json.dumps(result,indent=2)+"\n")
    return result,dst

def gated_archive_map_reference(audit_path,run_dir,source_reads_fasta,archive_path,out_dir,
                                *,strain_token="CBS1190",member=None,preset="map-hifi",
                                threads=4,minimap2="minimap2"):
    extract,ref=gated_extract_reference_archive(
        audit_path,archive_path,Path(out_dir)/"reference",
        strain_token=strain_token,member=member)
    if ref is None:
        return {"reference_extraction":extract},None
    result,target=gated_map_reference(
        audit_path,run_dir,source_reads_fasta,ref,Path(out_dir)/"evaluation",
        preset=preset,threads=threads,minimap2=minimap2)
    result["reference_extraction"]=extract
    target=Path(out_dir)/"POSTHOC_REFERENCE_EVALUATION.json"
    target.write_text(json.dumps(result,indent=2,default=str)+"\n")
    return result,target

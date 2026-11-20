"""Reference-only-after-feasibility evaluation of frozen, spelled assemblies.

Genome fraction is the primary assessment metric here. No result is sent
back to the solver and no reference is used to construct the assemblies.
"""
from pathlib import Path
import json,math,os,re,subprocess,sys,time
from runtime_utils import sha256_file
from posthoc_reference_gate import require_independent_audit_pass,_fasta_records


def _bound_file(record):
    p=Path(record['path']).expanduser().resolve()
    if sha256_file(p)!=record['sha256']:
        raise ValueError('frozen input SHA mismatch: '+str(p))
    return p


def read_genome_fraction_report(path, ids):
    lines=Path(path).read_text().splitlines()
    rows=[line.split('\t') for line in lines if line]
    if not rows or rows[0]!=['Assembly',*ids]:
        raise ValueError('QUAST assembly columns do not match frozen candidate IDs')
    table={}
    for row in rows[1:]:
        if len(row)!=len(ids)+1 or row[0] in table:
            raise ValueError('malformed or duplicate QUAST metric row')
        table[row[0]]=row[1:]
    if 'Genome fraction (%)' not in table:
        raise ValueError('QUAST did not report Genome fraction')
    secondary=['Total length','# misassemblies','# local misassemblies',
               'Duplication ratio','# mismatches per 100 kbp',
               '# indels per 100 kbp','NGA50']
    result=[]
    for i,name in enumerate(ids):
        raw=table['Genome fraction (%)'][i]
        value=None if raw=='-' else float(raw)
        if value is not None and (not math.isfinite(value) or not 0<=value<=100):
            raise ValueError('invalid Genome fraction value')
        result.append(dict(candidate_id=name,genome_fraction_percent=value,
                           status='ASSESSED' if value is not None else 'UNASSESSED',
                           secondary_metrics={k:table[k][i] for k in secondary if k in table}))
    return result


def evaluate_genome_fraction(manifest_path,out_dir,*,quast,threads=4,eukaryote=False):
    manifest_path=Path(manifest_path).resolve()
    manifest=json.loads(manifest_path.read_text())
    if (manifest.get('schema')!='PLANB_FROZEN_ASSEMBLY_BATCH_V1' or
            manifest.get('reference_used_for_construction') is not False):
        raise ValueError('requires a frozen, reference-free assembly batch')
    assemblies=manifest.get('assemblies',[])
    ids=[a['id'] for a in assemblies]
    if not ids or len(ids)!=len(set(ids)) or any(not re.fullmatch(r'[A-Za-z0-9_.-]+',s) for s in ids):
        raise ValueError('requires unique simple candidate IDs')
    if threads<1:
        raise ValueError('threads must be positive')
    frozen={str(manifest_path):sha256_file(manifest_path)}
    # All feasibility gates precede any reference stat/open/hash.
    prepared=[]
    for a in assemblies:
        audit_path=_bound_file(a['audit']);audit=require_independent_audit_pass(audit_path)
        chain=_bound_file(a['chain']);assembly=_bound_file(a['assembly']);proof_path=_bound_file(a['spelling'])
        if audit.get('chain_sha256')!=sha256_file(chain) or audit.get('graph',{}).get('sha256')!=manifest['graph_sha256']:
            raise ValueError('candidate chain or original graph differs from independent audit')
        proof=json.loads(proof_path.read_text())
        if proof.get('reference_used') is not False:
            raise ValueError('assembly spelling was not reference-free')
        nodes=chain.read_text().splitlines();spans=proof.get('spans',[])
        if nodes!=[s['node'] for s in spans] or not nodes or len(nodes)!=len(set(n[:-1] for n in nodes)):
            raise ValueError('spelling spans differ from the audited physical-once chain')
        records=list(_fasta_records(assembly))
        if len(records)!=1 or not records[0][1]:
            raise ValueError('one nonempty spelled chain per candidate is required')
        prepared.append((a,assembly,nodes,spans,records[0][1]))
        for p in [audit_path,chain,assembly,proof_path]:frozen[str(p)]=sha256_file(p)
    reads=_bound_file(manifest['source_reads']);frozen[str(reads)]=sha256_file(reads)
    wanted={n[:-1] for _,_,nodes,_,_ in prepared for n in nodes}
    seq={name:dna for name,dna in _fasta_records(reads) if name in wanted}
    if wanted-set(seq):
        raise ValueError('source read sequence missing')
    complement=str.maketrans('ACGTRYMKBDHVNacgtrymkbdhvn','TGCAYRKMVHDBNtgcayrkmvhdbn')
    for a,assembly,nodes,spans,dna in prepared:
        offset=0
        for node,span in zip(nodes,spans):
            if node[-1:] not in {'+','-'}:
                raise ValueError('chain lacks explicit read orientation')
            read=seq[node[:-1]] if node[-1]=='+' else seq[node[:-1]].translate(complement)[::-1]
            lo,hi=span['read_start'],span['read_end'];start,end=span['assembly_start'],span['assembly_end']
            if (not 0<=lo<=hi<=len(read) or start!=offset or end-start!=hi-lo or
                    dna[start:end]!=read[lo:hi]):
                raise ValueError('emitted DNA differs from original read spelling')
            offset=end
        if offset!=len(dna):
            raise ValueError('assembly contains bases outside the audited read spans')
    out_dir=Path(out_dir).resolve()
    if out_dir.exists():
        raise ValueError('use a new output directory to avoid QUAST alignment cache mixing')
    # Only now may the reference be accessed.
    reference=_bound_file(manifest['reference']);frozen[str(reference)]=sha256_file(reference)
    quast=Path(quast).expanduser().resolve()
    if not quast.is_file():
        raise FileNotFoundError(quast)
    frozen[str(quast)]=sha256_file(quast)
    out_dir.mkdir(parents=True)
    cmd=[sys.executable,str(quast),*[str(p[1]) for p in prepared],'-r',str(reference),
         '-o',str(out_dir/'quast'),'--labels',','.join(ids),'--threads',str(threads),
         '--no-plots','--no-icarus','--min-identity','95','--extensive-mis-size','1000',
         '--local-mis-size','200','--ambiguity-usage','one']
    if eukaryote:cmd.append('--eukaryote')
    (out_dir/'FROZEN_INPUTS.json').write_text(json.dumps(frozen,indent=2)+'\n')
    env=dict(os.environ)
    dependencies=quast.parent.parent/'python_deps'
    if dependencies.is_dir():env['PYTHONPATH']=str(dependencies)+os.pathsep+env.get('PYTHONPATH','')
    start=time.monotonic()
    with (out_dir/'quast.log').open('w') as log:
        process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
    if process.returncode:
        raise RuntimeError('QUAST failed: see '+str(out_dir/'quast.log'))
    if any(sha256_file(Path(p))!=h for p,h in frozen.items()):
        raise ValueError('input changed during reference evaluation')
    results=read_genome_fraction_report(out_dir/'quast/report.tsv',ids)
    assessed=[r for r in results if r['genome_fraction_percent'] is not None]
    best=max((r['genome_fraction_percent'] for r in assessed),default=None)
    status='PASS' if len(assessed)==len(results) else 'PARTIAL' if assessed else 'UNASSESSED'
    result=dict(schema='PLANB_GENOME_FRACTION_EVALUATION_V1',status=status,
                primary_metric='GENOME_FRACTION_PERCENT',results=results,
                evaluation_best_ids=[r['candidate_id'] for r in assessed if r['genome_fraction_percent']==best],
                command=cmd,seconds=time.monotonic()-start,quast_sha256=sha256_file(quast),
                frozen_inputs=frozen,reference_used_for_construction=False,
                solver_inputs_or_paths_modified=False,original_global_optimum_certified=False,
                biological_correctness_certified=False,
                interpretation='reference bases covered by spelled assembly alignments; overlapping coverage counted once; not read count or correctness percentage')
    (out_dir/'GENOME_FRACTION.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

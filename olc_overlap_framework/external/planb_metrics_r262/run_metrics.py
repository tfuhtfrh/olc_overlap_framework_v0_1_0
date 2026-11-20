#!/usr/bin/env python3
"""Delivery CLI; original evaluation modules are preserved under code/."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/kernels')]
from genome_fraction_evaluation import evaluate_genome_fraction,read_genome_fraction_report
p=argparse.ArgumentParser(description=__doc__)
s=p.add_subparsers(dest='command',required=True)
a=s.add_parser('parse',help='Parse an existing QUAST report; does not establish input provenance')
a.add_argument('report');a.add_argument('--ids',nargs='+',required=True)
a=s.add_parser('evaluate',help='Evaluate a previously frozen, audited batch')
a.add_argument('manifest');a.add_argument('out');a.add_argument('--quast',required=True)
a.add_argument('--threads',type=int,default=2);a.add_argument('--eukaryote',action='store_true')
a=s.add_parser('freeze',help='Audit and spell a final original-read path, then freeze inputs')
for name in ('graph','reads','chain','edges','reference','out'):a.add_argument('--'+name,required=True)
a.add_argument('--id',default='candidate01')
args=p.parse_args()
if args.command=='parse':
 result=read_genome_fraction_report(args.report,args.ids)
elif args.command=='evaluate':
 result=evaluate_genome_fraction(args.manifest,args.out,quast=args.quast,threads=args.threads,eukaryote=args.eukaryote)
else:
 from runtime_utils import sha256_file
 from independent_final_audit import write_audit
 from spell_read_path import spell
 out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
 auditdir=out/'audit';auditdir.mkdir()
 (auditdir/'final_summary.json').write_text(json.dumps({'status':'FEASIBLE_CANDIDATE','optimality_certified':False}))
 audit,ap=write_audit(args.graph,auditdir,chain_file=args.chain)
 if audit['status']!='PASS':raise RuntimeError('Audit failed; reference evaluation remains locked: '+str(ap))
 fasta,proof,_=spell(args.graph,args.reads,args.chain,out/'assembly',edge_ids_path=args.edges)
 def record(path):
  path=Path(path).resolve();return {'path':str(path),'sha256':sha256_file(path)}
 result={'schema':'PLANB_FROZEN_ASSEMBLY_BATCH_V1','reference_used_for_construction':False,
 'graph_sha256':sha256_file(args.graph),'source_reads':record(args.reads),'reference':record(args.reference),
 'assemblies':[{'id':args.id,'audit':record(ap),'chain':record(args.chain),'assembly':record(fasta),'spelling':record(proof)}]}
 (out/'FROZEN_BATCH.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2,ensure_ascii=False))

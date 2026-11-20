"""Spell one audited original-read path with reference-free, feasible overlap cuts."""
from pathlib import Path
import bisect,hashlib,json,re
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _spell_oriented(graph_path,reads_path,chain_path,out_prefix,*,edge_ids_path=None,_prepared_graph=None,_prepared_sequences=None):
 import networkx as nx
 import edlib
 from graph_common import load_graph
 g=_prepared_graph if _prepared_graph is not None else (nx.read_graphml(graph_path) if Path(graph_path).suffix.lower() in ('.graphml','.xml') else load_graph(graph_path))
 nodes=Path(chain_path).read_text().splitlines()
 assert nodes and len(nodes)==len({n[:-1]for n in nodes})and all(g.has_edge(u,v)for u,v in zip(nodes,nodes[1:]))
 from regenerate_hifi_overlaps import records
 wanted={n[:-1]for n in nodes};seq=({n:_prepared_sequences[n]for n in wanted} if _prepared_sequences is not None else {n:s for n,s in records(reads_path)if n in wanted});assert set(seq)==wanted
 trans=str.maketrans('ACGTRYMKBDHVNacgtrymkbdhvn','TGCAYRKMVHDBNtgcayrkmvhdbn')
 dna={n:seq[n[:-1]]if n[-1]=='+'else seq[n[:-1]].translate(trans)[::-1]for n in nodes}
 assert all(len(dna[n])==int(g.nodes[n]['length'])for n in nodes)
 def interval(n,a):
  side='q'if n[:-1]==a['q']else't';assert n[:-1]==a[side]
  lo,hi,L=int(a[side+'start']),int(a[side+'end']),int(a[side+'len'])
  return (lo,hi)if n[-1]=='+'else(L-hi,L-lo)
 edge_ids=Path(edge_ids_path).read_text().splitlines() if edge_ids_path else None
 if edge_ids is not None and len(edge_ids)!=len(nodes)-1:raise ValueError('edge witness length differs from path')
 if g.is_multigraph() and edge_ids is None:raise ValueError('parallel overlaps require the solver original-edge witness')
 joints=[];maps=[]
 for index,(u,v) in enumerate(zip(nodes,nodes[1:])):
  a=g[u][v]
  if g.is_multigraph():
   choices=[d for k,d in a.items() if str(d.get('edge_id',k))==edge_ids[index]]
   if len(choices)!=1:raise ValueError('original-edge witness is absent or ambiguous')
   a=choices[0]
  us,ue=interval(u,a);vs,ve=interval(v,a)
  z=edlib.align(dna[u][us:ue],dna[v][vs:ve],mode='NW',task='path')
  cigar=z['cigar'];ops=[(int(k),c)for k,c in re.findall(r'(\d+)([=XID])',cigar)]
  q=t=0;ends=[];starts=[];kinds=[]
  for k,c in ops:
   if c=='D':t+=k;continue
   starts.append(t+vs);kinds.append(c);q+=k;ends.append(q)
   if c in '=X':t+=k
  assert q==ue-us and t==ve-vs and sum(k for k,c in ops if c!='=')==z['editDistance']
  maps.append((us,vs,ends,starts,kinds))
  joints.append(dict(source=u,target=v,source_interval=[us,ue],target_interval=[vs,ve],cigar=cigar,global_overlap_edit_distance=z['editDistance']))
 def cut_map(i,cut):
  us,vs,ends,starts,kinds=maps[i];offset=cut-us
  if offset==0:return vs
  j=bisect.bisect_left(ends,offset);previous=ends[j-1]if j else 0
  return starts[j]+(offset-previous if kinds[j]in '=X'else 0)
 upper=[0]*len(joints);bound=len(dna[nodes[-1]])
 for i in range(len(joints)-1,-1,-1):
  lo,hi=joints[i]['source_interval']
  if cut_map(i,lo)>bound:raise ValueError('no globally monotone cut for the frozen path and overlap alignments')
  left,right=lo,hi
  while left<right:
   middle=(left+right+1)//2
   if cut_map(i,middle)<=bound:left=middle
   else:right=middle-1
  upper[i]=left;bound=left
 incoming=offset=0;parts=[];spans=[];cuts=[]
 for i,j in enumerate(joints):
  lo=max(incoming,j['source_interval'][0]);hi=upper[i]
  if lo>hi:raise ValueError('incompatible consecutive overlap cuts')
  cut=(lo+hi)//2;target=cut_map(i,cut);n=nodes[i];part=dna[n][incoming:cut]
  parts.append(part);spans.append(dict(node=n,read_start=incoming,read_end=cut,assembly_start=offset,assembly_end=offset+len(part)))
  cuts.append(dict(source_cut=cut,target_cut=target,feasible_source_cut=[lo,hi]));offset+=len(part);incoming=target
 n=nodes[-1];part=dna[n][incoming:];parts.append(part);spans.append(dict(node=n,read_start=incoming,read_end=len(dna[n]),assembly_start=offset,assembly_end=offset+len(part)))
 assembly=''.join(parts)
 assert all(assembly[s['assembly_start']:s['assembly_end']]==dna[s['node']][s['read_start']:s['read_end']]for s in spans)
 prefix=Path(out_prefix);prefix.parent.mkdir(parents=True,exist_ok=True);fasta=Path(str(prefix)+'.draft.fa');proof=Path(str(prefix)+'.spelling.json')
 with fasta.open('w')as f:
  f.write('>'+prefix.name+'\n')
  for k in range(0,len(assembly),80):f.write(assembly[k:k+80]+'\n')
 evidence=dict(reference_used=False,graph_sha256=sha(graph_path),source_reads_sha256=sha(reads_path),source_path_sha256=sha(chain_path),source_edge_ids_sha256=sha(edge_ids_path) if edge_ids_path else None,spans=spans,joints=joints,cuts=cuts,draft_bases=len(assembly),path_reads=len(nodes),cut_rule='backward_feasibility_then_forward_midpoint',assembly_sha256=sha(fasta))
 proof.write_text(json.dumps(evidence,indent=2))
 return fasta,proof,evidence


def spell(graph_path,reads_path,chain_path,out_prefix,*,edge_ids_path=None,canonical_rc=None,_prepared_graph=None,_prepared_sequences=None):
 """Use a common cut convention for a path and its reverse complement when enabled.

 Legacy calls retain their existing cut convention unless the marker-selection
 experiment is configured. Exact original PAF provenance is required for RC arcs.
 """
 import os,tempfile
 if canonical_rc is None:canonical_rc=bool(os.environ.get('PLANB_EMITTED_MARKER_SELECTION_CONFIG'))
 if not canonical_rc:return _spell_oriented(graph_path,reads_path,chain_path,out_prefix,edge_ids_path=edge_ids_path,_prepared_graph=_prepared_graph,_prepared_sequences=_prepared_sequences)
 if edge_ids_path is None:raise ValueError('canonical spelling requires original edge IDs')
 from graph_common import load_graph,build_mate_map
 g=_prepared_graph if _prepared_graph is not None else load_graph(graph_path);mate=build_mate_map(g)
 nodes=tuple(Path(chain_path).read_text().splitlines());ids=tuple(Path(edge_ids_path).read_text().splitlines())
 rc=tuple(mate[n]for n in reversed(nodes));rcids=[]
 for u,v,eid in zip(nodes,nodes[1:],ids):
  data=g[u][v][eid];row=data.get('paf_row')
  if row is None:raise ValueError('canonical spelling needs original PAF row provenance')
  choices=[k for k,d in g[mate[v]][mate[u]].items() if d.get('paf_row')==row]
  if len(choices)!=1:raise ValueError('canonical spelling RC original edge is ambiguous')
  rcids.append(choices[0])
 rcids=tuple(reversed(rcids));reverse=(rc,rcids)<(nodes,ids)
 lengths={n:int(g.nodes[n]['length'])for n in nodes};del g
 if not reverse:
  fasta,proof,d=_spell_oriented(graph_path,reads_path,chain_path,out_prefix,edge_ids_path=edge_ids_path,_prepared_graph=_prepared_graph,_prepared_sequences=_prepared_sequences)
  d['canonical_rc']=True;d['cut_rule']='canonical_rc_backward_feasibility_then_forward_midpoint'
  proof.write_text(json.dumps(d,indent=2));return fasta,proof,d
 prefix=Path(out_prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='canonical_spelling_',dir=prefix.parent)as td:
  td=Path(td);p=td/'path.txt';e=td/'edges.txt';p.write_text('\n'.join(rc)+'\n');e.write_text('\n'.join(rcids)+'\n')
  fa,_,original=_spell_oriented(graph_path,reads_path,p,td/'canonical',edge_ids_path=e,_prepared_graph=_prepared_graph,_prepared_sequences=_prepared_sequences)
  seq=''.join(l.strip()for l in fa.read_text().splitlines()if not l.startswith('>'))
 trans=str.maketrans('ACGTRYMKBDHVNacgtrymkbdhvn','TGCAYRKMVHDBNtgcayrkmvhdbn');seq=seq.translate(trans)[::-1]
 spans=[];joints=[];cuts=[];L=len(seq)
 for s in reversed(original['spans']):
  n=mate[s['node']];N=lengths[n]
  spans.append(dict(node=n,read_start=N-s['read_end'],read_end=N-s['read_start'],assembly_start=L-s['assembly_end'],assembly_end=L-s['assembly_start']))
 for j,c in zip(reversed(original['joints']),reversed(original['cuts'])):
  u,v=mate[j['target']],mate[j['source']];Nu,Nv=lengths[u],lengths[v]
  cigar=''.join(k+({'I':'D','D':'I'}.get(op,op))for k,op in reversed(re.findall(r'(\d+)([=XID])',j['cigar'])))
  joints.append(dict(source=u,target=v,source_interval=[Nu-j['target_interval'][1],Nu-j['target_interval'][0]],target_interval=[Nv-j['source_interval'][1],Nv-j['source_interval'][0]],cigar=cigar,global_overlap_edit_distance=j['global_overlap_edit_distance']))
  sc,tc=Nu-c['target_cut'],Nv-c['source_cut']
  cuts.append(dict(source_cut=sc,target_cut=tc,feasible_source_cut=[sc,sc]))
 fasta=Path(str(prefix)+'.draft.fa');proof=Path(str(prefix)+'.spelling.json')
 with fasta.open('w')as f:
  f.write('>'+prefix.name+'\n')
  for k in range(0,len(seq),80):f.write(seq[k:k+80]+'\n')
 d={**original,'source_path_sha256':sha(chain_path),'source_edge_ids_sha256':sha(edge_ids_path),'spans':spans,'joints':joints,'cuts':cuts,'assembly_sha256':sha(fasta),'canonical_rc':True,'cut_rule':'canonical_rc_backward_feasibility_then_forward_midpoint','feasible_cut_interval_scope':'single witnessed cut when reversing canonical spelling, not exhaustive interval'}
 proof.write_text(json.dumps(d,indent=2));return fasta,proof,d

"""Reference-free HiFi overlap regeneration with aligned identity and resumable batches."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,gzip,time,collections
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def records(p):
 name=None;parts=[]
 for line in Path(p).open():
  if line.startswith('>'):
   if name is not None:yield name,''.join(parts)
   name=line[1:].split()[0];parts=[]
  else:parts.append(line.strip())
 if name is not None:yield name,''.join(parts)
def main():
 a=argparse.ArgumentParser();a.add_argument('reads');a.add_argument('out');a.add_argument('--minimap2',default='/opt/homebrew/bin/minimap2');a.add_argument('--threads',type=int,default=8);a.add_argument('--batch-size',type=int,default=64);a.add_argument('--minimum-identity',type=float,default=.99);a.add_argument('--minimum-span',type=int,default=1000);x=a.parse_args()
 if x.threads<1 or x.batch_size<1:raise ValueError('positive threads and batch size required')
 if not 0<x.minimum_identity<=1 or x.minimum_span<1:raise ValueError('invalid overlap filter')
 reads=Path(x.reads).resolve();out=Path(x.out).resolve();out.mkdir(parents=True,exist_ok=True)
 sequence=list(records(reads));assert len(sequence)==len({n for n,s in sequence})and all(s for n,s in sequence)
 version=subprocess.check_output([x.minimap2,'--version'],text=True).strip()
 binding={'reads':str(reads),'reads_sha256':sha(reads),'read_count':len(sequence),'minimap2_sha256':sha(x.minimap2),'minimap2_version':version,'preset':'map-hifi','all_vs_all':True,'base_alignment':True,'identity_min':x.minimum_identity,'span_min':x.minimum_span,'batch_size':x.batch_size,'reference_used':False,'top_k_filter':False,'contained_removal':False}
 (out/'PRODUCER.json').write_text(json.dumps({'pid':os.getpid(),'reads_sha256':binding['reads_sha256']}))
 manifest=out/'INPUT_BINDING.json'
 if manifest.exists():assert json.loads(manifest.read_text())==binding,'cannot reuse checkpoint with different inputs or options'
 else:manifest.write_text(json.dumps(binding,indent=2))
 index=out/'reads.map_hifi.mmi'
 if not index.exists():
  temp=out/'reads.map_hifi.partial.mmi'
  with (out/'index.log').open('w')as log:subprocess.run([x.minimap2,'-x','map-hifi','-c','-t',str(x.threads),'-d',str(temp),str(reads)],stderr=log,check=True)
  temp.replace(index)
 started=time.monotonic();total=collections.Counter();progress={'completed_queries':0,'total_queries':len(sequence),'complete':False}
 for start in range(0,len(sequence),x.batch_size):
  tag=f'batch_{start:05d}';done=out/(tag+'.json');target=out/(tag+'.paf.gz')
  names=[n for n,s in sequence[start:start+x.batch_size]]
  if done.exists():
   receipt=json.loads(done.read_text());assert receipt['query_names']==names and receipt['paf_sha256']==sha(target)
   total.update(receipt['counts']);progress.update(completed_queries=min(start+x.batch_size,len(sequence)),counts=dict(total));continue
  query=out/(tag+'.fasta');query.write_text(''.join('>'+n+'\n'+s+'\n'for n,s in sequence[start:start+x.batch_size]))
  cmd=[x.minimap2,'-x','map-hifi','-X','-c','--eqx','-t',str(x.threads),'-K','1M',str(index),str(query)]
  partial=out/(tag+'.partial.paf.gz');counts=collections.Counter();begin=time.monotonic()
  with (out/(tag+'.log')).open('w')as log,gzip.open(partial,'wt')as dest:
   proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=log,text=True)
   try:
    for line in proc.stdout:
     f=line.rstrip().split('\t');counts['reported']+=1
     assert len(f)>=12 and f[0]in names and any(t.startswith('cg:Z:')for t in f[12:])
     if f[0]==f[5]:counts['self']+=1;continue
     if int(f[9])/int(f[10])<x.minimum_identity:counts['below_identity']+=1;continue
     if min(int(f[3])-int(f[2]),int(f[8])-int(f[7]))<x.minimum_span:counts['short']+=1;continue
     dest.write(line);counts['retained']+=1
    assert proc.wait()==0,'minimap2 failed; partial batch will not be committed'
   except BaseException:
    proc.kill();proc.wait();raise
  partial.replace(target)
  receipt={'query_names':names,'counts':dict(counts),'command':cmd,'seconds':time.monotonic()-begin,'paf_sha256':sha(target),'complete':True}
  done.write_text(json.dumps(receipt,indent=2));query.unlink();total.update(counts)
  progress={'completed_queries':min(start+x.batch_size,len(sequence)),'total_queries':len(sequence),'counts':dict(total),'seconds_this_invocation':time.monotonic()-started,'complete':False}
  (out/'PROGRESS.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress),flush=True)
 assert sha(reads)==binding['reads_sha256'] and sha(x.minimap2)==binding['minimap2_sha256']
 progress['complete']=True;(out/'PROGRESS.json').write_text(json.dumps(progress,indent=2));print('COMPLETE',json.dumps(progress),flush=True)
if __name__=='__main__':main()

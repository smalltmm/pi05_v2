import json,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import av
ROOT=(Path(__file__).resolve().parents[1] / "training_data" / "Robotwin2.0-lerobot-piper-v2")
def trim(job):
 ep,cam=job;i=ep['episode_index'];n=ep['length'];path=ROOT/f'videos/chunk-{i//1000:03d}/observation.images.{cam}/episode_{i:06d}.mp4';tmp=path.with_suffix('.trim.mp4')
 with av.open(str(path)) as src:
  s=src.streams.video[0]
  if s.frames==n:return 'ok'
  assert s.frames==n+1,(path,s.frames,n)
  assert not s.codec_context.has_b_frames,(path,'B frames')
  count=0
  with av.open(str(tmp),'w') as dst:
   out=dst.add_stream_from_template(s)
   for pkt in src.demux(s):
    if pkt.pts is None:continue
    frame_number=round(float(pkt.pts*pkt.time_base)*30)
    if frame_number>=n:continue
    assert frame_number==count,(path,frame_number,count)
    pkt.stream=out;dst.mux(pkt);count+=1
  assert count==n,(path,count,n)
 with av.open(str(tmp)) as verify:assert verify.streams.video[0].frames==n
 os.replace(tmp,path)
 return 'trimmed'
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--all',action='store_true');args=parser.parse_args()
 eps=[json.loads(x) for x in open(ROOT/'meta/episodes.jsonl')]
 if not args.all:eps=eps[:1]
 jobs=[(ep,c) for ep in eps for c in ['head','left_wrist','right_wrist']];counts={}
 with ThreadPoolExecutor(12) as pool:
  fs=[pool.submit(trim,j) for j in jobs]
  for i,f in enumerate(as_completed(fs),1):
   status=f.result();counts[status]=counts.get(status,0)+1
   if i%3000==0:print(i,counts,flush=True)
 print('DONE',len(jobs),counts,flush=True)

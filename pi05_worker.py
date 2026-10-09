"""Dependency-isolated inference worker for a JAX OpenPI PI05 checkpoint."""
from __future__ import annotations
import argparse, json, logging, os, sys, time, traceback
from checkpoint_assets import resolve_norm_stats
from pathlib import Path
import numpy as np

def args():
 p=argparse.ArgumentParser(); p.add_argument('--openpi-root',required=True); p.add_argument('--checkpoint-path',required=True); p.add_argument('--train-config-name',required=True); p.add_argument('--num-inference-steps',type=int,default=10); p.add_argument('--device'); p.add_argument('--compile-mode'); return p.parse_args()
def emit(x): print(json.dumps(x,ensure_ascii=False),flush=True)
def imports(root):
 for p in [root/'src',root/'packages'/'openpi-client'/'src']:
  if str(p) not in sys.path: sys.path.insert(0,str(p))
def load(a):
 imports(Path(a.openpi_root).resolve())
 from openpi.policies import policy_config
 from openpi.training import config as training_config
 from openpi.training import checkpoints
 config=training_config.get_config(a.train_config_name)
 checkpoint=Path(a.checkpoint_path).resolve()
 data_config=config.data.create(config.assets_dirs,config.model)
 norm_file=resolve_norm_stats(checkpoint,data_config.asset_id,os.environ.get('PI05_NORM_ASSET_ID'))
 logging.info("Checkpoint normalization statistics: %s",norm_file)
 norm_stats=checkpoints.load_norm_stats(norm_file.parent.parent,norm_file.parent.name)
 return policy_config.create_trained_policy(config,checkpoint,norm_stats=norm_stats,sample_kwargs={'num_steps':a.num_inference_steps})
def load_obs(path):
 with np.load(path,allow_pickle=False) as d:
  return {'state':np.asarray(d['observation/state'],dtype=np.float32),'images':{'cam_high':np.moveaxis(np.asarray(d['observation/image']),-1,0),'cam_left_wrist':np.moveaxis(np.asarray(d['observation/left_wrist_image']),-1,0),'cam_right_wrist':np.moveaxis(np.asarray(d['observation/right_wrist_image']),-1,0)},'prompt':str(d['prompt'].item())}
def main():
 a=args(); logging.basicConfig(level=logging.INFO,stream=sys.stderr,force=True); policy=load(a)
 emit({'ok':True,'event':'ready','config':a.train_config_name,'checkpoint':str(Path(a.checkpoint_path).resolve())})
 for line in sys.stdin:
  try:
   req=json.loads(line); cmd=req.get('cmd')
   if cmd=='shutdown': emit({'ok':True,'event':'shutdown'}); return
   if cmd=='reset': emit({'ok':True,'event':'reset'}); continue
   if cmd!='infer': raise ValueError(f'Unsupported worker command: {cmd!r}')
   obs=load_obs(req['obs_path']); start=time.perf_counter(); result=policy.infer(obs); actions=np.asarray(result['actions'],dtype=np.float32)
   if actions.ndim!=2: raise ValueError(f'OpenPI produced actions with shape {actions.shape}; expected [T,D].')
   actions=actions[:int(req.get('max_actions',actions.shape[0]))]
   emit({'ok':True,'actions':actions.tolist(),'infer_seconds':time.perf_counter()-start})
  except Exception: emit({'ok':False,'error':traceback.format_exc()})
if __name__=='__main__':
 try: main()
 except Exception: emit({'ok':False,'error':traceback.format_exc()}); raise


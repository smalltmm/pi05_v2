from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
from convert_robotwin_piper_v3_to_v2 import read_piper_episodes, camera_source_path, camera_output_path, extract_video_segment, CAMERAS
SOURCE=(Path(__file__).resolve().parents[1] / "training_data" / "Robotwin2.0-lerobot")
OUTPUT=(Path(__file__).resolve().parents[1] / "training_data" / "Robotwin2.0-lerobot-piper-v2")

def repair_one(job):
    i, rec = job
    done=0
    for cam in CAMERAS:
        dst=camera_output_path(OUTPUT,cam,i)
        tmp=dst.with_suffix('.repair.mp4')
        dst.parent.mkdir(parents=True,exist_ok=True)
        tmp.unlink(missing_ok=True)
        frames=extract_video_segment(camera_source_path(SOURCE,cam,rec), tmp, float(rec[f'videos/{cam}/from_timestamp']), float(rec[f'videos/{cam}/to_timestamp']))
        os.replace(tmp,dst)
        done += frames
    return i, done

if __name__=='__main__':
    records=read_piper_episodes(SOURCE,None)
    print('episodes',len(records),flush=True)
    jobs=list(enumerate(records))
    with ProcessPoolExecutor(max_workers=24) as ex:
        futures=[ex.submit(repair_one,j) for j in jobs]
        for n,f in enumerate(as_completed(futures),1):
            i,frames=f.result()
            if n%100==0: print('done',n,'last',i,'frames',frames,flush=True)
    print('complete',flush=True)


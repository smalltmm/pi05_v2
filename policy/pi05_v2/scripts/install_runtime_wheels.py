"""Install the SHA256-verified Linux x86_64 / CPython 3.11 runtime bundle."""
from pathlib import Path
import hashlib,json,subprocess,sys
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'runtime-wheel-manifest.json').read_text())
wheels=[]
for item in manifest:
    path=root/'runtime_wheels'/item['filename']
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''): h.update(chunk)
    if h.hexdigest()!=item['sha256']:
        raise RuntimeError(f'Runtime wheel checksum mismatch: {path.name}')
    wheels.append(str(path))
print(f'Verified {len(wheels)} runtime wheels',flush=True)
subprocess.run(['uv','pip','install','--python',sys.executable,'--no-index','--find-links',str(root/'runtime_wheels'),'--no-deps',*wheels],check=True)

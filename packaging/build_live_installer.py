"""Build a portable Live installer from a tested local Release build."""
import hashlib,json,shutil,zipfile
from pathlib import Path
from urllib.request import urlretrieve
root=Path(__file__).resolve().parents[1]
stage=root/'dist/live-build'
stage.mkdir(parents=True,exist_ok=True)
for p in (root/'packaging/live-installer').iterdir():
 if p.is_file() and p.name != "check_installer.py":shutil.copy2(p,stage/p.name)
shutil.copytree(root/'installer',stage/'base-installer',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','installer-settings.json'))
shutil.copytree(root/'backend/src',stage/'backend/src',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'))
with zipfile.ZipFile(root/'Posegraph/bin/Release/Posegraph/latest.zip') as z:z.extractall(stage/'plugin')
for relative in ('face/face_landmarker.task','pose/pose_landmarker_lite.task','hand/hand_landmarker.task'):
 p=stage/'models'/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/'backend/checkpoints'/relative,p)
(stage/'docs').mkdir(exist_ok=True)
shutil.copy2(root/'docs/live.zh-CN.md',stage/'docs/live.zh-CN.md')
for name in ('LICENSE','THIRD_PARTY_NOTICES.md'):shutil.copy2(root/name,stage/name)
urlretrieve('https://www.apache.org/licenses/LICENSE-2.0.txt',stage/'MediaPipe-APACHE-2.0.txt')
(stage/'Live-models.txt').write_text('MediaPipe Face, Pose Lite and Hand Landmarker models: Google / Apache-2.0. Download sources: https://storage.googleapis.com/mediapipe-models/ . cv2-enumerate-cameras wheel includes its distribution license. No camera images or local user settings are included.\n',encoding='utf-8')
if not list((stage/'wheels').glob('*.whl')):raise RuntimeError('Download cv2-enumerate-cameras==1.4.0 win_amd64 abi3 wheel to dist/live-build/wheels first')
manifest={p.relative_to(stage).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.rglob('*') if p.is_file() and p.name!='sha256.json'}
(stage/'sha256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
output=root/'dist/Posegraph-Live-Preview-Installer.zip'
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
 for p in stage.rglob('*'):
  if p.is_file():z.write(p,'Posegraph-Live/'+p.relative_to(stage).as_posix())
print(output)
print('Bytes:',output.stat().st_size)
print('SHA256:',hashlib.sha256(output.read_bytes()).hexdigest())

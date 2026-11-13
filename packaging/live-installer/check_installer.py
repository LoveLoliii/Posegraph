import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
spec=importlib.util.spec_from_file_location('installer',Path('packaging/live-installer/install_live.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as temp:
 base=Path(temp);root=base/'target';payload=base/'payload';appdata=base/'appdata'
 for directory in (root/'python',root/'backend/src/posegraph',payload/'backend/src/posegraph',payload/'plugin',payload/'models/pose',payload/'wheels',appdata/'XIVLauncherCN'):
  directory.mkdir(parents=True,exist_ok=True)
 (root/'python/python.exe').write_text('fake')
 (root/'backend/src/posegraph/app.py').write_text('old')
 (payload/'backend/src/posegraph/app.py').write_text('new')
 (payload/'plugin/Posegraph.dll').write_text('dll')
 (payload/'models/pose/test.task').write_text('model')
 manifest={p.relative_to(payload).as_posix():m.hashlib.sha256(p.read_bytes()).hexdigest() for p in payload.rglob('*') if p.is_file()}
 (payload/'sha256.json').write_text(json.dumps(manifest))
 config=appdata/'XIVLauncherCN/dalamudConfig.json'
 config.write_text(json.dumps({'OtherSetting':123,'DevPluginLoadLocations':{'$values':[{'Path':'C:/old/Posegraph.dll','IsEnabled':True}]}}))
 with patch.object(m,'HERE',payload),patch.object(sys,'argv',['installer','--root',str(root)]),patch.dict(m.os.environ,{'APPDATA':str(appdata)}),patch.object(m.subprocess,'run',return_value=SimpleNamespace(returncode=0,stderr='')):
  m.main()
 new=json.loads(config.read_text(encoding="utf-8"))
 assert new['OtherSetting']==123
 assert new['DevPluginLoadLocations']['$values'][0]['IsEnabled'] is False
 assert new['DevPluginLoadLocations']['$values'][-1]['IsEnabled'] is True
 assert (root/'backend/src/posegraph/app.py').read_text()=='new'
 backup=next((root/'live-backups').iterdir())
 assert (backup/'src/posegraph/app.py').read_text()=='old'
 assert json.loads((backup/'XIVLauncherCN-dalamudConfig.json').read_text())['OtherSetting']==123
 print('Isolated installer/config/backup checks passed')

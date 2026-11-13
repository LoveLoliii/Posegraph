"""Install release payload into a selected portable backend, preserving backups."""
import argparse, datetime, hashlib, json, os, shutil, subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def save(path, value):
    temp=path.with_suffix(path.suffix+'.live-tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);args=ap.parse_args()
    root=args.root.resolve();python=root/'python/python.exe';src=root/'backend/src'
    if not python.is_file() or not (src/'posegraph/app.py').is_file():raise RuntimeError('Invalid backend folder')
    # Validate package integrity before changing any installed files.
    for relative, digest in load(HERE/'sha256.json').items():
        path=(HERE/relative).resolve()
        if not path.is_relative_to(HERE):raise RuntimeError('Invalid manifest path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise RuntimeError('Package checksum failed: '+relative)
    probes=subprocess.run([str(python),'-c','import mediapipe,cv2,numpy,scipy;from mediapipe.tasks.python import vision'],capture_output=True,text=True)
    if probes.returncode:raise RuntimeError('基础后端缺少面捕依赖，请先用原项目安装器更新/修复。'+probes.stderr)
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    backup=root/'live-backups'/('installer-'+stamp);backup.mkdir(parents=True)
    shutil.copytree(src,backup/'src')
    plugin=root/'LivePlugin'
    if plugin.exists():shutil.copytree(plugin,backup/'LivePlugin')
    subprocess.run([str(python),'-m','pip','install','--no-deps','--no-index','--find-links',str(HERE/'wheels'),'cv2-enumerate-cameras==1.4.0'],check=True)
    try:
        shutil.copytree(HERE/'backend/src',src,dirs_exist_ok=True)
        shutil.copytree(HERE/'plugin',plugin,dirs_exist_ok=True)
        shutil.copytree(HERE/'models',root/'backend/checkpoints',dirs_exist_ok=True)
        env=os.environ.copy();env['PYTHONPATH']=str(src)
        subprocess.run([str(python),'-c','from posegraph.app import app; from posegraph.live import cameras; cameras(); print("Live backend OK")'],env=env,cwd=root/'backend',check=True)
    except Exception:
        shutil.copytree(backup/'src',src,dirs_exist_ok=True)
        raise
    dll=str(plugin/'Posegraph.dll')
    registered=[]
    for launcher in ('XIVLauncherCN','XIVLauncher'):
        config=Path(os.environ['APPDATA'])/launcher/'dalamudConfig.json'
        if not config.is_file():continue
        data=load(config)
        locations=data.get('DevPluginLoadLocations',[])
        entries=locations.get('$values',[]) if isinstance(locations,dict) else locations
        if not isinstance(entries,list):raise RuntimeError('Unknown DevPluginLoadLocations format')
        for entry in entries:
            if Path(entry.get('Path','')).name.lower()=='posegraph.dll':entry['IsEnabled']=False
        entries.append({'Path':dll,'IsEnabled':True,'Nickname':'Posegraph Live 开发版'})
        if isinstance(locations,dict):locations['$values']=entries
        else:locations=entries
        data['DevPluginLoadLocations']=locations
        devsettings=data.setdefault('DevPluginSettings',{})
        devsettings[dll]={'StartOnBoot':True,'NotifyForErrors':True,'AutomaticReloading':True}
        shutil.copy2(config,backup/(launcher+'-dalamudConfig.json'))
        save(config,data)
        pc=config.parent/'pluginConfigs/Posegraph.json';pc.parent.mkdir(parents=True,exist_ok=True)
        settings=load(pc) if pc.exists() else {}
        if pc.exists():shutil.copy2(pc,backup/(launcher+'-Posegraph.json'))
        settings.update(BackendDir=str(root/'backend'),BackendPython=str(python))
        save(pc,settings);registered.append(launcher)
    (root/'Live-installation.txt').write_text('Plugin: '+dll+'\nBackup: '+str(backup)+'\nLaunchers: '+','.join(registered),encoding='utf-8')
    if not registered:print('没有找到启动器配置。先运行一次 Dalamud，然后重跑安装器。插件位置：'+dll)
    print('Backup:',backup)

if __name__=='__main__':main()

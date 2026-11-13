"""Download the two official MediaPipe model bundles used by Live."""
from pathlib import Path
from urllib.request import urlretrieve
root = Path(__file__).resolve().parents[1] / 'checkpoints'
models = {
 'hand/hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
 'face/face_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
 'pose/pose_landmarker_lite.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
}
for relative, url in models.items():
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        temporary = path.with_suffix('.download')
        urlretrieve(url, temporary)
        temporary.replace(path)
    print(path)

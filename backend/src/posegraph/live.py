"""Opt-in camera sessions. One bounded frame per request; no queued inference."""
from __future__ import annotations
import threading
import time
import uuid
import base64
from typing import Annotated, Literal
import numpy as np
from scipy.spatial.transform import Rotation as R
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from .config import settings
from .retarget.brio_pose import BrioPose, BoneTransform
from . import face_rig
from .live_body import UpperBody
from .live_hands import LiveHands
from .live_camera_body import CameraBody
from .live_calibration import CalibrationGate

router = APIRouter(prefix='/live')
_lock = threading.RLock()
_session = None

def preview_jpeg(bgr):
    import cv2
    h, w = bgr.shape[:2]
    scale = min(1., 480 / max(w, h))
    small = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))))
    ok, data = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok: raise ValueError('摄像头预览编码失败')
    return base64.b64encode(data).decode('ascii')

@router.get('/cameras')
def cameras():
    import cv2
    from cv2_enumerate_cameras import enumerate_cameras
    return [dict(index=c.index, name=c.name, path=c.path) for c in enumerate_cameras(cv2.CAP_DSHOW)]

class MouthControls(BaseModel):
    mouth_strength: float = Field(default=1., ge=0, le=1)
    mouth_limit: float = Field(default=1., ge=0, le=1)
    face_strength: float = Field(default=1., ge=0, le=2)
    face_gains: dict[Literal['eyes', 'brows', 'mouth', 'jaw', 'gaze'], Annotated[float, Field(ge=0, le=2)]] = Field(default_factory=dict)

class Start(MouthControls):
    camera_pose: bool = True
    reference: str
    race: str
    camera: int = Field(default=0, ge=0, le=16)
    smoothing: float = Field(default=0.35, ge=0.05, le=1)
    camera_name: str | None = None
    camera_path: str | None = None

def check_camera(req, devices):
    if req.camera_name is None and req.camera_path is None:
        return
    current = next((c for c in devices if c['index'] == req.camera), None)
    if current is None or current['name'] != req.camera_name or (req.camera_path and current['path'] != req.camera_path):
        raise ValueError('摄像头列表已改变，请刷新设备并重新选择，避免打开其他摄像头。')

def tracking_overlay(face, body):
    lines = []
    landmarks = getattr(body, 'pose_landmarks', [])
    if landmarks:
        points = landmarks[0]
        for a, b in ((11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (11, 23), (12, 24), (23, 24)):
            if min(points[a].visibility, points[b].visibility) >= .65:
                lines.append([points[a].x, points[a].y, points[b].x, points[b].y])
    box = None
    if face.face_landmarks:
        p = face.face_landmarks[0]
        box = [min(v.x for v in p), min(v.y for v in p), max(v.x for v in p), max(v.y for v in p)]
    return dict(lines=lines, faceBox=box)

class Session:
    def __init__(self, req):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.cap = None
        self.face = None
        self.body = None
        self.hand = None
        self.id = uuid.uuid4().hex
        self.reference = BrioPose.loads(req.reference)
        if self.reference.get('j_kao') is None:
            raise ValueError('角色没有 j_kao 头部骨骼')
        self.race, self.alpha = req.race, req.smoothing
        self.mouth = MouthControls(**req.model_dump(include=set(MouthControls.model_fields)))
        self.upper = CameraBody(self.reference, self.race, self.alpha) if req.camera_pose else UpperBody(self.reference, self.alpha)
        self.fingers = LiveHands(self.upper.rest if req.camera_pose else self.reference, self.alpha)
        self.neutral = None
        self.previous = None
        self.last = time.monotonic()
        self.seq = 0
        self.calibration = None
        self.face_supported = face_rig.match(self.race, self.reference) is not None
        try:
            check_camera(req, cameras())
            self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=settings.face_model.read_bytes()),
                running_mode=vision.RunningMode.VIDEO, num_faces=1,
                output_face_blendshapes=True, output_facial_transformation_matrixes=True))
            self.cap = cv2.VideoCapture(req.camera, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise ValueError('无法打开所选摄像头。请停止其他 Live 会话，关闭占用摄像头的软件，并检查 Windows 摄像头权限。')
            model = settings.face_model.parent.parent / 'pose' / 'pose_landmarker_lite.task'
            self.body = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=model.read_bytes()),
                running_mode=vision.RunningMode.VIDEO, num_poses=1))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=settings.hand_model.read_bytes()),
                running_mode=vision.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=.6, min_hand_presence_confidence=.6))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception:
            self.close()
            raise

    def close(self):
        if self.cap is not None: self.cap.release()
        if self.face is not None: self.face.close()
        if self.body is not None: self.body.close()
        if self.hand is not None: self.hand.close()

    def frame(self):
        import cv2
        import mediapipe as mp
        self.last = time.monotonic()
        ok, bgr = self.cap.read()
        if not ok: raise ValueError('摄像头读取失败')
        preview = preview_jpeg(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.seq += 1
        stamp = max(int(time.monotonic() * 1000), getattr(self, 'stamp', 0) + 1)
        self.stamp = stamp
        result = self.face.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), stamp)
        body = self.body.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), stamp)
        overlay = tracking_overlay(result, body)
        overlay['framing'] = None
        if body.pose_landmarks:
            left, right = body.pose_landmarks[0][11], body.pose_landmarks[0][12]
            width = abs(left.x - right.x)
            if min(left.visibility, right.visibility) >= .65 and width > .04:
                overlay['framing'] = [(left.x+right.x)/2, (left.y+right.y)/2, width, rgb.shape[1]/rgb.shape[0]]
        hand = self.hand.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), stamp)
        gate = getattr(self, 'calibration', None)
        if gate is not None:
            ready, message = gate.update(hand, body.pose_landmarks[0] if body.pose_landmarks else None, time.monotonic())
            if not ready:
                return dict(sequence=self.seq, tracked=False, pose=None, bodyPose=None,
                            faceSupported=self.face_supported, preview=preview, calibrationMessage=message, **overlay)
            self.neutral = self.previous = None
            self.upper.reset()
            self.fingers.reset()
            self.calibration = None
            overlay['calibrationMessage'] = message
        body_pose = self.upper.solve(body.pose_world_landmarks[0]) if body.pose_world_landmarks else None
        combined = BrioPose.loads(body_pose) if body_pose else BrioPose({'Bones': {}})
        hands_tracked = self.fingers.solve(hand, body.pose_landmarks[0] if body.pose_landmarks else None, combined)
        torso_tracked = (bool(body.pose_world_landmarks) and self.upper.torso_tracked
                         if isinstance(self.upper, CameraBody)
                         else any(n.startswith('j_sebo_') for n in combined.bone_names))
        for landmarks in hand.hand_landmarks:
            for chain in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (5, 9, 10, 11, 12),
                          (9, 13, 14, 15, 16), (13, 17, 18, 19, 20), (0, 17)):
                for a, b in zip(chain, chain[1:]):
                    overlay['lines'].append([landmarks[a].x, landmarks[a].y, landmarks[b].x, landmarks[b].y])
        overlay['handsTracked'] = hands_tracked
        overlay['torsoTracked'] = torso_tracked
        overlay['fullPose'] = isinstance(self.upper, CameraBody)
        if isinstance(self.upper, CameraBody) and combined.bone_names:
            combined = self.upper.bake(combined)
        body_pose = combined.dumps() if combined.bone_names else None
        if body_pose is not None and not BrioPose.loads(body_pose).bone_names:
            body_pose = None
        if not result.face_landmarks or not result.facial_transformation_matrixes:
            # Keep the last applied rotation: reacquisition should blend from
            # the held pose instead of snapping directly to the new detection.
            return dict(sequence=self.seq, tracked=body_pose is not None, pose=None, bodyPose=body_pose, faceSupported=self.face_supported, preview=preview, **overlay)
        rotation = R.from_matrix(np.asarray(result.facial_transformation_matrixes[0])[:3, :3])
        if self.neutral is None: self.neutral = rotation
        delta = rotation * self.neutral.inv()
        if self.previous is not None:
            delta = self.previous * R.from_rotvec((self.previous.inv() * delta).as_rotvec() * self.alpha)
        self.previous = delta
        head = self.reference.get('j_kao')
        q = (delta * R.from_quat(head.rotation)).as_quat()
        if isinstance(self.upper, CameraBody):
            head = self.upper.head(rotation)
            q = head.rotation
            combined.raw['Bones'].pop('j_kao', None)
            combined.raw['Bones'].pop('j_kubi', None)
            combined = self.upper.bake(combined)
            body_pose = combined.dumps()
            head = combined.get('j_kao')
        out = BrioPose({'Bones': {}})
        out.set('j_kao', BoneTransform(head.position, q, head.scale))
        h, w = rgb.shape[:2]
        points = np.array([(p.x*w, p.y*h, p.z*w) for p in result.face_landmarks[0]])
        shapes = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
        bones, _, _ = face_rig.face_bones(self.race, self.reference,
            (head.position, q, head.scale), points, blendshapes=shapes, rotation=rotation.as_matrix(),
            strength=0.7 * self.mouth.face_strength,
            gains={**self.mouth.face_gains,
                   'mouth': self.mouth.face_gains.get('mouth', 1.) * self.mouth.mouth_strength,
                   'jaw': self.mouth.face_gains.get('jaw', 1.) * self.mouth.mouth_strength},
            mouth_limit=self.mouth.mouth_limit)
        for name, (p, r, s) in bones.items():
            if self.reference.get(name) is not None: out.set(name, BoneTransform(p, r, s))
        return dict(sequence=self.seq, tracked=True, pose=out.dumps(), bodyPose=body_pose, faceSupported=self.face_supported, preview=preview, **overlay)


def expire():
    global _session
    while True:
        time.sleep(1)
        with _lock:
            if _session is not None and time.monotonic() - _session.last > 8:
                _session.close()
                _session = None

threading.Thread(target=expire, daemon=True, name='live-camera-lease').start()

@router.post('/start')
def start(req: Start):
    global _session
    with _lock:
        if _session is not None: raise HTTPException(409, '已有 Live 会话，请先停止')
        try: _session = Session(req)
        except Exception as ex: raise HTTPException(400, str(ex)) from ex
        return dict(id=_session.id, faceSupported=_session.face_supported)

@router.post('/{sid}/frame')
def frame(sid: str, controls: MouthControls | None = None):
    global _session
    with _lock:
        if _session is None or _session.id != sid: raise HTTPException(404, 'Live 会话已结束')
        if controls is not None: _session.mouth = controls
        try:
            started = time.monotonic()
            result = _session.frame()
            result['frameMs'] = round((time.monotonic() - started) * 1000, 1)
            return result
        except Exception as ex:
            _session.close()
            _session = None
            raise HTTPException(400, str(ex)) from ex

@router.post('/{sid}/calibrate')
def calibrate(sid: str):
    with _lock:
        if _session is None or _session.id != sid: raise HTTPException(404, 'Live 会话已结束')
        _session.calibration = CalibrationGate()
        return {'ok': True}

@router.post('/{sid}/stop')
def stop(sid: str):
    global _session
    with _lock:
        if _session is not None and _session.id == sid:
            _session.close()
            _session = None
        return {'ok': True}

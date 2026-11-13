from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation as R
from fastapi import FastAPI
from fastapi.testclient import TestClient
from posegraph import live
from posegraph.live_body import UpperBody, swing
from posegraph.retarget.brio_pose import BrioPose


def reference():
    return BrioPose({'Bones': {'j_ude_a_l': {'Position': '0, 1, 0', 'Rotation': '0, 0, 0, 1', 'Scale': '1, 1, 1'}}})


def test_swing_opposite_and_identity():
    for target in ([1, 0, 0], [-1, 0, 0], [0, 1, 0]):
        np.testing.assert_allclose(swing([1, 0, 0], target).apply([1, 0, 0]), target, atol=1e-7)


def test_body_calibration_visibility_and_missing_bones():
    solver = UpperBody(reference(), 1)
    points = [SimpleNamespace(x=0., y=0., z=0., visibility=1.) for _ in range(33)]
    points[13].x = 1
    result = BrioPose.loads(solver.solve(points))
    assert result.bone_names == ['j_ude_a_l']
    np.testing.assert_allclose(result.get('j_ude_a_l').rotation, [0, 0, 0, 1])
    points[13].x, points[13].y = 0, 1
    result = BrioPose.loads(solver.solve(points))
    np.testing.assert_allclose(R.from_quat(result.get('j_ude_a_l').rotation).apply([-1, 0, 0]), [0, -1, 0], atol=1e-7)
    points[13].visibility = 0.1
    assert BrioPose.loads(solver.solve(points)).bone_names == []
    solver.reset()
    assert not solver.neutral


def test_session_ownership_and_release(monkeypatch):
    class Fake:
        def __init__(self, req):
            self.id, self.face_supported, self.closed = 'owned', False, False
        def close(self): self.closed = True
        def frame(self): raise ValueError('camera lost')
    monkeypatch.setattr(live, 'Session', Fake)
    monkeypatch.setattr(live, '_session', None)
    app = FastAPI()
    app.include_router(live.router)
    with TestClient(app) as client:
        args = dict(reference='{}', race='c0101')
        assert client.post('/live/start', json=args).status_code == 200
        session = live._session
        assert client.post('/live/start', json=args).status_code == 409
        assert client.post('/live/wrong/frame').status_code == 404
        client.post('/live/wrong/stop')
        assert not session.closed
        assert client.post('/live/owned/frame').status_code == 400
        assert session.closed and live._session is None
        assert client.post('/live/owned/stop').status_code == 200

def test_preview_is_bounded_jpeg():
    import base64
    import cv2
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :640, 2] = 255
    encoded = live.preview_jpeg(frame)
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (270, 480, 3)
    assert decoded[100, 50, 2] > 240
    assert decoded[100, 400, 2] < 10

def test_live_face_weights_validation_and_update(monkeypatch):
    import pytest
    from pydantic import ValidationError
    for invalid in ({'face_strength': 3}, {'face_gains': {'jaw': -1}}, {'face_gains': {'unknown': 1}}):
        with pytest.raises(ValidationError): live.MouthControls(**invalid)
    class Fake:
        id = 'weights'
        last = __import__('time').monotonic()
        mouth = None
        def frame(self): return self.mouth.model_dump()
    session = Fake()
    monkeypatch.setattr(live, '_session', session)
    app = FastAPI()
    app.include_router(live.router)
    with TestClient(app) as client:
        response = client.post('/live/weights/frame', json={'face_strength': 1.5, 'face_gains': {'jaw': 0.2, 'eyes': 1.2}})
        assert response.status_code == 200
        assert session.mouth.face_gains == {'jaw': 0.2, 'eyes': 1.2}
        assert session.mouth.face_strength == 1.5
        assert session.mouth.mouth_limit == 1

def test_head_reacquisition_blends_from_held_rotation(monkeypatch):
    s = live.Session.__new__(live.Session)
    s.cap = SimpleNamespace(read=lambda: (True, np.zeros((32, 32, 3), dtype=np.uint8)))
    missing = SimpleNamespace(face_landmarks=[], facial_transformation_matrixes=[])
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_euler('y', 90, degrees=True).as_matrix()
    found = SimpleNamespace(face_landmarks=[[SimpleNamespace(x=.5, y=.5, z=0)] * 478],
        facial_transformation_matrixes=[matrix], face_blendshapes=[[]])
    results = iter([missing, found])
    s.face = SimpleNamespace(detect_for_video=lambda *args: next(results))
    s.body = SimpleNamespace(detect_for_video=lambda *args: SimpleNamespace(pose_world_landmarks=[], pose_landmarks=[]))
    s.hand = SimpleNamespace(detect_for_video=lambda *args: SimpleNamespace(hand_landmarks=[]))
    s.fingers = SimpleNamespace(solve=lambda *args: [])
    s.upper = SimpleNamespace()
    s.seq, s.face_supported, s.race, s.alpha = 0, False, 'c1101', .5
    s.neutral = R.identity()
    s.previous = R.from_euler('y', 30, degrees=True)
    s.mouth = live.MouthControls()
    s.reference = BrioPose({'Bones': {'j_kao': {'Position': '0, 1, 0', 'Rotation': '0, 0, 0, 1', 'Scale': '1, 1, 1'}}})
    monkeypatch.setattr(live.face_rig, 'face_bones', lambda *a, **kw: ({}, None, {}))
    assert not s.frame()['tracked']
    np.testing.assert_allclose(s.previous.as_euler('xyz', degrees=True), [0, 30, 0], atol=1e-6)
    pose = BrioPose.loads(s.frame()['pose'])
    np.testing.assert_allclose(R.from_quat(pose.get('j_kao').rotation).as_euler('xyz', degrees=True), [0, 60, 0], atol=1e-6)

def test_camera_reorder_is_rejected():
    import pytest
    request = live.Start(reference='{}', race='c1101', camera=0, camera_name='HIK', camera_path='usb-a')
    live.check_camera(request, [{'index': 0, 'name': 'HIK', 'path': 'usb-a'}])
    for devices in ([], [{'index': 0, 'name': 'OBS', 'path': ''}], [{'index': 0, 'name': 'HIK', 'path': 'usb-b'}]):
        with pytest.raises(ValueError): live.check_camera(request, devices)


def test_overlay_filters_occluded_limbs_and_clears_missing_face():
    face = SimpleNamespace(face_landmarks=[[SimpleNamespace(x=.2,y=.3),SimpleNamespace(x=.6,y=.8)]])
    points = [SimpleNamespace(x=.4, y=.5, visibility=1.) for _ in range(33)]
    points[13].visibility = .1
    result = live.tracking_overlay(face, SimpleNamespace(pose_landmarks=[points]))
    assert result['faceBox'] == [.2,.3,.6,.8]
    assert len(result['lines']) == 6
    assert live.tracking_overlay(SimpleNamespace(face_landmarks=[]), SimpleNamespace(pose_landmarks=[])) == {'lines': [], 'faceBox': None}

def test_torso_rotation_is_distributed_and_hidden_torso_is_held():
    raw = {'Position': '0, 1, 0', 'Rotation': '0, 0, 0, 1', 'Scale': '1, 1, 1'}
    solver = UpperBody(BrioPose({'Bones': {n: dict(raw) for n in ['j_sebo_a','j_sebo_b','j_sebo_c']}}), 1)
    xyz = np.zeros((33,3)); xyz[11]=[-1,-1,0]; xyz[12]=[1,-1,0]; xyz[23]=[-.5,0,0]; xyz[24]=[.5,0,0]
    def points(x): return [SimpleNamespace(x=p[0],y=p[1],z=p[2],visibility=1.) for p in x]
    solver.solve(points(xyz))
    pose=BrioPose.loads(solver.solve(points(R.from_euler('y',30,degrees=True).apply(xyz))))
    angles=[R.from_quat(pose.get(n).rotation).magnitude() for n in ['j_sebo_a','j_sebo_b','j_sebo_c']]
    assert 0 < angles[0] < angles[1] < angles[2] < 1.1
    hidden=points(xyz); hidden[23].visibility=0
    assert BrioPose.loads(solver.solve(hidden)).bone_names == []


def test_fingers_support_ivcs_and_ignore_wrong_wrist():
    from posegraph.live_hands import LiveHands
    raw={'Position':'0, 0, 0','Rotation':'0, 0, 0, 1','Scale':'1, 1, 1'}
    ref=BrioPose({'Bones':{n:dict(raw) for n in ['j_te_l','j_hito_a_l','iv_hito_c_l']}})
    solver=LiveHands(ref,1)
    xyz=np.zeros((21,3)); xyz[5]=[1,1,0]; xyz[9]=[0,1,0]; xyz[17]=[-1,1,0]
    xyz[6]=[1,2,0];xyz[7]=[1,3,0];xyz[8]=[1,4,0]
    world=[SimpleNamespace(x=p[0],y=p[1],z=p[2]) for p in xyz]
    hand=[SimpleNamespace(x=.2,y=.2,z=0) for _ in range(21)]
    body=[SimpleNamespace(x=.9,y=.9,visibility=1) for _ in range(33)]
    body[15]=SimpleNamespace(x=.2,y=.2,visibility=1)
    result=SimpleNamespace(hand_landmarks=[hand],hand_world_landmarks=[world])
    out=BrioPose({'Bones':{}});solver.solve(result,body,out)
    assert 'iv_hito_c_l' in out.bone_names and 'j_te_r' not in out.bone_names
    world[8].z=1
    out=BrioPose({'Bones':{}});solver.solve(result,body,out)
    assert R.from_quat(out.get('iv_hito_c_l').rotation).magnitude() > .1
    body[15].visibility=0
    out=BrioPose({'Bones':{}});solver.solve(result,body,out)
    assert out.bone_names == []

def test_camera_pose_ignores_previous_head_twist_and_maps_seated_thigh():
    from posegraph.live_camera_body import CameraBody
    from posegraph.retarget.ffxiv import load_ffxiv_skeleton, forward_kinematics
    from posegraph.retarget.brio_pose import BoneTransform
    sk=load_ffxiv_skeleton('c1101');rots,pos=forward_kinematics(sk)
    ref=BrioPose({'Bones':{}})
    for i,n in enumerate(sk.names):ref.set(n,BoneTransform(pos[i],rots[i].as_quat(),np.ones(3)))
    old=ref.get('j_kao');old.rotation=(R.from_euler('y',120,degrees=True)*R.from_quat(old.rotation)).as_quat();ref.set('j_kao',old)
    solver=CameraBody(ref,'c1101',1)
    head=solver.head(R.identity())
    np.testing.assert_allclose(head.rotation,rots[sk.index('j_kao')].as_quat(),atol=1e-7)
    points=[SimpleNamespace(x=0.,y=0.,z=0.,visibility=0.) for _ in range(33)]
    points[23]=SimpleNamespace(x=0.,y=0.,z=0.,visibility=1.)
    points[25]=SimpleNamespace(x=0.,y=0.,z=-1.,visibility=1.)
    pose=BrioPose.loads(solver.solve(points))
    direction=pose.get('j_asi_b_l').position-pose.get('j_asi_a_l').position
    np.testing.assert_allclose(direction/np.linalg.norm(direction),[0,0,1],atol=1e-6)

def test_camera_body_does_not_overwrite_jaw_hair_or_ears():
    from posegraph.live_camera_body import CameraBody
    from posegraph.retarget.ffxiv import load_ffxiv_skeleton, forward_kinematics
    from posegraph.retarget.brio_pose import BoneTransform
    sk=load_ffxiv_skeleton('c1101'); rotations,positions=forward_kinematics(sk)
    ref=BrioPose({'Bones':{}})
    for i,name in enumerate(sk.names):
        ref.set(name,BoneTransform(positions[i],rotations[i].as_quat(),np.ones(3)))
    jaw=ref.get('j_ago');jaw.position += [.03,.02,.01];ref.set('j_ago',jaw)
    solver=CameraBody(ref,'c1101',1)
    solver.head(R.from_euler('y',30,degrees=True))
    pose=solver.bake()
    assert 'j_kao' in pose.bone_names
    assert 'j_sebo_c' in pose.bone_names
    assert not set(pose.bone_names) & {'j_ago','j_kami_a','j_mimi_l','j_mimi_r'}
    np.testing.assert_allclose(ref.get('j_ago').position,jaw.position)

def test_calibration_requires_two_upright_stable_hands():
    from posegraph.live_calibration import CalibrationGate
    body=[SimpleNamespace(x=.5,y=.5,visibility=1.) for _ in range(33)]
    palms=[]
    for x,wrist in ((.3,15),(.7,16)):
        h=[SimpleNamespace(x=x,y=.4,z=0.) for _ in range(21)]
        h[0].y=.55
        h[5].x=x+.04;h[17].x=x-.04
        for tip in (8,12,16,20):h[tip].y=.2
        palms.append(h)
        body[wrist]=SimpleNamespace(x=x,y=.55,visibility=1.)
    gate=CalibrationGate()
    hands=SimpleNamespace(hand_landmarks=palms)
    assert not gate.update(hands,body,0)[0]
    assert gate.update(hands,body,.7)[0]
    palms[0][8].y=.5
    assert not gate.update(hands,body,.8)[0]
    assert not gate.update(SimpleNamespace(hand_landmarks=[palms[1]]),body,2)[0]

def test_upright_hand_matches_camera_on_first_frame_and_after_calibration():
    from posegraph.live_hands import LiveHands, basis
    from posegraph.retarget.ffxiv import load_ffxiv_skeleton, forward_kinematics
    from posegraph.retarget.brio_pose import BoneTransform
    sk=load_ffxiv_skeleton('c1101'); rotations, positions=forward_kinematics(sk)
    ref=BrioPose({'Bones':{}})
    for i,n in enumerate(sk.names):ref.set(n,BoneTransform(positions[i],rotations[i].as_quat(),np.ones(3)))
    for side,wrist_idx in [('l',15),('r',16)]:
        solver=LiveHands(ref,1.)
        rest=np.zeros((21,3))
        for i,n in {0:f'j_te_{side}',5:f'j_hito_a_{side}',9:f'j_naka_a_{side}',17:f'j_ko_a_{side}'}.items():rest[i]=ref.get(n).position
        # An upright palm, with its long axis pointing up, regardless of bind pose.
        observed_frame=R.identity()
        delta=observed_frame*basis(rest).inv()
        camera=delta.apply(rest-rest[0])*[1,-1,-1]
        hand=[SimpleNamespace(x=.3,y=.4,z=0.) for _ in range(21)]
        result=SimpleNamespace(hand_landmarks=[hand],hand_world_landmarks=[[SimpleNamespace(x=p[0],y=p[1],z=p[2]) for p in camera]])
        body=[SimpleNamespace(x=.9,y=.9,visibility=1.) for _ in range(33)]
        body[wrist_idx]=SimpleNamespace(x=.3,y=.4,visibility=1.)
        for reset in (False,True):
            if reset:solver.reset()
            out=BrioPose({'Bones':{}});solver.solve(result,body,out)
            actual=R.from_quat(out.get(f'j_te_{side}').rotation)*R.from_quat(ref.get(f'j_te_{side}').rotation).inv()
            actual_palm=actual*basis(rest)
            np.testing.assert_allclose(actual_palm.as_matrix(),observed_frame.as_matrix(),atol=1e-6)

def test_camera_spine_bends_progressively_and_rejects_collapsed_tracking():
    from posegraph.live_camera_body import CameraBody
    from posegraph.retarget.ffxiv import load_ffxiv_skeleton,forward_kinematics
    from posegraph.retarget.brio_pose import BoneTransform
    sk=load_ffxiv_skeleton('c1101');rot,pos=forward_kinematics(sk)
    ref=BrioPose({'Bones':{}})
    for i,n in enumerate(sk.names):ref.set(n,BoneTransform(pos[i],rot[i].as_quat(),np.ones(3)))
    solver=CameraBody(ref,'c1101',1)
    xyz=np.zeros((33,3));xyz[23]=[.15,0,0];xyz[24]=[-.15,0,0]
    xyz[11]=[.2,.5,.3];xyz[12]=[-.2,.5,.3]
    points=[SimpleNamespace(x=v[0],y=-v[1],z=-v[2],visibility=1. if i in (11,12,23,24) else 0.) for i,v in enumerate(xyz)]
    out=BrioPose.loads(solver.solve(points))
    assert solver.torso_tracked
    angles=[]
    for n in ('j_kosi','j_sebo_a','j_sebo_b','j_sebo_c'):
        angles.append((R.from_quat(out.get(n).rotation)*rot[sk.index(n)].inv()).magnitude())
    assert angles[0] < 1e-7
    assert 0 < angles[1] < angles[2] < angles[3] < np.deg2rad(60)
    held=solver.targets['j_sebo_c'].as_quat().copy()
    points[11]=points[12]
    solver.solve(points)
    assert not solver.torso_tracked
    np.testing.assert_allclose(solver.targets['j_sebo_c'].as_quat(),held)
    name='j_sebo_c';before=solver.targets[name]
    solver.target(name,R.from_euler('y',170,degrees=True)*before)
    assert (before.inv()*solver.targets[name]).magnitude() <= np.deg2rad(20)+1e-7

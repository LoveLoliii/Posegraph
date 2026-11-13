"""Calibrated hand rotations, associated with body wrists rather than handedness labels."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from .live_body import swing
from .retarget.brio_pose import BoneTransform

FINGERS = [('oya', [1, 2, 3, 4]), ('hito', [5, 6, 7, 8]),
           ('naka', [9, 10, 11, 12]), ('kusu', [13, 14, 15, 16]), ('ko', [17, 18, 19, 20])]

def basis(points):
    x = points[5] - points[17]
    y = points[9] - points[0]
    if np.linalg.norm(x) < 1e-6 or np.linalg.norm(np.cross(x, y)) < 1e-8: return None
    x /= np.linalg.norm(x)
    z = np.cross(x, y); z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return R.from_matrix(np.column_stack([x, y, z]))

class LiveHands:
    def __init__(self, reference, alpha):
        self.reference, self.alpha = reference, alpha
        self.neutral, self.previous = {}, {}
        self.reference_palms = {}
        for side in ('l', 'r'):
            names = {0: f'j_te_{side}', 5: f'j_hito_a_{side}', 9: f'j_naka_a_{side}', 17: f'j_ko_a_{side}'}
            if all(reference.get(name) is not None for name in names.values()):
                points = np.zeros((21, 3))
                for index, name in names.items(): points[index] = reference.get(name).position
                frame = basis(points)
                if frame is not None: self.reference_palms[side] = frame

    def reset(self):
        self.neutral.clear(); self.previous.clear()

    def solve(self, result, body_points, out):
        if not body_points: return []
        candidates = []
        for i, hand in enumerate(result.hand_landmarks):
            for side, wrist in [('l', 15), ('r', 16)]:
                p = body_points[wrist]
                if p.visibility < .65: continue
                d = np.hypot(hand[0].x - p.x, hand[0].y - p.y)
                if d < .16: candidates.append((d, i, side))
        used_hands, used_sides, tracked = set(), set(), set()
        for _, index, side in sorted(candidates):
            if index in used_hands or side in used_sides: continue
            used_hands.add(index); used_sides.add(side)
            wrist = self.reference.get(f'j_te_{side}')
            if wrist is None: continue
            # Same camera -> model coordinates as CameraBody; reflecting x as
            # well inverted the hand relative to the arm.
            points = np.array([(p.x, p.y, p.z) for p in result.hand_world_landmarks[index]]) * [1, -1, -1]
            palm = basis(points)
            if palm is None: continue
            local = palm.inv().apply(points - points[0])
            initial_palm, initial = self.neutral.setdefault(side, (palm, local.copy()))
            # Align the actor's palm frame with the observed palm immediately.
            # Calibration changes finger neutral only, never makes an upright
            # observed palm map back to a horizontal bind-pose wrist.
            reference_palm = self.reference_palms.get(side, initial_palm)
            wrist_delta = palm * reference_palm.inv()
            target_wrist = wrist_delta * R.from_quat(wrist.rotation)
            self._set(out, f'j_te_{side}', target_wrist)
            tracked.add(side)
            for name, chain in FINGERS:
                for j, (a, b) in enumerate(zip(chain, chain[1:])):
                    bone_name = f'{"iv" if j == 2 else "j"}_{name}_{"abc"[j]}_{side}'
                    bone = self.reference.get(bone_name)
                    if bone is None: continue
                    old, new = initial[b] - initial[a], local[b] - local[a]
                    if min(np.linalg.norm(old), np.linalg.norm(new)) < 1e-6: continue
                    delta = swing(old, new)
                    # Conservative swing cap; no inferred axial finger twist.
                    vector = delta.as_rotvec(); angle = np.linalg.norm(vector)
                    limit = np.deg2rad(80 if name == 'oya' else 110)
                    if angle > limit: delta = R.from_rotvec(vector * limit / angle)
                    world_delta = reference_palm * delta * reference_palm.inv()
                    self._set(out, bone_name, wrist_delta * world_delta * R.from_quat(bone.rotation))
        return sorted(tracked)

    def _set(self, out, name, target):
        bone = self.reference.get(name)
        previous = self.previous.get(name, R.from_quat(bone.rotation))
        current = previous * R.from_rotvec((previous.inv() * target).as_rotvec() * self.alpha)
        self.previous[name] = current
        out.set(name, BoneTransform(bone.position, current.as_quat(), bone.scale))

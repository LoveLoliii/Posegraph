"""Relative, calibrated upper-body rotations; no inferred bone lengths or root motion."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from .retarget.brio_pose import BrioPose, BoneTransform

SEGMENTS = [('j_ude_a_l', 11, 13), ('j_ude_b_l', 13, 15),
            ('j_ude_a_r', 12, 14), ('j_ude_b_r', 14, 16)]

def swing(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    dot = np.clip(a @ b, -1, 1)
    axis = np.cross(a, b)
    if np.linalg.norm(axis) < 1e-8:
        if dot > 0: return R.identity()
        axis = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
    return R.from_rotvec(axis / np.linalg.norm(axis) * np.arccos(dot))

class UpperBody:
    def __init__(self, reference, alpha):
        self.reference, self.alpha = reference, alpha
        self.neutral, self.previous = {}, {}

    def reset(self):
        self.neutral.clear()
        self.previous.clear()

    def solve(self, points):
        out = BrioPose({'Bones': {}})
        if min(points[i].visibility for i in (11, 12, 23, 24)) >= .65:
            xyz = np.array([(p.x, p.y, p.z) for p in points]) * [-1, -1, -1]
            x = xyz[11] - xyz[12]
            y = (xyz[11] + xyz[12] - xyz[23] - xyz[24]) / 2
            z = np.cross(x, y)
            if np.linalg.norm(x) > 1e-6 and np.linalg.norm(z) > 1e-8:
                x /= np.linalg.norm(x); z /= np.linalg.norm(z)
                current = R.from_matrix(np.column_stack([x, np.cross(z, x), z]))
                initial = self.neutral.setdefault('torso', current)
                delta = current * initial.inv()
                rv = delta.as_rotvec()
                angle = np.linalg.norm(rv)
                if angle > np.deg2rad(60): rv *= np.deg2rad(60) / angle
                for name, weight in [('j_sebo_a', .33), ('j_sebo_b', .66), ('j_sebo_c', 1.)]:
                    bone = self.reference.get(name)
                    if bone is None: continue
                    target = R.from_rotvec(rv * weight)
                    previous = self.previous.get(name, R.identity())
                    target = previous * R.from_rotvec((previous.inv() * target).as_rotvec() * self.alpha)
                    self.previous[name] = target
                    out.set(name, BoneTransform(bone.position, (target * R.from_quat(bone.rotation)).as_quat(), bone.scale))
        for name, a, b in SEGMENTS:
            bone = self.reference.get(name)
            if bone is None or min(points[a].visibility, points[b].visibility) < 0.65: continue
            # MediaPipe camera axes to game model axes: x left, y up, z forward.
            v = (np.array([points[b].x, points[b].y, points[b].z]) -
                 np.array([points[a].x, points[a].y, points[a].z])) * [-1, -1, -1]
            if not np.isfinite(v).all() or np.linalg.norm(v) < 1e-6: continue
            initial = self.neutral.setdefault(name, v)
            delta = swing(initial, v)
            previous = self.previous.get(name, R.identity())
            delta = previous * R.from_rotvec((previous.inv() * delta).as_rotvec() * self.alpha)
            self.previous[name] = delta
            out.set(name, BoneTransform(bone.position, (delta * R.from_quat(bone.rotation)).as_quat(), bone.scale))
        return out.dumps()

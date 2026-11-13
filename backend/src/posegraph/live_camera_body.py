"""Absolute camera pose retargeting using the same race skeleton/FK as video.
World placement is deliberately separate from model-space posture.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from .retarget.ffxiv import load_ffxiv_skeleton, _with_reference_lengths, forward_kinematics
from .retarget.brio_pose import BrioPose, BoneTransform
from .live_body import swing

class CameraBody:
    def __init__(self, reference, race, alpha):
        self.reference, self.alpha = reference, alpha
        self.sk = _with_reference_lengths(load_ffxiv_skeleton(race), reference)
        self.bind_rot, self.bind_pos = forward_kinematics(self.sk)
        # The body skeleton contains generic jaw/hair/ear bones, not the
        # actor's actual face rig. Never export these as absolute transforms.
        self.head_children = set()
        head = self.sk.index('j_kao')
        for i in range(len(self.sk.names)):
            if self.sk.parents[i] == head or self.sk.parents[i] in self.head_children:
                self.head_children.add(i)
        self.targets = {}
        self.rest = BrioPose({'Bones': {}})
        for i, name in enumerate(self.sk.names):
            b = reference.get(name)
            if b is not None: self.rest.set(name, BoneTransform(self.bind_pos[i], self.bind_rot[i].as_quat(), b.scale))
        self.last = None
        self.torso_tracked = False

    def reset(self):
        # Calibration does not redefine a seated pose as a standing neutral pose.
        self.targets.clear()

    def target(self, name, rotation):
        if self.reference.get(name) is None or name not in self.sk.names: return
        old = self.targets.get(name, rotation)
        change = (old.inv() * rotation).as_rotvec()
        # Reject non-finite input and bound sudden tracker flips after acquisition.
        if not np.isfinite(change).all(): return
        angle = np.linalg.norm(change)
        if name in self.targets and angle > np.deg2rad(20):
            change *= np.deg2rad(20) / angle
        self.targets[name] = old * R.from_rotvec(change * self.alpha)

    def solve(self, points):
        xyz = np.array([(p.x,p.y,p.z) for p in points]) * [1,-1,-1]
        self.torso_tracked = False
        if min(points[i].visibility for i in (11,12,23,24)) >= .65:
            x = xyz[11]-xyz[12]
            up = (xyz[11]+xyz[12]-xyz[23]-xyz[24])/2
            z = np.cross(x,up)
            hips = xyz[23]-xyz[24]
            hips[1] = 0
            if (np.isfinite(xyz[[11,12,23,24]]).all() and np.linalg.norm(x) > .05
                    and np.linalg.norm(up) > .05 and np.linalg.norm(hips) > .05
                    and np.linalg.norm(z) > .2 * np.linalg.norm(x) * np.linalg.norm(up)):
                x /= np.linalg.norm(x); z /= np.linalg.norm(z)
                torso = R.from_matrix(np.column_stack([x,np.cross(z,x),z]))
                hips /= np.linalg.norm(hips)
                pelvis = R.from_matrix(np.column_stack([hips, [0,1,0], np.cross(hips,[0,1,0])]))
                bend = (pelvis.inv()*torso).as_rotvec()
                angle = np.linalg.norm(bend)
                if angle > np.deg2rad(60): bend *= np.deg2rad(60)/angle
                for name,weight in (('j_kosi',0),('j_sebo_a',.33),('j_sebo_b',.66),('j_sebo_c',1),('j_sako_l',1),('j_sako_r',1)):
                    if name in self.sk.names:
                        self.target(name,pelvis*R.from_rotvec(bend*weight)*self.bind_rot[self.sk.index(name)])
                self.torso_tracked = True
        segments = [('j_ude_a','j_ude_b',11,13),('j_ude_b','j_te',13,15),
                    ('j_asi_a','j_asi_b',23,25),('j_asi_c','j_asi_d',25,27)]
        for side,offset in [('l',0),('r',1)]:
            for parent,child,a,b in segments:
                name,tip=f'{parent}_{side}',f'{child}_{side}'
                if min(points[a+offset].visibility,points[b+offset].visibility)<.65: continue
                if name not in self.sk.names or tip not in self.sk.names: continue
                direction=xyz[b+offset]-xyz[a+offset]
                i,j=self.sk.index(name),self.sk.index(tip)
                rest=self.bind_pos[j]-self.bind_pos[i]
                if not np.isfinite(direction).all() or min(np.linalg.norm(direction),np.linalg.norm(rest))<1e-6: continue
                self.target(name,swing(rest,direction)*self.bind_rot[i])
        self.last = self.bake()
        return self.last.dumps()

    def head(self, rotation):
        i=self.sk.index('j_kao')
        # Face matrix is absolute in camera axes, not relative to the actor's old head pose.
        self.target('j_kao', rotation*self.bind_rot[i])
        if 'j_kubi' in self.sk.names:
            k=self.sk.index('j_kubi')
            chest=self.targets.get('j_sebo_c',self.bind_rot[self.sk.index('j_sebo_c')])
            chest_delta=chest*self.bind_rot[self.sk.index('j_sebo_c')].inv()
            neck_delta=chest_delta*R.from_rotvec((chest_delta.inv()*rotation).as_rotvec()*.5)
            self.target('j_kubi',neck_delta*self.bind_rot[k])
        self.last=self.bake()
        return self.last.get('j_kao')

    def bake(self, additions=None):
        targets=dict(self.targets)
        if additions is not None:
            for name in additions.bone_names:
                if name in self.sk.names: targets[name]=R.from_quat(additions.get(name).rotation)
        given=[targets.get(name) for name in self.sk.names]
        rotations,positions=forward_kinematics(self.sk,given)
        out=BrioPose({'Bones':{}})
        for i,name in enumerate(self.sk.names):
            b=self.reference.get(name)
            if b is None or i in self.head_children or name.startswith('j_f_') or not name.startswith(('j_','iv_')): continue
            out.set(name,BoneTransform(positions[i],rotations[i].as_quat(),b.scale))
        return out

"""A stable, two-hand upright open-hand calibration gate.
Palm-facing instruction is shown to the user: a monocular landmark model cannot
reliably distinguish palm/back in all views, so we check frontal orientation only.
"""
import numpy as np

class CalibrationGate:
    def __init__(self):
        self.since = None
        self.previous = None

    def update(self, hands, body, now):
        if len(hands.hand_landmarks) != 2 or not body:
            self.since = None
            return False, '请双手向前举起，手指向上、掌心朝摄像头；双手和手腕须完整入镜'
        wrists = []
        matched = set()
        for hand in hands.hand_landmarks:
            if any(not (0.02 < p.x < .98 and .02 < p.y < .98) for p in hand):
                self.since = None
                return False, '请让双手完整入镜，不要贴近画面边缘'
            p = np.array([(v.x, v.y, v.z) for v in hand])
            length = np.linalg.norm(p[9, :2] - p[0, :2])
            fingers = all(p[tip,1] < p[base,1] - length * .3 for base,tip in ((5,8),(9,12),(13,16),(17,20)))
            width = abs(p[5,0]-p[17,0])
            if length < .025 or not fingers or width < length * .3:
                self.since = None
                return False, '请张开手指并指向上方，掌心正对摄像头，不要侧掌'
            candidates = [(np.hypot(hand[0].x-body[i].x, hand[0].y-body[i].y), i) for i in (15,16) if body[i].visibility >= .65]
            if not candidates or min(candidates)[0] > .16 or min(candidates)[1] in matched:
                self.since = None
                return False, '请分开双手，让身体手腕与手掌同时入镜'
            matched.add(min(candidates)[1])
            wrists.append((min(candidates)[1], p[0,:2]))
        positions = np.array([p for _,p in sorted(wrists,key=lambda x:x[0])])
        if self.previous is None or np.max(np.abs(positions-self.previous)) > .025:
            self.since = now
        self.previous = positions
        if self.since is None: self.since = now
        ready = now-self.since >= .6
        return ready, '校准完成' if ready else '姿势已识别，请保持双手稳定片刻…'

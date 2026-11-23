"""Push a block in short strokes, using a fresh head-camera measurement between strokes."""
import numpy as np

from .assets import TABLE_Z
from .bench import Bench
from .policy import HOME


class FeedbackPush(Bench):
    max_strokes = 4
    stroke_length = .025
    tip_offset_x = -.003
    push_height = .016
    standoff = .022
    contact_gap = .014
    stop_error = .011
    max_ticks = 960

    def __init__(self, controller, rng, color, camera, reobserve=True):
        super().__init__(controller, rng, color, camera, 'push_feedback')
        self.stage = 'inspect'
        self.strokes = 0
        self.pad = None
        self.path = []
        self.reason = None
        self.reobserve = reobserve
        self.planned_pick = None

    def retract(self):
        self.angles += np.clip(HOME[5:8] - self.angles, -.03, .03)

    def finish(self, reason):
        self.reason = reason
        self.advance('release')

    def inspect(self):
        self.retract()
        if self.age < 42 or self.age % 5:
            return
        fresh = self.reobserve or self.planned_pick is None
        if fresh:
            self.seen.clear()
            self.observe()
            if self.pad is None and 'green' in self.seen:
                self.pad = self.seen['green'].copy()
            pick = self.seen.get(self.color)
        else:
            pick = self.planned_pick
        if pick is None or self.pad is None:
            if self.age >= 65:
                self.finish('lost_target')
            return
        delta = self.pad-pick
        distance = np.linalg.norm(delta)
        self.path.append({'t':round(self.d.time,2), 'xy_mm':(1000*pick).round(2).tolist(),
                          'source':'rgb' if fresh else 'planned', 'estimated_pad_error_mm':round(1000*float(distance),2)})
        if distance <= self.stop_error or self.strokes >= self.max_strokes:
            if fresh:
                self.final_seen = float(distance)
            self.finish('in_zone' if distance <= self.stop_error else 'stroke_budget')
            return
        direction = delta/distance
        self.strokes += 1
        offset = np.array([self.tip_offset_x,0.])
        self.start = pick - self.standoff*direction + offset
        self.planned_pick = pick + min(self.stroke_length,distance)*direction
        self.stop = self.planned_pick - self.contact_gap*direction + offset
        self.advance('align')

    def plan(self):
        self.age += 1
        if self.stage == 'inspect':
            self.inspect()
            return
        if self.stage == 'release':
            self.retract()
            return
        if self.stage == 'align':
            xy,height = self.start,.036
            if self.age >= 49:
                self.advance('descend')
        elif self.stage == 'descend':
            xy,height = self.start,self.push_height
            if self.age >= 34:
                self.advance('stroke')
        elif self.stage == 'stroke':
            u = min(self.age/63,1.)
            xy = self.start + (self.stop-self.start)*u
            height = self.push_height
            if self.age >= 66:
                self.advance('lift_clear')
        elif self.stage == 'lift_clear':
            xy,height = self.stop,.041
            if self.age >= 25:
                self.advance('inspect')
        else:
            raise ValueError(f'Unknown push phase {self.stage}')
        self.inverse_kinematics(np.array([*xy,TABLE_Z+height]))

    def result(self):
        result = super().result()
        result.update(strokes=self.strokes, observations=self.path, stop_reason=self.reason or 'timeout',
                      reobserve=self.reobserve)
        if self.path:
            initial_mm = np.asarray(self.path[0]['xy_mm'])
            result['initial_error_mm'] = round(float(np.linalg.norm(initial_mm - 1000*self.initial[:2])),2)
        return result

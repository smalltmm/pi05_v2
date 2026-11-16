"""Right-only observation pose after donor release; no hardware access."""

import json
from pathlib import Path

import numpy as np

from .config import PIPER_JOINT_MIN, PIPER_JOINT_MAX


DEFAULT_REFERENCE = Path(__file__).parent / 'calibration/items_handover_right_observation_20261113_093940.json'
DEFAULT_MODEL_REFERENCE = Path(__file__).parent / 'calibration/items_handover_model_right_waiting_10k.json'


class HandoverObservation:
    def __init__(self, reference, speed, hz, model_reference=DEFAULT_MODEL_REFERENCE):
        data = json.loads(Path(reference).read_text())
        if data.get('task') != 'items_handover' or data.get('pose_role') != 'right_observation':
            raise ValueError('invalid handover right observation reference')
        self.goal = np.asarray(data['command_right_joints_rad'], dtype=np.float32)
        if (self.goal.shape != (6,) or not np.isfinite(self.goal).all()
                or np.any(self.goal < PIPER_JOINT_MIN) or np.any(self.goal > PIPER_JOINT_MAX)):
            raise ValueError('right observation joints exceed limits')
        self.step = np.deg2rad(speed) / hz
        model_data = json.loads(Path(model_reference).read_text())
        if model_data.get('task') != 'items_handover' or model_data.get('pose_role') != 'model_right_waiting':
            raise ValueError('invalid handover model waiting reference')
        self.model_reference = str(model_reference)
        self.model_right = np.asarray(model_data['right_state_7'], dtype=np.float32)
        if (self.model_right.shape != (7,) or not np.isfinite(self.model_right).all()
                or not 0 <= self.model_right[6] <= 1):
            raise ValueError('invalid handover model right state')
        self.last = None
        self.phase = 'idle'

    def model_state(self, physical):
        result = np.asarray(physical, dtype=np.float32).copy()
        if self.phase != 'idle':
            result[7:14] = self.model_right
        return result

    def apply(self, physical, target, *, completed):
        if self.phase == 'idle':
            if not completed:
                return target
            self.last = np.asarray(physical[7:13], dtype=np.float32).copy()
            self.phase = 'moving'
        result = np.asarray(target).copy()
        result[7:13] = self.last + np.clip(self.goal - self.last, -self.step, self.step)
        result[13] = 1.  # Keep the donor open after transfer.
        return result

    def commit(self, sent):
        if self.phase == 'idle':
            return
        self.last = np.asarray(sent[7:13]).copy()
        self.phase = 'holding' if np.allclose(self.last, self.goal, atol=1e-6) else 'moving'

    def record(self):
        return dict(phase=self.phase, target_right_joints_rad=self.goal.tolist(),
                    model_state_overridden=self.phase != 'idle', model_right_state=self.model_right.tolist(),
                    model_state_reference=self.model_reference)


def add_arguments(parser):
    parser.add_argument('--handover-right-observe', action='store_true',
                        help='items_handover: move right arm to recorded observation pose after transfer')
    parser.add_argument('--handover-right-observe-reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--handover-right-observe-speed-deg-s', type=float, default=30.)
    parser.add_argument('--handover-right-observe-model-reference', type=Path, default=DEFAULT_MODEL_REFERENCE,
                        help='right waiting state supplied to model while physical right arm observes')


def validate_args(args):
    if args.handover_right_observe:
        if args.task != 'items_handover' or not args.handover_assist or args.relative_handover:
            raise ValueError('--handover-right-observe requires items_handover --handover-assist without --relative-handover')
        if not np.isfinite(args.handover_right_observe_speed_deg_s) or args.handover_right_observe_speed_deg_s <= 0:
            raise ValueError('handover right observation speed must be finite and positive')

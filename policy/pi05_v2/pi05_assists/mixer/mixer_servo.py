"""mixer_operating: left-arm visual servo for the toggle paddle, after the model's approach.

The usual trigger waits for the left gripper to close and the paddle's dark
slot to approach the fingertip in the left wrist camera. The real profile can
first wait for right pickup and a placement pose (or opening) and move the left arm to a saved
preparation pose, keeping left stationary during the right-arm model stage.
The servo then owns both arms
(right arm holds), aligns the slot to the contact pixel with image-based servoing,
inserts along the approach axis, strokes along link6 local +X, and retreats.

Geometry is taken from RoboSynChallenge/cobotmagic_Real_mixer_operating (60 episodes):
at contact the slot centroid is near (368, 346) px; the press is mainly +X local
(median 24.5 mm, IQR 12-30 mm) over ~4 s; the retreat is mainly -Z local.
Distances are mm in the link6 frame at the TCP (link6 +Z offset 135.8 mm).
The usual IK path uses numpy; scipy supplies a bounded fallback near joint limits.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import json
import math
from pathlib import Path

import cv2
import numpy as np

JOINT_MIN = np.array((-2.6179, 0.0, -2.967, -1.745, -1.22, -2.09439))
JOINT_MAX = np.array((2.6179, 3.14, 0.0, 1.745, 1.22, 2.09439))
IMAGE_SIZE = (640, 480)
PREPARE_JOINTS = (0.52682880, 2.02785450, -0.62518083, 1.61683041, -1.12297708, -0.28225625)

# piper_sdk C_PiperForwardKinematics(dh_is_offset=1), modified DH.
_DH_A = (0., 0., 285.03, -21.98, 0., 0.)
_DH_ALPHA = (0., -math.pi/2, 0., math.pi/2, -math.pi/2, math.pi/2)
_DH_THETA = (0., -math.pi*172.22/180, -102.78/180*math.pi, 0., 0., 0.)
_DH_D = (123., 0., 0., 250.75, 0., 91.)


def link6_transform(q):
    T = np.eye(4)
    for i in range(6):
        ct, st = math.cos(q[i]+_DH_THETA[i]), math.sin(q[i]+_DH_THETA[i])
        ca, sa = math.cos(_DH_ALPHA[i]), math.sin(_DH_ALPHA[i])
        T = T @ np.array([[ct, -st, 0, _DH_A[i]],
                          [st*ca, ct*ca, -sa, -sa*_DH_D[i]],
                          [st*sa, ct*sa, ca, ca*_DH_D[i]],
                          [0, 0, 0, 1]])
    return T


def tcp_pose(q, offset_mm):
    """TCP position (mm, base frame) and link6 rotation matrix."""
    T = link6_transform(np.asarray(q, float))
    return T[:3, 3] + T[:3, 2]*offset_mm, T[:3, :3]


def _rotation_error(R, goal):
    return .5*sum(np.cross(R[:, i], goal[:, i]) for i in range(3))


def rotation_distance(R, goal):
    return math.acos(float(np.clip((np.trace(goal @ R.T)-1.)/2., -1., 1.)))


def lateral_basis(direction):
    """Unit approach axis and a 3x2 orthonormal basis of the plane perpendicular to it."""
    axis = np.asarray(direction, float)/np.linalg.norm(direction)
    reference = np.eye(3)[0] if abs(axis[0]) < .9 else np.eye(3)[1]
    first = reference - (reference @ axis)*axis
    first /= np.linalg.norm(first)
    return axis, np.stack([first, np.cross(axis, first)], 1)


def solve_ik(q_seed, position, rotation, offset_mm, *, iterations=12):
    """Damped least squares from a nearby seed; None if inaccurate or out of limits."""
    q = np.clip(np.asarray(q_seed, float).copy(), JOINT_MIN, JOINT_MAX)
    for _ in range(iterations):
        p, R = tcp_pose(q, offset_mm)
        error = np.r_[position-p, 100.*_rotation_error(R, rotation)]
        if np.linalg.norm(error[:3]) < .05 and np.linalg.norm(error[3:]) < .05:
            break
        J = np.empty((6, 6))
        for j in range(6):
            dq = np.zeros(6)
            dq[j] = 1e-5
            p2, R2 = tcp_pose(q+dq, offset_mm)
            J[:3, j] = (p2-p)/1e-5
            J[3:, j] = 100.*_rotation_error(R, R2)/1e-5
        q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-2*np.eye(6), error), JOINT_MIN, JOINT_MAX)
    p, R = tcp_pose(q, offset_mm)
    if np.linalg.norm(p-position) <= .3 and rotation_distance(R, rotation) <= np.deg2rad(.5):
        return q
    # Clipping an unconstrained Newton step stalls at a bound even when the
    # remaining joints can satisfy the existing position/orientation tolerances.
    # TRF solves with the bounds active instead. No new seeds or joint-limit
    # relaxation: stay on the current nearby branch and verify the actual pose.
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    def residual(candidate):
        point, rot = tcp_pose(candidate, offset_mm)
        # Prefer preserving the fingertip path over spending the permitted
        # position tolerance at the boundary; orientation remains capped below.
        return np.r_[10.*(point-position), 100.*Rotation.from_matrix(rotation @ rot.T).as_rotvec()]
    result = least_squares(residual, q, bounds=(JOINT_MIN, JOINT_MAX),
                           max_nfev=60, ftol=1e-8, xtol=1e-8, gtol=1e-8)
    p, R = tcp_pose(result.x, offset_mm)
    if (np.isfinite(result.x).all() and np.linalg.norm(p-position) <= .3
            and rotation_distance(R, rotation) <= np.deg2rad(.5)):
        return result.x
    return None


@dataclass(frozen=True)
class MixerServoConfig:
    # Pixel where the slot centroid sits when the closed fingertip is at the paddle.
    target_uv: tuple = (368., 346.)
    # Fingertip silhouette in the 640x480 wrist image (official and local mounts).
    finger_polygon: tuple = ((335, 368), (385, 368), (490, 480), (235, 480))
    # Initial feature Jacobian du,dv per local [x, y, z] mm, fitted on the official data.
    image_jacobian: tuple = ((-.38, 2.58, .71), (-2.68, .38, .49))
    # Approach and insert direction in the link6 frame; lateral alignment uses
    # the two perpendicular axes. The simulator expert approaches off the tool axis.
    approach_local_dir: tuple = (0., 0., 1.)
    tcp_offset_mm: float = 135.8
    # Trigger: model approach has finished.
    trigger_radius_px: float | None = 200.  # None: no target-distance gate
    trigger_frames: int = 5
    # Missed detections tolerated while confirming; 0 on real images, where
    # gaps let seam/edge look-alikes accumulate. The simulator slot needs ~2.
    trigger_max_misses: int = 0
    trigger_max_tcp_speed_mm_s: float = 25.
    # Official arrival TCP z is 33-87 mm; early high triggers saw the housing corner.
    trigger_max_tcp_z_mm: float | None = 120.
    gripper_closed_below: float = .25
    # Slot-vs-surround brightness difference; the simulator's darker housing gives ~40-55.
    slot_min_contrast: float = 45.
    # Real profile: dark marks must belong to the white paddle, including
    # during tracking. The simulator's housing slot retains the basic detector.
    slot_require_paddle_support: bool = False
    slot_track_min_edge_support: float = 0.
    slot_track_reacquire_frames: int = 2
    slot_track_max_gap_seconds: float = .5
    # Alignment.
    align_tolerance_px: float = 12.
    align_frames: int = 2
    align_gain: float = .6
    align_step_mm: float = 8.
    max_align_mm: float | None = None  # None: no cap on total lateral alignment travel
    align_timeout_s: float = 15.
    # Optional aim line: where the slot appears when the fingertip axis passes
    # through it farther away. The fingertip then advances along +Z while the
    # slot is kept on the line from aim_far_uv to target_uv (parallax).
    aim_far_uv: tuple | None = None
    aim_tolerance_px: float = 20.
    approach_step_mm: float = 6.
    max_approach_mm: float | None = None  # None: unlimited; only contact, align_timeout_s or a lost target end the approach
    # Optional floor for the commanded TCP height (base frame, mm); moves that would go lower are clamped.
    min_tcp_z_mm: float | None = None
    # Opt-in posture preparation before visual alignment. Lift vertically
    # before rotating, preserving the fingertip position and both grippers.
    prepare_j4_margin_deg: float = 0.
    prepare_min_tcp_z_mm: float = 90.
    prepare_max_rotation_deg: float = 15.
    prepare_speed_mm_s: float = 20.
    prepare_angular_speed_deg_s: float = 10.
    prepare_timeout_s: float = 20.
    prepare_position_tolerance_mm: float = 2.
    prepare_orientation_tolerance_deg: float = .75
    # The real monitor supplies committed pickup/release evidence. Without
    # that input (standalone servo/direct mode), the ordinary visual gate applies.
    prepare_after_right_release: bool = False
    placement_trigger: str = 'release'
    # Right-base SDK link6 endpoint, WITHOUT the fingertip tool offset.
    placement_pose_xyz_mm: tuple = (340., 245., 285.)
    placement_pose_tolerance_xyz_mm: tuple = (35., 35., 35.)
    placement_pose_rpy_deg: tuple = (180., 35., -150.)
    placement_pose_orientation_tolerance_deg: float = 20.
    placement_pose_stable_seconds: float = .3
    placement_prepare_joints: tuple = (.4776, 1.6136, -.8825, -.9539, .9555, .7692)
    placement_open_threshold: float = .7
    placement_confirm_frames: int = 3
    placement_prepare_timeout_s: float = 45.
    # Optional final horizontal approach, in the LEFT BASE frame. Enabled only
    # by the real calibration profile; reaching a pixel target is not contact.
    floor_approach_base_dir: tuple | None = None
    floor_approach_mm: float = 10.
    floor_approach_step_mm: float = 2.
    floor_approach_speed_mm_s: float = 3.
    floor_approach_align_tolerance_px: float = 5.
    floor_approach_lag_mm: float = 2.
    floor_approach_lag_seconds: float = .3
    floor_approach_timeout_s: float = 12.
    # Template tracking after the first detection, finger silhouette masked.
    track_patch_px: int = 64
    track_search_px: int = 80
    # A template match farther than this from the previous position is treated as lost (template drift guard).
    max_track_step_px: float = 30.
    track_min_score: float = .5
    lost_timeout_s: float = 1.
    settle_tolerance_mm: float = 2.
    settle_timeout_s: float = 1.5
    # Insert along the approach direction until contact or the limit, then stroke and retreat.
    insert_mm: float = 8.
    contact_lag_mm: float = 4.
    stroke_local_mm: tuple = (25., 0., 0.)
    stroke_stall_lag_mm: float = 12.
    hold_s: float = .5
    retreat_local_mm: tuple = (-5., 0., -30.)
    # Speeds and safety.
    align_speed_mm_s: float = 25.
    press_speed_mm_s: float = 10.
    retreat_speed_mm_s: float = 25.
    max_joint_speed_rad_s: float = .8
    align_abort_lag_mm: float = 15.
    max_dt_s: float = .15
    vision_period_s: float = .08
    stale_image_s: float = .4
    max_attempts: int = 3
    cooldown_s: float = 3.
    left_gripper_command: float | None = None  # None keeps the value at takeover

    def __post_init__(self):
        if not isinstance(self.slot_require_paddle_support, bool):
            raise ValueError('mixer slot_require_paddle_support must be boolean')
        if (not np.isfinite(self.slot_track_min_edge_support)
                or not 0 <= self.slot_track_min_edge_support <= .65
                or isinstance(self.slot_track_reacquire_frames, bool)
                or not isinstance(self.slot_track_reacquire_frames, int)
                or self.slot_track_reacquire_frames < 1
                or not np.isfinite(self.slot_track_max_gap_seconds)
                or self.slot_track_max_gap_seconds <= 0):
            raise ValueError('invalid mixer slot tracking continuity settings')
        jacobian = np.asarray(self.image_jacobian, float)
        if jacobian.shape != (2, 3) or np.linalg.norm(self.approach_local_dir) < 1e-6:
            raise ValueError('mixer servo needs a 2x3 image Jacobian and a nonzero approach direction')
        if np.linalg.matrix_rank(jacobian @ lateral_basis(self.approach_local_dir)[1], tol=1e-3) < 2:
            raise ValueError('mixer servo lateral image Jacobian must be invertible')
        for name in ('align_step_mm', 'insert_mm', 'align_speed_mm_s', 'press_speed_mm_s',
                     'retreat_speed_mm_s', 'max_joint_speed_rad_s', 'max_track_step_px'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'mixer servo {name} must be positive')
        if self.trigger_radius_px is not None and (not np.isfinite(self.trigger_radius_px)
                                                   or self.trigger_radius_px <= 0):
            raise ValueError('mixer trigger radius must be positive or null for no distance limit')
        if not 0 < np.linalg.norm(self.stroke_local_mm) <= 45:
            raise ValueError('mixer servo stroke must be within (0, 45] mm')
        if self.insert_mm > 25 or (self.max_align_mm is not None and self.max_align_mm > 80) or (self.max_approach_mm is not None and not 0 < self.max_approach_mm <= 80):
            raise ValueError('mixer servo insert <= 25 mm, alignment and approach budgets <= 80 mm')
        if not 0 < self.approach_step_mm <= 10 or not 16 <= self.track_patch_px <= 160:
            raise ValueError('mixer servo approach step (0, 10] mm and track patch 16..160 px')
        if self.press_speed_mm_s > 30 or self.align_speed_mm_s > 50 or self.retreat_speed_mm_s > 50:
            raise ValueError('mixer servo speeds are limited to 30 mm/s press, 50 mm/s otherwise')
        if self.floor_approach_base_dir is not None:
            direction = np.asarray(self.floor_approach_base_dir, float)
            if (direction.shape != (3,) or not np.isfinite(direction).all()
                    or np.linalg.norm(direction[:2]) < 1e-6 or abs(direction[2]) > 1e-9):
                raise ValueError('mixer floor approach needs a nonzero horizontal left-base XYZ direction')
            if self.min_tcp_z_mm is None or not np.isfinite(self.min_tcp_z_mm):
                raise ValueError('mixer floor approach requires a finite min_tcp_z_mm')
        for name in ('floor_approach_mm', 'floor_approach_step_mm', 'floor_approach_speed_mm_s',
                     'floor_approach_align_tolerance_px', 'floor_approach_lag_mm',
                     'floor_approach_lag_seconds', 'floor_approach_timeout_s'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'mixer servo {name} must be finite and positive')
        if (self.floor_approach_mm > 100 or self.floor_approach_step_mm > 5
                or self.floor_approach_speed_mm_s > 10 or self.floor_approach_lag_mm > 4):
            raise ValueError('mixer floor approach limited to 100 mm travel, 5 mm steps, 10 mm/s and 4 mm lag')
        if not np.isfinite(self.prepare_j4_margin_deg) or not 0 <= self.prepare_j4_margin_deg <= 60:
            raise ValueError('mixer prepare J4 margin must be within 0..60 degrees')
        if (not np.isfinite(self.prepare_min_tcp_z_mm) or self.prepare_min_tcp_z_mm < 60
                or (self.prepare_j4_margin_deg > 0 and self.min_tcp_z_mm is not None
                    and self.prepare_min_tcp_z_mm < self.min_tcp_z_mm)):
            raise ValueError('mixer prepare height must be at least 60 mm and above the height floor')
        if not np.isfinite(self.prepare_speed_mm_s) or self.prepare_speed_mm_s <= 0:
            raise ValueError('mixer prepare_speed_mm_s must be finite and positive')
        for name, limit in (('prepare_max_rotation_deg', 20.),
                            ('prepare_angular_speed_deg_s', 15.), ('prepare_timeout_s', 60.),
                            ('prepare_position_tolerance_mm', 10.)):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value <= limit:
                raise ValueError(f'mixer {name} must be positive and <= {limit}')
        if (not np.isfinite(self.prepare_orientation_tolerance_deg)
                or self.prepare_orientation_tolerance_deg <= 0):
            raise ValueError('mixer prepare_orientation_tolerance_deg must be finite and positive')
        goal = np.asarray(self.placement_prepare_joints, float)
        if (goal.shape != (6,) or not np.isfinite(goal).all()
                or np.any(goal < JOINT_MIN) or np.any(goal > JOINT_MAX)):
            raise ValueError('mixer placement prepare joints must be finite and within joint limits')
        if (not isinstance(self.prepare_after_right_release, bool)
                or not np.isfinite(self.placement_open_threshold)
                or not 0 < self.placement_open_threshold <= 1
                or isinstance(self.placement_confirm_frames, bool)
                or not isinstance(self.placement_confirm_frames, int) or self.placement_confirm_frames < 1
                or not np.isfinite(self.placement_prepare_timeout_s)
                or not 0 < self.placement_prepare_timeout_s <= 60):
            raise ValueError('invalid mixer placement release confirmation/timeout settings')
        if self.placement_trigger not in ('release', 'pose'):
            raise ValueError('mixer placement trigger must be release or pose')
        for name in ('placement_pose_xyz_mm', 'placement_pose_tolerance_xyz_mm',
                     'placement_pose_rpy_deg'):
            value = np.asarray(getattr(self, name), float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f'mixer {name} must have three finite values')
        if (np.any(np.asarray(self.placement_pose_tolerance_xyz_mm) <= 0)
                or not np.isfinite(self.placement_pose_orientation_tolerance_deg)
                or not 0 < self.placement_pose_orientation_tolerance_deg <= 180
                or not np.isfinite(self.placement_pose_stable_seconds)
                or self.placement_pose_stable_seconds <= 0):
            raise ValueError('invalid mixer placement pose tolerances/dwell')

    @classmethod
    def from_mapping(cls, values):
        names = {f.name for f in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise ValueError(f'unknown mixer servo settings: {sorted(unknown)}')
        converted = {k: tuple(map(tuple, v)) if isinstance(v, list) and v and isinstance(v[0], list)
                     else tuple(v) if isinstance(v, list) else v for k, v in values.items()}
        return replace(cls(), **converted)


class SlotDetector:
    """Dark slot under the paddle lever: compact dark blob on a bright surface.

    The official paddle is white with two dark holes; the local replacement is a
    grey card with a black strip. Both share this structure. The red front panel
    (display, knob shadows) is the main look-alike and is rejected by colour.
    Localization only; the servo adds temporal gating and nearest-to-prior selection.
    """
    def __init__(self, finger_polygon, roi=(100, 90, 620, 440), min_contrast=45.,
                 require_paddle_support=False):
        self.min_contrast = min_contrast
        self.require_paddle_support = require_paddle_support
        self.finger = np.zeros(IMAGE_SIZE[::-1], np.uint8)
        cv2.fillPoly(self.finger, [np.asarray(finger_polygon, np.int32)], 1)
        self.roi = roi
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))

    def candidates(self, rgb, *, reference=None, prior=None, min_track_edges=0.):
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return []
        if rgb.shape[:2] != IMAGE_SIZE[::-1]:
            rgb = cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(rgb[..., :3])
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        red = ((hsv[..., 0] < 10) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 100) & (hsv[..., 2] > 60)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.kernel)
        mask = ((blackhat > 35) & (gray < 120) & (self.finger == 0)).astype(np.uint8)
        x0, y0, x1, y1 = self.roi
        mask[:y0], mask[y1:], mask[:, :x0], mask[:, x1:] = 0, 0, 0, 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        found = []
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if not 25 <= area <= 5000 or max(w, h) > 90 or area < .35*w*h:
                continue
            blob = (labels[y:y+h, x:x+w] == i).astype(np.uint8)
            if cv2.distanceTransform(np.pad(blob, 1), cv2.DIST_L2, 3).max() < 2.5:
                continue
            bx0, by0, bx1, by1 = max(0, x-10), max(0, y-10), min(640, x+w+10), min(480, y+h+10)
            ring = np.ones((by1-by0, bx1-bx0), bool)
            ring[y-by0:y-by0+h, x-bx0:x-bx0+w] = False
            ring &= self.finger[by0:by1, bx0:bx1] == 0
            if ring.sum() < 20:
                continue
            contrast = float(np.percentile(gray[by0:by1, bx0:bx1][ring], 75) - np.median(gray[y:y+h, x:x+w][blob > 0]))
            if contrast < self.min_contrast:
                continue
            rx0, ry0, rx1, ry1 = max(0, x-40), max(0, y-40), min(640, x+w+40), min(480, y+h+40)
            if red[ry0:ry1, rx0:rx1].mean() > .03:
                continue
            found.append({'uv': [float(centroids[i][0]), float(centroids[i][1])],
                          'bbox': [int(x), int(y), int(w), int(h)], 'area': int(area), 'contrast': contrast})
        if self.require_paddle_support:
            from .mixer_slot_support import supported_slots
            return supported_slots(rgb, found, reference=reference, prior=prior,
                                   min_track_edges=min_track_edges)
        return found

    def nearest(self, rgb, point, radius, **qualification):
        best = None
        for c in self.candidates(rgb, **qualification):
            distance = float(np.hypot(c['uv'][0]-point[0], c['uv'][1]-point[1]))
            if (radius is None or distance <= radius) and (best is None or distance < best['distance']):
                best = dict(c, distance=distance)
        return best


class SlotTracker:
    """Template tracking of the paddle around the first detected slot.

    A small slot can lose contrast or shrink while the camera moves (the
    simulator's dark housing); the lever around it stays matchable. The image
    is high-pass filtered and the fixed finger silhouette is zeroed in both
    template and search image. A nearby blob detection re-centres the track.
    """
    SCALES = (.9, 1., 1.1)

    def __init__(self, finger_mask, patch_px, search_px, min_score, max_step_px=30.):
        self.finger = cv2.dilate(finger_mask, np.ones((9, 9), np.uint8)) > 0
        self.half, self.search, self.min_score = patch_px//2, search_px, min_score
        self.max_step_px = max_step_px
        pad = self.half+self.search
        # Match centres inside the finger are rejected: zeroed pixels match each other.
        self.finger_padded = np.pad(self.finger, pad, constant_values=False)
        self.template = None
        self.uv = None

    def _prepare(self, rgb):
        rgb = np.asarray(rgb)[..., :3]
        if rgb.shape[:2] != IMAGE_SIZE[::-1]:
            rgb = cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
        image = gray - cv2.GaussianBlur(gray, (0, 0), 6)
        image[self.finger] = 0
        # Pad so templates and searches near the border keep a fixed size.
        return cv2.copyMakeBorder(image, self.half+self.search, self.half+self.search,
                                  self.half+self.search, self.half+self.search, cv2.BORDER_CONSTANT, value=0)

    def _patch(self, image, uv, half):
        pad = self.half+self.search
        u, v = int(round(uv[0]))+pad, int(round(uv[1]))+pad
        return image[v-half:v+half, u-half:u+half]

    def start(self, rgb, uv):
        self.uv = np.asarray(uv, float)
        self.template = self._patch(self._prepare(rgb), self.uv, self.half).copy()

    def _finger_fraction(self, uv):
        return float(self._patch(self.finger_padded, uv, self.half).mean())

    def track(self, rgb):
        """Return (uv, score) or None when the best match is below min_score."""
        image = self._prepare(rgb)
        region = self._patch(image, self.uv, self.half+self.search)
        finger = self._patch(self.finger_padded, self.uv, self.half+self.search)
        best = None
        for scale in self.SCALES:
            template = cv2.resize(self.template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            if np.std(template) < 1e-3:
                continue
            scores = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
            h, w = template.shape
            rows, cols = np.indices(scores.shape)
            centres = np.stack([cols + w/2., rows + h/2.], -1)
            # Prefer the nearest of similar matches; never a centre on the finger.
            ranked = scores - .002*np.linalg.norm(centres-(self.half+self.search), axis=-1)
            ranked[finger[(rows+h//2).clip(0, finger.shape[0]-1), (cols+w//2).clip(0, finger.shape[1]-1)]] = -2
            row, col = np.unravel_index(np.argmax(ranked), ranked.shape)
            if best is None or ranked[row, col] > best[0]:
                best = (ranked[row, col], scores[row, col], self.uv + centres[row, col] - (self.half+self.search))
        if best is None or best[1] < self.min_score:
            return None
        _, score, uv = best
        if np.linalg.norm(uv-self.uv) > self.max_step_px:
            # A jump this large between vision frames is a wrong match, not slot motion.
            return None
        self.uv = uv
        if score > .7 and self._finger_fraction(uv) < .15:
            self.template = self._patch(image, uv, self.half).copy()
        return uv, float(score)


@dataclass
class _Motion:
    """Joint-space ramp between IK-solved TCP waypoints at fixed orientation.

    IK runs once per waypoint, not per control tick. Segments are at most a
    few centimetres, so the joint-space path stays close to the straight line.
    """
    q: np.ndarray
    rotation: np.ndarray
    goal: np.ndarray
    q_from: np.ndarray = None
    q_to: np.ndarray = None
    elapsed: float = 0.
    duration: float = 0.
    goal_time: float = 0.

    def __post_init__(self):
        self.q_from = self.q.copy() if self.q_from is None else self.q_from
        self.q_to = self.q.copy() if self.q_to is None else self.q_to

    @property
    def reached(self):
        return self.elapsed >= self.duration

    def advance(self, dt):
        self.elapsed = min(self.duration, self.elapsed+dt)
        s = 1. if self.duration <= 0 else self.elapsed/self.duration
        s = .5-.5*math.cos(math.pi*s)
        self.q = self.q_from + s*(self.q_to-self.q_from)


class MixerServo:
    OWNING = ('placement_prepare', 'placement_ready', 'placement_hold',
              'prepare_lift', 'prepare_posture', 'prepare_hold', 'align',
              'floor_approach', 'floor_hold', 'insert', 'stroke', 'hold', 'retreat', 'done')

    def __init__(self, config: MixerServoConfig | None = None):
        self.config = config or MixerServoConfig()
        self.detector = SlotDetector(self.config.finger_polygon, min_contrast=self.config.slot_min_contrast,
                                    require_paddle_support=self.config.slot_require_paddle_support)
        self.tracker = SlotTracker(self.detector.finger, self.config.track_patch_px,
                                   self.config.track_search_px, self.config.track_min_score,
                                   self.config.max_track_step_px)
        self.axis, self.lateral = lateral_basis(self.config.approach_local_dir)
        self.jacobian0 = np.asarray(self.config.image_jacobian, float) @ self.lateral
        self.reset()

    def reset(self):
        self.phase, self.reason = 'model', 'waiting_for_approach'
        self.attempts = 0
        self.released_at = -math.inf
        self.last_now = None
        self.last_frame = None
        self.last_vision = -math.inf
        self.trigger_count = 0
        self.trigger_misses = 0
        self.trigger_uv = None
        self.trigger_tcp = None
        self.trigger_pair_reference = None
        self.trigger_pair_pose = None
        self.trigger_status = {'reason': 'not_evaluated', 'left_gripper': None,
                               'tcp_z_mm': None, 'tcp_speed_mm_s': None,
                               'slot_distance_px': None}
        self.match = None
        self.last_seen = None
        self.aligned_count = 0
        self.motion = None
        self.hold_target = None
        self.start_tcp = None
        self.phase_started = None
        self.jacobian = self.jacobian0.copy()
        self.vision_sample = None
        self.target_missing = False
        self.target_reacquire_count = 0
        self.target_reacquire_uv = None
        self.slot_track_reference = None
        self.slot_track_last_seen = None
        self.slot_track_current_pose = None
        self.slot_track_reference_pose = None
        self.approach_mm = 0.
        self.last_move_z = 0.
        self.floor_start_tcp = None
        self.floor_direction = None
        self.floor_started_at = None
        self.floor_planned_mm = 0.
        self.floor_measured_mm = 0.
        self.floor_lag_mm = 0.
        self.floor_lag_elapsed = 0.
        self.prepare_queue = []
        self.prepare_started_at = None
        self.prepare_report = None
        self.prepare_arrivals = 0
        self.ik_failure = None
        self.release_sequence_enabled = False
        self.placement_report = None
        self.placement_saw_lift = False
        self.placement_saw_release = False
        self.placement_open_count = 0
        self.placement_last_frame = None
        self.placement_pose_since = None
        self.placement_pose_count = 0
        self.wait_left_target = None
        self.events = []

    @property
    def owns_motion(self):
        return self.phase in self.OWNING

    @property
    def finished(self):
        return self.phase == 'done'

    def record(self):
        offset = None
        if self.motion is not None:
            command_tcp, _ = tcp_pose(self.motion.q, self.config.tcp_offset_mm)
            offset = np.round(self.motion.rotation.T @ (command_tcp-self.start_tcp), 2).tolist()
        return {'phase': self.phase, 'reason': self.reason, 'attempts': self.attempts,
                'match': self.match, 'trigger_count': self.trigger_count,
                'target_missing': self.target_missing,
                'target_reacquire_count': self.target_reacquire_count,
                'slot_track_locked': self.slot_track_reference is not None,
                'jacobian_px_per_mm': np.round(self.jacobian, 3).tolist(),
                'command_offset_local_mm': offset,
                'trigger_status': dict(self.trigger_status),
                'preparation': None if self.prepare_report is None else dict(self.prepare_report),
                'placement': None if self.placement_report is None else dict(self.placement_report),
                'j4_margin_deg': None if self.motion is None else float(np.rad2deg(
                    min(self.motion.q[3]-JOINT_MIN[3], JOINT_MAX[3]-self.motion.q[3]))),
                'ik_failure': self.ik_failure,
                'floor_approach': {
                    'enabled': self.config.floor_approach_base_dir is not None,
                    'frame': 'left_base_tcp',
                    'direction_base_xyz': None if self.floor_direction is None else self.floor_direction.tolist(),
                    'start_tcp_mm': None if self.floor_start_tcp is None else self.floor_start_tcp.tolist(),
                    'travel_limit_mm': self.config.floor_approach_mm,
                    'planned_mm': self.floor_planned_mm, 'measured_mm': self.floor_measured_mm,
                    'lag_mm': self.floor_lag_mm, 'lag_seconds': self.floor_lag_elapsed,
                    'success': None}}

    def _set_phase(self, phase, reason, now):
        self.phase, self.reason, self.phase_started = phase, reason, now
        self.events.append({'phase': phase, 'reason': reason, 'time': now})

    def _release(self, reason, now):
        """Give control back to the model; a new takeover needs a new trigger."""
        if self.release_sequence_enabled and self.placement_report is not None and self.placement_report.get('started'):
            # Once the cup is placed, retry only visual confirmation. Never let
            # the model re-grasp or undo the placed right-arm pose.
            self._stop_ramp()
            self.trigger_count, self.trigger_uv, self.trigger_tcp = 0, None, None
            self._set_phase('placement_ready', reason, now)
            return
        self._set_phase('model', reason, now)
        self.released_at = now
        self.trigger_count, self.trigger_uv, self.trigger_tcp = 0, None, None
        self.motion = None

    def step(self, state, left_rgb, *, frame_id, now, captured_at=None):
        """Return a 14-D absolute joint target while the servo owns motion, else None.

        ``frame_id`` changes for every new camera image. ``now`` is a monotonic
        clock in seconds; command ramps use its increments, capped at max_dt_s.
        """
        c = self.config
        state = np.asarray(state, float).reshape(-1)
        dt = 0. if self.last_now is None else min(c.max_dt_s, max(0., now-self.last_now))
        self.last_now = now
        fresh = frame_id != self.last_frame
        if fresh:
            self.last_frame = frame_id
        stale = captured_at is not None and not 0 <= now-captured_at <= c.stale_image_s
        tcp, measured_rotation = tcp_pose(state[:6], c.tcp_offset_mm)
        self.slot_track_current_pose = (tcp.copy(), measured_rotation.copy())
        vision_due = fresh and not stale and now-self.last_vision >= c.vision_period_s

        if self.phase == 'model':
            if self.release_sequence_enabled:
                if self.wait_left_target is None:
                    self.wait_left_target = state[:7].copy()
                if self._check_placement_release(state, frame_id, now, stale=stale):
                    return self._command(0.)
                return None
            if vision_due:
                self.last_vision = now
                self._check_trigger(state, tcp, left_rgb, now)
            return None if self.phase == 'model' else self._command(0.)
        if self.phase == 'done':
            return self.hold_target.copy()
        if self.phase in ('floor_hold', 'prepare_hold', 'placement_hold'):
            return self._command(0.)
        if self.phase == 'placement_prepare':
            return self._placement_prepare_step(state, tcp, now, dt, vision_due=vision_due, stale=stale)
        if self.phase == 'placement_ready':
            if vision_due:
                self.last_vision = now
                self._check_trigger(state, tcp, left_rgb, now)
            return self._command(0.)
        if self.phase in ('prepare_lift', 'prepare_posture'):
            return self._prepare_step(state, tcp, now, dt, vision_due=vision_due, stale=stale)
        if self.phase == 'floor_approach':
            return self._floor_step(tcp, left_rgb, now, dt, vision_due=vision_due, stale=stale)

        motion = self.motion
        lag = float(np.linalg.norm(tcp_pose(motion.q, c.tcp_offset_mm)[0]-tcp))
        if self.phase == 'align':
            if stale:
                self._release('stale_camera', now)
                return self._command(0.) if self.owns_motion else None
            # No contact detection here: tracking lag during normal motion is not evidence of
            # touching the paddle. Align keeps aligning and advancing until the operator stops it.
            if lag > c.align_abort_lag_mm:
                self._release('align_tracking_error', now)
                return self._command(0.) if self.owns_motion else None
            elif vision_due and self._settled(tcp, now):
                self.last_vision = now
                self._align_update(tcp, left_rgb, now)
                if self.phase == 'placement_ready':
                    return self._command(0.)
                if self.phase == 'model':
                    return None
        elif self.phase == 'insert':
            contact = lag > c.contact_lag_mm
            if contact or motion.reached:
                depth = self.axis @ motion.rotation.T @ (tcp-self.start_tcp)
                self._start_stroke(tcp if contact else None, now,
                                   f"{'contact' if contact else 'insert_limit'}_approach_mm={depth:.1f}")
        elif self.phase == 'stroke':
            stalled = lag > c.stroke_stall_lag_mm
            if stalled or motion.reached:
                if stalled:
                    self._goto(tcp, c.press_speed_mm_s, now)
                self._set_phase('hold', 'stroke_stalled' if stalled else 'stroke_complete', now)
        elif self.phase == 'hold':
            if motion.reached and now-self.phase_started >= c.hold_s:
                self._goto(tcp_pose(motion.q, c.tcp_offset_mm)[0] + motion.rotation @ np.asarray(c.retreat_local_mm, float),
                           c.retreat_speed_mm_s, now)
                self._set_phase('retreat', 'press_attempt_complete', now)
        elif self.phase == 'retreat':
            if motion.reached:
                self._set_phase('done', 'retreated_operator_judges_success', now)
        target = self._command(dt)
        if self.phase == 'done':
            self.hold_target = target.copy()
        return target

    def observe_right_grip(self, record):
        """Require confirmed pickup before the configured placement gate."""
        c = self.config
        if not c.prepare_after_right_release or record is None:
            return
        self.release_sequence_enabled = True
        if self.placement_report is None:
            self.placement_report = dict(status='waiting_for_right_pickup', started=False,
                open_samples=0, pose_samples=0, trigger=c.placement_trigger,
                reason='waiting_for_right_pickup')
        if self.placement_report['started']:
            return
        if record.get('phase') == 'holding':
            self.placement_saw_lift = True
        events = record.get('events', ())
        if self.placement_saw_lift and (record.get('phase') == 'released'
                or any(e in events for e in ('released_by_model', 'released_after_mixer_lowering'))):
            self.placement_saw_release = True
        status = ('waiting_for_right_pose' if c.placement_trigger == 'pose' and self.placement_saw_lift else
                  'waiting_for_right_open' if self.placement_saw_release else
                  'waiting_for_right_placement' if self.placement_saw_lift else
                  'waiting_for_right_pickup')
        self.placement_report.update(status=status, reason=status)
        self.reason = status
        self.trigger_status['reason'] = status

    def _check_placement_release(self, state, frame_id, now, *, stale):
        c = self.config
        fresh = frame_id != self.placement_last_frame
        self.placement_last_frame = frame_id
        if c.placement_trigger == 'pose':
            from scipy.spatial.transform import Rotation
            T = link6_transform(state[7:13])
            xyz_error = T[:3, 3]-np.asarray(c.placement_pose_xyz_mm)
            R_goal = Rotation.from_euler('xyz', c.placement_pose_rpy_deg, degrees=True).as_matrix()
            angle = math.degrees(rotation_distance(T[:3, :3], R_goal))
            inside = (not stale and self.placement_saw_lift
                      and np.all(abs(xyz_error) <= c.placement_pose_tolerance_xyz_mm)
                      and angle <= c.placement_pose_orientation_tolerance_deg)
            if not inside:
                self.placement_pose_since, self.placement_pose_count = None, 0
            elif fresh:
                if self.placement_pose_since is None:
                    self.placement_pose_since = now
                self.placement_pose_count += 1
            dwell = 0. if self.placement_pose_since is None else max(0., now-self.placement_pose_since)
            status = ('waiting_for_right_pickup' if not self.placement_saw_lift else
                      'placement_stale_camera' if stale else
                      'confirming_right_pose' if inside else 'waiting_for_right_pose')
            self.placement_report.update(status=status, reason=status,
                frame='right_base_sdk_endpoint',
                right_xyz_mm=T[:3, 3].tolist(),
                right_rpy_deg=Rotation.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True).tolist(),
                xyz_error_mm=xyz_error.tolist(), orientation_error_deg=angle,
                pose_samples=self.placement_pose_count, pose_dwell_seconds=dwell)
            self.reason = self.trigger_status['reason'] = status
            ready = (inside and fresh and self.placement_pose_count >= c.placement_confirm_frames
                     and dwell >= c.placement_pose_stable_seconds)
        else:
            opened = not stale and self.placement_saw_release and state[13] >= c.placement_open_threshold
            if not opened:
                self.placement_open_count = 0
            elif fresh:
                self.placement_open_count += 1
            self.placement_report['open_samples'] = self.placement_open_count
            ready = fresh and self.placement_open_count >= c.placement_confirm_frames
        if not ready:
            return False
        from .mixer_posture import plan_placement_preparation
        self.hold_target = state.copy()
        self.hold_target[:7] = self.wait_left_target
        q = self.wait_left_target[:6].copy()
        point, R = tcp_pose(q, c.tcp_offset_mm)
        self.motion = _Motion(q=q, rotation=R, goal=point, goal_time=now)
        self.start_tcp = point.copy()
        self.prepare_started_at = now
        self.prepare_queue, report = plan_placement_preparation(q, c)
        self.placement_report.update(report, started=True, open_samples=self.placement_open_count,
                                     right_open_feedback=float(state[13]))
        if report['status'] == 'failed':
            self._placement_hold(report['reason'], now)
        else:
            self._next_prepare_goal(now)
        return True

    def _placement_hold(self, reason, now):
        self.prepare_queue = []
        self.placement_report.update(status='failed', reason=reason)
        self._stop_ramp()
        self._set_phase('placement_hold', reason, now)

    def _placement_prepare_step(self, state, tcp, now, dt, *, vision_due, stale):
        c, motion = self.config, self.motion
        self.hold_target[6] = max(0., self.hold_target[6]-dt)
        if stale:
            self._placement_hold('placement_prepare_stale_camera', now)
        elif now-self.prepare_started_at > c.placement_prepare_timeout_s:
            self._placement_hold('placement_prepare_arrival_timeout', now)
        elif np.max(abs(state[:6]-motion.q)) > .15:
            self._placement_hold('placement_prepare_joint_tracking_error', now)
        elif motion.reached and vision_due:
            self.last_vision = now
            _, R = tcp_pose(state[:6], c.tcp_offset_mm)
            position_error = float(np.linalg.norm(tcp-motion.goal))
            orientation_error = float(np.rad2deg(rotation_distance(R, motion.rotation)))
            self.placement_report.update(position_error_mm=position_error,
                position_tolerance_mm=c.prepare_position_tolerance_mm,
                orientation_error_deg=orientation_error,
                orientation_tolerance_deg=c.prepare_orientation_tolerance_deg,
                waypoints_remaining=len(self.prepare_queue))
            arrived = (position_error <= c.prepare_position_tolerance_mm
                       and orientation_error <= c.prepare_orientation_tolerance_deg)
            self.prepare_arrivals = self.prepare_arrivals+1 if arrived else 0
            if self.prepare_arrivals >= 2:
                if self.prepare_queue:
                    self._next_prepare_goal(now)
                elif state[6] < c.gripper_closed_below:
                    self.placement_report.update(status='ready', reason='measured_left_prepare_arrival')
                    self.tracker.template = None
                    self.trigger_count, self.trigger_uv, self.trigger_tcp = 0, None, None
                    self._set_phase('placement_ready', 'left_prepared_waiting_for_slot', now)
        return self._command(0. if self.phase == 'placement_hold' else dt)

    def _check_trigger(self, state, tcp, rgb, now):
        c = self.config
        speed = (math.inf if self.trigger_tcp is None else
                 np.linalg.norm(tcp-self.trigger_tcp[0])/max(now-self.trigger_tcp[1], 1e-3))
        self.trigger_tcp = (tcp.copy(), now)
        self.trigger_status = dict(reason='checking', left_gripper=float(state[6]),
            tcp_z_mm=float(tcp[2]), tcp_speed_mm_s=float(speed) if np.isfinite(speed) else None,
            slot_distance_px=None)
        blocked = next((reason for condition, reason in (
            (self.attempts >= c.max_attempts, 'trigger_attempt_limit'),
            (now-self.released_at < c.cooldown_s, 'trigger_cooldown'),
            (state[6] >= c.gripper_closed_below, 'left_gripper_not_closed'),
            (c.trigger_max_tcp_z_mm is not None and tcp[2] > c.trigger_max_tcp_z_mm,
             'left_tcp_above_trigger_height')) if condition), None)
        if blocked is not None:
            self.trigger_count, self.trigger_uv, self.trigger_misses = 0, None, 0
            if c.slot_require_paddle_support:
                self.match = None
            self.trigger_status['reason'] = self.reason = blocked
            return
        prior = c.target_uv if self.trigger_uv is None else self.trigger_uv
        _, rotation = tcp_pose(state[:6], c.tcp_offset_mm)
        provisional_pair = (c.slot_require_paddle_support and self.trigger_count > 0
                            and self.trigger_pair_reference is not None
                            and self.trigger_pair_pose is not None
                            and np.linalg.norm(tcp-self.trigger_pair_pose[0]) <= 10.
                            and rotation_distance(rotation, self.trigger_pair_pose[1]) <= np.deg2rad(5.))
        if not provisional_pair:
            self.trigger_pair_reference = self.trigger_pair_pose = None
        qualification = (dict(reference=self.trigger_pair_reference, prior=self.trigger_uv,
                              min_track_edges=c.slot_track_min_edge_support)
                         if provisional_pair else {})
        match = self.detector.nearest(rgb, prior,
            c.trigger_radius_px if self.trigger_uv is None else 40., **qualification)
        distance = None if match is None else float(np.hypot(*np.subtract(match['uv'], c.target_uv)))
        self.trigger_status['slot_distance_px'] = distance
        if match is None or (c.trigger_radius_px is not None and distance > c.trigger_radius_px):
            if c.slot_require_paddle_support:
                self.match = None
            # A small, low-contrast slot is detected intermittently (simulator):
            # tolerate short gaps, but every confirmation is a fresh detection.
            self.trigger_misses += 1
            if self.trigger_misses > c.trigger_max_misses:
                self.trigger_count, self.trigger_uv = 0, None
            self.trigger_status['reason'] = self.reason = ('slot_not_found' if match is None
                                                          else 'slot_outside_trigger_radius')
            return
        self.trigger_misses = 0
        self.match, self.trigger_uv = match, match['uv']
        if match.get('paddle_support') in ('paired_marks', 'tracked_pair'):
            self.trigger_pair_reference = dict(match)
            self.trigger_pair_pose = (tcp.copy(), rotation.copy())
        # TCP speed is measured between vision frames; a fast approach restarts the count.
        self.trigger_count = self.trigger_count+1 if speed <= c.trigger_max_tcp_speed_mm_s else 1
        if self.trigger_count < c.trigger_frames:
            self.trigger_status['reason'] = self.reason = ('confirming_slot' if speed <= c.trigger_max_tcp_speed_mm_s
                                                          else 'left_tcp_moving_too_fast')
            return
        self.tracker.start(rgb, match['uv'])
        self.attempts += 1
        _, rotation = tcp_pose(state[:6], c.tcp_offset_mm)
        self.motion = _Motion(q=state[:6].copy(), rotation=rotation, goal=tcp.copy(), goal_time=now)
        self.start_tcp = tcp.copy()
        self.hold_target = state.copy()
        if c.left_gripper_command is not None:
            self.hold_target[6] = c.left_gripper_command
        self.last_seen = now
        self.slot_track_reference = (match if match.get('paddle_support') in ('paired_marks', 'tracked_pair')
                                     else self.trigger_pair_reference if provisional_pair else None)
        self.slot_track_last_seen = now
        self.slot_track_reference_pose = self.slot_track_current_pose
        self.target_missing = False
        self.target_reacquire_count = 0
        self.target_reacquire_uv = None
        self.aligned_count = 0
        self.vision_sample = None
        self.approach_mm = 0.
        self.last_move_z = 0.
        self.trigger_status['reason'] = 'triggered'
        self._set_phase('align', f'slot_near_fingertip_attempt_{self.attempts}', now)
        if c.prepare_j4_margin_deg > 0:
            self._begin_prepare(now)

    def _begin_prepare(self, now):
        from .mixer_posture import plan_preparation
        self.prepare_started_at = now
        self.prepare_queue, self.prepare_report = plan_preparation(self.motion.q, self.config)
        self.prepare_arrivals = 0
        if self.prepare_report['status'] == 'failed':
            self._prepare_hold(self.prepare_report['reason'], now)
        elif self.prepare_queue:
            self._next_prepare_goal(now)

    def _next_prepare_goal(self, now):
        phase, q, rotation = self.prepare_queue.pop(0)
        motion, c = self.motion, self.config
        start, R = tcp_pose(motion.q, c.tcp_offset_mm)
        goal, _ = tcp_pose(q, c.tcp_offset_mm)
        motion.q_from, motion.q_to = motion.q.copy(), q.copy()
        motion.elapsed = 0.
        motion.duration = math.pi/2*max(np.linalg.norm(goal-start)/c.prepare_speed_mm_s,
            np.max(abs(q-motion.q))/c.max_joint_speed_rad_s,
            rotation_distance(R, rotation)/np.deg2rad(c.prepare_angular_speed_deg_s))
        motion.goal, motion.goal_time, motion.rotation = goal, now, rotation
        self.prepare_arrivals = 0
        if self.phase != phase:
            self._set_phase(phase, 'right_placement_confirmed_moving_left_to_preparation'
                            if phase == 'placement_prepare' else 'preparing_j4_margin_before_alignment', now)

    def _prepare_hold(self, reason, now):
        self.prepare_queue = []
        self.prepare_report.update(status='failed', reason=reason)
        self._stop_ramp()
        self._set_phase('prepare_hold', reason, now)

    def _prepare_step(self, state, tcp, now, dt, *, vision_due, stale):
        motion, c = self.motion, self.config
        _, measured_R = tcp_pose(state[:6], c.tcp_offset_mm)
        position_error = float(np.linalg.norm(tcp-motion.goal))
        orientation_error = rotation_distance(measured_R, motion.rotation)
        self.prepare_report.update(position_error_mm=position_error,
            position_tolerance_mm=c.prepare_position_tolerance_mm,
            orientation_error_deg=float(np.rad2deg(orientation_error)),
            orientation_tolerance_deg=c.prepare_orientation_tolerance_deg,
            goal_tcp_mm=motion.goal.tolist(),
            waypoints_remaining=len(self.prepare_queue),
            elapsed_seconds=max(0., now-self.prepare_started_at))
        if stale:
            self._prepare_hold('prepare_stale_camera', now)
        elif now-self.prepare_started_at > c.prepare_timeout_s:
            self._prepare_hold('prepare_arrival_timeout', now)
        elif np.linalg.norm(tcp_pose(motion.q, c.tcp_offset_mm)[0]-tcp) > c.align_abort_lag_mm:
            self._prepare_hold('prepare_tracking_error', now)
        elif motion.reached and vision_due:
            self.last_vision = now
            arrived = (position_error <= c.prepare_position_tolerance_mm
                       and orientation_error <= np.deg2rad(c.prepare_orientation_tolerance_deg))
            self.prepare_arrivals = self.prepare_arrivals+1 if arrived else 0
            if self.prepare_arrivals >= 2:
                if self.prepare_queue:
                    self._next_prepare_goal(now)
                else:
                    self.prepare_report.update(status='complete', actual_j4_margin_deg=float(np.rad2deg(
                        min(state[3]-JOINT_MIN[3], JOINT_MAX[3]-state[3]))))
                    # Wrist-camera geometry changed: reacquire and align using
                    # new images, rather than reusing the pre-rotation template.
                    self.tracker.template = None
                    if c.slot_require_paddle_support:
                        self.tracker.uv = None
                        self.slot_track_reference = None
                        self.slot_track_last_seen = None
                        self.slot_track_reference_pose = None
                        self.target_missing = True
                        self.target_reacquire_count = 0
                        self.target_reacquire_uv = None
                    self.match = None
                    self.start_tcp = tcp.copy()
                    self.jacobian = self.jacobian0.copy()
                    self.vision_sample = None
                    self.last_move_z = self.approach_mm = 0.
                    self.aligned_count = 0
                    self._set_phase('align', 'prepared_j4_margin_reacquiring_slot', now)
        return self._command(0. if self.phase == 'prepare_hold' else dt)

    def _settled(self, tcp, now):
        motion = self.motion
        return motion.reached and (np.linalg.norm(tcp-motion.goal) <= self.config.settle_tolerance_mm
                                   or now-motion.goal_time >= self.config.settle_timeout_s)

    def _start_stroke(self, contact_tcp, now, reason):
        """Stroke from the measured surface on contact, else from the command."""
        c, motion = self.config, self.motion
        # On contact, start just inside the measured surface, not from the
        # command that is already pushing ahead of it.
        start = (contact_tcp + motion.rotation @ self.axis if contact_tcp is not None
                 else tcp_pose(motion.q, c.tcp_offset_mm)[0])
        self._goto(start + motion.rotation @ np.asarray(c.stroke_local_mm, float), c.press_speed_mm_s, now)
        self._set_phase('stroke', reason, now)

    def _track(self, rgb):
        if self.config.slot_require_paddle_support:
            # Strictly acquire first; maintain the verified pair when its
            # white side edges weaken. No template-only or blind motion.
            c = self.config
            now = self.last_now
            long_gap = (self.target_missing and now is not None and self.slot_track_last_seen is not None
                        and now-self.slot_track_last_seen > c.slot_track_max_gap_seconds)
            if long_gap:
                # Holding the same camera pose preserves object identity even
                # after a long pause. The live pair still needs confirmation.
                held_view = (self.slot_track_current_pose is not None
                             and self.slot_track_reference_pose is not None
                             and np.linalg.norm(self.slot_track_current_pose[0]
                                                -self.slot_track_reference_pose[0]) <= 10.
                             and rotation_distance(self.slot_track_current_pose[1],
                                                   self.slot_track_reference_pose[1]) <= np.deg2rad(5.))
                if not held_view:
                    self.slot_track_reference = None
                    self.slot_track_reference_pose = None
            prior = self.tracker.uv
            if prior is None:
                prior = self.config.target_uv
            match = self.detector.nearest(rgb, prior,
                40. if self.tracker.uv is not None else c.trigger_radius_px,
                reference=self.slot_track_reference, prior=self.tracker.uv,
                min_track_edges=c.slot_track_min_edge_support)
            if match is None:
                self.target_reacquire_count, self.target_reacquire_uv = 0, None
                return None
            uv = np.asarray(match['uv'], float)
            if self.target_missing:
                nearby = (self.target_reacquire_uv is not None
                          and np.linalg.norm(uv-self.target_reacquire_uv) <= 12.)
                self.target_reacquire_count = self.target_reacquire_count+1 if nearby else 1
                self.target_reacquire_uv = uv
                required = ((max(3, c.slot_track_reacquire_frames) if long_gap
                             else c.slot_track_reacquire_frames)
                            if self.slot_track_reference is not None else c.trigger_frames)
                if self.target_reacquire_count < required:
                    return None
            self.tracker.start(rgb, uv)
            if match.get('paddle_support') in ('paired_marks', 'tracked_pair'):
                # Perspective enlarges the pair during approach. Advance the
                # reference after every geometry/surface-verified live pair,
                # even when its side edges are too weak for new acquisition.
                self.slot_track_reference = dict(match)
            self.slot_track_last_seen = now
            self.slot_track_reference_pose = self.slot_track_current_pose
            return match
        if self.tracker.template is None:
            match = self.detector.nearest(rgb, self.config.target_uv, self.config.trigger_radius_px)
            if match is None:
                return None
            self.tracker.start(rgb, match['uv'])
            return match
        tracked = self.tracker.track(rgb)
        if tracked is None:
            # Template lost (occlusion, large view change): reacquire the slot nearby.
            blob = self.detector.nearest(rgb, self.tracker.uv, 40.)
            if blob is None:
                return None
            self.tracker.start(rgb, blob['uv'])
            tracked = (np.asarray(blob['uv'], float), 0.)
        uv, score = tracked
        # Re-centre on a nearby blob detection when the slot itself is visible.
        blob = self.detector.nearest(rgb, uv, 12.)
        if blob is not None:
            self.tracker.uv = uv = np.asarray(blob['uv'], float)
        half = self.config.track_patch_px//2
        return {'uv': [float(uv[0]), float(uv[1])], 'score': score,
                'bbox': [int(uv[0])-half//2, int(uv[1])-half//2, half, half]}

    def _align_error(self, uv):
        """Image error to remove laterally, and progress along the aim line (1 = contact)."""
        c = self.config
        target = np.asarray(c.target_uv, float)
        if c.aim_far_uv is None:
            return target-uv, 1.
        far = np.asarray(c.aim_far_uv, float)
        line = target-far
        t = float((uv-far) @ line/(line @ line))
        return far + min(t, 1.)*line - uv, t

    def _at_z_floor(self, tcp, margin_mm=1.5):
        """True when advancing would go down and the TCP (measured or commanded) is at min_tcp_z_mm."""
        c, motion = self.config, self.motion
        if c.min_tcp_z_mm is None or (motion.rotation @ self.axis)[2] >= 0:
            return False
        return min(float(tcp[2]), float(motion.goal[2])) <= c.min_tcp_z_mm + margin_mm

    def _begin_floor_approach(self, tcp, now):
        c = self.config
        direction = np.asarray(c.floor_approach_base_dir, float)
        self.floor_direction = direction / np.linalg.norm(direction)
        self.floor_start_tcp = tcp.copy()
        # Keep the current height (or correct a small measured undershoot).
        # X and orientation stay fixed for a pure base -Y approach.
        self.floor_start_tcp[2] = max(tcp[2], c.min_tcp_z_mm)
        self.floor_started_at = now
        self.floor_planned_mm = self.floor_measured_mm = self.floor_lag_mm = self.floor_lag_elapsed = 0.
        self._set_phase('floor_approach', 'z_floor_aligned_horizontal_approach', now)
        self._floor_next_goal(now)

    def _stop_ramp(self):
        motion = self.motion
        motion.q_from = motion.q_to = motion.q.copy()
        motion.goal = tcp_pose(motion.q, self.config.tcp_offset_mm)[0]
        motion.elapsed = motion.duration = 0.

    def _floor_hold(self, reason, now):
        self._stop_ramp()
        self._set_phase('floor_hold', reason, now)

    def _floor_next_goal(self, now):
        c = self.config
        distance = min(c.floor_approach_mm, self.floor_planned_mm + c.floor_approach_step_mm)
        goal = self.floor_start_tcp + self.floor_direction * distance
        if not self._goto(goal, c.floor_approach_speed_mm_s, now):
            self._floor_hold('floor_approach_ik_failed', now)
            return
        self.floor_planned_mm = float((self.motion.goal - self.floor_start_tcp) @ self.floor_direction)

    def _floor_step(self, tcp, rgb, now, dt, *, vision_due, stale):
        c, motion = self.config, self.motion
        self.floor_measured_mm = float((tcp-self.floor_start_tcp) @ self.floor_direction)
        commanded = tcp_pose(motion.q, c.tcp_offset_mm)[0]
        self.floor_lag_mm = max(0., float((commanded-tcp) @ self.floor_direction))
        if stale:
            self._floor_hold('floor_approach_stale_camera', now)
        elif now-self.floor_started_at >= c.floor_approach_timeout_s:
            self._floor_hold('floor_approach_timeout', now)
        elif self.floor_lag_mm >= c.floor_approach_lag_mm:
            # Freeze the ramp while checking sustained obstruction, rather
            # than integrating more pressure during the confirmation window.
            self.floor_lag_elapsed += dt
            if self.floor_lag_elapsed >= c.floor_approach_lag_seconds:
                self._floor_hold('floor_approach_obstructed_unverified_contact', now)
            return self._command(0.)
        else:
            self.floor_lag_elapsed = 0.
            if vision_due:
                self.last_vision = now
                match = self._track(rgb)
                if match is not None:
                    self.match, self.last_seen = match, now
                elif c.slot_require_paddle_support:
                    self.match, self.target_missing = None, True
                    self._floor_hold('floor_approach_target_lost', now)
                    return self._command(0.)
            if self.last_seen is None or now-self.last_seen > c.lost_timeout_s:
                self._floor_hold('floor_approach_target_lost', now)
            elif motion.reached:
                error = float(np.linalg.norm(tcp-motion.goal))
                if error <= c.settle_tolerance_mm:
                    if self.floor_planned_mm >= c.floor_approach_mm-1e-3:
                        self._floor_hold('floor_approach_travel_limit_unverified_contact', now)
                    else:
                        self._floor_next_goal(now)
                elif now-motion.goal_time >= c.settle_timeout_s + motion.duration:
                    self._floor_hold('floor_approach_tracking_error', now)
        # The image is monitored, but the old pixel target is not reapplied:
        # doing so would undo the deliberate base-frame approach to the paddle.
        return self._command(dt if self.phase == 'floor_approach' else 0.)

    def _align_update(self, tcp, rgb, now):
        c, motion = self.config, self.motion
        match = self._track(rgb)
        if match is None:
            if c.slot_require_paddle_support:
                # Freeze instead of advancing blind toward a former match.
                # Keep the right pose and allow same-target reacquisition.
                self.match = None
                self.target_missing = True
                self.vision_sample = None
                self.aligned_count = 0
                self._stop_ramp()
                self.reason = ('confirming_paddle_reacquisition' if self.target_reacquire_count
                               else 'paddle_target_lost_holding')
                return
            if c.floor_approach_base_dir is not None:
                self.aligned_count = 0
            if self.approach_mm <= 0:
                # Not advancing yet and no slot in view: hold and keep looking; never release on our own.
                return
            # Already advancing: the slot can slip under the fingertip mask or be occluded near the
            # surface. Keep advancing straight along the approach axis (no lateral correction without
            # vision) until contact, the approach budget or the align timeout ends 'align'.
            if c.max_approach_mm is not None and self.approach_mm >= c.max_approach_mm:
                self._release('approach_budget_exhausted', now)
                return
            if self._at_z_floor(tcp):
                # On the height floor without vision: nothing left to do but hold.
                return
            step_z = (c.approach_step_mm if c.max_approach_mm is None
                      else min(c.approach_step_mm, c.max_approach_mm-self.approach_mm))
            local = motion.rotation.T @ (tcp-self.start_tcp)
            if self._goto(self.start_tcp + motion.rotation @ (local + step_z*self.axis), c.align_speed_mm_s, now):
                self.approach_mm += step_z
                self.last_move_z = step_z
            else:
                self._release('alignment_ik_failed', now)
            return
        if c.slot_require_paddle_support:
            self.reason = 'paddle_target_reacquired' if self.target_missing else 'aligning_verified_paddle'
            self.target_missing = False
            self.target_reacquire_count = 0
            self.target_reacquire_uv = None
        self.last_seen, self.match = now, match
        uv = np.asarray(match['uv'])
        local = motion.rotation.T @ (tcp-self.start_tcp)
        lateral = self.lateral.T @ local
        # Only lateral moves are attributable to the 2x2 lateral Jacobian.
        if self.vision_sample is not None and self.last_move_z == 0:
            self._broyden(lateral-self.vision_sample[0], uv-self.vision_sample[1])
        self.vision_sample = (lateral.copy(), uv.copy())
        error, _ = self._align_error(uv)
        if c.floor_approach_base_dir is not None and self._at_z_floor(tcp):
            self.aligned_count = (self.aligned_count+1
                if (np.linalg.norm(error) <= c.floor_approach_align_tolerance_px
                    and np.linalg.norm(tcp-motion.goal) <= c.settle_tolerance_mm) else 0)
            if self.aligned_count >= c.align_frames:
                self._begin_floor_approach(tcp, now)
                return
        if c.aim_far_uv is None:
            # No aim line: being within tolerance never means arrived; keep advancing until contact.
            on_line = np.linalg.norm(error) <= c.align_tolerance_px
            aligned = False
        else:
            # With an aim line, being on the line never means arrived: keep advancing along the
            # approach axis and correct lateral error as it appears. 'align' ends only on measured
            # contact (commanded vs measured TCP lag, checked in step()) or when the approach budget runs out.
            on_line = np.linalg.norm(error) <= c.aim_tolerance_px
            aligned = False
        if aligned:
            self.aligned_count += 1
            if self.aligned_count >= c.align_frames:
                if self._goto(tcp + motion.rotation @ (c.insert_mm*self.axis), c.press_speed_mm_s, now):
                    self._set_phase('insert', f'aligned_error_px={np.linalg.norm(error):.1f}', now)
                else:
                    self._release('insert_ik_failed', now)
            return
        if c.floor_approach_base_dir is None or not self._at_z_floor(tcp):
            self.aligned_count = 0
        delta = c.align_gain*np.linalg.solve(self.jacobian, error)
        delta *= min(1., c.align_step_mm/max(np.linalg.norm(delta), 1e-9))
        step_z = 0.
        # On the height floor the approach step would only push x/y (its z part is clamped)
        # and fight the lateral correction: stop advancing, keep aligning laterally.
        if on_line and not self._at_z_floor(tcp):
            if c.max_approach_mm is not None and self.approach_mm >= c.max_approach_mm:
                self._release('approach_budget_exhausted', now)
                return
            step_z = (c.approach_step_mm if c.max_approach_mm is None
                      else min(c.approach_step_mm, c.max_approach_mm-self.approach_mm))
        goal_local = local + self.lateral @ delta + step_z*self.axis
        if c.max_align_mm is not None and np.linalg.norm(self.lateral.T @ goal_local) > c.max_align_mm:
            self._release('alignment_budget_exhausted', now)
        elif not self._goto(self.start_tcp + motion.rotation @ goal_local, c.align_speed_mm_s, now):
            self._release('alignment_ik_failed', now)
        else:
            self.approach_mm += step_z
            self.last_move_z = step_z

    def _broyden(self, dp, df):
        if np.linalg.norm(dp) < 1.5 or np.linalg.norm(df) > 200:
            return
        J0 = self.jacobian0
        candidate = self.jacobian + .5*np.outer(df-self.jacobian @ dp, dp)/(dp @ dp)
        singular = np.linalg.svd(candidate, compute_uv=False)
        scale = np.linalg.svd(J0, compute_uv=False)
        # Depth changes the feature gain; a sign flip or extreme anisotropy is noise.
        if (np.sign(np.linalg.det(candidate)) == np.sign(np.linalg.det(J0)) and
                singular[-1] >= .3*scale[-1] and singular[0] <= 3*scale[0]):
            self.jacobian = candidate

    def _goto(self, goal, speed, now):
        """Plan a ramp to the goal; shorten it if IK fails. False if nothing is reachable."""
        c, motion = self.config, self.motion
        start, _ = tcp_pose(motion.q, c.tcp_offset_mm)
        goal = np.asarray(goal, float).copy()
        if c.min_tcp_z_mm is not None and goal[2] < c.min_tcp_z_mm:
            goal[2] = max(goal[2], min(c.min_tcp_z_mm, start[2]))
        for fraction in (1., .75, .5, .25):
            position = start + fraction*(np.asarray(goal, float)-start)
            q_to = solve_ik(motion.q, position, motion.rotation, c.tcp_offset_mm)
            if q_to is not None:
                break
        else:
            self.ik_failure = dict(goal_tcp_mm=goal.tolist(),
                seed_joints_rad=motion.q.tolist(),
                j4_margin_deg=float(np.rad2deg(min(motion.q[3]-JOINT_MIN[3], JOINT_MAX[3]-motion.q[3]))))
            return False
        if fraction < 1:
            self.reason = f'ik_shortened_to_{fraction:.2f}'
        # Cosine profile peaks at pi/2 times the mean speed.
        duration = math.pi/2*max(np.linalg.norm(position-start)/speed,
                                 np.max(abs(q_to-motion.q))/c.max_joint_speed_rad_s)
        motion.q_from, motion.q_to = motion.q.copy(), q_to
        motion.elapsed, motion.duration = 0., duration
        motion.goal, motion.goal_time = position, now
        return True

    def _command(self, dt):
        self.motion.advance(dt)
        target = self.hold_target.copy()
        target[:6] = self.motion.q
        return target


def add_arguments(parser):
    parser.add_argument('--mixer-servo', action='store_true',
                        help='mixer_operating: left-arm visual servo for the paddle after the model approach')
    parser.add_argument('--mixer-servo-config', type=Path, default=None,
                        help='JSON overrides of MixerServoConfig fields')
    parser.add_argument('--mixer-servo-target-uv', type=float, nargs=2, default=None)
    parser.add_argument('--mixer-servo-stroke-mm', type=float, nargs=3, default=None,
                        help='press stroke in link6 local XYZ (default 25 0 0)')
    parser.add_argument('--mixer-servo-insert-mm', type=float, default=None)
    parser.add_argument('--mixer-servo-direct', action='store_true',
                        help='skip the model: move the left arm to the preparation pose, close the gripper, then servo')
    parser.add_argument('--mixer-servo-prepare-joints', type=float, nargs=6, default=None,
                        help='left-arm preparation joints (rad) for --mixer-servo-direct')
    parser.add_argument('--mixer-servo-observe-only', action='store_true',
                        help='log trigger/alignment decisions but never send servo commands')
    import argparse
    parser.add_argument('--mixer-servo-prepare-after-right-release',
                        action=argparse.BooleanOptionalAction, default=None,
                        help='enable pickup then placement gate before left preparation (pose or release mode)')
    parser.add_argument('--mixer-servo-placement-trigger', choices=('release', 'pose'), default=None)
    parser.add_argument('--mixer-servo-slot-require-paddle-support',
                        action=argparse.BooleanOptionalAction, default=None,
                        help='require white-paddle geometry for every dark-slot detection/tracking update')
    for name in ('placement_pose_xyz_mm', 'placement_pose_tolerance_xyz_mm', 'placement_pose_rpy_deg'):
        parser.add_argument('--mixer-servo-'+name.replace('_', '-'), type=float, nargs=3, default=None)
    parser.add_argument('--mixer-servo-placement-prepare-joints', type=float, nargs=6, default=None)
    parser.add_argument('--mixer-servo-trigger-radius-px',
                        type=lambda value: None if value.lower() in ('none', 'null') else float(value),
                        default=argparse.SUPPRESS,
                        help='maximum marker distance from target; none removes the distance gate')
    for name in ('trigger_max_tcp_z_mm', 'gripper_closed_below', 'prepare_j4_margin_deg',
                 'prepare_min_tcp_z_mm', 'prepare_max_rotation_deg', 'prepare_speed_mm_s',
                 'prepare_angular_speed_deg_s', 'prepare_timeout_s', 'prepare_position_tolerance_mm',
                 'prepare_orientation_tolerance_deg', 'placement_open_threshold',
                 'placement_confirm_frames', 'placement_prepare_timeout_s',
                 'placement_pose_orientation_tolerance_deg', 'placement_pose_stable_seconds',
                 'slot_track_min_edge_support', 'slot_track_reacquire_frames',
                 'slot_track_max_gap_seconds'):
        parser.add_argument('--mixer-servo-'+name.replace('_', '-'),
                            type=int if name.endswith('frames') else float, default=None)


def config_from_args(args):
    if not getattr(args, 'mixer_servo', False):
        if getattr(args, 'mixer_servo_direct', False):
            raise ValueError('--mixer-servo-direct requires --mixer-servo')
        return None
    if args.task != 'mixer_operating':
        raise ValueError('--mixer-servo requires --task mixer_operating')
    if getattr(args, 'mixer_offline', None) or getattr(args, 'mixer_deepseek', False):
        raise ValueError('--mixer-servo excludes the other mixer assists')
    values = json.loads(args.mixer_servo_config.read_text()) if args.mixer_servo_config else {}
    for key, value in (('target_uv', args.mixer_servo_target_uv), ('stroke_local_mm', args.mixer_servo_stroke_mm),
                       ('insert_mm', args.mixer_servo_insert_mm)):
        if value is not None:
            values[key] = value
    for name in ('prepare_after_right_release', 'placement_prepare_joints', 'placement_trigger',
                 'slot_require_paddle_support',
                 'slot_track_min_edge_support', 'slot_track_reacquire_frames',
                 'slot_track_max_gap_seconds',
                 'placement_pose_xyz_mm', 'placement_pose_tolerance_xyz_mm', 'placement_pose_rpy_deg',
                 'placement_pose_orientation_tolerance_deg', 'placement_pose_stable_seconds',
                 'trigger_max_tcp_z_mm', 'gripper_closed_below', 'prepare_j4_margin_deg',
                 'prepare_min_tcp_z_mm', 'prepare_max_rotation_deg', 'prepare_speed_mm_s',
                 'prepare_angular_speed_deg_s', 'prepare_timeout_s', 'prepare_position_tolerance_mm',
                 'prepare_orientation_tolerance_deg', 'placement_open_threshold',
                 'placement_confirm_frames', 'placement_prepare_timeout_s'):
        value = getattr(args, 'mixer_servo_'+name, None)
        if value is not None:
            values[name] = value
    if hasattr(args, 'mixer_servo_trigger_radius_px'):
        values['trigger_radius_px'] = args.mixer_servo_trigger_radius_px
    return MixerServoConfig.from_mapping(values)


def annotate(rgb, servo):
    out = np.array(rgb, copy=True)
    c = servo.config
    cv2.polylines(out, [np.asarray(c.finger_polygon, np.int32)], True, (255, 0, 0), 1)
    cv2.drawMarker(out, tuple(int(v) for v in c.target_uv), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
    if servo.match is not None:
        x, y, w, h = servo.match['bbox']
        cv2.rectangle(out, (x, y), (x+w, y+h), (255, 220, 0), 2)
    cv2.putText(out, f'{servo.phase} {servo.reason}'[:60], (8, 22), 0, .55, (255, 255, 0), 2)
    return out


__all__ = ['MixerServo', 'MixerServoConfig', 'SlotDetector', 'SlotTracker', 'add_arguments',
           'config_from_args', 'annotate', 'lateral_basis', 'tcp_pose', 'solve_ik']

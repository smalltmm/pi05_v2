"""Qualify dark slots by the surrounding paddle, never by distance alone."""
import cv2
import numpy as np


def _same_pair(lower, upper, reference):
    """Compare both marks to the latest verified pair after strict acquisition."""
    if reference is None or reference.get('paddle_support') not in ('paired_marks', 'tracked_pair'):
        return False
    old_vector = np.subtract(reference['support_uv'], reference['uv'])
    new_vector = np.subtract(upper['uv'], lower['uv'])
    old_length, new_length = np.linalg.norm(old_vector), np.linalg.norm(new_vector)
    if old_length < 1 or new_length < 1:
        return False
    cosine = float(old_vector @ new_vector/(old_length*new_length))
    width_scale = lower['bbox'][2]/reference['bbox'][2]
    height_scale = lower['bbox'][3]/reference['bbox'][3]
    return bool(.65 <= new_length/old_length <= 1.5
                and cosine >= np.cos(np.deg2rad(20.))
                and .45 <= width_scale <= 2. and .45 <= height_scale <= 2.)


def _pair_surface(gray, hsv, lower, upper):
    """Sample the exposed strip BETWEEN live marks, above the nearing fingers.

    The surface below the lower mark becomes occluded at contact. Requiring
    it for an already verified pair incorrectly treats the gripper as a loss.
    """
    top = upper['bbox'][1]+upper['bbox'][3]+2
    bottom = lower['bbox'][1]-2
    if bottom-top < 4:
        return None
    rows = np.arange(top, bottom, dtype=np.float32)
    u, v = lower['uv']
    ux, uv = upper['uv']
    centers = ux+(rows-uv)*(u-ux)/max(v-uv, 1.)
    half = max(2, int(.25*lower['bbox'][2]))
    xs = centers[:, None]+np.arange(-half, half+1, dtype=np.float32)
    ys = np.broadcast_to(rows[:, None], xs.shape).copy()
    if xs.min() < 0 or xs.max() >= gray.shape[1] or rows[0] < 0 or rows[-1] >= gray.shape[0]:
        return None
    values = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR)
    saturation = cv2.remap(hsv[:, :, 1], xs, ys, cv2.INTER_LINEAR)
    return float(np.mean((values >= 60) & (saturation <= 85)))


def supported_slots(rgb, candidates, *, reference=None, prior=None, min_track_edges=0.):
    """Accept the paired tape marks or a marker on a structurally verified strip.

    The paired branch chooses the larger LOWER mark, not the upper notch.
    The single-mark branch uses the existing elongated paddle geometry checks.
    A strictly acquired pair can retain its identity through weaker side
    edges, provided both current marks and the neutral strip remain visible.
    These are appearance heuristics, not calibrated confidence probabilities.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    accepted = []
    for lower in candidates:
        x, y, w, h = lower['bbox']
        u, v = lower['uv']
        for upper in candidates:
            ux, uy = upper['uv']
            if (not .08 <= upper['area']/lower['area'] <= .75
                    or abs(ux-u) > .6*w
                    or not max(10., .8*h) <= v-uy <= max(70., 5*h)):
                continue
            tracking = (reference is not None and prior is not None
                        and np.linalg.norm(np.subtract(lower['uv'], prior)) <= 40.
                        and _same_pair(lower, upper, reference))
            if reference is not None and not tracking:
                continue
            # Both marks must sit on a neutral light strip. Its long side
            # edges establish identity at acquisition; subsequent frames also
            # compare the live pair to that identity before allowing weaker edges.
            body_top, body_bottom = y+h+4, min(470, y+h+max(35, 2*h))
            half = max(3, int(.35*w))
            cx = int(round(u))
            if body_bottom-body_top < 25 or cx-half < 0 or cx+half >= gray.shape[1]:
                continue
            body = gray[body_top:body_bottom, cx-half:cx+half+1]
            sat = hsv[body_top:body_bottom, cx-half:cx+half+1, 1]
            surface = (_pair_surface(gray, hsv, lower, upper) if tracking else
                       float(np.mean((body >= 60) & (sat <= 85))))
            if surface is None or surface < .85:
                continue
            # Across the full vertical run, look for a rising and falling
            # edge on either side of the marker's centre, allowing a bent tip.
            # A complete mark can sit near the image border. Clip the edge
            # search to visible pixels instead of rejecting the whole paddle.
            x0, x1 = max(0, cx-2*w), min(gray.shape[1], cx+2*w+1)
            patch = cv2.GaussianBlur(gray, (3, 3), .6)[body_top:body_bottom, x0:x1].astype(float)
            grad = np.diff(patch, axis=1)
            left_search = grad[:, :cx-half-x0]
            right_search = grad[:, cx+half-x0:]
            if not left_search.size or not right_search.size:
                continue
            left = left_search.max(axis=1)
            right = -right_search.min(axis=1)
            edge_support = float(np.mean((left >= 2.) & (right >= 2.)))
            if edge_support < (.65 if not tracking else min_track_edges):
                continue
            support = 'paired_marks' if edge_support >= .65 else 'tracked_pair'
            accepted.append(dict(lower, paddle_support=support,
                support_uv=upper['uv'], strip_edge_support=edge_support,
                surface_region='between_marks' if tracking else 'below_mark',
                surface_neutral_fraction=surface))
            break
    # A single dark attachment is also valid on a verified paddle body/ledge.
    # Run the more expensive structural fallback only when the pair is absent.
    if not accepted and candidates and reference is None:
        from .mixer_paddle_features import structural_candidates
        paddles = structural_candidates(rgb)
        for slot in candidates:
            for paddle in paddles:
                attachments = paddle['features']['attachment_candidates']
                if any(np.linalg.norm(np.subtract(slot['uv'], a['uv'])) <= 12 for a in attachments):
                    accepted.append(dict(slot, paddle_support='verified_strip',
                                         support_uv=paddle['uv']))
                    break
    # Several distinct supported paddles are ambiguous. Never resolve that
    # ambiguity by choosing whichever dark patch is nearest the contact pixel.
    if accepted and any(np.linalg.norm(np.subtract(a['uv'], accepted[0]['uv'])) > 30
                        for a in accepted[1:]):
        return []
    return accepted

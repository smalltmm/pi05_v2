"""RGB-only structural cues for the white paddle shown in official references.

No local image is used as a template. These are scale-tolerant hypotheses about
the demonstrated part: an achromatic narrow strip, an exposed lower ledge, a
small dark attachment/notch above it, and locally smooth surrounding housing.
The scores are heuristic quality scores, never probabilities or contact labels.
"""
from __future__ import annotations

import cv2
import numpy as np


def finger_mask(x, y):
    """Fixed wrist-camera exclusion, deliberately wider than the finger tip."""
    return (y >= 374) & (np.abs(x-338) <= 48+.65*np.maximum(y-374, 0))


def strip_surface_features(gray, x, y, width):
    """Check a continuous, opaque strip, allowing tilt and lighting gradients.

    A glass edge can pass a short bright-line test. Over a longer segment its
    reflections/visible background usually interrupt the two edges or the
    smooth interior. The residual is normalized by local contrast, not exposure.
    This is a heuristic for the elongated fallback, not a material classifier.
    """
    rows = np.arange(y-90, y-12, 3, dtype=np.float32)
    if rows[0] < 0:
        return None
    offsets = np.arange(-1.5*width, 1.5*width+1, dtype=np.float32)
    smooth = cv2.GaussianBlur(gray, (3, 3), .6)
    middle = np.abs(offsets) <= .25*width
    left = (offsets >= -1.5*width) & (offsets <= -.75*width)
    right = (offsets >= .75*width) & (offsets <= 1.5*width)
    left_edge = (offsets[:-1] >= -width) & (offsets[:-1] <= -.2*width)
    right_edge = (offsets[:-1] >= .2*width) & (offsets[:-1] <= width)
    ry, rx = np.indices((len(rows), int(middle.sum())))
    plane = np.column_stack((np.ones(ry.size), ry.ravel(), rx.ravel()))
    plane_inverse = np.linalg.pinv(plane)
    best, best_support = None, -float('inf')
    for slope in np.linspace(-.35, .35, 15):
        centers = x+slope*(rows-y)
        xs = centers[:, None]+offsets[None, :]
        if xs.min() < 0 or xs.max() >= gray.shape[1]-1:
            continue
        ys = np.broadcast_to(rows[:, None], xs.shape).copy()
        patch = cv2.remap(smooth, xs.astype(np.float32), ys, cv2.INTER_LINEAR)
        interior = patch[:, middle]
        contrast = interior.mean(axis=1)-np.maximum(patch[:, left].mean(axis=1),
                                                    patch[:, right].mean(axis=1))
        support = float(np.mean(contrast > 5))
        median_contrast = float(np.median(contrast))
        # Choose the strip by continuity/contrast first. Do not select a flat
        # neighboring wall just because it has a small residual.
        strength = support*median_contrast
        if strength <= best_support:
            continue
        gradient = np.diff(patch, axis=1)
        paired = np.minimum(gradient[:, left_edge].max(axis=1),
                            -gradient[:, right_edge].min(axis=1))
        values = interior.ravel()
        residual = values-plane @ (plane_inverse @ values)
        rms = float(np.sqrt(np.mean(residual**2)))
        best_support = strength
        best = dict(axis_slope=float(slope), longitudinal_support=support,
                    strip_contrast=median_contrast,
                    paired_edges_fraction=float(np.mean(paired > 1.5)),
                    surface_residual=rms,
                    surface_residual_ratio=rms/max(10., median_contrast),
                    axis_uv=[[float(centers[0]), float(rows[0])], [float(x), float(y)]])
    return best


def structural_candidates(rgb):
    if np.asarray(rgb).shape != (480, 640, 3):
        return []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    yy, xx = np.indices(gray.shape)
    peaks = []

    def shifted(array, dx, dy):
        # The validity mask below excludes the rolled image boundaries.
        return np.roll(array, (-dy, -dx), axis=(0, 1))

    for width in (8, 12, 16, 24, 32):
        center = cv2.boxFilter(gray, -1, (width, 5))
        flank = cv2.boxFilter(gray, -1, (max(4, width//3), 5))
        left = shifted(flank, -width//2-width//4, -2)
        right = shifted(flank, width//2+width//4, -2)
        bottom = center-shifted(center, 0, 9)
        side_contrast = np.minimum(center-left, center-right)
        support = np.zeros_like(gray)
        for dy in (-10, -20, -30, -40):
            contrasts = []
            for slope in (-.3, 0., .3):
                dx = int(-dy*slope)
                value = shifted(center, dx, dy)
                contrasts.append(np.minimum(value-shifted(flank, dx-width//2-width//4, dy),
                                            value-shifted(flank, dx+width//2+width//4, dy)))
            support += np.maximum.reduce(contrasts) > 5
        score = (np.clip(bottom/np.maximum(15, .25*center), 0, 1)*
                 np.clip(side_contrast/np.maximum(6, .12*center), 0, 1)*support/4)
        valid = ((center > 65) & (hsv[:, :, 1] < 85) & (bottom > 12) &
                 (side_contrast > 4) & (support >= 3) & (yy > 55) & (yy < 374) &
                 (xx > 2*width) & (xx < 640-2*width) & ~finger_mask(xx, yy))
        score *= valid
        # Keep scale hypotheses separate until the independent shape checks.
        # An invalid, brighter narrow reflection must not suppress a valid
        # wider ledge at the same location.
        for _ in range(16):
            _, quality, _, location = cv2.minMaxLoc(score)
            if quality < .50:
                break
            peaks.append((quality, location, width))
            cv2.circle(score, location, 18, 0, -1)

    accepted = []
    # Examine more than the winning peak: machine reflections may score well
    # on one cue but fail the independent background/attachment checks.
    for quality, location, width in sorted(peaks, key=lambda peak: peak[0], reverse=True):
        x, y = location
        tip = gray[y-3:y+4, x-width//2:x+width//2]
        background = gray[y-60:y, x+width:min(640, x+width+35)]
        if background.size == 0:
            continue
        # Compare the body above the lip; a folded/slanted bottom mixes the
        # darker background into the endpoint window itself.
        strip_body = gray[y-15:y-5, x-width//2:x+width//2]
        relative_brightness = float(strip_body.mean()-background.mean())
        background_std = float(background.std())
        background_contrast_ratio = relative_brightness/max(20., float(background.mean()))
        if background_std > 20 or relative_brightness < 6 or background_contrast_ratio < .10:
            continue

        top = max(0, y-min(180, 12*width))
        if top < 10 or y-top < 60:
            continue
        xmin, xmax = max(0, x-2*width), min(640, x+2*width)
        above = gray[top:y-20, xmin:xmax]
        local_background = cv2.GaussianBlur(above, (0, 0), 5)
        darkness = local_background-above
        dark = (darkness > 15).astype(np.uint8)
        _, labels, stats, centers = cv2.connectedComponentsWithStats(dark, 8)
        slots = []
        for component, (stat, center) in enumerate(zip(stats[1:], centers[1:]), 1):
            _, _, w, h, area = stat
            cx, cy = center+[xmin, top]
            if (15 <= area < 1000 and 2 <= w < 2*width and 2 <= h < 4*width
                    and abs(cx-x) < width and y-cy >= 20):
                pixels = labels == component
                dark_fraction = float(np.median(darkness[pixels]/np.maximum(local_background[pixels], 1)))
                # A modest gray fluctuation on glass is not a dark slot. Use
                # relative darkness so ordinary exposure changes are tolerated.
                if dark_fraction >= .40:
                    slots.append(dict(uv=[float(cx), float(cy)], area=int(area),
                                      relative_darkness=dark_fraction))
        if not slots:
            continue
        # Refine the candidate to the lower ledge's actual downward edge.
        gradient = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        band = -gradient[max(0,y-2):min(480,y+14), x-width//2:x+width//2].mean(axis=1)
        end_y = int(max(0,y-2)+int(np.argmax(band)))
        if float(band.max()) < 30 or finger_mask(x, end_y):
            continue
        contrast = float(tip.mean()-gray[end_y+4:end_y+12, x-width//2:x+width//2].mean())
        if contrast < 12:
            continue
        surface = strip_surface_features(gray, x, end_y, width)
        if (surface is None or surface['longitudinal_support'] < .8 or
                surface['strip_contrast'] < 8 or surface['paired_edges_fraction'] < .8 or
                surface['surface_residual_ratio'] > .25):
            continue
        # Corroborating measurements contribute to a bounded quality score.
        # A score of .9 does NOT mean a 90% likelihood of correct localization.
        corroboration = (.35*quality + .25*surface['longitudinal_support'] +
                         .2*surface['paired_edges_fraction'] +
                         .2*(1-surface['surface_residual_ratio']/.25))
        accepted.append(dict(
            uv=[float(x), float(end_y)], score=float(.8+.14*corroboration),
            bbox=[x-width, top, 2*width, end_y-top+1], target='white_paddle',
            reference_source='official_video_geometry', detector_branch='structural_features',
            features=dict(strip_support_score=float(quality), width_px=width,
                          end_edge_strength=float(band.max()), end_contrast=contrast,
                          relative_brightness=relative_brightness, background_std=background_std,
                          background_contrast_ratio=background_contrast_ratio,
                          attachment_candidates=slots, **surface),
        ))
    accepted.sort(key=lambda candidate: candidate['score'], reverse=True)
    # Multiple separated high-quality tips are ambiguous, not extra evidence.
    if accepted and any(np.linalg.norm(np.array(accepted[0]['uv'])-other['uv']) > 25
                        for other in accepted[1:]):
        return []
    return accepted


def endpoint_supported(rgb, candidate):
    """Reject a source-template peak on a featureless wall or dark finger."""
    x, y = np.round(candidate['uv']).astype(int)
    if not (10 <= x < 630 and 12 <= y < 468) or finger_mask(x, y):
        return False
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    patch = gray[y-15:y+20, x-12:x+13]
    return bool(patch.size and np.percentile(patch, 90)-np.percentile(patch, 10) >= 12)

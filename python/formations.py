"""Target shapes: what the swarm is supposed to form.

Two sources, one interface (dist_dir for the reward's coverage measurement,
sample_points for the oracle's target selection):

  Stroke      a single fixed horizontal segment -- the default when no image
              pool is supplied, and what most tests run against
  Formation   a real target image, baked to an on-pixel set exactly the way
              unity/ImageLibrary.cs bakes it, so the replica and Unity agree
              on which pixels are "on" down to the threshold

Deliberately independent of the assignment logic that decides WHICH point a
given robot heads for -- reward.py's coverage measurement depends
on dist_dir, and the assignment must never influence it.

Positions are raw arena units throughout, not normalized.
"""

import os
import glob
import numpy as np

HALF_EXTENT = 100.0


class Stroke:
    """Horizontal segment y = 35, x in [-50, 50], in raw units."""

    def __init__(self, y = 35.0, x0 = -50.0, x1 = 50.0):
        self.y = y
        self.x0 = x0
        self.x1 = x1

    def dist_dir(self, pos):
        cx = np.clip(pos[:, 0], self.x0, self.x1)
        nearest = np.stack([cx, np.full(pos.shape[0], self.y)], axis = 1)
        delta = nearest - pos
        dist = np.linalg.norm(delta, axis = 1)
        safe = np.maximum(dist, 1e-9)
        direction = delta / safe[:, None]
        direction[dist < 1e-9] = 0.0
        return dist / HALF_EXTENT, direction

    def sample_points(self, k):
        # discretizes the stroke into k evenly spaced points, in the same raw
        # coordinate frame as dist_dir. Used for per-robot target
        # assignment (spatial_hash.py) -- kept
        # entirely separate from dist_dir, which reward.py's on/off-shape
        # computation depends on and which this must never alter
        xs = np.linspace(self.x0, self.x1, max(k, 2))
        return np.stack([xs, np.full_like(xs, self.y)], axis = 1)


class Formation:
    """A discrete on-pixel target loaded from a real formation image, the
    same-interface generalization of Stroke for image-agnostic replica work.
    Matches unity/ImageLibrary.cs's BakeImage: luminance threshold 0.5 on a
    [0,1]-normalized grayscale image, pixel (x,y) mapped to normalized
    (nx,nz) = (x/(w-1)*2-1, y/(h-1)*2-1). Unity's y here comes from
    Texture2D.GetPixels(), which is bottom-up (row 0 = bottom of the image,
    Unity's standard, OpenGL-derived texture convention) -- PIL's row order
    is top-down (row 0 = top, matching the raw PNG file directly), so the
    row index is flipped below before applying the same formula, to
    actually agree with what Unity bakes for the same image rather than
    just use the same-looking formula on an opposite ordering. Without the
    flip the two produce mirrored nz for identical input. Scaled to raw arena
    units (x HALF_EXTENT).
    """

    def __init__(self, path, on_threshold = 0.5):
        from PIL import Image
        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype = np.float64) / 255.0
        h, w = arr.shape
        ys_pil, xs = np.where(arr > on_threshold)
        if len(xs) == 0:
            raise ValueError("formation image %s has no on-pixels at threshold %.2f" % (path, on_threshold))
        ys = (h - 1) - ys_pil   # PIL top-down -> Unity GetPixels() bottom-up
        nx = (xs / (w - 1)) * 2.0 - 1.0
        nz = (ys / (h - 1)) * 2.0 - 1.0
        # Net identity on (nx, nz): a 90-degree CCW step and a later 90-degree
        # CW step compose to nothing, leaving only the row-flip baseline above,
        # which matches ImageLibrary.BakeImage. This orientation has been got
        # wrong more than once -- verify algebraically AND numerically before
        # changing it. See docs/code-history.md.
        self.points = np.stack([nx, nz], axis = 1) * HALF_EXTENT

    def dist_dir(self, pos):
        # same interface and semantics as Stroke.dist_dir: distance and
        # direction to the nearest on-pixel, normalized distance. This is
        # what reward.py's on/off-shape computation actually depends on
        diff = self.points[None, :, :] - pos[:, None, :]
        d = np.linalg.norm(diff, axis = 2)
        idx = np.argmin(d, axis = 1)
        nearest = self.points[idx]
        delta = nearest - pos
        dist = np.linalg.norm(delta, axis = 1)
        safe = np.maximum(dist, 1e-9)
        direction = delta / safe[:, None]
        direction[dist < 1e-9] = 0.0
        return dist / HALF_EXTENT, direction

    def sample_points(self, k):
        # coordination-aware oracle support (Arena._assign_targets): k is
        # accepted for interface parity with Stroke.sample_points but not
        # used to limit the count -- linear_sum_assignment handles more
        # candidate points than robots for free, picking the best subset, so
        # there is no reason to discard real on-pixels the formation actually
        # has just because k asked for fewer
        return self.points


def build_formation_pool(folder, pattern = ".png", limit = None, on_threshold = 0.5, exclude = None):
    # the geometric-target counterpart to images.build_image_pool, which
    # builds the ENCODER's preprocessed tensor pool -- this builds the
    # REWARD/oracle's on-pixel geometry pool. Uses images.formation_paths
    # directly, not a reimplementation, so the file ordering is guaranteed
    # identical to build_image_pool's rather than merely assumed to match --
    # index i must mean the same actual formation in both pools, since
    # Trainer._reset_arena picks one index and expects both the encoder and
    # the environment to agree on which formation it refers to. exclude is
    # threaded through for the same reason --
    # if a caller wants a guaranteed-non-overlapping training pool against
    # a held-out set, both this pool and build_image_pool's own need to
    # agree on the exclusion, not just one of them.
    from images import formation_paths
    paths = formation_paths(folder, pattern, limit, exclude)
    if not paths:
        raise ValueError("no %s files found in %s" % (pattern, folder))
    return [Formation(p, on_threshold = on_threshold) for p in paths]


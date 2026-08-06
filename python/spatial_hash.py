"""Decentralised target assignment: pure, stateless math.

On real hardware every robot carries a unique ID burned in at assembly, so it
can compute its own distinct target from (the shared formation, its own UID)
alone -- no per-robot upload and no communication. hilbert_order gives every
robot the same shape-derived spatial ordering; mix_hash turns a UID into a
well-distributed index into it.

Everything here takes plain arrays and integers and returns the same. The
stateful half -- sampling points, caching the ordering per arena -- lives in
observation.ensure_target.
"""

import numpy as np

# Pure math for hash-based, launch-decentralized target selection. These take
# plain arrays and integers and return the same, with no state to bind to.
#
# The underlying idea: on real hardware, every robot already carries a
# unique ID (kilo_uid) burned in at assembly, independent of whatever
# uniform program gets uploaded -- so a robot can compute its own,
# differentiated target from (the shared formation image, its own already-
# known UID) alone, with no per-robot upload individualization and no
# runtime communication needed for the computation itself. hilbert_order
# gives every robot the same, shape-derived spatial ordering (since it's a
# pure function of the shared formation data); mix_hash turns a robot's own
# UID (plus, optionally, something it has genuinely, independently observed
# this episode) into a well-distributed index into that ordering.


def hilbert_order(points, bits = 10):
    # Returns an array of indices into `points` (an (n, 2) array of raw x,y
    # coordinates) giving the order in which the ORIGINAL points are
    # visited along a Hilbert space-filling curve. Two points close
    # together in this order are close together spatially (not
    # guaranteed -- no space-filling curve preserves locality perfectly --
    # but far better than a raw row-major or index-order scan, which has no
    # such property at all).
    #
    # points get rescaled into an integer bits x bits grid using a SINGLE,
    # shared scale factor across both axes (the larger of the two spans),
    # not independent per-axis scaling -- an oblong shape squashed into a
    # square grid would distort distances non-uniformly, which would in
    # turn distort which points the curve treats as "close."
    #
    # bits=10 gives a 1024x1024 grid -- far finer than any formation image
    # actually used here (28x28, per the encoder), so distinct on-shape
    # points essentially never collide onto the same grid cell.
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    x_min = x.min()
    y_min = y.min()
    span = max(x.max() - x_min, y.max() - y_min, 1e-9)
    n = 1 << bits
    side_max = n - 1
    xi = np.clip(((x - x_min) / span * side_max).astype(np.int64), 0, side_max)
    yi = np.clip(((y - y_min) / span * side_max).astype(np.int64), 0, side_max)

    d = np.zeros(len(points), dtype = np.int64)
    s = n >> 1
    while s > 0:
        rx = ((xi & s) > 0).astype(np.int64)
        ry = ((yi & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        # rotate/flip this quadrant -- mirrors the standard reference
        # algorithm's rot(): the flip uses side_max (the FULL grid's fixed
        # max index), not s (which shrinks every iteration) -- using s here
        # instead of side_max is a real, easy-to-make bug (confirmed
        # directly: an earlier draft of this function did exactly that,
        # caught only by testing against a hand-verified 4x4 grid, where it
        # produced a path that revisited cells and skipped others instead
        # of the single, known-correct traversal)
        flip = (ry == 0) & (rx == 1)
        xi_f = np.where(flip, side_max - xi, xi)
        yi_f = np.where(flip, side_max - yi, yi)
        swap = (ry == 0)
        xi_new = np.where(swap, yi_f, xi_f)
        yi_new = np.where(swap, xi_f, yi_f)
        xi, yi = xi_new, yi_new
        s >>= 1
    return np.argsort(d, kind = "stable")


def mix_hash(*ints):
    # Deterministic, well-distributed unsigned 32-bit hash from one or more
    # small integers (a robot's own UID, a resample counter, an optional
    # extra entropy term). NOT cryptographic -- just needs to scramble
    # small, potentially-sequential inputs (real hardware UIDs are often
    # assigned close to sequentially at manufacture) into something that
    # doesn't cluster when reduced mod a candidate count. Standard
    # multiply-xorshift finalizer (in the family of MurmurHash3's own
    # finalizer and splitmix64), not something invented fresh here.
    x = 0
    for v in ints:
        x = (x + int(v) * 2654435761 + 40503) & 0xFFFFFFFF
        x ^= (x >> 15)
        x = (x * 2246822519) & 0xFFFFFFFF
        x ^= (x >> 13)
        x = (x * 3266489917) & 0xFFFFFFFF
        x ^= (x >> 16)
    return x


def assign_target_index(order, l, image_id):
    # `order` is hilbert_order's output for the formation's sampled points;
    # this resolves robot l's index into it. The stateful parts -- sampling the
    # points, caching the Hilbert order per arena -- live in
    # observation.ensure_target, keeping this as pure as the rest of the file.
    return int(order[mix_hash(l, image_id) % len(order)])

"""Formation image files: finding them and loading them into tensors.

formation_paths is the single source of file ordering, so an index means the
same formation to the encoder pool, the geometry pool and Unity's own lookup.
build_image_pool turns those into preprocessed tensors for the encoder.
"""

import os
import random

_formation_paths_cache = {}


def formation_paths(folder, pattern=".png", limit=None, exclude=None):
    # limit used to take names[:limit] -- the
    # first `limit` names after an alphabetical sort, e.g. 000000.png,
    # 000001.png, ... every single time, regardless of how many times the
    # script ran or how many arenas were requested. Confirmed directly as
    # the cause of a real, reported bug: with N arenas, every run only ever
    # showed the same first N formations, in that same order, no matter how
    # many times it was rerun -- since the pool itself never contained
    # anything else to draw from, no amount of randomness downstream (in
    # Trainer._pick_image, already genuinely random) could ever produce a
    # different formation once the available pool was this narrow. Now
    # samples `limit` names uniformly at random from the full, sorted set
    # instead of truncating to the front of it.
    #
    # Cached per (folder, pattern, limit) within this process: build_image_pool
    # (the encoder's tensor pool) and build_formation_pool (the reward/oracle's
    # geometry pool) are separate calls with identical arguments that must
    # agree index-for-index on which formation each index refers to
    # (Trainer._reset_arena picks one index and expects both pools to mean
    # the same actual file by it) -- caching the sampled subset here keeps
    # them in sync within a single run without threading a shared seed or
    # pre-sampled list through both call sites, while the subset itself is
    # still free to differ from one run of the script to the next.
    #
    # exclude: found while answering a direct
    # question about testing on formations the actor "hasn't seen before"
    # -- run_bc_monitored.py's own held-out val split (formation_split,
    # ensure_val_dir) is genuinely deterministic and separate, but the
    # TRAINING pool built here never excluded those same names, only
    # sampled independently from the full directory. With --limit 5000
    # training against --val-count 2000 out of roughly 172500 total, the
    # expected overlap by pure chance is real, not negligible (roughly 58
    # formations, ~1.2% of the training pool) -- confirmed by the actual
    # numbers, not asserted as a theoretical edge case. None by default,
    # so every existing caller's own behavior is completely unchanged;
    # only a caller that explicitly wants a guarantee (not just a low
    # probability) passes a set of names to keep out of contention
    # entirely, applied before sampling, not filtered after (filtering
    # after could silently return fewer than `limit` formations).
    names = []
    for name in os.listdir(folder):
        if name.endswith(pattern):
            names.append(name)
    names.sort()
    if exclude:
        names = [n for n in names if n not in exclude]

    if limit is not None and limit < len(names):
        key = (os.path.abspath(folder), pattern, limit, frozenset(exclude) if exclude else None)
        cached = _formation_paths_cache.get(key)
        if cached is not None:
            names = cached
        else:
            names = random.sample(names, limit)
            names.sort()
            _formation_paths_cache[key] = names

    paths = []
    for name in names:
        paths.append(os.path.join(folder, name))
    return paths


def build_image_pool(folder, preprocess, pattern=".png", limit=None, device=None, exclude=None):
    pool = []
    for path in formation_paths(folder, pattern, limit, exclude):
        image = preprocess(path)
        if device is not None:
            image = image.to(device)
        pool.append(image)
    return pool

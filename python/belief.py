"""Range-only pose belief tracker for the split-observation actor.

Per-robot particle filter over pose (x, y, theta) in the arena frame,
normalized units. Uses only signals the robot physically has:

  predict    executed motor chord + heading change per decision tick, applied
             in each particle's frame with motion-proportional noise
  seed fix   a seed in IR range gives strength s = 1/(1+d): a range to a
             landmark at a known arena position
  peer fix   a received neighbor message gives strength (a range to the
             sender) and, when belief comms are enabled, the sender's own
             broadcast pose estimate and confidence: a range to an
             approximately known point, which is how localization spreads
             through the swarm from seed-adjacent robots outward

One range beacon determines pose only up to rotation and reflection about it,
so the filter is deliberately conservative: resampling is gated on effective
sample size so a symmetric posterior is kept spread instead of collapsing to a
confidently wrong mode, and the first fix after a cold start re-seeds the
cloud on the measured ring. The ambiguity is broken by ranging a second
distinct beacon: another seed, or a confident neighbor.

The read-out exposes the coordinates that are observable even while absolute
pose is not: bearing and distance to the nearest seed are invariant under the
rotation symmetry, so they are useful for steering long before the cloud
collapses. Nothing here is learned; actor parameter cost is zero.
"""

import math
import torch

ARENA_HALF = 100.0
SPAWN_LIMIT = 0.9
SEED_LAYOUTS = {
    "corners": [[-0.9, 0.9], [0.9, 0.9], [-0.9, -0.9], [0.9, -0.9]],
    "cluster": [[0.22, 0.0], [0.11, 0.19], [-0.9, 0.9], [0.9, -0.9]],  # DEPRECATED, DO NOT USE -- not run in a long time. "corners" is the only supported layout; kept only for backward compatibility with old references and tests, not an alternative worth choosing.
}
SEED_POS = torch.tensor(SEED_LAYOUTS["corners"])
# every layout must be the same length: SEED_SIZE is a fixed tensor dimension
# used throughout the pipeline (actor input width, Tc, the reward's seed
# fields), not something that can vary with which layout is selected
assert all(len(v) == len(SEED_LAYOUTS["corners"]) for v in SEED_LAYOUTS.values()), \
    "all SEED_LAYOUTS entries must have the same length"
SEED_SIZE = len(SEED_LAYOUTS["corners"])

# Returns a generator on the same device as the tensor it is about to feed,
# so a mismatch cannot raise. Every generator= call site in this file, and the
# two in actor_io.py, routes through here.
#
# Cached per (generator identity, target device): a mismatch is corrected once
# and the corrected generator reused, so its state advances across calls.
# Rebuilding it each time would reseed the identical sequence and quietly
# destroy the particle diversity the filter depends on. Prints once, the first
# time a mismatch is actually caught. See docs/code-history.md.
_matched_generator_cache = {}
_mismatch_reported = False
def _matched_generator(generator, device):
    global _mismatch_reported
    if generator is None:
        return None
    target = torch.device(device)
    if generator.device.type == target.type:
        return generator
    key = (id(generator), target.type)
    cached = _matched_generator_cache.get(key)
    if cached is not None:
        return cached
    if not _mismatch_reported:
        _mismatch_reported = True
        print("belief.py: generator/tensor device mismatch caught and corrected "
              "(generator was %s, needed %s)" %
              (generator.device.type, target.type))
    fixed = torch.Generator(device = target)
    fixed.manual_seed(int(generator.initial_seed()))
    _matched_generator_cache[key] = fixed
    return fixed


def set_layout(name):
    global SEED_POS
    SEED_POS = torch.tensor(SEED_LAYOUTS[name])
IR_RANGE = 7.0   # the real Kilobot platform's short-range IR, in raw project units (cm)
IR_RANGE_NORM = IR_RANGE / ARENA_HALF
# spacing between the wall-lining seeds. Must match Unity SwarmManager's own
# constant -- kept here beside the rest of the arena geometry rather than in a
# controller module, since the observation pipeline needs it too.
WALL_SPACING = 8.0

WALL_SIZE = 4
# order: north, east, south, west. axis 1 = y (north/south), axis 0 = x (east/west)
WALL_NAMES = ["north", "east", "south", "west"]
WALL_AXIS = [1, 0, 1, 0]
# wall seeds sit WALL_SEED_INSET raw units in from the true boundary (Unity's
# physical wall barrier occupies the exact edge), so the fusion target is that
# inset position, not the boundary itself. Must match SwarmManager's own
# WALL_SEED_INSET.
WALL_SEED_INSET = 5.0 / ARENA_HALF
WALL_VAL = [1.0 - WALL_SEED_INSET, 1.0 - WALL_SEED_INSET,
           -(1.0 - WALL_SEED_INSET), -(1.0 - WALL_SEED_INSET)]

# Enough particles that resampling noise cannot move the unmeasured axis. A
# single-axis (wall) reading fused against a drifted prior produces a random
# jump in the axis the measurement says nothing about, and that jump shrinks
# with particle count. See docs/code-history.md.
BELIEF_PARTICLES = 256
BELIEF_FEATURES = 11
# Extra width belief_read returns when called with target= set (the relative
# target bearing/distance triple). Kept separate from BELIEF_FEATURES so that
# constant keeps meaning "belief_read's default, target=None output size" for
# its many other callers.
BELIEF_TARGET_FEATURES = 3
# Safety ceiling on the target-distance read-out, not a tuned scale. Unlike
# d_anchor's 1.5 (distance to the NEAREST seed, small by construction), a
# target can sit anywhere on the shape, up to the arena diagonal (2.83 in these
# normalized units); 3.0 stays clear without clipping real values.
TARGET_DIST_CLAMP = 3.0
MEAS_SIGMA = 0.02
PEER_SIGMA = 0.04
PEER_CONF_MIN = 0.85
LOCALIZED_CONF_THRESHOLD = 0.4  # conf_pos above this counts as "localized" for belief/frac_localized
# Concentration (r, column 5 of belief_read) below which actor_io's steering
# path stops trusting a fresh heading reading and falls back to the last one
# that cleared this bar. The heading direction is a circular mean, and dividing
# by a small r amplifies sampling noise into a wildly varying direction. A
# defensive backstop that should not trigger while HEADING_NOISE_SCALE is 0, and
# not swept. See docs/code-history.md.
HEADING_CONCENTRATION_MIN = 0.5
PEER_SIGMA_FLOOR = 0.12
PEER_SPREAD_FLOOR = 0.1
# Zero because KilobotMovement.cs no longer applies motor bias, noise or
# smoothing -- that randomization was the only thing this modelled. Applies
# unconditionally, unlike HEADING_NOISE_SCALE: position tracking runs for every
# robot regardless of oracle_known_start_heading.
MOTION_NOISE = 0.0
NOISE_FLOOR = 0.0
ESS_FRACTION = 0.5
COLD_SPREAD = 0.32
# Splits COLD_SPREAD's variance budget equally between the two axes
# (COLD_SPREAD_AXIS**2 * 2 == COLD_SPREAD**2). Used only by the per-axis
# directional-beacon injection gate below, not by landmark ring injection,
# which stays joint since a range constrains a 2D ring, not two independent
# axes.
COLD_SPREAD_AXIS = COLD_SPREAD / math.sqrt(2.0)

# The heading every robot is spawned facing, and the noise added to each
# particle's heading per predict step.
#
# pi/2, not 0: measured from real Unity position data, every robot's true
# starting heading sits within a few degrees of +pi/2. This is what
# SwarmManager.cs's spawn rotation has always physically corresponded to.
#
# HEADING_NOISE_SCALE is exactly 0 so every particle shares one heading by
# construction. belief_predict rotates each particle's position update by its
# own heading, so any per-particle heading disagreement turns identical
# physical motion into different positions and inflates position uncertainty
# for a reason unrelated to position noise. The cost of zero noise is that an
# unmodelled dtheta error produces a confidently wrong belief with no spread to
# signal it -- an accepted trade-off. See docs/code-history.md.
#
# Only meaningful when a robot's true spawn heading is physically fixed to this
# value, which is why it is gated behind oracle_known_start_heading rather than
# assumed.
KNOWN_START_HEADING = math.pi / 2
HEADING_NOISE_SCALE = 0.0

# The four cardinal spawn headings, so initial straight-line exploration does
# not send every robot toward the same wall. A robot knows which one it was
# given: a setup fact communicated at spawn, never a live read.
#
# Signs follow from KNOWN_START_HEADING by rotation math. Unity's left-handed
# Y-rotation turns a forward-facing object from +Z toward +X, and this project's
# kinematics fix heading=0 along +X and pi/2 along +Y counterclockwise -- so
# Unity's +Z is this project's +Y, and each successive 90-degree Unity rotation
# SUBTRACTS pi/2 from the Python heading. Only the first entry is measured; the
# other three are derived. SIMPLE_ORACLE_SPAWN_CHECK
# (KILOBOT_ORACLE_DEBUG_WALL_LOG=1) verifies them per robot at spawn.
CARDINAL_HEADINGS = [
    KNOWN_START_HEADING,                       # north (Unity Y-rotation   0 deg) -- measured
    KNOWN_START_HEADING - math.pi / 2,         # east  (Unity Y-rotation  90 deg) -- derived
    KNOWN_START_HEADING - math.pi,             # south (Unity Y-rotation 180 deg) -- derived
    KNOWN_START_HEADING - 3 * math.pi / 2,     # west  (Unity Y-rotation 270 deg) -- derived
]


def belief_init(n, generator, particles = BELIEF_PARTICLES, device = "cpu", known_start_heading = False):
    generator = _matched_generator(generator, device)
    p = torch.empty(n, particles, 3, device = device)
    p[:, :, 0].uniform_(-SPAWN_LIMIT, SPAWN_LIMIT, generator = generator)
    p[:, :, 1].uniform_(-SPAWN_LIMIT, SPAWN_LIMIT, generator = generator)
    if known_start_heading:
        p[:, :, 2] = KNOWN_START_HEADING
    else:
        p[:, :, 2].uniform_(-math.pi, math.pi, generator = generator)
    return p


def belief_predict(p, x_local, y_local, dtheta, generator, true_heading = None, heading_noise_scale = None):
    generator = _matched_generator(generator, p.device)
    # true_heading, when given, SETS particle heading rather than filtering it:
    # no measurement in this module constrains heading directly, so it is not a
    # latent quantity worth spending particles on when it is already known.
    # Only position stays genuinely latent.
    chord_x = x_local.unsqueeze(1) / ARENA_HALF
    chord_y = y_local.unsqueeze(1) / ARENA_HALF
    dth = dtheta.unsqueeze(1)
    if true_heading is not None:
        start_heading = (true_heading - dtheta).unsqueeze(1)
        c = torch.cos(start_heading)
        s = torch.sin(start_heading)
    else:
        c = torch.cos(p[:, :, 2])
        s = torch.sin(p[:, :, 2])
    move = torch.sqrt(chord_x * chord_x + chord_y * chord_y)
    sig_xy = MOTION_NOISE * move + NOISE_FLOOR
    # heading_noise_scale is decoupled from MOTION_NOISE/NOISE_FLOOR: a much
    # smaller flat scale keeps tracking accurate while staying detectable if
    # reality diverges from the model, which sharing MOTION_NOISE's larger,
    # position-motivated scale would not. None preserves the unconditional-share
    # behaviour for callers that do not pass it.
    if heading_noise_scale is not None:
        sig_th = torch.full_like(dth, heading_noise_scale)
    else:
        sig_th = MOTION_NOISE * dth.abs() + NOISE_FLOOR
    nx = torch.randn(p.shape[0], p.shape[1], device = p.device, generator = generator) * sig_xy
    ny = torch.randn(p.shape[0], p.shape[1], device = p.device, generator = generator) * sig_xy
    nt = torch.randn(p.shape[0], p.shape[1], device = p.device, generator = generator) * sig_th
    out = p.clone()
    out[:, :, 0] = p[:, :, 0] + chord_x * c - chord_y * s + nx
    out[:, :, 1] = p[:, :, 1] + chord_x * s + chord_y * c + ny
    if true_heading is not None:
        out[:, :, 2] = true_heading.unsqueeze(1).expand(-1, p.shape[1])
    else:
        out[:, :, 2] = p[:, :, 2] + dth + nt
    return out


def _spread(p):
    mx = p[:, :, 0].mean(dim = 1, keepdim = True)
    my = p[:, :, 1].mean(dim = 1, keepdim = True)
    var = ((p[:, :, 0] - mx) ** 2 + (p[:, :, 1] - my) ** 2).mean(dim = 1)
    return torch.sqrt(var)


def _range_from_strength(strength):
    """Invert the IR strength law s = 1/(1+d) back to a normalized distance."""
    return (1.0 / strength.clamp(min = 1e-6) - 1.0) / ARENA_HALF


def _range_log_w(p, cx, cy, d_meas, sigma, vis):
    dx = p[:, :, 0] - cx.unsqueeze(1)
    dy = p[:, :, 1] - cy.unsqueeze(1)
    d_part = torch.sqrt(dx * dx + dy * dy)
    err = (d_part - d_meas.unsqueeze(1)) / sigma.unsqueeze(1)
    return torch.where(vis.unsqueeze(1), -0.5 * err * err, torch.zeros_like(err))


def _wall_log_w(p, axis, val, d_meas, sigma, vis):
    # signed, not abs: the arena interior is always on one known side of a wall
    # (val - p for the +1 walls, p - val for the -1 walls), so unlike a point
    # beacon there is no real mirror ambiguity to preserve here
    sign = 1.0 if val > 0 else -1.0
    interior_dist = sign * (val - p[:, :, axis])
    err = (interior_dist - d_meas.unsqueeze(1)) / sigma.unsqueeze(1)
    return torch.where(vis.unsqueeze(1), -0.5 * err * err, torch.zeros_like(err))


def _wall_along_log_w(p, along_axis, along_val, sigma, vis):
    # The along-wall axis a plain wall reading never constrains -- the missing
    # half, possible only once the specific wall seed's identity is known rather
    # than aggregate band strength. No signed/interior-side handling, unlike
    # _wall_log_w: along the wall there is no privileged direction, so this is a
    # plain symmetric Gaussian around the seed's known along-wall coordinate.
    err = (p[:, :, along_axis] - along_val.unsqueeze(1)) / sigma.unsqueeze(1)
    return torch.where(vis.unsqueeze(1), -0.5 * err * err, torch.zeros_like(err))


class _Update:
    """One belief_update in progress: the cloud, this tick's evidence, the result.

    Split into phases so each one can be read on its own. They run in a fixed
    order, and that order is load-bearing -- both because later phases consume
    what earlier ones compute (`w` and `best_raw_log_w` come from resample) and
    because every phase that draws from `generator` must keep drawing in the
    same sequence, or an identically-seeded run stops reproducing.

        add_seed_ranges          \\
        add_wall_bands            |  evidence: what this tick's measurements
        add_peer_range            |  say about where the robot could be
        add_arrived_claim        /
        resample                    weight, then resample if ESS says to
        inject_seed_ring         \\
        inject_arrived_claim_ring |  fresh hypotheses for a cloud that is
        inject_wall_band          |  too spread out, or has lost track
        inject_random_particles  /
        floor_peer_only_spread      never sharpen past what peers can support

    Evidence accumulates into log_w (per particle) plus the per-robot
    bookkeeping the injection phases need: which beacon was strongest, and how
    strong. A wall hit constrains one coordinate to a band, not a ring, so it
    never touches best_seed/best_strength (the point-beacon ring injection does
    not apply to a line); it is tracked per axis instead, because North and
    South both constrain Y while East and West both constrain X, so whichever
    single wall side fires each tick (see sample_split_event) still
    needs routing to the one axis it actually measures.

    best_dir_* rather than best_wall_* is a naming leftover from a removed
    center-cluster reading that produced the same one-axis evidence;
    belief_triangulate refers to these fields by this name.
    """

    def __init__(self, p, generator, heading_noise_scale):
        self.p = p
        self.out = p
        self.generator = generator
        self.heading_noise_scale = heading_noise_scale
        self.n, self.k, _ = p.shape
        self.seeds = SEED_POS.to(p.device)
        self.spread = _spread(p)
        self.sigma = MEAS_SIGMA + 0.15 * self.spread
        self.log_w = torch.zeros(self.n, self.k, device = p.device)
        self.any_meas = torch.zeros(self.n, dtype = torch.bool, device = p.device)
        self.seed_meas = None       # any_meas as of before the peer/claim terms
        self.best_strength = torch.zeros(self.n, device = p.device)
        self.best_seed = torch.zeros(self.n, dtype = torch.long, device = p.device)
        self.best_dir_strength = [torch.zeros(self.n, device = p.device),
                                  torch.zeros(self.n, device = p.device)]
        self.best_dir_val = [torch.zeros(self.n, device = p.device),
                             torch.zeros(self.n, device = p.device)]
        self.d_meas_ac = None       # range implied by an arrived claim, reused by its injection
        self.w = None               # normalized particle weights, set by resample()
        self.best_raw_log_w = None  # raw fit quality, set by resample()
        self.cold = None            # robots the seed ring already re-seeded

    # -- evidence ----------------------------------------------------------

    def add_seed_ranges(self, seed_obs):
        """Each visible corner seed's strength is a range to a known point."""
        for s in range(self.seeds.shape[0]):
            strength = seed_obs[:, s]
            vis = strength > 0
            if not bool(vis.any()):
                continue
            self.any_meas = self.any_meas | vis
            better = vis & (strength > self.best_strength)
            self.best_strength = torch.where(better, strength, self.best_strength)
            self.best_seed = torch.where(better, torch.full_like(self.best_seed, s), self.best_seed)
            d_meas = _range_from_strength(strength)
            self.log_w = self.log_w + _range_log_w(self.p, self.seeds[s, 0].expand(self.n),
                                                   self.seeds[s, 1].expand(self.n), d_meas, self.sigma, vis)

    def add_wall_bands(self, wall_obs, wall_seed_xy):
        """A wall reading constrains the axis perpendicular to it, to a band."""
        if wall_obs is None:
            return
        for w in range(WALL_SIZE):
            strength = wall_obs[:, w]
            vis = strength > 0
            if not bool(vis.any()):
                continue
            self.any_meas = self.any_meas | vis
            axis = WALL_AXIS[w]
            better = vis & (strength > self.best_dir_strength[axis])
            self.best_dir_strength[axis] = torch.where(better, strength, self.best_dir_strength[axis])
            self.best_dir_val[axis] = torch.where(better, torch.full_like(self.best_dir_val[axis], WALL_VAL[w]),
                                                  self.best_dir_val[axis])
            d_meas = _range_from_strength(strength)
            self.log_w = self.log_w + _wall_log_w(self.p, axis, WALL_VAL[w], d_meas, self.sigma, vis)

            # The along-wall constraint, which is what stops a wrong heading
            # hypothesis going undetected -- nothing on this axis was ever
            # inconsistent with it before. Fires only when this band's
            # wall_seed_xy is nonzero, gated by sample_split_event's reception
            # competition, so it never exceeds what one IR receiver could have
            # captured. Sigma is derived: reception already implies euclidean
            # distance <= IR_RANGE, so along^2 <= IR_RANGE^2 - cross^2 bounds it
            # tighter than assuming the full IR_RANGE applies.
            if wall_seed_xy is None:
                continue
            along_axis = 1 - axis
            along_val = wall_seed_xy[:, along_axis] / ARENA_HALF
            seed_vis = vis & (wall_seed_xy.abs().sum(dim = 1) > 0)
            if bool(seed_vis.any()):
                along_bound_sq = (IR_RANGE_NORM ** 2 - d_meas ** 2).clamp(min = 0.0)
                along_sigma = torch.sqrt(along_bound_sq).clamp(min = MEAS_SIGMA) + 0.15 * self.spread
                self.log_w = self.log_w + _wall_along_log_w(self.p, along_axis, along_val, along_sigma, seed_vis)

    def add_peer_range(self, peer_pos, peer_conf, peer_strength):
        """One peer beacon per update: the most confident valid sender.

        Only one, so a clique of correlated senders cannot multiply into a
        false sharp likelihood; and the sender's own positional sigma (decoded
        from its broadcast confidence) inflates the measurement noise.
        """
        self.seed_meas = self.any_meas.clone()   # everything so far came from map geometry, not peers
        if peer_pos is None:
            return
        usable = (peer_strength > 0) & (peer_conf > PEER_CONF_MIN)
        score = torch.where(usable, peer_conf, torch.full_like(peer_conf, -1.0))
        best = score.argmax(dim = 1)
        rows_idx = torch.arange(self.p.shape[0], device = self.p.device)
        conf = peer_conf[rows_idx, best]
        strength = peer_strength[rows_idx, best]
        vis = usable[rows_idx, best]
        if not bool(vis.any()):
            return
        self.any_meas = self.any_meas | vis
        d_meas = _range_from_strength(strength)
        sig_peer = self._sender_sigma(conf)
        self.log_w = self.log_w + _range_log_w(self.p, peer_pos[rows_idx, best, 0],
                                               peer_pos[rows_idx, best, 1], d_meas, sig_peer, vis)

    def add_arrived_claim(self, pos, conf, valid, strength):
        """An arrived neighbour's claimed position: a range to a stopped robot.

        Deliberately separate from add_peer_range, which needs belief_comms:
        an always-on position broadcast from every robot double-counts the same
        evidence across hops. This is narrower and gated -- only a robot that
        has stopped (stop_on_arrival, itself requiring belief_conf >=
        LOCALIZED_CONF_THRESHOLD), and so is no longer accumulating drift, ever
        broadcasts. Same reweighting math as peer_pos. This term is close to
        inert on its own; inject_arrived_claim_ring is what rescues a lost
        robot.
        """
        if pos is None:
            return
        if strength is not None:
            self.d_meas_ac = _range_from_strength(strength)
        if not bool(valid.any()):
            return
        self.any_meas = self.any_meas | valid
        sig_ac = self._sender_sigma(conf)
        self.log_w = self.log_w + _range_log_w(self.p, pos[:, 0], pos[:, 1], self.d_meas_ac, sig_ac, valid)

    def has_measurement(self):
        return bool(self.any_meas.any())

    def _sender_sigma(self, conf):
        """Measurement noise for a range to a peer, inflated by its confidence."""
        sender_sig = torch.sqrt(-0.02 * torch.log(conf.clamp(min = 1e-3, max = 0.999)))
        return (PEER_SIGMA + 2.0 * sender_sig + 0.15 * self.spread).clamp(min = PEER_SIGMA_FLOOR)

    # -- weighting and resampling ------------------------------------------

    def resample(self):
        """Normalize the weights and, if ESS says the cloud has degenerated, resample.

        best_raw_log_w is captured before the max-subtraction destroys the
        absolute scale. It is the MEAN raw importance weight, per the standard
        aMCL formula (Thrun/Fox) -- not the max, which lets one lucky particle
        mask a cloud whose centre is far from truth and starves
        inject_random_particles exactly when it is needed. Computed by
        log-sum-exp: exp(log_w).mean() underflows to 0.0 in float32 for a
        genuinely bad fit, where log_w runs to -1000s.
        """
        log_w_max = self.log_w.max(dim = 1, keepdim = True).values
        mean_rel_w = torch.exp(self.log_w - log_w_max).mean(dim = 1).clamp(min = 1e-30)
        self.best_raw_log_w = log_w_max.squeeze(1) + torch.log(mean_rel_w)

        log_w = self.log_w - self.log_w.max(dim = 1, keepdim = True).values
        w = torch.exp(log_w)
        self.w = w / w.sum(dim = 1, keepdim = True).clamp(min = 1e-12)
        ess = 1.0 / (self.w * self.w).sum(dim = 1)
        do_resample = self.any_meas & (ess < ESS_FRACTION * self.k)

        n, k = self.n, self.k
        idx = torch.multinomial(self.w.clamp(min = 1e-12), k, replacement = True, generator = self.generator)
        resampled = torch.gather(self.p, 1, idx.unsqueeze(2).expand(n, k, 3))
        jxy = (0.05 * self.spread).clamp(min = 2e-3).view(n, 1, 1)
        jitter = torch.randn(n, k, 3, device = self.p.device, generator = self.generator)
        # Heading is only weakly observed by ranges, so a position collapse must
        # not freeze it: the default 0.15 rad of jitter keeps honest heading
        # spread, which fans the cloud during blind travel and lets the next
        # beacon prune heading through arrival position.
        #
        # That figure assumes heading IS uncertain, which stops being true under
        # oracle_known_start_heading -- hence the override. This jitter fires on
        # every resample and dominates belief_predict's own noise by orders of
        # magnitude (0.15 rad per resample vs ~0.0003 rad per predict tick), so
        # leaving it unconditional would let every landmark correction destroy an
        # exactly-tracked heading. See docs/code-history.md.
        heading_jitter_scale = self.heading_noise_scale if self.heading_noise_scale is not None else 0.15
        resampled = resampled + jitter * torch.cat([jxy, jxy, torch.full_like(jxy, heading_jitter_scale)], dim = 2)
        self.out = torch.where(do_resample.view(n, 1, 1), resampled, self.p)

    # -- injection ---------------------------------------------------------

    def _fresh_heading(self):
        """A random heading, unless heading is separately tracked already.

        Every ring/band injection refreshes heading on the particles it
        replaces, which is right by default -- a range or band reading says
        nothing about heading -- but wrong when heading_noise_scale is given,
        where heading is separately tracked and the injection is only meant to
        refresh position. The single place all three injection sites share.
        """
        if self.heading_noise_scale is not None:
            return self.p[:, :, 2]
        return torch.rand(self.n, self.k, device = self.p.device,
                          generator = self.generator) * 2.0 * math.pi - math.pi

    def _mix_into_worst_half(self, fresh_pose, where):
        """Keep the top-weight half untouched, replace the bottom half.

        A spread-out cloud meeting a beacon keeps its top-weight half unchanged
        (deterministic, no resampling noise) and gets fresh hypotheses injected
        into the bottom half: junk mass is purged in one step, the structure
        earlier beacons built is preserved, and an on-ring cloud has near-uniform
        weights so neither resampling nor re-injection churns it.
        """
        n, k = self.n, self.k
        order = torch.argsort(self.w, dim = 1, descending = True)
        ranked = torch.gather(self.p, 1, order.unsqueeze(2).expand(n, k, 3))
        half = torch.arange(k, device = self.p.device) < k // 2
        mixed = torch.where(half.view(1, k, 1), ranked, fresh_pose)
        return torch.where(where, mixed, self.out)

    def _ring_around(self, cx, cy, d_meas, radial_sigma):
        """Particles on the ring a range measurement implies, at random angles."""
        n, k = self.n, self.k
        ang = torch.rand(n, k, device = self.p.device, generator = self.generator) * 2.0 * math.pi
        noise = torch.randn(n, k, device = self.p.device, generator = self.generator)
        rad = d_meas.unsqueeze(1) + noise * radial_sigma
        return (cx.unsqueeze(1) + rad * torch.cos(ang),
                cy.unsqueeze(1) + rad * torch.sin(ang))

    def inject_seed_ring(self):
        """Re-seed a spread-out cloud onto the ring its strongest seed implies.

        Information accumulates across beacon visits instead of being reset by
        each one: the surviving half keeps whatever pose-heading structure
        earlier beacons built.
        """
        self.cold = self.any_meas & (self.spread > COLD_SPREAD) & (self.best_strength > 0)
        if not bool(self.cold.any()):
            return
        d_best = _range_from_strength(self.best_strength)
        cx, cy = self._ring_around(self.seeds[self.best_seed, 0], self.seeds[self.best_seed, 1],
                                   d_best, MEAS_SIGMA)
        ring = torch.stack([cx, cy, self._fresh_heading()], dim = 2)
        self.out = self._mix_into_worst_half(ring, self.cold.view(self.n, 1, 1))

    def inject_arrived_claim_ring(self, pos, conf, valid):
        """The same ring injection, centred on an arrived neighbour's claim.

        This, not the reweighting in add_arrived_claim, is what rescues a lost
        robot: a single point-measurement against an already-coherent cloud
        rarely differentiates particles enough to trigger resampling at all. It
        is also what makes the sending-side gate load-bearing rather than
        optional -- a wrong claim reaching here makes the receiver's error
        worse, not merely fails to help.

        Seed injection takes priority when both are available this tick (the
        mask excludes anything already cold-injected via a seed): direct map
        data over a peer's less certain estimate.

        The LOCALIZED_CONF_THRESHOLD floor is defence in depth -- the sending
        side already guarantees it -- so that a bug in that gate alone cannot
        cause harm. Kept as a hard floor rather than proportional scaling of the
        injected fraction, which still let harm through at low confidence.

        See docs/code-history.md, and config.py's oracle_arrived_claim_injection
        for why the feature is off by default.
        """
        if pos is None:
            return
        cold_ac = (self.any_meas & (self.spread > COLD_SPREAD) & valid & ~self.cold
                   & (conf >= LOCALIZED_CONF_THRESHOLD))
        if not bool(cold_ac.any()):
            return
        sender_sig = torch.sqrt(-0.02 * torch.log(conf.clamp(min = 1e-3, max = 0.999)))
        radial_sigma = (MEAS_SIGMA + sender_sig).unsqueeze(1)
        cx, cy = self._ring_around(pos[:, 0], pos[:, 1], self.d_meas_ac, radial_sigma)
        ring = torch.stack([cx, cy, self._fresh_heading()], dim = 2)
        self.out = self._mix_into_worst_half(ring, cold_ac.view(self.n, 1, 1))

    def inject_wall_band(self):
        """Re-seed onto the band a wall reading implies, per axis, independently.

        A cold cloud's first directional-beacon hit (a wall) needs the same kind
        of injection as a seed ring, for a different reason: a point-beacon ring
        only needs angular freedom, but a cloud drawn within SPAWN_LIMIT barely
        reaches a wall at all, so plain resampling has almost no nearby particles
        to select from and stalls short of the true band. Fresh particles get the
        constrained axis placed on the interior side at its measured distance;
        the other axis keeps each particle's existing value rather than being
        redrawn, since after the phase-10 fix only one directional beacon ever
        fires per tick, and redrawing the untouched axis on every injection would
        erase whatever an earlier tick's own injection had already established
        there -- the two axes would fight instead of accumulating. Robots that
        also got a point-seed hit this step are skipped, since the ring injection
        is strictly more informative for those.

        The gate deciding WHETHER to inject is per-axis, not joint. COLD_SPREAD
        is calibrated against the combined sqrt(var_x + var_y), so one axis
        locking in tight could pull the combined number under threshold and
        silently stop injection for the other axis before it was ever separately
        measured. Each axis gates independently against COLD_SPREAD_AXIS, so a
        confident X has no say over whether Y keeps receiving fresh
        hypotheses, and vice versa.
        """
        has_dir = (self.best_dir_strength[0] > 0) | (self.best_dir_strength[1] > 0)
        if not (bool(self.any_meas.any()) and bool(has_dir.any())):
            return
        n, k = self.n, self.k
        p = self.p
        mx_now = p[:, :, 0].mean(dim = 1, keepdim = True)
        my_now = p[:, :, 1].mean(dim = 1, keepdim = True)
        spread_axis = [torch.sqrt(((p[:, :, 0] - mx_now) ** 2).mean(dim = 1)),
                       torch.sqrt(((p[:, :, 1] - my_now) ** 2).mean(dim = 1))]

        band = []
        cold_axis = []
        for axis in [0, 1]:
            bs = self.best_dir_strength[axis]
            bv = self.best_dir_val[axis]
            has_axis = (bs > 0).view(n, 1)
            sign_t = torch.where(bv > 0, torch.ones_like(bv), -torch.ones_like(bv))
            constrained = bv - sign_t * _range_from_strength(bs)
            fixed = constrained.unsqueeze(1) + torch.randn(n, k, device = p.device,
                                                           generator = self.generator) * MEAS_SIGMA
            band.append(torch.where(has_axis, fixed, p[:, :, axis]))
            cold_axis.append(self.any_meas & (bs > 0) & (self.best_strength <= 0)
                             & (spread_axis[axis] > COLD_SPREAD_AXIS))

        cold_any = cold_axis[0] | cold_axis[1]
        if not bool(cold_any.any()):
            return
        band_pose = torch.stack([band[0], band[1], self._fresh_heading()], dim = 2)
        # per-coordinate, not per-robot: x only replaced where x itself is
        # cold-and-measured, y only where y is, heading refreshed whenever
        # either injects (heading must not freeze, same reasoning as the
        # resample jitter) -- unless heading_noise_scale is provided
        coord_mask = torch.stack([cold_axis[0], cold_axis[1], cold_any], dim = 1).view(n, 1, 3)
        self.out = self._mix_into_worst_half(band_pose, coord_mask)

    def floor_peer_only_spread(self):
        """Peer ranging may centre a cloud, but never tighten it past a floor.

        Repeated ranging off the same neighbour carries the same bias, so
        sharpening past this floor would be false information; and capping the
        spread caps confidence below the anchor threshold, so only direct seed
        geometry can mint new anchors.
        """
        peer_only = self.any_meas & (~self.seed_meas)
        if not bool(peer_only.any()):
            return
        n, k = self.n, self.k
        deficit = (PEER_SPREAD_FLOOR ** 2 - _spread(self.out) ** 2).clamp(min = 0.0)
        add = torch.sqrt(deficit).view(n, 1, 1)
        noise = torch.randn(n, k, 2, device = self.p.device, generator = self.generator) * add
        loosened = self.out.clone()
        loosened[:, :, 0:2] = self.out[:, :, 0:2] + noise
        self.out = torch.where(peer_only.view(n, 1, 1), loosened, self.out)

    def inject_random_particles(self, fit_ema):
        """Augmented-MCL rescue: fresh uniform particles when the fit goes bad.

        Thrun & Fox, "Robust Monte Carlo Localization for Mobile Robots". A
        collapsed cloud can be arbitrarily confident -- low spread, all particles
        agreeing -- while being systematically wrong, since spread measures
        internal agreement, never agreement with reality. Every other reset path
        above is gated on spread being WIDE, so none can fire once the cloud has
        collapsed onto a wrong point.

        This doesn't look at spread at all: it tracks, per robot, a fast- and
        slow-moving average of how well the CURRENT cloud fits each real
        measurement (best_raw_log_w, captured by resample before normalization
        erases its absolute scale). When recent fits are persistently much worse
        than the historical baseline -- the direct, literature-standard signal
        that the filter has lost track -- a growing fraction of fresh,
        uniformly-placed particles (the same cold-start distribution belief_init
        already uses) gets mixed in, giving the filter genuine, renewed
        hypotheses to test against the next measurement, rather than only ever
        refining a neighborhood the true state may have long since left.

        fit_ema is optional and mutated in place (not returned) so every existing
        call site and test keeps working unchanged; only a caller that threads
        persistent per-robot state through it opts in.
        """
        if fit_ema is None:
            return
        ALPHA_SLOW = 0.02
        ALPHA_FAST = 0.3
        DEFICIT_SCALE = 8.0
        MAX_RANDOM_FRAC = 0.3
        # a fit this bad in absolute terms (roughly an 8-sigma miss) should
        # trigger injection regardless of what the historical baseline says --
        # confirmed necessary directly: the purely relative (slow-vs-fast) signal
        # alone can be starved by a fit that's gradually improving yet still
        # absolutely terrible, since fast tracks that slow improvement faster than
        # slow can catch up, making fast look "better than usual" even while both
        # remain far from a genuine match
        ABSOLUTE_BAD_FIT = -30.0
        n, k = self.n, self.k
        slow = fit_ema[:, 0]
        fast = fit_ema[:, 1]
        # sentinel (+1, never a real log-likelihood, which is always <= 0) marks a
        # robot with no prior fit history -- seed both EMAs directly from this
        # first real measurement instead of blending from a synthetic zero start,
        # which would otherwise make the very first measurement look artificially
        # surprising (slow lagging optimistic at 0 while fast jumps immediately to
        # the real, negative value)
        seed_now = self.any_meas & (slow > 0.5)
        slow = torch.where(seed_now, self.best_raw_log_w, slow)
        fast = torch.where(seed_now, self.best_raw_log_w, fast)
        still_tracking = self.any_meas & ~seed_now
        new_slow = torch.where(still_tracking, slow + ALPHA_SLOW * (self.best_raw_log_w - slow), slow)
        new_fast = torch.where(still_tracking, fast + ALPHA_FAST * (self.best_raw_log_w - fast), fast)
        fit_ema[:, 0] = new_slow
        fit_ema[:, 1] = new_fast
        deficit = torch.maximum((new_slow - new_fast).clamp(min = 0.0),
                                (ABSOLUTE_BAD_FIT - new_fast).clamp(min = 0.0))
        p_random = (deficit / DEFICIT_SCALE).clamp(max = MAX_RANDOM_FRAC)
        inject = still_tracking & (p_random > 0)
        if not bool(inject.any()):
            return
        # Fresh particles keep the CURRENT tracked heading rather than being
        # re-randomized or reset to known_start_heading. This injection replaces
        # only a fraction of particles, so either alternative splits the
        # population into two heading groups and a later resample can land on
        # either one -- which is where the large oscillating heading jumps came
        # from. See docs/code-history.md.
        fresh = belief_init(n, self.generator, particles = k, device = self.p.device)
        if self.heading_noise_scale is not None:
            fresh = fresh.clone()
            fresh[:, :, 2] = self.out[:, :, 2]
        draw = torch.rand(n, k, device = self.p.device, generator = self.generator)
        replace = inject.view(n, 1) & (draw < p_random.view(n, 1))
        self.out = torch.where(replace.unsqueeze(2), fresh, self.out)


def belief_update(p, seed_obs, generator, peer_pos = None, peer_conf = None, peer_strength = None,
                  wall_obs = None, fit_ema = None, anchor = None, wall_seed_xy = None,
                  arrived_claim_pos = None, arrived_claim_conf = None, arrived_claim_valid = None,
                  arrived_claim_strength = None, heading_noise_scale = None):
    """Fold one tick of measurements into the particle cloud.

    See _Update above for what each phase does and why the order matters.
    Returns the new cloud; p itself is never modified. anchor and fit_ema, when
    given, are mutated in place.
    """
    generator = _matched_generator(generator, p.device)
    u = _Update(p, generator, heading_noise_scale)
    u.add_seed_ranges(seed_obs)
    u.add_wall_bands(wall_obs, wall_seed_xy)
    u.add_peer_range(peer_pos, peer_conf, peer_strength)
    u.add_arrived_claim(arrived_claim_pos, arrived_claim_conf, arrived_claim_valid, arrived_claim_strength)
    if not u.has_measurement():
        return p

    u.resample()
    u.inject_seed_ring()
    u.inject_arrived_claim_ring(arrived_claim_pos, arrived_claim_conf, arrived_claim_valid)
    u.inject_wall_band()
    u.floor_peer_only_spread()
    u.inject_random_particles(fit_ema)

    if anchor is not None:
        out, updated_anchor = belief_triangulate(u.out, anchor, u.best_seed, u.best_strength,
                                                 u.best_dir_strength, u.best_dir_val, u.seeds, generator,
                                                 fit_quality = u.best_raw_log_w)
        anchor[:] = updated_anchor
        return out
    return u.out


def belief_track_anchor(anchor, x_local, y_local, dtheta):
    # Called every tick alongside belief_predict, whether or not the anchor is
    # valid -- belief_triangulate gates on validity separately. Reads and writes
    # only indices 3:6 (af_x, af_y, rel_theta); see ANCHOR_STATE_SIZE for the
    # full layout.
    #
    # af (accumulated anchor-frame displacement) and rel_theta (net rotation
    # since the anchor) are tracked as a single noise-free particle starting at
    # heading 0 would be: each tick's local displacement rotated by the heading
    # accumulated SO FAR, not by the anchor's still-unknown starting heading.
    # Once a hypothesis theta_a is proposed, the true world-frame displacement is
    # rotate(af, theta_a) and the current heading is theta_a + rel_theta. The
    # whole path since the anchor is kept parametric in
    # the single unknown theta_a until a measurement arrives to solve it.
    rel_theta = anchor[:, 5]
    c = torch.cos(rel_theta)
    s = torch.sin(rel_theta)
    out = anchor.clone()
    out[:, 3] = anchor[:, 3] + x_local * c - y_local * s
    out[:, 4] = anchor[:, 4] + x_local * s + y_local * c
    out[:, 5] = rel_theta + dtheta
    return out


# raised from 0.06: a reading taken with tiny accumulated displacement
# since the anchor produces an almost flat likelihood across every
# candidate heading (rotating a near-zero vector barely moves its
# endpoint regardless of angle), so it counts toward the reading
# threshold above while contributing almost no real discriminating
# power. Requiring more displacement before a reading is even eligible
# filters these out specifically, rather than just requiring more of
# them. Validated together with the change above -- see that comment.
ANCHOR_MIN_DISPLACEMENT = 0.15
TRIANGULATE_SIGMA = MEAS_SIGMA
TRIANGULATE_REPLACE_FRAC = 0.5
TRIANGULATE_N_PROBE = 72   # 5-degree spacing
# Trawny & Roumeliotis (ICRA 2010) establish 3 range measurements as the minimal
# well-posed case for full 2D pose recovery from range-only data; 2 constraints
# generically leave a two-way mirror ambiguity. Set to 5 rather than that
# minimum because a short realistic 3-reading baseline can leave the true
# candidate and a spurious one nearly tied. See docs/code-history.md.
TRIANGULATE_MIN_READINGS = 5
# a redundant reading (same source, little new displacement) is not a
# genuinely new constraint, matching the "sufficient noncollinear anchors"
# requirement in Goudar et al. 2024 (arXiv:2309.09011) -- so readings only
# count toward the threshold if they differ from the last COUNTED one,
# either by source or by accumulated displacement since it
# kept independent of ANCHOR_MIN_DISPLACEMENT (was a derived constant,
# `= ANCHOR_MIN_DISPLACEMENT`, before that was raised to 0.15) -- the
# validation for the raised ANCHOR_MIN_DISPLACEMENT specifically tested it
# at the ORIGINAL diversity threshold (0.06), not the combination of both
# raised together, so this stays at the value actually exercised rather
# than silently inheriting an untested change
DIVERSITY_MIN_DISPLACEMENT = 0.06
ANCHOR_STATE_SIZE = 10 + TRIANGULATE_N_PROBE   # [valid, anchor_x, anchor_y, af_x, af_y, rel_theta, n_readings, last_src, last_af_x, last_af_y, probe_accum...]


def belief_triangulate(p, anchor, best_seed, best_strength, best_dir_strength, best_dir_val,
                       seeds, generator, fit_quality = None):
    generator = _matched_generator(generator, p.device)
    # Gives heading the same dedicated geometric injection that position gets
    # from the ring/band injection above. Heading never resolves through
    # ordinary predict/resample alone: a fresh position hypothesis is placed
    # exactly on the ring a measurement implies, while a fresh heading
    # hypothesis is only ever re-randomized, with no connection to the
    # measurement.
    #
    # The geometry: once a robot has a known anchor point and a known
    # anchor-frame displacement since it (belief_track_anchor above, exact
    # because the kinematics are exact whenever the recorded command equals the
    # executed one), a later range or wall reading constrains the single
    # heading-at-the-anchor unknown -- one equation, one unknown. Solving for
    # the anchor point jointly instead is underdetermined.
    #
    # A single reading leaves a two-way reflection ambiguity: the true heading
    # and one mirror candidate score exactly equally, since a range constrains
    # the robot to a ring and two headings can rotate the same path onto it.
    # Hence the accumulation of log-likelihood evidence across
    # TRIANGULATE_MIN_READINGS qualifying readings before injecting anything,
    # gridded over theta_anchor -- the heading AT the anchor, the one fixed
    # unknown shared by every reading since. See docs/code-history.md.
    n, k, _ = p.shape
    device = p.device
    has_anchor = anchor[:, 0] > 0.5
    af = anchor[:, 3:5]
    disp_mag = af.norm(dim = 1)
    ready = has_anchor & (disp_mag > ANCHOR_MIN_DISPLACEMENT)

    has_corner = best_strength > 0
    has_dir = (best_dir_strength[0] > 0) | (best_dir_strength[1] > 0)
    has_new = has_corner | has_dir
    ready = ready & has_new

    # source identity for this tick's reading: corner seed index (0..3),
    # or 10/11 for wall axis 0/1, or -1 if nothing arrived
    source_id = torch.where(has_corner, best_seed.to(best_strength.dtype),
                            torch.where(best_dir_strength[0] > 0, torch.full_like(best_strength, 10.0),
                                       torch.where(best_dir_strength[1] > 0, torch.full_like(best_strength, 11.0),
                                                  torch.full_like(best_strength, -1.0))))
    last_src = anchor[:, 7]
    last_af = anchor[:, 8:10]
    is_first = anchor[:, 6] < 0.5
    different_source = (last_src - source_id).abs() > 0.5
    new_displacement = (af - last_af).norm(dim = 1) > DIVERSITY_MIN_DISPLACEMENT
    diverse = is_first | different_source | new_displacement
    accumulate = ready & diverse

    out_p = p
    new_anchor = anchor.clone()
    just_injected = torch.zeros(n, dtype = torch.bool, device = device)

    if bool(accumulate.any()):
        theta_a = torch.linspace(-math.pi, math.pi, TRIANGULATE_N_PROBE + 1, device = device)[:TRIANGULATE_N_PROBE]
        theta_a = theta_a.unsqueeze(0).expand(n, TRIANGULATE_N_PROBE)
        ca = torch.cos(theta_a)
        sa = torch.sin(theta_a)
        af_x = anchor[:, 3:4]
        af_y = anchor[:, 4:5]
        p2_x = anchor[:, 1:2] + af_x * ca - af_y * sa
        p2_y = anchor[:, 2:3] + af_x * sa + af_y * ca

        contribution = torch.zeros(n, TRIANGULATE_N_PROBE, device = device)
        if bool(has_corner.any()):
            d_meas = (1.0 / best_strength.clamp(min = 1e-6) - 1.0) / ARENA_HALF
            sx = seeds[best_seed, 0].unsqueeze(1)
            sy = seeds[best_seed, 1].unsqueeze(1)
            d_hyp = torch.sqrt((p2_x - sx) ** 2 + (p2_y - sy) ** 2)
            err = (d_hyp - d_meas.unsqueeze(1)) / TRIANGULATE_SIGMA
            contribution = contribution + torch.where(has_corner.unsqueeze(1), -0.5 * err * err, torch.zeros_like(err))
        for axis in [0, 1]:
            bs = best_dir_strength[axis]
            bv = best_dir_val[axis]
            has_axis = bs > 0
            if not bool(has_axis.any()):
                continue
            sign_t = torch.where(bv > 0, torch.ones_like(bv), -torch.ones_like(bv)).unsqueeze(1)
            d_meas = ((1.0 / bs.clamp(min = 1e-6) - 1.0) / ARENA_HALF).unsqueeze(1)
            p2_axis = p2_x if axis == 0 else p2_y
            interior_dist = sign_t * (bv.unsqueeze(1) - p2_axis)
            err = (interior_dist - d_meas) / TRIANGULATE_SIGMA
            contribution = contribution + torch.where(has_axis.unsqueeze(1), -0.5 * err * err, torch.zeros_like(err))

        probe_accum = anchor[:, 10:10 + TRIANGULATE_N_PROBE]
        new_probe_accum = torch.where(accumulate.unsqueeze(1), probe_accum + contribution, probe_accum)
        n_readings = anchor[:, 6].clamp(min = 0.0)
        new_n_readings = torch.where(accumulate, n_readings + 1.0, n_readings)

        new_anchor[:, 10:10 + TRIANGULATE_N_PROBE] = new_probe_accum
        new_anchor[:, 6] = new_n_readings
        new_anchor[:, 7] = torch.where(accumulate, source_id, last_src)
        new_anchor[:, 8:10] = torch.where(accumulate.unsqueeze(1), af, last_af)

        do_inject = accumulate & (new_n_readings >= TRIANGULATE_MIN_READINGS)
        if bool(do_inject.any()):
            log_w_probe = new_probe_accum - new_probe_accum.max(dim = 1, keepdim = True).values
            w_probe = torch.exp(log_w_probe)
            w_probe = w_probe / w_probe.sum(dim = 1, keepdim = True).clamp(min = 1e-12)
            idx = torch.multinomial(w_probe.clamp(min = 1e-12), k, replacement = True, generator = generator)
            # current heading = theta_anchor + rel (the rotation accumulated since the anchor, up to now)
            cur_heading_probe = theta_a + anchor[:, 5:6]
            inj_theta = torch.gather(cur_heading_probe, 1, idx)
            inj_x = torch.gather(p2_x.expand(n, TRIANGULATE_N_PROBE), 1, idx)
            inj_y = torch.gather(p2_y.expand(n, TRIANGULATE_N_PROBE), 1, idx)
            jn_x = torch.randn(n, k, device = device, generator = generator) * TRIANGULATE_SIGMA
            jn_y = torch.randn(n, k, device = device, generator = generator) * TRIANGULATE_SIGMA
            injected = torch.stack([inj_x + jn_x, inj_y + jn_y, inj_theta], dim = 2)

            replace = torch.rand(n, k, device = device, generator = generator) < TRIANGULATE_REPLACE_FRAC
            mixed = torch.where(replace.unsqueeze(2), injected, p)
            out_p = torch.where(do_inject.view(n, 1, 1), mixed, p)
            just_injected = just_injected | do_inject
            # reset the accumulator (and diversity-tracking fields) for the
            # next round of evidence-gathering
            new_anchor[:, 6] = torch.where(do_inject, torch.zeros_like(new_n_readings), new_n_readings)
            new_anchor[:, 7] = torch.where(do_inject, torch.full_like(new_n_readings, -1.0), new_anchor[:, 7])
            new_anchor[:, 8:10] = torch.where(do_inject.unsqueeze(1), torch.zeros_like(new_anchor[:, 8:10]), new_anchor[:, 8:10])
            new_anchor[:, 10:10 + TRIANGULATE_N_PROBE] = torch.where(
                do_inject.unsqueeze(1), torch.zeros_like(new_probe_accum), new_probe_accum)

    # refresh the anchor to the filter's own current, confident mean
    # position whenever it's trustworthy enough to serve as one -- keeps
    # the reference point recent (bounding how much accumulated-path noise
    # can build up) and gives a not-yet-anchored robot its first one as
    # soon as it becomes available.
    #
    # Gated on genuine fit_quality (best_raw_log_w, the same absolute,
    # against-real-measurements signal fit_ema already tracks for exactly
    # this reason), not on position variance alone -- variance only
    # measures internal agreement among particles, never agreement with
    # reality, so a cloud that has confidently collapsed onto a wrong
    # candidate looks just as tight as a correct one. Also excludes any
    # robot triangulation just injected new candidates for on this
    # identical tick, since fit_quality evaluated against the same
    # reading(s) that generated an injection is not independent evidence
    # of it -- confirmed both of these were each necessary but,
    # individually, not sufficient; the accumulation above is what
    # actually closes the remaining gap.
    mx = out_p[:, :, 0].mean(dim = 1)
    my = out_p[:, :, 1].mean(dim = 1)
    var_x = ((out_p[:, :, 0] - mx.unsqueeze(1)) ** 2).mean(dim = 1)
    var_y = ((out_p[:, :, 1] - my.unsqueeze(1)) ** 2).mean(dim = 1)
    conf_x = torch.exp(-var_x / 0.02)
    conf_y = torch.exp(-var_y / 0.02)
    GOOD_FIT_THRESHOLD = -5.0
    # fixed: was requiring JOINT (var_x+var_y) confidence, which structurally
    # can never fire during pure wall-following -- a wall reading only ever
    # constrains one axis, leaving the other essentially unconstrained, so
    # the joint variance stays dominated by whichever axis wall contact
    # doesn't touch. That is why the
    # anchor never got set during exactly the phase where corner-triangulation
    # is supposed to take over from it. Per-axis OR, matching the same
    # conf_x/conf_y belief_read already computes, lets a single well-
    # constrained axis anchor the mean position on that axis while the
    # other axis's own uncertainty simply propagates into the anchor (still
    # the filter's best available estimate there, not a fabricated value)
    # rather than blocking the anchor from ever forming at all.
    if fit_quality is not None:
        refresh = ((conf_x > LOCALIZED_CONF_THRESHOLD) | (conf_y > LOCALIZED_CONF_THRESHOLD)) & (fit_quality > GOOD_FIT_THRESHOLD)
    else:
        refresh = (conf_x > LOCALIZED_CONF_THRESHOLD) | (conf_y > LOCALIZED_CONF_THRESHOLD)
    refresh = refresh & ~just_injected
    new_anchor[:, 0] = torch.where(refresh, torch.ones_like(anchor[:, 0]), new_anchor[:, 0])
    new_anchor[:, 1] = torch.where(refresh, mx, new_anchor[:, 1])
    new_anchor[:, 2] = torch.where(refresh, my, new_anchor[:, 2])
    new_anchor[:, 3] = torch.where(refresh, torch.zeros_like(anchor[:, 3]), new_anchor[:, 3])
    new_anchor[:, 4] = torch.where(refresh, torch.zeros_like(anchor[:, 4]), new_anchor[:, 4])
    new_anchor[:, 5] = torch.where(refresh, torch.zeros_like(anchor[:, 5]), new_anchor[:, 5])
    new_anchor[:, 6] = torch.where(refresh, torch.zeros_like(anchor[:, 6]), new_anchor[:, 6])
    new_anchor[:, 7] = torch.where(refresh, torch.full_like(anchor[:, 7], -1.0), new_anchor[:, 7])
    new_anchor[:, 8:10] = torch.where(refresh.unsqueeze(1), torch.zeros_like(anchor[:, 8:10]), new_anchor[:, 8:10])
    new_anchor[:, 10:10 + TRIANGULATE_N_PROBE] = torch.where(
        refresh.unsqueeze(1), torch.zeros_like(anchor[:, 10:10 + TRIANGULATE_N_PROBE]), new_anchor[:, 10:10 + TRIANGULATE_N_PROBE])

    return out_p, new_anchor


def belief_conf(p):
    # conf_pos exactly as belief_read reports it, for the reward bonus
    mx = p[:, :, 0].mean(dim = 1, keepdim = True)
    my = p[:, :, 1].mean(dim = 1, keepdim = True)
    var = ((p[:, :, 0] - mx) ** 2 + (p[:, :, 1] - my) ** 2).mean(dim = 1)
    return torch.exp(-var / 0.02)


def belief_read(p, target = None):
    n, k, _ = p.shape
    seeds = SEED_POS.to(p.device)
    mx = p[:, :, 0].mean(dim = 1)
    my = p[:, :, 1].mean(dim = 1)
    sin_m = torch.sin(p[:, :, 2]).mean(dim = 1)
    cos_m = torch.cos(p[:, :, 2]).mean(dim = 1)
    r = torch.sqrt(sin_m * sin_m + cos_m * cos_m).clamp(min = 1e-9)
    var_x = ((p[:, :, 0] - mx.unsqueeze(1)) ** 2).mean(dim = 1)
    var_y = ((p[:, :, 1] - my.unsqueeze(1)) ** 2).mean(dim = 1)
    conf_pos = torch.exp(-(var_x + var_y) / 0.02)
    conf_x = torch.exp(-var_x / 0.02)
    conf_y = torch.exp(-var_y / 0.02)

    mean_pos = torch.stack([mx, my], dim = 1)
    d_all = (mean_pos.unsqueeze(1) - seeds.unsqueeze(0)).norm(dim = 2)
    nearest = d_all.argmin(dim = 1)
    tx = seeds[nearest, 0].unsqueeze(1) - p[:, :, 0]
    ty = seeds[nearest, 1].unsqueeze(1) - p[:, :, 1]
    bearing = torch.atan2(ty, tx) - p[:, :, 2]
    sin_b = torch.sin(bearing).mean(dim = 1)
    cos_b = torch.cos(bearing).mean(dim = 1)
    d_anchor = torch.sqrt(tx * tx + ty * ty).mean(dim = 1).clamp(max = 1.5)
    out = torch.stack([mx, my, sin_m / r, cos_m / r, conf_pos, r, sin_b, cos_b, d_anchor,
                       conf_x, conf_y], dim = 1)
    if target is None:
        return out
    # Bearing and distance to this robot's assigned target point, in the same
    # egocentric, per-particle-averaged form as the nearest-seed bearing above.
    # Averaging sin/cos per particle before combining, rather than computing one
    # bearing from the mean position, means particle disagreement shortens the
    # resultant vector -- the same self-reported confidence every other angular
    # quantity here carries. target is (n, 2) from observation.ensure_target;
    # optional, so callers that omit it keep the 11-value read-out.
    ttx = target[:, 0].unsqueeze(1) - p[:, :, 0]
    tty = target[:, 1].unsqueeze(1) - p[:, :, 1]
    t_bearing = torch.atan2(tty, ttx) - p[:, :, 2]
    sin_t = torch.sin(t_bearing).mean(dim = 1)
    cos_t = torch.cos(t_bearing).mean(dim = 1)
    # TARGET_DIST_CLAMP (3.0), unlike d_anchor's 1.5, is a safety ceiling
    # only, not a tuned scale: a target can legitimately sit anywhere on
    # the shape, up to the arena's full diagonal away (2.83 in these
    # normalized units), where d_anchor's own 1.5 is specifically sized for
    # "distance to the NEAREST seed," which is small by construction (seeds
    # are dense) -- reusing 1.5 here would clip real, common values, not
    # just outliers.
    d_target = torch.sqrt(ttx * ttx + tty * tty).mean(dim = 1).clamp(max = TARGET_DIST_CLAMP)
    return torch.cat([out, torch.stack([sin_t, cos_t, d_target], dim = 1)], dim = 1)


def belief_population_stats(belief, m, device):
    # Population-level localization diagnostics for one arena: summed conf_pos,
    # conf_x, conf_y, and the count of robots with conf_pos above
    # LOCALIZED_CONF_THRESHOLD, over m robots. Robots with no belief entry yet
    # count as zero confidence, same convention as belief_confidence_bonus in
    # reward.py. Returns sums, not means, so the trainer can accumulate across
    # arenas and ticks before dividing at the end of a rollout.
    if not belief:
        return 0.0, 0.0, 0.0, 0
    idx = [l for l in belief if l < m]
    if not idx:
        return 0.0, 0.0, 0.0, 0
    p = torch.stack([belief[l] for l in idx]).to(device)
    mx = p[:, :, 0].mean(dim = 1, keepdim = True)
    my = p[:, :, 1].mean(dim = 1, keepdim = True)
    var_x = ((p[:, :, 0] - mx) ** 2).mean(dim = 1)
    var_y = ((p[:, :, 1] - my) ** 2).mean(dim = 1)
    conf_pos = torch.exp(-(var_x + var_y) / 0.02)
    conf_x = torch.exp(-var_x / 0.02)
    conf_y = torch.exp(-var_y / 0.02)
    localized = int((conf_pos > LOCALIZED_CONF_THRESHOLD).sum())
    return float(conf_pos.sum()), float(conf_x.sum()), float(conf_y.sum()), localized

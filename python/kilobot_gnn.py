"""The networks: actors, the critic, and the shapes they agree on.

Three actor families -- DeepSet over a message database, a GRU over one typed
message per tick, and the event-driven split-observation actor -- plus the graph
critic and build_actor, which picks one from a Config.

Also the width constants every side depends on (MESSAGE_SIZE, SEED_SIZE,
NODE_FEATURES, the SPLIT_* layout). Changing any of them changes the wire
format or the checkpoint format, so they are not free to edit.

Dead-reckoning math lives in kinematics.py, not here.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

from belief import BELIEF_FEATURES, BELIEF_TARGET_FEATURES, LOCALIZED_CONF_THRESHOLD, SEED_SIZE
from kinematics import PROP_SIZE

# ---- config (tweak these) ----
# recommended: H1 < SEED_SIZE + MESSAGE_SIZE + Z (= 13 + Z), and H2 < H1
Z = 64
H1 = 24
H2 = 16
H3 = 32
RECENCY_CLAMP = 32 # recency (messages-ago) fed to Psi is clamped to this; no DB rows are dropped

# fixed architectural sizes
# SEED_SIZE now comes from belief.py (single source of truth for the landmark
# table's own size) rather than being duplicated here; it was two
# independently-hardcoded 5s that only agreed by coincidence before
WALL_SIZE = 4 # coarse wall-band channel (N, E, S, W), split actor only, kept separate from SEED_SIZE
              # so the gru and base actors (which slice vector[:, 2:2+SEED_SIZE] directly) are unaffected
MESSAGE_SIZE = 9
TRANSMISSION_SIZE = 12 # 9 message + 1 robot_id + 2 checksum (checksum handled elsewhere)
DB_ROW_SIZE = 11 # 9 message + 1 robot_id + 1 age
DB_CAPACITY = 64 # padded rows per agent for batched inference; must be >= maxKilobots
MOTOR_SIZE = 2

# ---- critic config (tweak these) ----
NODE_FEATURES = 19 # P(2) + H(2) + |D|(1) + dir_D(2) + C(1) + M(2) + T(9)
CRITIC_HIDDEN = 32
CRITIC_HEADS = 4
CRITIC_HEAD_DIM = 16


def empty_database(device="cpu", dtype=torch.float32):
    return torch.zeros((0, DB_ROW_SIZE), device=device, dtype=dtype)


class TransmissionParser(nn.Module):
    def __init__(self, z=Z, h1=H1, h2=H2, extra=0):
        super().__init__()
        self.linear1 = nn.Linear(SEED_SIZE + extra + MESSAGE_SIZE + z, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, MESSAGE_SIZE)
        self.relu = nn.ReLU()
        self.tanh = nn.Hardtanh()

    def forward(self, z, seed_robots, transmission, database):
        message_in = transmission[:MESSAGE_SIZE]
        robot_id = transmission[MESSAGE_SIZE]

        x = torch.cat([seed_robots, message_in, z], dim=0)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        embedded = self.tanh(self.linear3(x))

        return self._update_database(database, embedded, robot_id)

    def _update_database(self, database, embedded, robot_id):
        n = database.shape[0]
        if n > 0:
            new_age = database[:, -1].max() + 1.0
        else:
            new_age = torch.tensor(1.0, dtype=database.dtype, device=database.device)

        new_row = torch.cat([embedded, robot_id.view(1), new_age.view(1)]).unsqueeze(0)

        if n == 0:
            return new_row

        same = database[:, MESSAGE_SIZE] == robot_id
        if same.any():
            if new_age <= database[same, -1].max():
                return database
            database = database[~same]
        return torch.cat([database, new_row], dim=0)


class DeepSet(nn.Module):
    def __init__(self, h3=H3):
        super().__init__()
        self.psi_1 = nn.Linear(MESSAGE_SIZE + 1, h3) # message + age, robot_id dropped
        self.psi_2 = nn.Linear(h3, h3)
        self.rho_1 = nn.Linear(h3, h3)
        self.rho_2 = nn.Linear(h3, MESSAGE_SIZE + MOTOR_SIZE)
        self.relu = nn.ReLU()
        self.tanh = nn.Hardtanh()
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, database):
        age = database[:, -1:]
        staleness = (age.max() - age).clamp(max=RECENCY_CLAMP) if database.shape[0] else age
        feats = torch.cat([database[:, :MESSAGE_SIZE], staleness], dim=1)
        x = self.relu(self.psi_1(feats))
        x = self.tanh(self.psi_2(x))
        aggregated = x.sum(dim=0)
        h = self.relu(self.rho_1(aggregated))
        raw = self.rho_2(h)
        output_transmission = raw[:MESSAGE_SIZE]
        motor_commands = raw[MESSAGE_SIZE:]
        return output_transmission, motor_commands


class Actor(nn.Module):
    def __init__(self, z=Z, h1=H1, h2=H2, h3=H3, extra=0, direct=False):
        super().__init__()
        self.parser = TransmissionParser(z=z, h1=h1, h2=h2, extra=extra)
        self.deepset = DeepSet(h3=h3)
        # Diagnostic: a direct motor head that reads the robot's OWN parsed
        # message, bypassing the database sum, to test the dilution hypothesis.
        self.direct_motor = bool(direct)
        if self.direct_motor:
            self.direct_head = nn.Linear(MESSAGE_SIZE, MOTOR_SIZE)

    def forward(self, z, seed_robots, transmission, database):
        database = self.parser(z, seed_robots, transmission, database)
        output_transmission, motor_commands = self.deepset(database)
        return output_transmission, motor_commands, database


def actor_forward_batch(actor, z_b, seed_b, msg_b, sender_b, db_rows, db_valid):
    # z_b (N,Z)  seed_b (N,SEED_SIZE)  msg_b (N,MESSAGE_SIZE)  sender_b (N,)
    # db_rows (N,K,DB_ROW_SIZE)  db_valid (N,K) bool
    # Returns mean (N, MESSAGE_SIZE+MOTOR_SIZE), updated db_rows, db_valid.
    parser = actor.parser
    deepset = actor.deepset
    n = z_b.shape[0]

    x = torch.cat([seed_b, msg_b, z_b], dim=1)
    x = parser.relu(parser.linear1(x))
    x = parser.relu(parser.linear2(x))
    embedded = parser.tanh(parser.linear3(x))

    ages = db_rows[:, :, -1]
    valid_ages = torch.where(db_valid, ages, torch.zeros_like(ages))
    new_age = valid_ages.max(dim=1).values + 1.0

    senders = db_rows[:, :, MESSAGE_SIZE]
    same = db_valid & (senders == sender_b.unsqueeze(1))
    has_same = same.any(dim=1)
    same_idx = torch.argmax(same.float(), dim=1)
    free = ~db_valid
    has_free = free.any(dim=1)
    free_idx = torch.argmax(free.float(), dim=1)
    oldest_idx = torch.argmin(torch.where(db_valid, ages, torch.full_like(ages, float("inf"))), dim=1)
    no_same_slot = torch.where(has_free, free_idx, oldest_idx)
    target = torch.where(has_same, same_idx, no_same_slot)

    new_row = torch.cat([embedded, sender_b.unsqueeze(1), new_age.unsqueeze(1)], dim=1)
    db_rows = db_rows.clone()
    db_valid = db_valid.clone()
    rows = torch.arange(n, device=db_rows.device)
    db_rows[rows, target] = new_row
    db_valid[rows, target] = True

    ages2 = db_rows[:, :, -1]
    neg = torch.full_like(ages2, float("-inf"))
    max_age = torch.where(db_valid, ages2, neg).max(dim=1, keepdim=True).values
    staleness = (max_age - ages2).clamp(max=RECENCY_CLAMP)
    feats = torch.cat([db_rows[:, :, :MESSAGE_SIZE], staleness.unsqueeze(-1)], dim=2)

    h = deepset.relu(deepset.psi_1(feats))
    h = deepset.tanh(deepset.psi_2(h))
    h = h * db_valid.unsqueeze(-1)
    aggregated = h.sum(dim=1)
    g = deepset.relu(deepset.rho_1(aggregated))
    raw = deepset.rho_2(g)
    motor_pre = raw[:, MESSAGE_SIZE:]
    if getattr(actor, "direct_motor", False):
        # motors from the robot's own parsed message, undiluted by neighbors
        motor_pre = actor.direct_head(embedded)
    actor._motor_preact = motor_pre.detach()
    # Unbounded action mean: message dims raw, motor dims raw. The policy squashes the
    # SAMPLE (tanh into range) with a Jacobian-corrected log-prob, so the executed action
    # is the one whose probability is optimized. The db representation (embedded) keeps
    # its own Hardtanh above and is unaffected.
    mean = torch.cat([raw[:, :MESSAGE_SIZE], motor_pre], dim=1)
    return mean, db_rows, db_valid


# ---------------------------------------------------------------------------
# Recurrent (GRU) actor. Replaces the message database and the DeepSet sum with
# a per-robot GRU hidden state. Each tick consumes ONE typed message plus the
# robot's own proprioception (dead-reckoned displacement) and the target latent
# z. The hidden state carries memory across ticks and is what produces the
# output message and the (unbounded) motor mean; the policy squashes the sample.
#
# Typed message layout fed to the parser each tick:
#   seed_b (SEED_SIZE) : the seed observation -- which known-anchor seed(s) are in
#                        range. A DISTINCT input, not blended with the peer message.
#                        Privileged dir/heading probe columns are appended here.
#   msg_b  (MESSAGE_SIZE) : the sampled peer transmission (zeros if none this tick).
# Seed and peer content live in separate slots, so the GRU can tell an anchor
# sighting from a neighbour's belief by where the signal arrives.
GRU_HIDDEN = 128



class RecurrentActor(nn.Module):
    def __init__(self, z=Z, hidden=GRU_HIDDEN, extra=0, direct=False):
        super().__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Hardtanh()
        msg_in = SEED_SIZE + MESSAGE_SIZE + extra
        self.msg1 = nn.Linear(msg_in, 32)
        self.msg2 = nn.Linear(32, 32)
        gru_in = 32 + PROP_SIZE + z
        self.gru = nn.GRUCell(gru_in, hidden)
        self.head_msg = nn.Linear(hidden, MESSAGE_SIZE)
        self.head_motor = nn.Linear(hidden, MOTOR_SIZE)
        self.hidden_size = hidden
        # kept for interface compatibility with the DeepSet actor
        self.direct_motor = False

    def initial_hidden(self, n, device=None):
        return torch.zeros(n, self.hidden_size, device=device)

    def forward(self, z, seed_robots, transmission, prop, h_prev):
        # single-robot convenience path (deployment); batched path below is used in training
        z = z.unsqueeze(0) if z.dim() == 1 else z
        seed_robots = seed_robots.unsqueeze(0) if seed_robots.dim() == 1 else seed_robots
        transmission = transmission.unsqueeze(0) if transmission.dim() == 1 else transmission
        prop = prop.unsqueeze(0) if prop.dim() == 1 else prop
        if h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)
        mean, h_new = recurrent_forward_batch(self, z, seed_robots, transmission, prop, h_prev)
        out_msg = mean[:, :MESSAGE_SIZE]
        motor_pre = mean[:, MESSAGE_SIZE:]
        return out_msg.squeeze(0), motor_pre.squeeze(0), h_new.squeeze(0)


def recurrent_forward_batch(actor, z_b, seed_b, msg_b, prop_b, h_prev):
    # z_b (N,Z)  seed_b (N,SEED_SIZE+extra)  msg_b (N,MESSAGE_SIZE)  prop_b (N,PROP_SIZE)
    # h_prev (N,HIDDEN). Returns mean (N, MESSAGE_SIZE+MOTOR_SIZE) UNBOUNDED, h_new (N,HIDDEN).
    x = actor.relu(actor.msg1(torch.cat([seed_b, msg_b], dim=1)))
    parsed = actor.tanh(actor.msg2(x))
    gru_in = torch.cat([parsed, prop_b, z_b], dim=1)
    h_new = actor.gru(gru_in, h_prev)
    out_msg = actor.head_msg(h_new)
    motor_pre = actor.head_motor(h_new)
    actor._motor_preact = motor_pre.detach()
    mean = torch.cat([out_msg, motor_pre], dim=1)
    return mean, h_new




# Event-driven split-observation actor. Tc is MESSAGE_SIZE + 1 + SEED_SIZE +
# WALL_SIZE wide, ordered neighbor | corner | wall: exactly one of the
# neighbor slot (message content + received signal strength), the corner-
# seed slot (one-hot times strength), or the wall slot (strength per wall
# side, N/E/S/W) is populated, the rest are zero. The neighbor strength
# column is the robot's only ranging measurement of a peer (strength =
# 1/(1+d) in the simulator); dropping it, as an earlier version did,
# removes the geometric grounding that seed strengths already provide, so
# it is carried explicitly at index MESSAGE_SIZE.
#
# Changing SPLIT_TC_SIZE changes the actor's first-layer input width
# (SPLIT_TC_SIZE + SPLIT_ODOM_SIZE), so an older checkpoint will not load
# into an actor built from a different value -- a new run, not a resume.
SPLIT_NEIGHBOR_SIZE = MESSAGE_SIZE + 1
SPLIT_SEED_OFFSET = SPLIT_NEIGHBOR_SIZE
SPLIT_WALL_OFFSET = SPLIT_SEED_OFFSET + SEED_SIZE
SPLIT_TC_SIZE = SPLIT_NEIGHBOR_SIZE + SEED_SIZE + WALL_SIZE
# 8 tracker values + belief_read's default 11-value read-out + its 3-value
# relative target bearing/distance (BELIEF_TARGET_FEATURES) = 22. There is no
# image branch: once a robot's assigned point is resolved, the oracle's
# navigating steering never references the target image again, so these values
# are a sufficient replacement and the actor carries neither Z nor the encoder
# that consumed it. Changing this breaks checkpoint compatibility.
SPLIT_ODOM_SIZE = 8 + BELIEF_FEATURES + BELIEF_TARGET_FEATURES
# sin and cos of (heading_now - heading_anchor); see config.py's
# use_turn_anchor. Deliberately NOT folded into SPLIT_ODOM_SIZE, which is
# unconditional and sizes every actor built from this module -- an opt-in +2 is
# added at construction time instead, the same way use_arrived_head's extra
# output head is.
TURN_ANCHOR_SIZE = 2
SPLIT_UPSCALE_HIDDEN = 40
# MUST stay in sync with config.py's split_gru_hidden. Two constants for one
# concept: this is what SplitObservationActor() uses when constructed with no
# arguments, e.g. in tests, while config.py's field is what build_actor(cfg)
# passes at real construction time. Letting them diverge once left a
# parameter-budget test silently checking a configuration nothing built.
SPLIT_GRU_HIDDEN = 59
SPLIT_HEAD_HIDDEN = 40
# simple_oracle's five states; must stay equal to len(bc_replay.BC_STATES),
# which is the label set the auxiliary state head is trained against. Not
# imported from there to keep this module free of a dependency on the BC side.
N_ORACLE_STATES = 5
# north, east, south, west, in simple_oracle.WALL_NAMES' order, which is also
# the order of the wall slot inside Tc.
N_WALLS = 4

# The hidden activation used at up1 and head1. ReLU was the only option for a
# long time and is what every pre-2026-08 checkpoint was trained with; it is
# also what produced the dying-unit collapse of docs/tuning.md phase 154, where
# 13 of head1's 40 outputs were exactly zero for all 28392 decisions measured --
# permanently, since a ReLU whose pre-activation is negative everywhere has zero
# gradient as well as zero output. head_hidden is 40 wide because of the 24KB
# parameter budget, and a narrow layer has no redundancy to absorb that. The
# alternatives here all keep a nonzero gradient on the negative side, cost no
# parameters, and leave the checkpoint format unchanged -- which of them to use
# is an empirical question, so it is a switch rather than a replacement.
# Indices of the belief heading inside prop: prop is
# [neighbour tracker 4][seed tracker 4][belief_read 11][target 3][turn anchor 2]
# and belief_read is [mx, my, sin_h, cos_h, conf_pos, r, sin_b, cos_b, d_anchor,
# conf_x, conf_y]. Used by the steering feature below.
PROP_SIN_H = 8 + 2
PROP_COS_H = 8 + 3
# Remaining prop indices the oracle-form head reads. prop is
# [neighbour tracker 4][seed tracker 4][belief_read 11][target 3][turn anchor 2];
# belief_read's 5th value is conf_pos and the target triple is
# (sin, cos, distance) of the bearing to this robot's OWN assigned point,
# measured relative to its own heading -- which is exactly the (cross, dot)
# pair simple_oracle._steer takes in `navigating`. Nothing here is a new input:
# every one of these is already in the observation every checkpoint was trained
# on. See docs/tuning.md phase 160.
PROP_CONF_POS = 8 + 4
PROP_SIN_T = 8 + BELIEF_FEATURES + 0
PROP_COS_T = 8 + BELIEF_FEATURES + 1
# The target triple is (sin, cos, distance) of the bearing to this robot's own
# assigned point, from belief_read's target= path; the distance is the
# per-particle-averaged distance in the same normalized units as cfg.tau_v.
PROP_DIST_T = 8 + BELIEF_FEATURES + 2
# simple_oracle's own steering constants, reused rather than re-derived: getting
# these out of step with the teacher is the whole failure mode this head exists
# to remove. Imported by value, not by import, to keep this module free of a
# dependency on the oracle (kilobot_gnn is imported BY simple_oracle).
ORACLE_REACQUIRE_TURN = 0.45
ORACLE_STEER_BASE = 0.9
ORACLE_STEER_GAIN = 0.35
ORACLE_TURN_MOTOR = (0.9, 0.15)
# wall_following's approach slowdown: simple_oracle.APPROACH_SLOWDOWN_CONF is
# 0.5 * LOCALIZED_CONF_THRESHOLD and APPROACH_SLOWDOWN_MIN_SCALE is 0.15.
ORACLE_SLOWDOWN_LO = 0.5 * LOCALIZED_CONF_THRESHOLD
ORACLE_SLOWDOWN_HI = LOCALIZED_CONF_THRESHOLD
ORACLE_SLOWDOWN_MIN = 0.15
# squash_action maps a pre-activation u to 0.5*(tanh(u)+1), which reaches 0 and
# 1 only at -+inf. The oracle-form head composes its command in MOTOR space and
# inverts the squash, so it needs a bound strictly inside [0, 1]; atanh of
# +-0.998 is +-3.45, comfortably reachable and worth an error of 0.001 on a
# saturated command against the 0.0015 the unstructured head achieves there.
ORACLE_MOTOR_EPS = 1e-3
# simple_oracle's arrival condition, in the actor's OWN observation. The teacher
# stops a robot ("arrived", terminal) when the belief filter's estimate of the
# distance to the robot's assigned target falls below cfg.tau_v (normalized),
# after it has localized -- simple_oracle only checks arrival inside
# `navigating`, which it enters once conf_pos crosses LOCALIZED_CONF_THRESHOLD.
# belief_read's target= path emits both quantities in prop: d_target as the
# per-particle-averaged distance to the robot's own assigned point and conf_pos
# as the position-spread confidence. Computing the terminal decision instead of
# predicting it is the same thing the oracle-form head did for steering: the
# learned arrived head is accurate in the tape's distribution (0.99 recall at
# 0.95, per run_o3's val log) but under-fires on the deployment localisation
# shift, and a closed form cannot drift. This is the single source for the
# actor's stop at deployment (`use_closed_form_arrived` in config.py).
def closed_form_arrived(prop_b, tau_v, localized_conf = LOCALIZED_CONF_THRESHOLD):
    dist = prop_b[..., PROP_DIST_T]
    conf = prop_b[..., PROP_CONF_POS]
    # sin_t/cos_t are zeroed together when a robot has no assigned target
    # (observation.gather_split_state zeroes the target triple when ensure_target
    # couldn't resolve one), in which case d_target reads 0 and the rule would
    # false-positive immediately. Excluding it avoids ever stranding a targetless
    # robot; the much rarer case of a genuine zero resultant from particle
    # disagreement is self-correcting -- the next decision re-reads the filter
    # once it tightens -- whereas a false switch-off is permanent.
    has_target = (prop_b[..., PROP_SIN_T] != 0.0) | (prop_b[..., PROP_COS_T] != 0.0)
    return (dist < tau_v) & (conf >= localized_conf) & has_target
# sin(theta_wall_tangent - theta_heading) for the four walls, in WALL_NAMES order
# (north, east, south, west), whose tangents point along 0, -pi/2, pi, +pi/2:
#   north -> -sin_h,  east -> -cos_h,  south -> +sin_h,  west -> +cos_h
# and the matching cosines. Both are exact linear functions of the heading the
# actor already observes -- what the network cannot learn is not these values but
# the SELECTION among them, which is a product with a discrete latent it has to
# remember. Computing that product from its own wall head is what the steering
# feature does.
SPLIT_ACTIVATIONS = {"relu": nn.ReLU, "elu": nn.ELU, "silu": nn.SiLU,
                     "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}


def split_activation(name):
    cls = SPLIT_ACTIVATIONS.get(str(name or "relu"))
    if cls is None:
        raise ValueError("unknown split activation %r (have %s)"
                         % (name, ", ".join(sorted(SPLIT_ACTIVATIONS))))
    return cls()


class SplitObservationActor(nn.Module):
    def __init__(self, upscale_hidden = SPLIT_UPSCALE_HIDDEN, gru_hidden = SPLIT_GRU_HIDDEN,
                 head_hidden = SPLIT_HEAD_HIDDEN, use_arrived_head = False, use_turn_anchor = False,
                 recurrent = True, activation = "relu", use_state_head = False,
                 use_wall_head = False, use_steer_feature = False, use_oracle_head = False,
                 oracle_residual = 0.05, oracle_residual_turn = 0.0):
        super().__init__()
        # Kept under the name `relu` because split_forward_batch, every
        # checkpoint's key layout, and the ablation tooling all already refer to
        # it by that name; `activation_name` is what actually says which one it
        # is. Activations hold no parameters, so this never affects a
        # state_dict.
        self.activation_name = str(activation or "relu")
        self.relu = split_activation(self.activation_name)
        self.tanh = nn.Hardtanh()
        up_in = SPLIT_TC_SIZE + SPLIT_ODOM_SIZE + (TURN_ANCHOR_SIZE if use_turn_anchor else 0)
        self.up1 = nn.Linear(up_in, upscale_hidden)
        self.up2 = nn.Linear(upscale_hidden, upscale_hidden)
        self.recurrent = bool(recurrent)
        if recurrent:
            self.gru = nn.GRUCell(upscale_hidden, gru_hidden)
        else:
            self.gru = MemorylessAggregator(upscale_hidden, gru_hidden)
        self.head1 = nn.Linear(gru_hidden, head_hidden)
        self.head_msg = nn.Linear(head_hidden, MESSAGE_SIZE)
        # +2 inputs when the steering feature is on: the wall-gated sine and
        # cosine of (tangent - heading), which IS the oracle's steering law in
        # wall_following. Measured before building it: that law reconstructs the
        # teacher's command to 0.0000 mean error from observed quantities alone,
        # while trained networks sit at 0.15-0.25 and 6.5x the parameters does
        # not help -- so the gap is this one operation, not capacity or data.
        self.use_steer_feature = bool(use_steer_feature)
        self.head_motor = nn.Linear(head_hidden + (2 if use_steer_feature else 0), MOTOR_SIZE)
        # config.py's own use_arrived_head has the full rationale. None,
        # not just an unused-but-present Linear, when off -- so an
        # existing checkpoint's own state_dict (saved before this head
        # existed) still loads cleanly with no unexpected/missing keys.
        self.head_arrived = nn.Linear(head_hidden, 1) if use_arrived_head else None
        # Auxiliary, training-only: which of simple_oracle's five states the
        # teacher was in. The deployed forward never reads it -- act() uses
        # head_motor and head_arrived and nothing else -- so it costs nothing at
        # inference and 205 int8 bytes in the checkpoint. Its purpose is to put
        # the state machine INTO the recurrent representation: cloning the motor
        # command alone leaves "which state am I in" as an unsupervised latent
        # the GRU has to invent, and the states that need it most
        # (wall_following, navigating) are exactly the ones where the closed
        # loop falls apart. Gated the same way head_arrived is, so a checkpoint
        # saved without it still loads.
        self.head_state = nn.Linear(head_hidden, N_ORACLE_STATES) if use_state_head else None
        # Auxiliary, training-only, same deal as head_state: which of the four
        # walls the robot last touched. That is the one latent wall_following
        # genuinely cannot do without -- the oracle steers along
        # WALL_TANGENT[wall_name], which differs by 90 degrees between walls, so
        # a network unsure which wall it is on hedges, and a hedge on a steering
        # command is a constant turn. Measured on a real rollout: the oracle's
        # own turn command during wall_following averages 0.0015 in magnitude
        # while the clone's averages +0.108 SIGNED -- a persistent one-way turn,
        # which is a circle, which is why the swarm never gets anywhere.
        self.head_wall = nn.Linear(head_hidden, N_WALLS) if use_wall_head else None
        # config.py's own use_oracle_head has the full rationale. It adds no
        # parameters at all -- it reuses head_state, head_wall and head_motor --
        # but it does change the deployed forward pass, so a checkpoint must be
        # loaded with the same setting. Requires both auxiliary heads, since the
        # two discrete latents are what it mixes by.
        self.use_oracle_head = bool(use_oracle_head)
        self.oracle_residual = float(oracle_residual)
        self.oracle_residual_turn = float(oracle_residual_turn)
        if self.use_oracle_head and (self.head_state is None or self.head_wall is None):
            raise ValueError("use_oracle_head needs use_state_head and use_wall_head: the "
                             "mixture weights ARE those two heads' posteriors")
        self.hidden_size = gru_hidden
        self.direct_motor = False

    def initial_hidden(self, n, device = None):
        return torch.zeros(n, self.hidden_size, device = device)

    def forward(self, tc, prop, h_prev):
        tc = tc.unsqueeze(0) if tc.dim() == 1 else tc
        prop = prop.unsqueeze(0) if prop.dim() == 1 else prop
        if h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)
        mean, h_new = split_forward_batch(self, tc, prop, h_prev)
        out_msg = mean[:, :MESSAGE_SIZE]
        motor_pre = mean[:, MESSAGE_SIZE:]
        return out_msg.squeeze(0), motor_pre.squeeze(0), h_new.squeeze(0)


class MemorylessAggregator(nn.Module):
    # Parameter-matched, recurrence-free stand-in for the GRUCell, used only
    # for the ablation that isolates recurrence. A GRUCell(40, 59) holds
    # 3*59*40 + 3*59*59 + 2*3*59 = 17877 parameters; Linear(40, 178) ->
    # ReLU -> Linear(178, 59) holds 17859. Matching the parameter count
    # rather than the layer count is what makes the comparison about
    # recurrence instead of about capacity.
    def __init__(self, in_size, hidden, mid = None):
        super().__init__()
        if mid is None:
            gru_params = 3 * hidden * in_size + 3 * hidden * hidden + 6 * hidden
            mid = max(1, int(round((gru_params - hidden) / float(in_size + 1 + hidden))))
        self.fc1 = nn.Linear(in_size, mid)
        self.fc2 = nn.Linear(mid, hidden)
        self.relu = nn.ReLU()

    def forward(self, x, h_prev):
        # h_prev accepted and ignored: the caller's signature is unchanged so
        # every downstream path (BC, the val tape, actor_io) works untouched.
        return self.fc2(self.relu(self.fc1(x)))


def _oracle_steer(cross, dot):
    """simple_oracle._steer, differentiable and batched.

    Same law, same constants: turn is the cross product with the direction being
    steered toward when that direction is ahead, and a hard +-REACQUIRE_TURN
    when it is behind, because a cross product alone cannot distinguish "aligned"
    from "exactly reversed". Returns the two wheel commands in [0, 1].
    """
    hard = torch.where(cross >= 0, torch.full_like(cross, ORACLE_REACQUIRE_TURN),
                       torch.full_like(cross, -ORACLE_REACQUIRE_TURN))
    turn = torch.where(dot < 0, hard, cross)
    left = (ORACLE_STEER_BASE - ORACLE_STEER_GAIN * turn).clamp(0.0, 1.0)
    right = (ORACLE_STEER_BASE + ORACLE_STEER_GAIN * turn).clamp(0.0, 1.0)
    return left, right


def oracle_form_motor(actor, g, prop_b):
    """The motor command as a soft mixture over the teacher's own five commands.

    Measured, and this is the whole reason the head exists (docs/tuning.md phase
    160): the wheel pair the oracle emits is dominated by its COMMON mode, which
    is nearly constant, while everything that decides where a robot ends up lives
    in the DIFFERENTIAL, whose spread during wall_following is 0.0093 in the
    oracle's own steering variable. A plain two-output linear head fitted by MSE
    on the wheel pair reproduces the common mode and gets the differential wrong
    by 0.0895 rms -- ten times the signal, R^2 = -109 on held-out oracle data.
    The loss cannot see a channel holding 0.1% of the target variance.

    So the differential is not regressed here at all. Each of the five oracle
    states has a command that is a closed form of quantities the actor ALREADY
    observes, and this composes them:

      go_north        (1, 1)
      turning         ORACLE_TURN_MOTOR
      wall_following  _oracle_steer against the latched wall's tangent, whose
                      alignment with the belief heading is an exact linear
                      function of prop's own sin/cos of that heading; which wall
                      is the wall head's posterior
      navigating      _oracle_steer against the bearing to this robot's own
                      assigned point -- prop's target triple IS that bearing,
                      already relative to the robot's heading, so (cross, dot)
                      is read straight out of the observation
      arrived         (0, 0)

    mixed by the state head's posterior. Both heads already exist, both already
    reach ~99.8% on the clone's own rollouts, and neither is given a new input --
    what changes is that the network now supplies the two DISCRETE latents it is
    good at and the continuous steering is computed rather than fitted.

    head_motor is kept as a bounded residual so the mixture is a strong prior
    rather than a cage: it can correct the places the closed form is wrong (most
    of all `navigating`, where the teacher steers by its own private particle
    filter and the observation genuinely cannot reproduce it -- median offset
    0.75 degrees but 56 degrees of spread).

    Returns a pre-activation, because that is what the caller's contract is: the
    command is built in motor space and the squash inverted, so squash_action
    reproduces it exactly.
    """
    ps = torch.softmax(actor.head_state(g), dim = -1)
    pw = torch.softmax(actor.head_wall(g), dim = -1)
    sin_h = prop_b[..., PROP_SIN_H]
    cos_h = prop_b[..., PROP_COS_H]
    # sin/cos of (wall tangent - heading) for north, east, south, west, whose
    # tangents point along 0, -pi/2, pi, +pi/2. Exact linear functions of the
    # observed heading; the wall head's posterior selects among them.
    cand_sin = torch.stack([-sin_h, -cos_h, sin_h, cos_h], dim = -1)
    cand_cos = torch.stack([cos_h, -sin_h, -cos_h, sin_h], dim = -1)
    cross_w = (pw * cand_sin).sum(dim = -1)
    dot_w = (pw * cand_cos).sum(dim = -1)
    lw, rw = _oracle_steer(cross_w, dot_w)
    # wall_following's approach slowdown, from the same conf_pos the oracle
    # thresholds. Scales both wheels, so it moves the command without touching
    # the steering angle it encodes.
    conf = prop_b[..., PROP_CONF_POS]
    frac = ((conf - ORACLE_SLOWDOWN_LO) / (ORACLE_SLOWDOWN_HI - ORACLE_SLOWDOWN_LO)).clamp(0.0, 1.0)
    scale = 1.0 - frac * (1.0 - ORACLE_SLOWDOWN_MIN)
    lw, rw = lw * scale, rw * scale
    # navigating: belief_read averages sin/cos of the bearing over particles, so
    # the pair is a mean resultant whose length reports particle disagreement.
    # Normalising recovers the unit direction the oracle's own steering uses.
    sin_t = prop_b[..., PROP_SIN_T]
    cos_t = prop_b[..., PROP_COS_T]
    rt = (sin_t * sin_t + cos_t * cos_t).clamp(min = 1e-6).sqrt()
    ln, rn = _oracle_steer(sin_t / rt, cos_t / rt)
    one = torch.ones_like(sin_h)
    zero = torch.zeros_like(sin_h)
    cmd_l = torch.stack([one, one * ORACLE_TURN_MOTOR[0], lw, ln, zero], dim = -1)
    cmd_r = torch.stack([one, one * ORACLE_TURN_MOTOR[1], rw, rn, zero], dim = -1)
    motor = torch.stack([(ps * cmd_l).sum(dim = -1), (ps * cmd_r).sum(dim = -1)], dim = -1)
    # The residual, read as (common, differential) rather than (left, right).
    # Measured on the first run of this head: a residual free to move the two
    # wheels independently spends itself correcting the SPEED -- which the closed
    # form gets slightly wrong, because the approach slowdown reads the actor's
    # own conf_pos and the teacher reads its own filter's -- and injects
    # differential noise while doing it. Turning it off afterwards dropped the
    # median wall_following steering error from 0.0403 to 0.0006, a factor of 67,
    # for no change in anything else. So the two modes get separate scales:
    # generous on the common mode, where a correction is real and harmless, and
    # bounded on the differential by the size of the signal itself. The oracle's
    # own turn during wall_following has a standard deviation of 0.0093, and the
    # differential bound below is chosen so the learned correction cannot exceed
    # it at the operating point: 2 * res_turn * 1.8 / (0.7 * (L + R)), which is
    # res_turn / 0.35 at the full-speed L + R = 1.8, so 0.003 buys 0.0086. The
    # common half cannot change the differential at all, which is the exact half
    # of the property.
    if actor.oracle_residual > 0 or actor.oracle_residual_turn > 0:
        res = torch.tanh(actor.head_motor(g))
        common = actor.oracle_residual * res[..., 0:1]
        diff = actor.oracle_residual_turn * res[..., 1:2]
        motor = motor + torch.cat([common - diff, common + diff], dim = -1)
    motor = motor.clamp(ORACLE_MOTOR_EPS, 1.0 - ORACLE_MOTOR_EPS)
    # invert squash_action: it applies 0.5*(tanh(u)+1), and atanh undoes it
    # exactly, so the composition is the identity and its gradient is 1 -- the
    # tanh's own 5x attenuation at the 0.9 operating point, which is half of why
    # the steering channel never trained, is gone with it.
    return torch.atanh((2.0 * motor - 1.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6))


def split_motor_from_head(actor, g, prop_b):
    """The motor pre-activation from the shared head output.

    Split out so there is exactly ONE definition of it: bc_offline.py rolls the
    network over whole sequences with its own loop rather than calling
    split_forward_batch per step, and a second copy of this is precisely how a
    training path and a deployed path drift apart.

    With use_steer_feature on, the motor head additionally receives the wall-
    gated sine and cosine of (wall tangent - heading) -- the quantity
    simple_oracle's wall_following command is literally a function of. The gate
    is the wall head's own softmax rather than an argmax, so the gradient
    reaches it.
    """
    if getattr(actor, "use_oracle_head", False):
        return oracle_form_motor(actor, g, prop_b)
    if getattr(actor, "use_steer_feature", False) and getattr(actor, "head_wall", None) is not None:
        p = torch.softmax(actor.head_wall(g), dim = -1)
        sin_h = prop_b[..., PROP_SIN_H]
        cos_h = prop_b[..., PROP_COS_H]
        cand_sin = torch.stack([-sin_h, -cos_h, sin_h, cos_h], dim = -1)
        cand_cos = torch.stack([cos_h, -sin_h, -cos_h, sin_h], dim = -1)
        steer = (p * cand_sin).sum(dim = -1, keepdim = True)
        align = (p * cand_cos).sum(dim = -1, keepdim = True)
        return actor.head_motor(torch.cat([g, steer, align], dim = -1))
    return actor.head_motor(g)


def split_forward_batch(actor, tc_b, prop_b, h_prev):
    # No image branch: the GRU's output goes straight into the policy head. The
    # relative target bearing/distance inside prop_b replaces what the image
    # branch could have taught the actor about where to go.
    x = actor.relu(actor.up1(torch.cat([tc_b, prop_b], dim = 1)))
    x = actor.tanh(actor.up2(x))
    h_new = actor.gru(x, h_prev)
    g = actor.relu(actor.head1(h_new))
    out_msg = actor.head_msg(g)
    motor_pre = split_motor_from_head(actor, g, prop_b)
    actor._motor_preact = motor_pre.detach()
    # config.py's own use_arrived_head has the full rationale. Deliberately
    # NOT detached, unlike _motor_preact just above -- that one is a pure
    # runtime diagnostic never used in any loss; this one's entire purpose
    # is to be trained by bc.py's own new BCE loss term, which needs a
    # real gradient path back into head_arrived's own weights to do
    # anything at all. Safe to read at inference time regardless, since
    # act() always runs under torch.no_grad() during collection anyway.
    actor._arrived_logit = actor.head_arrived(g) if actor.head_arrived is not None else None
    actor._state_logits = actor.head_state(g) if actor.head_state is not None else None
    actor._wall_logits = actor.head_wall(g) if getattr(actor, "head_wall", None) is not None else None
    mean = torch.cat([out_msg, motor_pre], dim = 1)
    return mean, h_new






class Critic(nn.Module):
    def __init__(self, node_features=NODE_FEATURES, z=Z, hidden=CRITIC_HIDDEN,
                 heads=CRITIC_HEADS, head_dim=CRITIC_HEAD_DIM):
        super().__init__()
        gnn_dim = heads * head_dim

        # Optional handicap: blind the critic to chosen privileged feature columns so it
        # can no longer trivially predict the distance-based return. Tests advantage collapse.
        # Columns: pos 0:2, heading 2:4, dist 4, dir 5:7, neighbor-dist 7, motors 8:10, msg 10:19.
        blind = os.environ.get("KILOBOT_CRITIC_BLIND", "none").strip().lower()
        groups = {"none": [], "dist": [4], "dir": [5, 6], "distdir": [4, 5, 6],
                  "spatial": [0, 1, 2, 3, 4, 5, 6]}
        cols = groups.get(blind)
        if cols is None:
            cols = [int(c) for c in blind.split(",") if c.strip() != ""]
        mask = torch.ones(node_features)
        for c in cols:
            if 0 <= c < node_features:
                mask[c] = 0.0
        self.register_buffer("in_mask", mask)
        self.blind_cols = cols

        self.encoder = nn.Sequential(
            nn.Linear(node_features, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU()
        )

        self.gat1 = GATv2Conv(hidden, head_dim, heads=heads, concat=True,
                              edge_dim=1, add_self_loops=True, fill_value=1.0)
        self.norm1 = nn.LayerNorm(gnn_dim)

        self.gat2 = GATv2Conv(gnn_dim, head_dim, heads=heads, concat=True,
                              edge_dim=1, add_self_loops=True, fill_value=1.0)
        self.norm2 = nn.LayerNorm(gnn_dim)

        self.head = nn.Sequential(
            nn.Linear(gnn_dim * 3 + z, gnn_dim),
            nn.ELU(),
            nn.Linear(gnn_dim, 1)
        )

    def forward(self, x, edge_attr, edge_index, z, batch=None):
        x = x * self.in_mask
        x = self.encoder(x)

        x = F.elu(self.norm1(self.gat1(x, edge_index, edge_attr)))
        x = F.elu(self.norm2(self.gat2(x, edge_index, edge_attr)))

        if batch is None:
            n = x.size(0)
            mean = x.mean(dim=0, keepdim=True).expand(n, -1)
            mx = x.amax(dim=0, keepdim=True).expand(n, -1)
            z = z.view(1, -1).expand(n, -1)
        else:
            mean = global_mean_pool(x, batch)[batch]
            mx = global_max_pool(x, batch)[batch]
            z = z[batch]

        x = torch.cat([x, mean, mx, z], dim=1)
        return self.head(x)

# --- Diagnostic instrumentation (overnight bottleneck study) ---------------
# Privileged actor observations: which node-feature columns to append to the
# seed vector that feeds the actor. Used only to localize the bottleneck; the
# normal training path uses mode "none" and is unchanged.
# node layout: P(0:2) H(2:4) |D|(4) dir_D(5:7) C(7) M(8:10) T(10:19)
PRIV_COLS = {
    "none": [],
    "dir": [5, 6],            # direction to nearest stroke pixel (privileged)
    "heading": [2, 3],        # own heading
    "dir_heading": [2, 3, 5, 6],  # heading + direction to shape (the oracle's inputs)
    "pose": [0, 1, 2, 3],     # own position + heading
    "full": [0, 1, 2, 3, 5, 6],  # pose + direction to shape
}


def priv_cols(mode):
    return PRIV_COLS.get(mode, [])


def widths_from_state_dict(sd):
    """The three split-actor widths, read off a checkpoint's own tensors.

    A checkpoint records which HEADS it was built with, but for a long time it
    recorded nothing about how WIDE they were, so anything trained with
    --gru-hidden could not be loaded back: build_actor used config.py's default
    and load_state_dict rejected it on a shape mismatch. The widths are fully
    determined by the tensors themselves, which is both simpler than a new meta
    field and correct for every checkpoint ever written, including the ones from
    before that field existed. save_actor now records them too; this is what
    reads them when it did not.

    Returns {} for a state_dict that is not a split actor.
    """
    if "gru.weight_hh" not in sd or "head1.weight" not in sd or "up1.weight" not in sd:
        return {}
    return {"gru_hidden": int(sd["gru.weight_hh"].shape[1]),
            "head_hidden": int(sd["head1.weight"].shape[0]),
            "upscale_hidden": int(sd["up1.weight"].shape[0])}


def build_actor(cfg):
    """The actor a Config asks for.

    Lived in two places for a long time -- launch.py and
    two entry points -- with fourteen modules importing it from the
    latter, which is why it now sits beside the actor classes themselves. Both
    old call sites still resolve: launch.py re-exports it, and the replica
    harness is gone.
    """
    actor_type = getattr(cfg, "actor_type", "deepset")
    if actor_type == "gru_split_observation":
        return SplitObservationActor(upscale_hidden = cfg.split_upscale_hidden,
                                     gru_hidden = cfg.split_gru_hidden,
                                     head_hidden = cfg.split_head_hidden,
                                     use_arrived_head = getattr(cfg, "use_arrived_head", False),
                                     use_turn_anchor = getattr(cfg, "use_turn_anchor", False),
                                     recurrent = getattr(cfg, "actor_recurrent", True),
                                     activation = getattr(cfg, "split_activation", "relu"),
                                     use_state_head = getattr(cfg, "use_state_head", False),
                                     use_wall_head = getattr(cfg, "use_wall_head", False),
                                     use_steer_feature = getattr(cfg, "use_steer_feature", False),
                                     use_oracle_head = getattr(cfg, "use_oracle_head", False),
                                     oracle_residual = getattr(cfg, "oracle_residual", 0.05),
                                     oracle_residual_turn = getattr(cfg, "oracle_residual_turn", 0.0))
    extra = len(priv_cols(cfg.actor_priv_mode))
    if actor_type == "gru":
        return RecurrentActor(extra = extra, hidden = cfg.gru_hidden)
    return Actor(extra = extra, direct = cfg.direct_motor)

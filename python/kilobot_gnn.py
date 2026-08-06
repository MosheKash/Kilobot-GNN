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

from belief import BELIEF_FEATURES, BELIEF_TARGET_FEATURES, SEED_SIZE
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


class SplitObservationActor(nn.Module):
    def __init__(self, upscale_hidden = SPLIT_UPSCALE_HIDDEN, gru_hidden = SPLIT_GRU_HIDDEN,
                 head_hidden = SPLIT_HEAD_HIDDEN, use_arrived_head = False, use_turn_anchor = False,
                 recurrent = True):
        super().__init__()
        self.relu = nn.ReLU()
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
        self.head_motor = nn.Linear(head_hidden, MOTOR_SIZE)
        # config.py's own use_arrived_head has the full rationale. None,
        # not just an unused-but-present Linear, when off -- so an
        # existing checkpoint's own state_dict (saved before this head
        # existed) still loads cleanly with no unexpected/missing keys.
        self.head_arrived = nn.Linear(head_hidden, 1) if use_arrived_head else None
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


def split_forward_batch(actor, tc_b, prop_b, h_prev):
    # No image branch: the GRU's output goes straight into the policy head. The
    # relative target bearing/distance inside prop_b replaces what the image
    # branch could have taught the actor about where to go.
    x = actor.relu(actor.up1(torch.cat([tc_b, prop_b], dim = 1)))
    x = actor.tanh(actor.up2(x))
    h_new = actor.gru(x, h_prev)
    g = actor.relu(actor.head1(h_new))
    out_msg = actor.head_msg(g)
    motor_pre = actor.head_motor(g)
    actor._motor_preact = motor_pre.detach()
    # config.py's own use_arrived_head has the full rationale. Deliberately
    # NOT detached, unlike _motor_preact just above -- that one is a pure
    # runtime diagnostic never used in any loss; this one's entire purpose
    # is to be trained by bc.py's own new BCE loss term, which needs a
    # real gradient path back into head_arrived's own weights to do
    # anything at all. Safe to read at inference time regardless, since
    # act() always runs under torch.no_grad() during collection anyway.
    actor._arrived_logit = actor.head_arrived(g) if actor.head_arrived is not None else None
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
                                     recurrent = getattr(cfg, "actor_recurrent", True))
    extra = len(priv_cols(cfg.actor_priv_mode))
    if actor_type == "gru":
        return RecurrentActor(extra = extra, hidden = cfg.gru_hidden)
    return Actor(extra = extra, direct = cfg.direct_motor)

import os
import sys
# these one-off probes live in tools/ but import the training modules that
# sit one level up, in python/ -- run them from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import torch

if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
    print("""usage: python tools/check_actor.py <actor.pt> <val_tape.pt>

Per-oracle-state mean motor, fraction near zero, and mean P(arrived) for a
trained checkpoint, scored against a recorded validation tape.""")
    raise SystemExit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
from config import Config
from policy import GaussianPolicy, squash_action
from kilobot_gnn import build_actor
from kilobot_gnn import MESSAGE_SIZE, split_forward_batch
from val_tape import load_tape
from bc_replay import BC_STATES

cfg = Config()
cfg.actor_type = "gru_split_observation"
cfg.use_arrived_head = True
cfg.use_turn_anchor = True
policy = GaussianPolicy(build_actor(cfg), cfg.log_std_init)
ck = torch.load(sys.argv[1], map_location = "cpu", weights_only = False)
policy.actor.load_state_dict(ck["actor"])
tape = load_tape(sys.argv[2])

actor = policy.actor
T, R = tape["valid"].shape
h = actor.initial_hidden(R)
sums = {}
counts = {}
zeros = {}
probs = {}
with torch.no_grad():
    for t in range(T):
        v = tape["valid"][t]
        if not bool(v.any()):
            continue
        mean, h_new = split_forward_batch(actor, tape["tc"][t], tape["prop"][t], h)
        h = torch.where(v.unsqueeze(1), h_new, h)
        mot = squash_action(mean)[:, MESSAGE_SIZE:]
        p = torch.sigmoid(actor._arrived_logit).squeeze(-1)
        st = tape["state"][t]
        for s, name in enumerate(BC_STATES):
            m = v & (st == s)
            if not bool(m.any()):
                continue
            sums[name] = sums.get(name, 0.0) + float(mot[m].sum())
            counts[name] = counts.get(name, 0) + int(m.sum()) * 2
            zeros[name] = zeros.get(name, 0) + int((mot[m] < 0.05).sum())
            probs[name] = probs.get(name, 0.0) + float(p[m].sum())

print("checkpoint iteration:", ck.get("iteration"))
print("%-16s %10s %14s %14s" % ("oracle state", "mean motor", "frac < 0.05", "mean P(arrived)"))
for name in BC_STATES:
    if name in counts:
        n = counts[name]
        print("%-16s %10.3f %14.4f %14.3f"
              % (name, sums[name] / n, zeros[name] / n, probs.get(name, 0.0) / (n / 2)))

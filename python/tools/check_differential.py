import os
import sys
# these one-off probes live in tools/ but import the training modules that
# sit one level up, in python/ -- run them from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import torch

if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
    print("""usage: python tools/check_differential.py <actor.pt> <val_tape.pt>

Predicted vs oracle motor differential (L-R) per oracle state -- how sharply
the actor turns compared with its teacher.""")
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
pd = {}
td = {}
ps = {}
ts = {}
n = {}
with torch.no_grad():
    for t in range(T):
        v = tape["valid"][t]
        if not bool(v.any()):
            continue
        mean, h_new = split_forward_batch(actor, tape["tc"][t], tape["prop"][t], h)
        h = torch.where(v.unsqueeze(1), h_new, h)
        mot = squash_action(mean)[:, MESSAGE_SIZE:]
        tgt = tape["tgt"][t]
        st = tape["state"][t]
        for s, name in enumerate(BC_STATES):
            m = v & (st == s)
            if not bool(m.any()):
                continue
            pd[name] = pd.get(name, 0.0) + float((mot[m][:, 0] - mot[m][:, 1]).sum())
            td[name] = td.get(name, 0.0) + float((tgt[m][:, 0] - tgt[m][:, 1]).sum())
            ps[name] = ps.get(name, 0.0) + float((mot[m][:, 0] - mot[m][:, 1]).abs().sum())
            ts[name] = ts.get(name, 0.0) + float((tgt[m][:, 0] - tgt[m][:, 1]).abs().sum())
            n[name] = n.get(name, 0) + int(m.sum())

print("checkpoint iteration:", ck.get("iteration"))
print("%-16s %8s %12s %12s %12s %12s" % ("state", "n", "pred L-R", "oracle L-R", "pred |L-R|", "oracle |L-R|"))
for name in BC_STATES:
    if name in n:
        c = float(n[name])
        print("%-16s %8d %12.4f %12.4f %12.4f %12.4f"
              % (name, n[name], pd[name] / c, td[name] / c, ps[name] / c, ts[name] / c))

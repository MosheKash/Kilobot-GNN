import os
import sys
# these one-off probes live in tools/ but import the training modules that
# sit one level up, in python/ -- run them from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import torch

if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
    print("""usage: python tools/sweep_threshold.py <actor.pt> <val_tape.pt>

Sweeps the arrived-head decision threshold and reports precision/recall/F1 at
each, against a recorded validation tape.""")
    raise SystemExit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
from config import Config
from policy import GaussianPolicy
from kilobot_gnn import build_actor
from kilobot_gnn import split_forward_batch
from val_tape import load_tape

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
ps = []
ls = []
with torch.no_grad():
    for t in range(T):
        v = tape["valid"][t] & tape["arrived_valid"][t]
        alive = tape["valid"][t]
        mean, h_new = split_forward_batch(actor, tape["tc"][t], tape["prop"][t], h)
        h = torch.where(alive.unsqueeze(1), h_new, h)
        if bool(v.any()):
            ps.append(torch.sigmoid(actor._arrived_logit).squeeze(-1)[v])
            ls.append(tape["arrived"][t][v] > 0.5)
p = torch.cat(ps)
lab = torch.cat(ls)
print("held-out arrived-label rate %.4f  (n=%d)" % (float(lab.float().mean()), int(lab.numel())))
for name, sel in (("arrived", lab), ("not arrived", ~lab)):
    if int(sel.sum()) == 0:
        print("  P(arrived) on %-12s no samples in this tape" % name)
        continue
    q = torch.quantile(p[sel].double(), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]).double())
    print("  P(arrived) on %-12s p5 %.3f  p25 %.3f  median %.3f  p75 %.3f  p95 %.3f"
          % (name, q[0], q[1], q[2], q[3], q[4]))
print()
print("%10s %10s %10s %10s" % ("threshold", "precision", "recall", "F1"))
for thr in (0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2):
    pred = p > thr
    tp = int((pred & lab).sum())
    fp = int((pred & (~lab)).sum())
    fn = int(((~pred) & lab).sum())
    pr = tp / float(tp + fp) if tp + fp else float("nan")
    rc = tp / float(tp + fn) if tp + fn else float("nan")
    f1 = 2 * pr * rc / (pr + rc) if (pr == pr and rc == rc and pr + rc > 0) else 0.0
    print("%10.2f %10.3f %10.3f %10.3f" % (thr, pr, rc, f1))

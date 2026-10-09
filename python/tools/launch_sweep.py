"""launch_sweep.py -- Path 2: gate retune sweep on the o3 actor.

Matrix style, all rows on the fixed seed set (--ticks 10000 --instances 2
--arenas 5 --swarm-rng 500 --seed 7 --bake-rotation-steps 0). Rows are
launched with setsid so they survive the caller; each row gets a distinct
--base-port range so the Unity players do not collide.

usage:
  python tools/launch_sweep.py --list
  python tools/launch_sweep.py --launch --batch 3     # start remaining rows
  python tools/launch_sweep.py --watch                # wait, then score into rows.tsv
"""

import argparse
import os
import subprocess
import sys
import time

PY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # python/
ROOT = os.path.dirname(PY)
OUT_DIR = os.path.join(ROOT, "results", "plan_execution", "sweep")
LOG_DIR = "/tmp/opencode"
WEIGHTS = os.path.join(ROOT, "results", "bc_v2", "run_o3", "actor_best.pt")
COMMON = ["--ticks", "10000", "--instances", "2", "--arenas", "5",
          "--swarm-rng", "500", "--seed", "7", "--bake-rotation-steps", "0"]

# name -> (arg list). Closed-form-only rows drop --closed-form-hybrid; AND
# rows use --closed-form-and; the learned-head threshold is swept only where it
# is load-bearing (hybrid / AND modes).
ROWS = [
    # OR hybrid: threshold sweep at the baseline arrival radius
    ("or_thr0p95_cf0p06", ["--arrived-threshold", "0.95", "--closed-form-arrived",
                           "--closed-form-dist", "0.06", "--closed-form-hybrid"]),
    ("or_thr0p95_cf0p10", ["--arrived-threshold", "0.95", "--closed-form-arrived",
                           "--closed-form-dist", "0.10", "--closed-form-hybrid"]),
    ("or_thr0p90_cf0p08", ["--arrived-threshold", "0.90", "--closed-form-arrived",
                           "--closed-form-dist", "0.08", "--closed-form-hybrid"]),
    ("or_thr0p80_cf0p08", ["--arrived-threshold", "0.80", "--closed-form-arrived",
                           "--closed-form-dist", "0.08", "--closed-form-hybrid"]),
    ("or_thr0p70_cf0p08", ["--arrived-threshold", "0.70", "--closed-form-arrived",
                           "--closed-form-dist", "0.08", "--closed-form-hybrid"]),
    ("or_thr0p50_cf0p08", ["--arrived-threshold", "0.50", "--closed-form-arrived",
                           "--closed-form-dist", "0.08", "--closed-form-hybrid"]),
    # OR hybrid: promising corner of both knobs
    ("or_thr0p80_cf0p10", ["--arrived-threshold", "0.80", "--closed-form-arrived",
                           "--closed-form-dist", "0.10", "--closed-form-hybrid"]),
    ("or_thr0p70_cf0p10", ["--arrived-threshold", "0.70", "--closed-form-arrived",
                           "--closed-form-dist", "0.10", "--closed-form-hybrid"]),
    # closed-form only: radius is the only knob
    ("cf_cf0p00", ["--closed-form-arrived", "--closed-form-dist", "0.00"]),
    ("cf_cf0p06", ["--closed-form-arrived", "--closed-form-dist", "0.06"]),
    ("cf_cf0p10", ["--closed-form-arrived", "--closed-form-dist", "0.10"]),
    # AND gate: stop only when BOTH branches agree
    ("and_thr0p95_cf0p08", ["--arrived-threshold", "0.95", "--closed-form-arrived",
                            "--closed-form-dist", "0.08", "--closed-form-and"]),
    ("and_thr0p80_cf0p08", ["--arrived-threshold", "0.80", "--closed-form-arrived",
                            "--closed-form-dist", "0.08", "--closed-form-and"]),
    ("and_thr0p95_cf0p10", ["--arrived-threshold", "0.95", "--closed-form-arrived",
                            "--closed-form-dist", "0.10", "--closed-form-and"]),
]

PORTS = [5500, 5800, 6100, 6400, 6700]

# Every row gets its OWN base port range so that parallel rows never try to
# re-bind a port a previous row's Unity players have not released yet (the 5
# shared ports above caused UnityWorkerInUseException in the 2nd+ batch).
ROW_PORT = {}


def _assign_ports():
    # ports derived from the row index; disjoint from PORTS and from each other
    for i, (n, _) in enumerate(ROWS):
        ROW_PORT[n] = 9000 + i * 80


_assign_ports()


def out(name):
    return os.path.join(OUT_DIR, "eval_%s.json" % name)


def log(name):
    return os.path.join(LOG_DIR, "sweep_%s.log" % name)


def finished(name):
    return os.path.exists(out(name))


def running(name):
    return os.path.exists(log(name)) and not finished(name)


def launch(rows, port):
    name, args = rows.pop(0)
    cmd = [sys.executable, "tools/eval_closed_loop.py", out(name), "--mode", "actor",
           "--weights", WEIGHTS, "--base-port", str(port)] + args + COMMON
    full = "setsid %s > %s 2>&1 < /dev/null & echo $!" % (" ".join(cmd), log(name))
    pid = subprocess.run(["bash", "-c", full], capture_output = True, text = True,
                         cwd = PY).stdout.strip()
    return name, pid


def launch_named(name, args, port):
    cmd = [sys.executable, "tools/eval_closed_loop.py", out(name), "--mode", "actor",
           "--weights", WEIGHTS, "--base-port", str(port)] + args + COMMON
    full = "setsid %s > %s 2>&1 < /dev/null & echo $!" % (" ".join(cmd), log(name))
    pid = subprocess.run(["bash", "-c", full], capture_output = True, text = True,
                         cwd = PY).stdout.strip()
    return pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action = "store_true")
    ap.add_argument("--launch", action = "store_true")
    ap.add_argument("--batch", type = int, default = 3)
    ap.add_argument("--watch", action = "store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok = True)
    os.makedirs(LOG_DIR, exist_ok = True)

    pending_names = [n for n, _ in ROWS if not finished(n)]
    if args.list:
        print("%d rows; %d pending" % (len(ROWS), len(pending_names)))
        for n, a in ROWS:
            print("  %24s  %s" % (n, " ".join(a)) + ("" if finished(n) else "   [pending]"))
        return

    if args.launch:
        todo = [[n, a] for n, a in ROWS if not finished(n)]
        slot = 0
        while todo:
            # count current live rows (any row with a log younger than 2h that
            # has not produced output)
            live = sum(1 for n, _ in ROWS
                       if os.path.exists(log(n)) and not finished(n)
                       and time.time() - os.path.getmtime(log(n)) < 7200)
            free = args.batch - live
            if free > 0 and todo:
                for _ in range(free):
                    if not todo:
                        break
                    name, args = todo.pop(0)
                    pid = launch_named(name, args, ROW_PORT[name])
                    slot += 1
                    print("%s launched (port %d, pid %s)" % (name, ROW_PORT[name], pid),
                          flush = True)
            if todo:
                time.sleep(60)
        print("all rows launched", flush = True)

    if args.watch:
        todo = [n for n, _ in ROWS if not finished(n)]
        if not todo:
            print("all rows already finished")
        while todo:
            done = [n for n in todo if finished(n)]
            for n in done:
                todo.remove(n)
                print("done %s" % n, flush = True)
            if todo:
                print("waiting on %d: %s" % (len(todo), ", ".join(todo)), flush = True)
                time.sleep(60)
        scorer = os.path.join(ROOT, "results", "plan_execution", "score_eval.py")
        tsv = os.path.join(OUT_DIR, "rows.tsv")
        jsons = [out(n) for n, _ in ROWS if finished(n)]
        if jsons:
            subprocess.run([sys.executable, scorer, "-o", tsv] + jsons, cwd = PY)
            with open(tsv) as f:
                sys.stdout.write(f.read())


if __name__ == "__main__":
    main()
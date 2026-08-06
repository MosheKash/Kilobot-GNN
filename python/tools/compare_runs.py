import os
import re
import sys

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

logs = sys.argv[1:] or [os.path.join(_LOG_DIR, n) for n in ("smoke_replay.log", "smoke_gate.log")]

for name in logs:
    try:
        lines = open(name).read().splitlines()
    except OSError as exc:
        print("== %s -- cannot read (%s)" % (name, exc))
        continue
    print("== %s" % name)

    tape = []
    for l in lines:
        m = re.search(r"iter (\d+)\s+val_tape ([\d.]+) \(smoothed ([\d.]+)\)\s+(.*?)\s+cold_start_moved ([\d.]+)", l)
        if m:
            body = m.group(4)
            states = dict((k, float(v)) for k, v in re.findall(r"([a-z_]+)\*? ([\d.]+)", body))
            head = re.search(r"arrived_head P (\S+) R (\S+) F1 ([\d.]+)", body)
            tape.append((int(m.group(1)), float(m.group(2)), float(m.group(3)),
                         states, head, float(m.group(5))))
    if tape:
        print("   selection criterion (held-out imitation error, lower is better)")
        for it, raw, sm, st, head, moved in tape:
            four = [st[k] for k in ("go_n", "turn", "wall", "navi") if k in st]
            hd = ""
            if head:
                hd = "  head P %s R %s F1 %s" % (head.group(1), head.group(2), head.group(3))
            print("     iter %3d  raw %.5f  smoothed %.5f  [%s]  arri %s%s  moved %.2f"
                  % (it, raw, sm,
                     " ".join("%.4f" % v for v in four),
                     ("%.4f" % st["arri"]) if "arri" in st else "n/a", hd, moved))
        sm = [t[2] for t in tape]
        print("   smoothed monotonically decreasing: %s" % all(b <= a for a, b in zip(sm, sm[1:])))
    else:
        print("   (no val_tape lines -- pre-phase-157 log)")

    agree = []
    for l in lines:
        m = re.search(r"iter (\d+)\s+arrived_agreement both=(\d+) actor_only=(\d+) shadow_only=(\d+)", l)
        if m:
            agree.append(tuple(int(x) for x in m.groups()))
    if agree:
        print("   arrived head, training-time census")
        last = agree[-1][0]
        for it, b, ao, so in agree:
            if it % 10 != 9 and it != last:
                continue
            pr = b / float(b + ao) if b + ao else float("nan")
            rc = b / float(b + so) if b + so else float("nan")
            print("     iter %3d  claims %6d  precision %.3f  recall %.3f" % (it, b + ao, pr, rc))

    res = []
    for l in lines:
        m = re.search(r"reservoir (\d+) samples over (\d+) states\s+(.*)", l)
        if m:
            res.append((int(m.group(1)), int(m.group(2)),
                        dict((k, int(v)) for k, v in re.findall(r"([a-z_]+)=(\d+)", m.group(3)))))
    if res:
        print("   reservoir: %d samples over %d states at the end" % (res[-1][0], res[-1][1]))
        print("     %s" % res[-1][2])

    mo = [float(x) for x in re.findall(r"motor_only ([\d.]+)", "\n".join(lines))]
    if mo:
        print("   motor_only: %.5f -> %.5f over %d iterations" % (mo[0], mo[-1], len(mo)))
    rss = [int(x) for x in re.findall(r"memory \(RSS\) (\d+) MB", "\n".join(lines))]
    vram = [int(x) for x in re.findall(r"memory \(VRAM reserved\) (\d+) MB", "\n".join(lines))]
    if rss:
        print("   RSS %d -> %d MB   VRAM %d -> %d MB" % (rss[0], rss[-1], vram[0], vram[-1]))
    print()

"""Print selected raw metrics from an ncu CSV export."""
import csv, sys
rows = list(csv.reader(open(sys.argv[1], encoding="utf-8-sig")))
h, units = rows[0], rows[1]
ki = h.index("Kernel Name")
for r in rows[2:]:
    print("==", r[ki])
    for i, c in enumerate(h):
        if c.startswith(("smsp", "sm__", "l1tex", "dram")):
            name = c.replace("smsp__average_warp_latency_issue_stalled_", "stall_")
            print(f"   {name:62s} {r[i]:>16s} {units[i]}")

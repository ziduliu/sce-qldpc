"""Print sbatch commands for the physical-error-rate sweep and the p=0.001
re-measurements: one deepverify_pl.sbatch job per (code, leg) on two
single-GPU nodes (one rank per basis), stopping at 1000 failures.

  python pl_sweep_gen2.py sweep > sweep.sh      # 20 codes x 5 legs
  python pl_sweep_gen2.py p001  > p001.sh       # luna pL champions at p=0.001
"""
import json
import sys

MODE = sys.argv[1]
SB = ("sbatch --parsable --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=8 --mem=60G "
      "--export=ALL,ARCHIVE=%s,OUT=%s,TAG=%s,LEGS=%s:%d,MAXERR=1000,HASH=\"c%d:%d:%s\" deepverify_pl.sbatch")
print("set -e; cd /u/zdliu/lp/plevolve_20260912")

if MODE == "sweep":
    OUT = "/u/zdliu/lp/oe_run/pl_sweep_20260928"
    DIST_ARCHIVE = "/u/zdliu/lp/oe_run/cc_v7_gpt6sol50_w12_10c_cmp_out/archive"
    PL_ARCHIVE = "/u/zdliu/lp/oe_run/pl10_gpt6sol50_out/archive"
    LEGS = [("0.0015", 2000000), ("0.002", 1000000), ("0.003", 300000), ("0.004", 100000), ("0.005", 50000)]
    dist = {0: (46, "333c995d3864fbe2"), 1: (37, "2577a31b81be5aaa"), 2: (117, "020a5c49a3eb579e"), 3: (48, "094fc553fda8aac6"),
            4: (19, "c96176f340d8a0c4"), 5: (32, "4133e0d218f93320"), 6: (55, "f78413d70c2d5124"), 7: (24, "51a7a7053f60c7ed"),
            8: (36, "783dd73a1bbf90d9"), 9: (28, "e10396e73a461640")}
    q_of = {e["cell"]: e["q"] for e in json.load(open("/u/zdliu/lp/oe_run/pl_champions_20260928/codes_pl_sol6.json"))}
    plc = {}
    for e in json.load(open("/u/zdliu/lp/oe_run/pl_champions_20260928/codes_deep_sol6.json")):
        c = int(str(e["cell"]).replace("cell", ""))
        plc[c] = (q_of[c], e["sha"])
    for name, arch, codes in (("dist", DIST_ARCHIVE, dist), ("plc", PL_ARCHIVE, plc)):
        for c in range(10):
            for p, shots in LEGS:
                tag = "%s_c%d_p%s" % (name, c, p.replace(".", ""))
                # this leg is measured separately
                if tag == "dist_c0_p0005":
                    continue
                print('echo -n "%s: "; ' % tag + SB % (arch, OUT, tag, p, shots, c, codes[c][0], codes[c][1]))
elif MODE == "p001":
    D = "/u/zdliu/lp/oe_run/pl_champions_20260928"
    tiers = {0: 5000000, 1: 5000000, 2: 5000000, 3: 5000000, 4: 500000, 5: 500000, 6: 500000, 7: 100000, 8: 100000, 9: 100000}
    skip = set(sys.argv[2:])   # tags of tiers already completed, e.g. lunaxhighpl_c7to9
    for tag, run in (("lunaxhigh", "/u/zdliu/lp/oe_run/pl10_gpt6lunaxhigh_out"), ("lunahigh", "/u/zdliu/lp/oe_run/pl10_gpt6lunahigh_out"),
                     ("lunalow", "/u/zdliu/lp/oe_run/pl10_gpt6lunalow_out")):
        pl = {e["cell"]: e for e in json.load(open("%s/codes_pl_%s.json" % (D, tag)))}
        deep = {int(str(e["cell"]).replace("cell", "")): e for e in json.load(open("%s/codes_deep_%s.json" % (D, tag)))}
        for c in range(10):
            tier = "c0to3" if c <= 3 else ("c4to6" if c <= 6 else "c7to9")
            if "%spl_%s" % (tag, tier) in skip or c not in pl or c not in deep:
                continue
            jt = "%spl_c%d_p001" % (tag, c)
            print('echo -n "%s: "; ' % jt + SB % (run + "/archive", D, jt, "0.001", tiers[c], c, pl[c]["q"], deep[c]["sha"]))

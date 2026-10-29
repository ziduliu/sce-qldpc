"""Tests of protocol_check_v5.py: a correct controller and worker pass; a wrong environment
value, a wrong api_base port, a wrong resolved config field, a repeated YAML key and an incomplete manifest
fail; credential variables are masked in the record and in mismatch reports."""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHK = os.path.join(HERE, "protocol_check_v5.py")
BASE_CFG = os.path.join(HERE, "..", "server_before", "circmean", "oe_circmean_cc_v7_rnd_w12_10cells.yaml")
T = tempfile.mkdtemp()


def cfg(path, port=23456, seed=42, timeout=400000, extra=""):
    txt = open(BASE_CFG).read().replace("__PORT__", str(port)).replace("\n  timeout: 40000\n", "\n  timeout: %d\n" % timeout)
    open(path, "w").write("random_seed: %d\n" % seed + txt + extra)
    return path


def run(role, manifest, env):
    m = os.path.join(T, "m.env")
    open(m, "w").write(manifest)
    e = dict(PATH=os.environ["PATH"], SLURM_JOB_ID=str(abs(hash(manifest + str(env))) % 10 ** 6), **env)
    r = subprocess.run([sys.executable, CHK, m, T, role], env=e, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


good_cfg = cfg(os.path.join(T, "good.yaml"))
COMMONE = dict(CM_NMIN="100", CM_NMAX="600", CM_WMAX="12", CM_RATE_LO="0.10", CM_RATE_STEP="0.05", CM_RATE_BINS="10", EM_SPOOL="sp")
CENV = dict(COMMONE, RND_V5="1", RND_LIBRARY="smallgrp", RND_ORDER_RULE="llm_n", RND_N_KEY="fom", RND_N_DIST="n.json",
            RND_SD_PATH="sd.tsv", RND_TMPL_PATH="t.tsv", RND_L3_SHAPE_P="0.63", RND_WCAPS="12", RND_LIB_PATH="lib.jsonl",
            PROTO_EVALUATOR="e", PROTO_SEEDS="s", PROTO_ITERS="520", PROTO_CONFIG=good_cfg, CM_LEDGER="l", EM_ARCH="r",
            OE_EVAL_TIMEOUT="400000", CM_ARCHIVE="a", CM_NSLOTS="60", CM_TABLE_TOP="0", CM_TABLE_CELLS="1", CM_INSP_BINS="4",
            CM_TOP_GLOBAL="1", CM_SHOW_FULL_TABLE="1", LAYOUT_BIN_ISLANDS="1", LAYOUT_RATE_BIN_N="10", LAYOUT_CELL_POOL="12",
            LAYOUT_SHAPE_PARENT_K="0", LAYOUT_RATE_EDGE_CELLS="1", LAYOUT_CELL_FOCUS_LEVELS="2", V2_CARD_SELF="0",
            V2_CARD_QUOTA="0", PROTO_RND_PORT="23456")
CMAN = "".join("%s=%s\n" % kv for kv in CENV.items() if kv[0] != "PROTO_RND_PORT") + \
    "CONFIGFIELD:random_seed=42\nCONFIGFIELD:evaluator.timeout=400000\nCONFIGFIELD:evaluator.parallel_evaluations=16\n"
WENV = dict(COMMONE, EM_SCRIPT="x.py", WK_EVAL_TIMEOUT="5400", WK_LEASE="7200", CM_DIST_MODE="cc", CM_D2S_DEEP="30",
            CM_D2S_BUDGET="2", CM_TOPK="5", CM_DCIRC_BUDGET="60")
PENV = dict(COMMONE, EM_SCRIPT="p.py", WK_EVAL_TIMEOUT="62000", WK_LEASE="70000", CM_DIST_MODE="pl", CM_D2S_BUDGET="2",
            CM_TOPK="5", PL_SCORE_P="0.002", PL_SCORE_ROUNDS="6", PL_SCORE_SHOTS="200000", PL_SCORE_MAX_ERRORS="100",
            PL_SCORE_SCHED_TIME="120", PL_SCORE_TIMEOUT="60000")
PMAN = "".join("%s=%s\n" % kv for kv in PENV.items())
WMAN = "".join("%s=%s\n" % kv for kv in WENV.items())
cases = [
    ("controller ok", "controller", CMAN, dict(CENV, OPENAI_API_KEY="sk-secret-1"), 0),
    ("worker ok", "worker", WMAN, WENV, 0),
    ("worker deep budget 1", "worker", WMAN, dict(WENV, CM_D2S_DEEP="1"), 1),
    ("wrong port", "controller", CMAN, dict(CENV, PROTO_RND_PORT="29999"), 1),
    ("resolved timeout 1", "controller", CMAN.replace(good_cfg, os.path.join(T, "t1.yaml")),
     dict(CENV, PROTO_CONFIG=cfg(os.path.join(T, "t1.yaml"), timeout=1)), 1),
    ("seed 43", "controller", CMAN.replace(good_cfg, os.path.join(T, "s43.yaml")),
     dict(CENV, PROTO_CONFIG=cfg(os.path.join(T, "s43.yaml"), seed=43)), 1),
    ("repeated key", "controller", CMAN.replace(good_cfg, os.path.join(T, "dup.yaml")),
     dict(CENV, PROTO_CONFIG=cfg(os.path.join(T, "dup.yaml"), extra="random_seed: 5\n")), 1),
    ("empty worker manifest", "worker", "", WENV, 1),
    ("controller without config fields", "controller", "".join(l + "\n" for l in CMAN.splitlines() if not l.startswith("CONFIGFIELD")), CENV, 1),
    ("secret in mismatch", "worker", WMAN + "MY_TOKEN=abc\n", dict(WENV, MY_TOKEN="zzz-secret"), 1),
    ("CLER worker ok", "worker", PMAN, PENV, 0),
    ("CLER worker rounds 1", "worker", PMAN, dict(PENV, PL_SCORE_ROUNDS="1"), 1),
    ("CLER worker manifest without rounds", "worker", "".join(l + "\n" for l in PMAN.splitlines() if not l.startswith("PL_SCORE_ROUNDS")), dict(PENV, PL_SCORE_ROUNDS="1"), 1),
    ("worker manifest without deep budget", "worker", "".join(l + "\n" for l in WMAN.splitlines() if not l.startswith("CM_D2S_DEEP")), dict(WENV, CM_D2S_DEEP="1"), 1),
    ("manifest without timeout field", "controller", CMAN.replace(good_cfg, os.path.join(T, "t2.yaml")).replace("CONFIGFIELD:evaluator.timeout=400000\n", ""),
     dict(CENV, PROTO_CONFIG=cfg(os.path.join(T, "t2.yaml"), timeout=1)), 1),
    ("repeated manifest entry", "worker", WMAN + "CM_D2S_DEEP=30\n", WENV, 1),
    ("controller manifest without EM_ARCH", "controller", "".join(l + "\n" for l in CMAN.splitlines() if not l.startswith("EM_ARCH")), CENV, 1),
    ("null api_base", "controller", CMAN.replace(good_cfg, os.path.join(T, "null.yaml")),
     dict(CENV, PROTO_CONFIG=os.path.join(T, "null.yaml")), 1),
    ("secret in config field", "controller", CMAN + "CONFIGFIELD:llm.api_key=sk-oldsecret\n",
     dict(CENV, PROTO_CONFIG=cfg(os.path.join(T, "key.yaml"), extra="")), 1),
]
# a configuration whose model has an explicit null endpoint
import re as _re
open(os.path.join(T, "null.yaml"), "w").write(_re.sub(r"api_base: \S+", "api_base: null", open(good_cfg).read()))
_txt = open(os.path.join(T, "key.yaml")).read()
open(os.path.join(T, "key.yaml"), "w").write(_txt.replace("llm:\n", "llm:\n  api_key: sk-newsecret\n", 1))
ok = True
for name, role, man, env, want in cases:
    rc, out = run(role, man, env)
    SECRETS = ("sk-secret-1", "zzz-secret", "sk-oldsecret", "sk-newsecret")
    leak = any(x in out for x in SECRETS)
    for f in os.listdir(T):
        if f.startswith("protocol_") and any(x in open(os.path.join(T, f)).read() for x in SECRETS):
            leak = True
    status = "PASS" if rc == want and not leak else "FAIL"
    ok &= status == "PASS"
    print("%-34s exit %d (want %d) %s%s | %s" % (name, rc, want, status, " LEAK" if leak else "", out.strip().splitlines()[-1][:150] if out.strip() else ""))
print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)

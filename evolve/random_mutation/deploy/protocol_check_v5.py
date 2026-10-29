#!/usr/bin/env python3
"""Protocol check of the v5 random-mutation runs. Called by the *_v5.sbatch scripts right
before Python starts the controller or the worker, i.e. after every startup file and module load.

  protocol_check_v5.py MANIFEST RECORD_DIR ROLE

MANIFEST lines:
  KEY=VALUE                  the environment variable KEY must equal VALUE
  CONFIGFIELD:path=VALUE     the engine's configuration, loaded from PROTO_CONFIG with the engine's own
                             loader (openevolve.config.load_config), must have str(field) == VALUE
                             (path: dotted attribute, e.g. evaluator.timeout)
For ROLE controller, every model of the loaded configuration must also point at
http://127.0.0.1:$PROTO_RND_PORT/v1, and the YAML file must not repeat a key in any mapping.
The manifest must hold the keys required for its role. Writes RECORD_DIR/protocol_<ROLE>_<jobid>.txt with
the effective values of the engine's, evaluators' and mutator's variables (credential variables are
masked) and the check result. Exit 1 on any mismatch.
"""
import os
import re
import sys
import time

COMMON = {"CM_NMIN", "CM_NMAX", "CM_WMAX", "CM_RATE_LO", "CM_RATE_STEP", "CM_RATE_BINS", "EM_SPOOL"}
REQUIRED = {"controller": COMMON | {"RND_V5", "RND_LIBRARY", "RND_ORDER_RULE", "RND_N_KEY", "RND_N_DIST", "RND_SD_PATH",
                                    "RND_TMPL_PATH", "RND_L3_SHAPE_P", "RND_WCAPS", "RND_LIB_PATH", "PROTO_EVALUATOR",
                                    "PROTO_SEEDS", "PROTO_ITERS", "PROTO_CONFIG", "CM_LEDGER", "EM_ARCH", "OE_EVAL_TIMEOUT",
                                    "CM_ARCHIVE", "CM_NSLOTS", "CM_TABLE_TOP", "CM_TABLE_CELLS", "CM_INSP_BINS",
                                    "CM_TOP_GLOBAL", "CM_SHOW_FULL_TABLE", "LAYOUT_BIN_ISLANDS", "LAYOUT_RATE_BIN_N",
                                    "LAYOUT_CELL_POOL", "LAYOUT_SHAPE_PARENT_K", "LAYOUT_RATE_EDGE_CELLS",
                                    "LAYOUT_CELL_FOCUS_LEVELS", "V2_CARD_SELF", "V2_CARD_QUOTA"},
            "worker": COMMON | {"EM_SCRIPT", "WK_EVAL_TIMEOUT", "WK_LEASE", "CM_DIST_MODE", "CM_D2S_BUDGET", "CM_TOPK"}}
REQUIRED_WORKER_MODE = {"cc": {"CM_D2S_DEEP", "CM_DCIRC_BUDGET"},
                        "pl": {"PL_SCORE_P", "PL_SCORE_ROUNDS", "PL_SCORE_SHOTS", "PL_SCORE_MAX_ERRORS",
                               "PL_SCORE_SCHED_TIME", "PL_SCORE_TIMEOUT"}}
REQUIRED_FIELDS = {"controller": {"random_seed", "evaluator.timeout", "evaluator.parallel_evaluations"}, "worker": set()}
SECRET = re.compile(r"(API_KEY|PASSWORD|SECRET|TOKEN)", re.I)     # env names and config paths (llm.api_key)


def shown(k, v):
    return "<masked>" if SECRET.search(k) else v


def clean(text):
    """Diagnostics may quote configuration values: mask anything that looks like a credential."""
    return re.sub(r"(sk-[A-Za-z0-9_\-]+|(?i:api_key|token|secret|password)\S*\s*[:=]\s*\S+)", "<masked>", str(text))


man, recdir, role = sys.argv[1], sys.argv[2], sys.argv[3]
expect, fields, bad = {}, {}, []
for raw in open(man):
    line = raw.rstrip("\n")
    if not line.strip() or line.startswith("#"):
        continue
    target = fields if line.startswith("CONFIGFIELD:") else expect
    k, v = (line[len("CONFIGFIELD:"):] if target is fields else line).split("=", 1)
    if k in target:
        bad.append(("manifest", "repeated entry", k))
    target[k] = v
need = set(REQUIRED[role]) | REQUIRED_WORKER_MODE.get(expect.get("CM_DIST_MODE"), set()) if role == "worker" else set(REQUIRED[role])
if role == "worker" and expect.get("CM_DIST_MODE") not in REQUIRED_WORKER_MODE:
    bad.append(("manifest", "worker CM_DIST_MODE must be one of %s" % sorted(REQUIRED_WORKER_MODE), expect.get("CM_DIST_MODE")))
missing = sorted(need - set(expect))
if missing:
    bad.append(("manifest", "required keys missing", missing))
missing_f = sorted(REQUIRED_FIELDS[role] - set(fields))
if missing_f:
    bad.append(("manifest", "required config fields missing", missing_f))
bad += [(k, shown(k, v), shown(k, os.environ.get(k) or "<unset>")) for k, v in sorted(expect.items()) if os.environ.get(k) != v]
if fields or role == "controller":
    cfg = os.environ.get("PROTO_CONFIG", "")
    try:
        import yaml

        class Strict(yaml.SafeLoader):
            pass

        def mapping(loader, node, deep=False):
            keys = [loader.construct_object(kn, deep=deep) for kn, _ in node.value]
            dup = sorted({k for k in keys if keys.count(k) > 1}, key=str)
            if dup:
                raise ValueError("repeated keys %s" % dup)
            return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)
        Strict.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        yaml.load(open(cfg), Loader=Strict)
        from openevolve.config import load_config
        c = load_config(cfg)
        for path, v in sorted(fields.items()):
            obj = c
            for part in path.split("."):
                obj = getattr(obj, part)
            if str(obj) != v:
                bad.append(("CONFIGFIELD " + path, clean(shown(path, v)), clean(shown(path, str(obj)))))
        if role == "controller":
            want = "http://127.0.0.1:%s/v1" % os.environ.get("PROTO_RND_PORT", "<unset>")
            models = list(getattr(c.llm, "models", []) or []) + list(getattr(c.llm, "evaluator_models", []) or [])
            # every model must resolve to this run's server; a missing (None) endpoint is a failure
            bases = [getattr(m, "api_base", None) for m in models]
            if not models or any(b != want for b in bases):
                bad.append(("CONFIG api_base", want, [clean(b) for b in bases]))
    except Exception as e:
        bad.append(("CONFIG", "loadable without repeated keys", clean("%s: %s" % (type(e).__name__, e))[:300]))
rec = {k: v for k, v in os.environ.items()
       if k.startswith(("CM_", "EM_", "OE_", "LAYOUT_", "V2_", "PL_", "RND_", "WK_", "PROTO_", "OPENAI_"))}
path = os.path.join(recdir, "protocol_%s_%s.txt" % (role, os.environ.get("SLURM_JOB_ID", "nojob")))
with open(path, "w") as f:
    f.write("# %s %s host=%s\n" % (role, time.strftime("%F %T"), os.uname().nodename))
    for k in sorted(rec):
        f.write("%s=%s\n" % (k, shown(k, rec[k])))
    f.write("CHECK: %s (%d values, %d config fields)\n" % ("OK" if not bad else "MISMATCH %r" % bad, len(expect), len(fields)))
if bad:
    sys.stderr.write("PROTOCOL MISMATCH (%s): %r\n" % (role, bad))
    sys.exit(1)
print("protocol check ok (%s): %d values, %d config fields -> %s" % (role, len(expect), len(fields), path), flush=True)

"""Persistent GPU worker: claims evaluation requests from a spool directory,
runs the evaluator script (EM_SCRIPT) on each with a WK_EVAL_TIMEOUT limit, and
writes the result plus a per-request log. Claims older than WK_LEASE seconds
are returned to the queue (recovers from worker death).
"""
import json
import os
import subprocess
import time

SPOOL = os.environ.get("EM_SPOOL", "/u/zdliu/lp/jobs/circmean_spool")
IDLE_EXIT = int(os.environ.get("WK_IDLE_EXIT", "7200"))
LEASE = int(os.environ.get("WK_LEASE", "9000"))
EVAL_TIMEOUT = int(os.environ.get("WK_EVAL_TIMEOUT", "7200"))
EVOMAP = os.environ.get("EM_SCRIPT",
                        "/u/zdliu/lp/circmean/eval_circmean.py")
PY = os.environ.get("EM_PY", "/u/zdliu/lpenv/bin/python")

os.makedirs(SPOOL, exist_ok=True)
wid = os.environ.get("SLURM_JOB_ID", str(os.getpid()))


def recover_stale():
    now = time.time()
    for f in os.listdir(SPOOL):
        path = os.path.join(SPOOL, f)
        try:
            if f.startswith("work_") and now - os.path.getmtime(path) > LEASE:
                rid = f.split("_", 2)[2]
                os.rename(path, os.path.join(SPOOL, "req_" + rid))
                os.utime(os.path.join(SPOOL, "req_" + rid), None)
                print("recovered stale %s" % f, flush=True)
            elif (f.startswith(("res_", "log_", "tmp_"))
                  and now - os.path.getmtime(path) > 86400):
                os.remove(path)            # orphaned results/logs
        except OSError:
            continue


def purge_node_tmp():
    """Evaluations killed by WK_EVAL_TIMEOUT never reach their own npz
    cleanup; sweep day-old leftovers at worker start (node-local /tmp)."""
    import glob
    now = time.time()
    for f in glob.glob("/tmp/em_*.npz"):
        try:
            if now - os.path.getmtime(f) > max(86400, 2 * EVAL_TIMEOUT):
                os.remove(f)
        except OSError:
            continue


last_work = time.time()
recover_stale()
purge_node_tmp()
print("worker %s watching %s" % (wid, SPOOL), flush=True)
while time.time() - last_work < IDLE_EXIT:
    reqs = sorted(f for f in os.listdir(SPOOL) if f.startswith("req_"))
    claimed = None
    for r in reqs:
        src = os.path.join(SPOOL, r)
        dst = os.path.join(SPOOL, "work_%s_%s" % (wid, r[4:]))
        try:
            os.rename(src, dst)          # atomic claim
            claimed = (r[4:], dst)
            break
        except OSError:
            continue
    if claimed is None:
        recover_stale()
        time.sleep(3)
        continue
    rid, path = claimed
    out = os.path.join(SPOOL, "tmp_%s" % rid)
    logp = os.path.join(SPOOL, "log_%s.txt" % rid.replace(".json", ""))
    env = dict(os.environ, EM_IN=path, EM_OUT=out)
    t0 = time.time()
    rc = None
    with open(logp, "w") as lg:
        # own process group so a timeout kill also reaps the LSD/d2s
        # children (they hold CUDA contexts)
        proc = subprocess.Popen([PY, "-u", EVOMAP], env=env,
                                stdout=lg, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=EVAL_TIMEOUT)
        except subprocess.TimeoutExpired:
            rc = "timeout"
            try:
                os.killpg(proc.pid, 9)
            except OSError:
                pass
            try:
                proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                print("WARN: %s unreapable (D-state?)" % rid, flush=True)
    if not os.path.exists(out):
        json.dump({"result": "error:no-output rc=%s" % rc,
                   "fitness": 0.0, "log": logp}, open(out, "w"))
    else:
        try:
            r = json.load(open(out))
            r["rc"] = rc
            r["log"] = logp
            json.dump(r, open(out, "w"))
        except Exception:
            pass
    os.replace(out, os.path.join(SPOOL, "res_%s" % rid))
    try:
        os.remove(path)
    except OSError:
        pass          # claim may have been recovered externally
    last_work = time.time()
    print("done %s rc=%s in %.0fs" % (rid, rc, time.time() - t0),
          flush=True)
print("worker %s idle-exit" % wid, flush=True)

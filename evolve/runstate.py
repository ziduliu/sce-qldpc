"""Sidecar: render the live RUN STATE text that the prompt sampler injects
into every prompt (EM_ARCH=<out>/runstate.txt). Parts, all facts of this run:

  BEDS     every (group, shapes) bed evaluated so far: programs, best
           fitness, iteration of the best, iterations since the bed last
           improved
  MOVES    what the previous generations' moves did, by ACTUAL level
           (1 = same group and shapes as the parent, 2 = same group /
           different shapes, 3 = different group), with the bed jumps
           listed (from -> to, child score vs the destination bed's best
           at that time)
  FRONTIER the best program of every rate band with its measured codes
           and the cells of its best slot
  SHAPES   admissible shape pairs per rate cell under the size limit

Sources: all numbered checkpoints (complete lineage: parent ids, iteration,
metrics incl. the stored slot table) plus <out>/eval_ledger.jsonl written
by the evaluator for the programs evaluated after the newest checkpoint
(fresh, no lineage). Written atomically (tmp + rename) every --interval
seconds. Usage:
  python runstate.py --out <run dir> [--interval 300] [--once]
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_circmean import RATE_BINS, RATE_LO, RATE_STEP, rate_bin_of  # noqa: E402
_FP_CACHE = {}
_FP_CACHE_PATH = None


def group_fp_of_code(code, sha):
    """(key, label) of the program's group from its multiplication table,
    by running get_candidates() in a subprocess (cached by code sha1)."""
    global _FP_CACHE_PATH
    if sha in _FP_CACHE:
        return tuple(_FP_CACHE[sha])
    script = (
        "import importlib.util, json, sys\n"
        "sys.path.insert(0, %r)\n"
        "from group_fp import fingerprint\n"
        "spec = importlib.util.spec_from_file_location('prog', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "r = m.get_candidates(); print(json.dumps(fingerprint(r['mul'], int(r['q']))))\n"
    ) % HERE
    import tempfile
    tf = tempfile.NamedTemporaryFile("w", prefix="rs_prog_", suffix=".py", delete=False)
    tf.write(code)
    tf.close()
    tmp = tf.name
    try:
        out = subprocess.run([sys.executable, "-c", script, tmp], capture_output=True,
                             text=True, timeout=600)
        res = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        # transient (timeout under load, disk full) or a broken program:
        # not cached, so a later cycle retries
        return ("unknown:" + sha, "unknown group")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    _FP_CACHE[sha] = res
    if _FP_CACHE_PATH:
        try:
            tmpc = _FP_CACHE_PATH + ".tmp"
            json.dump(_FP_CACHE, open(tmpc, "w"))
            os.replace(tmpc, _FP_CACHE_PATH)
        except Exception:
            pass
    return tuple(res)

_SHAPE = r"^\s*%s\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"
_MRS = re.compile(r"m\s*,\s*r\s*,\s*s\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")
_VALIDQ = re.compile(r"return\s+q\s*==\s*(\d+)")
_FAM = re.compile(r"^\s*FAMILY_NAME\s*=\s*[\"']([^\"']+)", re.M)
_CG = re.compile(r'^CUSTOM_GROUP\s*=\s*"""(.*?)"""', re.S | re.M)
_GTYPE = re.compile(r"^\s*GTYPE\s*=\s*[\"'](\w+)", re.M)
_A0 = re.compile(r"^\s*A0\s*=\s*(.+)$", re.M)


def parse_program(code):
    """(group_label, group_key, shapes, q) from a program text."""
    ma = re.search(_SHAPE % "A_SHAPE", code, re.M)
    mb = re.search(_SHAPE % "B_SHAPE", code, re.M)
    shapes = ((int(ma.group(1)), int(ma.group(2))),
              (int(mb.group(1)), int(mb.group(2)))) if ma and mb else None
    gtype = (_GTYPE.search(code) or [None, "?"])[1] if _GTYPE.search(code) else "?"
    cg = _CG.search(code)
    cgt = cg.group(1) if cg else ""
    q = None
    mq = _VALIDQ.search(cgt)
    if mq:
        q = int(mq.group(1))
    if gtype == "custom":
        mm = _MRS.search(cgt)
        if mm:
            m_, r_, s_ = (int(x) for x in mm.groups())
            label = ("Z%d" % m_) if r_ == 1 else ("Z%d:Z%d(s=%d)" % (m_, r_, s_) if s_ % m_ != 1 else "Z%dxZ%d" % (m_, r_))
            key = "mrs:%d:%d:%d" % (m_, r_, s_)
        else:
            norm = re.sub(r"\s+", "", cgt)
            h = hashlib.sha1(norm.encode()).hexdigest()[:6]
            cm = re.search(r"#\s*([^\n]{3,40})", cgt)
            label = "custom[%s]%s" % (h, (" " + cm.group(1).strip()) if cm else "")
            key = "custom:" + h
    else:
        a0 = _A0.search(code)
        label = "%s%s" % (gtype, (" A0=" + a0.group(1).strip()[:20]) if a0 else "")
        key = "%s:%s" % (gtype, (a0.group(1).strip() if a0 else ""))
    return label, key, shapes, q


def bed_of(code, q_hint=None, sha=None):
    label, key, shapes, q = parse_program(code)
    q = q or q_hint
    if shapes is None:
        return None
    if sha is not None:
        fkey, flabel = group_fp_of_code(code, sha)
        key = fkey
        if not _MRS.search(_CG.search(code).group(1) if _CG.search(code) else ""):
            label = flabel
        if q is None and fkey.startswith("q"):
            try:
                q = int(fkey[1:fkey.index("|")])
            except ValueError:
                pass
    (ra, ca), (rb, cb) = shapes
    n = (ca * cb + ra * rb) * q if q else None
    name = "%s q=%s (%d,%d)x(%d,%d) n=%s" % (label, q, ra, ca, rb, cb, n)
    return dict(name=name, key=(key, q, shapes), gkey=key, shapes=shapes, q=q, n=n)


def parse_table(txt):
    """Measured rows (idx, n, k, w, d2s, d_circ, cells) from a slot table.

    Columns are located by name from the header line, so every table layout
    parses; the cells line is searched for within the row's own block.
    """
    rows = []
    lines = (txt or "").splitlines()
    cols = None
    for i, ln in enumerate(lines):
        st = ln.strip()
        if st.startswith("idx ") and " n " in ln and " k " in ln:
            cols = st.split()
            continue
        if not st.endswith("measured") or cols is None:
            continue
        parts = st.split()
        if len(parts) != len(cols):
            continue
        rec = dict(zip(cols, parts))
        try:
            dread = rec.get("d30s", rec.get("d_circ"))
            vread = rec.get("kd30s^2/n", rec.get("kd_circ2/n"))
            row = dict(idx=int(rec["idx"]), n=int(rec["n"]), k=int(rec["k"]), w=int(rec["w"]),
                       d2s=int(rec["d2s"]), d_circ=int(dread), v=float(vread), cells=None)
        except (KeyError, ValueError, TypeError):
            continue
        # the cells line lives somewhere in this row's block (after the operator lines)
        for j in range(i + 1, min(i + 12, len(lines))):
            nxt = lines[j].strip()
            if nxt.startswith("cells "):
                row["cells"] = nxt[6:]
                break
            if re.match(r"\d+\s+\d+\s+\d+", nxt):   # next slot row
                break
        rows.append(row)
    rows.sort(key=lambda r: -r["v"])
    return rows


def load_checkpoint(out):
    """Programs from ALL numbered checkpoints, newest copy winning: the
    population is capped (60), so programs pruned from the newest
    checkpoint - and the parents of surviving children - still exist in
    older ones."""
    cps = [p for p in glob.glob(os.path.join(out, "checkpoints", "checkpoint_*"))
           if re.fullmatch(r"checkpoint_\d+", os.path.basename(p))]
    if not cps:
        return None, {}
    cps.sort(key=lambda p: int(p.rsplit("_", 1)[1]))
    P = {}
    for cp in cps:
        for f in glob.glob(os.path.join(cp, "programs", "*.json")):
            try:
                p = json.load(open(f))
            except Exception:
                continue
            P[p["id"]] = p
    return cps[-1], P


def load_ledger(out):
    path = os.path.join(out, "eval_ledger.jsonl")
    rows = []
    if os.path.exists(path):
        for ln in open(path):
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows


def bed_from_ledger_row(r):
    """Bed of an evaluated program from its ledger row: the evaluator's group
    fingerprint (gkey/glabel), order and shapes. None if the row lacks shapes."""
    a, b_ = r.get("a_shape"), r.get("b_shape")
    if not (a and b_):
        return None
    shapes = ((a[0], a[1]), (b_[0], b_[1]))
    fam = r.get("family") or "?"
    gkey = r.get("gkey") or ("led:%s:%s" % (fam.split("_")[0], r.get("q")))
    label = r.get("glabel") or ("%s(q=%s)" % (r.get("gtype", "?"), r.get("q")))
    mm = re.search(r"m(\d+)r(\d+)s(\d+)", fam)
    if mm and not (r.get("glabel") or "").startswith("Z"):
        m_, r_, s_ = (int(x) for x in mm.groups())
        label = ("Z%d" % m_) if r_ == 1 else ("Z%d:Z%d(s=%d)" % (m_, r_, s_) if s_ % m_ != 1 else "Z%dxZ%d" % (m_, r_))
    (ra, ca), (rb, cb) = shapes
    return dict(name="%s q=%s (%d,%d)x(%d,%d) n=%s" % (label, r.get("q"), ra, ca, rb, cb, r.get("n")),
                key=(gkey, r.get("q"), shapes), gkey=gkey, shapes=shapes, q=r.get("q"), n=r.get("n"))


def render(out, max_iter_hint=None):
    global _FP_CACHE_PATH, _FP_CACHE
    _FP_CACHE_PATH = os.path.join(out, "group_fp_cache.json")
    if not _FP_CACHE and os.path.exists(_FP_CACHE_PATH):
        try:
            _FP_CACHE = json.load(open(_FP_CACHE_PATH))
        except Exception:
            _FP_CACHE = {}
    cp, P = load_checkpoint(out)
    ledger = load_ledger(out)
    now = time.time()
    ledger_by_sha = {r.get("code_sha1"): r for r in ledger if r.get("code_sha1")}

    # ---- programs from the checkpoint (complete) ----
    progs = []
    migrant_of = {}
    for p in P.values():
        if (p.get("metadata") or {}).get("migrant"):
            # island-migration copy: same program, iteration_found reset
            # to 0 - it would front-run the timeline and steal records
            migrant_of[p["id"]] = p.get("parent_id")
            continue
        _sha = hashlib.sha1(p.get("code", "").encode()).hexdigest()[:16]
        # reuse the evaluator's group fingerprint from the ledger row with the
        # same code sha; otherwise run get_candidates() in a subprocess
        _lr = ledger_by_sha.get(_sha)
        b = bed_from_ledger_row(_lr) if (_lr and _lr.get("gkey")) else None
        if b is None:
            b = bed_of(p.get("code", ""), sha=_sha)
        if b is None:
            continue
        mt = p.get("metrics", {}) or {}
        art = mt.get("artifacts") or {}
        tbl = art.get("slot_table") if isinstance(art, dict) else None
        progs.append(dict(id=p["id"], sha=hashlib.sha1(p.get("code", "").encode()).hexdigest()[:16],
                          bed=b, score=float(mt.get("combined_score") or 0.0),
                          rate=mt.get("rate"), it=int(p.get("iteration_found") or 0),
                          parent=p.get("parent_id"), family=(_FAM.search(p.get("code", "")) or [None, "?"])[1]
                          if _FAM.search(p.get("code", "")) else "?",
                          measured=parse_table(tbl), n_dist=mt.get("n_distinct"),
                          n_dup=mt.get("n_archive_dup"), ts=float(p.get("timestamp") or 0)))
    by_id = {x["id"]: x for x in progs}
    for mid, orig in migrant_of.items():
        if orig in by_id:
            by_id[mid] = by_id[orig]
    known_sha = {x["sha"] for x in progs}
    last_it = max([x["it"] for x in progs] + [0])

    # ---- fresh programs from the ledger (after the checkpoint) ----
    fresh = []
    for r in ledger:
        if r.get("code_sha1") in known_sha:
            continue
        bed = bed_from_ledger_row(r)
        if bed is None:
            continue
        fam = r.get("family") or "?"
        meas = [dict(idx=t[0], n=t[1], k=t[2], w=t[3], d2s=t[4], d_circ=t[5],
                     v=round((t[2] or 0) * (t[5] or 0) ** 2 / (t[1] or 1), 2),
                     cells=("A=%s B=%s" % (json.dumps((r.get("best_cells") or {}).get("A"), separators=(",", ":")),
                                            json.dumps((r.get("best_cells") or {}).get("B"), separators=(",", ":"))))
                     if (i == 0 and r.get("best_cells")) else None)
                for i, t in enumerate(r.get("measured") or [])]
        fresh.append(dict(id="fresh", sha=r.get("code_sha1"), bed=bed, score=float(r.get("fitness") or 0),
                          rate=r.get("rate"), it=None, parent=None, family=fam,
                          measured=meas, n_dist=r.get("n_distinct"), n_dup=r.get("n_archive_dup"),
                          ts=float(r.get("ts") or 0)))
    allp = progs + fresh
    # one readable name per group: a program written with the metacyclic
    # template names its group (Z7:Z4(s=6)); the same group rewritten by
    # the model in another presentation only has the fingerprint label
    nice = {}
    for x in allp:
        lab = x["bed"]["name"].split(" q=")[0]
        if not lab.startswith(("abelian#", "nonabelian#", "unknown", "custom[")):
            nice.setdefault(x["bed"]["gkey"], lab)
    for x in allp:
        lab = x["bed"]["name"].split(" q=")[0]
        if x["bed"]["gkey"] in nice and lab != nice[x["bed"]["gkey"]]:
            x["bed"]["name"] = nice[x["bed"]["gkey"]] + " q=" + x["bed"]["name"].split(" q=", 1)[1]

    # ---- BEDS ----
    beds = {}
    order_key = lambda x: (x["it"] if x["it"] is not None else 10 ** 9, x["ts"])
    for x in sorted(allp, key=order_key):
        k = (x["bed"]["gkey"], x["bed"]["q"], x["bed"]["shapes"])
        d = beds.setdefault(k, dict(name=x["bed"]["name"], n=0, best=0.0, best_it=None,
                                    best_fam="", best_ts=None))
        d["n"] += 1
        if x["score"] > d["best"]:        # strict: the FIRST program at the current best
            d["best"], d["best_it"], d["best_fam"] = x["score"], x["it"], x["family"]
            d["best_ts"] = x["ts"]
    lines = []
    lines.append("RUN STATE rendered %s (live at prompt time; iteration %d in the newest checkpoint, %d programs in checkpoints, %d evaluated since)"
                 % (time.strftime("%Y-%m-%d %H:%M"), last_it, len(progs), len(fresh)))
    lines.append("")
    lines.append("BEDS - every (group, shapes) bed with a recorded evaluation in this run. "
                 "best-since = how long the bed's best has stood at its current value.")
    lines.append("  best  programs  best@iter  best-since  bed")
    for k, d in sorted(beds.items(), key=lambda kv: -kv[1]["best"]):
        bi = d["best_it"] if d["best_it"] is not None else "new"
        since = ("%.0f min" % ((now - d["best_ts"]) / 60)) if d.get("best_ts") else "-"
        lines.append("  %5.1f  %8d  %9s  %10s  %s" % (d["best"], d["n"], bi, since, d["name"]))

    # ---- MOVES (checkpoint lineage only) ----
    lines.append("")
    lines.append("MOVES - what earlier generations' moves did, by ACTUAL level "
                 "(1 same group+shapes as parent, 2 same group/new shapes, 3 new group), "
                 "over the children whose parent is recorded. "
                 "'improved' = child beat its parent; 'bed record' = child beat the best of its destination bed at that time.")
    stats = {1: [0, 0, 0, []], 2: [0, 0, 0, []], 3: [0, 0, 0, []]}
    jumps = []
    # bed best as of each iteration, to judge 'bed record'
    bybed_hist = {}
    for x in sorted(progs, key=lambda x: (x["it"], x["ts"])):
        k = (x["bed"]["gkey"], x["bed"]["q"], x["bed"]["shapes"])
        par = by_id.get(x["parent"]) if x["parent"] else None
        prev_best = bybed_hist.get(k, 0.0)
        if par is not None:
            pk = (par["bed"]["gkey"], par["bed"]["q"], par["bed"]["shapes"])
            if pk == k:
                lvl = 1
            elif par["bed"]["gkey"] == x["bed"]["gkey"]:
                lvl = 2
            else:
                lvl = 3
            st = stats[lvl]
            st[0] += 1
            if x["score"] > par["score"]:
                st[1] += 1
            if x["score"] > prev_best:
                st[2] += 1
            st[3].append(x["score"] - par["score"])
            if lvl in (2, 3):
                jumps.append((x["it"], lvl, par["bed"]["name"], x["bed"]["name"], x["score"],
                              prev_best if k in bybed_hist else None))
        bybed_hist[k] = max(prev_best, x["score"])
    for lvl in (1, 2, 3):
        n, imp, rec, gains = stats[lvl]
        if n == 0:
            lines.append("  level %d: 0 children" % lvl)
            continue
        gains.sort()
        lines.append("  level %d: %d children, %d improved on parent, %d set a bed record; gain median %+.1f, best %+.1f"
                     % (lvl, n, imp, rec, gains[len(gains) // 2], gains[-1]))
    if jumps:
        lines.append("  bed jumps (iteration, level, from -> to, child score, destination bed best before it):")
        for it, lvl, a, b, s, pb in jumps[-12:]:
            lines.append("    it %3d L%d  %s  ->  %s : %.1f  (dest best before: %s)"
                         % (it, lvl, a, b, s, ("%.1f" % pb) if pb is not None else "untried bed"))
        if len(jumps) > 12:
            lines.append("    (%d earlier jumps not shown)" % (len(jumps) - 12))

    # ---- FRONTIER ----
    lines.append("")
    lines.append("FRONTIER - best program of each rate band (the MAP-Elites cells: k/n from %.2f in steps of %.3f), its measured codes [[n,k,d]] w (d = the run's distance reading: the 30 s code-capacity search in cc mode, else d_circ), "
                 "and the cells of its best slot (in THAT program's CUSTOM_GROUP element encoding; sides A then B, protograph row-major)."
                 % (RATE_LO, RATE_STEP))
    bands = {}
    for x in allp:
        # band from the measured rate under the current grid (stored
        # rate_bin values may come from a different grid)
        rate = x.get("rate")
        if rate is None:
            (ra, ca), (rb_, cb) = x["bed"]["shapes"]
            kq, npq = (ca - ra) * (cb - rb_), ca * cb + ra * rb_
            if kq <= 0:
                continue
            rate = kq / npq
        rb = rate_bin_of(float(rate))
        if rb not in bands or x["score"] > bands[rb]["score"]:
            bands[rb] = x
    for rb, x in sorted(bands.items(), key=lambda kv: int(kv[0])):
        lo = RATE_LO + RATE_STEP * int(rb)
        codes = ", ".join("[[%d,%d,%d]] w%d" % (t["n"], t["k"], t["d_circ"], t["w"]) for t in x["measured"][:5]) or "(none measured)"
        lines.append("  band %2d (k/n~%.3f): %.1f  %s | %s" % (int(rb), lo, x["score"], x["bed"]["name"], codes))
        if x["measured"] and x["measured"][0].get("cells"):
            lines.append("      best slot cells: %s" % x["measured"][0]["cells"][:700])
    # ---- SHAPES: admissible shape pairs under the size law ----
    n_max = int(os.environ.get("CM_NMAX", "600"))
    side_max = int(os.environ.get("CM_SHAPE_SIDE_MAX", "8"))
    per_band = int(os.environ.get("CM_SHAPES_PER_BAND", "6"))

    def canon(ra, ca, rb, cb):
        # A<->B swap and transposing both sides give the same parameter family
        a = tuple(sorted([(ra, ca), (rb, cb)]))
        b = tuple(sorted([(ca, ra), (cb, rb)]))
        return min(a, b)

    by_shape = {}
    for x in allp:
        (ra, ca), (rb, cb) = x["bed"]["shapes"]
        key = canon(ra, ca, rb, cb)
        by_shape[key] = max(by_shape.get(key, 0.0), x["score"])
    fams = {}
    for ra in range(1, side_max + 1):
        for ca in range(1, side_max + 1):
            for rb in range(1, side_max + 1):
                for cb in range(1, side_max + 1):
                    kq = (ca - ra) * (cb - rb)
                    if kq <= 0:
                        continue
                    npq = ca * cb + ra * rb
                    qmax = (n_max - 1) // npq
                    if qmax < 8:
                        continue
                    key = canon(ra, ca, rb, cb)
                    if key not in fams:
                        fams[key] = (kq / npq, npq, qmax)
    bands = {}
    for key, (rate, npq, qmax) in fams.items():
        # group by MAP-Elites cell (same cells as FRONTIER)
        bands.setdefault(rate_bin_of(rate), []).append((qmax, rate, npq, key))
    lines.append("")
    lines.append("SHAPES - the size law admits %d distinct (A,B) shape families with sides up to %d and n < %d "
                 "(k-bound (ca-ra)(cb-rb) > 0; A/B swap and transposing both sides counted once). "
                 "Per 0.05 band of the guaranteed k/n = (ca-ra)(cb-rb)/(ca*cb+ra*rb): the %d shapes with the "
                 "largest lift order q_max = floor(%d/(ca*cb+ra*rb)), plus every shape this run has used "
                 "('best' = this run's best fitness on it, '-' = never used in this run)."
                 % (len(fams), side_max, n_max, per_band, n_max - 1))
    lines.append("  k/n>=  n/q  q_max   best  shape")
    for b in sorted(bands):
        rows_b = sorted(bands[b], key=lambda t: (-t[0], -t[1]))
        shown = rows_b[:per_band] + [t for t in rows_b[per_band:] if t[3] in by_shape]
        _c = RATE_LO + RATE_STEP * int(b)
        # open-ended end cells: rate_bin_of clamps, so the first cell holds
        # every rate below its upper edge and the last every rate above its
        # lower edge (shown as such under LAYOUT_RATE_EDGE_CELLS=1)
        _elo, _ehi = _c - RATE_STEP / 2, _c + RATE_STEP / 2
        _edge = os.environ.get("LAYOUT_RATE_EDGE_CELLS", "0") == "1"
        if _edge and int(b) == 0:
            _span = "k/n below %.4f (open-ended)" % _ehi
        elif _edge and int(b) == RATE_BINS - 1:
            _span = "k/n above %.4f (open-ended)" % _elo
        else:
            _span = "k/n %.4f-%.4f" % (_elo, _ehi)
        lines.append("  -- cell %d, %s, centre %.3f (%d shapes) --"
                     % (int(b), _span, _c, len(rows_b)))
        for qmax, rate, npq, key in shown:
            (ra, ca), (rb, cb) = key
            bst = by_shape.get(key)
            lines.append("  %.3f  %3d  %5d  %5s  (%d,%d)x(%d,%d)" % (rate, npq, qmax, ("%.1f" % bst) if bst else "-", ra, ca, rb, cb))
    txt = "\n".join(lines)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    target = os.path.join(a.out, "runstate.txt")
    src_mtime = os.path.getmtime(os.path.abspath(__file__))
    while True:
        if os.path.getmtime(os.path.abspath(__file__)) != src_mtime and not a.once:
            os.execv(sys.executable, [sys.executable] + sys.argv)   # re-exec on source change
        try:
            txt = render(a.out)
            tmp = target + ".tmp"
            open(tmp, "w").write(txt + "\n")
            os.replace(tmp, target)
            if a.once:
                print(txt)
                return
        except Exception as e:
            print("runstate error:", type(e).__name__, str(e)[:200], flush=True)
            if a.once:
                raise
        time.sleep(a.interval)


if __name__ == "__main__":
    main()

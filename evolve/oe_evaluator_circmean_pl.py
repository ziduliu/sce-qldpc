"""OpenEvolve evaluator (controller side) for the circuit-level pL objective.

Same spool protocol as oe_evaluator_circmean_v3.py; the slot table shown in
the next prompt adds, per measured code and basis, the shot and failure
counts, decoder-stage counts, ASC circuit depth, CNOT count and failure map.
"""
import importlib.util
import json
import os
import re
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_circmean_pl import rate_bin_of  # noqa: E402

SPOOL = os.environ.get("EM_SPOOL", "/u/zdliu/lp/jobs/circmean_spool")
TIMEOUT = int(os.environ.get("OE_EVAL_TIMEOUT", "7200"))
# per-run code archive (see eval_circmean.CodeArchive); empty = off
ARCHIVE = os.environ.get("CM_ARCHIVE", "")

_SHAPE_RE = r"^\s*%s\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"
# per-evaluation ledger read by runstate.py; empty = off
LEDGER = os.environ.get("CM_LEDGER", "")


def _ledger_write(program_path, req, rec, m):
    """One JSON line per evaluation: group, shapes, scores, measured codes,
    cells of the best measured slot, sha1 of the program text (joins to the
    checkpoint programs for lineage)."""
    import hashlib
    try:
        code = open(program_path, "rb").read()
        meas = [t for t in rec.get("slots", []) if t.get("status") == "measured"]
        meas.sort(key=lambda t: -(t.get("kd_circ2_n") or t.get("pl_score") or 0))
        try:
            from group_fp import fingerprint
            gkey, glabel = fingerprint(req["mul"], int(req["q"]))
        except Exception:
            gkey, glabel = None, None
        row = {
            "ts": round(time.time(), 1), "gkey": gkey, "glabel": glabel,
            "code_sha1": hashlib.sha1(code).hexdigest()[:16],
            "family": req.get("family"), "gtype": req.get("gtype"),
            "q": req.get("q"), "a_shape": req.get("a_shape"),
            "b_shape": req.get("b_shape"), "n": req.get("n"),
            "fitness": m.get("combined_score"), "rate": m.get("rate"),
            "rate_bin": m.get("rate_bin"),
            "n_distinct": m.get("n_distinct"), "n_measured": m.get("n_measured"),
            "n_archive_dup": m.get("n_archive_dup"),
            "measured": [[t.get("idx"), t.get("n"), t.get("k"), t.get("w"),
                          t.get("d2s"), t.get("d_circ")] for t in meas],
            # pL extras; the six "measured" columns above keep the layout
            # that runstate.py parses
            "pl": [[t.get("idx"), t.get("pl_score"), t.get("pl_basis"),
                    t.get("pl_ceiling"), t.get("depth"), t.get("cnots"),
                    {b: [d.get("shots"), d.get("logical_failures"),
                         d.get("stopped_on"), d.get("resolved_by_stage"),
                         # 5th: failed shots, the numerator of the score
                         d.get("shot_failures")]
                     for b, d in sorted((t.get("pl") or {}).items())}]
                   for t in meas],
            "pl_error": rec.get("pl_error"),
            "n_pl_failed": rec.get("n_pl_failed"),
            "best_cells": (meas[0].get("cells") if meas else None)
            if os.environ.get("CM_TABLE_CELLS", "0") == "1" else None,
        }
        line = (json.dumps(row, separators=(",", ":")) + "\n").encode()
        fd = os.open(LEDGER, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line)      # one write, O_APPEND: no interleaving
        finally:
            os.close(fd)
    except Exception as e:
        print("ledger write failed: %s: %s" % (type(e).__name__, str(e)[:100]))


def _bed_rate_bin(shapes=None, code=""):
    """MAP-Elites cell of a program that produced no measurable code,
    from the rank-count bound k/n = (ca-ra)(cb-rb)/(ca*cb+ra*rb) of its
    shapes (request a_shape/b_shape, else A_SHAPE/B_SHAPE in the text).
    Without a rate_bin metric the engine would park the program in cell 0,
    where it would be drawn as a parent as often as any elite."""
    try:
        if shapes:
            (ra, ca), (rb, cb) = shapes
        else:
            ma = re.search(_SHAPE_RE % "A_SHAPE", code, re.M)
            mb = re.search(_SHAPE_RE % "B_SHAPE", code, re.M)
            if not (ma and mb):
                return None
            ra, ca = int(ma.group(1)), int(ma.group(2))
            rb, cb = int(mb.group(1)), int(mb.group(2))
        kq = (ca - ra) * (cb - rb)
        nq = ca * cb + ra * rb
        if kq <= 0 or nq <= 0:
            return None
        return rate_bin_of(kq / nq)
    except Exception:
        return None


def _render_attrib(a):
    """One feedback line from a pl_attrib.FrameAttrib summary (same wording as pl_attrib.render)."""
    if not a or not a.get("failed"):
        return "failure map: no failed shot to attribute"

    def top(rows, k=4):
        rows = sorted([r for r in rows if r[1] is not None], key=lambda r: -r[1])
        return ", ".join("%s %.1fx" % (r[0], r[1]) for r in rows[:k])
    return ("failure map (from the %d failed of %d scored shots; enrichment = a block's share of the errors in "
            "failed shots / its share in all shots, 1.0 = no concentration): data blocks %s; check blocks %s; "
            "mean data-error weight %s in failed shots vs %s in all"
            % (a["failed"], a["shots"], top(a["data_blocks"]), top(a["check_blocks"]),
               a.get("mean_error_weight_failed"), a.get("mean_error_weight_all")))


def _witness_lines(r):
    """Witness lines for one measured slot (same wording as the distance evaluator)."""
    out = []
    if not r.get("wit_pool"):
        return out
    parts = []
    for side in ("x", "z"):
        p = r["wit_pool"].get(side)
        if not p:
            continue
        sec = p["sectors"]
        parts.append("%s-type: %s%d operators at weight %d (%d only in sector 1, %d only in sector 2, %d in both)" % (
            side.upper(), "at least " if p.get("capped") else "", p["n_ops"], p["weight"],
            sec["sector1_only"], sec["sector2_only"], sec["both"]))
    if parts:
        out.append("     minimum-weight logicals found by the 30 s search: " + "; ".join(parts))
    dmin = min(v["weight"] for v in r["wit_pool"].values())
    for side in ("x", "z"):
        p = r["wit_pool"].get(side)
        if p and p["weight"] == dmin and p.get("blocks_ops"):
            bl = ", ".join("%s:%d" % kv for kv in sorted(p["blocks_ops"].items(), key=lambda kv: (-kv[1], kv[0])))
            out.append("     blocks touched by the %s-type operators (block: how many of the %d%s operators use it): %s" % (
                side.upper(), p["n_ops"], " listed" if p.get("capped") else "", bl))
    if r.get("wit_spectrum"):
        out.append("     weight spectrum found (weight:verified operators, both types, uncapped): %s" % ", ".join(
            "%s:%s" % kv for kv in sorted(r["wit_spectrum"].items(), key=lambda kv: int(kv[0]))))
    return out


def _slot_table(rec):
    show_cells = os.environ.get("CM_TABLE_CELLS", "1") == "1"
    pl = str(rec.get("dcirc_mode", "")) == "pl_circuit"
    cc = str(rec.get("dcirc_mode", "")).startswith("cc")
    top_rows = int(os.environ.get("CM_TABLE_TOP", "0"))     # 0 = list every slot
    slots = list(rec.get("slots", []))
    lines = ["idx  n    k   k-kb  w  d2s  proxy  depth  cnots     s  s_max  status" if pl
             else "idx  n    k   k-kb  w  d2s  proxy   d30s   kd30s^2/n  status" if cc
             else "idx  n    k    w  d2s  proxy   d_circ  kd_circ2/n  status"]
    shown = slots
    hidden = []
    if (cc or pl) and top_rows > 0 and len(slots) > top_rows + 10:
        order = sorted(slots, key=lambda r: -(r.get("proxy") or 0))
        shown = sorted(order[:top_rows], key=lambda r: r.get("idx", 0))
        hidden = order[top_rows:]
    for r in shown:
        status = str(r.get("status", "?"))
        if status.startswith("dup_archive") and r.get("archived_d_circ"):
            status += " (archived d=%s)" % r["archived_d_circ"]
        kb = (r.get("k") - r["k_bound"]) if (r.get("k") is not None and r.get("k_bound") is not None) else "-"
        if pl:
            lines.append(
                "%3s %4s %4s %4s %3s %4s %6s %6s %6s %5s %6s  %s" % (
                    r.get("idx", "?"), r.get("n", "-"), r.get("k", "-"), kb,
                    r.get("w", "-"),
                    r.get("d2s", "-"), r.get("proxy", "-"), r.get("depth", "-"),
                    r.get("cnots", "-"),
                    ("%.2f" % r["pl_score"]) if r.get("pl_score") is not None else "-",
                    ("%.2f" % r["pl_ceiling"]) if r.get("pl_ceiling") is not None else "-",
                    status))
            for b in ("Z", "X"):
                d = (r.get("pl") or {}).get(b)
                if not d:
                    continue
                tl = ", ".join("#%d:%d" % (i, v) for i, v in (d.get("top_logicals") or []))
                lines.append(
                    "     %s basis: %s shots (%s), %s failed shots flipping %s logical "
                    "qubits in total on %s of %s [%s], worst #%s; cascade settled %s at "
                    "stages 1/2/3, %s unsatisfied; ASC depth %s; s=%s (ceiling %s); %ss"
                    % (b, d.get("shots"),
                       "stopped at the %s-failed-shot cap" % d.get("max_errors")
                       if d.get("stopped_on") == "errors"
                       else "shot cap %s reached" % d.get("max_shots"),
                       d.get("shot_failures"),
                       d.get("logical_failures"), d.get("logicals_failed"),
                       d.get("k"), tl or "none", d.get("worst_logical"),
                       "/".join(str(x) for x in (d.get("resolved_by_stage") or [])),
                       d.get("unsatisfied"), d.get("depth"),
                       ("%.3f" % d["score"]) if d.get("score") is not None else "-",
                       "%.2f" % (d.get("score_ceiling") or 0), d.get("secs")))
            for b in ("Z", "X"):
                d = (r.get("pl") or {}).get(b)
                if not d:
                    continue
                if d.get("attrib"):
                    lines.append("     %s basis %s" % (b, _render_attrib(d["attrib"])))
                elif d.get("attrib_error"):
                    lines.append("     %s basis failure map unavailable: %s" % (b, d["attrib_error"]))
            lines.extend(_witness_lines(r))
            if r.get("asc_sched") and r.get("asc_sched") != "cached":
                lines.append("     ASC schedule: %s" % r["asc_sched"])
        elif cc:
            lines.append(
                "%3s %4s %4s %4s %3s %4s %7s %6s %10s  %s" % (
                    r.get("idx", "?"), r.get("n", "-"), r.get("k", "-"), kb,
                    r.get("w", "-"), r.get("d2s", "-"), r.get("proxy", "-"),
                    r.get("d_circ", "-"), r.get("kd_circ2_n", "-"), status))
            if r.get("status") == "measured" and r.get("wit_pool"):
                parts = []
                for side in ("x", "z"):
                    p = r["wit_pool"].get(side)
                    if not p:
                        continue
                    sec = p["sectors"]
                    parts.append("%s-type: %s%d operators at weight %d (%d only in sector 1, %d only in sector 2, %d in both)" % (
                        side.upper(), "at least " if p.get("capped") else "", p["n_ops"], p["weight"], sec["sector1_only"], sec["sector2_only"], sec["both"]))
                lines.append("     minimum-weight logicals found by the 30 s search: " + "; ".join(parts))
                dmin = min(v["weight"] for v in r["wit_pool"].values())
                for side in ("x", "z"):
                    p = r["wit_pool"].get(side)
                    if p and p["weight"] == dmin and p["blocks_ops"]:
                        bl = ", ".join("%s:%d" % kv for kv in sorted(p["blocks_ops"].items(), key=lambda kv: (-kv[1], kv[0])))
                        lines.append("     blocks touched by the %s-type operators (block: how many of the %d%s operators use it): %s" % (
                            side.upper(), p["n_ops"], " listed" if p.get("capped") else "", bl))
                if r.get("wit_spectrum"):
                    lines.append("     weight spectrum found (weight:verified operators, both types, uncapped): %s" % ", ".join("%s:%s" % kv for kv in sorted(r["wit_spectrum"].items(), key=lambda kv: int(kv[0]))))
        else:
            lines.append(
                "%3s %4s %4s %3s %4s %7s %7s %10s  %s" % (
                    r.get("idx", "?"), r.get("n", "-"), r.get("k", "-"),
                    r.get("w", "-"), r.get("d2s", "-"), r.get("proxy", "-"),
                    r.get("d_circ", "-"), r.get("kd_circ2_n", "-"), status))
        if show_cells and r.get("cells"):
            c = r["cells"]
            lines.append("     cells A=%s B=%s" % (
                json.dumps(c.get("A"), separators=(",", ":")),
                json.dumps(c.get("B"), separators=(",", ":"))))
    if hidden:
        hist = {}
        for r in hidden:
            hist[str(r.get("d2s", "-"))] = hist.get(str(r.get("d2s", "-")), 0) + 1
        lines.append("(%d further slots not listed: top %d by proxy shown above; d2s histogram of the rest: %s)" % (
            len(hidden), top_rows, ", ".join("%s:%d" % kv for kv in sorted(hist.items(), key=lambda kv: (len(kv[0]), kv[0])))))
    if show_cells:
        lines.append("(cells: each cell lists its group elements in "
                     "your CUSTOM_GROUP element encoding, e.g. (a,b) = "
                     "x^a y^b in a metacyclic presentation; sides A "
                     "then B, protograph row-major)")
    if pl:
        pr = rec.get("pl_protocol") or {}
        lines.append(
            "COLUMNS: k-kb = k above the rank-count bound q(ca-ra)(cb-rb); d2s = the 2 s code-capacity distance estimate, used ONLY to pick which five slots are "
            "measured; proxy = k*d2s^2/n, the selection key, NOT the score; depth = depth of the ASC syndrome-extraction "
            "circuit; cnots = CNOTs in it (= (nnz(HX)+nnz(HZ)) * %s rounds); s = THE SCORE of the slot; s_max = the score "
            "a zero-failure sample of this size would have given this code, i.e. the ceiling s saturates at (it grows "
            "with k and with the shot count)." % pr.get("rounds", "?"))
        lines.append(
            "MEASUREMENT: circuit-level memory experiment on the ASC-scheduled circuit at physical error rate p=%s, %s "
            "rounds, both bases, decoded by relay-BP twice then BP+LSD order 1. Sampling stops at %s failed shots or "
            "%s shots, whichever comes first, so a code that fails often is measured on fewer shots - the per-basis "
            "line says which cap ended it. A shot FAILS when any of its k logical qubits is wrong; one failed shot "
            "usually flips many logical qubits at once. s = -log10(upper end of the 95%% Clopper-Pearson interval on "
            "the failed-shot rate, converted per round and divided by k) - the per-round per-logical-qubit rate as "
            "Bravyi et al. and the mitten paper define it - and the slot's s is the WORSE of its two bases. "
            "FITNESS = the sum of s over the five measured slots divided by 5."
            % (pr.get("p", "?"), pr.get("rounds", "?"), pr.get("max_errors", "?"),
               pr.get("max_shots", "?")))
        lines.append(
            "CASCADE STAGES: 'settled a/b/c at stages 1/2/3' counts the shots the first relay-BP pass resolved, the "
            "shots that needed the longer second pass, and the shots that fell through to BP+LSD. Stages 2 and 3 are "
            "the shots where the decoder struggled; they cost nearly all of the measurement time and almost every "
            "logical failure comes from them. It is a far cheaper signal than s itself.")
        lines.append(
            "FAILURE MAP: taken from the scored shots themselves. For every shot the simulator also records the "
            "physical errors that actually happened - the final error on each data qubit (the component that flips "
            "the measured basis) and every flipped check measurement over the rounds - summed per protograph block. "
            "For each block the map reports ENRICHMENT = its share of those errors in the FAILED shots divided by its "
            "share in ALL shots: 1.0 = the failed shots carry this block's errors in the usual proportion, 2.0 = the "
            "errors that beat the decoder sat there twice as often as usual. Labels are the witness lines' blocks: "
            "Acolj/Bcolb = data block of A-column j with B-column b (sector 1); Arowi/Browa = data block of A-row i "
            "with B-row a (sector 2); Xchk:Arowi/Bcolb = X-check block of A-row i with B-column b; Zchk:Acolj/Browa = "
            "Z-check block of A-column j with B-row a. The mean data-error "
            "weight of failed vs all shots says how much heavier the failing error patterns are. WITNESS LINES: the "
            "minimum-weight logical operators a 30 s code-capacity search found for the slot and the blocks they use "
            "(an upper bound on d; the same lines the distance objective shows).")
        if rec.get("pl_error"):
            lines.append("MEASUREMENT FAILED, so the zeros below are not a statement about these codes: %s. Log "
                         "tail:\n%s"
                         % (rec["pl_error"], str(rec.get("pl_log_tail", ""))[-1500:]))
        nf = rec.get("n_pl_failed") or 0
        if nf:
            lines.append("UNMEASURED: %d of the five selected slots returned no pL (no solvable ASC schedule, or the "
                         "sampler failed on them); they scored 0 into the fixed denominator of 5." % nf)
    if cc:
        lines.append("COLUMNS: k-kb = k above the rank-count bound q(ca-ra)(cb-rb); "
                     "d2s = 2 s code-capacity search on every distinct slot; d30s = 30 s search on the top-5 by proxy (the smaller of the two readings); "
                     "both are upper bounds on the code distance. No circuit-level tier. For measured slots every minimum-weight logical operator the 30 s "
                     "search found is located (there are usually many; the search finds some of them, never provably all): how many per type, how they split "
                     "between the two LP sectors, and which protograph blocks they use (Acol j/Bcol b = sector-1 block of A-column j and B-column b; "
                     "Arow i/Brow a = sector-2 block). The weight spectrum lists, for each weight the search reached, how many verified operators it found.")
    nr = rec.get("n_no_reading", 0)
    if nr and not (cc or pl):
        lines.append("DEM SEARCH: on %d of the measured codes the circuit-level search "
                     "returned no reading (timeout or error); their d_circ is the "
                     "code-capacity witness embedded in the circuit, an upper bound only."
                     % nr)
    nd = rec.get("n_archive_dup", 0)
    if nd:
        lines.append("ARCHIVE: %d of %d slots reproduce codes an earlier "
                     "program already produced (dup_archive); they were "
                     "not measured and score nothing." % (
                         nd, len(rec.get("slots", []))))
    return "\n".join(lines)


def evaluate(program_path):
    try:
        spec = importlib.util.spec_from_file_location("prog", program_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        req = mod.get_candidates()
        assert isinstance(req, dict) and req.get("slots"), "no slots"
    except Exception as e:
        m = {"combined_score": 0.0,
             "error": "program:" + str(e)[:200],
             "artifacts": {"program_error": str(e)[:1000]}}
        try:
            rb = _bed_rate_bin(code=open(program_path).read())
        except Exception:
            rb = None
        if rb is not None:
            m["rate_bin"] = rb
            m["k_bin"] = rb
        return m
    if ARCHIVE:
        req["archive"] = ARCHIVE
    os.makedirs(SPOOL, exist_ok=True)
    rid = uuid.uuid4().hex[:12]
    tmp = os.path.join(SPOOL, "tmpreq_%s" % rid)
    with open(tmp, "w") as f:
        json.dump(req, f)
    os.replace(tmp, os.path.join(SPOOL, "req_%s.json" % rid))
    res_path = os.path.join(SPOOL, "res_%s.json" % rid)
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        if os.path.exists(res_path):
            try:
                rec = json.load(open(res_path))
            except Exception:
                time.sleep(2)          # publisher mid-replace; retry once
                rec = json.load(open(res_path))
            try:
                os.remove(res_path)
            except OSError:
                pass
            m = {"combined_score": float(rec.get("fitness", 0.0)),
                 "result": rec.get("result", "?"),
                 "n_distinct": rec.get("n_distinct", 0),
                 "n_measured": rec.get("n_measured", 0),
                 "n_archive_dup": rec.get("n_archive_dup", 0),
                 "secs": rec.get("secs", -1)}
            # MAP-Elites axis; a program with nothing measurable still
            # gets the bin of its shapes (see _bed_rate_bin)
            for key in ("rate", "rate_bin"):
                if key in rec:
                    m[key] = rec[key]
            if "rate_bin" not in m:
                rb = _bed_rate_bin(shapes=(req.get("a_shape"), req.get("b_shape")))
                if rb is not None:
                    m["rate_bin"] = rb
            # k_bin: the engine's "top programs" section shows ONE elite per
            # occupied k_bin of the parent's island (process_parallel), so
            # with k_bin = rate_bin the model sees the best program of each
            # rate band instead of the N highest scores of one cell
            if "rate_bin" in m:
                m["k_bin"] = m["rate_bin"]
            m["artifacts"] = {"slot_table": _slot_table(rec)}
            if rec.get("result", "").startswith("error"):
                m["artifacts"]["eval_error"] = rec["result"]
            if rec.get("pl_error"):
                # never a silent zero: the sampler's own words, in the prompt
                m["artifacts"]["eval_error"] = "pL sampler failed: %s\n%s" % (
                    rec["pl_error"], str(rec.get("pl_log_tail", ""))[-2000:])
                m["pl_error"] = 1
            m["n_pl_failed"] = rec.get("n_pl_failed", 0)
            if LEDGER:
                # "archive" is the key the prompt sampler replaces, at
                # prompt-build time, with the live RUN STATE text (EM_ARCH)
                m["artifacts"]["archive"] = "(run state not available yet)"
                _ledger_write(program_path, req, rec, m)
            return m
        time.sleep(5)
    m = {"combined_score": 0.0, "error": "timeout-no-worker",
         "artifacts": {"eval_error":
                       "no worker answered within %ds" % TIMEOUT}}
    rb = _bed_rate_bin(shapes=(req.get("a_shape"), req.get("b_shape")))
    if rb is not None:
        m["rate_bin"] = rb
        m["k_bin"] = rb
    return m

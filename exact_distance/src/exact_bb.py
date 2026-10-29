"""Driver for dkp_bb: certified minimum-distance bounds of CSS codes.

  d(H, G) = min { wt(x) : H x = 0, x not in rowspan(G) }
  X-distance: H = HZ, G = HX      Z-distance: H = HX, G = HZ

A search at weight limit W ends in FOUND (an operator of weight <= W, re-verified here; an upper
bound only), NONE (d > W) or TIMEOUT (nothing proved). Results are reported as

  lower  : the largest W of a completed NONE search (d > lower); 0 when nothing is proved,
  upper  : the smallest weight of a verified FOUND operator (d <= upper),
  exact  : upper, but only when lower >= upper - 1.

A level split into frontier slices (--slice a/b) is certified only by --aggregate, which checks
every slice record against a manifest written before the slices ran.

  python exact_bb.py --npz code.npz --n 585 --type both --wmin 1 --wmax 24 --threads 8 --time 3600 --bin ./dkp_bb
  python exact_bb.py --make-manifest --run-id R --npz code.npz --n 585 --type X --W 23 --orbits o.json --slices 60 --bin ./dkp_bb --out m.json
  python exact_bb.py --expect m.json --slice 0/60 --threads 32 --time 400000 --out s0.json     (settings taken from the manifest)
  python exact_bb.py --aggregate s*.json --expect m.json --out agg.json
  python exact_bb.py --selftest --bin ./dkp_bb
"""
import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

EXIT_OK, EXIT_USAGE, EXIT_INCOMPLETE = 0, 2, 3


class InputError(Exception):
    pass


# ---------------------------------------------------------------- input
def load_code(npz, n=None):
    """Load HX, HZ from an npz as 0/1 uint8 arrays with n columns.

    Accepted: unpacked 0/1 arrays with n columns, or np.packbits(..., axis=1) arrays with ceil(n/8)
    columns and zero padding bits. n comes from --n or from an 'n' entry in the npz; packed input
    without n is rejected, and so is any shape or value that fits neither format."""
    z = np.load(npz)
    if "HX" not in z or "HZ" not in z:
        raise InputError("npz must contain HX and HZ")
    if n is None and "n" in z:
        n = int(z["n"])
    out = []
    for key in ("HX", "HZ"):
        M = np.asarray(z[key])
        if M.ndim != 2:
            raise InputError("%s is not a matrix" % key)
        binary = bool(np.isin(M, (0, 1)).all())
        if n is None:
            raise InputError("the number of qubits n is required (give --n or store 'n' in the npz); "
                             "a packed matrix cannot be told apart from a 0/1 matrix by its shape")
        if M.shape[1] == n and binary and not (M.shape[1] == (n + 7) // 8 and M.max(initial=0) > 1):
            A = M.astype(np.uint8)
        elif M.shape[1] == (n + 7) // 8 and M.dtype == np.uint8:
            A = np.unpackbits(M, axis=1)
            if A[:, n:].any():
                raise InputError("%s: packed rows have nonzero padding bits beyond column n=%d" % (key, n))
            A = A[:, :n]
        else:
            raise InputError("%s has shape %s and dtype %s, neither 0/1 with n=%d columns nor packed with %d" % (
                key, M.shape, M.dtype, n, (n + 7) // 8))
        out.append(A % 2)
    HX, HZ = out
    if HX.shape[1] != HZ.shape[1]:
        raise InputError("HX and HZ have different numbers of columns")
    if ((HX.astype(np.int64) @ HZ.T.astype(np.int64)) % 2).any():
        raise InputError("HX HZ^T != 0: not a CSS code")
    return HX, HZ


def matrix_fingerprint(HX, HZ):
    h = hashlib.sha256()
    h.update(b"CSS-HX-HZ-v1")
    h.update(np.asarray(HX.shape + HZ.shape, np.int64).tobytes())
    h.update(np.packbits(HX, axis=1).tobytes())
    h.update(np.packbits(HZ, axis=1).tobytes())
    return h.hexdigest()


def archive_name(HX, HZ):
    """The evaluator's archive file name (sha1 of the 0/1 arrays), for cross-reference."""
    return hashlib.sha1(HX.tobytes() + HZ.tobytes()).hexdigest()[:16]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(obj, path):
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------- GF(2)
def gf2_rank(M):
    rows = [int("".join("1" if b else "0" for b in r[::-1]) or "0", 2) for r in np.asarray(M, np.uint8) % 2]
    basis = {}
    for v in rows:
        while v:
            p = v.bit_length() - 1
            if p in basis:
                v ^= basis[p]
            else:
                basis[p] = v
                break
    return len(basis)


def verify_operator(H, G, support, n):
    if len(set(support)) != len(support) or any(not (0 <= v < n) for v in support):
        return False
    x = np.zeros(n, np.uint8)
    x[list(support)] = 1
    if ((H.astype(np.int64) @ x.astype(np.int64)) % 2).any():
        return False
    return gf2_rank(np.vstack([G, x[None, :]])) > gf2_rank(G)


# ---------------------------------------------------------------- orbits
def validate_orbits(ob, HX, HZ):
    """Check an orbits record against the loaded code and return the orbits regenerated from its
    verified generators, in canonical order (largest first, then by least element)."""
    n = HX.shape[1]
    if int(ob.get("n", -1)) != n:
        raise InputError("orbit file is for n=%s, the code has n=%d" % (ob.get("n"), n))
    fp = ob.get("matrix_sha256")
    if fp is not None and fp != matrix_fingerprint(HX, HZ):
        raise InputError("orbit file belongs to a different matrix (matrix_sha256 mismatch)")
    gens = ob.get("generators")
    if gens is None:
        raise InputError("orbit file has no generators to verify")
    rX, rZ = gf2_rank(HX), gf2_rank(HZ)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for gi, p in enumerate(gens):
        p = [int(v) for v in p]
        if sorted(p) != list(range(n)):
            raise InputError("generator %d is not a permutation of the n qubits" % gi)
        for M, r in ((HX, rX), (HZ, rZ)):
            if gf2_rank(np.vstack([M, M[:, p]])) != r:
                raise InputError("generator %d does not preserve the row space of %s" % (gi, "HX" if M is HX else "HZ"))
        for i, j in enumerate(p):
            a, b = find(i), find(j)
            if a != b:
                parent[a] = b
    comp = {}
    for i in range(n):
        comp.setdefault(find(i), []).append(i)
    regen = sorted((sorted(c) for c in comp.values()), key=lambda o: (-len(o), o[0]))
    stored = ob.get("orbits")
    if stored is not None and sorted(sorted(int(v) for v in o) for o in stored) != sorted(regen):
        raise InputError("stored orbits differ from the orbits generated by the verified generators")
    return regen


def orbit_roots(orbits):
    """Roots for dkp_bb (mode 2): the representative of each orbit, with every qubit of the
    orbits before it forbidden."""
    roots, before = [], []
    for orb in orbits:
        roots.append((int(orb[0]), 2, sorted(before)))
        before += [int(v) for v in orb]
    return roots


# ---------------------------------------------------------------- search
def problem_text(H, G, W, threads, roots):
    n = H.shape[1]
    lines = ["%d %d %d %d %d" % (n, H.shape[0], G.shape[0], W, threads)]
    for M in (H, G):
        for row in M:
            idx = np.flatnonzero(row)
            lines.append("%d %s" % (len(idx), " ".join(map(str, idx))))
    lines.append("%d" % len(roots))
    for rt in roots:
        if len(rt) == 3:
            lines.append("%d 2 %d %s" % (rt[0], len(rt[2]), " ".join(map(str, rt[2]))))
        else:
            lines.append("%d %d" % (rt[0], rt[1]))
    return "\n".join(lines) + "\n"


def problem_fingerprint(H, G, W, roots):
    return hashlib.sha256(problem_text(H, G, W, 0, roots).encode()).hexdigest()


def run_bb(binary, H, G, W, threads, tlimit, roots=None, slice_spec=None, depth=None):
    n = H.shape[1]
    if roots is None:
        roots = [(v, 1) for v in range(n)]
    cmd = [binary, repr(float(tlimit))]
    if slice_spec is not None:
        cmd += [str(slice_spec[0]), str(slice_spec[1])]
        if depth:
            cmd += [str(depth)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=problem_text(H, G, W, threads, roots), capture_output=True, text=True,
                           timeout=tlimit * 1.1 + 600)
        out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        out, err, rc = "", "process killed by the external time guard", -9
    except OSError as e:
        out, err, rc = "", "cannot run binary: %s" % e, -1
    res = dict(W=W, secs=round(time.time() - t0, 3), tlimit=tlimit, returncode=rc, stderr=err.strip()[-400:],
               status=None, weight=None, support=None, nodes=None, info=None,
               slice=list(slice_spec) if slice_spec is not None else None, n_roots=len(roots),
               problem_sha256=problem_fingerprint(H, G, W, roots))
    for line in out.strip().split("\n"):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "FOUND":
            res["status"] = "FOUND"; res["weight"] = int(parts[1]); res["support"] = [int(v) for v in parts[2:]]
        elif parts[0] == "NONE":
            res["status"] = "NONE" if int(parts[1]) == W else "ERROR"
        elif parts[0] == "TIMEOUT":
            res["status"] = "TIMEOUT"; res["progress"] = parts[1]
        elif parts[0] == "STATS":
            res["nodes"] = int(parts[1]); res["bb_secs"] = float(parts[2])
        elif parts[0] == "INFO":
            res["info"] = dict(kv.split("=", 1) for kv in parts[1:])
    if rc != 0 or res["status"] is None or res["info"] is None:
        res["status"] = "ERROR"
    if res["status"] == "FOUND":
        res["verified"] = bool(verify_operator(H, G, res["support"], n) and len(res["support"]) == res["weight"] <= W)
        if not res["verified"]:
            res["status"] = "BAD_FOUND"
    return res


def climb(binary, H, G, wmin, wmax, threads, tlimit, roots=None, log=print):
    """Searches W = wmin..wmax until a verified FOUND, a failure or wmax. Returns a dict with the
    proved lower bound (0 when nothing was proved), the upper bound (None if no operator was
    found), the exact value (only when they meet), completeness and the runs."""
    if wmin < 1 or wmax < wmin:
        raise InputError("need 1 <= wmin <= wmax (got %d, %d)" % (wmin, wmax))
    lower, upper, runs, complete = 0, None, [], True
    for W in range(wmin, wmax + 1):
        r = run_bb(binary, H, G, W, threads, tlimit, roots)
        runs.append(r)
        log("   W=%d: %s%s  nodes=%s  %.1f s" % (W, r["status"], (" weight %d" % r["weight"]) if r["weight"] else "", r["nodes"], r["secs"]))
        if r["status"] == "NONE":
            lower = max(lower, W)
        elif r["status"] == "FOUND":
            upper = r["weight"] if upper is None else min(upper, r["weight"])
            break
        else:
            complete = False
            break
    exact = upper if (upper is not None and lower >= upper - 1) else None
    return dict(lower=lower, upper=upper, exact=exact, complete=complete, runs=runs)


# ---------------------------------------------------------------- manifest and slices
def prepare(a):
    HX, HZ = load_code(a["npz"], a.get("n"))
    roots, orbit_info = None, None
    if a.get("orbits"):
        ob = json.load(open(a["orbits"]))
        orbs = validate_orbits(ob, HX, HZ)
        roots = orbit_roots(orbs)
        orbit_info = dict(path=a["orbits"], sha256=file_sha256(a["orbits"]), n_orbits=len(orbs))
    return HX, HZ, roots, orbit_info


def make_manifest(a):
    HX, HZ, roots, orbit_info = prepare(vars(a))
    t = a.type
    H, G = (HZ, HX) if t == "X" else (HX, HZ)
    n = HX.shape[1]
    rts = roots if roots is not None else [(v, 1) for v in range(n)]
    m = dict(run_id=a.run_id, npz=a.npz, n=n, matrix_sha256=matrix_fingerprint(HX, HZ), archive_name=archive_name(HX, HZ),
             type=t, W=a.W, slices=a.slices, depth=a.depth, orbits=orbit_info,
             problem_sha256=problem_fingerprint(H, G, a.W, rts), binary=os.path.abspath(a.bin), binary_sha256=file_sha256(a.bin),
             created=time.strftime("%Y-%m-%d %H:%M:%S"))
    atomic_json(m, a.out)
    print("manifest", a.out, "run", a.run_id, "problem", m["problem_sha256"][:16], "matrix", m["matrix_sha256"][:16])
    return EXIT_OK


def run_slice(a, m):
    """One frontier slice of the level described by the manifest m; refuses to run if the inputs
    on disk no longer match the manifest."""
    HX, HZ, roots, orbit_info = prepare(dict(npz=m["npz"], n=m["n"], orbits=(m["orbits"] or {}).get("path")))
    if matrix_fingerprint(HX, HZ) != m["matrix_sha256"]:
        raise InputError("matrix on disk differs from the manifest")
    if m["orbits"] and orbit_info["sha256"] != m["orbits"]["sha256"]:
        raise InputError("orbit file on disk differs from the manifest")
    if file_sha256(a.bin) != m["binary_sha256"]:
        raise InputError("binary differs from the manifest")
    sa, sb = (int(v) for v in a.slice.split("/"))
    if sb != m["slices"] or not (0 <= sa < sb):
        raise InputError("slice %d/%d does not fit the manifest's %d slices" % (sa, sb, m["slices"]))
    t = m["type"]
    H, G = (HZ, HX) if t == "X" else (HX, HZ)
    n = HX.shape[1]
    rts = roots if roots is not None else [(v, 1) for v in range(n)]
    r = run_bb(a.bin, H, G, m["W"], a.threads, a.time, rts, (sa, sb), m.get("depth"))
    if r["problem_sha256"] != m["problem_sha256"]:
        raise InputError("problem fingerprint differs from the manifest")
    rec = dict(kind="slice", run_id=m["run_id"], matrix_sha256=m["matrix_sha256"], problem_sha256=r["problem_sha256"],
               binary_sha256=m["binary_sha256"], type=t, W=m["W"], slice=[sa, sb], run=r,
               host=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID"))
    if a.out:
        atomic_json(rec, a.out)
    print("SLICE %d/%d of run %s: %s (nodes %s, %.1f s) - a slice proves nothing about the code on its own" % (
        sa, sb, m["run_id"], r["status"], r["nodes"], r["secs"]), flush=True)
    return EXIT_OK if r["status"] in ("NONE", "FOUND") else EXIT_INCOMPLETE


def aggregate(files, m, out=None):
    """Certify one level from its slice records, checked against the manifest m."""
    HX, HZ = load_code(m["npz"], m["n"])
    if matrix_fingerprint(HX, HZ) != m["matrix_sha256"]:
        raise InputError("matrix on disk differs from the manifest")
    t, W, b = m["type"], m["W"], m["slices"]
    H, G = (HZ, HX) if t == "X" else (HX, HZ)
    problems, recs = [], {}
    for f in files:
        try:
            r = json.load(open(f))
        except Exception as e:
            problems.append("%s unreadable (%s)" % (f, e)); continue
        for key in ("run_id", "matrix_sha256", "problem_sha256", "binary_sha256", "type", "W"):
            if r.get(key) != m.get(key):
                problems.append("%s: %s=%s, manifest %s" % (os.path.basename(f), key, str(r.get(key))[:16], str(m.get(key))[:16]))
        sl = r.get("slice") or [None, None]
        if sl[1] != b:
            problems.append("%s: slice count %s, manifest %d" % (os.path.basename(f), sl[1], b))
        if sl[0] in recs:
            problems.append("slice %s appears twice" % sl[0])
        # the record's own run data must agree with its outer label (a relabelled or copied record is refused)
        run = r.get("run") or {}
        ri = run.get("info") or {}
        try:
            if run.get("problem_sha256") != m.get("problem_sha256"):
                problems.append("%s: the search ran a different problem than the manifest" % os.path.basename(f))
            if run.get("slice") is not None and list(run["slice"]) != list(sl):
                problems.append("%s: labelled slice %s but the search ran slice %s" % (os.path.basename(f), sl, run["slice"]))
            if int(ri["W"]) != W:
                problems.append("%s: the binary ran W=%s, manifest W=%d" % (os.path.basename(f), ri["W"], W))
            if int(ri["slice"]) != sl[0] or int(ri["slices"]) != sl[1]:
                problems.append("%s: the binary ran slice %s/%s, record labelled %s/%s" % (os.path.basename(f), ri["slice"], ri["slices"], sl[0], sl[1]))
            N = int(ri["frontier_total"])
            expected = (N - int(sl[0]) + b - 1) // b if int(sl[0]) < N else 0    # subtrees with index % b == slice
            if int(ri["frontier_slice"]) != expected:
                problems.append("%s: frontier share %s, expected %d of %d" % (os.path.basename(f), ri["frontier_slice"], expected, N))
            if run.get("status") not in ("NONE", "FOUND"):
                problems.append("%s: status %s" % (os.path.basename(f), run.get("status")))
        except (KeyError, TypeError, ValueError):
            problems.append("%s: incomplete search information" % os.path.basename(f))
        recs[sl[0]] = r
    if sorted(k for k in recs if k is not None) != list(range(b)):
        problems.append("slices present %s, expected exactly 0..%d" % (sorted(k for k in recs if k is not None)[:20], b - 1))
    totals = {(r["run"].get("info") or {}).get("frontier_total") for r in recs.values()}
    depths = {(r["run"].get("info") or {}).get("depth") for r in recs.values()}
    if len(totals) != 1 or None in totals or len(depths) != 1 or None in depths:
        problems.append("slices disagree on the frontier (totals %s, depths %s)" % (sorted(map(str, totals)), sorted(map(str, depths))))
    found = []
    for k, r in recs.items():
        st = r["run"]["status"]
        if st == "FOUND":
            if verify_operator(H, G, r["run"]["support"], HX.shape[1]) and len(r["run"]["support"]) <= W:
                found.append(len(r["run"]["support"]))
            else:
                problems.append("slice %s: FOUND operator fails verification" % k)
        elif st != "NONE":
            problems.append("slice %s: status %s" % (k, st))
    verdict = dict(run_id=m["run_id"], type=t, W=W, slices=b, matrix_sha256=m["matrix_sha256"], problem_sha256=m["problem_sha256"])
    if found and not [p for p in problems if "verification" in p]:
        verdict.update(result="FOUND", upper=min(found), level_complete=not problems, problems=problems)
        line = "%s W=%d: FOUND verified operator of weight %d - upper bound d_%s <= %d (exact only together with a certified level W=%d)%s" % (
            t, W, min(found), t, min(found), min(found) - 1,
            "" if not problems else "; the level itself is not complete: " + "; ".join(problems[:4]))
        code = EXIT_OK
    elif not problems:
        verdict.update(result="CERTIFIED", lower=W)
        nodes = sum(r["run"].get("nodes") or 0 for r in recs.values())
        line = "%s W=%d: all %d slices NONE, manifest checks passed - certified d_%s > %d (nodes %d)" % (t, W, b, t, W, nodes)
        code = EXIT_OK
    else:
        verdict.update(result="INCOMPLETE", problems=problems)
        line = "%s W=%d: INCOMPLETE - %s" % (t, W, "; ".join(problems[:6]))
        code = EXIT_INCOMPLETE
    print(line, flush=True)
    if out:
        atomic_json(verdict, out)
    return code


# ---------------------------------------------------------------- self-test
def hamming(rbits):
    cols = list(range(1, 2 ** rbits))
    return np.array([[(c >> b) & 1 for c in cols] for b in range(rbits)], np.uint8)


def toric(L):
    n = 2 * L * L
    def e_h(i, j): return (i % L) * L + (j % L)
    def e_v(i, j): return L * L + (i % L) * L + (j % L)
    HX = np.zeros((L * L, n), np.uint8); HZ = np.zeros((L * L, n), np.uint8)
    for i in range(L):
        for j in range(L):
            v = i * L + j
            for e in (e_h(i, j), e_h(i, j - 1), e_v(i, j), e_v(i - 1, j)):
                HX[v, e] ^= 1
            for e in (e_h(i, j), e_h(i + 1, j), e_v(i, j), e_v(i, j + 1)):
                HZ[v, e] ^= 1
    return HX, HZ


def brute_distance(H, G, wmax):
    n = H.shape[1]
    rG = gf2_rank(G)
    for w in range(1, wmax + 1):
        for comb in itertools.combinations(range(n), w):
            x = np.zeros(n, np.uint8); x[list(comb)] = 1
            if ((H.astype(np.int64) @ x.astype(np.int64)) % 2).any():
                continue
            if gf2_rank(np.vstack([G, x[None, :]])) > rG:
                return w
    return None


def selftest(binary):
    rng = np.random.default_rng(7)
    cases = []
    H7 = hamming(3); cases.append(("Steane [[7,1,3]]", H7, H7, 3))
    H15 = hamming(4); cases.append(("[[15,7,3]]", H15, H15, 3))
    for L in (3, 4, 5):
        HX, HZ = toric(L); cases.append(("toric L=%d" % L, HX, HZ, L))
    for trial in range(3):
        while True:
            A = (rng.random((3, 6)) < 0.4).astype(np.uint8); B = (rng.random((2, 5)) < 0.5).astype(np.uint8)
            if A.sum(1).min() and B.sum(1).min():
                break
        mA, nA = A.shape; mB, nB = B.shape
        HX = np.hstack([np.kron(A, np.eye(nB, dtype=np.uint8)), np.kron(np.eye(mA, dtype=np.uint8), B.T)]) % 2
        HZ = np.hstack([np.kron(np.eye(nA, dtype=np.uint8), B), np.kron(A.T, np.eye(mB, dtype=np.uint8))]) % 2
        cases.append(("HGP %d" % trial, HX, HZ, None))
    ok = True
    for name, HX, HZ, dknown in cases:
        for t, (H, G) in (("X", (HZ, HX)), ("Z", (HX, HZ))):
            dk = brute_distance(H, G, 6) if dknown is None else dknown
            if dk is None:
                continue
            r = climb(binary, H, G, 1, dk + 1, 1, 600, log=lambda s: None)
            good = r["exact"] == dk and r["complete"] and r["lower"] == dk - 1
            ok &= good
            print("%-18s %s-type: expected %s got exact=%s (d > %d, d <= %s) %s" % (name, t, dk, r["exact"], r["lower"], r["upper"], "ok" if good else "MISMATCH"))
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz"); ap.add_argument("--n", type=int)
    ap.add_argument("--type", default="both", choices=["X", "Z", "both"])
    ap.add_argument("--wmin", type=int, default=1); ap.add_argument("--wmax", type=int, default=40)
    ap.add_argument("--threads", type=int, default=1); ap.add_argument("--time", type=float, default=3600)
    ap.add_argument("--bin", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dkp_bb"))
    ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--orbits", help="orbits JSON from code_orbits.py, re-validated on the loaded matrices")
    ap.add_argument("--make-manifest", action="store_true"); ap.add_argument("--run-id"); ap.add_argument("--W", type=int)
    ap.add_argument("--slices", type=int); ap.add_argument("--depth", type=int)
    ap.add_argument("--expect", help="manifest of the level (slice runs and aggregation)")
    ap.add_argument("--slice", help="a/b, with --expect")
    ap.add_argument("--aggregate", nargs="+", help="slice records of one level, with --expect")
    a = ap.parse_args()
    try:
        if a.selftest:
            return EXIT_OK if selftest(a.bin) else EXIT_INCOMPLETE
        if a.make_manifest:
            if not (a.run_id and a.npz and a.W and a.slices and a.type in ("X", "Z") and a.out):
                raise InputError("--make-manifest needs --run-id --npz --n --type X|Z --W --slices --out")
            return make_manifest(a)
        if a.aggregate:
            if not a.expect:
                raise InputError("--aggregate needs --expect MANIFEST")
            return aggregate(a.aggregate, json.load(open(a.expect)), a.out)
        if a.slice:
            if not a.expect:
                raise InputError("--slice needs --expect MANIFEST written before the slices were submitted")
            return run_slice(a, json.load(open(a.expect)))
        HX, HZ, roots, orbit_info = prepare(vars(a))
        n = HX.shape[1]
        res = dict(kind="climb", npz=a.npz, n=n, matrix_sha256=matrix_fingerprint(HX, HZ), archive_name=archive_name(HX, HZ),
                   orbits=orbit_info, types={})
        code = EXIT_OK
        for t in (["X", "Z"] if a.type == "both" else [a.type]):
            H, G = (HZ, HX) if t == "X" else (HX, HZ)
            print("== %s-type: H %s G %s, W from %d to %d" % (t, H.shape, G.shape, a.wmin, a.wmax), flush=True)
            r = climb(a.bin, H, G, a.wmin, a.wmax, a.threads, a.time, roots, log=lambda s: print(s, flush=True))
            res["types"][t] = r
            if not r["complete"]:
                code = EXIT_INCOMPLETE
            print("   %s-type: d_%s > %d%s%s%s" % (t, t, r["lower"], "" if r["upper"] is None else ", d_%s <= %d" % (t, r["upper"]),
                  "" if r["exact"] is None else ", exact d_%s = %d" % (t, r["exact"]), "" if r["complete"] else "  [INCOMPLETE]"), flush=True)
        # the quantum distance is min(d_X, d_Z): a type that was not searched contributes the lower bound 0
        lows = [res["types"][t]["lower"] if t in res["types"] else 0 for t in ("X", "Z")]
        ups = [v["upper"] for v in res["types"].values() if v["upper"] is not None]
        res["d_lower_exclusive"] = min(lows)                     # d > this
        res["d_upper"] = min(ups) if ups else None
        res["d_exact"] = res["d_upper"] if (res["d_upper"] is not None and len(res["types"]) == 2 and min(lows) >= res["d_upper"] - 1) else None
        if a.out:
            atomic_json(res, a.out)
        print("RESULT d > %d, d <= %s, exact d = %s%s" % (res["d_lower_exclusive"], res["d_upper"], res["d_exact"],
              "" if len(res["types"]) == 2 else "  (only type %s searched: no bound on the other type)" % a.type), flush=True)
        return code
    except InputError as e:
        print("INPUT ERROR: %s" % e, file=sys.stderr, flush=True)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())

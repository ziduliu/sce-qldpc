"""Worker-side evaluator (distance objective): same request, pipeline and
record as eval_circmean.py, using dem_distance_v2; in code-capacity mode
(CM_DIST_MODE=cc) each measured slot also records its 4-cycle counts and the
pooled minimum-weight logical operators found by the deep search.
"""
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("LAYOUT_SQFAST_PATH", "/u/zdliu/lp/dopt/agentA")
for _p in ("/u/zdliu/lp/circmean", "/u/zdliu/lp/lps_graftgate",
           "/u/zdliu/lp/dopt/bdirect_v2", "/u/zdliu/lp/lifted_product_search"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

N_MIN = int(os.environ.get("CM_NMIN", "100"))
N_MAX = int(os.environ.get("CM_NMAX", "600"))
W_MAX = int(os.environ.get("CM_WMAX", "12"))
# optional rate floor (0 = off)
RATE_FLOOR = float(os.environ.get("CM_RATE_FLOOR", "0"))
TOP_K = int(os.environ.get("CM_TOPK", "5"))
D2S_BUDGET = float(os.environ.get("CM_D2S_BUDGET", "2"))
DCIRC_BUDGET = float(os.environ.get("CM_DCIRC_BUDGET", "60"))
# DEM search tier of the circuit-level distance; off by default for the
# n in [1000, 1500) window (embedding and exact w<=2 checks only).
DEM_SEARCH_ENABLED = os.environ.get(
    "CM_DEM_SEARCH", "0" if (N_MIN, N_MAX) == (1000, 1500) else "1") == "1"
# CM_DIST_MODE=cc: no circuit-level tier. The top-K by proxy get a longer
# code-capacity search (CM_D2S_DEEP seconds) and its result is the d that
# enters the fitness. The field names d_circ/kd_circ2_n are kept; dcirc_mode
# says which quantity they hold.
DIST_MODE = os.environ.get("CM_DIST_MODE", "circ")
D2S_DEEP = float(os.environ.get("CM_D2S_DEEP", "30"))
ISO_EXACT = os.environ.get("CM_ISO_EXACT", "0") == "1"
# archive collisions on the fingerprint alone are NOT trusted: over a
# run-wide archive of thousands of codes the false-merge rate of the
# fingerprint is far higher than inside one 20-slot batch, so a
# fingerprint hit is confirmed with VF2++ (timeout -> distinct)
ARCH_ISO_TIMEOUT = int(os.environ.get("CM_ARCH_ISO_TIMEOUT", "20"))
# MAP-Elites axis: rate_bin = cell index of the mean k/n of the program's
# measured codes (engine convention: a *_bin metric is the bin index); cells
# are RATE_STEP wide, centred at RATE_LO + RATE_STEP * b, and rates outside
# the grid are clipped into the end cells.
RATE_LO = float(os.environ.get("CM_RATE_LO", "0.15"))
RATE_STEP = float(os.environ.get("CM_RATE_STEP", "0.025"))
RATE_BINS = int(os.environ.get("CM_RATE_BINS", "15"))


def rate_bin_of(rate):
    b = int(round((rate - RATE_LO) / RATE_STEP))
    return max(0, min(RATE_BINS - 1, b))


class CodeArchive:
    """Run-wide registry of every code produced so far.

    Layout under root:
      codes/<h>.json  {family, idx, n, k, w, d2s, d_circ, ts}
      codes/<h>.npz   packed HX, HZ (for the exact isomorphism confirm)
      iso/<ikey_dir>/<h>   marker; ikey_dir = n_mX_mZ_k_<fingerprint>
    h = sha1(HX bytes + HZ bytes)[:16]. Files are written atomically
    (tmp + rename); two workers may register concurrently.
    """

    def __init__(self, root):
        self.root = root
        os.makedirs(os.path.join(root, "codes"), exist_ok=True)
        os.makedirs(os.path.join(root, "iso"), exist_ok=True)

    def _meta_path(self, h):
        return os.path.join(self.root, "codes", h + ".json")

    def _iso_dir(self, n, HX, HZ, k, ikey):
        tag = ikey if ikey.startswith("%d_" % n) else "%d_%d_%d_%d_%s" % (n, HX.shape[0], HZ.shape[0], k, ikey)
        return os.path.join(self.root, "iso", tag)

    def lookup(self, h, n, HX, HZ, k, ikey, iso_dedup):
        """Return the archived meta dict of a duplicate, or None."""
        p = self._meta_path(h)
        if os.path.exists(p):
            return self._load(p)
        d = self._iso_dir(n, HX, HZ, k, ikey)
        if not os.path.isdir(d):
            return None
        import numpy as np
        for other in sorted(os.listdir(d)):
            if other == h:
                continue
            try:
                z = np.load(os.path.join(self.root, "codes", other + ".npz"))
                HX2 = np.unpackbits(z["HX"], axis=1)[:, :n]
                HZ2 = np.unpackbits(z["HZ"], axis=1)[:, :n]
            except Exception:
                continue
            if HX2.shape != HX.shape or HZ2.shape != HZ.shape:
                continue
            if iso_dedup.is_isomorphic_exact(HX, HZ, HX2, HZ2,
                                             timeout=ARCH_ISO_TIMEOUT):
                return self._load(self._meta_path(other))
        return None

    def register(self, h, n, HX, HZ, k, ikey, meta):
        import numpy as np
        p_npz = os.path.join(self.root, "codes", h + ".npz")
        if not os.path.exists(p_npz):
            tmp = p_npz + ".%d.tmp" % os.getpid()
            with open(tmp, "wb") as f:
                np.savez_compressed(f, HX=np.packbits(HX, axis=1),
                                    HZ=np.packbits(HZ, axis=1))
            os.replace(tmp, p_npz)
        self._write_meta(h, meta)
        d = self._iso_dir(n, HX, HZ, k, ikey)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, h), "a").close()

    def unregister(self, h, n, HX, HZ, k, ikey):
        d = self._iso_dir(n, HX, HZ, k, ikey)
        for p in (os.path.join(d, h), self._meta_path(h),
                  os.path.join(self.root, "codes", h + ".npz")):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(d)
        except OSError:
            pass

    def update(self, h, **fields):
        p = self._meta_path(h)
        meta = self._load(p) or {}
        meta.update(fields)
        self._write_meta(h, meta)

    def _write_meta(self, h, meta):
        p = self._meta_path(h)
        tmp = p + ".%d.tmp" % os.getpid()
        with open(tmp, "w") as f:
            json.dump(meta, f)
        os.replace(tmp, p)

    @staticmethod
    def _load(p):
        try:
            return json.load(open(p))
        except Exception:
            return {}


def _ops_from_table(mul_table, q):
    """Group ops over elements 0..q-1 from the multiplication table."""
    ident = None
    for a in range(q):
        if all(mul_table[a][h] == h for h in range(q)):
            ident = a
            break
    if ident is None:
        raise ValueError("mul table has no identity")
    inv = [None] * q
    for a in range(q):
        for b in range(q):
            if mul_table[a][b] == ident:
                inv[a] = b
                break
        if inv[a] is None:
            raise ValueError("element %d has no inverse" % a)
    return dict(q=q, elements=list(range(q)), ident=ident,
                mul=lambda g, h: mul_table[int(g) % q][int(h) % q],
                inv=lambda g: inv[int(g) % q],
                idx=lambda g: int(g) % q,
                red=lambda g: int(g) % q)



# ---- structural measurements reported with the slot table ----
def _cycle4_count(H):
    """Number of 4-cycles in the Tanner graph of H = sum over check pairs of C(overlap, 2)."""
    import numpy as _np
    Hi = _np.asarray(H, _np.int64)
    C = Hi @ Hi.T
    _np.fill_diagonal(C, 0)
    return int((C * (C - 1) // 2).sum() // 2)


def _witness_blocks(support, q, ra, ca, rb, cb):
    """Map a data-qubit support to the two LP sectors and the protograph blocks it touches.
    Sector 1 columns: (A-column j, B-column beta) blocks; sector 2: (A-row i, B-row alpha) blocks
    (layout of random_search_all_levels._lifted_product)."""
    n1 = ca * cb * q
    w1 = sum(1 for c in support if c < n1)
    w2 = len(support) - w1
    blocks = {}
    for c in support:
        b = c // q
        if c < n1:
            j, beta = divmod(b, cb)
            key = "Acol%d/Bcol%d" % (j, beta)
        else:
            i, alpha = divmod(b - ca * cb, rb)
            key = "Arow%d/Brow%d" % (i, alpha)
        blocks[key] = blocks.get(key, 0) + 1
    return w1, w2, blocks


def _pool_witnesses(sups, q, ra, ca, rb, cb):
    """Summarise a family of minimum-weight operators (list of supports):
    how many, how they split between the two LP sectors, and how many of
    them touch each protograph block (block -> number of operators)."""
    sectors = {"sector1_only": 0, "sector2_only": 0, "both": 0}
    blocks_ops = {}
    for sup in sups:
        w1, w2, bl = _witness_blocks(sup, q, ra, ca, rb, cb)
        key = "sector1_only" if w2 == 0 else ("sector2_only" if w1 == 0 else "both")
        sectors[key] += 1
        for b in bl:
            blocks_ops[b] = blocks_ops.get(b, 0) + 1
    return {"n_ops": len(sups), "sectors": sectors, "blocks_ops": blocks_ops}


def main():
    import numpy as np
    import fast_gf2
    from fast_gf2 import logicals
    from random_search_all_levels import _lift as rs_lift, \
        _lifted_product as rs_lp
    import dem_distance_v2 as DD
    import iso_dedup

    EM_IN = os.environ["EM_IN"]
    EM_OUT = os.environ.get("EM_OUT", EM_IN + "_out.json")
    req = json.load(open(EM_IN))
    t0 = time.time()
    rec = {"family": req.get("family", "?"),
           "result": "ok", "slots": [], "fitness": 0.0,
           "dem_search_enabled": DEM_SEARCH_ENABLED and DIST_MODE != "cc",
           "dcirc_mode": ("cc_deep%ds" % int(D2S_DEEP)) if DIST_MODE == "cc"
                         else ("dem_search" if DEM_SEARCH_ENABLED else "embedding_w2")}
    try:
        q = int(req["q"])
        n_req = int(req["n"])
        ra, ca = int(req["a_shape"][0]), int(req["a_shape"][1])
        rb, cb = int(req["b_shape"][0]), int(req["b_shape"][1])
        if not (N_MIN <= n_req < N_MAX):
            raise ValueError("n=%d outside [%d, %d)" % (n_req, N_MIN, N_MAX))
        ops = _ops_from_table(req["mul"], q)
        archive = CodeArchive(req["archive"]) if req.get("archive") else None
        rec["n_archive_dup"] = 0
        built = {}                      # slot idx -> build dict
        hashes = {}                     # slot idx -> (h, ikey)
        seen_hash = {}
        seen_iso = {}
        for slot in req["slots"]:
            s = int(slot.get("idx", -1))
            row = {"idx": s}
            rec["slots"].append(row)
            if "error" in slot:
                err = str(slot["error"])[:80]
                if err.startswith("(") and err.endswith(")"):
                    err = "element %s is not in the group's element list (CELL must return elements exactly as custom_ops(q)['elements'] lists them)" % err
                row["status"] = "labeling:" + err
                continue
            try:
                HX, HZ = rs_lp(rs_lift(slot["A"], ops, "left"),
                               rs_lift(slot["B"], ops, "right"), q)
                HX = np.ascontiguousarray(np.asarray(HX, np.uint8) % 2)
                HZ = np.ascontiguousarray(np.asarray(HZ, np.uint8) % 2)
                assert not np.any((HX @ HZ.T.astype(np.int64)) % 2), \
                    "CSS violated"
                n = HX.shape[1]
                if n != n_req or not (N_MIN <= n < N_MAX):
                    row["status"] = "n=%d outside [%d,%d) or != request %d" % (n, N_MIN, N_MAX, n_req)
                    continue
                k = int(n - fast_gf2.rank(HX) - fast_gf2.rank(HZ))
                w = int(max(HX.sum(1).max(), HZ.sum(1).max()))
                row.update(n=n, k=k, w=w)
                if k <= 0:
                    row["status"] = "k<=0"
                    continue
                if w > W_MAX:
                    row["status"] = "w>%d" % W_MAX
                    continue
                if RATE_FLOOR > 0 and k / n < RATE_FLOOR:
                    row["status"] = "k/n<%.2f" % RATE_FLOOR
                    continue
                h = hashlib.sha1(HX.tobytes() + HZ.tobytes()).hexdigest()[:16]
                if h in seen_hash:
                    row["status"] = "dup_of_%d" % seen_hash[h]
                    continue
                seen_hash[h] = s
                # isomorphism-aware dedup on (n, rows, k) plus the Tanner-graph
                # fingerprint: isomorphic slots count once
                ikey = "%d_%d_%d_%d_%s" % (n, HX.shape[0], HZ.shape[0], k,
                                          iso_dedup.iso_key(HX, HZ))
                dup_of = None
                for prev in seen_iso.get(ikey, []):
                    # CM_ISO_EXACT=1 confirms with VF2++ (3-9 s per true
                    # duplicate); default 0 = a fingerprint collision counts
                    # as a duplicate
                    if ISO_EXACT:
                        pb = built[prev]
                        if not iso_dedup.is_isomorphic_exact(
                                HX, HZ, pb["HX"], pb["HZ"]):
                            continue
                    dup_of = prev
                    break
                if dup_of is not None:
                    row["status"] = "iso_dup_of_%d" % dup_of
                    continue
                seen_iso.setdefault(ikey, []).append(s)
                # persist the slot's genotype (cell grids) in the record;
                # the adapter shows it to the mutator when CM_TABLE_CELLS=1
                row["cells"] = {"A": slot.get("Ar", slot["A"]),
                                "B": slot.get("Br", slot["B"])}
                if archive is not None:
                    old = archive.lookup(h, n, HX, HZ, k, ikey, iso_dedup)
                    if old is not None:
                        row["status"] = "dup_archive:%s#%s" % (
                            old.get("family", "?"), old.get("idx", "?"))
                        row["archived_d_circ"] = old.get("d_circ")
                        rec["n_archive_dup"] += 1
                        continue
                LZ = np.asarray(logicals(HX, HZ), np.uint8) % 2
                LX = np.asarray(logicals(HZ, HX), np.uint8) % 2
                built[s] = dict(HX=HX, HZ=HZ, LX=LX, LZ=LZ, n=n, k=k, w=w)
                hashes[s] = (h, ikey)
                row["status"] = "distinct"
                if DIST_MODE == "cc":
                    row["k_bound"] = int(q * (ca - ra) * (cb - rb))
                    row["cyc4_x"] = _cycle4_count(built[s]["HX"])
                    row["cyc4_z"] = _cycle4_count(built[s]["HZ"])
            except Exception as e:
                row["status"] = "build:%s" % str(e)[:80]
        rec["n_distinct"] = len(built)
        if archive is not None:
            # register BEFORE the screen so a concurrent copy on the other
            # worker already sees these codes
            for s, (h, ikey) in hashes.items():
                b = built[s]
                archive.register(h, b["n"], b["HX"], b["HZ"], b["k"], ikey,
                                 dict(family=rec["family"], idx=s, n=b["n"],
                                      k=b["k"], w=b["w"], ts=round(t0, 1)))

        order = sorted(built)
        wit_res = DD.d2s_batch(
            [(built[s]["HX"], built[s]["HZ"], built[s]["LX"],
              built[s]["LZ"]) for s in order], budget=D2S_BUDGET)
        ranked = []
        by_idx = {r["idx"]: r for r in rec["slots"]}
        for s, wres in zip(order, wit_res):
            row = by_idx[s]
            if not isinstance(wres, dict):
                row["status"] = "d2s:" + (str(wres)[:60] if wres else "died")
                continue
            b = built[s]
            row["d2s"] = int(wres["d"])
            row["proxy"] = round(b["k"] * wres["d"] ** 2 / b["n"], 3)
            built[s]["wits"] = wres
            ranked.append((row["proxy"], s))
        ranked.sort(key=lambda t: (-t[0], t[1]))
        chosen = [s for _pr, s in ranked[:TOP_K]]
        rec["measured"] = chosen

        vals = []
        if DIST_MODE == "cc":
            deep = DD.d2s_batch(
                [(built[s]["HX"], built[s]["HZ"], built[s]["LX"],
                  built[s]["LZ"]) for s in chosen], budget=D2S_DEEP)
            for s, wres in zip(chosen, deep):
                b = built[s]
                row = by_idx[s]
                row["dcirc_est"] = "cc_deep"
                if not isinstance(wres, dict):
                    row["status"] = "no_deep_reading"
                    continue
                d = int(min(wres["d"], row["d2s"]))       # both are upper bounds
                row["d_circ"] = d
                try:
                    # all minimum-weight operators the deep search found, per side
                    row["wit_dx"] = wres.get("dx"); row["wit_dz"] = wres.get("dz")
                    row["wit_pool"] = {}
                    for side in ("x", "z"):
                        sups = wres.get("w%s_all" % side) or []
                        if not sups and wres.get("w" + side):
                            sups = [wres["w" + side]]
                        if sups:
                            row["wit_pool"][side] = _pool_witnesses(sups, q, ra, ca, rb, cb)
                            row["wit_pool"][side]["weight"] = int(wres.get("d" + side))
                            # the transmitted list is capped (dem_distance_v2.WIT_ALL_CAP);
                            # the spectrum still counts every verified operator
                            row["wit_pool"][side]["capped"] = len(sups) >= getattr(DD, "WIT_ALL_CAP", 300)
                    row["wit_spectrum"] = wres.get("spectrum")
                except Exception:
                    pass
                row["kd_circ2_n"] = round(b["k"] * d ** 2 / b["n"], 3)
                row["status"] = "measured"
                vals.append(row["kd_circ2_n"])
        else:
            dcs = DD.dcirc_batch(
                [(built[s]["HX"], built[s]["HZ"], built[s]["LX"],
                  built[s]["LZ"]) for s in chosen], q,
                budget=DCIRC_BUDGET if DEM_SEARCH_ENABLED else 0,
                wits_list=[built[s]["wits"] for s in chosen])
            for s, dc in zip(chosen, dcs):
                b = built[s]
                row = by_idx[s]
                row["dcirc_z"] = dc["z"].get("value")
                row["dcirc_x"] = dc["x"].get("value")
                row["dcirc_src"] = {ba: {kk: dc[ba].get(kk) for kk in
                                         ("w2", "emb", "reading")}
                                    for ba in ("z", "x")}
                row["dcirc_est"] = dc.get("est_status")
                if dc["d_circ"] is None:
                    row["status"] = "no_witnessed_dcirc"
                    continue
                row["d_circ"] = int(dc["d_circ"])
                row["kd_circ2_n"] = round(
                    b["k"] * dc["d_circ"] ** 2 / b["n"], 3)
                row["status"] = "measured"
                vals.append(row["kd_circ2_n"])
        rec["n_measured"] = len(vals)
        rated = [built[s] for s in chosen
                 if by_idx[s].get("status") == "measured"] or list(built.values())
        if rated:
            rec["rate"] = round(sum(b["k"] / b["n"] for b in rated) / len(rated), 4)
            rec["rate_bin"] = rate_bin_of(rec["rate"])
        rec["n_no_reading"] = sum(1 for s in chosen
                                  if str(by_idx[s].get("dcirc_est", "")).startswith(("timeout", "error")))
        if archive is not None:
            for s, (h, _ik) in hashes.items():
                row = by_idx[s]
                if str(row.get("status", "")).startswith("d2s:"):
                    # infrastructure failure, not a measurement: let a
                    # retry measure this code instead of calling it a dup
                    archive.unregister(h, built[s]["n"], built[s]["HX"], built[s]["HZ"], built[s]["k"], _ik)
                    continue
                archive.update(h, d2s=row.get("d2s"), d_circ=row.get("d_circ"),
                               status=row.get("status"))
        # fixed denominator: fewer than TOP_K measured slots count as zeros,
        # so a single good code cannot collect a full five-code score
        rec["fitness"] = round(sum(vals) / TOP_K, 4) if vals else 0.0
    except Exception as e:
        rec["result"] = "error:%s:%s" % (type(e).__name__, str(e)[:120])
        rec["fitness"] = 0.0
    rec["secs"] = round(time.time() - t0, 1)
    tmp = EM_OUT + ".tmp"
    json.dump(rec, open(tmp, "w"), indent=1)
    os.replace(tmp, EM_OUT)
    # full record, including the per-slot table, also goes to the worker log
    # (the result file is consumed and deleted by the controller-side evaluator)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()

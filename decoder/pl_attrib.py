"""Block-level failure attribution from the same shots that produce the pL score.

Shots are drawn with stim's FlipSimulator; per chunk, the final data-qubit
Pauli frame and the flipped check measurements are summed per protograph block,
over failed shots and over all shots. The report per block is the enrichment =
share of the failed-shot error mass / share of the all-shot error mass.

Block labels (in A/B terms, independent of the basis):
  Acolj/Bcolb        data block of sector 1: A-column j with B-column b (ca*cb blocks)
  Arowi/Browa        data block of sector 2: A-row i with B-row a       (ra*rb blocks)
  Xchk:Arowi/Bcolb   X-check block: A-row i with B-column b             (ra*cb blocks)
  Zchk:Acolj/Browa   Z-check block: A-column j with B-row a             (ca*rb blocks)
"""
import numpy as np


def _block_maps(n, q, shapes, groups):
    """qubit index -> block label. groups = [(kind, rows), ...] in the order the
    ancilla groups follow the data qubits in the circuit; kind "X" labels a
    block by (A-row i, B-column b), kind "Z" by (A-column j, B-row a)."""
    ra, ca, rb, cb = shapes
    n1 = ca * cb * q
    assert n == n1 + ra * rb * q, "n=%d does not match shapes %s and q=%d" % (n, shapes, q)
    lab = {}
    for c in range(n):
        if c < n1:
            b = c // q
            lab[c] = "Acol%d/Bcol%d" % (b // cb, b % cb)
        else:
            b = (c - n1) // q
            lab[c] = "Arow%d/Brow%d" % (b // rb, b % rb)
    base = n
    for kind, rows in groups:
        if kind == "X":
            assert rows == ra * cb * q, "X-check rows %d vs ra*cb*q=%d" % (rows, ra * cb * q)
        else:
            assert rows == ca * rb * q, "Z-check rows %d vs ca*rb*q=%d" % (rows, ca * rb * q)
        for r in range(rows):
            b = r // q
            lab[base + r] = (("Xchk:Arow%d/Bcol%d" % (b // cb, b % cb)) if kind == "X"
                             else ("Zchk:Acol%d/Brow%d" % (b // rb, b % rb)))
        base += rows
    return lab


def measurement_qubits(circ):
    """qubit measured by each measurement record index, in circuit order."""
    out = []
    for inst in circ.flattened():
        if inst.name in ("M", "MZ", "MX", "MY", "MR", "MRZ", "MRX", "MRY"):
            for t in inst.targets_copy():
                out.append(int(t.value))
        elif inst.name in ("MPP",):
            raise ValueError("MPP measurements are not mapped to a single qubit")
    assert len(out) == circ.num_measurements, "measurement map %d vs %d" % (len(out), circ.num_measurements)
    return out


class FrameAttrib(object):
    """Per-unit accumulator. per_shot(sim) turns a FlipSimulator batch into
    per-block counts; add(extra, failed_idx) accumulates; summary() reports."""

    def __init__(self, circ, n, q, shapes, basis, nax, naz):
        groups = [("X", nax), ("Z", naz)] if basis == "Z" else [("Z", nax), ("X", naz)]
        lab = _block_maps(n, q, shapes, groups)
        self.n = n
        # Both bases are built as Z-type memories (the X basis swaps HX and HZ
        # before building), so the errors that flip the measured observables
        # are the X frame component in both.
        self.part = "x"
        self.data_labels = sorted({lab[c] for c in range(n)}, key=lambda s: (0 if s.startswith("Acol") else 1, s))
        di = {l: i for i, l in enumerate(self.data_labels)}
        self.data_ind = np.zeros((n, len(self.data_labels)), np.float32)
        for c in range(n):
            self.data_ind[c, di[lab[c]]] = 1.0
        mq = measurement_qubits(circ)
        self.check_meas = np.array([i for i, qb in enumerate(mq) if qb >= n], np.int64)
        self.check_labels = sorted({lab[mq[i]] for i in self.check_meas})
        ci = {l: i for i, l in enumerate(self.check_labels)}
        self.check_ind = np.zeros((len(self.check_meas), len(self.check_labels)), np.float32)
        for row, i in enumerate(self.check_meas):
            self.check_ind[row, ci[lab[mq[i]]]] = 1.0
        self.fb_fail = np.zeros(len(self.data_labels)); self.fb_all = np.zeros(len(self.data_labels))
        self.mb_fail = np.zeros(len(self.check_labels)); self.mb_all = np.zeros(len(self.check_labels))
        self.fw_fail = 0.0; self.fw_all = 0.0
        self.n_fail = 0; self.n_all = 0

    def per_shot(self, sim, m):
        """(frame block counts (m x nD), check-flip block counts (m x nC),
        frame weight (m)) for one FlipSimulator batch of m shots."""
        xs, zs = sim.to_numpy(transpose=True, output_xs=True, output_zs=True)[:2]
        fr = (xs if self.part == "x" else zs)[:m, :self.n].astype(np.float32)
        mf = sim.get_measurement_flips().T[:m][:, self.check_meas].astype(np.float32)
        return fr @ self.data_ind, mf @ self.check_ind, fr.sum(axis=1)

    def add(self, extra, failed_idx):
        fb, mb, fw = extra
        m = fb.shape[0]
        self.fb_all += fb.sum(axis=0); self.mb_all += mb.sum(axis=0); self.fw_all += float(fw.sum())
        self.n_all += m
        if failed_idx:
            idx = np.asarray(failed_idx, np.int64)
            self.fb_fail += fb[idx].sum(axis=0); self.mb_fail += mb[idx].sum(axis=0)
            self.fw_fail += float(fw[idx].sum()); self.n_fail += len(idx)

    @staticmethod
    def _enrich(labels, fail, alls):
        tf, ta = fail.sum(), alls.sum()
        rows = []
        for i, l in enumerate(labels):
            sf = (fail[i] / tf) if tf else 0.0
            sa = (alls[i] / ta) if ta else 0.0
            rows.append([l, round(sf / sa, 2) if sa else None, round(sf, 3)])
        return rows

    def summary(self):
        return self._summary(self.data_labels, self.check_labels, self.fb_fail, self.fb_all,
                             self.mb_fail, self.mb_all, self.fw_fail, self.fw_all,
                             self.n_fail, self.n_all)

    @classmethod
    def _summary(cls, dl, cl, fb_fail, fb_all, mb_fail, mb_all, fw_fail, fw_all, n_fail, n_all):
        return dict(shots=int(n_all), failed=int(n_fail),
                    mean_error_weight_failed=round(fw_fail / n_fail, 2) if n_fail else None,
                    mean_error_weight_all=round(fw_all / n_all, 2) if n_all else None,
                    data_blocks=cls._enrich(dl, fb_fail, fb_all),
                    check_blocks=cls._enrich(cl, mb_fail, mb_all),
                    raw=dict(data_labels=list(dl), check_labels=list(cl),
                             fb_fail=[float(x) for x in fb_fail], fb_all=[float(x) for x in fb_all],
                             mb_fail=[float(x) for x in mb_fail], mb_all=[float(x) for x in mb_all],
                             fw_fail=float(fw_fail), fw_all=float(fw_all),
                             n_fail=int(n_fail), n_all=int(n_all)))

    @classmethod
    def merge_summaries(cls, summaries):
        """Sum the raw counts of several shards' summaries and recompute the enrichments."""
        raws = [s["raw"] for s in summaries]
        dl, cl = raws[0]["data_labels"], raws[0]["check_labels"]
        for r in raws[1:]:
            if r["data_labels"] != dl or r["check_labels"] != cl:
                raise ValueError("attribution summaries with different block labels cannot be merged")
        tot = lambda key: np.sum([np.asarray(r[key], float) for r in raws], axis=0)
        return cls._summary(dl, cl, tot("fb_fail"), tot("fb_all"), tot("mb_fail"), tot("mb_all"),
                            sum(r["fw_fail"] for r in raws), sum(r["fw_all"] for r in raws),
                            sum(r["n_fail"] for r in raws), sum(r["n_all"] for r in raws))


def render(a):
    """One feedback line."""
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

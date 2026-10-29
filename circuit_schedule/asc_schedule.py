"""Depth-optimal syndrome extraction schedules (ASC, Auto-Stabilizer-Check) for
lifted-product codes, at the level of layer units.

Each q x q block joining a check block to a data block is split into perfect
matchings (layer units); all q CNOTs of a unit share one tick. solve_schedule
assigns ticks with CP-SAT (or a complete backtracking fallback) so that units
sharing a check block or a data block get different ticks and every X/Z parity
condition of parity_conditions holds, searching the depth upward from the
degree bound. The returned depth is minimal among schedules of this unit form.

build_circuit writes the interleaved memory circuit with the same detectors,
CX orientations, reset/measure placement and observables as
circ_1side.build_circuit. Idling noise is off unless idle_p is given.
"""
import itertools
import os
import sys

import numpy as np
import stim

# --------------------------------------------------------------------------
# units: one layer (a perfect matching of a block), all q gates in one TICK
# --------------------------------------------------------------------------


class Unit(object):
    __slots__ = ("kind", "rb", "cb", "mt")

    def __init__(self, kind, rb, cb, mt):
        self.kind = kind          # "x" or "z"
        self.rb = rb              # check block index within its side
        self.cb = cb              # data block index
        self.mt = mt              # mt[i] = data offset touched by ancilla i

    def __repr__(self):
        return "Unit(%s,rb=%d,cb=%d)" % (self.kind, self.rb, self.cb)


def perfect_matchings(sub):
    """Split a q x q 0-1 block of constant row/column weight a into a matchings.

    Same augmenting-path search as circ_1side.perfect_matchings; the matchings
    are not in general the permutation matrices of the cell's group elements.
    """
    q = sub.shape[0]
    rem = sub.copy()
    out = []
    while rem.sum():
        match_r = [-1] * q
        match_c = [-1] * q
        adj = [np.flatnonzero(rem[r]).tolist() for r in range(q)]
        for r in range(q):
            seen = [False] * q
            stack = [(r, iter(adj[r]))]
            path = []
            while stack:
                rr, it = stack[-1]
                adv = False
                for cc in it:
                    if seen[cc]:
                        continue
                    seen[cc] = True
                    path.append((rr, cc))
                    if match_c[cc] == -1:
                        for (a, b) in path:
                            match_r[a] = b
                            match_c[b] = a
                        stack = []
                        adv = True
                        break
                    stack.append((match_c[cc], iter(adj[match_c[cc]])))
                    adv = True
                    break
                if not adv:
                    stack.pop()
                    if path:
                        path.pop()
        if any(m == -1 for m in match_r):
            raise AssertionError("block is not regular: no perfect matching")
        out.append(list(match_r))
        for r, c in enumerate(match_r):
            rem[r, c] = 0
            adj[r].remove(c)
    return out


def build_units(HX, HZ, q):
    """All layer units (perfect matchings of the blocks) of both sides, as one list."""
    units = []
    for kind, H in (("x", HX), ("z", HZ)):
        m, n = H.shape
        nrb, ncb = m // q, n // q
        assert m % q == 0 and n % q == 0, "matrix is not a q-lift"
        for rb in range(nrb):
            for cb in range(ncb):
                sub = H[rb * q:(rb + 1) * q, cb * q:(cb + 1) * q]
                if not sub.any():
                    continue
                for mt in perfect_matchings(sub):
                    units.append(Unit(kind, rb, cb, mt))
    return units


def block_members(units):
    """check-block -> unit indices, data-block -> unit indices."""
    checks, datas = {}, {}
    for i, u in enumerate(units):
        checks.setdefault((u.kind, u.rb), []).append(i)
        datas.setdefault(u.cb, []).append(i)
    return checks, datas


def degree_bound(units):
    """Lower bound on the depth = max block degree of the multigraph.

    Equals Delta(G) of the Tanner graph: a check block's degree is that check's row
    weight, a data block's degree is that data qubit's total (X+Z) degree.
    """
    checks, datas = block_members(units)
    best = 0
    for v in itertools.chain(checks.values(), datas.values()):
        best = max(best, len(v))
    return best


# --------------------------------------------------------------------------
# X/Z parity conditions on the unit graph
# --------------------------------------------------------------------------


def parity_conditions(units, q):
    """Group X/Z unit pairs into even-parity conditions.

    Returns (conditions, regular) where conditions is a list of lists of (iu, iv)
    index pairs; each list must contain an EVEN number of pairs with t_iu < t_iv.
    """
    xs = [i for i, u in enumerate(units) if u.kind == "x"]
    zs = [i for i, u in enumerate(units) if u.kind == "z"]
    inv_cache = {}

    def inv(iv):
        r = inv_cache.get(iv)
        if r is None:
            mt = units[iv].mt
            r = [0] * q
            for b, l in enumerate(mt):
                r[l] = b
            inv_cache[iv] = r
        return r

    groups = {}
    for iu in xs:
        u = units[iu]
        for iv in zs:
            v = units[iv]
            if v.cb != u.cb:
                continue
            vi = inv(iv)
            sigma = tuple(vi[l] for l in u.mt)      # X-ancilla a -> Z-ancilla b
            groups.setdefault((u.rb, v.rb, sigma), []).append((iu, iv))

    # one condition per sigma class is valid only if, within one (X block, Z block),
    # distinct sigmas agree nowhere; otherwise use one condition per realised (a, b)
    by_blocks = {}
    for (rbx, rbz, sigma) in groups:
        by_blocks.setdefault((rbx, rbz), []).append(sigma)
    regular = True
    for sigs in by_blocks.values():
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                if any(a == b for a, b in zip(sigs[i], sigs[j])):
                    regular = False
                    break
            if not regular:
                break
        if not regular:
            break

    if regular:
        return list(groups.values()), True

    # fallback: one condition per (X block, Z block, a, b) actually realised
    fine = {}
    for (rbx, rbz, sigma), pairs in groups.items():
        for a, b in enumerate(sigma):
            fine.setdefault((rbx, rbz, a, b), []).extend(pairs)
    return list(fine.values()), False


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------


def _solve_cpsat(units, conds, T, time_limit, seed, workers):
    from ortools.sat.python import cp_model

    checks, datas = block_members(units)
    model = cp_model.CpModel()
    t = [model.NewIntVar(0, T - 1, "t%d" % i) for i in range(len(units))]

    for idxs in itertools.chain(checks.values(), datas.values()):
        if len(idxs) > 1:
            model.AddAllDifferent([t[i] for i in idxs])

    for c, pairs in enumerate(conds):
        if not pairs:
            continue
        lits = []
        for (iu, iv) in pairs:
            # iu and iv share a data block, so AllDifferent already forces
            # t[iu] != t[iv]; the negated branch is therefore a strict >.
            b = model.NewBoolVar("b%d_%d_%d" % (c, iu, iv))
            model.Add(t[iu] < t[iv]).OnlyEnforceIf(b)
            model.Add(t[iu] > t[iv]).OnlyEnforceIf(b.Not())
            lits.append(b)
        if len(lits) == 1:
            model.Add(lits[0] == 0)
        else:
            half = model.NewIntVar(0, len(lits) // 2, "h%d" % c)
            model.Add(sum(lits) == 2 * half)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.random_seed = int(seed)
    solver.parameters.num_search_workers = int(workers)
    st = solver.Solve(model)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [solver.Value(x) for x in t], "sat"
    if st == cp_model.INFEASIBLE:
        return None, "unsat"
    return None, "timeout"


def _solve_dfs(units, conds, T, time_limit, seed):
    """Complete backtracking fallback, used when ortools is not installed.

    Complete: an exhausted search is a genuine UNSAT proof for this depth. Returns
    "timeout" only if the wall clock runs out first.
    """
    import random
    import time

    checks, datas = block_members(units)
    conflict = [set() for _ in units]
    for idxs in itertools.chain(checks.values(), datas.values()):
        for i in idxs:
            for j in idxs:
                if i != j:
                    conflict[i].add(j)

    cond_of = [[] for _ in units]
    for c, pairs in enumerate(conds):
        for (iu, iv) in pairs:
            cond_of[iu].append(c)
            cond_of[iv].append(c)
    cond_of = [sorted(set(v)) for v in cond_of]

    order = sorted(range(len(units)), key=lambda i: -len(conflict[i]))
    rng = random.Random(seed)
    t = [-1] * len(units)
    deadline = time.time() + time_limit
    hit_deadline = [False]

    def cond_ok(c, final):
        odd = 0
        for (iu, iv) in conds[c]:
            if t[iu] < 0 or t[iv] < 0:
                if final:
                    return True          # not yet decidable
                return True
            if t[iu] < t[iv]:
                odd ^= 1
        if not final:
            return True
        return odd == 0

    def all_assigned(c):
        return all(t[iu] >= 0 and t[iv] >= 0 for (iu, iv) in conds[c])

    def rec(pos):
        if time.time() > deadline:
            hit_deadline[0] = True
            return False
        if pos == len(order):
            return True
        i = order[pos]
        used = {t[j] for j in conflict[i] if t[j] >= 0}
        cand = [v for v in range(T) if v not in used]
        rng.shuffle(cand)
        for v in cand:
            t[i] = v
            ok = True
            for c in cond_of[i]:
                if all_assigned(c) and not cond_ok(c, True):
                    ok = False
                    break
            if ok and rec(pos + 1):
                return True
            t[i] = -1
        return False

    if rec(0):
        return list(t), "sat"
    return None, ("timeout" if hit_deadline[0] else "unsat")


def solve_schedule(HX, HZ, q, t_max=None, time_limit=120.0, seed=0, workers=8,
                   backend=None, verbose=True):
    """Search depth upward from the degree bound; return the first feasible schedule.

    Returns dict with units, ticks, depth, lower_bound, backend, regular, status.
    """
    units = build_units(HX, HZ, q)
    conds, regular = parity_conditions(units, q)
    lb = degree_bound(units)
    hi = t_max if t_max is not None else 2 * lb

    if backend is None:
        try:
            import ortools.sat.python.cp_model  # noqa: F401
            backend = "cpsat"
        except ImportError:
            backend = "dfs"

    if verbose:
        npairs = sum(len(c) for c in conds)
        print("units=%d (x=%d z=%d)  parity conditions=%d over %d pairs  "
              "regular=%s  Delta=%d  backend=%s"
              % (len(units), sum(1 for u in units if u.kind == "x"),
                 sum(1 for u in units if u.kind == "z"),
                 len(conds), npairs, regular, lb, backend), flush=True)

    for T in range(lb, hi + 1):
        if backend == "cpsat":
            ticks, st = _solve_cpsat(units, conds, T, time_limit, seed, workers)
        else:
            ticks, st = _solve_dfs(units, conds, T, time_limit, seed)
        if verbose:
            print("  depth %2d -> %s" % (T, st), flush=True)
        if st == "sat":
            return {"units": units, "ticks": ticks, "depth": T, "lower_bound": lb,
                    "backend": backend, "regular": regular, "status": "sat",
                    "conditions": conds}
        if st == "timeout":
            return {"units": units, "ticks": None, "depth": None, "lower_bound": lb,
                    "backend": backend, "regular": regular, "status": "timeout",
                    "conditions": conds}
    return {"units": units, "ticks": None, "depth": None, "lower_bound": lb,
            "backend": backend, "regular": regular, "status": "unsat",
            "conditions": conds}


def greedy_separated(HX, HZ, q):
    """The circ_1side schedule, for comparison: X fully before Z, each side
    coloured greedily. Returns (ticks, depth) with Z ticks offset after X."""
    units = build_units(HX, HZ, q)
    depth = {}
    ticks = [0] * len(units)
    for kind in ("x", "z"):
        busy_rb, busy_cb = {}, {}
        top = 0
        for i, u in enumerate(units):
            if u.kind != kind:
                continue
            t = 0
            while t in busy_rb.get(u.rb, ()) or t in busy_cb.get(u.cb, ()):
                t += 1
            busy_rb.setdefault(u.rb, set()).add(t)
            busy_cb.setdefault(u.cb, set()).add(t)
            ticks[i] = t
            top = max(top, t + 1)
        depth[kind] = top
    for i, u in enumerate(units):
        if u.kind == "z":
            ticks[i] += depth["x"]
    return units, ticks, depth["x"] + depth["z"], depth["x"], depth["z"]


# --------------------------------------------------------------------------
# circuit
# --------------------------------------------------------------------------


def build_circuit(HX, HZ, LZ, rounds, p, q, units, ticks, depth, idle_p=None):
    """Interleaved memory circuit from a unit schedule.

    Detector convention, CX orientations, reset/measure placement and observables are
    identical to circ_1side.build_circuit; only the gate ORDER changes. Z-memory:
    both check types are extracted (the X gates are where most hook errors come from
    and must stay) but only the Z checks carry detectors.
    """
    n = HX.shape[1]
    nax, naz = HX.shape[0], HZ.shape[0]
    ax = [n + i for i in range(nax)]
    az = [n + nax + i for i in range(naz)]
    allq = list(range(n + nax + naz))

    by_tick = [[] for _ in range(depth)]
    for i, u in enumerate(units):
        by_tick[ticks[i]].append(i)

    c = stim.Circuit()
    rec, tot = {}, [0]

    for r in range(rounds):
        c.append("R", ax + az)
        c.append("DEPOLARIZE1", ax + az, p)
        c.append("H", ax)
        for t in range(depth):
            pairs = []
            touched = set()
            for i in by_tick[t]:
                u = units[i]
                for j in range(q):
                    d = u.cb * q + u.mt[j]
                    if u.kind == "x":
                        a = ax[u.rb * q + j]
                        pairs += [a, d]
                    else:
                        a = az[u.rb * q + j]
                        pairs += [d, a]
                    assert d not in touched and a not in touched, \
                        "tick %d double-books a qubit -- schedule is invalid" % t
                    touched.add(d)
                    touched.add(a)
            if pairs:
                c.append("CX", pairs)
                c.append("DEPOLARIZE2", pairs, p)
            if idle_p:
                idle = [x for x in allq if x not in touched]
                if idle:
                    c.append("DEPOLARIZE1", idle, idle_p)
        c.append("H", ax)
        c.append("DEPOLARIZE1", ax + az, p)
        c.append("M", ax)
        for i in range(nax):
            rec[(r, "x", i)] = tot[0] + i
        tot[0] += nax
        c.append("M", az)
        for i in range(naz):
            rec[(r, "z", i)] = tot[0] + i
        tot[0] += naz
        for i in range(naz):
            if r == 0:
                c.append("DETECTOR", [stim.target_rec(rec[(0, "z", i)] - tot[0])])
            else:
                c.append("DETECTOR",
                         [stim.target_rec(rec[(r, "z", i)] - tot[0]),
                          stim.target_rec(rec[(r - 1, "z", i)] - tot[0])])

    c.append("DEPOLARIZE1", list(range(n)), p)
    c.append("M", list(range(n)))
    dstart = tot[0]
    tot[0] += n
    for j in range(naz):
        tt = [stim.target_rec(dstart + int(qq) - tot[0])
              for qq in np.flatnonzero(HZ[j]).tolist()]
        tt.append(stim.target_rec(rec[(rounds - 1, "z", j)] - tot[0]))
        c.append("DETECTOR", tt)
    for i, lz in enumerate(LZ):
        c.append("OBSERVABLE_INCLUDE",
                 [stim.target_rec(dstart + int(qq) - tot[0])
                  for qq in np.flatnonzero(lz).tolist()], i)
    return c


def detectors_deterministic(HX, HZ, LZ, q, units, ticks, depth, rounds=2, shots=64):
    """Noiseless build must fire no detector; a violated parity condition makes
    the circuit measure something other than the intended stabilisers."""
    c = build_circuit(HX, HZ, LZ, rounds, 0.0, q, units, ticks, depth)
    s = c.compile_detector_sampler().sample(shots=shots)
    return not bool(s.any())

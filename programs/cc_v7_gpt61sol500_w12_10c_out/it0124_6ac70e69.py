"""Initial program: lifted-product CSS code, ONE (group, shape) bed per
program, N_SLOTS (env CM_NSLOTS) freely-filled candidates per generation.

SEARCH SPACE (the only constraints; everything else is free):
- The group and the protograph shapes are fixed for the whole program
  (genotype, editable between generations). Each of the 20 slots fills
  the cells however it wants: PER SLOT, every cell holds 0..5 DISTINCT
  group elements (an empty cell is a zero block) - term counts AND
  elements are both the slot's choice. Identical fillings are
  deduplicated downstream and waste the slot.
- Size law: n = (ca*cb + ra*rb) * q for A of shape (ra, ca), B of shape
  (rb, cb), |G| = q. n must land in [CM_NMIN, CM_NMAX) (default
  [100, 600)); the scaffold picks the largest valid q.
- PER-SLOT validity (scaffold-checked; a violating slot is voided, not
  repaired): max row weight <= CM_WMAX (an HX row sums A-row i with
  B-column j; an HZ row sums a B-row with an A-column - check BOTH),
  no all-zero protograph row or column on either side (a zero A-row
  leaves its sector-2 block untouched by every Z check, which is a
  weight-1 logical - d=1 - unless the matching B column is a single
  1-term cell; symmetric for the other three cases), cells distinct,
  sizes 0..5.
- k > 0 required; k is measured on the built matrices per slot.

GROUPS: GTYPE "cyclic" (Z_q), "abelian" (Z_A0 x Z_{q/A0}) or "custom".
A custom group is a code string defining valid_q(q) -> bool and
custom_ops(q) -> dict(elements, ident, mul, inv). Any finite group you
can write down qualifies; the scaffold validates identity, inverses and
associativity on samples before building.
"""
import json
import os
import random

N_MIN = int(os.environ.get("CM_NMIN", "100"))
N_MAX = int(os.environ.get("CM_NMAX", "600"))
W_MAX = int(os.environ.get("CM_WMAX", "12"))
CELL_MAX = 5
N_SLOTS = int(os.environ.get("CM_NSLOTS", "20"))
BASE_SEED = 20260903


# EVOLVE-BLOCK-START
CONCEPT = ("Dic12, order 48, shapes (1,2)x(2,5): normal-subgroup kernel screening followed by independent component optimization should exceed five-code fitness 38.25 and test whether the measured weight-12 plateau can be removed.")
GROUP = ("Frozen under L1. The current Dic12 bed measures five k=153, d30s=12 codes. Its normal cyclic subgroups of orders 2,3,4,6,8,12,24 provide structured witness subspaces without changing the group or assuming that Dic11's distance transfers.")
SHAPES = ("Frozen (1,2)x(2,5) for cell 0.250: n=576, kb=144, guaranteed rate 1/4; the size window permits q<=49 and this group uses q=48. Component variable/check counts are 96/48 and 240/96. Parent minima were entirely sector-one Acol/Bcol operators, broadly involving both A columns and several B columns. Verified ranks retain [0.225,0.275).")
FILLING = ("Twenty support/occupancy cards use four-term A cells and degree-three/four B columns, with both CSS row caps <=12. Within each cell require distinct images modulo the order-three and order-four normal subgroups; vary outer-coset occupancy and same-coset binomial orders. Search A before B. Verified component syndromes supply normal-subgroup norm columns; enumerate small restricted kernels and probe larger ones. Independently rescreen finalists, retain every lower reading, and rank conservatively by k times the squared limiting witness reading. Restricted exact searches and randomized readings do not certify quantum distance.")
FAMILY_NAME = "r025_Dic12_12x25_joint_logical_v1_norm_rescreen_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    # Dic12: (a,b)=x^a y^b, x^24=1, y^2=x^12.
    if not valid_q(q):
        raise ValueError("Dic12 requires order 48")
    els = [(a,b) for b in range(2) for a in range(24)]

    def red(g):
        a,b = g
        return ((a + 12*(b//2)) % 24, b % 2)

    def mul(g, h):
        a,b = red(g)
        c,d = red(h)
        return ((a + (c if b == 0 else -c) + 12*b*d) % 24,
                (b+d) % 2)

    def inv(g):
        a,b = red(g)
        return ((-a) % 24, 0) if b == 0 else ((a+12) % 24, 1)

    def idx(g):
        a,b = red(g)
        return 24*b+a

    return dict(elements=els, ident=(0,0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)

# Each card is (A cell sizes, B column sizes).
# First three: HX<=12, HZ<=12. Fourth: HX<=11, HZ<=12.
# Fifth: HX<=10, HZ<=12. Every row and column is nonzero.
_DIC_SUPPORTS = (
    ((4,4), ((2,1), (2,1), (1,2), (1,2), (2,2))),
    ((4,4), ((3,0), (0,3), (2,1), (1,2), (2,2))),
    ((4,4), ((3,1), (1,3), (1,2), (2,1), (1,1))),
    ((3,5), ((2,1), (2,1), (1,2), (1,2), (1,1))),
    ((3,3), ((2,2), (2,2), (2,2), (2,1), (1,2))),
)


def _dic_cell(rng, G, pools, count, outer, long_binomial):
    """Prescribed coset occupancy; exclude the unique involution difference."""
    from math import gcd
    if count == 0:
        return []
    quota = [0]*(count-outer) + [1]*outer
    for _ in range(64):
        rng.shuffle(quota)
        cell = []
        for coset in quota:
            available = []
            for g in pools[coset]:
                legal = True
                for h in cell:
                    relative = G.mul(G.inv(h), g)
                    if relative == G.e or relative == (12,0):
                        legal = False
                        break
                    if long_binomial and count == 2 and relative[1] == 0:
                        if 24 // gcd(relative[0], 24) < 8:
                            legal = False
                            break
                if legal:
                    available.append(g)
            if not available:
                break
            cell.append(rng.choice(available))
        if len(cell) == count:
            return cell
    raise ValueError("dicyclic cell rejection exhausted")


def _dic_grid(rng, G, pools, counts, mode, side, phase):
    grid = []
    for i, row in enumerate(counts):
        out = []
        for j, count in enumerate(row):
            if count <= 1:
                outer = count*((phase+i+j) % 2)
            elif side == "A":
                if mode == 0:
                    outer = count//2 + (count % 2)*(j % 2)
                elif mode == 1:
                    outer = 1
                elif mode == 2:
                    outer = 1 if j == 0 else count-1
                else:
                    outer = count//2 if j == 0 else count-1
            elif mode == 0:
                outer = (count*((i+j+phase) % 2) if count == 2
                         else count//2 + (count % 2)*((i+j+phase) % 2))
            elif mode == 1:
                outer = 1
            elif mode == 2:
                outer = 0 if count == 2 else count-1
            else:
                outer = (count*((i+j) % 2) if count == 2
                         else rng.randrange(count+1))
            out.append(_dic_cell(rng, G, pools, count, outer,
                                 side == "B" and mode in (0,2,3)))
        grid.append(out)
    return grid


def _extract_components(hx, hz, q):
    """Extract both component maps from the scaffold's actual lifts."""
    ca, cb, rb = A_SHAPE[1], B_SHAPE[1], B_SHAPE[0]
    mask = (1 << q) - 1
    ar = [sum(((row >> (j * cb * q)) & mask) << (j * q)
              for j in range(ca))
          for row in hx[:q]]
    br = [row & ((1 << (cb * q)) - 1) for row in hz[:rb * q]]
    return ar, br


def _short_kernel(cols, through_six=False):
    """Exact component screening through four, optionally through six.

    Sentinel 5/7 means no witness through four/six, not a distance
    estimate. After excluding weights <=4, triple collisions are
    disjoint and each weight-six word has ten triple partitions.
    """
    if 0 in cols:
        return (1, 1)
    if len(set(cols)) != len(cols):
        return (2, 1)
    low = {s: 1 for s in cols}
    for i, x in enumerate(cols):
        for j in range(i):
            s = x ^ cols[j]
            previous = low.get(s, 0)
            if previous:
                return (3 if previous == 1 else 4, 1)
            low[s] = 2
    if not through_six:
        return (5, 0)

    seen = {}
    hits = 0
    for i in range(2, len(cols)):
        x = cols[i]
        for j in range(1, i):
            xy = x ^ cols[j]
            for h in range(j):
                s = xy ^ cols[h]
                if not s:
                    return (3, 1)
                if s in low:
                    return (5, 1)
                previous = seen.get(s, 0)
                hits += previous
                seen[s] = previous + 1
    return (6, hits // 10) if hits else (7, 0)


def _kernel_probe(cols, rng, trials):
    """Randomized classical kernel witnesses: upper bounds only."""
    width = len(cols)
    best = width + 1
    for _ in range(trials):
        order = list(range(width))
        rng.shuffle(order)
        pivots = {}
        deps = []
        for c in order:
            s, v = cols[c], 1 << c
            while s:
                p = s.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (s, v)
                    break
                s ^= old[0]
                v ^= old[1]
            if not s:
                deps.append(v)
                best = min(best, v.bit_count())

        for _ in range(2):
            changed = False
            for a in range(len(deps)):
                v = deps[a]
                weight = v.bit_count()
                for b in range(len(deps)):
                    if a == b:
                        continue
                    z = v ^ deps[b]
                    zw = z.bit_count()
                    if 0 < zw < weight:
                        v, weight = z, zw
                        changed = True
                deps[a] = v
                best = min(best, weight)
            if not changed:
                break
    return best


def _component_columns(rows, width):
    """Column syndromes of an extracted, verified lifted component."""
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _component_four_cycles(cols):
    seen = {}
    cycles = 0
    for syndrome in cols:
        checks = []
        while syndrome:
            bit = syndrome & -syndrome
            checks.append(bit.bit_length() - 1)
            syndrome ^= bit
        for a in range(len(checks)):
            for b in range(a):
                pair = (checks[b], checks[a])
                cycles += seen.get(pair, 0)
                seen[pair] = seen.get(pair, 0) + 1
    return cycles


def _small_logical_4(checks, stabilizers, width):
    """Exact quantum-logical existence test through weight four.

    Equal syndromes of supports of size <=2 cover every commuting
    support through four. One representative suffices: if a collision
    is a stabilizer, it preserves the quotient class of that syndrome.
    """
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old

    def outside(v):
        while v:
            p = v.bit_length() - 1
            old = basis.get(p)
            if old is None:
                return True
            v ^= old
        return False

    cols = _component_columns(checks, width)
    seen = {0: 0}
    bits = [1 << i for i in range(width)]
    for i, s in enumerate(cols):
        old = seen.get(s)
        if old is None:
            seen[s] = bits[i]
        elif outside(bits[i] ^ old):
            return True

    for i, x in enumerate(cols):
        for j in range(i):
            s = x ^ cols[j]
            v = bits[i] | bits[j]
            old = seen.get(s)
            if old is None:
                seen[s] = v
            elif outside(v ^ old):
                return True
    return False


def _quantum_probe(checks, stabilizers, width, rng, trials):
    """Find genuine logical witnesses, including both sectors.

    Kernel dependencies are classified modulo the exact stabilizer span.
    Dependency sums and addition of check rows preserve commutation.
    A high reading is only an upper bound, never a distance certificate.
    """
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length()-1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old
    ordered = sorted(basis.items(), reverse=True)

    def logical_class(word):
        for p, row in ordered:
            if (word >> p) & 1:
                word ^= row
        return word

    cols = _component_columns(checks, width)
    order = list(range(width))
    best = width+1
    for _ in range(trials):
        rng.shuffle(order)
        pivots, deps = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                cls = logical_class(word)
                deps.append((word, cls))
                if cls:
                    best = min(best, word.bit_count())

        short = sorted(deps, key=lambda x: x[0].bit_count())[:64]
        seeds = []
        for word, cls in short:
            if cls:
                seeds.append((word, cls))
        for i, (word, cls) in enumerate(short):
            for other, oc in short[:i]:
                if cls ^ oc:
                    v = word ^ other
                    best = min(best, v.bit_count())
                    seeds.append((v, cls ^ oc))
        seeds.sort(key=lambda x: x[0].bit_count())

        # Sparse stabilizer rows explicitly allow cross-sector reduction.
        moves = short + [(row, 0) for row in stabilizers]
        for word, cls in seeds[:4]:
            weight = word.bit_count()
            for sweep in range(3):
                changed = False
                rng.shuffle(moves)
                for other, oc in moves:
                    trial, tc = word ^ other, cls ^ oc
                    tw = trial.bit_count()
                    if tc and tw < weight:
                        word, cls, weight = trial, tc, tw
                        changed = True
                if not changed:
                    break
            best = min(best, weight)
        if best <= 4:
            break
    return best


def NEW_CTX(rng, G, idx):
    """Optimize independent components against explicit normal-fiber words."""
    from math import gcd

    q = G.q
    assert q == 48
    profile, mode = idx % 5, (idx // 5) % 4
    plans = (
        ((2,1), (2,1), (1,2), (1,2), (2,2)),
        ((3,0), (0,3), (2,1), (1,2), (2,2)),
        ((3,1), (1,3), (1,2), (2,1), (1,1)),
        ((3,0), (1,3), (1,2), (1,2), (2,1)),
        ((2,1), (2,1), (2,1), (1,2), (1,2)),
    )
    sa = [[4, 4]]
    sb = [[c[i] for c in plans[profile]] for i in range(2)]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    def draw(counts, side, phase):
        """Occupancy and internal-order constraints, not orbit transforms."""
        grid = []
        for i, row in enumerate(counts):
            out = []
            for j, weight in enumerate(row):
                if weight <= 1:
                    out.append([] if not weight else [G.pick(rng)])
                    continue
                if side == "A":
                    outer = ((2,2), (1,1), (1,3), (2,1))[mode][j]
                elif weight == 2:
                    # Same-coset orders 24,12,8 contrast with order-four
                    # mixed-coset pairs. Independent labels are optimized.
                    outer = (1 if mode == 1 else
                             2*((i+j+phase) % 2))
                elif mode == 2:
                    outer = weight-1
                else:
                    outer = 1 + ((i+j+mode+phase) % (weight-1))
                quota = [0]*(weight-outer) + [1]*outer
                selected = None
                for restart in range(128):
                    rng.shuffle(quota)
                    cell = []
                    for b in quota:
                        available = []
                        for a in range(24):
                            g = (a, b)
                            # These are label constraints only. Screening
                            # maps below come exclusively from verified lifts.
                            if any(
                                b == h[1] and
                                (a % 8 == h[0] % 8 or
                                 a % 6 == h[0] % 6)
                                for h in cell
                            ):
                                continue
                            if weight == 2 and cell and b == cell[0][1]:
                                order = 24 // gcd((a-cell[0][0]) % 24, 24)
                                wanted = (24, 12, 24, 8)[mode]
                                if order != wanted:
                                    continue
                            available.append(g)
                        if not available:
                            break
                        cell.append(rng.choice(available))
                    if len(cell) == weight:
                        selected = cell
                        break
                if selected is None:
                    raise ValueError("normal-image cell recipe exhausted")
                out.append(selected)
            grid.append(out)
        return grid

    # H=<x^(24/h)> is normal. Its coset indicators have disjoint
    # supports of size h. Do not construct a separate quotient lift.
    fibers = {}
    for h in (2, 3, 4, 6, 8, 12, 24):
        remaining = set(range(q))
        partition = []
        while remaining:
            g = G.elements[min(remaining)]
            orbit = tuple(sorted(
                G.idx(G.mul(g, (t*(24//h), 0))) for t in range(h)))
            assert len(set(orbit)) == h
            remaining.difference_update(orbit)
            partition.append(orbit)
        fibers[h] = tuple(partition)

    def norm_columns(cols, h):
        result = []
        for block in range(len(cols)//q):
            for orbit in fibers[h]:
                syndrome = 0
                for t in orbit:
                    syndrome ^= cols[block*q+t]
                result.append(syndrome)
        return result

    def cheap_norm_bound(cols):
        """Known structured witness weight, or a no-witness sentinel."""
        bound = len(cols)+1
        for h in (3, 4, 6, 8, 12, 24):
            flag, _ = _short_kernel(norm_columns(cols, h))
            if flag <= 4:
                bound = min(bound, h*flag)
        return bound

    def norm_reading(cols, enumeration_cap):
        """Exact small restricted kernels; sampled larger restricted kernels.

        Each dependency expands to a genuine classical kernel word of
        weight h times its coefficient weight. This does not establish
        the unrestricted classical or quantum minimum.
        """
        best = len(cols)+1
        for h in fibers:
            compressed = norm_columns(cols, h)
            flag, _ = _short_kernel(
                compressed, through_six=(h == 2 and len(cols) == 2*q))
            if flag <= (6 if h == 2 and len(cols) == 2*q else 4):
                best = min(best, h*flag)

            pivots, dependencies = {}, []
            for c, syndrome in enumerate(compressed):
                word = 1 << c
                while syndrome:
                    p = syndrome.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    dependencies.append(word)
            if not dependencies:
                continue
            if len(dependencies) <= enumeration_cap:
                # Gray traversal enumerates the entire restricted kernel
                # without storing its exponentially many words.
                word = 0
                for t in range(1, 1 << len(dependencies)):
                    word ^= dependencies[(t & -t).bit_length()-1]
                    best = min(best, h*word.bit_count())
                    if best <= 4:
                        return best
            else:
                reading = _kernel_probe(compressed, rng, trials=16)
                if reading <= len(compressed):
                    best = min(best, h*reading)
        return best

    def a_merit(item):
        return (item["da"], -item["cycles"], item["rank"])

    # A is shared across five sector-one embeddings and repeatedly limits
    # the Z reading. Spend its search budget independently of B.
    skeleton = [[[G.e] for _ in range(5)] for _ in range(2)]
    a_pool = []
    for attempt in range(80 + 4*(idx//20)):
        A = draw(sa, "A", idx+attempt)
        hx, hz, _ = lifted_matrices(G, A, skeleton)
        ar, _ = _extract_components(hx, hz, q)
        ac = _component_columns(ar, 2*q)
        flag, _ = _short_kernel(ac)
        da = min(_kernel_probe(ac, rng, trials=2),
                 cheap_norm_bound(ac))
        if flag <= 4:
            da = min(da, flag)
        item = dict(A=A, ac=ac, da=da, rank=gf2_rank(ar),
                    cycles=_component_four_cycles(ac))
        a_pool.append(item)
        a_pool.sort(key=a_merit, reverse=True)
        del a_pool[6:]

    for item in a_pool:
        item["da"] = min(
            item["da"], _kernel_probe(item["ac"], rng, trials=48),
            norm_reading(item["ac"], enumeration_cap=18))
    a_pool.sort(key=a_merit, reverse=True)
    a_pool = a_pool[:4]

    def merit(item):
        d = min(item["da"], item["db"])
        return (item["k"]*d*d, d, item["da"]+item["db"],
                -item["cycles"])

    finalists = []
    for attempt in range(72 + 4*(idx//20)):
        ai = a_pool[attempt % len(a_pool)]
        A = ai["A"]
        B = draw(sb, "B", idx+attempt)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (225*n <= 1000*k < 275*n):
            continue
        _, br = _extract_components(hx, hz, q)
        bc = _component_columns(br, 5*q)
        flag, _ = _short_kernel(bc)
        db = min(_kernel_probe(bc, rng, trials=2),
                 cheap_norm_bound(bc))
        if flag <= 4:
            db = min(db, flag)
        item = dict(A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                    ac=ai["ac"], bc=bc, da=ai["da"], db=db,
                    cycles=ai["cycles"]+_component_four_cycles(bc))
        finalists.append(item)
        finalists.sort(key=merit, reverse=True)
        del finalists[6:]

    if not finalists:
        raise ValueError("no norm-screened candidate in target rate cell")

    # Fresh searches only lower previous readings. A lucky preliminary
    # information set must not erase a witness found by another search.
    for item in finalists:
        item["da"] = min(item["da"],
                         _kernel_probe(item["ac"], rng, trials=32))
        item["db"] = min(
            item["db"], _kernel_probe(item["bc"], rng, trials=24),
            norm_reading(item["bc"], enumeration_cap=14))
    finalists.sort(key=merit, reverse=True)

    winner, winner_score = None, None
    for item in finalists[:3]:
        clean = (
            not _small_logical_4(item["hx"], item["hz"], item["n"])
            and not _small_logical_4(item["hz"], item["hx"], item["n"]))
        dx = _quantum_probe(item["hz"], item["hx"], item["n"],
                            rng, trials=12)
        dz = _quantum_probe(item["hx"], item["hz"], item["n"],
                            rng, trials=12)
        # Component witnesses may be stabilizers after embedding, so
        # their readings are conservative selection heuristics, not
        # claimed quantum upper bounds. Full probes reject stabilizers.
        d = min(dx, dz, item["da"], item["db"])
        score = (int(clean), item["k"]*d*d, d,
                 min(dx, dz), item["da"]+item["db"], -item["cycles"])
        if winner_score is None or score > winner_score:
            winner, winner_score = item, score

    A, B = winner["A"], winner["B"]
    hx, hz, n = winner["hx"], winner["hz"], winner["n"]
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx+hz) <= W_MAX
    assert 225*n <= 1000*winner["k"] < 275*n
    assert all(len(cell) == len(set(cell))
               for grid in (A, B) for row in grid for cell in row)
    return {"idx": idx, "cells": {"A": A, "B": B},
            "k": winner["k"], "selection": winner_score}


def CELL(rng, G, side, i, j, ctx):
    """Return the independently optimized, normal-image-constrained cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted by parent slots 4/5/6/12/13: legal weight-12 rolled fillings with monomial-rich A do not prevent sector-1 Z logical witnesses of weight 6-8.
# REFUTED: C9 monomial voltage rows with isolated odd trinomial upgrades and exact weight<=3 logical rejection did not exceed distance four: all five measured parent codes had k=283 and d30s=4, with minima entirely in sector 1.
# REFUTED: GL(2,3) determinant cards, component probes and exact weight<=4 logical screening did not eliminate weight-12 sector-one logicals: all five measured parent codes had d30s=12.
# REFUTED: Dic12 central-involution exclusion and joint full-logical information-set selection raised the five measured readings above twelve; all five measured codes instead had k=153 and d30s=12, with minima entirely in sector one.
# EVOLVE-BLOCK-END


class _G(object):
    def __init__(self, ops, q):
        self.q = q
        self.elements = list(ops["elements"])
        self.e = ops["ident"]
        self.mul = ops["mul"]
        self.inv = ops["inv"]
        self._pos = {self._key(g): i for i, g in enumerate(self.elements)}

    @staticmethod
    def _key(g):
        return tuple(g) if isinstance(g, (list, tuple)) else g

    def idx(self, g):
        return self._pos[self._key(g)]

    def pick(self, rng):
        return self.elements[rng.randrange(self.q)]


# ---- verified construction helpers (2026-09-09) --------------------------
# lifted_matrices(G, cells_A, cells_B) builds EXACTLY the HX / HZ the
# evaluator builds (same lift conventions, same column order), as Python-int
# bit rows; gf2_rank(rows) is the rank behind k; lifted_k(G, A, B) is the k the table will report;
# css_ok(hx, hz) checks HX HZ^T = 0. Use these for any in-program screening -
# a hand-written lift that gets the left/right actions wrong screens a
# different (usually non-CSS) matrix pair than the one that is measured.
def lifted_matrices(G, cells_A, cells_B):
    """cells_A[i][j], cells_B[a][b]: lists of group elements (as CELL returns
    them) or element indices. Returns (hx_rows, hz_rows, n): lists of Python
    ints, bit c set = a 1 in column c; n = number of columns (qubits)."""
    q = G.q
    ra, ca = len(cells_A), len(cells_A[0])
    rb, cb = len(cells_B), len(cells_B[0])
    els = G.elements

    def gi(g):
        try:
            return G.idx(g)
        except (KeyError, TypeError):
            if isinstance(g, int) and 0 <= g < q:
                return g
            raise
    mul_idx = [[G.idx(G.mul(els[x], els[y])) for y in range(q)] for x in range(q)]
    inv_idx = [G.idx(G.inv(els[x])) for x in range(q)]
    A = [[sorted(set(gi(g) for g in cell)) for cell in row] for row in cells_A]
    B = [[sorted(set(gi(g) for g in cell)) for cell in row] for row in cells_B]
    n1 = ca * cb * q
    n = n1 + ra * rb * q
    hx = []
    for i in range(ra):
        for beta in range(cb):
            for r in range(q):
                row = 0
                for j in range(ca):                       # sector 1: A_{ij} left-lifted
                    base = (j * cb + beta) * q
                    for g in A[i][j]:
                        row ^= 1 << (base + mul_idx[inv_idx[g]][r])   # h with g*h = r
                for alpha in range(rb):                   # sector 2: B_{alpha beta}^T right-lifted
                    base = n1 + (i * rb + alpha) * q
                    for g in B[alpha][beta]:
                        row ^= 1 << (base + mul_idx[r][inv_idx[g]])   # c = r*g^-1
                hx.append(row)
    hz = []
    for j in range(ca):
        for alpha in range(rb):
            for r in range(q):
                row = 0
                for beta in range(cb):                    # sector 1: B_{alpha beta} right-lifted
                    base = (j * cb + beta) * q
                    for g in B[alpha][beta]:
                        row ^= 1 << (base + mul_idx[r][g])           # h with h*g^-1 = r
                for i in range(ra):                       # sector 2: A_{ij}^T left-lifted
                    base = n1 + (i * rb + alpha) * q
                    for g in A[i][j]:
                        row ^= 1 << (base + mul_idx[g][r])           # c = g*r
                hz.append(row)
    return hx, hz, n


def _popcount(x):
    return bin(x).count("1")



def gf2_rank(rows):
    """Rank over GF(2) of a list of bit rows."""
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break
    return len(pivots)


def lifted_k(G, cells_A, cells_B):
    """k = n - rank(HX) - rank(HZ) of the code the evaluator will build."""
    hx, hz, n = lifted_matrices(G, cells_A, cells_B)
    return n - gf2_rank(hx) - gf2_rank(hz)


def css_ok(hx, hz):
    """True iff every X check commutes with every Z check (HX HZ^T = 0)."""
    return all(_popcount(x & z) % 2 == 0 for x in hx for z in hz)


def _grid_weights(sa, sb):
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    wx = max(sum(sa[i]) + sum(sb[r_][j] for r_ in range(rb))
             for i in range(ra) for j in range(cb))
    wz = max(sum(sb[r_]) + sum(sa[i][c] for i in range(ra))
             for r_ in range(rb) for c in range(ca))
    return max(wx, wz)


def _no_dead_lines(sa, sb):
    return all(
        all(sum(row) > 0 for row in g)
        and all(sum(r_[c] for r_ in g) > 0 for c in range(len(g[0])))
        for g in (sa, sb))


def roll_supports(rng):
    """Random per-cell term counts (1..CELL_MAX), repaired down to the
    weight cap with 1 as the floor - the roller never emits zero blocks
    (a slot's CELL may still return [] deliberately). A scaffold
    utility available to NEW_CTX; always returns legal grids."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    sa = [[rng.randint(1, CELL_MAX) for _ in range(ca)] for _ in range(ra)]
    sb = [[rng.randint(1, CELL_MAX) for _ in range(cb)] for _ in range(rb)]
    for _ in range(4000):
        if _grid_weights(sa, sb) <= W_MAX:
            return sa, sb
        g = sa if rng.random() < 0.5 else sb
        i = rng.randrange(len(g))
        c = rng.randrange(len(g[0]))
        if g[i][c] >= 2:            # 1 is the floor: the roller never
            g[i][c] -= 1            # emits zero blocks (evolution may)
    raise ValueError("roll_supports could not fit the weight cap "
                     "(all-ones row exceeds it: shape too wide)")


def _registry_ops(gtype, q, a0):
    if gtype == "cyclic":
        return dict(elements=list(range(q)), ident=0,
                    mul=lambda g, h: (g + h) % q,
                    inv=lambda g: (-g) % q)
    if gtype == "abelian":
        a = int(a0)
        if a < 1 or q % a:
            raise ValueError("A0 must divide q")
        b = q // a
        return dict(elements=[(x, y) for x in range(a) for y in range(b)],
                    ident=(0, 0),
                    mul=lambda g, h: ((g[0] + h[0]) % a, (g[1] + h[1]) % b),
                    inv=lambda g: ((-g[0]) % a, (-g[1]) % b))
    raise ValueError("GTYPE %r needs CUSTOM_GROUP" % gtype)


def _make_ops(q):
    if GTYPE == "custom":
        ns = {}
        exec(CUSTOM_GROUP, ns)
        if not ns["valid_q"](q):
            raise ValueError("valid_q rejects q=%d" % q)
        return ns["custom_ops"](q)
    return _registry_ops(GTYPE, q, A0)


def _valid_q(q):
    if GTYPE == "custom":
        ns = {}
        exec(CUSTOM_GROUP, ns)
        return bool(ns["valid_q"](q))
    if GTYPE == "abelian":
        return A0 is not None and int(A0) >= 1 and q % int(A0) == 0
    return q >= 2


def _validate_group(G):
    rng = random.Random(7)
    sample = [G.e] + [G.pick(rng) for _ in range(min(12, G.q))]
    for g in sample:
        assert G._key(G.mul(G.e, g)) == G._key(g), "identity fails"
        assert G._key(G.mul(g, G.inv(g))) == G._key(G.e), "inverse fails"
    for g in sample[:6]:
        for h in sample[:6]:
            for k in sample[:6]:
                assert G._key(G.mul(G.mul(g, h), k)) \
                    == G._key(G.mul(g, G.mul(h, k))), "associativity fails"
    assert len(G._pos) == len(G.elements) == G.q, "element list broken"


def _choose_q():
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    if min(ra, ca, rb, cb) < 1:
        raise ValueError("protograph sides need >= 1 row and >= 1 column")
    f = ca * cb + ra * rb
    best = None
    for q in range(max(2, N_MIN // f), N_MAX // f + 1):
        if f * q < N_MIN or f * q >= N_MAX:
            continue
        if _valid_q(q):
            best = q
    if best is None:
        raise ValueError("no valid q puts n in [%d, %d)" % (N_MIN, N_MAX))
    return best


def _fill_slot(rng, G, idx):
    ctx = NEW_CTX(rng, G, idx)
    grids = {}
    rgrids = {}
    for side, (rows, cols) in (("A", A_SHAPE), ("B", B_SHAPE)):
        grid = []
        rgrid = []
        for i in range(rows):
            row = []
            rrow = []
            for j in range(cols):
                cell = CELL(rng, G, side, i, j, ctx)
                if not isinstance(cell, (list, tuple)):
                    raise ValueError("%s[%d][%d]: not a list" % (side, i, j))
                if len(cell) > CELL_MAX:
                    raise ValueError("%s[%d][%d]: >%d terms"
                                     % (side, i, j, CELL_MAX))
                keys = {G._key(g) for g in cell}
                if len(keys) != len(cell):
                    raise ValueError("%s[%d][%d]: duplicate element"
                                     % (side, i, j))
                pairs = sorted((G.idx(g), G._key(g)) for g in cell)
                row.append([i for i, _ in pairs])
                rrow.append([list(e) if isinstance(e, tuple) else e
                             for _, e in pairs])
            grid.append(row)
            rgrid.append(rrow)
        grids[side] = grid
        rgrids[side] = rgrid
    sa = [[len(c) for c in row] for row in grids["A"]]
    sb = [[len(c) for c in row] for row in grids["B"]]
    w = _grid_weights(sa, sb)
    if w > W_MAX:
        raise ValueError("row weight %d exceeds cap %d" % (w, W_MAX))
    if not _no_dead_lines(sa, sb):
        raise ValueError("all-zero protograph row or column")
    return grids["A"], grids["B"], rgrids["A"], rgrids["B"]


def get_candidates():
    """-> request dict for the worker: shared group data + 20 slots."""
    q = _choose_q()
    ops = _make_ops(q)
    G = _G(ops, q)
    _validate_group(G)
    mul_table = [[G.idx(G.mul(G.elements[a], G.elements[b]))
                  for b in range(q)] for a in range(q)]
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    slots = []
    for s in range(N_SLOTS):
        rng = random.Random(BASE_SEED * 10007 + s)
        try:
            A, B, Ar, Br = _fill_slot(rng, G, s)
            slots.append({"idx": s, "A": A, "B": B, "Ar": Ar, "Br": Br})
        except Exception as e:
            slots.append({"idx": s, "error": str(e)[:120]})
    return {"family": FAMILY_NAME, "concept": CONCEPT,
            "gtype": GTYPE, "q": q, "mul": mul_table,
            "a_shape": [ra, ca], "b_shape": [rb, cb],
            "n": (ca * cb + ra * rb) * q,
            "slots": slots}


if __name__ == "__main__":
    r = get_candidates()
    ok = sum(1 for s in r["slots"] if "error" not in s)
    print(json.dumps({"family": r["family"], "q": r["q"], "n": r["n"],
                      "slots_ok": ok}))

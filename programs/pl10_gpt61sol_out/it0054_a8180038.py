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
CONCEPT = ("On A4 x C3, order 36, shapes (1,3)x(1,4), redistributing singleton/binomial/trinomial budgets and minimizing verified weight-three/four logical multiplicities should improve worse-basis circuit reliability over four-cycle-only selection.")
GROUP = ("Frozen by L1. This order-36 bed has best recorded fitness about 3.5, versus the current parent's 3.2178. Its order-six differences provide useful binomial options, but excluding involutory differences did not remove the parent's weight-three logical ambiguity.")
SHAPES = ("Frozen unequal components: A has 36 checks and 108 variables, B has 36 checks and 144 variables. n=468 and kb=216 give guaranteed rate 6/13; verified complete-code ranks retain k=216..222 in cell 0.450. Parent short logicals occupy sector-one Acol/Bcol blocks, and failure enrichment is diffuse at roughly 1.0-1.1, not evidence for one local repair.")
FILLING = ("Twenty briefs cross five support allocations with four ratio/phase policies. Singleton-plus-three-trinomial B designs test stronger protection against dominant Z failures; other allocations test the opposite tradeoff. Component searches reject duplicate syndromes, limit rank deficits and minimize weight-three/four kernel multiplicities. Finalists are selected by actual short quantum logical counts using verified lifts and stabilizer membership, with cycles only a secondary criterion. Target the parent's 112..137 Z failures per 250 shots and collective logical flips; neither these screens nor distance is asserted to predict circuit fitness exactly.")
FAMILY_NAME = "r045_A4xC3_13x14_syndrome_budget_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    from itertools import permutations
    if not valid_q(q):
        raise ValueError("A4 x C3 requires order 36")
    perms = [p for p in permutations(range(4))
             if sum(p[i] > p[j] for i in range(4)
                    for j in range(i + 1, 4)) % 2 == 0]
    els = [p + (c,) for p in perms for c in range(3)]
    pos = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(int(g[i]) for i in range(4)) + (int(g[4]) % 3,)

    def mul(g, h):
        return tuple(g[h[i]] for i in range(4)) + ((g[4] + h[4]) % 3,)

    def inv(g):
        p = [0] * 4
        for i in range(4):
            p[g[i]] = i
        return tuple(p) + ((-g[4]) % 3,)

    def idx(g):
        return pos[red(g)]

    return dict(elements=els, ident=(0, 1, 2, 3, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


def _orders(G):
    result = []
    for g in G.elements:
        h = G.e
        for t in range(1, G.q + 1):
            h = G.mul(h, g)
            if G._key(h) == G._key(G.e):
                result.append(t)
                break
        else:
            raise ValueError("element order missing")
    return result


def _designed_cell(rng, G, count, target_order, mode, orders):
    """Avoid order-two differences; prescribe binomial cycle lengths."""
    for _ in range(512):
        cell = rng.sample(G.elements, count)
        # Distinct projected permutations prevent pure-central pairs.
        if len({tuple(g[:4]) for g in cell}) != count:
            continue
        relative = [
            orders[G.idx(G.mul(G.inv(cell[a]), cell[b]))]
            for a in range(count) for b in range(a + 1, count)
        ]
        if min(relative) < 3:
            continue
        if count == 2:
            if relative[0] != target_order:
                continue
        else:
            central_span = len({g[4] for g in cell})
            wanted = 3 if mode == 0 else 2
            if central_span != wanted:
                continue
        return cell
    raise ValueError("could not realize cell design")


def _same_basis_cycles(rows):
    """Number of Tanner four-cycles within one check basis."""
    total = 0
    for i, x in enumerate(rows):
        for j in range(i):
            overlap = (x & rows[j]).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def NEW_CTX(rng, G, idx):
    """Optimize one support/ratio brief, then classify quantum ambiguities."""
    profiles = (
        ((2, 2, 2), (1, 3, 3, 3)),
        ((2, 2, 2), (2, 2, 3, 3)),
        ((2, 2, 3), (1, 2, 3, 3)),
        ((3, 3, 3), (1, 2, 3, 3)),
        ((2, 3, 3), (2, 2, 2, 3)),
    )
    case = idx % 20
    pa, pb = profiles[case // 4]
    mode = case % 4
    assert _grid_weights([list(pa)], [list(pb)]) <= W_MAX
    assert _no_dead_lines([list(pa)], [list(pb)])
    q, e = G.q, G.e
    orders = _orders(G)

    six = [g for g in G.elements if orders[G.idx(g)] == 6]
    three = [g for g in G.elements
             if orders[G.idx(g)] == 3 and tuple(g[:4]) != tuple(e[:4])]
    central = [g for g in G.elements
               if orders[G.idx(g)] == 3 and tuple(g[:4]) == tuple(e[:4])]

    # Fix the first cell term to identity. Search relative supports,
    # rather than spending proposals on arbitrary whole-cell translations.
    nonidentity = [g for g in G.elements if G._key(g) != G._key(e)]
    triples = []
    for a, g in enumerate(nonidentity):
        for h in nonidentity[a + 1:]:
            cell = [e, g, h]
            span = len({x[4] for x in cell})
            projections = len({tuple(x[:4]) for x in cell})
            if mode == 0 and not (span >= 2 and projections == 3):
                continue
            if mode == 1 and span != 2:
                continue
            if mode == 2 and span != 1:
                continue
            if mode == 3 and span != 3:
                continue
            triples.append(cell)
    if not triples:
        raise ValueError("empty trinomial policy")

    def columns(rows, ncols):
        result = [0] * ncols
        for r, row in enumerate(rows):
            bit = 1 << r
            while row:
                low = row & -row
                result[low.bit_length() - 1] |= bit
                row ^= low
        return result

    def component_quality(side, T):
        # A one-cell opposite-side probe extracts the exact action used
        # by the scaffold. No independent left/right lift is implemented.
        if side == "A":
            rows = lifted_matrices(G, T, [[[e]]])[0]
        else:
            rows = lifted_matrices(G, [[[e]]], T)[1]
        ncols = len(T[0]) * q
        mask = (1 << ncols) - 1
        rows = [row & mask for row in rows]
        deficit = q - gf2_rank(rows)
        if deficit > 1:
            return None
        cols = columns(rows, ncols)
        if 0 in cols or len(set(cols)) != ncols:
            return None
        singles = set(cols)
        pairs = {}
        n3 = n4 = cycles = 0
        weights = [c.bit_count() for c in cols]
        for a, x in enumerate(cols):
            for b in range(a):
                s = x ^ cols[b]
                n3 += int(s in singles)
                old = pairs.get(s, 0)
                n4 += old
                pairs[s] = old + 1
                overlap = (weights[a] + weights[b] - s.bit_count()) // 2
                cycles += overlap * (overlap - 1) // 2
        # Nonzero distinct columns make these exact classical counts.
        assert n3 % 3 == 0 and n4 % 3 == 0
        return (n3 // 3, n4 // 3, cycles, deficit)

    def design_side(side, counts):
        nb = counts.count(2)
        if mode == 0:
            policies = [six] * min(nb, 2) + [three] * max(0, nb - 2)
        elif mode == 1:
            policies = ([six] + [three] * (nb - 1)) if nb else []
        elif mode == 2:
            policies = [three] * nb
        else:
            policies = ([central] if nb else [])
            if nb >= 2:
                policies.append(six)
            policies.extend([three] * max(0, nb - 2))

        options = []
        b = 0
        for w in counts:
            if w == 1:
                options.append([[e]])
            elif w == 2:
                options.append([[e, g] for g in policies[b]])
                b += 1
            else:
                options.append(triples)

        def valid_ratios(T):
            used = set()
            for cell in T[0]:
                if len(cell) != 2:
                    continue
                g = cell[1]
                key, inverse = G._key(g), G._key(G.inv(g))
                if key in used or inverse in used:
                    return False
                used.update((key, inverse))
            return True

        def fingerprint(T):
            return tuple(tuple(sorted(G.idx(g) for g in cell))
                         for cell in T[0])

        pool = []
        seen = set()

        def retain(T):
            signature = fingerprint(T)
            if signature in seen or not valid_ratios(T):
                return None
            seen.add(signature)
            quality = component_quality(side, T)
            if quality is None:
                return None
            pool.append((quality, signature, T))
            pool.sort(key=lambda item: (item[0], item[1]))
            del pool[5:]
            return quality

        for trial in range(80):
            T = [[list(rng.choice(choices)) for choices in options]]
            retain(T)
        if not pool:
            raise ValueError("no distinct-syndrome component passed rank screen")

        # Whole-relative-cell mutations preserve this slot's count and
        # ratio/phase architecture. Equal-cost moves explore plateaus.
        starts = list(pool[:3])
        mutable = [j for j, choices in enumerate(options) if len(choices) > 1]
        for quality, signature, initial in starts:
            current = [[list(cell) for cell in initial[0]]]
            current_quality = quality
            for step in range(32):
                j = rng.choice(mutable)
                trial = [[list(cell) for cell in current[0]]]
                trial[0][j] = list(rng.choice(options[j]))
                candidate_quality = retain(trial)
                if candidate_quality is not None and (
                        candidate_quality < current_quality or
                        (candidate_quality == current_quality
                         and rng.random() < 0.35)):
                    current, current_quality = trial, candidate_quality
        return pool

    apool = design_side("A", pa)
    bpool = design_side("B", pb)

    def logical_counts(checks, stabilizers, n):
        """Exact non-stabilizer kernel-support counts through weight four."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        def nontrivial(v):
            while v:
                p = v.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    return True
                v ^= old
            return False

        cols = columns(checks, n)
        bits = [1 << a for a in range(n)]
        singles = {}
        result = [0, 0, 0, 0]
        for a, syndrome in enumerate(cols):
            singles.setdefault(syndrome, []).append(a)
            if syndrome == 0 and nontrivial(bits[a]):
                result[0] += 1
        pairs = {}
        for a, x in enumerate(cols):
            for b in range(a + 1, n):
                s = x ^ cols[b]
                support = bits[a] | bits[b]
                if s == 0 and nontrivial(support):
                    result[1] += 1
                for c in singles.get(s, ()):
                    if c > b and nontrivial(support | bits[c]):
                        result[2] += 1
                bucket = pairs.setdefault(s, [])
                # c < d < a < b selects exactly one partition of each
                # ordered weight-four support; overlapping pairs are ignored.
                for c, d, previous in bucket:
                    if d < a and nontrivial(previous | support):
                        result[3] += 1
                bucket.append((a, b, support))
        return tuple(result)

    finalists = []
    for aq, af, A in apool:
        for bq, bf, B in bpool:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (216 <= k <= 222):
                continue
            key = (max(aq[0], bq[0]), aq[0] + bq[0],
                   max(aq[1], bq[1]), aq[1] + bq[1],
                   aq[2] + bq[2], af, bf)
            finalists.append((key, A, B, hx, hz, n, k))
    finalists.sort(key=lambda item: item[0])

    best = None
    for component_key, A, B, hx, hz, n, k in finalists[:3]:
        if not css_ok(hx, hz):
            continue
        x = logical_counts(hz, hx, n)
        z = logical_counts(hx, hz, n)
        short = tuple(v for t in range(4)
                      for v in (max(x[t], z[t]), x[t] + z[t]))
        edges = sum(row.bit_count() for row in hx + hz)
        key = short + (edges, component_key)
        if best is None or key < best[0]:
            best = (key, A, B, k)
    if best is None:
        raise ValueError("no screened filling retained the 0.450 rate cell")
    key, A, B, k = best
    return {"idx": idx, "case": case, "k": k, "screen": key,
            "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without further randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Parent Z17 slots 31/50 had d2s=5 but worse-basis s=3.691/3.892 and large basis gaps; that screen value does not establish balanced circuit protection.
# REFUTED: Excluding involutory differences and minimizing four-cycles suffices for reliable r045 memory: fitness was 3.2178, measured Z failures were 112..137/250 shots, and every measured code had weight-three X-type logical witnesses.
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

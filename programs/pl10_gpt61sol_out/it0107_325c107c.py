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
CONCEPT = ("For A4 x C2, order 24, shapes (1,3)x(2,7), unit-trinomial A cells and an independently labeled two-row B component should improve worse-basis circuit fitness above 3.4383 by reducing short-logical ambiguity under check weight 12.")
GROUP = ("L2 order adjustment only: retain the direct product of even permutations on four letters with a cyclic factor, replacing C3 by C2. The new shape requires q<=26; q=24 gives n=552. The normal 2-subgroup V4 x C2 has quotient C3, allowing odd trinomials with monomial quotient image to be units. This algebra does not establish circuit reliability.")
SHAPES = ("A=(1,3), B=(2,7): n=552, kb=240 and guaranteed rate 10/23=0.43478 in cell 0.450. Components have 24 checks/72 variables and 48 checks/168 variables. The frontier's related (1,3)x(2,7) bed scores about 4.0, motivating this shape test. Parent weight-three Z witnesses occupied Acol2 across every B column, while failure enrichments were diffuse, at roughly 1.0-1.2x, without a consistent Brow hotspot. Exact final ranks require k=240.")
FILLING = ("Twenty briefs cross four support architectures with five A quotient/central-span policies. Three architectures use A=(3,3,3); one uses (2,3,3). B column degrees are 2-4, row totals 8-9, and both CSS weight sums are <=12. Independently optimize relative supports and inter-row voltages using verified lifts; require full component ranks, reject duplicate syndrome columns, and shortlist weight-three/four kernels. Finalists minimize exact nontrivial short logical multiplicities and shared-check hook opportunities. CNOT budgets are 20736-22032 versus the parent's 15120, an explicit circuit-risk tradeoff. Target the parent's 48 weight-three Z witnesses and X failures of 108/250 to 181/500 despite first-stage settlement; only measured pL confirms improvement.")
FAMILY_NAME = "r045_A4xC2_13x27_unit_twocheck_hook_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    from itertools import permutations
    if not valid_q(q):
        raise ValueError("A4 x C2 requires order 24")
    perms = [p for p in permutations(range(4))
             if sum(p[i] > p[j] for i in range(4)
                    for j in range(i + 1, 4)) % 2 == 0]
    els = [p + (c,) for p in perms for c in range(2)]
    pos = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(int(g[i]) for i in range(4)) + (int(g[4]) % 2,)

    def mul(g, h):
        return tuple(g[h[i]] for i in range(4)) + ((g[4] + h[4]) % 2,)

    def inv(g):
        p = [0] * 4
        for i in range(4):
            p[g[i]] = i
        return tuple(p) + ((-g[4]) % 2,)

    def idx(g):
        return pos[red(g)]

    return dict(elements=els, ident=(0, 1, 2, 3, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


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
    """Four support architectures crossed with five relative-support briefs."""
    q = G.q
    architecture = (idx // 5) % 4
    mode = idx % 5
    extension = idx // 20
    sa = [[3, 3, 3]] if architecture < 3 else [[2, 3, 3]]
    column_profiles = (
        ((2, 1), (2, 1), (1, 2), (1, 2),
         (1, 1), (1, 1), (1, 1)),
        ((2, 1), (1, 2), (1, 1), (1, 1),
         (1, 1), (1, 1), (1, 1)),
        ((2, 1), (1, 2), (0, 3), (3, 0),
         (1, 1), (1, 1), (1, 1)),
        ((2, 2), (2, 1), (1, 2), (1, 1),
         (1, 1), (1, 1), (1, 1)),
    )
    profile = column_profiles[architecture]
    sb = [[column[i] for column in profile] for i in range(2)]
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    # Further harness batches have different B ratio-order prescriptions.
    ratio_orders = ((6, 6), (6, 3), (3, 3))[extension % 3]

    orders = _orders(G)
    v4 = [g for g in G.elements
          if g[4] == 0 and orders[G.idx(g)] <= 2]

    def quotient(g):
        # Quotient by the normal 2-subgroup V4 x C2.
        return min(tuple(G.mul(h, g)[:4]) for h in v4)

    quotient_id = quotient(G.e)
    quotients = {G.idx(g): quotient(g) for g in G.elements}
    nonidentity = [g for g in G.elements
                   if G._key(g) != G._key(G.e)]
    ratio_pools = {
        order: [g for g in G.elements
                if orders[G.idx(g)] == order
                and quotients[G.idx(g)] != quotient_id]
        for order in (3, 6)
    }
    # A trinomials are identity-normalized. Quotient span one or two
    # gives a monomial image in F2[C3], hence a unit modulo the
    # nilpotent ideal associated with the normal 2-subgroup.
    quotient_patterns = (
        (2, 2, 2), (2, 2, 2), (1, 2, 2),
        (1, 1, 2), (2, 2, 2),
    )
    central_patterns = (
        (2, 2, 2), (1, 1, 1), (2, 2, 2),
        (2, 2, 2), (1, 2, 2),
    )

    def triple(number, side):
        if side == "A":
            wanted_q = quotient_patterns[mode][number % 3]
            wanted_c = central_patterns[mode][number % 3]
        else:
            wanted_q = 1 if mode == 3 and number % 2 == 0 else 2
            wanted_c = 1 if mode == 1 else 2
        for _ in range(1024):
            u, v = rng.sample(nonidentity, 2)
            relative = [G.e, u, v]
            if len({quotients[G.idx(g)] for g in relative}) != wanted_q:
                continue
            if len({g[4] for g in relative}) != wanted_c:
                continue
            if wanted_q == 2 and (
                    G._key(G.mul(u, v)) == G._key(G.mul(v, u))):
                continue
            if side == "A":
                return relative
            # Inter-row B voltages are genuine design variables; do
            # not normalize every B cell to the same monomial anchor.
            base = G.pick(rng)
            return [G.mul(g, base) for g in relative]
        raise ValueError("unit-trinomial brief could not be realized")

    def draw_grid(side, support):
        grid = []
        heavy = 0
        for row_counts in support:
            row = []
            used = set()
            binary = 0
            for count in row_counts:
                if count == 0:
                    cell = []
                elif count == 1:
                    cell = [G.pick(rng)]
                elif count == 2:
                    order = 6 if side == "A" else ratio_orders[binary % 2]
                    choices = [g for g in ratio_pools[order]
                               if G._key(g) not in used]
                    if not choices:
                        raise ValueError("binomial brief exhausted")
                    delta = rng.choice(choices)
                    used.update((G._key(delta), G._key(G.inv(delta))))
                    base = G.e if side == "A" else G.pick(rng)
                    cell = [base, G.mul(delta, base)]
                    binary += 1
                else:
                    cell = triple(heavy, side)
                    heavy += 1
                assert len(cell) == count
                assert len({G._key(g) for g in cell}) == count
                row.append(cell)
            grid.append(row)
        return grid

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= tag
                row ^= bit
        return result

    def components(hx, hz):
        # Extract exclusively from the helper-built complete matrices.
        # Bcol0 X checks contain the left-lifted A component.
        mask = (1 << q) - 1
        ar = []
        for row in hx[:q]:
            compact = 0
            for j in range(A_SHAPE[1]):
                base = j * B_SHAPE[1] * q
                compact |= ((row >> base) & mask) << (j * q)
            ar.append(compact)
        # Acol0 Z checks contain BOTH right-lifted B check rows.
        width = B_SHAPE[1] * q
        br = [row & ((1 << width) - 1)
              for row in hz[:B_SHAPE[0] * q]]
        return ar, br

    def kernel_stats(cols):
        """Exact weight-three/four kernel counts for unique columns."""
        if 0 in cols or len(set(cols)) != len(cols):
            return None
        singles = set(cols)
        pairs = {}
        triples = collisions = cycles = 0
        weights = [c.bit_count() for c in cols]
        for i, x in enumerate(cols):
            for j in range(i):
                s = x ^ cols[j]
                triples += int(s in singles)
                old = pairs.get(s, 0)
                collisions += old
                pairs[s] = old + 1
                overlap = (weights[i] + weights[j] - s.bit_count()) // 2
                cycles += overlap * (overlap - 1) // 2
        assert triples % 3 == 0 and collisions % 3 == 0
        return triples // 3, collisions // 3, cycles

    def logical_stats(checks, stabilizers, n):
        """Exact logical counts and shared-check hook opportunities.

        Unique nonzero columns exclude weights one and two. Quotient
        residues classify collisions modulo the stabilizer row span.
        Hook statistics count possible fault-pair partitions, not
        ASC-certified faults or an estimated circuit failure rate.
        """
        cols = columns(checks, n)
        if 0 in cols or len(set(cols)) != n:
            return None
        spread = columns(stabilizers, n)
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        ordered = sorted(pivots.items(), reverse=True)
        residuals = []
        for i in range(n):
            v = 1 << i
            for p, row in ordered:
                if (v >> p) & 1:
                    v ^= row
            residuals.append(v)

        singles = {s: residuals[i] for i, s in enumerate(cols)}
        totals = {}
        classes = {}
        triples = fours = double_hook = single_hook = 0
        for i, x in enumerate(cols):
            ri = residuals[i]
            hi = spread[i]
            for j in range(i):
                s = x ^ cols[j]
                residue = ri ^ residuals[j]
                single = singles.get(s)
                if single is not None and residue != single:
                    triples += 1
                hook = (hi & spread[j]).bit_count()
                key = (s, residue)
                nt, ht = totals.get(s, (0, 0))
                nc, hc = classes.get(key, (0, 0))
                different = nt - nc
                different_hooks = ht - hc
                fours += different
                double_hook += hook * different_hooks
                single_hook += hook * different + different_hooks
                totals[s] = (nt + 1, ht + hook)
                classes[key] = (nc + 1, hc + hook)
        assert triples % 3 == 0 and fours % 3 == 0
        return triples // 3, fours // 3, double_hook, single_hook

    shortlist = []
    exact_seen = set()
    for trial in range(192):
        try:
            A = draw_grid("A", sa)
            B = draw_grid("B", sb)
        except ValueError:
            continue
        identity = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row
        )
        if identity in exact_seen:
            continue
        exact_seen.add(identity)

        hx, hz, n = lifted_matrices(G, A, B)
        ar, br = components(hx, hz)
        if gf2_rank(ar) != A_SHAPE[0] * q:
            continue
        if gf2_rank(br) != B_SHAPE[0] * q:
            continue
        ast = kernel_stats(columns(ar, A_SHAPE[1] * q))
        bst = kernel_stats(columns(br, B_SHAPE[1] * q))
        if ast is None or bst is None:
            continue

        # Component words can replicate across opposite-side columns.
        # This coarse weighting is only a shortlisting heuristic.
        a3, b3 = B_SHAPE[1] * ast[0], A_SHAPE[1] * bst[0]
        a4, b4 = B_SHAPE[1] * ast[1], A_SHAPE[1] * bst[1]
        ca, cb = ast[2] / len(ar), bst[2] / len(br)
        key = (max(a3, b3), a3 + b3,
               max(a4, b4), a4 + b4,
               max(ca, cb), ca + cb)
        shortlist.append((key, identity, A, B, hx, hz, n))
        shortlist.sort(key=lambda item: item[:2])
        del shortlist[8:]

    best = None
    best_key = None
    for component_key, identity, A, B, hx, hz, n in shortlist:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 10 * q or not (425 * n <= 1000 * k < 475 * n):
            continue
        lx = logical_stats(hx, hz, n)
        lz = logical_stats(hz, hx, n)
        if lx is None or lz is None:
            continue
        fx = _same_basis_cycles(hx)
        fz = _same_basis_cycles(hz)
        # Exclude short logical mechanisms before comparing possible
        # two-hook and one-hook fault-pair ambiguities in both bases.
        key = (max(lx[0], lz[0]), lx[0] + lz[0],
               max(lx[2], lz[2]), lx[2] + lz[2],
               max(lx[3], lz[3]), lx[3] + lz[3],
               max(lx[1], lz[1]), lx[1] + lz[1],
               max(fx / len(hx), fz / len(hz)),
               fx + fz, component_key, identity)
        if best_key is None or key < best_key:
            best_key = key
            best = (A, B, hx, hz, k, lx, lz)

    if best is None:
        raise ValueError("no full-rank two-check filling passed screening")
    A, B, hx, hz, k, lx, lz = best
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    edges = sum(row.bit_count() for row in hx + hz)
    expected = q * (
        (B_SHAPE[1] + B_SHAPE[0]) * sum(map(sum, sa))
        + (A_SHAPE[1] + A_SHAPE[0]) * sum(map(sum, sb)))
    assert edges == expected
    return {"idx": idx, "architecture": architecture, "mode": mode,
            "extension": extension, "k": k, "screen": best_key,
            "logical_profile": (lx, lz), "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without further randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Parent Z17 slots 31/50 had d2s=5 but worse-basis s=3.691/3.892 and large basis gaps; that screen value does not establish balanced circuit protection.
# REFUTED: Excluding involutory cell differences and minimizing four-cycles alone did not protect this A4 x C3 bed: fitness 3.2178, Z failures 112..137/250, and widespread weight-three sector-1 witnesses despite mostly stage-1 settlement.
# REFUTED: Low gate count and connected binomial A suffice for reliable balanced memory: the triangle-transfer parent scored 3.4383, retained 48 weight-three Z witnesses, and failed 108/250 to 181/500 X shots despite first-stage settlement.
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

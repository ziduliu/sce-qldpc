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
CONCEPT = ("A4 x C3, order 36, shapes (1,3)x(1,4): replacing the B trinomial by a fourth binomial and using quartet-containing or certified-unit A supports should improve worse-basis circuit reliability through lower gate exposure, even when more weight-three logicals are admitted.")
GROUP = ("Frozen by L1. The current order-36 bed scores 3.4843. Two distinct order-six directions and two noncentral order-three subgroups can connect B's lift positions without duplicate binomial edge types. Verified component and complete-code ranks test this architecture; nonabelian labels alone do not establish reliability.")
SHAPES = ("Frozen (1,3)x(1,4): component lengths 108 and 144, n=468, and kb=216 give guaranteed rate 6/13. Require actual k=218..222, still in cell 0.450. Parent weight-three witnesses occupied all three A columns at Bcol0, while failed-shot error enrichment was diffuse at about 1.0-1.1x across Acol/Bcol and check blocks.")
FILLING = ("Twenty briefs cross five A support profiles with four B phase architectures. B has four binomials: two distinct order-six directions and two distinct noncentral order-three subgroups. A uses certified unit trinomials and, in four profiles, new quartets. Verified component screening rejects duplicate syndromes and minimizes short kernels and small boundaries. Final selection uses exact stabilizer-aware weight-three/four logical multiplicities weighted by incident gate degree. Six-round CNOT counts are 12096, 12960 or 13824, versus 15336 for the parent's measured codes. The falsifiable tradeoff is fewer circuit faults versus additional short logicals; target the parent's 140-176 Z failures per 500 shots and collective flips, not distance as fitness.")
FAMILY_NAME = "r045_A4xC3_13x14_quartet_gate_tradeoff_v3"

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
    """Lower-gate structural briefs, not relabelings of saved codes."""
    profiles = (
        (3, 3, 3),
        (1, 3, 4),
        (1, 4, 4),
        (2, 3, 4),
        (3, 3, 4),
    )
    phases = ((0, 0), (0, 1), (1, 1), (1, 2))
    case = idx % 20
    pa = profiles[case % 5]
    pb = (2, 2, 2, 2)
    phase_pair = phases[case // 5]
    variant = (idx // 20) % 3
    q, e = G.q, G.e
    assert _grid_weights([list(pa)], [list(pb)]) <= W_MAX
    assert _no_dead_lines([list(pa)], [list(pb)])

    orders = _orders(G)
    klein = [g for g, o in zip(G.elements, orders)
             if o == 2 and g[4] == 0]
    threes = [g for g, o in zip(G.elements, orders)
              if o == 3 and g[4] == 0
              and tuple(g[:4]) != tuple(e[:4])]
    nonidentity = [g for g in G.elements if G._key(g) != G._key(e)]

    # Four projected order-three subgroups, distinguished from their
    # inverse orientations. Sampling two avoids parallel edge families.
    three_families = {}
    for g in threes:
        family = tuple(sorted((G.idx(g), G.idx(G.inv(g)))))
        three_families.setdefault(family, []).append(g)
    family_keys = sorted(three_families)

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= tag
                row ^= bit
        return result

    def component_rows(side, grid):
        # Extract the exact action through the scaffold, including
        # B's right lift. The sector-two probe is masked away.
        if side == "A":
            rows = lifted_matrices(G, grid, [[[e]]])[0]
        else:
            rows = lifted_matrices(G, [[[e]]], grid)[1]
        width = len(grid[0]) * q
        mask = (1 << width) - 1
        return [row & mask for row in rows], width

    def component_stats(rows, width):
        """Exact weight-three/four kernels for unique nonzero columns."""
        col = columns(rows, width)
        if 0 in col or len(set(col)) != width:
            return None
        singles = set(col)
        weights = [v.bit_count() for v in col]
        pairs = {}
        triples = collisions = cycles = boundary = 0
        for i, x in enumerate(col):
            for j in range(i):
                s = x ^ col[j]
                triples += int(s in singles)
                collisions += pairs.get(s, 0)
                pairs[s] = pairs.get(s, 0) + 1
                overlap = (weights[i] + weights[j] - s.bit_count()) // 2
                cycles += overlap * (overlap - 1) // 2
                w = s.bit_count()
                if w < 3:
                    boundary += (3 - w) ** 2
        assert triples % 3 == 0 and collisions % 3 == 0
        return triples // 3, collisions // 3, boundary, cycles

    def unit_trinomial(j):
        # The involutory pair cancels in the quotient by normal V4,
        # leaving a monomial. Certify the actual group-ring unit anyway.
        u = rng.choice(klein)
        v = rng.choice(threes)
        coordinate = (j + variant) % 3
        v = tuple(v[:4]) + (coordinate,)
        cell = [e, u, v]
        rows, _, _ = lifted_matrices(G, [[cell]], [[[]]])
        if gf2_rank(rows) != q:
            raise ValueError("unit-trinomial certificate failed")
        return cell

    def quartet():
        # Extra harness slots change heavy-cell phase architecture,
        # rather than merely applying a group automorphism.
        for _ in range(512):
            cell = [e] + rng.sample(nonidentity, 3)
            span = len({g[4] for g in cell})
            wanted = (2, 2, 3)[variant]
            if span == wanted:
                return cell
        raise ValueError("quartet phase architecture unavailable")

    def draw_A():
        row = []
        for j, count in enumerate(pa):
            if count == 1:
                row.append([e])
            elif count == 2:
                u = rng.choice(klein)
                delta = tuple(u[:4]) + (rng.choice((1, 2)),)
                row.append([e, delta])
            elif count == 3:
                row.append(unit_trinomial(j))
            else:
                row.append(quartet())
        return [row]

    def draw_B():
        directions = rng.sample(klein, 2)
        row = [[e, tuple(u[:4]) + (rng.choice((1, 2)),)]
               for u in directions]
        chosen = rng.sample(family_keys, 2)
        for family, coordinate in zip(chosen, phase_pair):
            g = rng.choice(three_families[family])
            row.append([e, tuple(g[:4]) + (coordinate,)])
        return [row]

    def select_components(side):
        pool = []
        seen = set()
        wanted_rank = q if side == "A" else q - 1
        for trial in range(256):
            if trial >= 96 and pool:
                break
            grid = draw_A() if side == "A" else draw_B()
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for cell in grid[0])
            if signature in seen:
                continue
            seen.add(signature)
            rows, width = component_rows(side, grid)
            if gf2_rank(rows) != wanted_rank:
                continue
            stats = component_stats(rows, width)
            if stats is None:
                continue
            pool.append((stats, signature, grid))
            pool.sort(key=lambda item: (item[0], item[1]))
            del pool[6:]
        if not pool:
            raise ValueError("no full-rank distinct-syndrome component")
        return pool

    apool = select_components("A")
    bpool = select_components("B")

    def census(checks, stabilizers, n, degree):
        """Exact logical counts and pair-degree exposure through weight four.

        Exposure is only a fault-opportunity heuristic. It does not
        model the ASC schedule, correlated hooks or decoder behavior.
        """
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        bits = [1 << i for i in range(n)]
        residues = list(bits)
        for p in sorted(pivots):
            row = pivots[p] ^ bits[p]
            residue = 0
            while row:
                bit = row & -row
                residue ^= residues[bit.bit_length() - 1]
                row ^= bit
            residues[p] = residue

        col = columns(checks, n)
        singles = {}
        counts = [0, 0, 0, 0]
        risk = [0, 0, 0, 0]
        for a, syndrome in enumerate(col):
            singles.setdefault(syndrome, []).append(a)
            if syndrome == 0 and residues[a]:
                counts[0] += 1

        pairs = {}
        for a in range(n):
            x, ra, da = col[a], residues[a], degree[a]
            for b in range(a + 1, n):
                syndrome = x ^ col[b]
                residue = ra ^ residues[b]
                db = degree[b]
                if syndrome == 0 and residue:
                    counts[1] += 1
                for c in singles.get(syndrome, ()):
                    if c > b and residue ^ residues[c]:
                        dc = degree[c]
                        counts[2] += 1
                        risk[2] += da * db + da * dc + db * dc

                ds = da + db
                dq = da * da + db * db
                bucket = pairs.get(syndrome)
                entry = (b, residue, ds, dq)
                if bucket is None:
                    pairs[syndrome] = [entry]
                else:
                    # c < d < a < b counts each four-support once.
                    for d, previous, ps, pq in bucket:
                        if d < a and previous ^ residue:
                            counts[3] += 1
                            total = ds + ps
                            risk[3] += (total * total - dq - pq) // 2
                    bucket.append(entry)
        return tuple(counts), tuple(risk)

    finalists = []
    for ast, af, A in apool:
        for bst, bf, B in bpool:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (218 <= k <= 222):
                continue
            # Sector-one component kernels replicate over the other
            # side's columns. This is a shortlist heuristic only.
            a3, b3 = 4 * ast[0], 3 * bst[0]
            a4, b4 = 4 * ast[1], 3 * bst[1]
            key = (max(a3, b3), a3 + b3,
                   max(a4, b4), a4 + b4,
                   ast[2] + bst[2], ast[3] + bst[3], af, bf)
            finalists.append((key, A, B, hx, hz, n, k))
    finalists.sort(key=lambda item: item[0])

    best = None
    best_key = None
    for component_key, A, B, hx, hz, n, k in finalists[:8]:
        xc, zc = columns(hx, n), columns(hz, n)
        degree = [x.bit_count() + z.bit_count()
                  for x, z in zip(xc, zc)]
        x, rx = census(hz, hx, n, degree)
        z, rz = census(hx, hz, n, degree)
        if x[0] + z[0] + x[1] + z[1]:
            continue

        # Allow the architectural triangle/gate tradeoff. The small
        # weight-four coefficient is a regularizer, not a pL estimate.
        ex = 100 * rx[2] + rx[3]
        ez = 100 * rz[2] + rz[3]
        edges = sum(degree)
        cx = _same_basis_cycles(hx) / len(hx)
        cz = _same_basis_cycles(hz) / len(hz)
        key = (max(ex, ez), ex + ez,
               max(x[2], z[2]), x[2] + z[2],
               edges, max(cx, cz), cx + cz, component_key)
        if best_key is None or key < best_key:
            best_key = key
            best = (A, B, hx, hz, k, edges)

    if best is None:
        raise ValueError("no lower-gate filling passed the logical census")
    A, B, hx, hz, k, edges = best
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert 6 * edges in (12096, 12960, 13824)
    return {"idx": idx, "case": case, "k": k, "screen": best_key,
            "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete stabilizer-aware screened cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Parent Z17 slots 31/50 had d2s=5 but worse-basis s=3.691/3.892 and large basis gaps; that screen value does not establish balanced circuit protection.
# REFUTED: Excluding involutory cell differences and minimizing four-cycles did not ensure reliable A4 x C3 memory: fitness 3.2178, measured Z failures 112-137/250, and 72-216 weight-three X witnesses.
# REFUTED: Exactly 36 weight-three X logicals and zero weight-three Z logicals did not suffice for low circuit failure rates: fitness 3.4843, Z failures 140-176/500, and collective flips across essentially all 216 logical qubits.
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

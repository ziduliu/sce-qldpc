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
CONCEPT = ("A4 x C3, order 36, shapes (1,3)x(1,4): central-charge separation between A-trinomial differences and B-triangle edges, followed by verified short-logical hook-exposure screening, should improve worse-basis circuit fitness above 3.2178.")
GROUP = ("Frozen by L1. The order-36 bed has measured scores 3.2178 and 3.4843 under different fillings. Conjugation preserves the central C3 coordinate, allowing selected triangle pairs to be excluded from opposite-basis co-checks by construction; this is a circuit-reliability hypothesis, not a distance guarantee.")
SHAPES = ("Frozen (1,3)x(1,4): n=468, guaranteed k>=216 and rate 6/13 in cell 0.450; component lengths are 108 and 144. These shapes permit q<=46, while the frozen group fixes q=36. Require exact k=218. Parent Z failures were 112..137/250 shots, with widespread sector-1 weight-three witnesses and diffuse Acol/Bcol enrichment near 1.0..1.1x.")
FILLING = ("Four A support profiles (1,3,3), (2,2,3), (2,3,3), (3,3,3) pair with B=(2,2,2,2). A trinomials have central coordinate zero; B has two distinct order-six differences and two noncentral order-three differences with nonzero central charge. Verified component screens precede an exact stabilizer-aware logical census through weight four, including co-check pair exposure. A deterministic library selects five different full-code invariant signatures per support profile, not labeling orbits. Maximum check weight is 11 and six-round CNOT counts are 14472, 15552 or 16632. The objective is reducing circuit failures, not maximizing the distance proxy.")
FAMILY_NAME = "r045_A4xC3_13x14_charge_hook_library_v1"

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
    """Four support architectures, each with five invariant-distinct codes.

    Rebuilding the same local library is intentional: idx chooses a
    designed outcome, not a new random draw or a translated seed.
    """
    profile, choice = divmod(idx % 20, 5)
    pa = ((1, 3, 3), (2, 2, 3),
          (2, 3, 3), (3, 3, 3))[profile]
    pb = (2, 2, 2, 2)
    assert _grid_weights([list(pa)], [list(pb)]) <= min(W_MAX, 11)
    rr = random.Random(92317561 + 104729 * profile)
    q = G.q
    orders = _orders(G)
    v4 = [g for g in G.elements
          if g[4] == 0 and orders[G.idx(g)] <= 2]
    involutions = [g for g in v4 if orders[G.idx(g)] == 2]
    threes = [g for g in G.elements
              if g[4] == 0 and orders[G.idx(g)] == 3]
    zero_charge = [g for g in G.elements
                   if g[4] == 0 and G._key(g) != G._key(G.e)]

    def quotient(g):
        return min(tuple(G.mul(v, g)[:4]) for v in v4)

    def charged(g, c):
        return tuple(g[:4]) + (c,)

    def triple(span):
        for _ in range(512):
            u, v = rr.sample(zero_charge, 2)
            cell = [G.e, u, v]
            if len({quotient(g) for g in cell}) != span:
                continue
            if span > 1:
                if G._key(G.mul(u, v)) == G._key(G.mul(v, u)):
                    continue
            return cell
        raise ValueError("central-zero triple policy unavailable")

    def draw(mode):
        av = rr.sample(involutions, 2)
        ar = []
        binary = heavy = 0
        for count in pa:
            if count == 1:
                ar.append([G.e])
            elif count == 2:
                ar.append([G.e, charged(av[binary], rr.choice((1, 2)))])
                binary += 1
            else:
                # The first heavy cell is a unit modulo the V4 radical:
                # its quotient support has span one or two. Later
                # cells can test a nonunit three-coset support.
                if heavy == 0:
                    span = 1 if mode == 4 else 2
                else:
                    span = 2 if mode < 2 else 3
                ar.append(triple(span))
                heavy += 1

        bv = rr.sample(involutions, 2)
        p = rr.choice(threes)
        same_projection = mode in (0, 2) or (
            mode == 4 and rr.randrange(2) == 0)
        if same_projection:
            td = [charged(p, 1), charged(p, 2)]
        else:
            partners = [h for h in threes
                        if quotient(h) == quotient(p)
                        and G._key(h) != G._key(p)]
            h = rr.choice(partners)
            td = [charged(p, 1), charged(h, 1)]
        bd = [charged(v, rr.choice((1, 2))) for v in bv] + td
        # These are actual right-component edge displacements.
        B = [[[G.e, G.inv(d)] for d in bd]]
        return [ar], B

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
        mask = (1 << q) - 1
        ar = []
        for row in hx[:q]:
            compact = 0
            for j in range(3):
                compact |= ((row >> (j * 4 * q)) & mask) << (j * q)
            ar.append(compact)
        br = [row & ((1 << (4 * q)) - 1) for row in hz[:q]]
        return ar, br

    def kernel_counts(rows, width):
        cols = columns(rows, width)
        if 0 in cols or len(set(cols)) != width:
            return None
        singles = set(cols)
        pairs = {}
        triples = fours = 0
        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                triples += int(syndrome in singles)
                old = pairs.get(syndrome, 0)
                fours += old
                pairs[syndrome] = old + 1
        assert triples % 3 == 0 and fours % 3 == 0
        return triples // 3, fours // 3

    def logical_exposure(checks, stabilizers, n):
        """Exact logical counts and co-check pair exposure at weights 3/4.

        Unique nonzero syndrome columns exclude weights one and two.
        Opposite-row-span residues distinguish logicals from stabilizers.
        Each support occurs in three pair partitions. Co-check exposure
        is a hook-opportunity statistic, not an ASC fault certificate.
        """
        cols = columns(checks, n)
        if 0 in cols or len(set(cols)) != n:
            return None
        co = columns(stabilizers, n)
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

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

        singles = {s: i for i, s in enumerate(cols)}
        pairs = {}
        n3 = h3 = n4 = h4 = 0
        for i, x in enumerate(cols):
            ri = residues[i]
            for j in range(i):
                syndrome = x ^ cols[j]
                residue = ri ^ residues[j]
                k = singles.get(syndrome)
                if k is not None and residue ^ residues[k]:
                    n3 += 1
                    h3 += ((co[i] & co[j]).bit_count()
                           + (co[i] & co[k]).bit_count()
                           + (co[j] & co[k]).bit_count())

                bucket = pairs.get(syndrome)
                if bucket is None:
                    pairs[syndrome] = [(i, j, residue)]
                    continue
                for a, b, previous in bucket:
                    if previous == residue:
                        continue
                    # Equal pair syndromes cannot share a coordinate
                    # when the individual columns are distinct.
                    n4 += 1
                    h4 += ((co[i] & co[j]).bit_count()
                           + (co[a] & co[b]).bit_count()
                           + (co[i] & co[a]).bit_count()
                           + (co[i] & co[b]).bit_count()
                           + (co[j] & co[a]).bit_count()
                           + (co[j] & co[b]).bit_count())
                bucket.append((i, j, residue))
        assert all(v % 3 == 0 for v in (n3, h3, n4, h4))
        return n3 // 3, h3 // 3, n4 // 3, h4 // 3

    def cycle_signature(rows):
        """Exact four/six-cycle counts and check-overlap histogram."""
        m = len(rows)
        overlap = [[0] * m for _ in range(m)]
        adj = [0] * m
        hist = [0] * 13
        four = six = 0
        for i, row in enumerate(rows):
            for j in range(i):
                t = (row & rows[j]).bit_count()
                hist[t] += 1
                four += t * (t - 1) // 2
                if t:
                    overlap[i][j] = overlap[j][i] = t
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        for i in range(m):
            earlier = adj[i] & ((1 << i) - 1)
            while earlier:
                bit = earlier & -earlier
                j = bit.bit_length() - 1
                common = adj[i] & adj[j] & ((1 << j) - 1)
                ij = rows[i] & rows[j]
                x = overlap[i][j]
                while common:
                    low = common & -common
                    h = low.bit_length() - 1
                    y, z = overlap[j][h], overlap[i][h]
                    t = (ij & rows[h]).bit_count()
                    six += x * y * z - t * (x + y + z) + 2 * t
                    common ^= low
                earlier ^= bit
        return four, six, tuple(hist)

    library = {}
    exact_seen = set()
    for batch in range(3):
        buckets = {}
        for trial in range(144):
            mode = (trial + 144 * batch) % 5
            A, B = draw(mode)
            identity = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for grid in (A, B) for row in grid for cell in row)
            if identity in exact_seen:
                continue
            exact_seen.add(identity)

            hx, hz, n = lifted_matrices(G, A, B)
            ar, br = components(hx, hz)
            if gf2_rank(ar) != q or gf2_rank(br) != q - 1:
                continue
            sa = kernel_counts(ar, 3 * q)
            sb = kernel_counts(br, 4 * q)
            if sa is None or sb is None:
                continue
            if sa[0] != 0 or sb[0] != 24:
                continue
            ca, cb = _same_basis_cycles(ar), _same_basis_cycles(br)
            fast = (max(sa[1], sb[1]), sa[1] + sb[1],
                    max(ca, cb), ca + cb)
            bucket_key = (sa[1], sb[1], ca, cb, mode)
            bucket = buckets.setdefault(bucket_key, [])
            if len(bucket) < 2:
                bucket.append((fast, identity, A, B, hx, hz, n))

        pool = [item for bucket in buckets.values() for item in bucket]
        pool.sort(key=lambda item: (item[0], item[1]))
        for _, identity, A, B, hx, hz, n in pool[:40]:
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if k != 218:
                continue
            x = logical_exposure(hz, hx, n)
            z = logical_exposure(hx, hz, n)
            if x is None or z is None:
                continue
            if x[0] != 72 or z[0] != 0 or x[1] or z[1]:
                continue

            cx, cz = cycle_signature(hx), cycle_signature(hz)
            # Logical support counts, co-check pair counts, and Tanner
            # cycle signatures are unchanged by basis-preserving graph
            # relabelings. Sorting also identifies a basis-swap image.
            invariant = tuple(sorted(((x, cx), (z, cz))))
            key = (max(x[3], z[3]), x[3] + z[3],
                   max(x[2], z[2]), x[2] + z[2],
                   max(cx[0] / len(hx), cz[0] / len(hz)),
                   cx[0] + cz[0],
                   max(cx[1] / len(hx), cz[1] / len(hz)),
                   cx[1] + cz[1])
            old = library.get(invariant)
            candidate = (key, identity, A, B, hx, hz)
            if old is None or candidate[:2] < old[:2]:
                library[invariant] = candidate
        if len(library) >= 5:
            break

    if len(library) < 5:
        raise ValueError("fewer than five invariant-distinct hook-screened codes")
    ordered = sorted(library.values(), key=lambda item: item[:2])
    _, _, A, B, hx, hz = ordered[choice]
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= 11
    expected = 6 * q * (5 * sum(pa) + 4 * sum(pb))
    assert 6 * sum(row.bit_count() for row in hx + hz) == expected
    return {"idx": idx, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without further randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Parent Z17 slots 31/50 had d2s=5 but worse-basis s=3.691/3.892 and large basis gaps; that screen value does not establish balanced circuit protection.
# REFUTED: Excluding involutory cell differences and minimizing same-basis four-cycles alone did not ensure reliable A4 x C3 memory: fitness 3.2178, measured Z failures 112..137/250, and widespread weight-three witnesses despite predominantly stage-1 settlement.
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

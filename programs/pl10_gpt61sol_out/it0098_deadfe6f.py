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
CONCEPT = ("On metacyclic Z7:Z2(s=6), order 14, shapes (1,4)x(2,7), low-multiplicity A=[1,2,3,3] components and mostly degree-three B columns should improve worse-basis circuit reliability over abundant weight-four logicals, provided their short logicals have sufficiently little possible hook exposure.")
GROUP = ("Keep the frozen order-14 group and multiplication rule. This bed scored 3.2001 and the related (2,4)x(2,5) bed scored 4.7; neither reading isolates a group defect. This is a filling-only comparison.")
SHAPES = ("Keep (1,4)x(2,7): n=420, kb=210, guaranteed rate 0.500, with unequal component matrices 14x56 and 28x98. Parent failures affected all 213 logical qubits, with diffuse 1.0-1.2x Acol/Bcol enrichment and no consistent Brow hotspot. Protect both components rather than patch one block.")
FILLING = ("Twenty canonical support architectures use A counts [1,2,3,3] and B column degrees two or three, with 17-18 B terms and row sums <=9. Both CSS check weights are <=12; six-round CNOT counts are 13944 or 14364, explicitly higher than the parent's 13440. Enumerate A translation classes with minimum weight-three multiplicity, optimize independent B voltages, and classify actual short CSS logicals using verified lifts and stabilizer spans. Prefer fewer possible hooks on triples, then fewer triples and lower quartet exposure. Rotation-separated A binomials and reflection-separated B doubletons test a fault-geometry tradeoff, not a distance improvement. The target is the parent's 121-133 Z failures per 250 shots and collective logical flips despite almost universal first-pass settlement.")
FAMILY_NAME = "r050_m7r2s6_14x27_sparse_triples_denseB_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 14

def custom_ops(q):
    # metacyclic Z_7 x| Z_2: y x y^-1 = x^6 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 7, 2, 6
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def _distinct_terms(rng, G, count):
    terms = []
    seen = set()
    while len(terms) < count:
        g = G.pick(rng)
        key = G._key(g)
        if key not in seen:
            seen.add(key)
            terms.append(g)
    return terms


def _b_profile(rng, extra, zeros):
    """Seven live columns; fixed total terms and fixed zero-cell count."""
    for _ in range(2000):
        triple = set(rng.sample(range(7), extra))
        sparse = set(rng.sample(range(7), zeros))
        sb = [[0] * 7 for _ in range(2)]
        for j in range(7):
            degree = 3 if j in triple else 2
            if j in sparse:
                a, b = 0, degree
            else:
                a, b = 1, degree - 1
            if rng.randrange(2):
                a, b = b, a
            sb[0][j], sb[1][j] = a, b
        if 0 < sum(sb[0]) <= 9 and 0 < sum(sb[1]) <= 9:
            return sb
    raise ValueError("could not realize deliberate B profile")


def _syndrome_risk(rows, n):
    """Ambiguity counts, not a distance certificate or circuit score."""
    columns = [0] * n
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= check
            row ^= bit
    if 0 in columns or len(set(columns)) != n:
        return None
    singles = set(columns)
    pairs = {}
    triples = 0
    for a in range(n):
        x = columns[a]
        for b in range(a):
            syndrome = x ^ columns[b]
            triples += syndrome in singles
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    collisions = sum(c * (c - 1) // 2 for c in pairs.values())
    return triples // 3, collisions


def NEW_CTX(rng, G, idx):
    """Trade short-word multiplicity against B protection and hook exposure."""
    from itertools import combinations, combinations_with_replacement

    q = G.q
    els = G.elements
    e = G.e
    mask = (1 << q) - 1
    mul = [[G.idx(G.mul(g, h)) for h in els] for g in els]
    autos = [
        [G.idx(((u * g[0] + v * g[1]) % 7, g[1]))
         for g in els]
        for u in range(1, 7) for v in range(7)
    ]

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def four_cycles(cols):
        overlaps = {}
        for syndrome in cols:
            checks = []
            while syndrome:
                bit = syndrome & -syndrome
                checks.append(bit.bit_length() - 1)
                syndrome ^= bit
            for pair in combinations(checks, 2):
                overlaps[pair] = overlaps.get(pair, 0) + 1
        return sum(t * (t - 1) // 2 for t in overlaps.values())

    def component_fours(cols, weights):
        """Exact weight-four words; ordered pairs count each support once."""
        pairs = {}
        count = exposure = 0
        participation = [0] * len(cols)
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                syndrome = cols[a] ^ cols[b]
                bucket = pairs.setdefault(syndrome, [])
                for c, d in bucket:
                    if d < a:
                        count += 1
                        ids = (c, d, a, b)
                        exposure += sum(weights[x] * weights[y]
                                        for x, y in combinations(ids, 2))
                        for x in ids:
                            participation[x] += 1
                bucket.append((a, b))
        return exposure, count, max(participation)

    # Right translations are column gauges. Enumerate their trinomial
    # orbits rather than repeatedly drawing relabeled copies.
    tri_type = {}
    representatives = []
    for triple in combinations(range(q), 3):
        if triple in tri_type:
            continue
        orbit = {
            tuple(sorted(mul[g][h] for g in triple))
            for h in range(q)
        }
        t = len(representatives)
        representatives.append(min(orbit))
        for image in orbit:
            tri_type[image] = t

    type_actions = [
        [tri_type[tuple(sorted(aut[g] for g in triple))]
         for triple in representatives]
        for aut in autos
    ]

    # Obtain every component column from the verified full construction.
    # Only extraction/concatenation is performed here, not a separate lift.
    tri_columns = []
    B_probe = [[[e] for _ in range(7)] for _ in range(2)]
    unit_columns = None
    for triple in representatives:
        A_probe = [[[e], [els[g] for g in triple], [e], [e]]]
        hx, hz, n = lifted_matrices(G, A_probe, B_probe)
        ar = [((hx[t] >> (7 * q)) & mask) for t in range(q)]
        tri_columns.append(columns(ar, q))
        if unit_columns is None:
            unit_columns = columns([hx[t] & mask for t in range(q)], q)

    def component_threes(cs):
        """Exact count when columns are distinct and nonzero."""
        if 0 in cs or len(set(cs)) != len(cs):
            return None
        singles = set(cs)
        hits = sum((x ^ cs[b]) in singles
                   for a, x in enumerate(cs) for b in range(a))
        assert hits % 3 == 0
        return hits // 3

    # Normalize the sole binomial to the order-seven displacement x.
    # Only automorphisms preserving its undirected step canonicalize A.
    bin_actions = [type_actions[7 * (u - 1) + v]
                   for u in (1, 6) for v in range(7)]
    A_probe = [[[e], [e, (1, 0)], [e], [e]]]
    hx, _, _ = lifted_matrices(G, A_probe, B_probe)
    bin_rows = [(hx[t] >> (7 * q)) & mask for t in range(q)]
    bin_columns = columns(bin_rows, q)

    A_book = []
    seen_A = set()
    for triple_types in combinations(range(len(representatives)), 2):
        canonical = min(tuple(sorted(action[t] for t in triple_types))
                        for action in bin_actions)
        if canonical in seen_A:
            continue
        seen_A.add(canonical)
        ac = list(unit_columns) + list(bin_columns)
        for t in canonical:
            ac.extend(tri_columns[t])
        threes = component_threes(ac)
        if threes is None:
            continue
        # B-column degrees vary, so these are only coarse A costs.
        risk = component_fours(
            ac, [3] * q + [4] * q + [5] * (2 * q))
        A_book.append(((threes,) + risk + (four_cycles(ac),), canonical))
    A_book.sort()
    if not A_book:
        raise ValueError("no distinct A columns")
    minimum_three = A_book[0][0][0]
    A_book = [entry for entry in A_book
              if entry[0][0] == minimum_three]

    # Enumerate support architectures, not twenty random support draws.
    # Each column has degree 2 or 3; canonicalization removes independent
    # B-column permutations and exchange of its two rows.
    column_types = ((1, 1), (2, 0), (0, 2),
                    (2, 1), (1, 2), (3, 0), (0, 3))
    support_book = {}
    for card in combinations_with_replacement(column_types, 7):
        total = sum(a + b for a, b in card)
        if total not in (17, 18):
            continue
        if max(sum(a for a, b in card),
               sum(b for a, b in card)) > 9:
            continue
        # Avoid repeated pure binomial/trinomial blocks on one row.
        if any(sum(a == degree and b == 0 for a, b in card) > 1
               or sum(b == degree and a == 0 for a, b in card) > 1
               for degree in (2, 3)):
            continue
        normal = tuple(sorted(card))
        swapped = tuple(sorted((b, a) for a, b in card))
        canonical = min(normal, swapped)
        zero_cells = sum(not a or not b for a, b in canonical)
        triples = sum(a == 3 or b == 3 for a, b in canonical)
        support_book[canonical] = (zero_cells, triples, total, canonical)
    cards = sorted(support_book, key=lambda card: support_book[card])
    if len(cards) < 20:
        raise ValueError("insufficient deliberate B architectures")

    design = idx % 20
    card = cards[design]
    sb = [[column[i] for column in card] for i in range(2)]
    sa = [[1, 2, 3, 3]]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    a_entry = A_book[(idx // 20) % min(8, len(A_book))]
    a_types = a_entry[1]
    A = [[[e], [e, (1, 0)]] +
         [[els[g] for g in representatives[t]] for t in a_types]]
    bmask = (1 << (7 * q)) - 1

    def span(rows):
        pivots = {}
        for row in rows:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        return pivots

    def logical_risk(check_cols, stab_cols, stab_rows, degrees):
        """Exact nontrivial logicals through weight four, plus hook proxy."""
        pivots = span(stab_rows)

        def nontrivial(v):
            while v:
                p = v.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    return True
                v ^= old
            return False

        width = len(check_cols)
        bits = [1 << a for a in range(width)]
        singles = {}
        counts = [0, 0, 0, 0]
        hooks = exposure = triple_hooks = 0
        for a, syndrome in enumerate(check_cols):
            bucket = singles.setdefault(syndrome, [])
            if syndrome == 0 and nontrivial(bits[a]):
                counts[0] += 1
            for b in bucket:
                if nontrivial(bits[a] | bits[b]):
                    counts[1] += 1
            bucket.append(a)

        pairs = {}
        for a in range(width):
            for b in range(a + 1, width):
                syndrome = check_cols[a] ^ check_cols[b]
                support = bits[a] | bits[b]
                for c in singles.get(syndrome, ()):
                    if c > b and nontrivial(support | bits[c]):
                        counts[2] += 1
                        for x, y in combinations((a, b, c), 2):
                            triple_hooks += (
                                stab_cols[x] & stab_cols[y]).bit_count()
                bucket = pairs.setdefault(syndrome, [])
                for c, d, previous in bucket:
                    if d >= a or not nontrivial(previous | support):
                        continue
                    counts[3] += 1
                    ids = (c, d, a, b)
                    # Common same-error-basis checks can propagate pairs.
                    # This is pessimistic: the actual ASC ordering is not
                    # available here, so it is not a circuit certificate.
                    for x, y in combinations(ids, 2):
                        hooks += (stab_cols[x] & stab_cols[y]).bit_count()
                        exposure += degrees[x] * degrees[y]
                bucket.append((a, b, support))
        return tuple(counts), triple_hooks, hooks, exposure

    def draw_B():
        """Independent relative voltages; never an image of a saved code."""
        B = [[[] for _ in range(7)] for _ in range(2)]
        for j, (upper, lower) in enumerate(card):
            for i, weight in enumerate((upper, lower)):
                if not weight:
                    continue
                base = e if i == 0 else G.pick(rng)
                if weight == 1:
                    cell = [base]
                elif weight == 2:
                    # Pure degree-two blocks need order-seven ratios;
                    # mixed-row doubletons use reflection differences.
                    opposite = lower if i == 0 else upper
                    direction = ((rng.choice((2, 3, 4, 5)), 0)
                                 if opposite == 0
                                 else (rng.randrange(7), 1))
                    cell = [base, G.mul(base, direction)]
                else:
                    # Its only rotational pair avoids the A step +/-1.
                    direction = (rng.choice((2, 3, 4, 5)), 0)
                    cell = [base, G.mul(base, direction),
                            G.mul(base, (rng.randrange(7), 1))]
                assert len({G._key(g) for g in cell}) == weight
                B[i][j] = cell
        return B

    # Screen the B projection cheaply, before full quantum classification.
    # Weight-three B words are penalized, not silently assumed absent.
    pool = []
    seen = set()
    for trial in range(192):
        B = draw_B()
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in B for cell in row)
        if signature in seen:
            continue
        seen.add(signature)
        hx, hz, n = lifted_matrices(G, A, B)
        bc = columns([row & bmask for row in hz[:2 * q]], 7 * q)
        b3 = component_threes(bc)
        if b3 is None:
            continue
        b4 = component_fours(bc, [c.bit_count() + 2 for c in bc])
        coarse = (b3, b4[1], b4[2], four_cycles(bc), b4[0])
        pool.append((coarse, trial, B, hx, hz, n))
        pool.sort(key=lambda item: (item[0], item[1]))
        del pool[6:]

    best = None
    best_key = None
    for coarse, trial, B, hx, hz, n in pool:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 19 * n <= 40 * k < 21 * n):
            continue
        xc, zc = columns(hx, n), columns(hz, n)
        degrees = [x.bit_count() + z.bit_count() for x, z in zip(xc, zc)]
        zrisk = logical_risk(xc, zc, hz, degrees)
        xrisk = logical_risk(zc, xc, hx, degrees)
        # This experiment intentionally allows a small triple family,
        # but never a nontrivial weight-one or weight-two logical.
        if any(zrisk[0][:2]) or any(xrisk[0][:2]):
            continue
        cx, cz = four_cycles(xc), four_cycles(zc)
        key = (
            max(xrisk[1], zrisk[1]), xrisk[1] + zrisk[1],
            max(xrisk[0][2], zrisk[0][2]),
            xrisk[0][2] + zrisk[0][2],
            max(xrisk[2], zrisk[2]), xrisk[2] + zrisk[2],
            max(xrisk[3], zrisk[3]), xrisk[3] + zrisk[3],
            max(xrisk[0][3], zrisk[0][3]),
            xrisk[0][3] + zrisk[0][3],
            max(cx / len(hx), cz / len(hz)), cx + cz,
            k - 210,
        )
        if best_key is None or key < best_key:
            best_key = key
            best = {"A": A, "B": B, "design": design,
                    "A_three": minimum_three, "risk": key, "k": k}
    if best is None:
        raise ValueError("no target-rate filling passed weight-one/two screen")
    hx, hz, n = lifted_matrices(G, best["A"], best["B"])
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    return best


def CELL(rng, G, side, i, j, ctx):
    """Return the complete multiplicity- and hook-screened cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone did not give reliable memory on the parent D14 (6,2)x(6,2) bed: fitness 3.4511, ubiquitous weight-four sector-2 witnesses, and frequent failures in both bases.
# REFUTED: First-pass syndrome settlement certifies reliable memory on the immediate parent: slot 0 X settled 246/4/0 yet failed 120/250 shots, affecting all 213 logical qubits.
# REFUTED: Odd systematic A=[1,3,3,3] plus monomial bipartite B and quartet hook balancing suffices to improve this bed: fitness fell from 3.2219 to 3.2001, only six distinct codes survived 60 slots, and measured codes retained thousands of weight-four witnesses with 121-133 Z failures per 250 shots.
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

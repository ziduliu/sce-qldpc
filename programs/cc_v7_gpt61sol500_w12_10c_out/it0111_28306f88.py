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
CONCEPT = ("Affine C7:C6(s=3), order 42, shapes (1,2)x(3,5): degree-four A columns and degree-three/four B columns selected against short component dependencies should produce five measured codes exceeding the rate-0.150 frontier fitness 29.2.")
GROUP = ("Jump from cyclic C9 to the nonabelian affine group C7:C6. The action is legal because 3^6=1 mod 7; order 42 gives n=546. This bed is absent from BEDS. Strong measured nonabelian component-screening beds motivate the experiment, without establishing a distance advantage for this group.")
SHAPES = ("Target cell 0.150, whose fitness 29.2 trails both neighbouring cells. A=(1,2), B=(3,5) give n=546, kb=84 and guaranteed rate 2/13. Component variable/check counts are 84/42 and 210/126. Parent measured weight-four logicals occupied sector-1 Acol/Bcol blocks broadly. Replace both dense degree-two geometries; verified ranks require 0.125<=k/n<0.175.")
FILLING = ("Twenty cards cross A counts (4,4)/(3,5), five B support profiles, and two relative-order palettes. A row sum eight plus B column sums <=4 gives HX<=12; B row sums <=7 plus A column sums <=5 gives HZ<=12. Avoid involution-separated terms and commuting multi-term cells. Extract components only from verified lifts, shortlist by exact dependencies through four and four-cycles, refine by randomized kernel witnesses, and exclude weight<=3 quantum logicals. Different uncoloured Tanner-incidence fingerprints exclude emitted orbit copies. Witness readings are heuristics, not quantum-distance certificates.")
FAMILY_NAME = "r015_affine7x6_12x35_deg4_kernelprobe_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    # Affine group over F7:
    # x^7=y^6=1, y*x*y^-1=x^3; (a,b)=x^a*y^b.
    if not valid_q(q):
        raise ValueError("C7:C6 requires order 42")
    els = [(a,b) for b in range(6) for a in range(7)]
    powers = [pow(3, b, 7) for b in range(6)]

    def red(g):
        return (g[0] % 7, g[1] % 6)

    def mul(g, h):
        a,b = red(g)
        c,d = red(h)
        return ((a + powers[b]*c) % 7, (b+d) % 6)

    def inv(g):
        a,b = red(g)
        bi = (-b) % 6
        return ((-powers[bi]*a) % 7, bi)

    def idx(g):
        a,b = red(g)
        return 7*b+a

    return dict(elements=els, ident=(0,0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)

from functools import lru_cache

# Begin with a full 3x5 monomial B; double the indicated cell
# in each column. -1 leaves that column all-monomial.
# These profiles differ in total degrees or row-degree distributions.
_AFFINE_B_PROFILES = (
    (-1, -1, -1, -1, -1),
    (0, 1, 2, -1, -1),
    (0, 0, 1, 2, -1),
    (0, 0, 1, 1, 2),
    (0, 0, 1, 1, -1),
)


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


def _small_logical_3(checks, stabilizers, width):
    """Exactly detect any non-stabilizer logical of weight one to three."""
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p not in basis:
                basis[p] = row
                break
            row ^= basis[p]

    def in_span(v):
        while v:
            p = v.bit_length() - 1
            if p not in basis:
                return False
            v ^= basis[p]
        return True

    syndromes = _component_columns(checks, width)
    singles = {}
    for i, syndrome in enumerate(syndromes):
        v = 1 << i
        if syndrome == 0 and not in_span(v):
            return True
        if syndrome in singles:
            if not in_span(v ^ singles[syndrome]):
                return True
        else:
            singles[syndrome] = v

    # After the singleton test, equal-syndrome singleton representatives
    # differ only by stabilizers, so one representative per syndrome suffices.
    for i in range(width):
        vi, si = 1 << i, syndromes[i]
        for j in range(i):
            syndrome = si ^ syndromes[j]
            v = vi | (1 << j)
            if syndrome == 0 and not in_span(v):
                return True
            if syndrome in singles:
                if not in_span(v ^ singles[syndrome]):
                    return True
    return False


@lru_cache(maxsize=1)
def _affine_label_data(G):
    """Algebraic label inventory, not a separately implemented lift."""
    orders = {}
    for g in G.elements:
        h = G.e
        for order in range(1, G.q + 1):
            h = G.mul(h, g)
            if h == G.e:
                orders[g] = order
                break
        else:
            raise ValueError("invalid element order")
    compatible = {
        h: frozenset(
            g for g in G.elements
            if orders[G.mul(G.inv(h), g)] > 2)
        for h in G.elements
    }
    pools = {
        order: tuple(g for g in G.elements if orders[g] == order)
        for order in (3, 6, 7)
    }
    return orders, compatible, pools


def _affine_cell(rng, G, size, palette, first_order, data):
    """Noncommuting multi-term cells without involution-separated terms."""
    if size == 1:
        return [G.pick(rng)]
    orders, compatible, pools = data
    allowed = [g for g in G.elements if orders[g] in palette]
    for _ in range(96):
        relatives = [G.e, rng.choice(pools[first_order])]
        while len(relatives) < size:
            choices = [
                g for g in allowed
                if all(g in compatible[h] for h in relatives)
            ]
            if not choices:
                break
            relatives.append(rng.choice(choices))
        if len(relatives) != size:
            continue
        if size >= 3 and not any(
                G.mul(g, h) != G.mul(h, g)
                for i, g in enumerate(relatives[1:])
                for h in relatives[1:i+1]):
            continue
        anchor = G.pick(rng)
        cell = [G.mul(anchor, g) for g in relatives]
        assert len(set(cell)) == size
        return cell
    raise ValueError("affine relative-label recipe exhausted")


def _affine_grids(rng, G, sa, sb, model, data):
    # The palettes change relative orders, not just a group automorphism.
    palette = (6, 7) if model == 0 else (3, 7)
    A = [[_affine_cell(
        rng, G, size, palette,
        (7 if j == 0 else palette[0]), data)
        for j, size in enumerate(sa[0])]]
    B = []
    for i, sizes in enumerate(sb):
        row = []
        for j, size in enumerate(sizes):
            wanted = (7, 6)[(model + i + j) % 2]
            row.append(_affine_cell(
                rng, G, size, (6, 7), wanted, data))
        B.append(row)
    return A, B


def _affine_components(hx, hz, q):
    """Read the two component actions from the verified construction."""
    mask = (1 << q) - 1
    cb = B_SHAPE[1]
    ar = [
        (row & mask) | (((row >> (cb*q)) & mask) << q)
        for row in hx[:q]
    ]
    bm = (1 << (cb*q)) - 1
    br = [row & bm for row in hz[:B_SHAPE[0]*q]]
    return (_component_columns(ar, 2*q),
            _component_columns(br, cb*q))


def _affine_short_floor(columns):
    """Exact component dependency existence through four.

    Five is only an exclusion flag, not a distance reading.
    """
    singles = set(columns)
    if 0 in singles:
        return 1
    if len(singles) != len(columns):
        return 2
    pairs = set()
    four = False
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                four = True
            pairs.add(syndrome)
    return 4 if four else 5


def _affine_kernel_probe(columns, rng, trials):
    """Find actual classical kernel witnesses; high minima are not proofs."""
    best = len(columns) + 1
    order = list(range(len(columns)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots, words = {}, []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                words.append(word)
                best = min(best, word.bit_count())

        # All reductions preserve nonzero component-kernel membership.
        short = sorted(words, key=int.bit_count)[:36]
        for i, word in enumerate(short):
            for other in short[:i]:
                combined = word ^ other
                if combined:
                    best = min(best, combined.bit_count())
        tiny = short[:10]
        for i, word in enumerate(tiny):
            for j in range(i):
                pair = word ^ tiny[j]
                for h in range(j):
                    combined = pair ^ tiny[h]
                    if combined:
                        best = min(best, combined.bit_count())
        if best <= 4:
            break
    return best


def _affine_incidence_fingerprint(hx, hz, n, k):
    """Invariant even for an uncoloured qubit/check Tanner graph.

    Unequal fingerprints rule out graph isomorphism. Equal fingerprints
    are conservatively rejected, not asserted to prove isomorphism.
    """
    rows = hx + hz
    weights = [row.bit_count() for row in rows]
    profiles = []
    for i, row in enumerate(rows):
        histogram = {}
        for j, other in enumerate(rows):
            if i == j:
                continue
            key = (weights[j], (row & other).bit_count())
            histogram[key] = histogram.get(key, 0) + 1
        profiles.append((weights[i], tuple(sorted(histogram.items()))))
    degrees = [0] * n
    for row in rows:
        while row:
            bit = row & -row
            degrees[bit.bit_length() - 1] += 1
            row ^= bit
    return (n, k, tuple(sorted(degrees)), tuple(sorted(profiles)))


@lru_cache(maxsize=1)
def _affine_design_bank(G):
    """Optimize structural cards jointly, excluding emitted orbit copies."""
    data = _affine_label_data(G)
    bank, used = [], set()

    for slot in range(N_SLOTS):
        card = slot % 20
        profile = card // 4
        shape_model, label_model = divmod(card % 4, 2)
        sa = [[4, 4] if shape_model == 0 else [3, 5]]
        sb = [[1] * 5 for _ in range(3)]
        for j, i in enumerate(_AFFINE_B_PROFILES[profile]):
            if i >= 0:
                sb[i][j] += 1
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        local = random.Random(BASE_SEED*17011 + 420150 + slot)
        winner = None

        for batch in range(2):
            pool = []
            for attempt in range(48):
                A, B = _affine_grids(
                    local, G, sa, sb, label_model, data)
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (k > 0 and 125*n <= 1000*k < 175*n):
                    continue
                ac, bc = _affine_components(hx, hz, G.q)
                fa = _affine_short_floor(ac)
                fb = _affine_short_floor(bc)
                cycles = (_component_four_cycles(ac),
                          _component_four_cycles(bc))
                key = (min(fa, fb), fa + fb,
                       -max(cycles), -sum(cycles))
                pool.append({
                    "key": key, "A": A, "B": B, "hx": hx, "hz": hz,
                    "n": n, "k": k, "ac": ac, "bc": bc,
                    "fa": fa, "fb": fb, "cycles": cycles,
                })

            finalists = sorted(
                pool, key=lambda item: item["key"], reverse=True)[:6]
            for item in finalists:
                item["da"] = (
                    item["fa"] if item["fa"] < 5 else
                    _affine_kernel_probe(item["ac"], local, 12))
                item["db"] = (
                    item["fb"] if item["fb"] < 5 else
                    _affine_kernel_probe(item["bc"], local, 12))

            def merit(item):
                d = min(item["da"], item["db"])
                return (min(item["fa"], item["fb"]),
                        item["k"]*d*d, d,
                        item["da"] + item["db"],
                        -sum(item["cycles"]))

            finalists = sorted(finalists, key=merit, reverse=True)[:3]
            for item in finalists:
                item["da"] = min(
                    item["da"],
                    _affine_kernel_probe(item["ac"], local, 40))
                item["db"] = min(
                    item["db"],
                    _affine_kernel_probe(item["bc"], local, 40))

            for item in sorted(finalists, key=merit, reverse=True):
                hx, hz, n = item["hx"], item["hz"], item["n"]
                fingerprint = _affine_incidence_fingerprint(
                    hx, hz, n, item["k"])
                if fingerprint in used:
                    continue
                if _small_logical_3(hz, hx, n):
                    continue
                if _small_logical_3(hx, hz, n):
                    continue
                score = merit(item)
                if winner is None or score > winner[0]:
                    winner = (score, item, fingerprint)
            if winner is not None:
                break

        if winner is None:
            bank.append(None)
            continue
        _, item, fingerprint = winner
        assert css_ok(item["hx"], item["hz"])
        assert max(r.bit_count() for r in item["hx"]+item["hz"]) <= W_MAX
        used.add(fingerprint)
        bank.append((item["A"], item["B"]))

    if sum(item is not None for item in bank) < min(5, N_SLOTS):
        raise ValueError("fewer than five fresh rate-0.150 designs")
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return the independently optimized filling of this indexed card."""
    bank = _affine_design_bank(G)
    if idx >= len(bank) or bank[idx] is None:
        raise ValueError("card found no fresh screened rate-0.150 filling")
    A, B = bank[idx]
    return {"idx": idx, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell without additional random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted by parent slots 4/5/6/12/13: legal weight-12 rolled fillings with monomial-rich A do not prevent sector-1 Z logical witnesses of weight 6-8.
# REFUTED: Omitted-offset and triple-template differences necessarily distinguish C9 codes: current slots 1,4,5,7,9 and many later slots were isomorphic duplicates.
# REFUTED: Odd triple upgrades remove the C9 (2,7)x(2,8) weight-four bottleneck: all five measured codes retained d30s=4 with thousands of sector-1 witnesses.
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

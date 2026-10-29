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
CONCEPT = ("S3 x C4, order 24, A=B=(1,3): six cubic cells selected against short verified component dependencies and complete CSS logical witnesses should produce five fresh codes with measured fitness above the 0.400 frontier's 25.6.")
GROUP = ("Keep the exact S3 x C4 direct-product rule and order 24. The parent measured five distance-four codes, but ten principal deletion slots were isomorphic duplicates. A shape change isolates the effect of cubic columns without attributing improvement to a different group.")
SHAPES = ("Deliberately target 0.400 instead of 0.550: (1,3)x(1,3) gives n=240, kb=96 and guaranteed rate 0.400. Equal component lengths are 72 with 24 checks each. Parent weight-four minima occupied sector-1 Acol/Bcol blocks broadly on both sides; replace both binomial-rich components. Verified ranks require 0.375<=k/n<0.425.")
FILLING = ("Twenty cards vary rotation-pair orders three/six and placements of pure-rotation triples among noncommuting triples. Every cell has three distinct terms: component row sums nine and both CSS row weights twelve. Optimize components extracted from verified lifts, refine kernel witnesses, then screen complete CSS matrices and stabilizer-tested logical witnesses. Distinct row-intersection fingerprints reject orbit copies. Probe minima are upper bounds, not distance certificates.")
FAMILY_NAME = "r040_S3xC4_13x13_cubic_ordercards_verified"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    # (a,b,c) = r^a t^b z^c:
    # r^3=t^2=z^4=1, t*r*t=r^-1, z central.
    if not valid_q(q):
        raise ValueError("S3 x C4 requires order 24")
    els = [(a, b, c) for c in range(4)
           for b in range(2) for a in range(3)]

    def red(g):
        return (g[0] % 3, g[1] % 2, g[2] % 4)

    def mul(g, h):
        return ((g[0] + (1 if g[1] == 0 else -1) * h[0]) % 3,
                (g[1] + h[1]) % 2, (g[2] + h[2]) % 4)

    def inv(g):
        return ((-g[0] if g[1] == 0 else g[0]) % 3,
                g[1], (-g[2]) % 4)

    def idx(g):
        a, b, c = red(g)
        return (2*c + b)*3 + a

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _no_kernel_le3(rows, n):
    """Exhaustive conservative test: no nonzero kernel of weight <=3.
    Rejecting even a small stabilizer is intentional here."""
    columns = [0] * n
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= bit
            row ^= low
    singles = set(columns)
    if 0 in singles or len(singles) != n:
        return False
    for i, x in enumerate(columns):
        for j in range(i):
            if (x ^ columns[j]) in singles:
                return False
    return True


from functools import lru_cache

# 0: noncommuting triple with rotation-pair order three.
# 1: noncommuting triple with rotation-pair order six.
# 2: pure-rotation triple, screened rather than presumed beneficial.
_CUBIC_SIDE_MODELS = (
    (0, 0, 0), (1, 1, 1), (0, 0, 1),
    (0, 1, 1), (0, 1, 2), (0, 0, 2),
)
_CUBIC_CARDS = tuple(
    (a, b) for a in range(6) for b in range(a, 6)
    if (a, b) != (5, 5)
)


def _cubic_columns(rows, width):
    columns = [0] * width
    for r, source in enumerate(rows):
        while source:
            low = source & -source
            columns[low.bit_length() - 1] |= 1 << r
            source ^= low
    return columns


def _cubic_basis(rows):
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old
    return basis


def _cubic_outside(v, basis):
    while v:
        p = v.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        v ^= old
    return False


def _cubic_kernel4(columns):
    """Exclude weights 1..3 and count weight-four kernel supports.

    Once nonzero distinct singles and singleton/pair collisions are
    excluded, colliding pairs are disjoint. Each quartet is counted
    by its three pair partitions. These need not be quantum logicals.
    """
    singles = set(columns)
    if 0 in singles or len(singles) != len(columns):
        return None
    pairs = {}
    collisions = 0
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            if syndrome in singles:
                return None
            old = pairs.get(syndrome, 0)
            collisions += old
            pairs[syndrome] = old + 1
    return collisions // 3


def _cubic_probe(columns, rng, trials, stabilizers=()):
    """Find actual kernel/logical witnesses; a high reading is not proof."""
    width = len(columns)
    best = width + 1
    order = list(range(width))
    basis = _cubic_basis(stabilizers)

    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        dependencies = []
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
                dependencies.append(word)

        # Preserve a genuine logical fallback before short-word reduction.
        if stabilizers:
            for word in dependencies:
                if _cubic_outside(word, basis):
                    best = min(best, word.bit_count())

        words = sorted(dependencies, key=int.bit_count)[:32]
        for _ in range(2):
            for i, word in enumerate(words):
                weight = word.bit_count()
                for j, other in enumerate(words):
                    if i == j:
                        continue
                    candidate = word ^ other
                    cw = candidate.bit_count()
                    if 0 < cw < weight:
                        word, weight = candidate, cw
                words[i] = word

        for word in words:
            if stabilizers and not _cubic_outside(word, basis):
                continue
            weight = word.bit_count()
            for _ in range(2 if stabilizers else 0):
                changed = False
                for row in stabilizers:
                    candidate = word ^ row
                    cw = candidate.bit_count()
                    if cw < weight:
                        word, weight = candidate, cw
                        changed = True
                if not changed:
                    break
            if word:
                best = min(best, weight)
    return best


def _cubic_intersections(rows, width):
    """Permutation-invariant weighted check-intersection statistics."""
    columns = _cubic_columns(rows, width)
    adjacency = [{} for _ in rows]
    for syndrome in columns:
        checks = []
        while syndrome:
            low = syndrome & -syndrome
            checks.append(low.bit_length() - 1)
            syndrome ^= low
        for t, i in enumerate(checks):
            for j in checks[:t]:
                adjacency[i][j] = adjacency[i].get(j, 0) + 1
                adjacency[j][i] = adjacency[j].get(i, 0) + 1

    histogram = {}
    cycles = 0
    edges = 0
    for i, neighbors in enumerate(adjacency):
        for j, overlap in neighbors.items():
            if j < i:
                histogram[overlap] = histogram.get(overlap, 0) + 1
                cycles += overlap * (overlap - 1) // 2
                edges += 1
    histogram[0] = len(rows) * (len(rows) - 1) // 2 - edges

    third, fourth = 0, 0
    for i, neighbors in enumerate(adjacency):
        walks = {}
        for j, ij in neighbors.items():
            for h, jh in adjacency[j].items():
                walks[h] = walks.get(h, 0) + ij * jh
        third += sum(ij * walks.get(j, 0)
                     for j, ij in neighbors.items())
        fourth += sum(x * x for x in walks.values())

    signature = (
        tuple(sorted(row.bit_count() for row in rows)),
        tuple(sorted(c.bit_count() for c in columns)),
        tuple(sorted(histogram.items())), third, fourth,
    )
    return cycles, signature


def _cubic_cell(rng, G, model):
    if model < 2:
        # Relative reflection terms have order four; the rotation
        # pair has order three or six. No relative involution occurs.
        return [G.e, (rng.choice((1, 2)), 0, 2 * model),
                (rng.randrange(3), 1, rng.choice((1, 3)))]

    rotations = [g for g in G.elements
                 if g[1] == 0 and g != G.e]
    for _ in range(96):
        x, y = rng.sample(rotations, 2)
        relatives = (x, y, G.mul(G.inv(x), y))
        if any(G.mul(g, g) == G.e for g in relatives):
            continue
        if not any(g[2] % 2 for g in (x, y)):
            continue
        return [G.e, x, y]
    raise ValueError("pure-rotation cubic recipe failed")


def _cubic_component_pool(rng, G, side, models, quota):
    """Extract components exclusively from the scaffold's verified lift."""
    q = G.q
    mask = (1 << q) - 1
    stub = [[[G.e] for _ in range(3)]]
    shortlist = []
    seen = set()

    for attempt in range(quota):
        cells = [[_cubic_cell(rng, G, model) for model in models]]
        key = tuple(tuple(sorted(cell)) for cell in cells[0])
        if key in seen:
            continue
        seen.add(key)
        A, B = (cells, stub) if side == "A" else (stub, cells)
        hx, hz, _ = lifted_matrices(G, A, B)
        if side == "A":
            rows = [
                sum(((row >> (j * 3 * q)) & mask) << (j * q)
                    for j in range(3))
                for row in hx[:q]
            ]
        else:
            rows = [row & ((1 << (3 * q)) - 1)
                    for row in hz[:q]]

        rank = gf2_rank(rows)
        if rank < q - 4:
            continue
        columns = _cubic_columns(rows, 3 * q)
        quartets = _cubic_kernel4(columns)
        if quartets is None:
            continue
        cycles, signature = _cubic_intersections(rows, 3 * q)
        score = (quartets, cycles, -rank, attempt)
        shortlist.append((score, cells, columns, signature))
        shortlist.sort(key=lambda item: item[0])
        del shortlist[12:]

    ranked = []
    for score, cells, columns, signature in shortlist:
        reading = _cubic_probe(columns, rng, 24)
        ranked.append((reading, -score[0], -score[1],
                       -score[2], cells, columns, signature))
    ranked.sort(key=lambda item: item[:4], reverse=True)

    # Rescreen finalists with more independent information sets.
    refined = []
    for item in ranked[:6]:
        reading = min(item[0], _cubic_probe(item[5], rng, 64))
        refined.append((reading,) + item[1:])
    refined.sort(key=lambda item: item[:4], reverse=True)
    return refined[:4]


@lru_cache(maxsize=1)
def _cubic_bank(G):
    """Optimize structural cards and exclude invariant-equivalent outputs."""
    assert G.q == 24
    assert _grid_weights([[3] * 3], [[3] * 3]) <= W_MAX
    fingerprints = set()
    bank = []

    for idx in range(N_SLOTS):
        card = _CUBIC_CARDS[idx % len(_CUBIC_CARDS)]
        local = random.Random(BASE_SEED * 10009 + 40024031 + idx)
        chosen = None

        for retry in range(3):
            quota = 96 + 48 * retry
            ap = _cubic_component_pool(
                local, G, "A", _CUBIC_SIDE_MODELS[card[0]], quota)
            bp = _cubic_component_pool(
                local, G, "B", _CUBIC_SIDE_MODELS[card[1]], quota)
            candidates = []

            for a in ap:
                for b in bp:
                    A, B = a[4], b[4]
                    hx, hz, n = lifted_matrices(G, A, B)
                    rx, rz = gf2_rank(hx), gf2_rank(hz)
                    k = n - rx - rz
                    if not (375 * n <= 1000 * k < 425 * n):
                        continue
                    cx = _cubic_columns(hx, n)
                    cz = _cubic_columns(hz, n)
                    qx, qz = _cubic_kernel4(cx), _cubic_kernel4(cz)
                    if qx is None or qz is None:
                        continue
                    component_d = min(a[0], b[0])
                    score = (qx + qz == 0, component_d,
                             -(qx + qz), k, a[2] + b[2])
                    candidates.append(
                        (score, A, B, hx, hz, n, k, cx, cz, rx, rz))

            candidates.sort(key=lambda item: item[0], reverse=True)
            finalists = []
            for candidate in candidates[:6]:
                score, A, B, hx, hz, n, k, cx, cz, rx, rz = candidate
                dx = _cubic_probe(cz, local, 16, stabilizers=hx)
                dz = _cubic_probe(cx, local, 16, stabilizers=hz)
                d = min(score[1], dx, dz)
                merit = (k * d * d / n, d, score[0],
                         score[2], min(dx, dz))
                finalists.append((merit, candidate))

            finalists.sort(key=lambda item: item[0], reverse=True)
            for merit, candidate in finalists:
                _, A, B, hx, hz, n, k, cx, cz, rx, rz = candidate
                fingerprint = (
                    k, tuple(sorted((rx, rz))),
                    tuple(sorted((
                        _cubic_intersections(hx, n)[1],
                        _cubic_intersections(hz, n)[1]))))
                if fingerprint in fingerprints:
                    continue
                assert css_ok(hx, hz)
                assert max(row.bit_count() for row in hx + hz) <= W_MAX
                fingerprints.add(fingerprint)
                chosen = (
                    tuple(tuple(tuple(cell) for cell in row) for row in A),
                    tuple(tuple(tuple(cell) for cell in row) for row in B),
                    k, merit)
                break
            if chosen is not None:
                break

        if chosen is None:
            raise ValueError("no fresh invariant-distinct cubic card design")
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return the optimized indexed design without applying a relabeling."""
    A, B, k, merit = _cubic_bank(G)[idx]
    return {"idx": idx, "cells": {"A": A, "B": B},
            "k": k, "witness_merit": merit}


def CELL(rng, G, side, i, j, ctx):
    """Return the stored full cell without additional random labeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Uniform legal support rolling on C17, (2,5)x(2,6), did not suppress sector-1 weight-five logicals: all five measured parent codes had d30s=5.
# REFUTED: Fifteen odd-central deletion prescriptions provide fifteen inequivalent improvements: ten principal parent slots were isomorphic duplicates, and all five measured survivors had d30s=4 with sector-1 minima.
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

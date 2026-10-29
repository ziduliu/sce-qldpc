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
CONCEPT = ("S3 x C4, order 24, shapes (1,4)x(1,5): witness-guided joint support and relative-label repair should find unarchived codes with no weight<=3 logicals and k>297, improving the parent's distance-four fitness.")
GROUP = ("Frozen L1 group and order. This bed measured five [504,297,4] codes. The related central-pair experiment scored 5.1964 with measured distance three, so useful rank deficits must be selected only after removing actual short logicals.")
SHAPES = ("Frozen (1,4)x(1,5): n=504, kb=288, guaranteed rate 4/7 in open-ended cell 0.550. Component lengths are 96 and 120 with 24 checks each; the size window permits q<=28 but this round retains q=24. Parent minima occupied sector-1 Acol/Bcol blocks broadly across both components; screen the complete CSS matrices, including sector two.")
FILLING = ("Twenty cards cross five support architectures with four relative-element-order preferences. Exact stabilizer-quotient counts through weight three guide legal term transfers and label replacements; eliminate witnesses before rewarding k. Finalists receive an exact weight-four logical census. Invariant fingerprints exclude the parent's archived deletion codes and separate selected outputs. Screening certificates are limited to the enumerated weights; longer distance readings remain evaluator upper bounds.")
FAMILY_NAME = "r055_S3xC4_14x15_witness_cutwalk_v2"

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

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (1, 5)               # (rb, cb)


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
from collections import Counter

_CUT_PROFILES = (
    ((2, 2, 2, 2), (2, 2, 2, 2, 2)),
    ((2, 2, 2, 3), (1, 2, 2, 2, 2)),
    ((1, 2, 2, 2), (1, 2, 2, 2, 3)),
    ((1, 2, 2, 3), (1, 2, 2, 2, 2)),
    ((1, 2, 3, 3), (1, 2, 2, 2, 2)),
)


def _cut_freeze(A, B):
    return tuple(tuple(tuple(tuple(cell) for cell in row)
                       for row in grid) for grid in (A, B))


def _cut_columns_quotient(checks, stabilizers, n):
    """Use only verified rows; compute linear stabilizer normal forms."""
    basis = {}
    for source in stabilizers:
        row = source
        while row:
            p = row.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old

    # Every pivot row contains only lower-index terms besides its pivot.
    # Ascending recursion avoids reducing n unit vectors independently.
    residues = []
    for c in range(n):
        row = basis.get(c)
        if row is None:
            residues.append(1 << c)
            continue
        tail = row ^ (1 << c)
        residue = 0
        while tail:
            bit = tail & -tail
            residue ^= residues[bit.bit_length() - 1]
            tail ^= bit
        residues.append(residue)

    columns = [0] * n
    for r, source in enumerate(checks):
        row = source
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns, residues


def _cut_three(columns, residues):
    """Exact numbers of logical supports of weights one, two and three."""
    locations = {}
    one = 0
    for i, syndrome in enumerate(columns):
        locations.setdefault(syndrome, []).append(i)
        one += syndrome == 0 and residues[i] != 0

    two = three = 0
    for i, si in enumerate(columns):
        ri = residues[i]
        for j in range(i):
            syndrome = si ^ columns[j]
            residue = ri ^ residues[j]
            if syndrome == 0 and residue:
                two += 1
            # t<j<i counts each disjoint triple exactly once.
            for t in locations.get(syndrome, ()):
                if t >= j:
                    break
                if residue ^ residues[t]:
                    three += 1
    return one, two, three


def _cut_pair_profile(columns, residues, lows):
    """Exact quartet count when lighter logicals are absent.

    Different stabilizer residues at equal pair syndrome give a logical.
    With no weight-two logical, such pairs cannot overlap. Each logical
    quartet has three unordered pair partitions.
    """
    totals = {}
    classes = {}
    collisions = 0
    for i, si in enumerate(columns):
        ri = residues[i]
        for j in range(i):
            syndrome = si ^ columns[j]
            residue = ri ^ residues[j]
            key = syndrome, residue
            total = totals.get(syndrome, 0)
            equal = classes.get(key, 0)
            collisions += total - equal
            totals[syndrome] = total + 1
            classes[key] = equal + 1

    histogram = Counter((s.bit_count(), count)
                        for s, count in totals.items())
    quotient_histogram = Counter(classes.values())
    quartets = collisions // 3 if not any(lows) else -1
    if quartets >= 0:
        assert collisions % 3 == 0
    return (quartets, tuple(sorted(histogram.items())),
            tuple(sorted(quotient_histogram.items())))


def _cut_evaluate(G, A, B):
    hx, hz, n = lifted_matrices(G, A, B)
    rx, rz = gf2_rank(hx), gf2_rank(hz)
    cx, ux = _cut_columns_quotient(hz, hx, n)
    cz, uz = _cut_columns_quotient(hx, hz, n)
    lx, lz = _cut_three(cx, ux), _cut_three(cz, uz)
    counts = tuple(x + z for x, z in zip(lx, lz))
    floor = next((w for w, count in enumerate(counts, 1) if count), 4)
    k = n - rx - rz
    # Unlike rank-first random selection, eliminate the actual witnesses
    # before rewarding extra encoded qubits.
    score = (floor, -counts[0], -counts[1], -counts[2], k)
    return dict(A=A, B=B, hx=hx, hz=hz, n=n, rx=rx, rz=rz,
                k=k, cx=cx, cz=cz, ux=ux, uz=uz,
                lx=lx, lz=lz, score=score)


def _cut_finish(candidate):
    px = _cut_pair_profile(candidate["cx"], candidate["ux"],
                           candidate["lx"])
    pz = _cut_pair_profile(candidate["cz"], candidate["uz"],
                           candidate["lz"])
    hx, hz = candidate["hx"], candidate["hz"]
    types = tuple(sorted((
        (candidate["rz"], tuple(sorted(r.bit_count() for r in hz)),
         candidate["lx"], px),
        (candidate["rx"], tuple(sorted(r.bit_count() for r in hx)),
         candidate["lz"], pz),
    )))
    degrees = tuple(sorted(zip(
        (c.bit_count() for c in candidate["cx"]),
        (c.bit_count() for c in candidate["cz"]))))
    swapped = tuple(sorted((b, a) for a, b in degrees))
    fingerprint = (candidate["n"], candidate["k"], types,
                   min(degrees, swapped))

    floor = candidate["score"][0]
    if floor == 4 and px[0] == 0 and pz[0] == 0:
        floor = 5
    quartets = max(0, px[0]) + max(0, pz[0])
    if floor >= 4:
        score = (floor, candidate["k"], -quartets)
    else:
        score = candidate["score"]
    return fingerprint, score


@lru_cache(maxsize=1)
def _cut_bank(G):
    """Optimize deliberate cards once; retain invariant-distinct outputs."""
    def order(g):
        h = G.e
        for t in range(1, G.q + 1):
            h = G.mul(h, g)
            if h == G.e:
                return t
        raise ValueError("invalid group element order")

    orders = {g: order(g) for g in G.elements}
    noninvolutory = [g for g in G.elements if orders[g] > 2]
    preferred = (
        [g for g in noninvolutory if g[2] % 2],
        [g for g in noninvolutory if orders[g] == 6],
        [g for g in noninvolutory if orders[g] == 12],
        noninvolutory,
    )

    # Blacklist measured parent constructions by matrix invariants, not
    # just by their literal labels. These matrices are never submitted.
    archived = set()
    generators = [(a, b, 1) for b in range(2) for a in range(3)]
    for missing_a in ((0, 3), (0, 1), (3, 4), (3, 1), (1, 2)):
        for missing_b in (0, 3, 1):
            A = [[[G.e, g] for t, g in enumerate(generators)
                  if t not in missing_a]]
            B = [[[G.e, g] for t, g in enumerate(generators)
                  if t != missing_b]]
            archived.add(_cut_finish(_cut_evaluate(G, A, B))[0])

    def initial(rng, counts, mode):
        row = []
        used = set()
        for count in counts:
            cell = [G.e]
            if count >= 2:
                legal = [g for g in preferred[mode]
                         if g not in used and G.inv(g) not in used]
                if not legal:
                    legal = [g for g in noninvolutory
                             if g not in used and G.inv(g) not in used]
                g = rng.choice(legal or noninvolutory)
                cell.append(g)
                used.update((g, G.inv(g)))
            while len(cell) < count:
                cell.append(rng.choice([g for g in G.elements
                                        if g not in cell]))
            row.append(cell)
        return [row]

    def mutate(rng, A, B, mode):
        A = [[list(cell) for cell in row] for row in A]
        B = [[list(cell) for cell in row] for row in B]
        cells = A[0] + B[0]
        if rng.random() < 0.35:
            # A coupled count move, rather than an illegal independent
            # addition. The verified budgets are checked before evaluation.
            donors = [c for c in cells if len(c) > 1]
            recipients = [c for c in cells if len(c) < CELL_MAX]
            donor, recipient = rng.choice(donors), rng.choice(recipients)
            if donor is recipient:
                return None
            donor.pop(rng.randrange(1, len(donor)))
            recipient.append(rng.choice([g for g in G.elements
                                         if g not in recipient]))
        else:
            cell = rng.choice([c for c in cells if len(c) > 1])
            t = rng.randrange(1, len(cell))
            pool = (preferred[mode] if len(cell) == 2
                    and rng.random() < 0.8 else G.elements)
            legal = [g for g in pool if g not in cell]
            if not legal:
                legal = [g for g in G.elements if g not in cell]
            cell[t] = rng.choice(legal)
        sa, sb = [[len(c) for c in A[0]]], [[len(c) for c in B[0]]]
        if _grid_weights(sa, sb) > W_MAX:
            return None
        return A, B

    keep = {}
    cards = max(20, N_SLOTS)
    for card in range(cards):
        profile, mode = divmod(card % 20, 4)
        ac, bc = _CUT_PROFILES[profile]
        rng = random.Random(BASE_SEED * 10009 + 5502429 + card)
        pool = {}
        evaluated = set()

        def remember(candidate):
            key = _cut_freeze(candidate["A"], candidate["B"])
            old = pool.get(key)
            if old is None or candidate["score"] > old["score"]:
                pool[key] = candidate
            if len(pool) > 18:
                worst = min(pool, key=lambda key: pool[key]["score"])
                del pool[worst]

        for restart in range(4):
            current = _cut_evaluate(
                G, initial(rng, ac, mode), initial(rng, bc, mode))
            remember(current)
            for step in range(18):
                proposal = mutate(rng, current["A"], current["B"], mode)
                if proposal is None:
                    continue
                key = _cut_freeze(*proposal)
                if key in evaluated:
                    continue
                evaluated.add(key)
                trial = _cut_evaluate(G, *proposal)
                remember(trial)
                better = trial["score"] >= current["score"]
                # Permit small sideways excursions, but never deliberately
                # trade away the current absence of lighter logicals.
                sideways = (
                    trial["score"][0] == current["score"][0]
                    and rng.random() < 0.16)
                if better or sideways:
                    current = trial

        for candidate in sorted(pool.values(),
                                key=lambda c: c["score"], reverse=True):
            fingerprint, score = _cut_finish(candidate)
            if fingerprint in archived:
                continue
            previous = keep.get(fingerprint)
            if previous is None or score > previous[0]:
                A, B = _cut_freeze(candidate["A"], candidate["B"])
                keep[fingerprint] = (score, A, B)

    selected = sorted(keep.values(), key=lambda item: item[0],
                      reverse=True)[:N_SLOTS]
    if len(selected) < min(5, N_SLOTS):
        raise ValueError("fewer than five new invariant-distinct designs")

    bank = []
    for score, A, B in selected:
        hx, hz, n = lifted_matrices(G, A, B)
        assert css_ok(hx, hz)
        assert n - gf2_rank(hx) - gf2_rank(hz) >= 288
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        assert all(1 <= len(cell) <= CELL_MAX
                   and len(set(cell)) == len(cell)
                   for grid in (A, B) for row in grid for cell in row)
        bank.append((A, B, score))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign a jointly optimized code, never an orbit-copy fallback."""
    bank = _cut_bank(G)
    if idx >= len(bank):
        raise ValueError("no further invariant-distinct design")
    A, B, score = bank[idx]
    return {"idx": idx, "cells": {"A": A, "B": B},
            "selection": score}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without changing the screening."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Uniform legal support rolling on C17, (2,5)x(2,6), did not suppress sector-1 weight-five logicals: all five measured parent codes had d30s=5.
# REFUTED: Fifteen prescribed odd-central deletion slots provide fifteen inequivalent codes: ten principal parent slots were isomorphic duplicates; all five measured survivors had d30s=4.
# REFUTED: Central-involution pairing plus rank-first quotient selection improves this bed's 9.429 fitness: the shown paired-support program scored 5.1964, with all five measured minima at weight three.
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

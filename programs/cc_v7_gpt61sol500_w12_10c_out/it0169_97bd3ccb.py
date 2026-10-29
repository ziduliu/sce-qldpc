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
CONCEPT = ("On D7 x C3, order 42, A=(1,2), B=(3,5), rank-controlled degree-four constituents selected against short dependencies should produce five measured distance readings of at least sixteen in cell 0.150.")
GROUP = ("Jump from the order-55 Frobenius presentation to the direct product D7 x C3. This order-42 centre-three bed is absent from the rendered BEDS list. Rotation, central and mixed coordinates supply relative orders 7,21,6; any distance advantage is a hypothesis, not a consequence of noncommutativity.")
SHAPES = ("Target cell 0.150: n=546, kb=84 and guaranteed rate 2/13. Components have variable/check counts 84/42 and 210/126. Require A rank 41 and B rank at least 124, then verify full CSS k and the target rate. Parent measured weight-eight minima occupied sector-one Acol/Bcol blocks; replace its cubic three-column constituents with two degree-four A columns and five B columns spread over three check rows.")
FILLING = ("A counts are (4,4). Each B column has one double cell and two singleton cells, with B row sums at most eight: HX has weight 8+4=12 and HZ has weight at most 8+4=12. Twenty cards cross four A quotient-occupancy recipes with five B support/relative-order policies. Compile component contributions only from verified unit-cell lifts, optimize actual kernel witnesses, and screen full logicals through four modulo stabilizers. A deterministic bank rejects equal uncoloured union-Tanner fingerprints. Component and logical probes remain witness upper bounds, never distance certificates.")
FAMILY_NAME = "r015_D7xC3_12x35_degree4_rankcards_distinct_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    # Direct product D7 x C3:
    # (a,b,c)=x^a t^b z^c, x^7=t^2=z^3=1,
    # t*x*t=x^-1, and z commutes with x and t.
    if not valid_q(q):
        raise ValueError("D7 x C3 requires order 42")
    els = [(a, b, c) for c in range(3)
           for b in range(2) for a in range(7)]

    def red(g):
        return (g[0] % 7, g[1] % 2, g[2] % 3)

    def mul(g, h):
        a, b, c = red(g)
        d, e, f = red(h)
        return ((a + (d if b == 0 else -d)) % 7,
                (b + e) % 2, (c + f) % 3)

    def inv(g):
        a, b, c = red(g)
        return ((-a if b == 0 else a) % 7, b, (-c) % 3)

    def idx(g):
        a, b, c = red(g)
        return (2*c + b)*7 + a

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)

from functools import lru_cache

# Quotient coordinates (reflection parity, central C3 coordinate).
# Repeated quotient positions receive distinct rotation coordinates.
# Cross-parity pairs have different central coordinates, avoiding
# involutory relative elements inside these four-term cells.
_D42_MODELS = (
    ((0, 0), (0, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 2), (1, 2)),
    ((0, 0), (0, 0), (0, 1), (1, 2)),
    ((0, 0), (1, 1), (1, 1), (1, 2)),
)
_D42_A_CARDS = ((2, 2), (2, 3), (0, 2), (1, 2))

# Double-cell row assignments and prescribed relative element orders.
# Row sums are respectively (8,7,5), (8,6,6), or (7,7,6).
_D42_B_CARDS = (
    ((0, 0, 0, 1, 1), 7),
    ((0, 0, 0, 1, 2), 21),
    ((0, 0, 1, 1, 2), 6),
    ((0, 0, 1, 1, 2), 21),
    ((0, 0, 0, 1, 2), 6),
)


def _f55_columns(rows, width):
    """Transpose only matrices obtained from the verified constructor."""
    cols = [0] * width
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            cols[low.bit_length() - 1] |= bit
            row ^= low
    return cols


def _d42_basis(rows):
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


def _d42_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _d42_small(cols, basis=None):
    """Exact existence test through weight four, without parity assumptions.

    With basis=None, detect any nonzero classical kernel word.
    Otherwise detect only words outside the actual stabilizer row space.
    Singleton/pair collisions cover every support of weight <=4.
    """
    seen = {0: 0}

    def insert(syndrome, word):
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = word
            return False
        difference = old ^ word
        return bool(difference) and (
            basis is None or _d42_outside(difference, basis))

    for i, syndrome in enumerate(cols):
        if insert(syndrome, 1 << i):
            return True
    for i, syndrome in enumerate(cols):
        bit = 1 << i
        for j in range(i):
            if insert(syndrome ^ cols[j], bit | (1 << j)):
                return True
    return False


def _f55_cycles(cols):
    """Count four-cycles through repeated check-neighbor pairs."""
    seen = {}
    total = 0
    for col in cols:
        neighbors = []
        while col:
            low = col & -col
            neighbors.append(low.bit_length() - 1)
            col ^= low
        for i, a in enumerate(neighbors):
            for b in neighbors[:i]:
                pair = (b, a)
                old = seen.get(pair, 0)
                total += old
                seen[pair] = old + 1
    return total


@lru_cache(maxsize=1)
def _d42_atlas(G):
    """Compile both component actions exclusively from verified unit lifts.

    Subsequent XOR and block placement are linear superposition of these
    verified contributions, not a separate regular-representation lift.
    """
    q = G.q
    mask = (1 << q) - 1
    tables = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            A = [[[G.e] for _ in range(2)]]
            B = [[[G.e] for _ in range(5)] for _ in range(3)]
            if side == "A":
                A = [[[g], []]]
            else:
                B = [[[] for _ in range(5)] for _ in range(3)]
                B[0][0] = [g]
            hx, hz, _ = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            rows = [row & mask for row in source[:q]]
            blocks.append(tuple(_f55_columns(rows, q)))
        tables[side] = tuple(blocks)
    return tables


def _d42_component(G, M, side):
    """Place verified unit contributions into the constituent check rows."""
    q = G.q
    atlas = _d42_atlas(G)[side]
    cols = [0] * (len(M[0]) * q)
    for i, row in enumerate(M):
        for j, cell in enumerate(row):
            for g in cell:
                contribution = atlas[G.idx(g)]
                for h, syndrome in enumerate(contribution):
                    cols[j*q + h] ^= syndrome << (i*q)
    return cols


def _d42_connected(cols, height):
    """Connectivity of a constituent's verified Tanner incidence."""
    parent = list(range(height))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for syndrome in cols:
        if not syndrome:
            return False
        first = (syndrome & -syndrome).bit_length() - 1
        syndrome &= syndrome - 1
        while syndrome:
            bit = syndrome & -syndrome
            other = bit.bit_length() - 1
            parent[root(other)] = root(first)
            syndrome ^= bit
    return len({root(a) for a in range(height)}) == 1


def _f55_probe(rng, cols, rounds, stabilizers=()):
    """Actual kernel/logical witnesses with dependency and span reduction.

    Every generated word has zero syndrome in the supplied verified matrix.
    Stabilizer-tested readings count only genuine nontrivial logicals.
    A no-witness sentinel is never a distance certificate.
    """
    best = len(cols) + 1
    basis = _d42_basis(stabilizers)
    generators = list(stabilizers)
    memory = []

    def record(word):
        nonlocal best
        if (word and word.bit_count() < best
                and (not generators or _d42_outside(word, basis))):
            best = word.bit_count()

    for _ in range(rounds):
        order = list(range(len(cols)))
        rng.shuffle(order)
        pivots, dependencies = {}, []
        for j in order:
            syndrome, word = cols[j], 1 << j
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
                record(word)

        words = sorted(set(dependencies + memory),
                       key=int.bit_count)[:48]
        for sweep in range(3):
            changed = False
            rng.shuffle(words)
            for i, word in enumerate(words):
                weight = word.bit_count()
                for j, other in enumerate(words):
                    if i == j:
                        continue
                    combined = word ^ other
                    new_weight = combined.bit_count()
                    if 0 < new_weight < weight:
                        word, weight = combined, new_weight
                        changed = True
                words[i] = word
                record(word)
            if not changed:
                break

        discoveries = set(words)
        for i, word in enumerate(words):
            for other in words[:i]:
                combined = word ^ other
                record(combined)
                if combined and combined.bit_count() <= best + 4:
                    discoveries.add(combined)
        small = sorted(discoveries, key=int.bit_count)[:12]
        for i, word in enumerate(small):
            for j in range(i):
                pair = word ^ small[j]
                for t in range(j):
                    record(pair ^ small[t])

        if generators:
            for word in sorted(discoveries, key=int.bit_count)[:24]:
                if not _d42_outside(word, basis):
                    continue
                weight = word.bit_count()
                for sweep in range(3):
                    changed = False
                    for row in generators:
                        combined = word ^ row
                        new_weight = combined.bit_count()
                        if new_weight < weight:
                            word, weight = combined, new_weight
                            changed = True
                    if not changed:
                        break
                record(word)
                discoveries.add(word)

        memory = sorted(discoveries, key=int.bit_count)[:48]
        if best <= 4:
            break
    return best


def _d42_draw(rng, G, side, idx):
    """Fill a prescribed card; never transport a previously selected code."""
    card, lane = idx % 20, idx // 20
    if side == "A":
        models = _D42_A_CARDS[card // 5]
        row = []
        for model in models:
            positions = _D42_MODELS[(model + lane) % len(_D42_MODELS)]
            used, cell = set(), []
            for b, c in positions:
                legal = [a for a in range(7) if (a, b, c) not in used]
                g = (rng.choice(legal), b, c)
                used.add(g)
                cell.append(g)
            row.append(cell)
        return [row]

    doubles, order = _D42_B_CARDS[card % 5]
    orders = [order] * 5
    # Extra harness lanes change individual relative-order constraints,
    # not just random labels or a common group automorphism.
    palette = (7, 21, 6)
    for t in range(min(lane, 5)):
        j = (card + 2*t) % 5
        orders[j] = palette[(palette.index(order) + 1 + t) % 3]
    pools = {
        7: [g for g in G.elements
            if g[1] == 0 and g[2] == 0 and g[0] != 0],
        21: [g for g in G.elements
             if g[1] == 0 and g[2] != 0 and g[0] != 0],
        6: [g for g in G.elements if g[1] == 1 and g[2] != 0],
    }
    M = [[[G.pick(rng)] for _ in range(5)] for _ in range(3)]
    for j, i in enumerate(doubles):
        first = M[i][j][0]
        second = G.mul(first, rng.choice(pools[orders[j]]))
        assert first != second
        M[i][j].append(second)
    return M


def _d42_pool(rng, G, side, idx, quota):
    """Rank-controlled component search using verified atlas syndromes."""
    raw, seen = [], set()
    height = G.q if side == "A" else 3*G.q
    for attempt in range(quota):
        M = _d42_draw(rng, G, side, idx)
        label_key = tuple(tuple(sorted(cell)) for row in M for cell in row)
        if label_key in seen:
            continue
        seen.add(label_key)
        cols = _d42_component(G, M, side)
        rank = gf2_rank(cols)
        if side == "A":
            if rank != G.q - 1:
                continue
        elif rank < 3*G.q - 2:
            continue
        if not _d42_connected(cols, height):
            continue
        cycles = _f55_cycles(cols)
        short = _d42_small(cols)
        raw.append((int(short), cycles, attempt, M, cols, rank))
        raw.sort(key=lambda item: item[:3])
        del raw[8:]

    scored = []
    for short, cycles, attempt, M, cols, rank in raw:
        reading = _f55_probe(rng, cols, 12 if side == "A" else 18)
        if reading > len(cols):
            continue
        key = (int(not short), reading, -cycles, rank, -attempt)
        scored.append((key, M, cols))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:4]


def _d42_signature(hx, hz, n):
    """Invariant of the UNCOLOURED union-Tanner graph.

    Unequal fingerprints certify nonisomorphism. Equal fingerprints are
    conservatively rejected; equality is not an isomorphism certificate.
    No X/Z colouring or constituent/block labels enter the fingerprint.
    """
    from collections import Counter
    from hashlib import blake2b

    rows = hx + hz
    weights = [row.bit_count() for row in rows]
    profiles = [Counter() for _ in rows]
    overlaps = Counter()
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            overlaps[overlap] += 1
            if overlap:
                profiles[i][(weights[j], overlap)] += 1
                profiles[j][(weights[i], overlap)] += 1
    labels = Counter(
        (weights[i], tuple(sorted(profile.items())))
        for i, profile in enumerate(profiles))
    cols = _f55_columns(rows, n)
    fingerprint = (
        tuple(sorted(Counter(weights).items())),
        tuple(sorted(Counter(c.bit_count() for c in cols).items())),
        tuple(sorted(overlaps.items())),
        tuple(sorted(labels.items())),
    )
    return blake2b(repr(fingerprint).encode(), digest_size=24).digest()


def _d42_extract(hx, hz, q):
    """Independent extraction from the final VERIFIED complete LP lift."""
    mask = (1 << q) - 1
    ar = [
        (row & mask) | (((row >> (5*q)) & mask) << q)
        for row in hx[:q]
    ]
    br = [row & ((1 << (5*q)) - 1) for row in hz[:3*q]]
    return _f55_columns(ar, 2*q), _f55_columns(br, 5*q)


def _d42_optimise(rng, G, idx, used, quota):
    ap = _d42_pool(rng, G, "A", idx, quota)
    bp = _d42_pool(rng, G, "B", idx, quota)
    pairs = []
    for ai, a in enumerate(ap):
        for bi, b in enumerate(bp):
            ka, kb = a[0], b[0]
            key = (min(ka[0], kb[0]), min(ka[1], kb[1]),
                   ka[1] + kb[1], ka[2] + kb[2], -ai, -bi)
            pairs.append((key, a, b))
    pairs.sort(key=lambda item: item[0], reverse=True)

    finalists = []
    for component_key, a, b in pairs:
        A, B = a[1], b[1]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _grid_weights(sa, sb) <= 12
        assert _no_dead_lines(sa, sb)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 546 or not (125*n <= 1000*k < 175*n):
            continue
        ac, bc = _d42_extract(hx, hz, G.q)
        assert ac == a[2] and bc == b[2], "verified atlas mismatch"
        xc, zc = _f55_columns(hz, n), _f55_columns(hx, n)
        if _d42_small(xc, _d42_basis(hx)):
            continue
        if _d42_small(zc, _d42_basis(hz)):
            continue
        signature = _d42_signature(hx, hz, n)
        if signature in used:
            continue

        # Include both LP sectors and the actual stabilizer quotient.
        dx = _f55_probe(rng, xc, 6, stabilizers=hx)
        dz = _f55_probe(rng, zc, 6, stabilizers=hz)
        d = min(a[0][1], b[0][1], dx, dz)
        score = (k*d*d, d, min(dx, dz),
                 a[0][1] + b[0][1], a[0][2] + b[0][2])
        finalists.append((score, A, B, hx, hz, k, signature))
        if len(finalists) >= 3:
            break

    if not finalists:
        return None
    score, A, B, hx, hz, k, signature = max(
        finalists, key=lambda item: item[0])
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= 12
    assert all(len(cell) <= 5 and len(set(cell)) == len(cell)
               for M in (A, B) for row in M for cell in row)
    return {"idx": idx, "cells": {"A": A, "B": B}, "k": k,
            "selection": score, "fingerprint": signature}


@lru_cache(maxsize=1)
def _d42_bank(G):
    """Pure deterministic bank of independently optimized card designs."""
    assert G.q == 42
    assert A_SHAPE == (1, 2) and B_SHAPE == (3, 5)
    used, bank = set(), []
    for idx in range(N_SLOTS):
        local = random.Random(BASE_SEED*10007 + 420150731 + 7919*idx)
        chosen = None
        for retry in range(3):
            chosen = _d42_optimise(
                local, G, idx, used, quota=96 + 48*retry)
            if chosen is not None:
                break
        if chosen is not None:
            used.add(chosen["fingerprint"])
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return this card's fresh target-rate, invariant-distinct design."""
    chosen = _d42_bank(G)[idx]
    if chosen is None:
        raise ValueError("no fresh target-rate design passed this card")
    return chosen


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without further random filling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports with uniform dihedral labels reliably suppress short logicals on (1,3)x(1,3); many parent slots found d2s=2 and all five measured parent codes retained d30s=4.
# REFUTED: Changing to C11 semidirect C5 and screening cubic component weight-four/six kernels suffices to exceed distance eight; all five measured current-parent codes returned d30s=8 with sector-one minima.
# REFUTED: Action-population and rotation-gap cards alone prevent equivalent emissions; current-parent slots 12,13,43 were collapsed as isomorphic copies.
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

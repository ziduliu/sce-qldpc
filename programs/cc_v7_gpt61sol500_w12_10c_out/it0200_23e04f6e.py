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
CONCEPT = ("AGL(1,7), the affine matrix group of order 42, shapes (1,3)x(1,4): compatible singular trinomials and long-order binomial voltages should produce five fresh k=270 codes with measured distance readings at least six; exact logical-four-screened designs are the reserve.")
GROUP = ("Jump from D18 to affine matrices over F7. AGL(1,7) has an order-seven translation subgroup, an order-six multiplicative complement, centre one and seven conjugacy classes. RUN STATE records 7.7 on the related order-42 centre-one/seven-class bed at these shapes. Challenge that reading with verified singularity and logical screening, not an assumed group advantage.")
SHAPES = ("Target cell 0.500 with (1,3)x(1,4): n=546, kb=252 and guaranteed rate 6/13. Components have 126/168 variables and 42 checks each. Component ranks 36 and 42 target verified k=270, rate 0.4945; q=42 fits the shape's q<=46 window. The parent's weight-four minima occupied sector-one Acol/Bcol blocks broadly, including its monomial Acol3 anchor. Remove that anchor and its degree-two B architecture.")
FILLING = ("A counts [3,3,3] and B counts [2,2,2,3] give both check-weight maxima twelve. Compile component actions only from verified unit-cell lifts. Assemble clean A triples by containment in a rank-36 column space. Twenty cards cross four rotation-voltage ratios with five cubic quotient occupancies. Prefer component exclusion through five and exact stabilizer-aware full exclusion through five, using odd-column parity and an exhaustive mixed-sector triple test. Fresh exact logical-four-screened designs are explicit reserves. Verified k must equal 270; uncoloured incidence invariants reject repeated outputs. Search misses never certify a distance above the exact exclusion threshold.")
FAMILY_NAME = "r050_AGL1F7_13x14_singular_cliques_longvoltage_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    # Affine matrices [[u,v],[0,1]] over F7, u nonzero.
    # Multiplication is matrix composition, not the parent's rule.
    if not valid_q(q):
        raise ValueError("AGL(1,7) has order 42")
    els = [(u, v) for u in range(1, 7) for v in range(7)]

    def red(g):
        u, v = g[0] % 7, g[1] % 7
        if u == 0:
            raise ValueError("affine multiplier must be nonzero")
        return (u, v)

    def mul(g, h):
        u, v = red(g)
        s, t = red(h)
        return ((u * s) % 7, (v + u * t) % 7)

    def inv(g):
        u, v = red(g)
        ui = pow(u, -1, 7)
        return (ui, (-ui * v) % 7)

    def idx(g):
        u, v = red(g)
        return 7 * (u - 1) + v

    return dict(elements=els, ident=(1, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


from functools import lru_cache


def _inc_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= marker
            row ^= bit
    return columns


def _inc_profile(columns, detailed=False):
    """Exactly exclude kernel words through three; census weight four.

    Once singleton syndromes are distinct and nonzero, equal pair
    syndromes have disjoint supports. Every quartet contributes three
    pair collisions. Kernel words need not be logical operators.
    """
    singles = set(columns)
    if 0 in singles or len(singles) != len(columns):
        return None
    pairs = {}
    collisions = 0
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome in singles:
                return None
            old = pairs.get(syndrome, 0)
            collisions += old
            pairs[syndrome] = old + 1
    if not detailed:
        return collisions // 3
    histogram = {}
    for syndrome, count in pairs.items():
        key = (count, syndrome.bit_count())
        histogram[key] = histogram.get(key, 0) + 1
    return (collisions // 3, tuple(sorted(histogram.items())),
            tuple(c.bit_count() for c in columns))


def _jump_span(words):
    """Canonical binary span of columns from verified component lifts."""
    basis = {}
    for word in words:
        while word:
            p = word.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = word
                break
            word ^= old
    for p in sorted(basis):
        for other in sorted(basis):
            if other > p and ((basis[other] >> p) & 1):
                basis[other] ^= basis[p]
    return tuple(basis[p] for p in sorted(basis))


def _jump_rows(cols, checks):
    rows = [0] * checks
    for c, syndrome in enumerate(cols):
        while syndrome:
            bit = syndrome & -syndrome
            rows[bit.bit_length() - 1] |= 1 << c
            syndrome ^= bit
    return rows


def _jump_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            size = (row & other).bit_count()
            total += size * (size - 1) // 2
    return total


def _jump_six(cols):
    """Exact component exclusion through five, then a weight-six census.

    Called only after exclusion through four. Any triple collision with
    a support of size <=2 reveals a kernel word of weight <=5. Otherwise
    equal triples are disjoint; each six-word has ten triple partitions.
    """
    low = {0}
    low.update(cols)
    for i, x in enumerate(cols):
        for y in cols[:i]:
            low.add(x ^ y)
    triples, hits = {}, 0
    for i in range(2, len(cols)):
        x = cols[i]
        for j in range(1, i):
            xy = x ^ cols[j]
            for h in range(j):
                syndrome = xy ^ cols[h]
                if syndrome in low:
                    return None
                old = triples.get(syndrome, 0)
                hits += old
                triples[syndrome] = old + 1
    return hits // 10


def _jump_small_logical(checks, stabilizers, n, start, through_five):
    """Exact logical test through four, optionally mixed-sector five.

    All supports of size <=2 are compared modulo the stabilizer span.
    For the fifth-weight extension, enumerate every triple touching
    sector two. The caller separately proves that sector-one kernels
    through five are absent using the verified B component.
    """
    basis = {}
    for word in stabilizers:
        while word:
            p = word.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = word
                break
            word ^= old

    def outside(word):
        while word:
            p = word.bit_length() - 1
            old = basis.get(p)
            if old is None:
                return True
            word ^= old
        return False

    cols = _inc_columns(checks, n)
    bits = [1 << i for i in range(n)]
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = bits[i]
        elif outside(bits[i] ^ old):
            return True
    for i, x in enumerate(cols):
        for j in range(i):
            syndrome = x ^ cols[j]
            word = bits[i] | bits[j]
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif outside(word ^ old):
                return True

    if through_five:
        # Every five-word touching sector two has a 3+2 partition
        # whose triple touches sector two. Its largest index is >=start.
        for i in range(start, n):
            x = cols[i]
            for j in range(1, i):
                xy = x ^ cols[j]
                for h in range(j):
                    old = seen.get(xy ^ cols[h])
                    if old is not None:
                        word = bits[i] | bits[j] | bits[h]
                        if outside(word ^ old):
                            return True
    return False


def _jump_fingerprint(hx, hz, n):
    """Invariant of the combined Tanner graph, without X/Z colouring.

    Unequal signatures exclude arbitrary check/qubit relabelings.
    Equal signatures are conservatively discarded, not declared
    isomorphic. Weighted overlap triangles supplement degree profiles.
    """
    from collections import Counter
    rows = list(hx) + list(hz)
    weights = [r.bit_count() for r in rows]
    adj = [dict() for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                adj[i][j] = overlap
                adj[j][i] = overlap

    profiles = []
    for i, neighbors in enumerate(adj):
        items = list(neighbors.items())
        triangles = 0
        for a, (s, ws) in enumerate(items):
            for t, wt in items[:a]:
                triangles += ws * wt * adj[s].get(t, 0)
        profiles.append((
            weights[i],
            tuple(sorted(Counter(
                (weights[j], overlap)
                for j, overlap in items).items())),
            triangles,
        ))

    qubits = [[] for _ in range(n)]
    for i, original in enumerate(rows):
        row = original
        while row:
            bit = row & -row
            qubits[bit.bit_length() - 1].append(weights[i])
            row ^= bit
    return (
        tuple(sorted(Counter(profiles).items())),
        tuple(sorted(Counter(
            tuple(sorted(v)) for v in qubits).items())),
    )


@lru_cache(maxsize=1)
def _jump_bank(G):
    """Compile singular cliques and optimize fresh long-voltage cards."""
    from itertools import combinations

    q = G.q
    assert q == 42
    mask = (1 << q) - 1
    dummy_a = [[[G.e] for _ in range(3)]]
    dummy_b = [[[G.e] for _ in range(4)]]
    assert _grid_weights([[3, 3, 3]], [[2, 2, 2, 3]]) == 12
    assert _no_dead_lines([[3, 3, 3]], [[2, 2, 2, 3]])

    # XOR sums of verified unit-cell lifts give the component blocks.
    # There is no independently implemented regular representation.
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            if side == "A":
                unit = [[[g], [G.e], [G.e]]]
                hx, hz, _ = lifted_matrices(G, unit, dummy_b)
                source = hx
            else:
                unit = [[[g], [G.e], [G.e], [G.e]]]
                hx, hz, _ = lifted_matrices(G, dummy_a, unit)
                source = hz
            blocks.append(tuple(_inc_columns(
                [row & mask for row in source[:q]], q)))
        atlas[side] = tuple(blocks)

    def block(cell, side):
        out = [0] * q
        for g in cell:
            contribution = atlas[side][G.idx(g)]
            for h in range(q):
                out[h] ^= contribution[h]
        return tuple(out)

    def clean4(cols):
        return _inc_profile(cols) == 0

    # Every cubic A cell has a representative containing e after a
    # common right factor, which merely permutes its left-lift columns.
    # Remove those single-block incidence copies before assembly.
    raw = {}
    nonidentity = [g for g in G.elements if g != G.e]
    for g, h in combinations(nonidentity, 2):
        cell = (G.e, g, h)
        cols = block(cell, "A")
        space = _jump_span(cols)
        if q - len(space) < 6 or not clean4(cols):
            continue
        incidence = tuple(sorted(cols))
        raw.setdefault(space, {}).setdefault(incidence, (cell, cols))

    targets = {space for space in raw if len(space) == 36}
    # If necessary, allow unequal, more singular blocks whose union
    # supplies the intended rank-36 image instead of equal image spaces.
    if not targets:
        spaces = list(raw)
        for i, first in enumerate(spaces):
            for second in spaces[:i]:
                if gf2_rank(first + second) == 36:
                    targets.add(_jump_span(first + second))

    apool, aseen = [], set()

    def add_a(entries):
        cols = tuple(s for _, cc in entries for s in cc)
        if gf2_rank(cols) != 36 or not clean4(cols):
            return False
        cells = tuple(sorted(cell for cell, _ in entries))
        if cells in aseen:
            return False
        aseen.add(cells)
        # Every component column is cubic, so odd kernel weights are
        # impossible. Clean through four therefore means clean through five.
        rows = _jump_rows(cols, q)
        apool.append((_jump_cycles(rows), cells))
        return True

    compiler_rng = random.Random(74260191)
    for target in sorted(targets):
        compatible = {}
        for space, bucket in raw.items():
            if gf2_rank(target + space) != 36:
                continue
            compatible.update(bucket)
        entries = list(compatible.values())
        compiler_rng.shuffle(entries)
        if len(entries) < 3:
            continue
        neighbors = [set() for _ in entries]
        for i in range(len(entries)):
            for j in range(i):
                if clean4(entries[j][1] + entries[i][1]):
                    neighbors[j].add(i)
        accepted = 0
        for i in range(len(entries)):
            for j in sorted(neighbors[i]):
                for t in sorted(neighbors[i] & neighbors[j]):
                    if add_a([entries[z] for z in (i, j, t)]):
                        accepted += 1
                    if accepted >= 6:
                        break
                if accepted >= 6:
                    break
            if accepted >= 6:
                break
        if len(apool) >= 96:
            break

    # Broader rank-first assembly is a fresh search, not a copied fallback.
    if not apool:
        entries = [entry for bucket in raw.values()
                   for entry in bucket.values()]
        if len(entries) >= 3:
            for attempt in range(12000):
                add_a(compiler_rng.sample(entries, 3))
                if len(apool) >= 24:
                    break
    apool.sort()
    if not apool:
        return (None,) * N_SLOTS

    # y=(3,0) and x*y^-1=(5,1) have order six and do not commute.
    # x^c=(1,c) has order seven. Unlike the old monomial anchor,
    # these binomials do not themselves force weight-four kernels.
    quotient_models = (
        (0, 0, 1), (0, 1, 3), (0, 2, 5),
        (1, 2, 4), (0, 3, 5),
    )
    fingerprints, bank = set(), []

    for slot in range(N_SLOTS):
        policy, mode = divmod(slot % 20, 5)
        local = random.Random(74260217 + 100003 * slot)
        c = 1 + policy
        skeleton = ((G.e, (3, 0)), (G.e, (5, 1)),
                    (G.e, (1, c)))
        base_cols = tuple(
            syndrome for cell in skeleton
            for syndrome in block(cell, "B"))
        reserve, chosen = None, None
        bseen = set()

        for batch in range(2):
            bpool = []
            for attempt in range(96 + 32 * batch):
                phase = local.randrange(6)
                triple = []
                for exponent in quotient_models[mode]:
                    u = pow(3, (exponent + phase) % 6, 7)
                    choices = [(u, v) for v in range(7)
                               if (u, v) not in triple]
                    triple.append(local.choice(choices))
                sig = tuple(sorted(triple))
                if sig in bseen:
                    continue
                bseen.add(sig)
                cols = base_cols + block(triple, "B")
                if gf2_rank(cols) != q or not clean4(cols):
                    continue
                rows = _jump_rows(cols, q)
                B = [[list(cell) for cell in skeleton] + [triple]]
                bpool.append((_jump_cycles(rows), B, cols))
                bpool.sort(key=lambda item: item[0])
                del bpool[12:]

            screened_b = []
            for cycles, B, cols in bpool:
                six = _jump_six(cols)
                floor = 5 if six is None else 6
                screened_b.append((floor, cycles,
                                   10**9 if six is None else six, B))
            screened_b.sort(key=lambda item: (-item[0], item[2], item[1]))
            screened_b = screened_b[:4]

            ordered_a = list(apool)
            local.shuffle(ordered_a)
            ordered_a.sort(key=lambda item: item[0])
            proposals = []
            for acycles, cells in ordered_a[:12]:
                A = [[list(cell) for cell in cells]]
                for floor, bcycles, six, B in screened_b:
                    proposals.append(
                        ((-floor, six, acycles + bcycles), A, B))
            local.shuffle(proposals)
            proposals.sort(key=lambda item: item[0])

            five_tests = 0
            for key, A, B in proposals[:24]:
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if k != 270 or not (475*n <= 1000*k < 525*n):
                    continue
                start = 3 * 4 * q
                if _jump_small_logical(hx, hz, n, start, False):
                    continue
                if _jump_small_logical(hz, hx, n, start, False):
                    continue
                fp = _jump_fingerprint(hx, hz, n)
                if fp in fingerprints:
                    continue
                assert css_ok(hx, hz)
                assert max(row.bit_count() for row in hx + hz) <= W_MAX
                assert all(len(cell) == len(set(cell))
                           for grid in (A, B)
                           for row in grid for cell in row)
                item = (A, B, k, key, fp, 5)
                if reserve is None:
                    reserve = item

                if -key[0] != 6 or five_tests >= 6:
                    continue
                five_tests += 1
                # HX columns have weights three or nine. Thus no odd
                # HX-kernel support exists, including weight five.
                assert all(s.bit_count() & 1
                           for s in _inc_columns(hx, n))
                # HZ sector one is a direct sum of verified B components,
                # each already clean through five. The remaining test
                # exhausts all fifth-weight words touching sector two.
                if _jump_small_logical(hz, hx, n, start, True):
                    continue
                chosen = (A, B, k, key, fp, 6)
                break
            if chosen is not None:
                break

        selected = chosen if chosen is not None else reserve
        if selected is None:
            bank.append(None)
            continue
        A, B, k, key, fp, floor = selected
        fingerprints.add(fp)
        bank.append({
            "cells": {
                "A": tuple(tuple(tuple(cell) for cell in row) for row in A),
                "B": tuple(tuple(tuple(cell) for cell in row) for row in B),
            },
            "k": k, "card": slot % 20,
            "exact_logical_exclusion": floor - 1,
            "selection_score": key,
        })
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign a fresh affine-group card, never an orbit-copy fallback."""
    item = _jump_bank(G)[idx]
    if item is None:
        raise ValueError("no fresh rank-270 affine filling passed logical screening")
    return item


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without further draws or relabeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Uniform labels with rolled counts at q=58, (1,3)x(1,3), reliably avoid weight<=3 logicals: many parent slots had d2s=2 or 3; its measured top-five all had d30s=4.
# REFUTED: Voltage-autocorrelation classes and reflection-offset optimization ensure five distinct D18 (1,4)x(2,7) codes: sixty parent slots collapsed to two distinct codes, both with d30s=4.
# REFUTED: D18 odd-anchor/incidence-card optimization removes the weight-four bottleneck: all five measured current codes had [540,276,4], with thousands of sector-one weight-four witnesses.
# REFUTED: The parent's separate-type incidence fingerprints prevent every Tanner-isomorphic submission: multiple current slots, including 10,14,20 and 29, were reported as iso_dup despite its fingerprint filter.
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

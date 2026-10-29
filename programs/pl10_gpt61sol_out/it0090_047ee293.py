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
CONCEPT = ("For metacyclic Z7:Z2(s=6), order 14, shapes (2,4)x(2,5), constant degree-three columns and reduced weight-six/eight component-kernel multiplicities should improve worse-basis circuit fitness above the parent's 4.7482.")
GROUP = ("Keep the exact frozen group and order. This bed produced s=5.101 in slot 0, so filling remains a meaningful controlled experiment. Same-coset binomials retain order-seven relative rotations; no advantage of larger-group beds is attributed to this fixed lift.")
SHAPES = ("Frozen (2,4)x(2,5): n=336, kb=84, guaranteed rate 0.250; require measured k=84..92. Components have 56 and 70 variables and 28 checks each. Parent failures spread across Acol/Bcol blocks with only 1.1-1.2x enrichment; protect both components rather than infer a single defective block.")
FILLING = ("Twenty explicit support architectures keep every component column at degree three, every check weight <=11, and the six-round CNOT count at 14616. Most introduce zero/trinomial alternatives to the parent's nonzero-only cells. Search fresh labels with verified component lifts, exact exclusion of kernels through weight four, exact weight-six multiplicities and, when those vanish, exact weight-eight multiplicities. Also screen component connectivity, rank deficits, small syndrome boundaries, final CSS and logical weights <=4. Extra harness batches change trinomial coset policies. Parent slots 37/57 used 17472 CNOTs and had much worse Z failure rates than slots 0/40; reduced load and ambiguity are pL hypotheses, not circuit measurements.")
FAMILY_NAME = "r025_m7r2s6_24x25_degree3_exact68_v2"

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

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _balanced_counts(rng, totals):
    """Two nonzero rows; column totals prescribed, row totals <=8."""
    import itertools
    choices = []
    total = sum(totals)
    for first in itertools.product(*(range(1, t) for t in totals)):
        if max(sum(first), total - sum(first)) <= 8:
            choices.append(first)
    first = choices[rng.randrange(len(choices))]
    return [list(first), [t - x for t, x in zip(totals, first)]]


def _coset_grid(rng, pools, counts):
    grid = []
    for row in counts:
        out = []
        for size in row:
            b = rng.randrange(2)
            if size == 1:
                cell = rng.sample(pools[b], 1)
            elif size == 2:
                # Relative element is a nonidentity order-seven rotation.
                cell = rng.sample(pools[b], 2)
            else:
                # Odd support, and genuinely mixed-coset multiplication.
                cell = rng.sample(pools[b], 2) + rng.sample(pools[1-b], 1)
            out.append(cell)
        grid.append(out)
    return grid


def _component_quality(rows, width, rng):
    """Components extracted from the verified lift, not a separate lift.
    Exactly exclude kernel weights 1..4; estimate larger kernel weights."""
    syndromes = [0] * width
    pair_hits = {}
    cycles = 0
    for r, row in enumerate(rows):
        positions = []
        bits = row
        while bits:
            bit = bits & -bits
            c = bit.bit_length() - 1
            syndromes[c] |= 1 << r
            positions.append(c)
            bits ^= bit
        for a in range(len(positions)):
            for b in range(a):
                key = (positions[b], positions[a])
                old = pair_hits.get(key, 0)
                cycles += old
                pair_hits[key] = old + 1

    if 0 in syndromes or len(set(syndromes)) != width:
        return None
    singles = set(syndromes)
    pairs = set()
    for a in range(width):
        for b in range(a):
            s = syndromes[a] ^ syndromes[b]
            if s in singles or s in pairs:
                # With unique nonzero columns, these certify weight 3 or 4.
                return None
            pairs.add(s)

    pivots = {}
    for row in rows:
        v = row
        while v:
            p = v.bit_length() - 1
            if p in pivots:
                v ^= pivots[p]
            else:
                pivots[p] = v
                break
    ordered = sorted(pivots.items())
    basis = []
    for c in range(width):
        if c in pivots:
            continue
        v = 1 << c
        for p, row in ordered:
            if _popcount(row & v) & 1:
                v ^= 1 << p
        basis.append(v)

    # Every vector considered here is an actual component-kernel vector.
    distance = min(_popcount(v) for v in basis)
    for a in range(len(basis)):
        for b in range(a):
            ab = basis[a] ^ basis[b]
            distance = min(distance, _popcount(ab))
            for c in range(b):
                distance = min(distance, _popcount(ab ^ basis[c]))
    for _ in range(1024):
        v = 0
        for c in rng.sample(range(len(basis)), min(4, len(basis))):
            v ^= basis[c]
        distance = min(distance, _popcount(v))
    return distance, cycles


def NEW_CTX(rng, G, idx):
    """Search fresh labels within a deliberate constant-load architecture.

    All algebraic screening uses matrices returned by lifted_matrices.
    Kernel multiplicities and small syndrome boundaries are surrogates;
    none is asserted to measure the scored circuit failure probability.
    """
    # Each tuple specifies the top row. The bottom row is 3-top.
    # A profiles are inequivalent under row/column permutations.
    # B's top sums are seven; exchanging rows gives sums eight.
    atop = (
        (1, 1, 2, 2),
        (0, 1, 2, 3),
        (0, 2, 2, 2),
    )
    btop = (
        (1, 1, 1, 2, 2),
        (0, 1, 2, 2, 2),
        (0, 1, 1, 2, 3),
        (1, 1, 1, 1, 3),
        (0, 0, 2, 2, 3),
        (0, 0, 1, 3, 3),
    )
    design = idx % 20
    if design < 18:
        ta, tb = atop[design // 6], btop[design % 6]
    else:
        # Two genuinely different A row-exposure designs, sums five/seven.
        ta = ((1, 1, 1, 2), (0, 1, 1, 3))[design - 18]
        tb = btop[0]
    sa = [list(ta), [3 - t for t in ta]]
    sb = [list(tb), [3 - t for t in tb]]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(11, W_MAX)

    q = G.q
    cosets = [[g for g in G.elements if g[1] == b] for b in range(2)]
    mode = (idx // 20) % 3

    def draw(counts):
        grid = []
        for row in counts:
            out = []
            for weight in row:
                b = rng.randrange(2)
                if weight == 0:
                    cell = []
                elif weight <= 2:
                    cell = rng.sample(cosets[b], weight)
                elif mode == 0:
                    cell = (rng.sample(cosets[b], 2)
                            + rng.sample(cosets[1 - b], 1))
                elif mode == 1:
                    cell = rng.sample(cosets[b], 3)
                else:
                    cell = rng.sample(G.elements, 3)
                out.append(cell)
            grid.append(out)
        return grid

    def component_rows(side, grid):
        # Zero opposite-side probes are never submitted as codes.
        # Restriction retains the exact left action for A and right for B.
        if side == "A":
            rows = lifted_matrices(G, grid, [[[]]])[0]
        else:
            rows = lifted_matrices(G, [[[]]], grid)[1]
        mask = (1 << (len(grid[0]) * q)) - 1
        return [row & mask for row in rows]

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= tag
                row ^= bit
        return result

    def connected(cols, checks):
        # Connectivity of the component check-intersection graph.
        adjacency = [0] * checks
        for syndrome in cols:
            bits = syndrome
            while bits:
                bit = bits & -bits
                adjacency[bit.bit_length() - 1] |= syndrome ^ bit
                bits ^= bit
        seen = frontier = 1
        while frontier:
            neighbors = 0
            bits = frontier
            while bits:
                bit = bits & -bits
                neighbors |= adjacency[bit.bit_length() - 1]
                bits ^= bit
            frontier = neighbors & ~seen
            seen |= frontier
        return seen == (1 << checks) - 1

    def no_kernel_four(cols):
        if 0 in cols or len(set(cols)) != len(cols):
            return False
        singles = set(cols)
        pairs = set()
        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                if syndrome in singles or syndrome in pairs:
                    return False
                pairs.add(syndrome)
        return True

    def four_cycles(rows):
        total = 0
        for i, row in enumerate(rows):
            for other in rows[:i]:
                overlap = (row & other).bit_count()
                total += overlap * (overlap - 1) // 2
        return total

    def six_and_boundary(cols, limit):
        """Exact weight-six count after excluding kernel weights <=4.

        Degree-three columns make every kernel word even. Triple
        collisions must therefore have disjoint supports and weight six;
        each six-set has ten unordered partitions into two triples.
        The boundary statistic is only a decoder-ambiguity heuristic.
        """
        seen = {}
        collisions = boundary = 0
        bound = None if limit is None else 10 * limit
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    old = seen.get(syndrome, 0)
                    collisions += old
                    if bound is not None and collisions > bound:
                        return None
                    seen[syndrome] = old + 1
                    weight = syndrome.bit_count()
                    if weight <= 3:
                        boundary += (4 - weight) ** 2
        assert collisions % 10 == 0
        return collisions // 10, boundary

    def eight_count(cols, limit):
        """Exact weight-eight count ONLY after weights <=6 are absent.

        Quartet collisions then have disjoint supports. Each eight-set
        has 35 unordered partitions into two quartets.
        """
        seen = {}
        collisions = 0
        bound = None if limit is None else 35 * limit
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    triple = pair ^ cols[h]
                    for t in range(h):
                        syndrome = triple ^ cols[t]
                        old = seen.get(syndrome, 0)
                        collisions += old
                        if bound is not None and collisions > bound:
                            return None
                        seen[syndrome] = old + 1
        assert collisions % 35 == 0
        return collisions // 35

    def optimize(side, counts):
        width = len(counts[0]) * q
        pool = []
        identities = set()
        for trial in range(512):
            grid = draw(counts)
            identity = tuple(tuple(sorted(G.idx(g) for g in cell))
                             for row in grid for cell in row)
            if identity in identities:
                continue
            identities.add(identity)
            rows = component_rows(side, grid)
            deficit = len(rows) - gf2_rank(rows)
            if deficit > 1:
                continue
            cols = columns(rows, width)
            assert all(s.bit_count() == 3 for s in cols)
            if not connected(cols, len(rows)) or not no_kernel_four(cols):
                continue

            limit = None if not pool else pool[0][0][0]
            brief = six_and_boundary(cols, limit)
            if brief is None:
                continue
            six, boundary = brief
            four = four_cycles(rows)
            key = (six, deficit, four, boundary)
            pool.append((key, trial, grid, cols))
            pool.sort(key=lambda entry: (entry[0], entry[1]))
            # Once a lower six-count appears, higher counts cannot win.
            minimum = pool[0][0][0]
            pool = [entry for entry in pool if entry[0][0] == minimum][:6]

        if not pool:
            raise ValueError("no connected short-kernel-safe component")

        finalists = []
        best_eight = None
        for key, trial, grid, cols in pool:
            six, deficit, four, boundary = key
            if six == 0:
                eight = eight_count(cols, best_eight)
                if eight is None:
                    continue
                best_eight = eight if best_eight is None else min(
                    best_eight, eight)
            else:
                # Do not mislabel quartet collisions as an eight-spectrum
                # when lower-weight component words are present.
                eight = 10**9
            finalists.append({
                "grid": grid, "six": six, "eight": eight,
                "deficit": deficit, "four": four,
                "boundary": boundary, "trial": trial,
            })
        finalists.sort(key=lambda entry: (
            entry["six"], entry["eight"], entry["deficit"],
            entry["four"], entry["boundary"], entry["trial"]))
        return finalists

    def has_small_logical(checks, stabilizers, n):
        """Exact logical existence test through weight four."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        def trivial(word):
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    return False
                word ^= old
            return True

        cols = columns(checks, n)
        seen = {0: 0}
        for i, syndrome in enumerate(cols):
            word = 1 << i
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif not trivial(old ^ word):
                return True
        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                word = (1 << i) | (1 << j)
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = word
                elif not trivial(old ^ word):
                    return True
        return False

    for restart in range(3):
        aa = optimize("A", sa)
        bb = optimize("B", sb)
        options = []
        for a in aa:
            for b in bb:
                key = (
                    max(a["six"], b["six"]), a["six"] + b["six"],
                    max(a["eight"], b["eight"]),
                    a["eight"] + b["eight"],
                    a["four"] + b["four"],
                    a["boundary"] + b["boundary"],
                    a["deficit"] + b["deficit"],
                    a["trial"], b["trial"],
                )
                options.append((key, a["grid"], b["grid"]))
        options.sort(key=lambda entry: entry[0])
        for key, A, B in options:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not 84 <= k <= 92:
                continue
            if has_small_logical(hx, hz, n):
                continue
            if has_small_logical(hz, hx, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= min(11, W_MAX)
            assert 6 * sum(row.bit_count() for row in hx + hz) == 14616
            return {"idx": idx, "design": design, "mode": mode,
                    "k": k, "screen": key,
                    "cells": {"A": A, "B": B}}

    raise ValueError("no target-rate code passed final logical screening")


def CELL(rng, G, side, i, j, ctx):
    """All cells are already selected; return a fresh full-cell list."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled D14 (6,2)x(6,2) fillings do not provide adequate sector-2 protection: all five measured codes had weight-4 witnesses there and scores only 3.30..3.59.
# REFUTED: Balanced same-coset support profiles uniformly yield basis-balanced memory: parent slot 37 failed 100/5500 Z shots versus 100/59750 X shots, scoring 4.370 versus 5.409.
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

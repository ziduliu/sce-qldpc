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
CONCEPT = ("Metacyclic C4 semidirect C2 with action 3, order 8, shapes (2,8)x(2,8): balanced degree-two/three components with reduced nontrivial logical hook collisions should improve worse-basis circuit fitness above the parent's 2.9485.")
GROUP = ("Retain the metacyclic construction rule. The new size factor 68 would give n=952 at order 14, so change its parameters to m=4,r=2,s=3; 3^2=1 modulo 4 and q=8 gives n=544. This bed previously reached approximately 3.2, motivating a new filling experiment without claiming a causal group advantage.")
SHAPES = ("Equal 2x8 components have 16 checks and 64 variables each. n=544 and kb=288 guarantee rate 9/17 in target cell 0.550; final verified ranks require k=288. Parent weight-three witnesses and collective failures occupied many sector-one Acol/Bcol blocks, with mostly 1.0-1.1x check-block enrichment and no established Acol/Brow hotspot. Two-row components permit protection without retaining the parent's singleton columns.")
FILLING = ("Nine explicit column architectures use component column totals two/three and row totals at most nine, making both CSS weight sums <=12. Include zero/trinomial columns absent from the parent. Independently search a deterministic catalogue, retain full-row-rank components and different uncolored rooted Tanner fingerprints, and reserve unused unordered component-class pairs across slots. Four finalists receive exact quotient-label screening for logical triples and pair collisions involving shared-opposite-check hook surrogates in both bases. This targets the parent's 109-144 failures per 250 shots and widespread logical flips despite near-universal first-pass settlement. Fingerprints are conservative component invariants, not full-code isomorphism certificates; hook exposure is not an ASC ordering certificate or measured pL.")
FAMILY_NAME = "r055_m4r2s3_28x28_componentclass_hook_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("This metacyclic parameter set requires q=8")
    # Same construction rule: x^4=y^2=1, y*x*y^-1=x^3.
    # Order 14 is inadmissible for the new size factor 68.
    m, r, s = 4, 2, 3
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 8)               # (ra, ca)
B_SHAPE = (2, 8)               # (rb, cb)


def _short_logical_counts(checks, stabilizers, n):
    """Exact counts of nontrivial logical supports of weights 1..4.
    Matrices come exclusively from the verified scaffold lift."""
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old

    def nontrivial(v):
        while v:
            p = v.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                return True
            v ^= old
        return False

    columns = [0] * n
    for r, row in enumerate(checks):
        syndrome_bit = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= syndrome_bit
            row ^= bit

    singles = {}
    counts = [0, 0, 0, 0]
    bits = [1 << a for a in range(n)]
    for a, syndrome in enumerate(columns):
        singles.setdefault(syndrome, []).append(a)
        if syndrome == 0 and nontrivial(bits[a]):
            counts[0] += 1

    pairs = {}
    for a in range(n):
        for b in range(a + 1, n):
            syndrome = columns[a] ^ columns[b]
            support = bits[a] | bits[b]
            if syndrome == 0 and nontrivial(support):
                counts[1] += 1

            # a < b < c counts each weight-three support once.
            for c in singles.get(syndrome, ()):
                if c > b and nontrivial(support | bits[c]):
                    counts[2] += 1

            bucket = pairs.get(syndrome)
            if bucket is None:
                pairs[syndrome] = [(a, b, support)]
            else:
                # c < d < a < b is the unique ordered partition
                # used to count each weight-four support once.
                for c, d, previous in bucket:
                    if d < a and nontrivial(previous | support):
                        counts[3] += 1
                bucket.append((a, b, support))
    return tuple(counts)


def _designed_side(rng, G, columns):
    """Draw one architecture; never manufacture a saved-code orbit."""
    grid = [[], []]
    nonidentity = [g for g in G.elements if g != G.e]
    mono_labels = iter(rng.sample(
        G.elements, columns.count((1, 1))))
    for a, b in columns:
        if (a, b) == (1, 1):
            cells = ([G.e], [next(mono_labels)])
        elif not a or not b:
            weight = a or b
            # A one-row doubleton must not have an involutory ratio.
            cell = ([G.e, (1, 0)] if weight == 2 else
                    [G.e] + rng.sample(nonidentity, weight - 1))
            cells = (cell, []) if a else ([], cell)
        else:
            cells = (
                [G.e] if a == 1 else
                [G.e] + rng.sample(nonidentity, a - 1),
                rng.sample(G.elements, b),
            )
        grid[0].append(list(cells[0]))
        grid[1].append(list(cells[1]))
    return grid


def NEW_CTX(rng, G, idx):
    """Pure deterministic class catalogue, followed by joint hook screening."""
    from itertools import combinations_with_replacement
    from collections import Counter
    import hashlib

    q = G.q
    assert q == 8
    mono = ((1, 1),)
    profiles = (
        mono * 7 + ((1, 2),),
        mono * 6 + ((1, 2), (2, 1)),
        mono * 6 + ((0, 3), (3, 0)),
        mono * 6 + ((0, 2), (2, 1)),
        mono * 5 + ((0, 2), (2, 0), (1, 2)),
        mono * 4 + ((0, 2), (2, 0), (1, 2), (2, 1)),
        mono * 5 + ((0, 2), (2, 1), (2, 1)),
        mono * 5 + ((0, 3), (2, 1), (2, 0)),
        mono * 6 + ((0, 3), (2, 0)),
    )
    cards = [
        (2, 2), (0, 2), (1, 2), (1, 1), (0, 1),
        (2, 8), (1, 7), (7, 7), (0, 7), (7, 8),
        (1, 5), (2, 5), (5, 7), (0, 0), (0, 3),
        (3, 7), (4, 7), (6, 7), (6, 6), (8, 8),
    ]
    cards.extend(pair for pair in combinations_with_replacement(
        range(len(profiles)), 2) if pair not in cards)

    supports = [
        [[column[i] for column in profile] for i in range(2)]
        for profile in profiles
    ]
    for sa in supports:
        assert _no_dead_lines(sa, sa)
        assert _grid_weights(sa, sa) <= min(12, W_MAX)
        assert all(sum(row) <= 9 for row in sa)
        assert all(2 <= sum(c) <= 3 for c in zip(*sa))

    # Singleton actions come exclusively from the verified helper.
    mask = (1 << q) - 1
    actions = {}
    for g in G.elements:
        rows = lifted_matrices(G, [[[g]]], [[[]]])[0]
        actions[G._key(g)] = tuple(row & mask for row in rows)

    def component(grid):
        rows = []
        for cells in grid:
            out = [0] * q
            for j, cell in enumerate(cells):
                for g in cell:
                    for t, word in enumerate(actions[G._key(g)]):
                        out[t] ^= word << (j * q)
            rows.extend(out)
        return rows

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def brief(rows):
        cols = columns(rows, 8 * q)
        if 0 in cols or len(set(cols)) != len(cols):
            return None
        if gf2_rank(rows) != 2 * q:
            return None
        singles = set(cols)
        pairs = {}
        triples = collisions = boundary = 0
        for i, si in enumerate(cols):
            for sj in cols[:i]:
                s = si ^ sj
                triples += int(s in singles)
                collisions += pairs.get(s, 0)
                pairs[s] = pairs.get(s, 0) + 1
                w = s.bit_count()
                if w <= 2:
                    boundary += (3 - w) ** 2
        four = sum(
            c * (c - 1) // 2
            for i, row in enumerate(rows) for other in rows[:i]
            for c in [(row & other).bit_count()])
        # With distinct nonzero columns these normalized word counts
        # are exact. Boundary burden remains only a decoding surrogate.
        return triples // 3, collisions // 3, boundary, four

    def fingerprint(rows):
        """Uncolored rooted component invariant, not full-code canonization."""
        width = 8 * q
        adjacency = [[] for _ in range(len(rows) + width)]
        for c, syndrome in enumerate(columns(rows, width)):
            while syndrome:
                bit = syndrome & -syndrome
                r = bit.bit_length() - 1
                adjacency[r].append(len(rows) + c)
                adjacency[len(rows) + c].append(r)
                syndrome ^= bit
        images = []
        # Group translations are transitive within each protograph row.
        for root in (0, q):
            colors = [0] * len(rows) + [1] * width
            colors[root] = 2
            digest = hashlib.sha256()
            for _ in range(12):
                signatures = [
                    (colors[v], tuple(sorted(colors[w]
                                             for w in adjacency[v])))
                    for v in range(len(adjacency))
                ]
                digest.update(repr(sorted(
                    Counter(signatures).items())).encode("ascii"))
                palette = {s: t for t, s in enumerate(
                    sorted(set(signatures)))}
                colors = [palette[s] for s in signatures]
            images.append(digest.hexdigest())
        return min(images)

    books = []
    for architecture, profile in enumerate(profiles):
        crng = random.Random(804113 + 1009 * architecture)
        unique = {}
        for trial in range(192):
            grid = _designed_side(crng, G, profile)
            rows = component(grid)
            quality = brief(rows)
            if quality is None:
                continue
            identity = fingerprint(rows)
            labeling = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in grid for cell in row)
            entry = (quality, identity, labeling, grid)
            old = unique.get(identity)
            if old is None or (quality, labeling) < (old[0], old[2]):
                unique[identity] = entry
        book = sorted(unique.values(), key=lambda e: e[:3])
        if not book:
            raise ValueError("No full-rank distinct-syndrome component class")
        books.append(book[:12])

    catalogues = []
    for pa, pb in cards:
        candidates = {}
        for a in books[pa]:
            for b in books[pb]:
                identity = tuple(sorted((a[1], b[1])))
                quality = tuple(
                    value for t in range(4)
                    for value in (max(a[0][t], b[0][t]),
                                  a[0][t] + b[0][t]))
                entry = (quality, identity, a[3], b[3])
                old = candidates.get(identity)
                if old is None or quality < old[0]:
                    candidates[identity] = entry
        catalogues.append(sorted(
            candidates.values(), key=lambda e: e[:2]))

    # Reconstruct the same reservation prefix in every call. No mutable
    # cross-slot cache and no orbit transformation of a saved filling.
    reserved = set()
    chosen = None
    actual_card = None
    for slot in range(idx + 1):
        card = slot % len(cards)
        available = [e for e in catalogues[card]
                     if e[1] not in reserved]
        if not available:
            alternatives = [
                (e[0], e[1], c, e)
                for c, catalogue in enumerate(catalogues)
                for e in catalogue if e[1] not in reserved
            ]
            if not alternatives:
                raise ValueError("Unused component-class pair catalogue exhausted")
            alternatives.sort(key=lambda e: e[:3])
            card = alternatives[0][2]
            available = [e for e in catalogues[card]
                         if e[1] not in reserved]
        chosen = available[:4]
        actual_card = card
        reserved.update(e[1] for e in chosen)

    def quotient_columns(rows, width):
        pivots = {}
        for row in rows:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        positions = sorted(pivots, reverse=True)
        residuals = []
        for c in range(width):
            word = 1 << c
            for p in positions:
                if (word >> p) & 1:
                    word ^= pivots[p]
            residuals.append(word)
        return residuals

    def hook_profile(checks, stabilizers, width):
        syndrome = columns(checks, width)
        if 0 in syndrome or len(set(syndrome)) != width:
            return None
        spread = columns(stabilizers, width)
        residual = quotient_columns(stabilizers, width)
        singles = dict(zip(syndrome, residual))
        pairs = {}
        three = four = double_hook = exposure = 0
        for i in range(width):
            si, ri, hi = syndrome[i], residual[i], spread[i]
            for j in range(i):
                s, label = si ^ syndrome[j], ri ^ residual[j]
                third = singles.get(s)
                if third is not None and third != label:
                    three += 1
                hook = (hi & spread[j]).bit_count()
                old = pairs.get(s)
                if old is None:
                    pairs[s] = [1, hook, {label: [1, hook]}]
                    continue
                same = old[2].get(label)
                count, hooks = same if same else (0, 0)
                different = old[0] - count
                different_hooks = old[1] - hooks
                four += different
                double_hook += hook * different_hooks
                exposure += hook * different + different_hooks
                old[0] += 1
                old[1] += hook
                if same is None:
                    old[2][label] = [1, hook]
                else:
                    same[0] += 1
                    same[1] += hook
        return three // 3, double_hook, exposure, four // 3

    best = None
    for coarse, identity, A, B in chosen:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 544 or k != 288:
            continue
        # Certify cached A actions against the actual final construction.
        ar = [
            sum(((hx[(i * 8) * q + t] >> (j * 8 * q)) & mask)
                << (j * q) for j in range(8))
            for i in range(2) for t in range(q)
        ]
        assert ar == component(A)
        br = [row & ((1 << (8 * q)) - 1) for row in hz[:2 * q]]
        if brief(br) is None:
            continue
        px = hook_profile(hz, hx, n)
        pz = hook_profile(hx, hz, n)
        if px is None or pz is None:
            continue
        key = tuple(
            value for t in range(4)
            for value in (max(px[t], pz[t]), px[t] + pz[t]))
        key += (coarse, identity)
        if best is None or key < best[0]:
            best = (key, A, B, hx, hz, px, pz)

    if best is None:
        raise ValueError("No exact-bound joint hook-screened finalist")
    key, A, B, hx, hz, px, pz = best
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    assert css_ok(hx, hz)
    return {"idx": idx, "case": cards[actual_card], "k": 288,
            "screen": key, "hook_profile": (px, pz),
            "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the already-screened complete cell without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Distance-four screening plus first-pass settlement does not certify low circuit failure: parent slot 17 Z failed 113/250 shots despite settlement 250/0/0.
# REFUTED: The parent's single-row short-logical/gate-balance change improved circuit fitness: fitness fell from 3.4511 to 2.9485; all five measured codes had d2s=3, and slot 2 failed 114/250 shots in both bases. Shape and labeling changed together, so this does not isolate either cause.
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

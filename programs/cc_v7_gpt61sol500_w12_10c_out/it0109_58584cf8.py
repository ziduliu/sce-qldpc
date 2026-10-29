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
CONCEPT = ("For cyclic Z11 (metacyclic Z11:Z1), order 11, shapes (2,6)x(3,8), difference-packed degree-three columns and exact component screening through eight should produce five measured distance-eight codes in cell 0.350; this is a falsifiable measured-distance hypothesis.")
GROUP = ("Preserve the parent's trivial-action metacyclic construction. The new size factor 54 makes order 17 inadmissible; order 11 is the largest compatible cyclic lift. The parent measured five distance-eight codes, motivating a shape experiment rather than a new group family.")
SHAPES = ("A=(2,6), B=(3,8): n=594, kb=220 and guaranteed rate 20/54=0.37037 in cell 0.350. Components have 66/22 and 88/33 variables/checks. Parent minimum witnesses occupied sector-1 Acol/Bcol blocks; add a third B check row and require verified k=220. Five distance-eight readings would score 23.7037, above 22.588.")
FILLING = ("Twenty cards cross two A incidence profiles, five balanced B profiles and two relative-binomial-gap geometries. Every component column has three terms, A rows total nine and B rows eight, giving HX<=12 and HZ<=11. B augmentation has rank three. Independently optimize difference-packed voltages, exclude component words through six, then minimize exact weight-eight orbit counts among finalists. Verify assembled ranks, weights and CSS. Component exclusion is not a quantum-distance certificate.")
FAMILY_NAME = "r035_m11r1s1_26x38_balanced_orbit8_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 11

def custom_ops(q):
    # Same trivial-action metacyclic construction, resized for n=54*11.
    # Z_11 x| Z_1: y x y^-1 = x, with s^r = 1 mod m.
    m, r, s = 11, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 6)               # (ra, ca)
B_SHAPE = (3, 8)               # (rb, cb)


def _retired_parent_context(rng, G, idx):
    """Historical parent filling; the new shape never calls this function."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    kb = q * (ca - ra) * (cb - rb)

    # Entries are column term counts (top row, bottom row).
    # Each A profile has row totals 7/8; each B profile has 9/9.
    aprofiles = [
        [(2, 1), (2, 1), (1, 2), (1, 2), (1, 2)],
        [(3, 0), (2, 1), (1, 2), (1, 2), (1, 2)],
        [(3, 0), (0, 3), (2, 1), (1, 2), (1, 2)],
    ]
    bprofiles = [
        [(2, 1)] * 3 + [(1, 2)] * 3,
        [(3, 0), (2, 1)] + [(1, 2)] * 4,
        [(3, 0), (0, 3)] + [(2, 1)] * 2 + [(1, 2)] * 2,
    ]
    design = idx % 20
    if design < 9:
        acounts = aprofiles[design // 3]
        bcounts = bprofiles[design % 3]
    elif design < 18:
        acounts = aprofiles[(design - 9) // 3]
        t = (design - 9) % 3 + 1
        bcounts = ([(1, 1)] * t + [(2, 2)] * t
                   + [(2, 1)] * (3 - t) + [(1, 2)] * (3 - t))
    elif design == 18:
        acounts = aprofiles[0]
        bcounts = [(1, 1), (3, 1), (2, 1)] + [(1, 2)] * 3
    else:
        acounts = aprofiles[2]
        bcounts = [(1, 1), (4, 0)] + [(1, 2)] * 4

    sa = [[col[i] for col in acounts] for i in range(2)]
    sb = [[col[i] for col in bcounts] for i in range(2)]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    def pack(counts):
        """Construct a cyclic pair packing, not relabel an existing code.

        A within-row check-pair orbit uses an unsigned difference;
        a cross-row orbit uses an oriented difference. Never reuse
        either orbit within this component.
        """
        width = len(counts)
        for restart in range(64):
            used = [set(), set()]
            cross_used = set()
            out = [None] * width
            order = list(range(width))
            rng.shuffle(order)
            order.sort(key=lambda j: -sum(counts[j]))
            complete = True
            for j in order:
                a, b = counts[j]
                chosen = None
                for attempt in range(256):
                    x = rng.sample(range(q), a)
                    y = rng.sample(range(q), b)
                    anchor = x[0] if x else y[0]
                    x = sorted((v - anchor) % q for v in x)
                    y = sorted((v - anchor) % q for v in y)
                    local = []
                    legal = True
                    for row, vals in enumerate((x, y)):
                        diffs = []
                        for u in range(len(vals)):
                            for v in range(u + 1, len(vals)):
                                d = (vals[v] - vals[u]) % q
                                diffs.append(min(d, q - d))
                        ds = set(diffs)
                        if len(ds) != len(diffs) or ds & used[row]:
                            legal = False
                            break
                        local.append(ds)
                    if not legal:
                        continue
                    cross = [(u - v) % q for u in x for v in y]
                    cs = set(cross)
                    if len(cs) != len(cross) or cs & cross_used:
                        continue
                    chosen = (x, y)
                    used[0].update(local[0])
                    used[1].update(local[1])
                    cross_used.update(cs)
                    break
                if chosen is None:
                    complete = False
                    break
                out[j] = chosen
            if complete:
                return [[[G.elements[v] for v in out[j][i]]
                         for j in range(width)] for i in range(2)]
        return None

    def syndrome_columns(rows, positions):
        """Read columns of a verified matrix; do not implement a lift."""
        result = []
        for pos in positions:
            syndrome = 0
            for r, row in enumerate(rows):
                if (row >> pos) & 1:
                    syndrome |= 1 << r
            result.append(syndrome)
        return result

    def short_words(columns):
        """Exact rejection of words <=5; count weight-six words.

        Return 7 when no word <=6 exists: this is a screening flag,
        not an assertion that the component distance equals seven.
        After excluding words <=4, colliding triples are disjoint,
        so each weight-six word contributes ten triple collisions.
        """
        low = {}
        for s in columns:
            if not s:
                return (1, 1)
            if s in low:
                return (2, 1)
            low[s] = 1
        length = len(columns)
        for i in range(length):
            x = columns[i]
            for j in range(i + 1, length):
                s = x ^ columns[j]
                if not s:
                    return (2, 1)
                previous = low.get(s, 0)
                if previous:
                    return (3 if previous == 1 else 4, 1)
                low[s] = 2

        triples = {}
        get = triples.get
        collisions = 0
        for i in range(length - 2):
            x = columns[i]
            for j in range(i + 1, length - 1):
                xy = x ^ columns[j]
                for h in range(j + 1, length):
                    s = xy ^ columns[h]
                    if not s:
                        return (3, 1)
                    if s in low:
                        # A singleton collision would imply weight four,
                        # already excluded by the pair enumeration.
                        return (5, 1)
                    previous = get(s, 0)
                    collisions += previous
                    triples[s] = previous + 1
        if collisions:
            return (6, collisions // 10)
        return (7, 0)

    best = None
    best_score = None
    accepted = 0
    # Beyond twenty slots, retain the structural design but enlarge its
    # independently seeded optimization search; do not emit orbit copies.
    quota = 28 + 4 * (idx // 20)
    for trial in range(3 * quota):
        A = pack(acounts)
        B = pack(bcounts)
        if A is None or B is None:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != kb:
            continue

        # HX at Bcol=0 restricts to the left-lifted A component.
        arows = [hx[(i * cb) * q + r]
                 for i in range(ra) for r in range(q)]
        apos = [j * cb * q + r for j in range(ca) for r in range(q)]
        # HZ at Acol=0 restricts to the right-lifted B component.
        brows = hz[:rb * q]
        bpos = list(range(cb * q))
        pa = short_words(syndrome_columns(arows, apos))
        if pa[0] <= 4:
            continue
        pb = short_words(syndrome_columns(brows, bpos))
        if pb[0] <= 4:
            continue

        score = (min(pa[0], pb[0]), pa[0] + pb[0],
                 -max(pa[1], pb[1]), -(pa[1] + pb[1]))
        if best_score is None or score > best_score:
            best_score = score
            best = (A, B)
        accepted += 1
        if accepted >= quota:
            break

    if best is None:
        raise ValueError("no rank-bound candidate passed short-word screening")
    A, B = best
    hx, hz, n = lifted_matrices(G, A, B)
    assert css_ok(hx, hz)
    assert n - gf2_rank(hx) - gf2_rank(hz) == kb
    return {"idx": idx, "cells": {"A": A, "B": B},
            "component_screen": best_score}


def NEW_CTX(rng, G, idx):
    """Optimize twenty new-shape incidence/gap cards using verified lifts."""
    q = G.q
    assert q == 11
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    card = idx % 20
    aprofile = card // 10
    bprofile = (card // 2) % 5
    geometry = card % 2

    acatalogue = (
        ((3, 0), (2, 1)) + ((1, 2),) * 4,
        ((3, 0), (0, 3)) + ((2, 1),) * 2 + ((1, 2),) * 2,
    )
    # Each profile has eight degree-three columns and row totals (8,8,8).
    # Their parity columns span all three augmentation-check coordinates.
    cycle = ((2, 1, 0), (0, 2, 1), (1, 0, 2))
    bcatalogue = (
        cycle + ((1, 1, 1),) * 5,
        cycle * 2 + ((1, 1, 1),) * 2,
        ((3, 0, 0), (0, 3, 0), (0, 0, 3))
            + ((1, 1, 1),) * 5,
        ((3, 0, 0), (0, 2, 1), (0, 1, 2))
            + ((1, 1, 1),) * 5,
        ((3, 0, 0), (0, 3, 0), (0, 1, 2), (1, 0, 2))
            + ((1, 1, 1),) * 4,
    )
    counts = {"A": acatalogue[aprofile],
              "B": bcatalogue[bprofile]}
    sa = [[c[i] for c in counts["A"]] for i in range(ra)]
    sb = [[c[i] for c in counts["B"]] for i in range(rb)]
    assert [sum(row) for row in sa] == [9, 9]
    assert [sum(row) for row in sb] == [8, 8, 8]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert gf2_rank([
        sum((c[i] & 1) << i for i in range(rb))
        for c in counts["B"]]) == rb

    # Ratios between gaps on different check rows are not changed by
    # a common cyclic automorphism. The two cards use ratios 2 and 3.
    first_top = 1 if aprofile == 0 else 2
    first_bottom = 2 if aprofile == 0 else 4
    fixed_gaps = {(0, first_top): 1,
                  (1, first_bottom): 2 + geometry}

    def rotate(x, t):
        return ((x >> t) | (x << (q - t))) & mask

    # For prime q, every nonconstant binary block has a unique orientation.
    shifts = [0] * (1 << q)
    for x in range(1, mask):
        shifts[x] = min(range(q), key=lambda t: rotate(x, t))

    def normalize_syndrome(s, height):
        t = 0
        for row in range(height):
            block = (s >> (row * q)) & mask
            if block not in (0, mask):
                t = shifts[block]
                break
        key = sum(rotate((s >> (row * q)) & mask, t) << (row * q)
                  for row in range(height))
        return key, t

    def normalize_word(word, width):
        first = (word & -word).bit_length() - 1
        block = first // q
        t = shifts[(word >> (block * q)) & mask]
        return sum(rotate((word >> (j * q)) & mask, t) << (j * q)
                   for j in range(width))

    def packed(side):
        """Pack within-row and oriented cross-row check-pair orbits."""
        profile = counts[side]
        height = len(profile[0])
        width = len(profile)
        row_pairs = [(a, b) for a in range(height)
                     for b in range(a + 1, height)]
        for restart in range(24):
            used = [set() for _ in range(height)]
            cross_used = {pair: set() for pair in row_pairs}
            out = [None] * width
            order = list(range(width))
            rng.shuffle(order)
            order.sort(key=lambda j: -max(profile[j]))
            for j in order:
                chosen = None
                for attempt in range(128):
                    vals = [rng.sample(range(q), c) for c in profile[j]]
                    if side == "A":
                        for i in range(height):
                            gap = fixed_gaps.get((i, j))
                            if gap is not None:
                                base = rng.randrange(q)
                                vals[i] = [base, (base + gap) % q]
                    anchor = next(row[0] for row in vals if row)
                    vals = [sorted((v - anchor) % q for v in row)
                            for row in vals]
                    local = []
                    legal = True
                    for i, row in enumerate(vals):
                        ds = []
                        for a in range(len(row)):
                            for b in range(a):
                                d = (row[a] - row[b]) % q
                                ds.append(min(d, q - d))
                        if len(ds) != len(set(ds)) or set(ds) & used[i]:
                            legal = False
                            break
                        local.append(set(ds))
                    if not legal:
                        continue
                    cross = {}
                    for a, b in row_pairs:
                        ds = [(x - y) % q
                              for x in vals[a] for y in vals[b]]
                        if (len(ds) != len(set(ds))
                                or set(ds) & cross_used[(a, b)]):
                            legal = False
                            break
                        cross[(a, b)] = set(ds)
                    if not legal:
                        continue
                    for i in range(height):
                        used[i].update(local[i])
                    for pair in row_pairs:
                        cross_used[pair].update(cross[pair])
                    chosen = vals
                    break
                if chosen is None:
                    break
                out[j] = chosen
            if all(column is not None for column in out):
                return [[[G.elements[v] for v in out[j][i]]
                         for j in range(width)] for i in range(height)]
        return None

    zero_A = [[[] for _ in range(ca)] for _ in range(ra)]
    zero_B = [[[] for _ in range(cb)] for _ in range(rb)]

    def component_columns(side, grid):
        """Extract actions solely from the scaffold's verified matrices."""
        if side == "A":
            hx, _, _ = lifted_matrices(G, grid, zero_B)
            width, height = ca, ra
            rows = []
            for i in range(ra):
                for t in range(q):
                    source = hx[(i * cb) * q + t]
                    rows.append(sum(
                        ((source >> (j * cb * q)) & mask) << (j * q)
                        for j in range(ca)))
        else:
            _, hz, _ = lifted_matrices(G, zero_A, grid)
            width, height = cb, rb
            rows = [row & ((1 << (width * q)) - 1)
                    for row in hz[:height * q]]
        cols = [0] * (width * q)
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        assert all(s.bit_count() == 3 for s in cols)
        # Validate the symmetry used below on the actual extracted action.
        for j in range(width):
            base = cols[j * q]
            for t in range(q):
                expected = sum(
                    rotate((base >> (r * q)) & mask, (-t) % q)
                    << (r * q) for r in range(height))
                assert cols[j * q + t] == expected
        return cols

    def orbit_quality(cols, height, width, through_eight=False):
        """Exact component existence/count screening, not quantum distance.

        Every subset orbit has a representative with position zero in
        its first occupied protograph block. Equal normalized syndromes
        give kernel words. Cubic columns exclude odd kernel weights.
        """
        bits = [[1 << (j * q + (t - shift) % q)
                 for j in range(width) for t in range(q)]
                for shift in range(q)]
        seen = {0: 0}
        words = set()

        def insert(s, positions):
            key, shift = normalize_syndrome(s, height)
            word = 0
            for p in positions:
                word |= bits[shift][p]
            previous = seen.get(key)
            if previous is None:
                seen[key] = word
                return 0
            if previous == word:
                return 0
            difference = previous ^ word
            weight = difference.bit_count()
            if weight <= 4:
                return weight
            words.add(normalize_word(difference, width))
            return 0

        length = len(cols)
        if not through_eight:
            for a in range(0, length, q):
                hit = insert(cols[a], (a,))
                if hit:
                    return hit, 1
            for a in range(0, length, q):
                for b in range(a + 1, length):
                    hit = insert(cols[a] ^ cols[b], (a, b))
                    if hit:
                        return hit, 1
            for a in range(0, length, q):
                for b in range(a + 1, length - 1):
                    ab = cols[a] ^ cols[b]
                    for c in range(b + 1, length):
                        hit = insert(ab ^ cols[c], (a, b, c))
                        if hit:
                            return hit, 1
            return (6, len(words)) if words else (8, 0)

        # The caller has excluded all component words through six.
        # Then each quartet syndrome has at most two distinct support
        # representatives: three would imply a word of weight <=6.
        seen.clear()
        for a in range(0, length, q):
            for b in range(a + 1, length - 2):
                ab = cols[a] ^ cols[b]
                for c in range(b + 1, length - 1):
                    abc = ab ^ cols[c]
                    for d in range(c + 1, length):
                        hit = insert(abc ^ cols[d], (a, b, c, d))
                        if hit:
                            raise ValueError("component screening mismatch")
        return (8, len(words)) if words else (10, 0)

    def optimize(side):
        height = ra if side == "A" else rb
        width = ca if side == "A" else cb
        finalists = []
        signatures = set()
        base_budget = 112 if side == "A" else 80
        for batch in range(2):
            for attempt in range(base_budget):
                grid = packed(side)
                if grid is None:
                    continue
                signature = tuple(
                    tuple(sorted(G.idx(g) for g in cell))
                    for row in grid for cell in row)
                if signature in signatures:
                    continue
                signatures.add(signature)
                cols = component_columns(side, grid)
                if gf2_rank(cols) != height * q:
                    continue
                floor, multiplicity = orbit_quality(cols, height, width)
                if floor <= 4:
                    continue
                key = (floor, -multiplicity)
                finalists.append((key, signature, grid, cols))
                finalists.sort(key=lambda item: (item[0], item[1]),
                               reverse=True)
                del finalists[4:]
            if finalists and finalists[0][0][0] == 8:
                break
        if not finalists:
            raise ValueError("no full-rank packed component passed screening")

        winner = None
        for key, signature, grid, cols in finalists:
            if key[0] == 8:
                floor, multiplicity = orbit_quality(
                    cols, height, width, through_eight=True)
                deep_key = (floor, -multiplicity)
            else:
                deep_key = key
            if winner is None or deep_key > winner[0]:
                winner = (deep_key, grid)
        return winner[1], winner[0]

    A, pa = optimize("A")
    B, pb = optimize("B")
    hx, hz, n = lifted_matrices(G, A, B)
    k = n - gf2_rank(hx) - gf2_rank(hz)
    kb = q * (ca - ra) * (cb - rb)
    assert n == 594 and kb == 220 and k == kb
    assert 325 * n <= 1000 * k < 375 * n
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert css_ok(hx, hz)
    return {"idx": idx, "cells": {"A": A, "B": B},
            "component_screen": (pa, pb),
            "card": (aprofile, bprofile, geometry)}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete new-shape cell without further random labeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled supports with uniform labels do not reliably eliminate weight-5 sector-1 logicals on Z17 (2,5)x(2,6): all five measured parent slots have d30s=5.
# REFUTED: Difference packing and component screening through six remove the weight-eight plateau on Z17 (2,5)x(2,6): all five measured current-parent codes still had d30s=8, with minimum witnesses exclusively in sector 1.
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

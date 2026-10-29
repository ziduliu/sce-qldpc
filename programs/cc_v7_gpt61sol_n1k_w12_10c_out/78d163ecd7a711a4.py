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
CONCEPT = ("AGL(1,11), order 110, A=(1,3), B=(1,4): all-even supports A=(2,2,2), B=(2,2,2,4), with verified small-kernel exclusion, should produce five new measured readings >=6 with k>=667, improving the parent's 15.6254 fitness. Larger readings remain provisional.")
GROUP = ("Keep the frozen faithful affine C11:C10 group and order 110. This bed repeatedly measured distance six, whereas the current parent's selected readings were 6,6,6,6,5. Quotient parity supplies explicit constraints on actual syndrome columns; it is not assumed to improve distance without verification.")
SHAPES = ("Keep frozen A=(1,3), B=(1,4): n=1430, k>=660 and guaranteed rate 6/13 target cell 0.450. Components have 330 and 440 variables with 110 checks each. All-even cells add four explicit HX dependencies and three HZ dependencies, giving k>=667. Parent minima occupied only sector-1 Acol/Bcol blocks, prominently Acol1 across Bcol1..3. Exact ranks retain k/n<0.475.")
FILLING = ("Twenty cards cross two feasible primitive-multiplier A patterns with ten quartic quotient profiles different from the earlier quartic deck. A has three binomials; B has three binomials and one quartic, giving CSS maxima ten and twelve. Every binomial has odd quotient exponent; the quartic has three even and one odd exponent. Verified monomial templates supply component matrices by GF2 linearity. Require component rank 109, exclude kernels through four, and bound B-row overlap by four. Exact full ranks select pairs; full kernel-four tests and directly checked parity/overlap premises exclude weight five. Cycle objectives diversify independently constructed fillings. No random circuit reading is used as a distance certificate.")
HOW_AB = FILLING
FAMILY_NAME = "r045_affineF11_13x14_even_quartic_fullfive_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 110

def custom_ops(q):
    # Affine maps t -> 2^b*t + a over F11.
    # 2 has multiplicative order ten modulo 11.
    if not valid_q(q):
        raise ValueError("faithful affine-F11 bed requires q=110")
    powers = tuple(pow(2, b, 11) for b in range(10))
    els = [(a, b) for b in range(10) for a in range(11)]

    def red(g):
        return (g[0] % 11, g[1] % 10)

    def mul(g, h):
        return ((g[0] + powers[g[1]] * h[0]) % 11,
                (g[1] + h[1]) % 10)

    def inv(g):
        bi = (-g[1]) % 10
        return ((-powers[bi] * g[0]) % 11, bi)

    def idx(g):
        a, b = red(g)
        return 11 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


# Distinct allocations, not column permutations of one support.
# For single-row sides: wx=sum(A)+max(B), wz=sum(B)+max(A).
_DENSITY_CARDS = (
    ((3, 3, 3), (3, 2, 2, 2)),  # wx=12, wz=12
    ((4, 3, 2), (2, 2, 2, 2)),  # wx=11, wz=12
    ((3, 3, 2), (3, 2, 2, 2)),  # wx=11, wz=12
    ((3, 3, 3), (2, 2, 2, 2)),  # wx=11, wz=11
    ((2, 2, 2), (3, 3, 2, 2)),  # wx=9,  wz=12
)


def _bits(word):
    while word:
        low = word & -word
        yield low.bit_length() - 1
        word ^= low


def _affine_cells(rng, G, counts, mode, side):
    """Independent normalized directions; no orbit-copy construction."""
    patterns = ((0, 1, 2), (0, 1, 3),
                (0, 1, 4), (0, 2, 3))
    gaps = ((1, 1), (3, 3), (1, 3), (1, 7))[mode]
    fixed_points = rng.sample(range(11), len(counts))
    cells = []
    for j, count in enumerate(counts):
        if count == 2:
            gap = gaps[(j + side) % 2]
            # d fixes p and has order ten because gcd(gap,10)=1.
            p = fixed_points[j]
            slope = pow(2, gap, 11)
            d = (((1 - slope) * p) % 11, gap)
            first = G.pick(rng)
            # Normalizing on the right gives [1,d].
            cell = [first, G.mul(d, first)]
        else:
            if count == 3:
                bs = patterns[mode]
            else:
                # Four distinct slopes, with no order-two pair ratio.
                bs = (0, 1, 2, 3) if mode < 2 else (0, 1, 3, 4)
            shift = rng.randrange(10)
            cell = [(rng.randrange(11), (b + shift) % 10)
                    for b in bs[:count]]
        assert len({G.idx(g) for g in cell}) == count
        cells.append(cell)
    return [cells]


def _component_data(rows, width):
    """Statistics of components extracted from the verified LP lift."""
    columns = [0] * width
    for r, row in enumerate(rows):
        for v in _bits(row):
            columns[v] |= 1 << r
    if not all(columns) or len(set(columns)) != width:
        return None

    adjacency = [0] * len(rows)
    check_pairs = {}
    four = 0
    forced = 0
    for syndrome in columns:
        ns = list(_bits(syndrome))
        degree = len(ns)
        forced += degree * (degree - 1) * (degree - 2) // 6
        for a, u in enumerate(ns):
            adjacency[u] |= syndrome ^ (1 << u)
            for v in ns[:a]:
                key = v * len(rows) + u
                old = check_pairs.get(key, 0)
                four += old
                check_pairs[key] = old + 1

    visited = frontier = 1
    while frontier:
        reached = 0
        for u in _bits(frontier):
            reached |= adjacency[u]
        frontier = reached & ~visited
        visited |= frontier
    if visited != (1 << len(rows)) - 1:
        return None

    triangles = 0
    for u, neighbors in enumerate(adjacency):
        for v in _bits(neighbors & (-1 << (u + 1))):
            triangles += (neighbors & adjacency[v] &
                          (-1 << (v + 1))).bit_count()
    # Exact six-cycle count only when four=0; otherwise a surrogate.
    return columns, four, max(0, triangles - forced)


def _kernel_screen(columns, checks):
    """Exact <=4 screen; detected connected-triple five/six witnesses.

    Returning seven means these enumerations found no word <=6,
    not an exhaustive general weight-five/six exclusion.
    """
    width = len(columns)
    singles = set(columns)
    if 0 in singles:
        return 1
    if len(singles) != width:
        return 2

    pairs = {}
    for a, x in enumerate(columns):
        for b in range(a):
            syndrome = x ^ columns[b]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                return 4
            pairs[syndrome] = (1 << a) | (1 << b)

    incidence = [[] for _ in range(checks)]
    for v, syndrome in enumerate(columns):
        for r in _bits(syndrome):
            incidence[r].append(v)
    adjacency = [set() for _ in columns]
    for variables in incidence:
        for v in variables:
            adjacency[v].update(variables)
            adjacency[v].discard(v)

    seen = {}
    found_six = False
    for b, neighbors in enumerate(adjacency):
        ns = sorted(neighbors)
        for t, a in enumerate(ns):
            for c in ns[t + 1:]:
                word = (1 << a) | (1 << b) | (1 << c)
                syndrome = columns[a] ^ columns[b] ^ columns[c]
                pair = pairs.get(syndrome)
                if pair is not None and not (pair & word):
                    return 5
                old = seen.get(syndrome)
                if old is not None:
                    if old != word and not (old & word):
                        found_six = True
                else:
                    seen[syndrome] = word
    return 6 if found_six else 7


def _circuit_reading(rng, columns, trials):
    """Classical kernel upper reading, never a distance certificate."""
    best = len(columns) + 1
    order = list(range(len(columns)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        basis = []
        for v in order:
            syndrome, word = columns[v], 1 << v
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                basis.append(word)
                best = min(best, word.bit_count())
        light = sorted(basis, key=int.bit_count)[:16]
        for a in light:
            for b in basis:
                if a != b:
                    best = min(best, (a ^ b).bit_count())
        if best <= 4:
            break
    return best


def NEW_CTX(rng, G, idx):
    """Construct all-even cards and check their actual small-kernel premises."""
    q = G.q
    mask = (1 << q) - 1
    card, cohort = idx % 20, idx // 20
    pattern = ((1, 1, 3), (1, 3, 3))[card // 10]
    quartics = (
        (0, 0, 2, 1), (0, 0, 4, 3),
        (0, 2, 4, 3), (0, 2, 6, 1),
        (0, 2, 8, 3), (0, 4, 6, 1),
        (0, 4, 8, 3), (0, 2, 4, 5),
        (0, 2, 6, 5), (0, 4, 8, 5),
    )
    sa, sb = [[2, 2, 2]], [[2, 2, 2, 4]]
    assert _grid_weights(sa, sb) == 12
    assert W_MAX >= 12 and _no_dead_lines(sa, sb)

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            for v in _bits(row):
                out[v] |= 1 << r
        return out

    def rows_of(cols):
        rows = [0] * q
        for v, value in enumerate(cols):
            for r in _bits(value):
                rows[r] |= 1 << v
        return rows

    def free_four(cols):
        """Exhaustive nonzero kernel exclusion through weight four."""
        singles = set(cols)
        if 0 in singles or len(singles) != len(cols):
            return False
        pairs = set()
        for i, value in enumerate(cols):
            for j in range(i):
                syndrome = value ^ cols[j]
                if syndrome in singles or syndrome in pairs:
                    return False
                pairs.add(syndrome)
        return True

    # Obtain both elemental conventions from the authoritative helper.
    # XOR of these columns is GF2 linearity, not a separate group lift.
    hx, _, _ = lifted_matrices(
        G, [[[g] for g in G.elements]], [[[]]])
    _, hz, _ = lifted_matrices(
        G, [[[]]], [[[g] for g in G.elements]])
    templates = {}
    for side, source in (("A", hx), ("B", hz)):
        templates[side] = [
            columns([(row >> (e*q)) & mask for row in source], q)
            for e in range(q)
        ]

    charge = sum(1 << t for t, g in enumerate(G.elements) if g[1] & 1)

    def assemble(cells, side):
        cols = []
        for cell in cells:
            block = [0] * q
            for g in cell:
                action = templates[side][G.idx(g)]
                for t in range(q):
                    block[t] ^= action[t]
            cols.extend(block)
        return cols

    def cycle_stats(rows):
        """Exact Tanner C4/C6 counts, allowing multi-term cells."""
        common = [[0] * q for _ in range(q)]
        upper = [[] for _ in rows]
        four = six = 0
        overlap = 0
        for a in range(q):
            for b in range(a + 1, q):
                count = (rows[a] & rows[b]).bit_count()
                overlap = max(overlap, count)
                if count:
                    common[a][b] = count
                    upper[a].append(b)
                    four += count * (count - 1) // 2
        for a, neighbors in enumerate(upper):
            for t, b in enumerate(neighbors):
                ab = common[a][b]
                for c in neighbors[t + 1:]:
                    bc = common[b][c]
                    if bc:
                        ac = common[a][c]
                        triple = (rows[a] & rows[b] & rows[c]).bit_count()
                        six += ab*ac*bc - triple*(ab+ac+bc-2)
        return four, six, overlap

    def inspect(cells, side):
        cols = assemble(cells, side)
        # Check the quotient-character assertion on VERIFIED columns.
        if not all((value & charge).bit_count() & 1 for value in cols):
            return None
        if not free_four(cols):
            return None
        rows = rows_of(cols)
        if gf2_rank(rows) != q - 1:
            return None
        stats = cycle_stats(rows)
        if side == "B" and stats[2] > 4:
            return None
        if side == "A":
            # Degree-two component graph: closed walks discriminate
            # genuine relation geometries, rather than labeling copies.
            adjacency = [[] for _ in range(q)]
            for value in cols:
                a, b = list(_bits(value))
                adjacency[a].append(b)
                adjacency[b].append(a)
            counts = [0] * q
            counts[0] = 1
            spectrum = []
            for length in range(1, 9):
                nxt = [0] * q
                for a, count in enumerate(counts):
                    for b in adjacency[a]:
                        nxt[b] += count
                counts = nxt
                if length in (6, 8):
                    spectrum.append(counts[0])
            objective = (spectrum[0], spectrum[1])
        elif (card + cohort) % 3 == 0:
            objective = (stats[1], stats[0], stats[2])
        elif (card + cohort) % 3 == 1:
            objective = (stats[0], stats[1], stats[2])
        else:
            objective = (stats[1] + 8*stats[0], stats[2], stats[0])
        return {"cells": cells, "cols": cols, "rows": rows,
                "stats": stats, "objective": objective}

    # Conjugation can normalize two distinct affine fixed points.
    # Exhaust the remaining parameter instead of repeatedly drawing
    # automorphism copies of a small A-component library.
    apool = []
    order = list(range(11))
    rng.shuffle(order)
    for a in order:
        cells = [[G.e, (0, pattern[0])],
                 [G.e, (1, pattern[1])],
                 [G.e, (a, pattern[2])]]
        record = inspect(cells, "A")
        if record is not None:
            apool.append(record)
    if not apool:
        raise ValueError("no normalized A graph passed exact kernel-four")
    apool.sort(key=lambda record: record["objective"])
    apool = apool[:4]

    def draw_b(profile):
        # Distinct fixed points prevent pairwise commuting directions.
        points = rng.sample(range(11), 3)
        bins = pattern if cohort % 2 == 0 else pattern[::-1]
        cells = []
        for p, b in zip(points, bins):
            g = (((1 - pow(2, b, 11))*p) % 11, b)
            cells.append([G.e, g])
        bs = quartics[profile]
        for _ in range(80):
            cell = [G.e] + [(rng.randrange(11), b) for b in bs[1:]]
            if len({G.idx(g) for g in cell}) == 4:
                cells.append(cell)
                return cells
        raise ValueError("distinct quartic construction failed")

    bpool, signatures = [], set()
    for trial in range(768):
        # Enrich the named geometry first. If it is infeasible, retain
        # all parity premises while exploring the other NEW quartic cards.
        profile = card % 10 if trial < 384 else (
            card + 1 + trial // 32) % 10
        cells = draw_b(profile)
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for cell in cells)
        if signature in signatures:
            continue
        signatures.add(signature)
        record = inspect(cells, "B")
        if record is None:
            continue
        record["quartic_profile"] = profile
        bpool.append(record)
        if len(bpool) >= 16:
            break
    if not bpool:
        raise ValueError("no new B quartic passed verified screens")
    bpool.sort(key=lambda record: record["objective"])
    bpool = bpool[:4]

    def full_five_free(hx, hz, n, brows):
        """Full <=4 exclusion plus verified premises excluding weight five.

        HZ's character is odd on every column, excluding odd kernels.
        HX's character distinguishes its sectors. An odd kernel therefore
        has an odd number of sector-2 variables.

        One sector-2 column weighs ten, exceeding four degree-two columns.
        Three have XOR weight at least 30-2*(4+4+4)=6, exceeding two
        degree-two columns. Five cannot sum to zero: the B transpose
        kernel is exactly the all-ones word of weight q.
        """
        xc, zc = columns(hx, n), columns(hz, n)
        n1 = 12*q
        xcharge = sum(charge << (b*q) for b in range(4))
        zcharge = sum(charge << (j*q) for j in range(3))
        if not all((value & zcharge).bit_count() & 1 for value in zc):
            return False
        if not all(value.bit_count() == 2 and
                   (value & xcharge).bit_count() & 1
                   for value in xc[:n1]):
            return False
        if not all(value.bit_count() == 10 and
                   not ((value & xcharge).bit_count() & 1)
                   for value in xc[n1:]):
            return False
        if xc[n1:] != brows or gf2_rank(brows) != q - 1:
            return False
        total = 0
        for row in brows:
            total ^= row
        if total:
            return False
        if max((row & other).bit_count()
               for i, row in enumerate(brows)
               for other in brows[:i]) > 4:
            return False
        return free_four(xc) and free_four(zc)

    proposals = []
    for a in apool:
        for b in bpool:
            A, B = [a["cells"]], [b["cells"]]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k >= 667 and 425*n <= 1000*k < 475*n):
                continue
            key = (-k, a["objective"], b["objective"])
            proposals.append((key, a, b, A, B, hx, hz, n, k))
    proposals.sort(key=lambda item: item[0])

    for key, a, b, A, B, hx, hz, n, k in proposals:
        if not full_five_free(hx, hz, n, b["rows"]):
            continue
        # Confirm cached GF2 superposition against a fresh full lift.
        ar = []
        for source in hx[:q]:
            ar.append(sum(((source >> (j*4*q)) & mask) << (j*q)
                          for j in range(3)))
        br = [row & ((1 << (4*q)) - 1) for row in hz[:q]]
        assert ar == a["rows"] and br == b["rows"]
        assert n == 1430 and max(row.bit_count() for row in hx + hz) <= 12
        assert css_ok(hx, hz)
        return {"idx": idx, "card": card,
                "cells": {"A": A, "B": B}, "k": k,
                "cycles": (a["stats"], b["stats"]),
                "quartic_profile": b["quartic_profile"],
                "screen": "verified full kernels through five excluded"}
    raise ValueError("no all-even pair passed the full small-kernel/rate gate")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete verified binomial or quartic, without new draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolled supports with uniform C2 x C22 labels reliably suppress sector-1 weight-six logicals; all five measured parent codes instead had distance six and exclusively sector-1 minimum witnesses.
# REFUTED: Affine order-ten density cards with component geometry and circuit selection yield five readings >=8; the measured top five instead returned 6,6,6,6,5, with abundant sector-1 witnesses.
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

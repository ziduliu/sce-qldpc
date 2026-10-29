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
CONCEPT = ("S3 x C4, order 24, shapes (1,4)x(1,5): an exhaustive inverse-distinct binomial atlas should provide five fresh codes with no logical below weight three and k>=304, improving the current 5.3966 fitness.")
GROUP = ("Frozen L1 group and order. This exact bed measured [504,304,3] codes and archived distance-four leaders. Its eight non-involutory inverse classes permit a finite architecture census; rank surplus alone did not preserve distance four.")
SHAPES = ("Frozen (1,4)x(1,5) for open-ended cell 0.550: n=504, kb=288, guaranteed rate 4/7. Components have 96 and 120 variables with 24 checks each; q<=28 fits the size window, while this round retains q=24. Current weight-three witnesses occupied sector-1 Acol/Bcol blocks broadly on both sides.")
FILLING = ("Enumerate all 70 four-class A subsets and 56 five-class B subsets, excluding the archived all-odd family. Reconstruct current outputs solely for archive screening and blacklist shown cyclic and mixed-binomial controls. Verified lifts determine ranks and logical floors. Strong uncoloured weighted-check-intersection closed-walk signatures separate emitted geometries. Five fresh [504,304,3] codes would score 5.4286; neither archive novelty nor beating the 9.429 frontier is assumed.")
FAMILY_NAME = "r055_S3xC4_14x15_exhaustive_binomial_walkatlas_v1"

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
from itertools import combinations_with_replacement


def _rf_columns(rows, n):
    cols = [0] * n
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            lo = row & -row
            cols[lo.bit_length() - 1] |= check
            row ^= lo
    return cols


def _rf_logical_floor(checks, stabilizers, n, through_four=False):
    """Exact logical minimum through three/four.

    Return 4/5 respectively when no logical was found through the
    screened range. These sentinels are lower floors, not upper readings.
    Equal-syndrome representatives can be collapsed only after checking
    that their difference belongs to the stabilizer span.
    """
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old

    def outside(word):
        while word:
            p = word.bit_length() - 1
            old = basis.get(p)
            if old is None:
                return True
            word ^= old
        return False

    cols = _rf_columns(checks, n)
    bits = [1 << i for i in range(n)]
    for i, syndrome in enumerate(cols):
        if syndrome == 0 and outside(bits[i]):
            return 1

    singles = {0: 0}
    for i, syndrome in enumerate(cols):
        old = singles.get(syndrome)
        if old is None:
            singles[syndrome] = bits[i]
        elif outside(bits[i] ^ old):
            return 2

    # Pair/single collisions cover all supports through weight three.
    for i, syndrome in enumerate(cols):
        for j in range(i):
            old = singles.get(syndrome ^ cols[j])
            if old is not None and outside(bits[i] ^ bits[j] ^ old):
                return 3

    if not through_four:
        return 4

    # After excluding logicals through three, one representative per
    # pair syndrome suffices even when low-weight stabilizers exist.
    seen = dict(singles)
    for i, syndrome in enumerate(cols):
        for j in range(i):
            s = syndrome ^ cols[j]
            word = bits[i] | bits[j]
            old = seen.get(s)
            if old is None:
                seen[s] = word
            elif outside(word ^ old):
                return 4
    return 5


def _rf_signature(hx, hz, n):
    """Conservative invariant, including a possible global X/Z swap.

    Different signatures prove nonisomorphism. Equal signatures are
    discarded conservatively; they do not prove isomorphism.
    """
    xc, zc = _rf_columns(hx, n), _rf_columns(hz, n)
    qubits = tuple(sorted(
        (x.bit_count(), z.bit_count()) for x, z in zip(xc, zc)))
    swapped = tuple(sorted((z, x) for x, z in qubits))

    def check_profile(rows, opposite):
        own_weights = [r.bit_count() for r in rows]
        other_weights = [r.bit_count() for r in opposite]
        profiles = []
        for i, row in enumerate(rows):
            own, other = {}, {}
            for j, neighbor in enumerate(rows):
                if i == j:
                    continue
                key = (own_weights[j], (row & neighbor).bit_count())
                own[key] = own.get(key, 0) + 1
            for j, neighbor in enumerate(opposite):
                key = (other_weights[j], (row & neighbor).bit_count())
                other[key] = other.get(key, 0) + 1
            profiles.append((own_weights[i],
                             tuple(sorted(own.items())),
                             tuple(sorted(other.items()))))
        return tuple(sorted(profiles))

    return (min(qubits, swapped),
            tuple(sorted((check_profile(hx, hz),
                          check_profile(hz, hx)))))


@lru_cache(maxsize=1)
def _rf_bank(G, slots):
    """Pure deterministic search; caching only avoids repeated work."""
    search = random.Random(550240417)
    q = G.q
    assert q == 24
    elements = list(G.elements)

    # These descriptors choose labels, never implement a second lift.
    representatives = []
    for g in elements:
        if g == G.e or G.mul(g, g) == G.e:
            continue
        if G.idx(g) <= G.idx(G.inv(g)):
            representatives.append(g)
    odd = [g for g in representatives if g[2] & 1]
    involutions = [g for g in elements
                   if g != G.e and G.mul(g, g) == G.e]

    # Exclude every known principal parent orbit by an actual
    # permutation-invariant signature, not by an element-label hash.
    archived = set()
    generators = [(a, b, 1) for b in range(2) for a in range(3)]
    for missing_a in ((0, 3), (0, 1), (3, 4), (3, 1), (1, 2)):
        for missing_b in (0, 3, 1):
            A = [[[G.e, g] for t, g in enumerate(generators)
                  if t not in missing_a]]
            B = [[[G.e, g] for t, g in enumerate(generators)
                  if t != missing_b]]
            hx, hz, n = lifted_matrices(G, A, B)
            archived.add(_rf_signature(hx, hz, n))

    profiles = []
    for aw in combinations_with_replacement(range(1, 6), 4):
        for bw in combinations_with_replacement(range(1, 6), 5):
            if sum(aw) + max(bw) > W_MAX:
                continue
            if sum(bw) + max(aw) > W_MAX:
                continue
            profiles.append((aw, bw))
    profiles.sort(key=lambda p: (
        -min(p[0]) - min(p[1]),
        -min(p[0]), -sum(p[0]) - sum(p[1]), p))
    limit = min(len(profiles), max(28, slots))
    profiles = profiles[:limit]

    def draw_side(counts, model):
        row = []
        if all(c == 2 for c in counts):
            if model == 0:
                pool = odd
            elif model == 1:
                pool = representatives
            elif model == 2:
                pool = representatives + involutions
            else:
                pool = [g for g in elements if g != G.e]
            steps = search.sample(pool, len(counts))
            return [[[G.e, g] for g in steps]]

        for j, count in enumerate(counts):
            if count == 1:
                cell = [search.choice(elements)]
            elif count == 2:
                pool = (odd if model == 0 else representatives
                        if model == 1 else involutions
                        if model == 2 else elements[1:])
                cell = [G.e, search.choice(pool)]
            elif model == 0:
                # Odd-central terms plus an even anchor.
                cell = [G.e] + search.sample(
                    [g for g in elements if g[2] & 1], count - 1)
            elif model == 1:
                # Prescribed mixed central parity, without cancellation.
                amount = 1 + j % (count - 1)
                cell = (search.sample(
                            [g for g in elements if not (g[2] & 1)],
                            amount)
                        + search.sample(
                            [g for g in elements if g[2] & 1],
                            count - amount))
            elif model == 2:
                # Permit central-pair dependencies: exact logical
                # screening decides whether they are harmful.
                g = search.choice(elements)
                h = G.mul(g, (0, 0, 2))
                remaining = [u for u in elements if u not in (g, h)]
                cell = [g, h] + search.sample(remaining, count - 2)
            else:
                cell = search.sample(elements, count)
            assert len(cell) == len(set(cell))
            row.append(cell)
        return [row]

    keep = {}
    seen_labels = set()
    capacity = max(100, 4 * slots)
    principal = ((2, 2, 2, 2), (2, 2, 2, 2, 2))
    schedule = []
    # Search internal geometry deeply on the distance-four architecture,
    # but also examine genuine count/rank architectures.
    for t in range(max(512, 16 * slots)):
        schedule.append((principal, t % 4))
    for p in profiles:
        for t in range(32):
            schedule.append((p, t % 4))

    for (aw, bw), model in schedule:
        A, B = draw_side(aw, model), draw_side(bw, model)
        label_key = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for grid in (A, B)
                          for row in grid for cell in row)
        if label_key in seen_labels:
            continue
        seen_labels.add(label_key)

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        fx = _rf_logical_floor(hz, hx, n)
        fz = _rf_logical_floor(hx, hz, n)
        floor = min(fx, fz)
        provisional = (k * floor * floor, floor, k)
        if len(keep) >= capacity:
            worst = min(item[0][:3] for item in keep.values())
            # Passing through three may still improve through four.
            if floor < 4 and provisional < worst:
                continue

        signature = _rf_signature(hx, hz, n)
        if signature in archived:
            continue

        if floor == 4:
            fx = _rf_logical_floor(hz, hx, n, through_four=True)
            fz = _rf_logical_floor(hx, hz, n, through_four=True)
            floor = min(fx, fz)
        # This is a screening-floor objective, NOT a reported distance.
        objective = (k * floor * floor, floor, k, fx + fz,
                     -sum(row.bit_count() ** 2 for row in hx + hz))
        previous = keep.get(signature)
        if previous is None or objective > previous[0]:
            keep[signature] = (objective, A, B, k, floor)
        if len(keep) > capacity:
            weakest = min(keep, key=lambda key: keep[key][0])
            del keep[weakest]

    selected = sorted(keep.values(), key=lambda item: item[0],
                      reverse=True)[:slots]
    if len(selected) < min(5, slots):
        raise ValueError("fewer than five new invariant-distinct fillings")

    bank = []
    for objective, A, B, k, floor in selected:
        hx, hz, n = lifted_matrices(G, A, B)
        assert css_ok(hx, hz)
        assert n - gf2_rank(hx) - gf2_rank(hz) == k
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        # Immutable storage makes the cache observationally transparent.
        bank.append((tuple(tuple(tuple(cell) for cell in row) for row in A),
                     tuple(tuple(tuple(cell) for cell in row) for row in B),
                     k, floor))
    return tuple(bank)


def _atlas_signature(hx, hz, n):
    """Uncoloured Tanner invariants, strengthened by closed walks.

    The weighted check-intersection graph is obtained from actual lifted
    rows. Its second, third and fourth closed-walk counts, and squared-
    adjacency profiles, are invariant under arbitrary Tanner relabeling.
    Unequal signatures certify nonisomorphism; equality is rejected
    conservatively and is never asserted to prove isomorphism.
    """
    from collections import Counter

    rows = list(hx) + list(hz)
    weights = [row.bit_count() for row in rows]
    adjacency = [{} for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                adjacency[i][j] = overlap
                adjacency[j][i] = overlap

    profiles = []
    for i, neighbours in enumerate(adjacency):
        square = {}
        for j, multiplicity in neighbours.items():
            for t, other in adjacency[j].items():
                square[t] = square.get(t, 0) + multiplicity * other

        neighbour_profile = Counter(
            (weights[j], multiplicity)
            for j, multiplicity in neighbours.items())
        square_profile = Counter(
            (weights[t], value) for t, value in square.items())
        closed_three = sum(
            neighbours.get(t, 0) * value
            for t, value in square.items())
        closed_four = sum(value * value for value in square.values())
        profiles.append((
            weights[i],
            tuple(sorted(neighbour_profile.items())),
            square.get(i, 0),
            closed_three,
            closed_four,
            tuple(sorted(square_profile.items())),
        ))

    incidents = [[] for _ in range(n)]
    for weight, source in zip(weights, rows):
        while source:
            bit = source & -source
            incidents[bit.bit_length() - 1].append(weight)
            source ^= bit

    return (
        tuple(sorted(Counter(profiles).items())),
        tuple(sorted(Counter(
            tuple(sorted(profile)) for profile in incidents).items())),
    )


@lru_cache(maxsize=1)
def _atlas_bank(G, slots):
    """Pure finite census; prior matrices are used only as a blacklist."""
    from itertools import combinations

    assert G.q == 24
    representatives = tuple(
        g for g in G.elements
        if g != G.e and G.mul(g, g) != G.e
        and G.idx(g) < G.idx(G.inv(g)))
    assert len(representatives) == 8

    def cells(side):
        return (tuple((G.e, g) for g in side),)

    archived = set()

    def blacklist(A, B):
        hx, hz, n = lifted_matrices(G, A, B)
        archived.add(_atlas_signature(hx, hz, n))

    # The displayed current program was evaluated with sixty slots.
    # Reconstructing its deterministic outputs is archive screening only:
    # none of these matrices is used as an emitted candidate or fallback.
    for A, B, k, floor in _rf_bank(G, max(60, slots)):
        blacklist(A, B)

    # Shown C12 controls: all five B inverse classes and each A deletion.
    # Their apparently different omitted-step geometries included actual
    # Tanner isomorphisms, so blacklist matrices rather than label hashes.
    generator = (1, 0, 1)
    powers = [G.e]
    for exponent in range(1, 12):
        powers.append(G.mul(powers[-1], generator))
    assert len(set(powers)) == 12
    for omitted in range(1, 6):
        blacklist(
            cells(tuple(powers[t] for t in range(1, 6)
                        if t != omitted)),
            cells(tuple(powers[t] for t in range(1, 6))),
        )

    # Other displayed all-binomial controls from the witness-cutwalk
    # program. Nonuniform-count controls have different check-degree
    # profiles and cannot be isomorphic to this all-binomial atlas.
    shown = (
        (
            ((1,0,3), (1,0,1), (0,1,1), (2,0,2)),
            ((2,0,1), (2,1,3), (0,0,3), (1,1,3), (0,1,3)),
        ),
        (
            ((1,0,3), (1,0,1), (1,1,3), (2,0,2)),
            ((2,0,1), (2,1,1), (0,0,3), (2,0,3), (0,1,3)),
        ),
        (
            ((2,0,0), (1,0,1), (0,1,3), (2,1,1)),
            ((1,0,3), (2,0,3), (0,0,1), (2,1,3), (0,1,3)),
        ),
        (
            ((1,1,3), (1,0,0), (2,0,2), (2,0,1)),
            ((0,1,1), (2,0,3), (2,1,3), (1,0,3), (0,0,1)),
        ),
    )
    for left, right in shown:
        blacklist(cells(left), cells(right))

    alib = tuple(combinations(representatives, 4))
    blib = tuple(combinations(representatives, 5))
    assert len(alib) == 70 and len(blib) == 56

    plans = []
    for left in alib:
        for right in blib:
            # The six all-odd inverse classes and their four/five-element
            # deletion codes have already been exhaustively archived.
            if all(g[2] & 1 for g in left + right):
                continue
            A, B = cells(left), cells(right)
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            assert n == 504 and k >= 288
            label_key = (
                tuple(G.idx(g) for g in left),
                tuple(G.idx(g) for g in right),
            )
            plans.append((k, label_key, A, B))

    # Rank-ordering is appropriate only after the short-logical screen.
    # Store compact plans, not thousands of full matrix pairs.
    plans.sort(key=lambda item: (-item[0], item[1]))
    fingerprints = set()
    keep = []

    for k, label_key, A, B in plans:
        hx, hz, n = lifted_matrices(G, A, B)
        signature = _atlas_signature(hx, hz, n)
        if signature in archived or signature in fingerprints:
            continue

        fx = _rf_logical_floor(hz, hx, n)
        fz = _rf_logical_floor(hx, hz, n)
        floor = min(fx, fz)
        if floor < 3:
            continue
        if floor >= 4:
            fx = _rf_logical_floor(hz, hx, n, through_four=True)
            fz = _rf_logical_floor(hx, hz, n, through_four=True)
            floor = min(fx, fz)

        # These are exhaustive small-weight floors, not GPU distance
        # readings. Actual short kernels remain subject to the exact
        # stabilizer-quotient tests above.
        objective = (k * floor * floor, floor, k, fx + fz)
        fingerprints.add(signature)
        keep.append((objective, A, B, k, floor))
        keep.sort(key=lambda item: item[0], reverse=True)
        del keep[slots:]

        # Every remaining code has no larger k. If the retained codes
        # all have floor three, retain a margin of additional architectures
        # before stopping; do not stop on just five lucky discoveries.
        if (len(keep) == slots and floor == 3
                and k < keep[-1][3]
                and len(fingerprints) >= max(120, 3 * slots)):
            break

    if len(keep) < min(5, slots):
        raise ValueError("fewer than five fresh atlas geometries")

    bank = []
    for objective, A, B, k, floor in keep:
        hx, hz, n = lifted_matrices(G, A, B)
        assert css_ok(hx, hz)
        assert n - gf2_rank(hx) - gf2_rank(hz) == k
        assert _grid_weights([[2]*4], [[2]*5]) == 12
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        assert all(len(cell) == len(set(cell)) == 2
                   for grid in (A, B) for row in grid for cell in row)
        bank.append((A, B, k, floor))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign a fresh exhaustive-atlas geometry, never an orbit fallback."""
    bank = _atlas_bank(G, N_SLOTS)
    if idx >= len(bank):
        raise ValueError("no further fresh atlas geometry")
    A, B, k, floor = bank[idx]
    return {"cells": {"A": A, "B": B}, "k": k,
            "exact_small_logical_floor": floor}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete screened atlas cell without further labeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Uniform legal support rolling on C17, (2,5)x(2,6), did not suppress sector-1 weight-five logicals: all five measured parent codes had d30s=5.
# REFUTED: Fifteen odd-central deletion cards are fifteen distinct codes: parent slots 0-14 collapsed to five isomorphism classes, and all five measured readings remained four.
# REFUTED: Unscreened mixed-count support experiments improve this bed's leaders: parent slots 15-59 had d2s at most three and did not advance over the distance-four principal codes.
# REFUTED: Random rank-flex logical screening supplies fresh distance-four leaders on this bed: all five current measured codes had d30s=3, and fitness declined from 9.4290 to 5.3966 despite rank surpluses of 13-16.
# REFUTED: Four C12 omitted-step geometries provide four inequivalent rank-306 codes: the shown cyclic program's slots 1-3 were isomorphic duplicates of slot 0.
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

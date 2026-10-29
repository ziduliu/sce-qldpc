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
CONCEPT = ("On cyclic C17, order 17, A=(2,5), B=(2,6), independently packed cubic-column constituents and exact component-six screening should produce five fresh distance readings at least ten, exceeding the 0.350 frontier fitness 23.5.")
GROUP = ("L2 order repair within the unchanged cyclic construction rule. The new factor 34 excludes order 23 because n would be 782; order 17 is the largest admissible lift and gives n=578. This exact bed previously reached about 23.0. That motivates deeper constituent selection, not a presumed cyclic-group advantage.")
SHAPES = ("Target 0.350 rather than the requested half-rate cell: A=(2,5), B=(2,6), n=578, kb=204 and guaranteed rate 6/17=0.352941. Components have 85 variables/34 checks and 102 variables/34 checks. Both sides now have two check rows, removing the parent's one-row commuting-trinomial ceiling. Parent minima were entirely sector 1, especially Acol1 with Bcol2..6. Verified complete ranks require 0.325<=k/n<0.375.")
FILLING = ("Twenty cards vary nine cubic-column support geometries and intrinsic inverse-gap multiplicities. A row totals are seven/eight and B row totals nine; every component column has three terms, hence HX<=11 and HZ<=12. Compile actions exclusively from verified unit lifts, exclude component words through four, test six exactly, optimize retained genuine kernel witnesses and independently deepen finalists. Screen complete logical quotients through four, validate embedded constituent witnesses and search stabilizer dressings. Unequal uncoloured incidence signatures separate emitted designs. Restricted exclusions and sampled readings are not quantum-distance certificates.")
FAMILY_NAME = "r035_C17_25x26_cubic_column_exact6_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 17

def custom_ops(q):
    if q != 17:
        raise ValueError("this cyclic bed requires order 17")
    els = [(a,) for a in range(q)]

    def red(g):
        return (g[0] % q,)

    def mul(g, h):
        return ((g[0] + h[0]) % q,)

    def inv(g):
        return ((-g[0]) % q,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul,
                inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def _q85_basis(rows):
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in basis:
                row ^= basis[p]
            else:
                basis[p] = row
                break
    return basis


def _q85_outside(v, basis):
    while v:
        p = v.bit_length() - 1
        if p not in basis:
            return True
        v ^= basis[p]
    return False


def _q85_short_screen(hx, hz, n):
    """Exact existence test for nontrivial CSS logicals of weight <=4.

    Every such support splits into two supports of size <=2.
    Equal-syndrome collisions are tested modulo the actual stabilizer
    row space. Keeping one representative per syndrome is sufficient:
    if all collisions with it are stabilizers, their pairwise differences
    are stabilizers too.
    """
    cycles = 0
    for checks, stabilizers in ((hz, hx), (hx, hz)):
        basis = _q85_basis(stabilizers)
        cols = [0] * n
        for r, row in enumerate(checks):
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= 1 << r
                row ^= bit

        # Count actual Tanner 4-cycles, not protograph coincidences.
        overlaps = {}
        for col in cols:
            ids = []
            v = col
            while v:
                bit = v & -v
                ids.append(bit.bit_length() - 1)
                v ^= bit
            for a in range(len(ids)):
                for b in range(a):
                    key = (ids[b], ids[a])
                    old = overlaps.get(key, 0)
                    cycles += old
                    overlaps[key] = old + 1

        representatives = {0: 0}
        bits = [1 << i for i in range(n)]
        for i, syndrome in enumerate(cols):
            mask = bits[i]
            if syndrome in representatives:
                v = mask ^ representatives[syndrome]
                if _q85_outside(v, basis):
                    return v.bit_count(), cycles
            else:
                representatives[syndrome] = mask

        for i in range(n):
            si, bi = cols[i], bits[i]
            for j in range(i):
                syndrome = si ^ cols[j]
                mask = bi | bits[j]
                if syndrome in representatives:
                    v = mask ^ representatives[syndrome]
                    if _q85_outside(v, basis):
                        return v.bit_count(), cycles
                else:
                    representatives[syndrome] = mask

    # Sentinel: no logical of weight <=4, not a distance estimate.
    return 5, cycles


def _z23_columns(rows, length):
    cols = [0] * length
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _z23_components(hx, hz, q):
    """Extract every component row from the scaffold's exact lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for r in range(q):
            row = hx[(i * cb) * q + r]
            compact = 0
            for j in range(ca):
                compact |= ((row >> (j * cb * q)) & mask) << (j * q)
            ar.append(compact)
    bmask = (1 << (cb * q)) - 1
    br = [row & bmask for row in hz[:rb * q]]
    return (_z23_columns(ar, ca * q),
            _z23_columns(br, cb * q))


def _z23_short(cols, through_five=False):
    """Exact existence screening, not a heuristic distance reading.

    Singles/pairs cover weight <=4. Triples against their representatives
    cover weight <=5 without storing the much larger triple dictionary.
    The sentinel is 5 or 6 when the requested weights were excluded.
    """
    seen = {0: 0}
    bits = [1 << i for i in range(len(cols))]
    for i, syndrome in enumerate(cols):
        word = bits[i]
        old = seen.get(syndrome)
        if old is not None:
            return (old ^ word).bit_count()
        seen[syndrome] = word
    for i, ci in enumerate(cols):
        for j in range(i):
            syndrome = ci ^ cols[j]
            word = bits[i] | bits[j]
            old = seen.get(syndrome)
            if old is not None:
                return (old ^ word).bit_count()
            seen[syndrome] = word
    if not through_five:
        return 5
    for i, ci in enumerate(cols):
        for j in range(i):
            syndrome = ci ^ cols[j]
            word = bits[i] | bits[j]
            for t in range(j):
                old = seen.get(syndrome ^ cols[t])
                if old is not None:
                    return (old ^ word ^ bits[t]).bit_count()
    return 6


def _z23_reading(cols, rng, trials):
    """Upper-bound readings from actual nonzero component-kernel words."""
    best = len(cols) + 1
    order = list(range(len(cols)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots, words = {}, []
        for i in order:
            syndrome, word = cols[i], 1 << i
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
        words.sort(key=int.bit_count)
        for i, word in enumerate(words):
            for other in words[:i]:
                best = min(best, (word ^ other).bit_count())
        small = words[:10]
        for i, word in enumerate(small):
            for j in range(i):
                pair = word ^ small[j]
                for t in range(j):
                    best = min(best, (pair ^ small[t]).bit_count())
        if best <= 4:
            break
    return best


def _z23_cycles(cols):
    total = 0
    for i, col in enumerate(cols):
        for other in cols[:i]:
            overlap = (col & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _z23_candidate(rng, G, sb, ap_count):
    """Fresh labels for the card, never an automorphism-copy generator."""
    q, cb = G.q, B_SHAPE[1]
    A = [[]]
    for j in range(A_SHAPE[1]):
        if j < ap_count:
            step = rng.randrange(1, q)
            offsets = (0, step, (2 * step) % q)
        else:
            while True:
                u, v = rng.sample(range(1, q), 2)
                # None of the three terms is an AP midpoint.
                if (v != (2 * u) % q and u != (2 * v) % q
                        and (u + v) % q != 0):
                    offsets = (0, u, v)
                    break
        base = G.pick(rng)
        A[0].append([G.mul(base, (a,)) for a in offsets])

    # Distinct relative labels prevent repeated all-monomial columns.
    relatives = rng.sample(range(q), cb)
    B = [[None] * cb for _ in range(2)]
    for j, relative in enumerate(relatives):
        top = G.pick(rng)
        bottom = G.mul(top, (relative,))
        B[0][j], B[1][j] = [top], [bottom]
    for i in range(2):
        for j in range(cb):
            if sb[i][j] == 2:
                delta = (rng.randrange(1, q),)
                B[i][j].append(G.mul(B[i][j][0], delta))
    return A, B


def _retired_zero3_context(rng, G, idx):
    """Inactive parent optimizer; never called for the new shapes."""
    recipes = (
        ((3, 0), (0, 3)) + ((1, 1),) * 5,
        ((3, 0), (1, 2), (1, 2)) + ((1, 1),) * 4,
        ((3, 0), (0, 3), (2, 1), (1, 2)) + ((1, 1),) * 3,
        ((3, 0), (3, 0), (0, 3), (0, 3)) + ((1, 1),) * 3,
        ((3, 0), (1, 2), (1, 2), (1, 2)) + ((1, 1),) * 3,
    )
    card = idx % 20
    columns = recipes[card // 4]
    mode = card % 4
    q, cb = G.q, B_SHAPE[1]
    sa = [[3, 3, 3]]
    sb = [[column[i] for column in columns] for i in range(2)]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    positions = [(i, j) for i in range(2) for j in range(cb)
                 if sb[i][j]]
    mono = [j for j in range(cb) if columns[j] == (1, 1)]

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def triple(ap):
        if ap:
            d = rng.randrange(1, q)
            offsets = (0, d, 2*d % q)
        else:
            while True:
                u, v = rng.sample(range(1, q), 2)
                if (v != 2*u % q and u != 2*v % q
                        and (u + v) % q != 0):
                    offsets = (0, u, v)
                    break
        return [(a,) for a in offsets]

    def fresh_A():
        # Modes alter intrinsic triple geometry, not global translations.
        ap_counts = (0, 1, 2, 3)
        return [[triple(j < ap_counts[mode]) for j in range(3)]]

    def relatives():
        # Prefer distinct ordered differences between mono columns.
        # A failed packing is not assumed good: exact screens follow.
        best, best_cost = None, None
        for _ in range(24):
            labels = rng.sample(range(q), len(mono))
            differences = {}
            for a in labels:
                for b in labels:
                    if a != b:
                        d = (a - b) % q
                        differences[d] = differences.get(d, 0) + 1
            cost = sum(t*(t-1)//2 for t in differences.values())
            if best_cost is None or cost < best_cost:
                best, best_cost = labels, cost
            if cost == 0:
                break
        return best

    def fresh_B():
        B = [[[] for _ in range(cb)] for _ in range(2)]
        for j, d in zip(mono, relatives()):
            B[0][j], B[1][j] = [G.e], [(d,)]
        for i, j in positions:
            if B[i][j]:
                continue
            count = sb[i][j]
            if count == 1:
                B[i][j] = [G.pick(rng)]
            elif count == 2:
                base = rng.randrange(q)
                delta = rng.randrange(1, q)
                B[i][j] = [(base,), ((base + delta) % q,)]
            else:
                # AP occupancy varies independently of A's policy.
                cell = triple((mode + i + j) % 4 == 0)
                base = G.pick(rng)
                B[i][j] = [G.mul(base, g) for g in cell]
        return B

    def signature(A, B):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for grid in (A, B) for row in grid for cell in row)

    def mutate(B):
        out = copy_grid(B)
        i, j = rng.choice(positions)
        cell = out[i][j]
        t = rng.randrange(len(cell))
        available = [g for g in G.elements if g not in cell]
        cell[t] = rng.choice(available)
        return out

    finalists, seen = [], set()
    elite_B, elite_key = None, None
    for attempt in range(128):
        A = fresh_A()
        # Keep independent restarts throughout; mutation is not a
        # generator of relabeled copies of the current elite.
        if elite_B is not None and attempt % 4:
            B = mutate(elite_B)
        else:
            B = fresh_B()
        sig = signature(A, B)
        if sig in seen:
            continue
        seen.add(sig)

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (425*n <= 1000*k < 475*n):
            continue
        ac, bc = _z23_components(hx, hz, q)
        qa, qb = _z23_short(ac), _z23_short(bc)
        ca, cb_cycles = _z23_cycles(ac), _z23_cycles(bc)

        db = (_z23_reading(bc, rng, 4) if qb == 5 else qb)
        bkey = (qb == 5, min(db, 6), -cb_cycles)
        if elite_key is None or bkey > elite_key:
            elite_B, elite_key = copy_grid(B), bkey

        if qa != 5 or qb != 5:
            continue
        da = _z23_reading(ac, rng, 4)
        d = min(da, db, 6)
        key = (k*d*d, d, min(db, 6), -(ca + cb_cycles))
        item = {
            "key": key, "A": A, "B": B, "hx": hx, "hz": hz,
            "n": n, "k": k, "ac": ac, "bc": bc,
            "da": da, "db": db, "cycles": ca + cb_cycles,
        }
        finalists.append(item)
        finalists.sort(key=lambda x: x["key"], reverse=True)
        del finalists[8:]

    if not finalists:
        raise ValueError("zero/triple recipe found no kernel-four-clean filling")

    # Do not let a short-budget witness miss determine the emitted code.
    for item in finalists:
        item["da"] = min(item["da"],
                         _z23_reading(item["ac"], rng, 24))
        item["db"] = min(item["db"],
                         _z23_reading(item["bc"], rng, 24))
        d = min(item["da"], item["db"], 6)
        item["key"] = (item["k"]*d*d, d, -item["cycles"])
    finalists.sort(key=lambda x: x["key"], reverse=True)

    winner = None
    for item in finalists[:6]:
        qa = _z23_short(item["ac"], through_five=True)
        qb = _z23_short(item["bc"], through_five=True)
        low, cycles = _q85_short_screen(
            item["hx"], item["hz"], item["n"])
        if low != 5:
            continue
        # Sentinels certify only the finite screened weight range.
        clean5 = qa == qb == 6
        d = min(item["da"], item["db"], qa, qb, 6)
        key = (clean5, item["k"]*d*d, d, -cycles)
        if winner is None or key > winner[0]:
            winner = (key, item)

    if winner is None:
        raise ValueError("no finalist passed full stabilizer-aware CSS screen")
    key, item = winner
    A, B = item["A"], item["B"]
    assert css_ok(item["hx"], item["hz"])
    assert _grid_weights(sa, sb) <= 12
    assert all(len(cell) == len({G._key(g) for g in cell})
               for grid in (A, B) for row in grid for cell in row)
    return {"idx": idx, "cells": {"A": A, "B": B},
            "screen": key, "k": item["k"]}


from functools import lru_cache


_C17_A_TOP = (
    (2, 2, 1, 1, 1),
    (3, 1, 1, 1, 1),
    (3, 2, 1, 1, 0),
)
_C17_B_TOP = (
    (2, 2, 2, 1, 1, 1),
    (3, 2, 1, 1, 1, 1),
    (3, 2, 2, 1, 1, 0),
)


def _c17_clean6(cols):
    """Exact six exclusion after component words through four are absent.

    Every column is cubic, so odd kernel words are impossible.
    Colliding distinct triples therefore detect exactly weight six.
    """
    seen = set()
    for i, x in enumerate(cols):
        for j in range(i):
            pair = x ^ cols[j]
            for h in range(j):
                syndrome = pair ^ cols[h]
                if syndrome in seen:
                    return False
                seen.add(syndrome)
    return True


def _c17_probe(cols, rng, trials, seeds=(), stabilizers=None):
    """Retain actual kernel words; logical readings exclude stabilizers."""
    span = None if stabilizers is None else _q85_basis(stabilizers)
    memory = sorted({w for w in seeds if w}, key=int.bit_count)[:40]
    best = len(cols) + 1
    order = list(range(len(cols)))

    def syndrome(word):
        value = 0
        while word:
            bit = word & -word
            value ^= cols[bit.bit_length() - 1]
            word ^= bit
        return value

    def observe(word):
        nonlocal best
        if word and (span is None or _q85_outside(word, span)):
            best = min(best, word.bit_count())

    memory = [w for w in memory if syndrome(w) == 0]
    for word in memory:
        observe(word)

    for _ in range(trials):
        rng.shuffle(order)
        pivots, words = {}, []
        for c in order:
            value, word = cols[c], 1 << c
            while value:
                p = value.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (value, word)
                    break
                value ^= old[0]
                word ^= old[1]
            if not value:
                words.append(word)
                observe(word)

        donors = sorted(set(words + memory), key=int.bit_count)[:32]
        for sweep in range(2):
            reduced = []
            for word in words:
                weight = word.bit_count()
                for other in donors:
                    trial = word ^ other
                    tw = trial.bit_count()
                    if 0 < tw < weight:
                        word, weight = trial, tw
                reduced.append(word)
                observe(word)
            words = reduced
            donors = sorted(set(words + memory), key=int.bit_count)[:32]

        discoveries = set(words + memory)
        short = sorted(discoveries, key=int.bit_count)[:32]
        for i, word in enumerate(short):
            for other in short[:i]:
                trial = word ^ other
                if trial:
                    observe(trial)
                    discoveries.add(trial)

        if stabilizers is not None:
            logical = sorted(
                (w for w in discoveries if _q85_outside(w, span)),
                key=int.bit_count)[:6]
            for word in logical:
                for sweep in range(2):
                    changed = False
                    for row in stabilizers:
                        trial = word ^ row
                        if trial.bit_count() < word.bit_count():
                            word = trial
                            changed = True
                    if not changed:
                        break
                observe(word)
                discoveries.add(word)

        memory = sorted(discoveries, key=int.bit_count)[:40]
        if best <= 4:
            break

    assert all(syndrome(w) == 0 for w in memory)
    return best, tuple(memory)


def _c17_fingerprint(hx, hz, n):
    """Invariant of the combined uncoloured incidence graph.

    Unequal signatures certify nonisomorphism. Equal signatures are
    conservatively rejected, not asserted to prove isomorphism.
    """
    from collections import Counter

    rows = list(hx) + list(hz)
    weights = [row.bit_count() for row in rows]
    adjacency = [0] * len(rows)
    neighborhoods = [Counter() for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
                neighborhoods[i][weights[j], overlap] += 1
                neighborhoods[j][weights[i], overlap] += 1

    profiles = []
    for i, neighbors in enumerate(adjacency):
        remaining, triangles = neighbors, 0
        while remaining:
            bit = remaining & -remaining
            j = bit.bit_length() - 1
            triangles += (neighbors & adjacency[j]).bit_count()
            remaining ^= bit
        profiles.append((
            weights[i], tuple(sorted(neighborhoods[i].items())),
            triangles // 2))

    qubits = [[] for _ in range(n)]
    for i, source in enumerate(rows):
        while source:
            bit = source & -source
            qubits[bit.bit_length() - 1].append(weights[i])
            source ^= bit
    return (
        n, tuple(sorted(Counter(profiles).items())),
        tuple(sorted(Counter(tuple(sorted(v)) for v in qubits).items())))


@lru_cache(maxsize=1)
def _c17_bank(G):
    """Fresh two-row optimization; never invokes the archived parent."""
    assert G.q == 17
    assert A_SHAPE == (2, 5) and B_SHAPE == (2, 6)
    q, mask = G.q, (1 << G.q) - 1
    za = [[[] for _ in range(5)] for _ in range(2)]
    zb = [[[] for _ in range(6)] for _ in range(2)]

    # Compile actual permutation blocks from verified unit-cell lifts.
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            A = [[list(c) for c in row] for row in za]
            B = [[list(c) for c in row] for row in zb]
            (A if side == "A" else B)[0][0] = [g]
            hx, hz, _ = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            block = tuple(_z23_columns(
                [row & mask for row in source[:q]], q))
            assert all(c.bit_count() == 1 for c in block)
            blocks.append(block)
        atlas[side] = tuple(blocks)

    def compile_grid(side, grid):
        cols = [0] * (len(grid[0]) * q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = atlas[side][G.idx(g)]
                    for h, value in enumerate(block):
                        cols[j*q+h] ^= value << (i*q)
        assert all(c.bit_count() == 3 for c in cols)
        return tuple(cols)

    def embeddings(words, side):
        if side == "B":
            return [word << (j * 6 * q)
                    for word in words[:12] for j in range(5)]
        return [
            sum(((word >> (j*q)) & mask) << ((j*6+b)*q)
                for j in range(5))
            for word in words[:12] for b in range(6)]

    bank, emitted = [], set()
    for slot in range(N_SLOTS):
        card = slot % 20
        geometry, mode = card % 9, card // 9
        ai, bi = divmod(geometry, 3)
        sa = [list(_C17_A_TOP[ai]),
              [3-t for t in _C17_A_TOP[ai]]]
        sb = [list(_C17_B_TOP[bi]),
              [3-t for t in _C17_B_TOP[bi]]]
        assert _grid_weights(sa, sb) <= 12
        assert _no_dead_lines(sa, sb)
        local = random.Random(
            BASE_SEED * 100003 + 17350261 + 104729 * slot)

        def draw(side):
            counts = sa if side == "A" else sb
            grid = [[[] for _ in row] for row in counts]
            pairs = [(i, j) for i, row in enumerate(counts)
                     for j, count in enumerate(row) if count == 2]

            # Intrinsic inverse-gap multiplicities, not common scalings.
            if mode == 0:
                gaps = [local.randrange(1, 9) for _ in pairs]
            elif mode == 1:
                gaps = local.sample(range(1, 9), len(pairs))
            else:
                first, second = local.sample(range(1, 9), 2)
                gaps = [first if t % 2 == 0 else second
                        for t in range(len(pairs))]
            gap_map = dict(zip(pairs, gaps))

            for i, row in enumerate(counts):
                for j, count in enumerate(row):
                    if count == 0:
                        continue
                    if count == 1:
                        grid[i][j] = [G.pick(local)]
                    elif count == 2:
                        base = local.randrange(q)
                        delta = gap_map[i, j] * local.choice((-1, 1))
                        grid[i][j] = [
                            (base,), ((base + delta) % q,)]
                    else:
                        # Avoid imposing the parent's AP rule everywhere.
                        if mode == 2:
                            base = local.randrange(q)
                            step = local.randrange(1, q)
                            grid[i][j] = [
                                ((base+t*step) % q,) for t in range(3)]
                        else:
                            grid[i][j] = local.sample(G.elements, 3)
            return grid

        def pool(side, quota):
            retained, seen = [], set()
            for attempt in range(quota):
                grid = draw(side)
                labels = tuple(tuple(sorted(cell))
                               for row in grid for cell in row)
                if labels in seen:
                    continue
                seen.add(labels)
                cols = compile_grid(side, grid)
                if _z23_short(cols) != 5:
                    continue
                clean6 = _c17_clean6(cols)
                cycles = _z23_cycles(cols)
                reading, words = _c17_probe(cols, local, 3)
                item = dict(
                    grid=grid, cols=cols, clean6=clean6,
                    cycles=cycles, d=reading, words=words,
                    labels=labels)
                item["key"] = (
                    clean6, min(reading, 12), -cycles, labels)
                retained.append(item)
                retained.sort(key=lambda x: x["key"], reverse=True)
                del retained[8:]

            for item in retained:
                d, words = _c17_probe(
                    item["cols"], local, 28, item["words"])
                item["d"] = min(item["d"], d)
                item["words"] = words
                item["key"] = (
                    item["clean6"], min(item["d"], 12),
                    -item["cycles"], item["labels"])
            retained.sort(key=lambda x: x["key"], reverse=True)
            return retained[:4]

        chosen = None
        for retry in range(2):
            ap = pool("A", 160 + 96*retry)
            bp = pool("B", 160 + 96*retry)
            proposals = []
            for a in ap:
                for b in bp:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if not (325*n <= 1000*k < 375*n):
                        continue
                    ac, bc = _z23_components(hx, hz, q)
                    assert tuple(ac) == a["cols"]
                    assert tuple(bc) == b["cols"]
                    d = min(a["d"], b["d"], 12)
                    key = (
                        k*d*d, a["clean6"] and b["clean6"],
                        -a["cycles"]-b["cycles"])
                    proposals.append((key, a, b, hx, hz, n, k))
            proposals.sort(key=lambda x: x[0], reverse=True)

            finalists = []
            for _, a, b, hx, hz, n, k in proposals:
                fp = _c17_fingerprint(hx, hz, n)
                if fp in emitted:
                    continue
                low, cycles = _q85_short_screen(hx, hz, n)
                if low != 5:
                    continue
                xc, zc = _z23_columns(hz, n), _z23_columns(hx, n)
                dx, xwords = _c17_probe(
                    xc, local, 6, embeddings(b["words"], "B"), hx)
                dz, zwords = _c17_probe(
                    zc, local, 6, embeddings(a["words"], "A"), hz)
                finalists.append(dict(
                    A=a["grid"], B=b["grid"], hx=hx, hz=hz,
                    n=n, k=k, fp=fp, xc=xc, zc=zc,
                    dx=dx, dz=dz, xwords=xwords, zwords=zwords,
                    da=a["d"], db=b["d"], cycles=cycles,
                    clean6=a["clean6"] and b["clean6"]))
                if len(finalists) == 4:
                    break

            winner, best_key = None, None
            for item in finalists:
                dx, _ = _c17_probe(
                    item["xc"], local, 18,
                    item["xwords"], item["hx"])
                dz, _ = _c17_probe(
                    item["zc"], local, 18,
                    item["zwords"], item["hz"])
                d = min(
                    item["da"], item["db"],
                    item["dx"], item["dz"], dx, dz, 12)
                # A no-witness sentinel is never treated as a large reading.
                if dx > item["n"] or dz > item["n"]:
                    continue
                key = (
                    item["k"]*d*d, d,
                    item["clean6"], -item["cycles"])
                if best_key is None or key > best_key:
                    winner, best_key = item, key

            if winner is None:
                continue
            A, B = winner["A"], winner["B"]
            assert css_ok(winner["hx"], winner["hz"])
            assert winner["n"] == 578
            assert max(row.bit_count()
                       for row in winner["hx"] + winner["hz"]) <= 12
            assert all(len(cell) == len(set(cell)) <= 3
                       for grid in (A, B) for row in grid for cell in row)
            emitted.add(winner["fp"])
            chosen = dict(
                cells={
                    side: tuple(tuple(tuple(c) for c in row)
                                for row in grid)
                    for side, grid in (("A", A), ("B", B))},
                k=winner["k"], card=card, screen=best_key)
            break

        # No archived filling or transported design replaces a failed card.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    selected = _c17_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh two-row cubic design passed this card")
    return selected


def CELL(rng, G, side, i, j, ctx):
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's uniform rolled D14 fillings did not suppress sector-2 weight-4 logicals: all five measured codes had d30s=4.
# REFUTED: Q8 x Z5 order-20 quaternion-axis binomials plus exact-four candidate selection did not break the weight-four plateau: all five measured codes had d30s=4, with minima entirely in sector 1.
# REFUTED: A trinomials and distinct relative labels on seven all-monomial B columns exclude weight-four logicals; parent slots 0..3 instead had d2s=4.
# REFUTED: Zero/triple B rewiring and deeper component screening improve the Z23 (1,3)x(2,7) five-code score; all five current measurements remained k=230, d30s=6, fitness 15.652.
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

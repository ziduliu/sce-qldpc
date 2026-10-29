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
CONCEPT = ("For cyclic Z17 (metacyclic Z17:Z1, s=1), order 17, shapes (2,5)x(2,6), balanced degree-three components selected by exact kernel-orbit multiplicity and potential ancilla-spread exposure should raise worse-basis circuit fitness above the rolled baseline 4.0702.")
GROUP = ("Frozen L1 group: retain the exact Z17 construction and order. This bed improved from 4.0702 to 4.7738 with degree-three columns. Order 17 is the largest admitted by size factor 34, giving n=578. These measurements support another filling experiment, not a group-only performance claim.")
SHAPES = ("Frozen (2,5)x(2,6), targeting cell 0.350: kb=204 and guaranteed rate 12/34=0.352941. A has 85 variables/34 checks and B has 102 variables/34 checks; their unequal lengths warrant screening both independently. Parent failures affected all 204 logicals, with only diffuse 1.1-1.2x Acol/Bcol and Acol/Brow enrichment. Slot31 implicated Acol0/Bcol0,3,5; slot50 implicated several A-columns at Bcol3. No unique hotspot is established.")
FILLING = ("Twenty contracts cross four balanced count architectures with five joint difference-card prescriptions. All component columns have three terms, A rows total 7/8 and B rows 9/9: HX<=11, HZ=12, and six-round extraction uses 25092 CNOTs. Canonical relative-geometry books remove obvious orbit copies; voltage walks optimize only genuine inter-row offsets. Verified singleton actions support exact weight-six/eight kernel translation-orbit censuses. Full-CSS pairing prefers no logical <=4 and fewer low-weight words coverable by few stabilizer-support fault locations. This potential-hook cover ignores ASC ordering and is only a circuit-pL hypothesis. Target collective Z failures such as slot31's 146/750 and slot50's 125/1000, despite their equal d2s=5; do not reward rank deficits or an optimistic distance estimate.")
FAMILY_NAME = "r035_m17r1s1_25x26_orbit8_hookcover_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 17

def custom_ops(q):
    # metacyclic Z_17 x| Z_1: y x y^-1 = x^1 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 17, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Twenty count/difference contracts; screen only verified actions."""
    import math
    from itertools import combinations
    from functools import lru_cache

    q = G.q
    assert q == 17
    design, batch = idx % 20, idx // 20
    architecture, card = divmod(design, 5)
    mask = (1 << q) - 1

    # Upper-row counts; lower rows complete every column to degree three.
    # These are support architectures, not permutations of one filling.
    profiles = (
        ((2, 2, 1, 1, 1), (2, 2, 2, 1, 1, 1)),
        ((3, 0, 2, 1, 1), (2, 2, 2, 1, 1, 1)),
        ((3, 0, 2, 1, 1), (0, 2, 2, 2, 2, 1)),
        ((3, 0, 0, 2, 2), (2, 2, 2, 1, 1, 1)),
    )
    pa, pb = profiles[architecture]
    counts = {
        "A": [list(pa), [3 - t for t in pa]],
        "B": [list(pb), [3 - t for t in pb]],
    }
    assert _no_dead_lines(counts["A"], counts["B"])
    assert _grid_weights(counts["A"], counts["B"]) <= W_MAX
    assert sorted(map(sum, counts["A"])) == [7, 8]
    assert list(map(sum, counts["B"])) == [9, 9]

    # (number of A directions, B directions, shared directions).
    # Sparse trinomial rows require larger direction palettes.
    direction_cards = (
        ((3, 3, 0), (3, 4, 0), (4, 3, 0), (4, 4, 0), (4, 4, 1)),
        ((5, 3, 0), (5, 4, 1), (6, 3, 1), (6, 4, 2), (6, 5, 3)),
        ((5, 4, 1), (5, 5, 2), (6, 4, 2), (6, 5, 3), (7, 4, 3)),
        ((6, 3, 1), (6, 4, 2), (6, 5, 3), (7, 3, 2), (7, 4, 3)),
    )
    na, nb, overlap = direction_cards[architecture][card]

    def direction(x):
        x %= q
        return min(x, q - x)

    def differences(cell):
        return frozenset(direction(x - y)
                         for x, y in combinations(cell, 2))

    triples = []
    for a, b in combinations(range(1, q), 2):
        cell = (0, a, b)
        ds = differences(cell)
        if len(ds) == 3:
            triples.append((cell, ds))

    def normalized(cell, multiplier):
        values = [(multiplier * x) % q for x in cell]
        if not values:
            return ()
        return min(tuple(sorted((x - origin) % q for x in values))
                   for origin in values)

    def geometry_key(grids):
        """Obvious automorphisms/gauges only; Tanner testing is authoritative."""
        images = []
        for u in range(1, q):
            sides = []
            for side in ("A", "B"):
                grid = grids[side]
                columns = [
                    (normalized(grid[0][j], u),
                     normalized(grid[1][j], u))
                    for j in range(len(grid[0]))
                ]
                sides.append(min(tuple(sorted(columns)),
                                 tuple(sorted((b, a) for a, b in columns))))
            images.append(tuple(sides))
        return min(images)

    book_rng = random.Random(170052601 + 1031 * design)
    seen_geometries = set()
    accepted = 0
    geometry = None
    for proposal in range(30000):
        ap = set(book_rng.sample(range(1, 9), na))
        bp = set(book_rng.sample(sorted(ap), overlap))
        bp.update(book_rng.sample(sorted(set(range(1, 9)) - ap),
                                  nb - overlap))
        grids = {}
        legal = True
        for side, palette in (("A", ap), ("B", bp)):
            grid = []
            used_side = set()
            for row in counts[side]:
                out = [[] for _ in row]
                used = set()
                # Fill trinomials before binomials so row differences
                # are distinct by construction, excluding intra-row C4.
                sites = sorted(range(len(row)), key=lambda j: -row[j])
                for j in sites:
                    t = row[j]
                    available = palette - used
                    if t == 3:
                        choices = [(cell, ds) for cell, ds in triples
                                   if ds <= available]
                        if not choices:
                            legal = False
                            break
                        cell, ds = book_rng.choice(choices)
                        out[j] = list(cell)
                        used.update(ds)
                    elif t == 2:
                        if not available:
                            legal = False
                            break
                        d = book_rng.choice(sorted(available))
                        out[j] = [0, d]
                        used.add(d)
                    elif t == 1:
                        out[j] = [0]
                if not legal:
                    break
                used_side.update(used)
                grid.append(out)
            if not legal or used_side != palette:
                legal = False
                break
            grids[side] = grid
        if not legal:
            continue
        key = geometry_key(grids)
        if key in seen_geometries:
            continue
        seen_geometries.add(key)
        if accepted == batch:
            geometry = grids
            break
        accepted += 1
    if geometry is None:
        raise ValueError("no unused joint relative-difference geometry")

    def transpose(rows, width):
        columns = [0] * width
        limit = (1 << width) - 1
        for r, row in enumerate(rows):
            bits = row & limit
            while bits:
                bit = bits & -bits
                columns[bit.bit_length() - 1] ^= 1 << r
                bits ^= bit
        return columns

    # Cache only blocks actually returned by the verified helper.
    actions = {"A": {}, "B": {}}
    zero = [[[]]]
    for g in G.elements:
        ar = lifted_matrices(G, [[[g]]], zero)[0]
        br = lifted_matrices(G, zero, [[[g]]])[1]
        actions["A"][G._key(g)] = tuple(transpose(ar, q))
        actions["B"][G._key(g)] = tuple(transpose(br, q))

    def component(side, grid):
        width = len(grid[0]) * q
        columns = [0] * width
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = actions[side][G._key(g)]
                    for t, value in enumerate(block):
                        columns[j * q + t] ^= value << (i * q)
        assert all(c.bit_count() == 3 for c in columns)
        return columns

    def rows_of(columns):
        return transpose(columns, 2 * q)

    def rot_block(value, shift):
        shift %= q
        return ((value << shift) | (value >> (q - shift))) & mask

    def rotate(word, shift, blocks):
        return sum(rot_block((word >> (j * q)) & mask, shift) << (j * q)
                   for j in range(blocks))

    def canonical_word(word, blocks):
        """Unique anchored representative of a simultaneous translation orbit."""
        first = ((word & -word).bit_length() - 1) // q
        positions = (word >> (first * q)) & mask
        best = None
        while positions:
            bit = positions & -positions
            shift = -(bit.bit_length() - 1)
            image = rotate(word, shift, blocks)
            best = image if best is None else min(best, image)
            positions ^= bit
        return best

    # A nonconstant 17-bit pattern has a unique minimal rotation.
    # Populate its entire necklace at once, avoiding 17 rotations per
    # quadruple. This cache is local to this pure NEW_CTX invocation.
    necklaces = {0: (0, 0)}

    def syndrome_origin(syndrome):
        low, high = syndrome & mask, (syndrome >> q) & mask
        leading = high or low
        assert leading and leading != mask
        if leading not in necklaces:
            orbit = [rot_block(leading, s) for s in range(q)]
            origin = min(range(q), key=lambda s: orbit[s])
            least = orbit[origin]
            for s, value in enumerate(orbit):
                necklaces[value] = (least, (origin - s) % q)
        shift = necklaces[leading][1]
        normal = rot_block(low, shift) | (rot_block(high, shift) << q)
        return normal, shift

    def orbit_census(columns, half):
        """Exact weight 2*half kernel orbits, after weights <=4 vanish.

        Half-supports are enumerated once per translation orbit. Equal
        syndrome orbits are aligned using their unique syndrome origin.
        Only disjoint supports count. A six-word has ten partitions;
        an eight-word has 35. This counts component words, not logicals.
        """
        width = len(columns)
        blocks = width // q
        # Verify the symmetry on actual helper-derived columns.
        assert all(columns[j * q + t] == rotate(columns[j * q], t, 2)
                   for j in range(blocks) for t in range(q))
        bits = [1 << i for i in range(width)]
        buckets = {}
        words = set()
        collisions = visited = 0
        for j in range(blocks):
            anchor = j * q
            for tail in combinations(range(anchor + 1, width), half - 1):
                word = bits[anchor]
                syndrome = columns[anchor]
                for c in tail:
                    word |= bits[c]
                    syndrome ^= columns[c]
                if canonical_word(word, blocks) != word:
                    continue
                visited += 1
                normal, shift = syndrome_origin(syndrome)
                aligned = rotate(word, shift, blocks)
                old = buckets.get(normal)
                if old is None:
                    buckets[normal] = aligned
                    continue
                entries = old if isinstance(old, list) else (old,)
                for other in entries:
                    if aligned & other:
                        continue
                    collisions += 1
                    words.add(canonical_word(aligned ^ other, blocks))
                if isinstance(old, list):
                    old.append(aligned)
                else:
                    buckets[normal] = [old, aligned]
        divisor = 10 if half == 3 else 35
        assert visited * q == math.comb(width, half)
        assert collisions == divisor * len(words)
        return len(words), words

    def short4(columns):
        singles = {}
        n2 = 0
        for value in columns:
            n2 += singles.get(value, 0)
            singles[value] = singles.get(value, 0) + 1
        pairs = {}
        collisions = 0
        for i, x in enumerate(columns):
            for y in columns[:i]:
                syndrome = x ^ y
                collisions += pairs.get(syndrome, 0)
                pairs[syndrome] = pairs.get(syndrome, 0) + 1
        # Exact when columns are distinct; otherwise a lower-priority proxy.
        return n2, collisions // 3

    def cycles(columns):
        """Exact component four/six-cycle counts for degree-three columns."""
        adjacency = [0] * (2 * q)
        overlaps = {}
        four = 0
        neighbors = []
        for value in columns:
            ns = []
            while value:
                bit = value & -value
                ns.append(bit.bit_length() - 1)
                value ^= bit
            neighbors.append(ns)
            for u, v in combinations(ns, 2):
                key = (u, v)
                old = overlaps.get(key, 0)
                four += old
                overlaps[key] = old + 1
                adjacency[u] |= 1 << v
                adjacency[v] |= 1 << u
        six = 0
        for (u, v), uv in overlaps.items():
            common = adjacency[u] & adjacency[v] & ~((1 << (v + 1)) - 1)
            while common:
                bit = common & -common
                w = bit.bit_length() - 1
                six += uv * overlaps[u, w] * overlaps[v, w]
                common ^= bit
        for u, v, w in neighbors:
            six -= overlaps[u, v] + overlaps[u, w] + overlaps[v, w] - 2
        assert six >= 0
        return four, six

    def cell_at(side, i, j, origin):
        return [G.mul((origin, 0), (value, 0))
                for value in geometry[side][i][j]]

    def draw(side):
        grid = [[cell_at(side, i, j, 0) for j in range(len(row))]
                for i, row in enumerate(counts[side])]
        for j in range(len(grid[0])):
            if grid[0][j] and grid[1][j]:
                grid[1][j] = cell_at(side, 1, j, rng.randrange(q))
        return grid

    def voltage_signature(grid):
        # Column gauges are fixed by draw; remove the remaining common
        # inter-row shift when retaining search finalists.
        values = [grid[1][j][0][0] for j in range(len(grid[0]))
                  if grid[0][j] and grid[1][j]]
        return tuple((v - values[0]) % q for v in values)

    def portfolio(side):
        sites = [j for j in range(len(counts[side][0]))
                 if counts[side][0][j] and counts[side][1][j]]
        pool, seen = [], set()

        def offer(grid, columns, stats):
            sig = voltage_signature(grid)
            if sig in seen:
                return
            seen.add(sig)
            if gf2_rank(rows_of(columns)) != 2 * q:
                return
            key = short4(columns) + stats
            saved = [[list(cell) for cell in row] for row in grid]
            pool.append((key, sig, saved, list(columns)))
            pool.sort(key=lambda item: item[:2])
            del pool[4:]

        for restart in range(3):
            grid = draw(side)
            columns = component(side, grid)
            stats = cycles(columns)
            local = stats[0]
            offer(grid, columns, stats)
            steps = 260 + 20 * (batch % 3)
            for step in range(steps):
                j = rng.choice(sites)
                trial = [list(row) for row in grid]
                trial[1][j] = cell_at(side, 1, j, rng.randrange(q))
                proposed_columns = component(side, trial)
                proposed = cycles(proposed_columns)
                delta = ((60 * proposed[0] + proposed[1])
                         - (60 * stats[0] + stats[1])) / q
                temperature = 9.0 * (0.982 ** step) + 0.08
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    grid, columns, stats = trial, proposed_columns, proposed
                    improved = stats[0] < local
                    local = min(local, stats[0])
                    if improved or (step % 12 == 0 and stats[0] <= local):
                        offer(grid, columns, stats)
            offer(grid, columns, stats)

        if not pool:
            raise ValueError("no full-rank component in relative-geometry card")
        refined = []
        for key, sig, grid, columns in pool:
            if not any(key[:2]):
                n6, words6 = orbit_census(columns, 3)
            else:
                n6, words6 = 10**9, set()
            refined.append((key[:2] + (n6,) + key[2:],
                            sig, grid, columns, words6))
        refined.sort(key=lambda item: item[:2])
        result = []
        for key, sig, grid, columns, words6 in refined[:2]:
            if not any(key[:2]):
                n8, words8 = orbit_census(columns, 4)
            else:
                n8, words8 = 10**9, set()
            witnesses = tuple(sorted(words6 | words8,
                                     key=lambda word: (word.bit_count(), word)))[:64]
            result.append({"grid": grid, "short": key[:3], "n8": n8,
                           "words": witnesses, "cycles": key[3:]})
        return result

    def span_basis(rows):
        pivots = {}
        for value in rows:
            while value:
                p = value.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = value
                    break
                value ^= old
        return pivots

    def in_span(word, pivots):
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                return False
            word ^= old
        return True

    def small_logical(checks, stabilizers, n):
        """Exact existence test through weight four, modulo stabilizers."""
        columns = transpose(checks, n)
        pivots = span_basis(stabilizers)
        seen = {0: 0}
        for i, syndrome in enumerate(columns):
            word = 1 << i
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif not in_span(old ^ word, pivots):
                return True
        for i, x in enumerate(columns):
            for j in range(i):
                syndrome = x ^ columns[j]
                word = (1 << i) | (1 << j)
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = word
                elif not in_span(old ^ word, pivots):
                    return True
        return False

    def hook_profile(stabilizers, side, witnesses, n):
        """Potential fault-location covers, NOT ASC hook probabilities.

        One check-support location may cover any subset of its touched
        witness qubits. Unknown CNOT ordering can make such covers
        unrealizable. Individual data-fault locations are always allowed.
        Stabilizer images are excluded using the actual full row space.
        """
        shared = transpose(stabilizers, n)
        pivots = span_basis(stabilizers)
        repeats = B_SHAPE[1] if side == "A" else A_SHAPE[1]
        minimum, exposure = 8, 0
        for component_word in witnesses:
            positions = []
            bits = component_word
            while bits:
                bit = bits & -bits
                positions.append(bit.bit_length() - 1)
                bits ^= bit
            for repetition in range(repeats):
                indices = []
                for position in positions:
                    block, t = divmod(position, q)
                    j, beta = ((block, repetition) if side == "A"
                               else (repetition, block))
                    indices.append((j * B_SHAPE[1] + beta) * q + t)
                image = sum(1 << c for c in indices)
                if in_span(image, pivots):
                    continue
                check_patterns = {}
                for local, c in enumerate(indices):
                    bits = shared[c]
                    while bits:
                        bit = bits & -bits
                        check = bit.bit_length() - 1
                        check_patterns[check] = (
                            check_patterns.get(check, 0) | (1 << local))
                        bits ^= bit
                patterns = set(check_patterns.values())
                choices = [
                    tuple(p for p in patterns
                          if p & (1 << i) and p.bit_count() >= 2)
                    for i in range(len(indices))
                ]

                @lru_cache(None)
                def cover(remaining):
                    if not remaining:
                        return 0
                    bit = remaining & -remaining
                    i = bit.bit_length() - 1
                    best = 1 + cover(remaining ^ bit)
                    for pattern in choices[i]:
                        best = min(best, 1 + cover(remaining & ~pattern))
                    return best

                faults = cover((1 << len(indices)) - 1)
                minimum = min(minimum, faults)
                exposure += 2 ** max(0, 8 - faults) - 1
        return minimum, exposure

    apool, bpool = portfolio("A"), portfolio("B")
    winner = None
    for a in apool:
        for b in bpool:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 578 or k != 204:
                continue
            bad = int(small_logical(hx, hz, n)
                      or small_logical(hz, hx, n))
            az = hook_profile(hz, "A", a["words"], n)
            bx = hook_profile(hx, "B", b["words"], n)
            short = tuple(x + y for x, y in zip(a["short"], b["short"]))
            full_c4 = sum(
                t * (t - 1) // 2
                for rows in (hx, hz)
                for i, row in enumerate(rows)
                for other in rows[:i]
                for t in [(row & other).bit_count()])
            key = ((bad,) + short
                   + (max(0, 5 - min(az[0], bx[0])),
                      max(az[1], bx[1]), az[1] + bx[1],
                      6 * a["n8"] + 5 * b["n8"], full_c4,
                      sum(a["cycles"]) + sum(b["cycles"])))
            if winner is None or key < winner[0]:
                winner = key, A, B, hx, hz, k

    if winner is None:
        raise ValueError("no verified full-rank pairing in cell 0.350")
    key, A, B, hx, hz, k = winner
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert 6 * sum(row.bit_count() for row in hx + hz) == 25092

    # Independently verify the final cached-action composition.
    for side, grid in (("A", A), ("B", B)):
        rows = (lifted_matrices(G, grid, zero)[0] if side == "A"
                else lifted_matrices(G, zero, grid)[1])
        assert transpose(rows, len(grid[0]) * q) == component(side, grid)
    return {"idx": idx, "cells": {"A": A, "B": B}, "k": k,
            "contract": (architecture, card, batch), "screen": key}


def CELL(rng, G, side, i, j, ctx):
    """Return a copy of the complete, already screened cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Equal (n,k,w,d2s) determines circuit reliability: Z17 seed slots1/31 both had (578,204,12,5), but scored 4.347/3.691; slot50 was strongly Z-limited despite weight-seven Z-type witnesses.
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

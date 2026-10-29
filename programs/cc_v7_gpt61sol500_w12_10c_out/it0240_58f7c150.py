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
CONCEPT = ("C37, order 37, A=(2,3), B=(2,4): independently optimizing commuting-cofactor witnesses and dressing verified logical embeddings should exceed the rate-0.150 frontier fitness 41.1, rather than reproduce the parent's measured distance-eight plateau.")
GROUP = ("Jump from dihedral order 58 to the genuinely different cyclic prime-order C37. Every nonidentity element has order 37. C46 reached fitness 60.3 on (2,3)x(2,3), while Dic9 reached 41.1 on the proposed shapes; these motivate an untried C37 bed, not a transferred distance guarantee. Order 37 is the largest lift allowed by n=16q<600.")
SHAPES = ("Target cell 0.150 with (2,3)x(2,4): n=592, kb=74 and guaranteed rate 1/8. Constituents have 111/74 and 148/74 variables/checks. Even A-cell counts create augmentation deficits; accept only exact verified rates in [0.130,0.175). Parent minima were entirely sector-one Acol/Bcol witnesses, prominently Acol1 and Bcol1; screen both new constituents and dressed full-matrix representatives.")
FILLING = ("Twenty tasks cross five B count geometries with four prescribed A gap geometries. All A cells are doubletons; B rows sum to eight and columns to at most five, so HX<=11 and HZ<=12. Compile component actions exclusively from verified single-term lifts. Retain syndrome-verified cofactor and information-set kernel words, independently deepen finalists, test exact stabilizer membership and dress logical embeddings. Reject equal uncoloured Tanner signatures across the deterministic bank. Found readings are upper estimates, never distance certificates.")
FAMILY_NAME = "r015_C37_23x24_cofactor_gapcards_dressed_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 37

def custom_ops(q):
    # Cyclic prime-order family, not a metacyclic inversion presentation.
    if not valid_q(q):
        raise ValueError("this C37 bed requires order 37")
    els = [(a,) for a in range(37)]

    def red(g):
        return (g[0] % 37,)

    def mul(g, h):
        return ((g[0] + h[0]) % 37,)

    def inv(g):
        return ((-g[0]) % 37,)

    def idx(g):
        return g[0] % 37

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _legacy_dihedral_ctx(rng, G, idx):
    """Inactive historical optimizer; NEW_CTX below replaces this policy."""
    from itertools import combinations

    q = G.q
    m = 29
    n1 = 9 * q

    def gap(x):
        x %= m
        return min(x, m - x)

    def triangular(t):
        a, b, c = sorted(t)
        return a + b == c or a + b + c == m

    def canonical(a, b):
        images = []
        for scale in range(1, 15):
            aa = tuple(sorted(gap(scale * x) for x in a))
            bb = tuple(sorted(gap(scale * x) for x in b))
            images.append(tuple(sorted((aa, bb))))
        return min(images)

    # Distinct gap profiles modulo common group automorphisms,
    # column permutations, and exchange of the two sides.
    designs = []
    for a in ((1, 4, 10), (1, 5, 12), (1, 6, 11)):
        remaining = [x for x in range(1, 15) if x not in a]
        for b in combinations(remaining, 3):
            if not triangular(b):
                designs.append((a, b))
    designs.sort(key=lambda z: (
        sum((i + 3) * x * x for i, x in enumerate(z[0] + z[1])) % 997,
        z))
    seen = set()
    profiles = []
    for a, b in designs:
        key = canonical(a, b)
        if key not in seen:
            seen.add(key)
            profiles.append((a, b))
    ga, gb = profiles[idx % len(profiles)]

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            bit = 1 << r
            while row:
                low = row & -row
                out[low.bit_length() - 1] |= bit
                row ^= low
        return out

    def pair_quality(cols):
        # All these columns have odd weight. Distinct nonzero singles
        # and distinct nonzero pair syndromes exclude weights 1..4.
        if 0 in cols or len(set(cols)) != len(cols):
            return None
        pairs = {0}
        low2 = low4 = 0
        for i, x in enumerate(cols):
            for y in cols[:i]:
                s = x ^ y
                if s in pairs:
                    return None
                pairs.add(s)
                w = s.bit_count()
                low2 += (w == 2)
                low4 += (w == 4)
        return (10 + idx % 5) * low2 + low4

    def six_collisions(cols, cap=2500):
        # Following the pair test, two equal triple syndromes have
        # disjoint supports and therefore yield a weight-six kernel word.
        syndromes = set()
        count = 0
        for i in range(len(cols)):
            x = cols[i]
            for j in range(i):
                xy = x ^ cols[j]
                for k in range(j):
                    s = xy ^ cols[k]
                    if s in syndromes:
                        count += 1
                        if count >= cap:
                            return cap
                    else:
                        syndromes.add(s)
        return count

    def basis(rows):
        pivots = {}
        for row in rows:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        return pivots

    def remainder(v, pivots):
        while v:
            p = v.bit_length() - 1
            if p not in pivots:
                return v
            v ^= pivots[p]
        return 0

    def no_small_logical(cols, stabilizers):
        # Exact syndrome-collision test through weight four.
        # Keep one representative per syndrome; stabilizer-equivalent
        # collisions are harmless.
        representatives = {0: 0}
        for i, s in enumerate(cols):
            v = 1 << i
            if s in representatives:
                if remainder(v ^ representatives[s], stabilizers):
                    return False
            else:
                representatives[s] = v
        for i, x in enumerate(cols):
            bit = 1 << i
            for j in range(i):
                s = x ^ cols[j]
                v = bit | (1 << j)
                if s in representatives:
                    if remainder(v ^ representatives[s], stabilizers):
                        return False
                else:
                    representatives[s] = v
        return True

    def mixed_sixes(cols, stabilizers, x_checks):
        # Enumerate actual 3-sector-2 + 3-sector-1 witnesses.
        # This is a selection statistic, not an exhaustive distance search.
        block_maps = [{} for _ in range(3)]
        mask = (1 << q) - 1
        for c in range(n1):
            block = (c // q) % 3 if x_checks else (c // q) // 3
            s = (cols[c] >> (block * q)) & mask
            block_maps[block][s] = 1 << c
        tail = cols[n1:]
        found = 0
        for i in range(q):
            for j in range(i):
                xy = tail[i] ^ tail[j]
                for k in range(j):
                    s = xy ^ tail[k]
                    v = (1 << (n1 + i)) | (1 << (n1 + j)) | (1 << (n1 + k))
                    for block in range(3):
                        match = block_maps[block].get(
                            (s >> (block * q)) & mask)
                        if match is None:
                            break
                        v |= match
                    else:
                        if remainder(v, stabilizers):
                            found += 1
        return found

    amask = sum(((1 << q) - 1) << (3 * j * q) for j in range(3))
    bmask = (1 << (3 * q)) - 1
    pool = []
    fallback = None

    # Offset search, not independent uniform final fillings.
    for trial in range(96):
        centers = set()
        cells = {}
        for side, gaps in (("A", ga), ("B", gb)):
            row = []
            for d in gaps:
                u = rng.randrange(m)
                while (2 * u + d) % m in centers:
                    u = rng.randrange(m)
                centers.add((2 * u + d) % m)
                row.append([G.e, (u, 1), ((u + d) % m, 1)])
            cells[side] = [row]

        hx, hz, n = lifted_matrices(G, cells["A"], cells["B"])
        if fallback is None:
            fallback = cells

        # Extract component columns from the verified construction.
        aa = columns([r & amask for r in hx[:q]], n)
        ac = [aa[3 * j * q + t] for j in range(3) for t in range(q)]
        bc = columns([r & bmask for r in hz[:q]], 3 * q)
        qa = pair_quality(ac)
        qb = pair_quality(bc)
        if qa is None or qb is None:
            continue

        cheap = max(qa, qb) + (qa + qb) // 4
        pool.append((cheap, trial, cells, hx, hz, n, ac, bc))
        pool.sort(key=lambda z: z[:2])
        del pool[4:]

    best = None
    best_score = None
    for cheap, trial, cells, hx, hz, n, ac, bc in pool:
        if n - gf2_rank(hx) - gf2_rank(hz) != 232:
            continue
        bx, bz = basis(hx), basis(hz)
        sx, sz = columns(hx, n), columns(hz, n)
        if not no_small_logical(sx, bz):
            continue
        if not no_small_logical(sz, bx):
            continue

        a6, b6 = six_collisions(ac), six_collisions(bc)
        z6 = mixed_sixes(sx, bz, True)
        x6 = mixed_sixes(sz, bx, False)
        score = (max(a6, b6) // 10 + x6 + z6,
                 a6 + b6, x6 + z6, cheap, trial)
        if best_score is None or score < best_score:
            best_score, best = score, cells

    # Preserve legal candidates if an unusually difficult profile fails
    # the preferred screening criteria; the evaluator measures it as-is.
    if best is None:
        best = pool[0][2] if pool else fallback
    hx, hz, n = lifted_matrices(G, best["A"], best["B"])
    assert css_ok(hx, hz)
    assert n - gf2_rank(hx) - gf2_rank(hz) > 0
    return best


# Five support geometries, each with B row sums eight.
# A row sums are six and column sums four.
# Therefore HX <= 6+5=11 and HZ <= 8+4=12.
_C37_SUPPORTS = (
    ((3, 3, 1, 1), (1, 1, 3, 3)),
    ((3, 2, 2, 1), (1, 2, 2, 3)),
    ((2, 2, 2, 2), (2, 2, 2, 2)),
    ((2, 3, 1, 2), (2, 1, 2, 3)),
    ((3, 2, 2, 1), (1, 1, 2, 4)),
)
_C37_GAPS = (
    (1, 2, 4, 8, 12, 17),
    (1, 3, 7, 9, 14, 18),
    (1, 1, 3, 5, 8, 13),
    (1, 2, 5, 6, 11, 16),
)


def _c37_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length()-1] |= 1 << r
            row ^= bit
    return cols


def _c37_short(cols):
    """Exact kernel exclusion through four, for arbitrary column parity."""
    seen = {0}
    for col in cols:
        if col in seen:
            return False
        seen.add(col)
    for i, col in enumerate(cols):
        for other in cols[:i]:
            syndrome = col ^ other
            if syndrome in seen:
                return False
            seen.add(syndrome)
    return True


def _c37_cycles(cols):
    counts = {}
    total = 0
    for col in cols:
        neighbors = []
        while col:
            bit = col & -col
            neighbors.append(bit.bit_length()-1)
            col ^= bit
        for i, a in enumerate(neighbors):
            for b in neighbors[:i]:
                key = (b, a)
                old = counts.get(key, 0)
                total += old
                counts[key] = old + 1
    return total


def _c37_basis(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length()-1
            old = pivots.get(p)
            if old is None:
                pivots[p] = word
                break
            word ^= old
    return pivots


def _c37_outside(word, span):
    while word:
        p = word.bit_length()-1
        old = span.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _c37_probe(cols, rng, trials, seeds=()):
    """Retain actual kernel words; a high reading is not an exclusion."""
    memory = sorted({w for w in seeds if w},
                    key=lambda w: (w.bit_count(), w))[:40]
    order = list(range(len(cols)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                dependencies.append(word)

        words = sorted(set(memory+dependencies),
                       key=lambda w: (w.bit_count(), w))
        for sweep in range(2):
            reducers = words[:32]
            reduced = []
            for word in words:
                weight = word.bit_count()
                for other in reducers:
                    trial = word ^ other
                    tw = trial.bit_count()
                    if 0 < tw < weight:
                        word, weight = trial, tw
                reduced.append(word)
            words = sorted(set(reduced),
                           key=lambda w: (w.bit_count(), w))
        discoveries = set(words+memory)
        short = words[:32]
        for i, word in enumerate(short):
            for other in short[:i]:
                trial = word ^ other
                if trial:
                    discoveries.add(trial)
        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:40]
    return tuple(memory)


def _c37_cofactors(G, grid, cols, inverse):
    """Commuting minors, accepted only after verified syndrome checks.

    These are group-algebra word constructions, not another matrix lift.
    Each three-column restriction of a two-row matrix supplies a syzygy.
    """
    from itertools import combinations

    q, count = G.q, len(grid[0])
    entries = [
        [set(G.inv(g) if inverse else g for g in cell) for cell in row]
        for row in grid
    ]

    def product(left, right):
        result = set()
        for g in left:
            for h in right:
                term = G.mul(g, h)
                if term in result:
                    result.remove(term)
                else:
                    result.add(term)
        return result

    found = set()
    for triple in combinations(range(count), 3):
        minors = {}
        for omitted in triple:
            s, t = [j for j in triple if j != omitted]
            minors[omitted] = (
                product(entries[0][s], entries[1][t])
                ^ product(entries[0][t], entries[1][s]))
        for shift in G.elements:
            word = 0
            for j, minor in minors.items():
                for g in minor:
                    word ^= 1 << (j*q + G.idx(G.mul(g, shift)))
            if not word:
                continue
            syndrome, remaining = 0, word
            while remaining:
                bit = remaining & -remaining
                syndrome ^= cols[bit.bit_length()-1]
                remaining ^= bit
            if syndrome == 0:
                found.add(word)
    return tuple(found)


def _c37_draw(rng, G, counts, mode, side, lane):
    """Prescribed relative-gap cards with independently optimized voltages."""
    gaps = _C37_GAPS[mode]
    larger = (
        None,
        (0, 1, 4, 10, 18),
        (0, 1, 2, 3, 4),
        (0, 1, 3, 7, 15),
    )
    grid = []
    for i, sizes in enumerate(counts):
        row = []
        for j, size in enumerate(sizes):
            if size == 0:
                row.append([])
                continue
            anchor = rng.randrange(37)
            if size == 1:
                residues = [anchor]
            elif size == 2:
                position = 3*i+j if side == "A" else (4*i+j+mode) % 6
                gap = (gaps[position] + 3*lane) % 37 or 1
                gap *= rng.choice((-1, 1))
                residues = [anchor, (anchor+gap) % 37]
            elif mode == 0:
                residues = rng.sample(range(37), size)
            else:
                step = rng.randrange(1, 37)
                residues = [(anchor+step*t) % 37
                            for t in larger[mode][:size]]
            cell = [(a,) for a in residues]
            assert len(cell) == len(set(cell)) == size
            row.append(cell)
        grid.append(row)
    return grid


def _c37_dressed(item, q):
    """Actual component embeddings, exact membership and pair dressings."""
    mask = (1 << q)-1
    best = item["n"]+1
    for side, seeds, checks, stabilizers in (
            ("A", item["a"]["words"], item["hx"], item["hz"]),
            ("B", item["b"]["words"], item["hz"], item["hx"])):
        span = _c37_basis(stabilizers)
        for seed in seeds[:10]:
            for fixed in range(4 if side == "A" else 3):
                if side == "A":
                    word = sum(
                        ((seed >> (j*q)) & mask)
                        << ((4*j+fixed)*q) for j in range(3))
                else:
                    word = seed << (fixed*4*q)
                if any((word & row).bit_count() & 1 for row in checks):
                    continue
                if not _c37_outside(word, span):
                    continue
                best = min(best, word.bit_count())
                for sweep in range(3):
                    chosen, weight = word, word.bit_count()
                    nearby = sorted(
                        (((word ^ row).bit_count(), row)
                         for row in stabilizers),
                        key=lambda x: x[0])
                    for cw, row in nearby:
                        if cw < weight:
                            chosen, weight = word ^ row, cw
                    moves = [row for _, row in nearby[:12]]
                    for i, row in enumerate(moves):
                        for other in moves[:i]:
                            trial = word ^ row ^ other
                            tw = trial.bit_count()
                            if tw < weight:
                                chosen, weight = trial, tw
                    if chosen == word:
                        break
                    word = chosen
                    best = min(best, weight)
    return best


def _c37_full_probe(checks, stabilizers, n, rng):
    """Bounded mixed-sector logical search; sentinels are not witnesses."""
    cols = _c37_columns(checks, n)
    span = _c37_basis(stabilizers)
    order = list(range(n))
    best = n+1
    for _ in range(3):
        rng.shuffle(order)
        pivots, words = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                words.append(word)
        reducers = sorted(words, key=int.bit_count)[:32]
        logicals = sorted(
            (w for w in words if _c37_outside(w, span)),
            key=int.bit_count)[:10]
        for word in logicals:
            for sweep in range(2):
                chosen, weight = word, word.bit_count()
                for other in reducers:
                    trial = word ^ other
                    tw = trial.bit_count()
                    if (0 < tw < weight
                            and _c37_outside(trial, span)):
                        chosen, weight = trial, tw
                if chosen == word:
                    break
                word = chosen
            # Actual stabilizers preserve the nonzero logical class.
            for row in stabilizers:
                trial = word ^ row
                if trial.bit_count() < word.bit_count():
                    word = trial
            best = min(best, word.bit_count())
    return best


def _c37_signature(hx, hz, n):
    """Uncoloured Tanner invariant; inequality certifies nonisomorphism."""
    from collections import Counter

    rows = hx + hz
    weights = [r.bit_count() for r in rows]
    profiles = [Counter() for _ in rows]
    adjacency = [0] * len(rows)
    histogram = Counter()
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            histogram[overlap] += 1
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
                profiles[i][weights[j], overlap] += 1
                profiles[j][weights[i], overlap] += 1
    refined = Counter()
    for i, neighbors in enumerate(adjacency):
        triangles, remaining = 0, neighbors
        while remaining:
            bit = remaining & -remaining
            j = bit.bit_length()-1
            triangles += (neighbors & adjacency[j]).bit_count()
            remaining ^= bit
        refined[weights[i], tuple(sorted(profiles[i].items())),
                triangles//2] += 1
    degrees = Counter(c.bit_count() for c in _c37_columns(rows, n))
    return (n, tuple(sorted(histogram.items())),
            tuple(sorted(refined.items())), tuple(sorted(degrees.items())))


@__import__("functools").lru_cache(maxsize=1)
def _c37_bank(G):
    """Pure deterministic bank: twenty tasks, not twenty random outputs."""
    assert G.q == 37
    q = G.q
    mask = (1 << q)-1
    zero_a = [[[] for _ in range(3)] for _ in range(2)]
    zero_b = [[[] for _ in range(4)] for _ in range(2)]

    # Compile both actions from scaffold-generated single-term blocks.
    # Subsequent compilation is only F2-linear superposition of these.
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[] for _ in range(3 if side == "A" else 4)]
                    for _ in range(2)]
            unit[0][0] = [g]
            A, B = (unit, zero_b) if side == "A" else (zero_a, unit)
            hx, hz, _ = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            blocks.append(tuple(_c37_columns(
                [row & mask for row in source[:q]], q)))
        atlas[side] = tuple(blocks)

    def component(side, grid):
        width = len(grid[0])*q
        cols = [0] * width
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = atlas[side][G.idx(g)]
                    for h, syndrome in enumerate(block):
                        cols[j*q+h] ^= syndrome << (i*q)
        return cols

    def component_pool(local, side, counts, mode, slot, batch):
        pool, seen = [], set()

        def merit(item):
            return (item["d"], -item["load"], -item["cycles"])

        for attempt in range(56 + 24*batch):
            grid = _c37_draw(
                local, G, counts, mode, side, slot//20)
            labeling = tuple(tuple(sorted(cell))
                             for row in grid for cell in row)
            if labeling in seen:
                continue
            seen.add(labeling)
            cols = component(side, grid)
            if not _c37_short(cols):
                continue
            seeds = _c37_cofactors(G, grid, cols, side == "B")
            words = _c37_probe(cols, local, 2, seeds)
            if not words:
                continue
            d = words[0].bit_count()
            item = dict(
                grid=grid, cols=cols, words=words, d=d,
                cycles=_c37_cycles(cols),
                load=sum(w.bit_count() == d for w in words))
            pool.append(item)
            pool.sort(key=merit, reverse=True)
            del pool[8:]

        for item in pool:
            item["words"] = _c37_probe(
                item["cols"], local, 20, item["words"])
            item["d"] = min(item["d"], item["words"][0].bit_count())
            item["load"] = sum(w.bit_count() == item["d"]
                               for w in item["words"])
        pool.sort(key=merit, reverse=True)
        pool = pool[:4]
        for item in pool:
            item["words"] = _c37_probe(
                item["cols"], local, 48, item["words"])
            item["d"] = min(item["d"], item["words"][0].bit_count())
            item["load"] = sum(w.bit_count() == item["d"]
                               for w in item["words"])
        return sorted(pool, key=merit, reverse=True)

    def assembled_merit(item):
        d = min(item["a"]["d"], item["b"]["d"],
                item.get("logical", item["n"]+1),
                item.get("full", item["n"]+1))
        return (item["k"]*d*d/item["n"], d,
                -item["a"]["cycles"]-item["b"]["cycles"])

    bank, emitted = [], set()
    for slot in range(N_SLOTS):
        geometry, mode = divmod(slot % 20, 4)
        sa = ((2, 2, 2), (2, 2, 2))
        sb = _C37_SUPPORTS[geometry]
        assert _grid_weights(sa, sb) <= 12
        assert _no_dead_lines(sa, sb)
        local = random.Random(
            BASE_SEED*100003 + 37015241 + 104729*slot)
        chosen = None
        for batch in range(3):
            ap = component_pool(local, "A", sa, mode, slot, batch)
            bp = component_pool(local, "B", sb, mode, slot, batch)
            proposals = []
            for a in ap:
                for b in bp:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (130*n <= 1000*k < 175*n):
                        continue
                    # Check compiled columns against a complete verified lift.
                    ar = []
                    for i in range(2):
                        for h in range(q):
                            source = hx[(i*4)*q+h]
                            ar.append(sum(
                                ((source >> (j*4*q)) & mask) << (j*q)
                                for j in range(3)))
                    br = [row & ((1 << (4*q))-1)
                          for row in hz[:2*q]]
                    assert _c37_columns(ar, 3*q) == a["cols"]
                    assert _c37_columns(br, 4*q) == b["cols"]
                    proposals.append(dict(
                        A=A, B=B, hx=hx, hz=hz, n=n, k=k, a=a, b=b))

            proposals.sort(key=assembled_merit, reverse=True)
            finalists = []
            for item in proposals[:6]:
                if not _c37_short(_c37_columns(item["hx"], item["n"])):
                    continue
                if not _c37_short(_c37_columns(item["hz"], item["n"])):
                    continue
                item["logical"] = _c37_dressed(item, q)
                finalists.append(item)
            finalists.sort(key=assembled_merit, reverse=True)
            for item in finalists[:2]:
                item["full"] = min(
                    _c37_full_probe(
                        item["hx"], item["hz"], item["n"], local),
                    _c37_full_probe(
                        item["hz"], item["hx"], item["n"], local))

            for item in sorted(finalists, key=assembled_merit, reverse=True):
                signature = _c37_signature(
                    item["hx"], item["hz"], item["n"])
                if signature in emitted:
                    continue
                assert item["n"] == 592 and item["k"] >= 74
                assert css_ok(item["hx"], item["hz"])
                assert max(r.bit_count()
                           for r in item["hx"]+item["hz"]) <= 12
                assert all(
                    len(cell) == len(set(cell)) <= 5
                    for M in (item["A"], item["B"])
                    for row in M for cell in row)
                emitted.add(signature)
                chosen = tuple(
                    tuple(tuple(tuple(cell) for cell in row) for row in M)
                    for M in (item["A"], item["B"]))
                break
            if chosen is not None:
                break
        # Never fill a failed task with an already emitted orbit copy.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return the independently optimized task's immutable bank entry."""
    item = _c37_bank(G)[idx]
    if item is None:
        raise ValueError("no fresh C37 filling in target rate band")
    A, B = item
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without additional randomness."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled counts with uniform labels suffice to escape short sector-1 logicals at rate 0.400; many seed slots have d2s=2, and all five measured seed codes have d30s=4 with sector-1 minima.
# REFUTED: Canonical reflection-gap profiles ensure Tanner-nonisomorphic outputs; the measured current parent emitted only 35 distinct codes from 60 slots, with the remaining slots collapsed as isomorphic copies.
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

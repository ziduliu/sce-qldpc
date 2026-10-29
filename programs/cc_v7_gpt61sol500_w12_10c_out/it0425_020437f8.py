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
CONCEPT = ("Dihedral D15, order 30, shapes (1,3)x(2,5): aligning A trinomial images in the quotient D3 while independently varying their C5 voltages should produce fresh verified k>=188 codes retaining five d30s>=8 readings and improve cell 0.350 beyond 23.5290. This is a falsifiable measured-score hypothesis.")
GROUP = ("L3 jump from Q32 to D15: x^15=y^2=1 and y*x*y^-1=x^-1. Different order and noncentral involutions rule out a group relabeling. No order-30 bed is listed. The quotient D15 -> D3 supplies a candidate two-dimensional nontrivial A rank deficit; actual component and full ranks, not the quotient argument, decide acceptance.")
SHAPES = ("Keep A=(1,3), B=(2,5) for cell 0.350. At q=30, n=510, kb=180 and the guaranteed rate is 6/17; component variable/check counts are 90/30 and 150/60. The shapes permit q<=35, while this family fixes q=30. Parent measured minima were entirely sector-one Acol/Bcol witnesses, especially Acol1, not Arow/Brow. Gate assembled rates in [0.325,0.375); k=188 or 190 at reading eight would score 23.5922 or 23.8431.")
FILLING = ("Twenty cards cross five intrinsic A gap geometries with four B architectures. A cells are {e,x^d,x^a*y}, with gcd(d,15)=1 and a=d mod 3, targeting a common quotient-D3 image and verified A rank 28. B tests rotation-only, full-dihedral and three-active-column architectures; pure trinomials use order-five rotation gaps. Every component column is cubic. A row total nine and B totals 8/7 or 6/9 give HX<=12 and HZ<=12. Compile only verified unit lifts, exactly exclude component kernels through six before shortlisting, deepen actual witness searches, and test both complete logical quotients with stabilizer dressings. Full ranks and CSS are checked; unequal uncoloured Tanner invariants separate emitted codes. Found-witness readings remain upper bounds.")
FAMILY_NAME = "r035_D15_13x25_D3image_rank28_preclean6_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 30

def custom_ops(q):
    # D15: (a,b)=x^a*y^b, x^15=y^2=1.
    if not valid_q(q):
        raise ValueError("D15 requires order 30")
    els = [(a,b) for b in range(2) for a in range(15)]

    def red(g):
        return (g[0] % 15, g[1] % 2)

    def mul(g, h):
        a,b = red(g)
        c,d = red(h)
        return ((a + (c if b == 0 else -c)) % 15, (b+d) % 2)

    def inv(g):
        a,b = red(g)
        return ((-a if b == 0 else a) % 15, b)

    def idx(g):
        a,b = red(g)
        return 15*b+a

    return dict(elements=els, ident=(0,0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


from functools import lru_cache
from collections import Counter

_Q32_A_STEPS = (
    (1, 2, 3), (1, 2, 4), (1, 3, 5),
    (1, 3, 6), (1, 4, 7),
)
# Active even-row cells, outer-coset cells, prescribed differences.
# Every difference is in <x^2>; each card includes an order-eight one.
_Q32_B_CARDS = (
    (4, 1, (2, 4, 6, 10)),
    (4, 2, (2, 2, 6, 6)),
    (4, 3, (2, 4, 10, 14)),
    (3, 1, (2, 6, 10)),
)


def _verified_component_columns(G, side, cells):
    """Extract component syndromes from the verified LP construction."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    if side == "A":
        zero = [[[] for _ in range(cb)] for _ in range(rb)]
        hx, _, _ = lifted_matrices(G, cells, zero)
        rows = [hx[(i * cb) * q + t]
                for i in range(ra) for t in range(q)]
        positions = [j * cb * q + t
                     for j in range(ca) for t in range(q)]
    else:
        zero = [[[] for _ in range(ca)] for _ in range(ra)]
        _, hz, _ = lifted_matrices(G, zero, cells)
        rows = hz[:rb * q]
        positions = range(cb * q)
    return [sum(((row >> p) & 1) << r
                for r, row in enumerate(rows)) for p in positions]


@lru_cache(maxsize=2)
def _q32_atlas(G, side):
    """Compile F2-linear contributions exclusively from verified lifts."""
    shape = A_SHAPE if side == "A" else B_SHAPE
    blocks = []
    for g in G.elements:
        unit = [[[] for _ in range(shape[1])]
                for _ in range(shape[0])]
        unit[0][0] = [g]
        blocks.append(tuple(
            _verified_component_columns(G, side, unit)[:G.q]))
    return tuple(blocks)


def _component_columns(G, side, cells):
    """Superpose verified unit blocks; never implement group actions."""
    q = G.q
    width = len(cells[0])*q
    cols = [0]*width
    atlas = _q32_atlas(G, side)
    for i, row in enumerate(cells):
        for j, cell in enumerate(row):
            for g in cell:
                block = atlas[G.idx(g)]
                for t, syndrome in enumerate(block):
                    cols[j*q+t] ^= syndrome << (i*q)
    return cols


def _q32_columns(rows, width):
    cols = [0]*width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            cols[bit.bit_length()-1] |= 1 << r
            source ^= bit
    return cols


def _q32_short4(cols):
    """Exact component exclusion through four."""
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


def _q32_probe(cols, rng, trials, memory=()):
    """Retain true classical kernel witnesses; readings are upper bounds."""
    width = len(cols)
    memory = sorted(set(memory), key=lambda w: (w.bit_count(), w))[:24]
    order = list(range(width))
    best = min((w.bit_count() for w in memory), default=width+1)
    for trial in range(trials):
        rng.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = syndrome, word
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                dependencies.append(word)
                best = min(best, word.bit_count())

        words = sorted(set(dependencies+memory),
                       key=lambda w: (w.bit_count(), w))
        for sweep in range(2):
            reducers = words[:24]
            reduced = []
            for word in words:
                weight = word.bit_count()
                for other in reducers:
                    combined = word ^ other
                    cw = combined.bit_count()
                    if combined and cw < weight:
                        word, weight = combined, cw
                reduced.append(word)
                best = min(best, weight)
            words = sorted(set(reduced),
                           key=lambda w: (w.bit_count(), w))

        short = sorted(set(words+memory),
                       key=lambda w: (w.bit_count(), w))[:24]
        discoveries = list(short)
        for i, word in enumerate(short):
            for other in short[:i]:
                combined = word ^ other
                if combined:
                    discoveries.append(combined)
                    best = min(best, combined.bit_count())
        if trials >= 12 and trial % 4 == 0:
            for i, word in enumerate(short[:8]):
                for j in range(i):
                    for h in range(j):
                        combined = word ^ short[j] ^ short[h]
                        if combined:
                            discoveries.append(combined)
                            best = min(best, combined.bit_count())
        memory = sorted(set(discoveries),
                        key=lambda w: (w.bit_count(), w))[:24]
        if best <= 6:
            break
    return best, tuple(memory)


def _four_cycle_penalty(cols, q):
    pairs = {}
    collisions = 0
    for syndrome in cols:
        neighbors = []
        while syndrome:
            bit = syndrome & -syndrome
            neighbors.append(bit.bit_length() - 1)
            syndrome ^= bit
        for a in range(len(neighbors)):
            for b in range(a):
                key = (neighbors[b], neighbors[a])
                collisions += pairs.get(key, 0)
                pairs[key] = pairs.get(key, 0) + 1
    return collisions / float(q)


def _kernel_score(cols):
    """Detect all possible kernel weights <=6 by subset collisions.

    A reading of 7 means no such word was detected by this exhaustive
    component test; it says nothing conclusive about mixed-sector logicals.
    """
    seen = {0: 0}
    best, hits = 7, 0

    def record(syndrome, mask):
        nonlocal best, hits
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = mask
        else:
            difference = previous ^ mask
            if difference:
                weight = difference.bit_count()
                if weight <= 6:
                    best = min(best, weight)
                    hits += 1
            if mask.bit_count() < previous.bit_count():
                seen[syndrome] = mask

    bits = [1 << i for i in range(len(cols))]
    for i, ci in enumerate(cols):
        record(ci, bits[i])
    for i, ci in enumerate(cols):
        for j in range(i):
            record(ci ^ cols[j], bits[i] | bits[j])
    for i, ci in enumerate(cols):
        for j in range(i):
            syndrome = ci ^ cols[j]
            mask = bits[i] | bits[j]
            for t in range(j):
                record(syndrome ^ cols[t], mask | bits[t])
    return best, -hits


def _copy_cells(cells):
    return [[list(cell) for cell in row] for row in cells]


def _support_design(rng, idx):
    active, _, _ = _Q32_B_CARDS[idx % 4]
    sa = [[3, 3, 3]]
    sb = [[2]*active + [0]*(5-active),
          [1]*active + [3]*(5-active)]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    return sa, sb


def _q32_draw(rng, G, side, idx):
    """Prescribed relative geometries, not common relabeling operations."""
    if side == "A":
        steps = _Q32_A_STEPS[(idx % 20)//4]
        return [[
            [G.e, (d, 0), (0 if j == 0 else rng.randrange(16), 1)]
            for j, d in enumerate(steps)]]

    active, reflected, differences = _Q32_B_CARDS[idx % 4]
    top, bottom = [], []
    for j, difference in enumerate(differences):
        a = rng.randrange(16)
        b = int(j < reflected)
        top.append([(a, b), ((a+difference) % 16, b)])
        bottom.append([G.e])
    for j in range(active, 5):
        top.append([])
        if (idx+j) % 2:
            bottom.append([G.e, (rng.randrange(1, 16), 0),
                           (rng.randrange(16), 1)])
        else:
            u, v = rng.sample(range(16), 2)
            bottom.append([G.e, (u, 1), (v, 1)])
    return [top, bottom]


def _design_component(rng, G, side, counts, idx):
    """Independent proposals plus repairs directed by actual kernel words."""
    required_rank = 32 if side == "A" else 60
    pool, labels = [], set()
    width = len(counts[0])

    def merit(item):
        return (min(item["reading"], 10),
                -sum(w.bit_count() <= 8 for w in item["words"]),
                -item["cycles"])

    def admit(cells):
        signature = tuple(tuple(sorted(cell))
                          for row in cells for cell in row)
        if signature in labels:
            return
        labels.add(signature)
        cols = _component_columns(G, side, cells)
        assert all(col.bit_count() == 3 for col in cols)
        if gf2_rank(cols) != required_rank or not _q32_short4(cols):
            return
        reading, words = _q32_probe(cols, rng, 3)
        item = dict(cells=cells, cols=cols, reading=reading, words=words,
                    cycles=_four_cycle_penalty(cols, G.q))
        pool.append(item)
        pool.sort(key=merit, reverse=True)
        del pool[8:]

    quota = 72 if side == "A" else 112
    for attempt in range(quota):
        admit(_q32_draw(rng, G, side, idx))

    for step in range(40 if side == "A" else 56):
        if not pool:
            admit(_q32_draw(rng, G, side, idx))
            continue
        parent = pool[rng.randrange(min(4, len(pool)))]
        pressure = [
            1 + sum(((word >> (j*G.q)) & ((1 << G.q)-1)).bit_count()
                    for word in parent["words"][:8])
            for j in range(width)]
        j = rng.choices(range(width), weights=pressure, k=1)[0]
        proposal = _q32_draw(rng, G, side, idx)
        cells = _copy_cells(parent["cells"])
        positions = {j}
        if step % 7 == 0:
            positions.add(rng.randrange(width))
        for column in positions:
            for i in range(len(cells)):
                cells[i][column] = proposal[i][column]
        admit(cells)

    for item in pool:
        reading, words = _q32_probe(
            item["cols"], rng, 12, item["words"])
        item["reading"] = min(item["reading"], reading)
        item["words"] = words
    pool.sort(key=merit, reverse=True)

    finalists = []
    for item in pool[:4]:
        flag, negative_hits = _kernel_score(item["cols"])
        # Cubic columns prohibit odd-weight kernel words. Flag seven
        # therefore excludes component words through seven, not LP words.
        item["floor"] = 8 if flag == 7 else flag
        item["hits"] = -negative_hits
        reading, words = _q32_probe(
            item["cols"], rng, 32, item["words"])
        item["reading"] = min(item["reading"], reading)
        item["words"] = words
        finalists.append(item)
    finalists.sort(key=lambda item: (
        min(item["floor"], 8),) + merit(item), reverse=True)
    return finalists[:3]


def _q32_basis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length()-1
            old = basis.get(p)
            if old is None:
                basis[p] = word
                break
            word ^= old
    return basis


def _q32_outside(word, basis):
    while word:
        p = word.bit_length()-1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _q32_small_logical(checks, stabilizers, n):
    """Exact stabilizer-aware logical exclusion through weight four."""
    span = _q32_basis(stabilizers)
    cols = _q32_columns(checks, n)
    seen = {0: 0}
    bits = [1 << i for i in range(n)]
    for i, syndrome in enumerate(cols):
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = bits[i]
        elif _q32_outside(bits[i] ^ old, span):
            return True
    for i, ci in enumerate(cols):
        for j in range(i):
            syndrome = ci ^ cols[j]
            word = bits[i] | bits[j]
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif _q32_outside(word ^ old, span):
                return True
    return False


def _q32_dressed(item, q):
    """Actual logical witnesses, verified before stabilizer dressings."""
    mask = (1 << q)-1
    best = item["n"]+1
    for side, seeds, checks, stabilizers in (
            ("A", item["a"]["words"], item["hx"], item["hz"]),
            ("B", item["b"]["words"], item["hz"], item["hx"])):
        span = _q32_basis(stabilizers)
        fixed_count = B_SHAPE[1] if side == "A" else A_SHAPE[1]
        for seed in seeds[:5]:
            for fixed in range(fixed_count):
                if side == "A":
                    word = sum(
                        ((seed >> (j*q)) & mask)
                        << ((j*B_SHAPE[1]+fixed)*q)
                        for j in range(A_SHAPE[1]))
                else:
                    word = seed << (fixed*B_SHAPE[1]*q)
                if any((word & row).bit_count() & 1 for row in checks):
                    continue
                if not _q32_outside(word, span):
                    continue
                best = min(best, word.bit_count())
                for sweep in range(3):
                    chosen, weight = word, word.bit_count()
                    nearby = []
                    for row in stabilizers:
                        candidate = word ^ row
                        cw = candidate.bit_count()
                        nearby.append((cw, row))
                        if cw < weight:
                            chosen, weight = candidate, cw
                    nearby.sort(key=lambda pair: pair[0])
                    moves = [row for _, row in nearby[:10]]
                    for i, row in enumerate(moves):
                        for other in moves[:i]:
                            candidate = word ^ row ^ other
                            cw = candidate.bit_count()
                            if cw < weight:
                                chosen, weight = candidate, cw
                    if chosen == word:
                        break
                    word = chosen
                    best = min(best, weight)
    return best


def _q32_fingerprint(hx, hz, n):
    """Different uncoloured incidence invariants certify nonisomorphism."""
    rows = hx + hz
    adjacency = [dict() for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                adjacency[i][j] = overlap
                adjacency[j][i] = overlap
    profiles = []
    for r, neighbors in enumerate(adjacency):
        items = list(neighbors.items())
        triangles = 0
        for i, (s, ws) in enumerate(items):
            for t, wt in items[:i]:
                triangles += 2*ws*wt*adjacency[s].get(t, 0)
        profiles.append((
            rows[r].bit_count(),
            tuple(sorted(Counter(
                (rows[s].bit_count(), weight)
                for s, weight in items).items())),
            sum(weight*weight for _, weight in items),
            triangles))
    degrees = Counter(col.bit_count() for col in _q32_columns(rows, n))
    return (tuple(sorted(Counter(profiles).items())),
            tuple(sorted(degrees.items())))


@lru_cache(maxsize=1)
def _q32_bank(G):
    """Deterministic card bank; never substitute an emitted orbit copy."""
    assert G.q == 32
    bank, emitted = [], set()
    for slot in range(N_SLOTS):
        sa, sb = _support_design(None, slot)
        chosen = None
        for batch in range(2):
            local = random.Random(
                BASE_SEED*100003 + 32013560 + 7919*slot + 104729*batch)
            ap = _design_component(local, G, "A", sa, slot)
            bp = _design_component(local, G, "B", sb, slot)
            proposals = []
            for a in ap:
                for b in bp:
                    A, B = a["cells"], b["cells"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if n != 544 or k != 200:
                        continue
                    # Explicitly verify the compiled atlas on full components.
                    assert _verified_component_columns(G, "A", A) == a["cols"]
                    assert _verified_component_columns(G, "B", B) == b["cols"]
                    d = min(a["reading"], b["reading"], 10)
                    key = (min(a["floor"], b["floor"], 8),
                           k*d*d, a["reading"]+b["reading"],
                           -a["cycles"]-b["cycles"])
                    proposals.append((key, dict(
                        A=A, B=B, a=a, b=b, hx=hx, hz=hz, n=n, k=k)))

            proposals.sort(key=lambda pair: pair[0], reverse=True)
            tested = []
            for key, item in proposals[:5]:
                item["logical"] = _q32_dressed(item, G.q)
                d = min(item["a"]["reading"], item["b"]["reading"],
                        item["logical"], 10)
                tested.append((
                    (key[0], item["k"]*d*d, d, key[2], key[3]), item))
            tested.sort(key=lambda pair: pair[0], reverse=True)

            for key, item in tested:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                if _q32_small_logical(hx, hz, n):
                    continue
                if _q32_small_logical(hz, hx, n):
                    continue
                fingerprint = _q32_fingerprint(hx, hz, n)
                if fingerprint in emitted:
                    continue
                assert css_ok(hx, hz)
                assert max(row.bit_count() for row in hx+hz) <= W_MAX
                assert 325*n <= 1000*item["k"] < 375*n
                assert all(len(cell) == len(set(cell)) <= 5
                           for grid in (item["A"], item["B"])
                           for row in grid for cell in row)
                emitted.add(fingerprint)
                chosen = tuple(
                    tuple(tuple(tuple(cell) for cell in row) for row in grid)
                    for grid in (item["A"], item["B"]))
                break
            if chosen is not None:
                break
        bank.append(chosen)
    return tuple(bank)


# The historical Q32 optimizer is inactive. Its generic syndrome,
# atlas, basis, witness and incidence helpers remain group-independent.
_D30_A_GAPS = (
    (1,2,4), (1,2,7), (1,4,8), (2,4,7), (1,7,11),
)


def _d30_clean6(cols, q, height):
    """Exact component exclusion through six after short-four exclusion.

    Cubic columns exclude odd kernel weights. The singleton/pair table
    is unique after the preceding test. Branching has no node cutoff.
    Coordinate-zero anchors use component regular-action transitivity
    only; no transitivity of the full quantum code is assumed.
    """
    bits = [1 << j for j in range(len(cols))]
    tails = {0: 0}
    incidence = [[] for _ in range(height)]
    for j, syndrome in enumerate(cols):
        assert syndrome.bit_count() == 3
        assert syndrome not in tails
        tails[syndrome] = bits[j]
        value = syndrome
        while value:
            bit = value & -value
            incidence[bit.bit_length()-1].append(j)
            value ^= bit
    for j, syndrome in enumerate(cols):
        for h in range(j):
            value = syndrome ^ cols[h]
            assert value not in tails
            tails[value] = bits[j] | bits[h]

    def decode(target, forbidden):
        def visit(value, remaining, word):
            if not value:
                return word
            if value.bit_count() > 3*remaining:
                return None
            if remaining <= 2:
                tail = tails.get(value)
                if (tail is not None and tail.bit_count() <= remaining
                        and not tail & (word | forbidden)):
                    return word | tail
                return None
            check = (value & -value).bit_length()-1
            choices = sorted(
                incidence[check],
                key=lambda j: -(cols[j] & value).bit_count())
            for j in choices:
                if bits[j] & (word | forbidden):
                    continue
                hit = visit(value ^ cols[j], remaining-1,
                            word | bits[j])
                if hit is not None:
                    return hit
            return None
        return visit(target, 5, 0)

    for anchor in range(0, len(cols), q):
        tail = decode(cols[anchor], bits[anchor])
        if tail is not None:
            word = tail | bits[anchor]
            value, syndrome = word, 0
            while value:
                bit = value & -value
                syndrome ^= cols[bit.bit_length()-1]
                value ^= bit
            assert syndrome == 0 and word.bit_count() <= 6
            return False
    return True


@lru_cache(maxsize=5)
def _d30_a_pool(G, geometry):
    """Enumerate actual quotient-aligned voltages, not translated copies."""
    from itertools import product

    gaps = _D30_A_GAPS[geometry]
    local = random.Random(BASE_SEED*100003 + 301328 + geometry)
    pool = []
    for parameters in product(range(5), repeat=3):
        A = [[
            [G.e, (d,0), ((d % 3)+3*t,1)]
            for d,t in zip(gaps, parameters)]]
        cols = _component_columns(G, "A", A)
        if gf2_rank(cols) != 28 or not _q32_short4(cols):
            continue
        if not _d30_clean6(cols, G.q, G.q):
            continue
        d, words = _q32_probe(cols, local, 4)
        cycles = _four_cycle_penalty(cols, G.q)
        pool.append(dict(
            cells=A, cols=cols, reading=d, words=words,
            cycles=cycles, floor=8))
        pool.sort(key=lambda item: (
            min(item["reading"], 12), -item["cycles"]),
            reverse=True)
        del pool[10:]

    # Independently confirm the bounded decoder on retained components.
    for item in pool:
        assert _kernel_score(item["cols"])[0] == 7
        d, words = _q32_probe(
            item["cols"], local, 64, item["words"])
        item["reading"] = min(item["reading"], d)
        item["words"] = words
        assert item["reading"] >= 8
    pool.sort(key=lambda item: (
        min(item["reading"], 12),
        -sum(w.bit_count() <= 8 for w in item["words"]),
        -item["cycles"]), reverse=True)
    return tuple(pool[:6])


def _d30_unit_triple(local, G):
    """Mixed trinomial with an order-five rotation gap; rank is still tested."""
    a = local.randrange(15)
    b = local.randrange(2)
    delta = local.choice((3,6,9,12))
    return [(a,b), ((a+delta) % 15,b),
            (local.randrange(15),1-b)]


def _d30_draw_b(local, G, mode):
    """Four genuinely different difference/incidence construction policies."""
    if mode == 0:
        directions = [(d,0) for d in (1,2,4,7)]
    elif mode == 1:
        directions = [(1,0), (2,0), (0,1), (1,1)]
    elif mode == 2:
        directions = [(1,0), (4,0), (7,0)]
    else:
        directions = [(1,0), (5,0), (2,0), (7,0)]
    local.shuffle(directions)
    top, bottom = [], []
    for direction in directions:
        anchor = G.pick(local)
        if local.randrange(2):
            direction = G.inv(direction)
        top.append([anchor, G.mul(direction, anchor)])
        bottom.append([G.pick(local)])
    while len(top) < 5:
        top.append([])
        bottom.append(_d30_unit_triple(local, G))
    return [top, bottom]


def _d30_b_pool(local, G, mode, retry):
    """Screen ordinary short words before applying heuristic selection."""
    pool, seen = [], set()
    admitted = 0

    def key(item):
        return (min(item["reading"], 12),
                -sum(w.bit_count() <= 8 for w in item["words"]),
                -item["cycles"], -item["rank"])

    def admit(B):
        nonlocal admitted
        signature = tuple(tuple(sorted(cell))
                          for row in B for cell in row)
        if signature in seen:
            return
        seen.add(signature)
        cols = _component_columns(G, "B", B)
        rank = gf2_rank(cols)
        # Permit either the rotation-subgroup or full-dihedral dependency.
        if rank not in (58,59):
            return
        if not _q32_short4(cols):
            return
        if not _d30_clean6(cols, G.q, 2*G.q):
            return
        d, words = _q32_probe(cols, local, 4)
        assert d >= 8
        pool.append(dict(
            cells=B, cols=cols, rank=rank, reading=d, words=words,
            cycles=_four_cycle_penalty(cols, G.q), floor=8))
        pool.sort(key=key, reverse=True)
        del pool[8:]
        admitted += 1

    for attempt in range(320+160*retry):
        admit(_d30_draw_b(local, G, mode))
        if admitted >= 20:
            break

    for step in range(64):
        if not pool:
            admit(_d30_draw_b(local, G, mode))
            continue
        parent = local.choice(pool[:min(3, len(pool))])
        B = _copy_cells(parent["cells"])
        fresh = _d30_draw_b(local, G, mode)
        pressure = [
            1+sum(((w >> (j*G.q)) & ((1 << G.q)-1)).bit_count()
                  for w in parent["words"][:8])
            for j in range(5)]
        j = local.choices(range(5), weights=pressure, k=1)[0]
        # Bottom-only changes preserve top difference multiplicities.
        B[1][j] = fresh[1][j]
        if step % 7 == 0 and B[0][j]:
            active = sum(bool(c) for c in B[0])
            replacement = local.randrange(active)
            B[0][j] = fresh[0][replacement]
        admit(B)

    for item in pool:
        d, words = _q32_probe(
            item["cols"], local, 24, item["words"])
        item["reading"] = min(item["reading"], d)
        item["words"] = words
    pool.sort(key=key, reverse=True)
    finalists = pool[:4]
    for item in finalists:
        assert _kernel_score(item["cols"])[0] == 7
        d, words = _q32_probe(
            item["cols"], local, 64, item["words"])
        item["reading"] = min(item["reading"], d)
        item["words"] = words
    return sorted(finalists, key=key, reverse=True)


def _d30_logical_probe(checks, stabilizers, n, local, trials):
    """Search true full-code logicals, with stabilizer-aware descent."""
    cols = _q32_columns(checks, n)
    span = _q32_basis(stabilizers)
    order = list(range(n))
    best = n+1
    memory = []

    def observe(word):
        nonlocal best
        if not word or not _q32_outside(word, span):
            return
        for sweep in range(2):
            changed = False
            for row in stabilizers:
                candidate = word ^ row
                if candidate.bit_count() < word.bit_count():
                    word = candidate
                    changed = True
            if not changed:
                break
        best = min(best, word.bit_count())

    for trial in range(trials):
        local.shuffle(order)
        if trial % 3 != 2:
            split = 15*(n//17)
            order.sort(key=lambda c:
                       (c < split) if trial % 3 == 0 else (c >= split))
        pivots, words = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = syndrome, word
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                words.append(word)

        logical = sorted(
            {w for w in words+memory if _q32_outside(w, span)},
            key=int.bit_count)[:16]
        reducers = sorted(set(words+memory), key=int.bit_count)[:32]
        discoveries = set(logical)
        for word in logical:
            for sweep in range(2):
                chosen, weight = word, word.bit_count()
                for other in reducers:
                    candidate = word ^ other
                    if (candidate and candidate.bit_count() < weight
                            and _q32_outside(candidate, span)):
                        chosen, weight = candidate, candidate.bit_count()
                if chosen == word:
                    break
                word = chosen
            observe(word)
            discoveries.add(word)
        for i, word in enumerate(logical):
            for other in logical[:i]:
                candidate = word ^ other
                observe(candidate)
                if candidate and _q32_outside(candidate, span):
                    discoveries.add(candidate)
        memory = sorted(discoveries, key=int.bit_count)[:24]
        if best <= 6:
            break
    return best


@lru_cache(maxsize=1)
def _d30_bank(G):
    """Twenty structural cards with complete-matrix rank and diversity gates."""
    assert G.q == 30
    assert A_SHAPE == (1,3) and B_SHAPE == (2,5)
    bank, emitted = [], set()

    def merit(item):
        d = min(item["a"]["reading"], item["b"]["reading"],
                item["logical"], item["dx"], item["dz"], 12)
        return (item["k"]*d*d/item["n"], d,
                item["a"]["reading"]+item["b"]["reading"],
                -item["a"]["cycles"]-item["b"]["cycles"])

    for slot in range(N_SLOTS):
        geometry, mode = divmod(slot % 20, 4)
        ap = _d30_a_pool(G, geometry)
        winner = None
        for retry in range(2):
            local = random.Random(
                BASE_SEED*100003+301350190+104729*slot+7919*retry)
            bp = _d30_b_pool(local, G, mode, retry)
            proposals = []
            for a in ap:
                for b in bp:
                    A, B = a["cells"], b["cells"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if n != 510 or not (325*n <= 1000*k < 375*n):
                        continue
                    if k < 188:
                        continue
                    assert _verified_component_columns(G, "A", A) == a["cols"]
                    assert _verified_component_columns(G, "B", B) == b["cols"]
                    d = min(a["reading"], b["reading"], 12)
                    proposals.append((
                        (k*d*d/n, d, -a["cycles"]-b["cycles"]),
                        dict(A=A, B=B, a=a, b=b,
                             hx=hx, hz=hz, n=n, k=k)))
            proposals.sort(key=lambda pair: pair[0], reverse=True)

            finalists = []
            for _, item in proposals:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                if _q32_small_logical(hx, hz, n):
                    continue
                if _q32_small_logical(hz, hx, n):
                    continue
                fingerprint = _q32_fingerprint(hx, hz, n)
                if fingerprint in emitted:
                    continue
                item["fingerprint"] = fingerprint
                item["logical"] = _q32_dressed(item, G.q)
                item["dx"] = _d30_logical_probe(hz, hx, n, local, 8)
                item["dz"] = _d30_logical_probe(hx, hz, n, local, 8)
                finalists.append(item)
                if len(finalists) == 3:
                    break

            # Equal fresh full-code budgets for every joint finalist.
            for item in finalists:
                item["dx"] = min(item["dx"], _d30_logical_probe(
                    item["hz"], item["hx"], item["n"], local, 24))
                item["dz"] = min(item["dz"], _d30_logical_probe(
                    item["hx"], item["hz"], item["n"], local, 24))
                if winner is None or merit(item) > merit(winner):
                    winner = item
            if winner is not None and merit(winner)[1] >= 8:
                break

        if winner is None:
            bank.append(None)
            continue
        A, B = winner["A"], winner["B"]
        hx, hz, n = winner["hx"], winner["hz"], winner["n"]
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        assert css_ok(hx, hz)
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        assert max(row.bit_count() for row in hx+hz) <= W_MAX
        assert all(len(c) == len(set(c)) <= 5
                   for grid in (A,B) for row in grid for c in row)
        emitted.add(winner["fingerprint"])
        frozen = tuple(
            tuple(tuple(tuple(cell) for cell in row) for row in grid)
            for grid in (A,B))
        bank.append(dict(
            cells={"A": frozen[0], "B": frozen[1]},
            k=winner["k"], card=(geometry, mode),
            selection=merit(winner)))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return this card's fresh D15 design; never invoke the Q32 bank."""
    selected = _d30_bank(G)[idx]
    if selected is None:
        raise ValueError("D15 quotient-image card found no fresh target-rate code")
    return selected


def CELL(rng, G, side, i, j, ctx):
    """Return every selected D15 term without further sampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Saturating the row-weight cap alone excludes weight-five logicals; all five measured parent codes had w=12 and d30s=5, with sector-1 witnesses.
# REFUTED: The third-B-row cyclic C13 geometry with component cycle/six-word optimization exceeds the two-row cubic cell-0.350 record: its five measured leaders instead were [598,195,8], fitness 20.8700, below 23.0590.
# REFUTED: Normal-C8 rank control and witness-directed repairs alone lift the Q32 cubic bed above its distance-eight plateau: all five measured current-parent leaders were [544,200,8], with thousands of sector-one weight-eight witnesses and fitness 23.5290.
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

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
CONCEPT = ("AGL(1,5) x C6, order 120, A=B=(1,3): entirely cubic "
           "components sharing verified quotient dependencies should "
           "produce five new measured readings >=10 and improve "
           "cell 0.450 beyond fitness 28.5684. Readings are upper bounds.")
GROUP = ("Preserve the AGL(1,5) x cyclic-group construction rule; resize "
         "only the central factor C3 to C6. The factor-ten shape cannot "
         "use q=60 inside the size window; q=120 gives n=1200. The "
         "parent's order-60 bed repeatedly measured eight. Characteristic "
         "two changes the C6 quotient module, but scaffold ranks and "
         "genuine logical witnesses, not element orders, decide utility.")
SHAPES = ("Change to equal A=B=(1,3), with 360 variables and 120 checks "
          "per component. Factor ten gives n=1200 and k>=480, guaranteed "
          "rate 0.400; q_max=149. Target cell 0.450 through measured rank "
          "gain, enforcing 510<=k<570. Parent minima occupied sector-one "
          "Acol/Bcol blocks, especially Acol1/2 across Bcol0..2 and "
          "later-Bcol families. Remove its unequal seven-column component "
          "and all degree-two backbone columns.")
FILLING = ("All six cells contain three distinct terms: both row totals "
           "nine and column degrees three give HX=HZ weight twelve. "
           "Twenty cards cross five relative C5 commutator compositions "
           "with four central-coordinate prescriptions. Check generating "
           "quotient templates and quotient deficit eight using native "
           "scaffold blocks; vary affine translations without producing "
           "completed-code automorphism copies. Remove obvious component "
           "column-gauge duplicates, retain genuine kernel words, repair "
           "implicated cells and independently retest finalists. Select "
           "full pairings using original-span-verified localized logicals "
           "and conservative readings capped at twelve. Exact full ranks, "
           "CSS, weights and [0.425,0.475) remain gates. Neither component "
           "screens nor exhausted searches certify quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r045_AGL5xC6_13x13_cubic_quotient8_logicalpair_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 120

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("AGL(1,5) x C6 requires order 120")
    # Same direct-product rule; only the central cyclic order changes.
    powers = (1, 2, 4, 3)
    central_order = 6
    assert pow(2, 4, 5) == 1
    els = [(u,v,c) for c in range(central_order)
           for v in range(4) for u in range(5)]

    def red(g):
        return (int(g[0]) % 5, int(g[1]) % 4,
                int(g[2]) % central_order)

    def mul(g, h):
        u,v,c = red(g)
        w,t,d = red(h)
        return ((u+powers[v]*w) % 5, (v+t) % 4,
                (c+d) % central_order)

    def inv(g):
        u,v,c = red(g)
        vi = (-v) % 4
        return ((-powers[vi]*u) % 5, vi, (-c) % central_order)

    def idx(g):
        u,v,c = red(g)
        return 20*c+5*v+u

    return dict(elements=els, ident=(0,0,0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


# Five ternary relative-order spectra, not automorphism images:
# mode 0: (5,10,10); mode 1: (5,5,5);
# mode 2: (10,10,11); mode 3: (5,5,11);
# mode 4: (2,5,10).
# Cross these with the count of order-five B binomial ratios.
_ODD_V = (1, 3, 7, 9)
_EVEN_V = (2, 4, 6, 8)


def _affine_trinomial(rng, G, mode):
    """Anchor one term; preserve the prescribed relative-order spectrum."""
    if mode == 0:
        v, w = rng.sample(_ODD_V, 2)
        g, h = (rng.randrange(11), v), (rng.randrange(11), w)
    elif mode == 1:
        v, w = rng.sample(_EVEN_V, 2)
        g, h = (rng.randrange(11), v), (rng.randrange(11), w)
    elif mode in (2, 3):
        pool = _ODD_V if mode == 2 else _EVEN_V
        g = (rng.randrange(1, 11), 0)
        h = (rng.randrange(11), rng.choice(pool))
    else:
        v = rng.choice(_ODD_V)
        g = (rng.randrange(11), v)
        h = (rng.randrange(11), (v + 5) % 10)
    cell = [G.e, g, h]
    assert len({G._key(x) for x in cell}) == 3
    return cell


def _affine_side(rng, G, side, card):
    """Twenty deliberate order-spectrum cards with a full-action bridge."""
    mode, five = card % 5, card // 5
    if side == "A":
        # The first column's odd multipliers bridge the index-two subgroup.
        return [_affine_trinomial(rng, G, 0),
                _affine_trinomial(rng, G, mode),
                _affine_trinomial(rng, G, mode)]
    cells = []
    for j in range(3):
        pool = _EVEN_V if j < five else _ODD_V
        cells.append([G.e, (rng.randrange(11), rng.choice(pool))])
    cells.append(_affine_trinomial(rng, G, 0))
    return cells


def _verified_single_component(G, cells, side):
    """Use the scaffold lift, with an empty dummy side for extraction only."""
    if side == "A":
        hx, _, _ = lifted_matrices(G, [cells], [[[]]])
        rows = hx
    else:
        _, hz, _ = lifted_matrices(G, [[[]]], [cells])
        rows = hz
    width = len(cells) * G.q
    mask = (1 << width) - 1
    return [row & mask for row in rows]


def _component_kernel4(rows, width):
    """Exact exclusion of component kernel words of weights one through four."""
    columns = [0] * width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            columns[bit.bit_length() - 1] |= 1 << r
            source ^= bit
    singles = set(columns)
    if 0 in singles or len(singles) != width:
        return False
    pairs = set()
    for i, a in enumerate(columns):
        for b in columns[:i]:
            syndrome = a ^ b
            if syndrome in singles or syndrome in pairs:
                return False
            pairs.add(syndrome)
    return True


def _component_geometry(rows):
    """Connectivity and exact four/six cycles of a verified bipartite lift."""
    count = len(rows)
    common = [[0] * count for _ in range(count)]
    upper = [[] for _ in rows]
    adjacent = [0] * count
    four = 0
    for a in range(count):
        for b in range(a + 1, count):
            overlap = (rows[a] & rows[b]).bit_count()
            if overlap:
                common[a][b] = overlap
                upper[a].append(b)
                adjacent[a] |= 1 << b
                adjacent[b] |= 1 << a
                four += overlap * (overlap - 1) // 2

    reached = frontier = 1
    while frontier:
        neighbors = 0
        while frontier:
            bit = frontier & -frontier
            neighbors |= adjacent[bit.bit_length() - 1]
            frontier ^= bit
        frontier = neighbors & ~reached
        reached |= frontier
    if reached != (1 << count) - 1:
        return None

    six = 0
    for a, neighbors in enumerate(upper):
        for t, b in enumerate(neighbors):
            ab = common[a][b]
            for c in neighbors[t + 1:]:
                bc = common[b][c]
                if bc:
                    ac = common[a][c]
                    triple = (rows[a] & rows[b] & rows[c]).bit_count()
                    six += ab * ac * bc - triple * (ab + ac + bc - 2)
    return four, six


def _kernel_witness_reading(rows, width, rng, trials):
    """Randomized information-set search for genuine classical kernel words.

    A high reading means this search missed lighter words. These are
    component witnesses, not certified quantum logical distances.
    """
    best = width + 1
    for _ in range(trials):
        permutation = list(range(width))
        rng.shuffle(permutation)
        mapping = [1 << p for p in permutation]
        pivots = {}
        for source in rows:
            row = 0
            while source:
                bit = source & -source
                row ^= mapping[bit.bit_length() - 1]
                source ^= bit
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        ordered = sorted(pivots)
        for at, p in enumerate(ordered):
            bit = 1 << p
            reduced = pivots[p]
            for higher in ordered[at + 1:]:
                if pivots[higher] & bit:
                    pivots[higher] ^= reduced

        words = []
        for f in range(width):
            if f in pivots:
                continue
            bit = 1 << f
            word = bit
            for p in ordered:
                if pivots[p] & bit:
                    word |= 1 << p
            best = min(best, word.bit_count())
            words.append(word)
        words.sort(key=int.bit_count)
        shortlist = words[:24]
        for i, word in enumerate(shortlist):
            for other in shortlist[:i]:
                best = min(best, (word ^ other).bit_count())
        if best <= 5:
            break
    return best


def _screened_component_pool(rng, G, side, card):
    """Compare independent fillings of one prescribed order-spectrum card."""
    pool = []
    width = (3 if side == "A" else 4) * G.q
    for trial in range(96):
        cells = _affine_side(rng, G, side, card)
        rows = _verified_single_component(G, cells, side)
        geometry = _component_geometry(rows)
        if geometry is None:
            continue
        if not _component_kernel4(rows, width):
            continue
        # Full component rank avoids selecting rank-deficient fallbacks.
        if gf2_rank(rows) != G.q:
            continue
        pool.append((geometry, trial, cells, rows))
        if len(pool) >= 24:
            break
    if not pool:
        raise ValueError("no connected full-rank component passed kernel-4")

    pool.sort(key=lambda record: (record[0], record[1]))
    ranked = []
    for geometry, trial, cells, rows in pool[:6]:
        reading = _kernel_witness_reading(rows, width, rng, 8)
        key = (reading, -geometry[0], -geometry[1])
        ranked.append((key, trial, cells, rows, geometry))
    ranked.sort(key=lambda record: record[0], reverse=True)

    finalists = []
    for key, trial, cells, rows, geometry in ranked[:3]:
        reading = min(key[0], _kernel_witness_reading(
            rows, width, rng, 32))
        final_key = (reading, -geometry[0], -geometry[1])
        finalists.append((final_key, trial, cells))
    finalists.sort(key=lambda record: record[0], reverse=True)
    return finalists


def NEW_CTX(rng, G, idx):
    """Equal cubic components; every binary diagnostic uses native lifts."""
    q = G.q
    assert q == 120 and A_SHAPE == B_SHAPE == (1, 3)
    card, cohort = idx % 20, idx // 20
    composition, model = card % 5, card // 5
    seed = rng.randrange(1 << 62)
    mask = (1 << q)-1
    ident = G.idx(G.e)
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    blocks = {"A": [], "B": []}

    # Small batches cache authoritative monomial blocks, not another lift.
    for start in range(0, q, 16):
        ids = list(range(start, min(q, start+16)))
        gallery = [[[G.elements[g]] for g in ids]]
        xx, _, _ = lifted_matrices(G, gallery, [[[]]])
        _, zz, _ = lifted_matrices(G, [[[]]], gallery)
        for j in range(len(ids)):
            blocks["A"].append([(r >> (j*q)) & mask for r in xx])
            blocks["B"].append([(r >> (j*q)) & mask for r in zz])

    def bits(word):
        while word:
            low = word & -word
            yield low.bit_length()-1
            word ^= low

    def syndrome(cols, word):
        value = 0
        for v in bits(word):
            value ^= cols[v]
        return value

    def span(rows):
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

    def outside(word, pivots):
        while word:
            old = pivots.get(word.bit_length()-1)
            if old is None:
                return True
            word ^= old
        return False

    def closure(a, b):
        reached, pending = {ident}, [ident]
        for x in pending:
            for y in (a, b):
                image = table[x][y]
                if image not in reached:
                    reached.add(image)
                    pending.append(image)
        return len(reached)

    def cell_rows(cell, side):
        return [blocks[side][cell[0]][r] ^
                blocks[side][cell[1]][r] ^
                blocks[side][cell[2]][r] for r in range(q)]

    # Balanced C3 residues, with different C6 associations.
    central_pairs = ((1,2), (1,5), (4,5), (2,1))
    c, d = central_pairs[model]
    templates = []
    for v in range(1, 4):
        for t in range(1, 4):
            a, b = G.idx((0,v,c)), G.idx((0,t,d))
            if closure(a, b) != 24:
                continue
            # This subgroup is C4 x C6. Its regular action occurs five
            # times; divide the authoritative nullity by five exactly.
            nullity = q-gf2_rank(cell_rows((ident,a,b), "A"))
            if nullity == 5*8:
                templates.append((v,t,c,d))
    if not templates:
        raise ValueError("no quotient template has verified deficit eight")
    v, t, c, d = templates[(composition+cohort) % len(templates)]

    libraries = {z: [] for z in range(1, 5)}
    for u in range(5):
        for w in range(5):
            a, b = G.idx((u,v,c)), G.idx((w,t,d))
            comm = table[table[a][b]][inverse[table[b][a]]]
            z, mv, phase = G.elements[comm]
            assert mv == phase == 0
            if z and closure(a, b) == q:
                libraries[z].append((ident,a,b))
    assert all(libraries.values())
    aprofiles = ((1,1,1), (1,1,2), (1,2,3),
                 (1,2,4), (1,3,3))
    bprofiles = ((1,1,1), (1,1,2), (1,2,3), (1,2,4))
    plans = {"A": aprofiles[composition],
             "B": tuple(2*z % 5 for z in bprofiles[model])}

    def canonical(cell):
        # Recognize these particular column gauges only; no claim to
        # recognize every Tanner isomorphism. Never emit gauge images.
        return min(tuple(sorted(table[g][inverse[root]] for g in cell))
                   for root in cell)

    def draw(side):
        for _ in range(128):
            cells = [list(rng.choice(libraries[z])) for z in plans[side]]
            if len({canonical(cell) for cell in cells}) == 3:
                return cells
        raise ValueError("three distinct cubic operator classes unavailable")

    def inspect(cells, side):
        rows = [0]*q
        for j, cell in enumerate(cells):
            for r, value in enumerate(cell_rows(cell, side)):
                rows[r] |= value << (j*q)
        rank = gf2_rank(rows)
        if not q-12 <= rank <= q-8:
            return None
        cols = [0]*(3*q)
        for r, row in enumerate(rows):
            for x in bits(row):
                cols[x] |= 1 << r
        four = sum(
            (rows[a] & rows[b]).bit_count() *
            ((rows[a] & rows[b]).bit_count()-1)//2
            for a in range(q) for b in range(a))
        return dict(cells=[list(x) for x in cells], rows=rows, cols=cols,
                    rank=rank, four=four, words=set(), d=3*q+1)

    def short(cols):
        singles, pairs = {}, {}
        for x, value in enumerate(cols):
            word = 1 << x
            if not value:
                return word
            if value in singles:
                return word ^ singles[value]
            singles[value] = word
        for x, value in enumerate(cols):
            for y in range(x):
                value2 = value ^ cols[y]
                word = (1 << x) | (1 << y)
                old = singles.get(value2)
                if old is None:
                    old = pairs.get(value2)
                if old is not None:
                    return word ^ old
                pairs[value2] = word
        return None

    def challenge(rec, offset, trials):
        """Retained genuine kernel words; finite search gives no exclusion."""
        local = random.Random(seed+offset)
        cols = rec["cols"]
        best = short(cols)
        retained = sorted(rec["words"], key=int.bit_count)[:24]

        def observe(word):
            nonlocal best
            if word and (best is None or word.bit_count() < best.bit_count()):
                best = word

        for word in retained:
            assert syndrome(cols, word) == 0
            observe(word)
        order = list(range(len(cols)))
        for _ in range(trials):
            local.shuffle(order)
            pivots, basis = {}, []
            for x in order:
                value, word = cols[x], 1 << x
                while value:
                    p = value.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    observe(word)
            light = sorted(set(basis+retained), key=int.bit_count)[:16]
            grown = list(light)+retained
            for word in light:
                for other in basis:
                    proposed = word ^ other
                    observe(proposed)
                    if proposed and proposed.bit_count() <= 16:
                        grown.append(proposed)
            moves = basis+retained
            local.shuffle(moves)
            for initial in light[:5]:
                word = initial
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        proposed = word ^ other
                        if proposed and proposed.bit_count() < word.bit_count():
                            word, changed = proposed, True
                    observe(word)
                    if not changed:
                        break
                grown.append(word)
            retained = sorted(set(grown), key=int.bit_count)[:24]
            if best is not None and best.bit_count() <= 4:
                break
        assert best and syndrome(cols, best) == 0
        rec["words"] = set(retained+[best])
        rec["d"] = min(rec["d"], best.bit_count())
        rec["free4"] = short(cols) is None

    def pool(side):
        seen, bank = set(), []
        offset = 100003*int(side == "B")

        def remember(cells):
            sig = tuple(sorted(canonical(cell) for cell in cells))
            if sig in seen:
                return None
            seen.add(sig)
            rec = inspect(cells, side)
            if rec is not None:
                bank.append(rec)
            return rec

        for _ in range(160):
            remember(draw(side))
            if len(bank) >= 64:
                break
        bank.sort(key=lambda r: (r["four"], -r["rank"]))
        tested = []
        for at, rec in enumerate(bank[:14]):
            challenge(rec, offset+1009*at, 12)
            tested.append(rec)
        if not tested:
            raise ValueError("empty quotient-eight component portfolio")

        def merit(rec):
            return (rec["free4"], min(12,rec["d"]), rec["d"],
                    -rec["four"], rec["rank"])

        current = max(tested, key=merit)
        for step in range(16):
            cells = [list(x) for x in current["cells"]]
            words = sorted(current["words"], key=int.bit_count)[:6]
            occupied = sorted({x//q for x in bits(rng.choice(words))})
            j = rng.choice(occupied)
            cells[j] = list(rng.choice(libraries[plans[side][j]]))
            if len({canonical(cell) for cell in cells}) != 3:
                continue
            rec = remember(cells)
            if rec is None:
                continue
            rec["words"].update(
                word for word in current["words"]
                if syndrome(rec["cols"], word) == 0)
            challenge(rec, offset+300007+1013*step, 8)
            tested.append(rec)
            if merit(rec) >= merit(current):
                current = rec
        finalists = sorted(tested, key=merit, reverse=True)[:4]
        for at, rec in enumerate(finalists):
            challenge(rec, offset+700001+1009*at, 48)
            challenge(rec, offset+1700021+1013*at, 64)
        return sorted(finalists, key=merit, reverse=True)[:3]

    def logicals(a, b, hx, hz, n):
        """Commuting nontrivial representatives in original coordinates."""
        readings = []
        for kind, rec, opposite, stabilizers in (
                ("Z",a,hx,hz), ("X",b,hz,hx)):
            pivots = span(stabilizers)
            candidates = set()
            for word in sorted(rec["words"], key=int.bit_count)[:5]:
                for fixed in range(3):
                    actual = (sum(
                        ((word >> (j*q)) & mask) << ((3*j+fixed)*q)
                        for j in range(3)) if kind == "Z"
                        else word << (3*fixed*q))
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)
            reading = n+1
            for actual in sorted(candidates, key=int.bit_count)[:6]:
                for sweep in range(3):
                    before = actual.bit_count()
                    for row in stabilizers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                    nearby = sorted(
                        stabilizers, key=lambda r: (r & actual).bit_count(),
                        reverse=True)[:24]
                    chosen = actual
                    for i, first in enumerate(nearby):
                        for second in nearby[:i]:
                            proposed = actual ^ first ^ second
                            if proposed.bit_count() < chosen.bit_count():
                                chosen = proposed
                    actual = chosen
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            readings.append(reading)
        return tuple(readings)

    aa, bb = pool("A"), pool("B")
    proposals = []
    for a in aa:
        for b in bb:
            A = [[[G.elements[g] for g in cell] for cell in a["cells"]]]
            B = [[[G.elements[g] for g in cell] for cell in b["cells"]]]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (n == 1200 and 425*n <= 1000*k < 475*n):
                continue
            ar = [sum(((hx[r] >> (3*j*q)) & mask) << (j*q)
                      for j in range(3)) for r in range(q)]
            br = [row & ((1 << (3*q))-1) for row in hz[:q]]
            assert ar == a["rows"] and br == b["rows"]
            upper = logicals(a, b, hx, hz, n)
            detected = min(12, a["d"], b["d"], *upper)
            quality = (a["free4"] and b["free4"],
                       k*detected*detected, detected,
                       a["d"]+b["d"], -a["four"]-b["four"])
            proposals.append((quality,A,B,hx,hz,k,a,b,upper))
    proposals.sort(key=lambda item: item[0], reverse=True)
    for quality,A,B,hx,hz,k,a,b,upper in proposals:
        if not _component_kernel4(hx, 1200):
            continue
        if not _component_kernel4(hz, 1200):
            continue
        assert _grid_weights([[3,3,3]], [[3,3,3]]) == 12
        assert _no_dead_lines([[3,3,3]], [[3,3,3]])
        assert max(row.bit_count() for row in hx+hz) == 12
        assert css_ok(hx, hz)
        assert all(len({G.idx(g) for g in cell}) == 3
                   for grid in (A,B) for row in grid for cell in row)
        return dict(
            idx=idx, cells={"A":A,"B":B}, k=k,
            card=(composition,model,cohort), quotient_template=(v,t,c,d),
            verified_quotient_deficit=8, quality=quality,
            component_ranks=(a["rank"],b["rank"]),
            component_readings=(a["d"],b["d"]),
            verified_localized_logical_readings=upper,
            target_cell=0.450)
    raise ValueError("equal cubic pairings missed full rate/kernel-four gates")


def _retired_sharedmodule_ctx(rng, G, idx):
    """Inactive order-60 unequal-shape experiment; never called."""
    from itertools import combinations
    from math import exp

    q = G.q
    assert q == 60 and A_SHAPE == (1, 3) and B_SHAPE == (2, 7)
    mask = (1 << q)-1
    card, cohort = idx % 20, idx // 20
    composition, model = card % 5, card // 5
    diagnostic_seed = rng.randrange(1 << 62)
    ident = G.idx(G.e)
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]

    class Probe:
        # Cache group operations, not a separately implemented lift.
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = ident

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverses[g]

    probe = Probe()
    raw_a, _, _ = lifted_matrices(
        probe, [[[g] for g in range(q)]], [[[]]])
    _, raw_b, _ = lifted_matrices(
        probe, [[[]]], [[[g] for g in range(q)]])
    blocks = {
        "A": [[(row >> (g*q)) & mask for row in raw_a]
              for g in range(q)],
        "B": [[(row >> (g*q)) & mask for row in raw_b]
              for g in range(q)],
    }

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def syndrome(cols, word):
        value = 0
        for v in bits(word):
            value ^= cols[v]
        return value

    orders = []
    for g in range(q):
        value = ident
        for order in range(1, q+1):
            value = table[value][g]
            if value == ident:
                orders.append(order)
                break
        else:
            raise ValueError("invalid finite group")

    def generates(a, b):
        reached = {ident}
        todo = [ident]
        for x in todo:
            for y in (a, b):
                z = table[x][y]
                if z not in reached:
                    reached.add(z)
                    todo.append(z)
        return len(reached) == q

    # Full-generating normalized cubic spectra, not automorphism images.
    # The C3 factor can unwind affine order-four/five relations.
    spectra = ((6,12,12), (12,12,15), (4,12,15))
    libraries = [[] for _ in spectra]
    general = []
    for a in range(q):
        if a == ident:
            continue
        for b in range(a):
            if b == ident or not generates(a, b):
                continue
            triple = (ident, a, b)
            spectrum = tuple(sorted((
                orders[a], orders[b], orders[table[inverses[a]][b]])))
            general.append((spectrum, triple))
            if spectrum in spectra:
                libraries[spectra.index(spectrum)].append(triple)
    assert general
    # Enrichment is deliberate, but an empty preferred catalog is not
    # allowed to destroy every slot.
    for kind, target in enumerate(spectra):
        if not libraries[kind]:
            ranked = sorted(general, key=lambda item: sum(
                abs(x-y) for x, y in zip(item[0], target)))
            libraries[kind] = [item[1] for item in ranked[:24]]

    def left_kernel(rows):
        """Exact row dependencies in authoritative check coordinates."""
        pivots, basis = {}, []
        for r, source in enumerate(rows):
            word = 1 << r
            while source:
                p = source.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (source, word)
                    break
                source ^= old[0]
                word ^= old[1]
            if not source:
                basis.append(word)
        return basis

    # p -> p*h is a single-column gauge. Catalog one operator class,
    # rather than treating different anchor choices as different designs.
    operator_rows, memberships = {}, {}
    for spectrum, triple in general:
        support = tuple(sorted(triple))
        representative = min(
            tuple(sorted(table[g][inverses[root]] for g in support))
            for root in support)
        if support != representative:
            continue
        rows = [blocks["A"][support[0]][r] ^
                blocks["A"][support[1]][r] ^
                blocks["A"][support[2]][r] for r in range(q)]
        operator_rows[support] = rows
        basis = left_kernel(rows)
        if not basis or len(basis) > 14:
            continue
        # Capped proposal enumeration, not a kernel-exclusion claim.
        values = [0]
        for word in basis[:9]:
            values += [v ^ word for v in values]
        for word in set(values[1:]) | set(basis):
            quotient_only = all(
                ((word >> (5*t)) & 31) in (0, 31)
                for t in range(12))
            if not quotient_only:
                memberships.setdefault(word, []).append(support)

    shared_catalog = [
        (functional, sorted(set(cells)))
        for functional, cells in memberships.items()
        if len(set(cells)) >= 3]
    shared_catalog.sort(key=lambda item: (-len(item[1]), item[0]))
    desired_defect = (2, 4, 6, 8, 10)[composition]

    def joined_rows(signature):
        return [
            operator_rows[signature[0]][r] |
            (operator_rows[signature[1]][r] << q) |
            (operator_rows[signature[2]][r] << (2*q))
            for r in range(q)]

    shared_seeds, joins, joined_seen = [], [], set()
    if shared_catalog:
        turn = (11*card + 17*cohort) % len(shared_catalog)
        catalog = shared_catalog[turn:] + shared_catalog[:turn]
        for functional, members in catalog[:32]:
            for attempt in range(32):
                signature = tuple(sorted(rng.sample(members, 3)))
                if signature in joined_seen:
                    continue
                joined_seen.add(signature)
                rows = joined_rows(signature)
                defect = q-gf2_rank(rows)
                if 2 <= defect <= 10:
                    joins.append((abs(defect-desired_defect),
                                  -defect, signature))
        joins.sort()
        for _, negative_defect, signature in joins[:96]:
            rows = joined_rows(signature)
            if not _component_kernel4(rows, 3*q):
                continue
            module = left_kernel(rows)
            assert len(module) == -negative_defect
            shared_seeds.append((signature, module))
            if len(shared_seeds) == 8:
                break

    def module_options(module):
        """Operators annihilated by EVERY verified module basis vector."""
        answer = []
        for cell, rows in operator_rows.items():
            legal = True
            for functional in module:
                image = 0
                for r in bits(functional):
                    image ^= rows[r]
                if image:
                    legal = False
                    break
            if legal:
                answer.append(cell)
        return answer

    def record(grid, side):
        """GF2 superposition of monomial blocks from the verified helper."""
        nr, nc = len(grid), len(grid[0])
        rows = [0]*(nr*q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    action = blocks[side][g]
                    for t, value in enumerate(action):
                        rows[i*q+t] ^= value << (j*q)
        cols = [0]*(nc*q)
        for r, row in enumerate(rows):
            for v in bits(row):
                cols[v] |= 1 << r

        uses, four = {}, 0
        adjacency = [0]*len(rows)
        forced = 0
        for col in cols:
            checks = list(bits(col))
            degree = len(checks)
            forced += degree*(degree-1)*(degree-2)//6
            for r in checks:
                adjacency[r] |= col ^ (1 << r)
            for a, b in combinations(checks, 2):
                key = (a, b)
                four += uses.get(key, 0)
                uses[key] = uses.get(key, 0)+1
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in bits(neighbors & (-1 << (a+1))):
                triangles += (neighbors & adjacency[b]
                              & (-1 << (b+1))).bit_count()
        # Projection excess is only a cycle-selection heuristic.
        excess = max(0, triangles-forced)
        base = (500*(1+model)*four + (1+cohort % 3)*excess)
        return {"grid": copy_grid(grid), "rows": rows,
                "cols": cols, "base": base, "cycles": (four, excess)}

    def backbone_word(rec):
        """Exact shortest cycle word of the verified degree-two component."""
        cols = rec["cols"]
        adjacency = [[] for _ in rec["rows"]]
        for v, col in enumerate(cols):
            checks = list(bits(col))
            assert len(checks) == 2
            a, b = checks
            adjacency[a].append((b, v))
            adjacency[b].append((a, v))
        best = None
        for root in range(len(adjacency)):
            distance = [-1]*len(adjacency)
            paths = [0]*len(adjacency)
            parent_edge = [-1]*len(adjacency)
            distance[root] = 0
            todo = [root]
            for a in todo:
                for b, edge in adjacency[a]:
                    if distance[b] < 0:
                        distance[b] = distance[a]+1
                        paths[b] = paths[a] ^ (1 << edge)
                        parent_edge[b] = edge
                        todo.append(b)
                    elif edge != parent_edge[a] and edge != parent_edge[b]:
                        word = paths[a] ^ paths[b] ^ (1 << edge)
                        if word and (best is None or
                                     word.bit_count() < best.bit_count()):
                            best = word
        assert best is not None and syndrome(cols, best) == 0
        return best

    backbone_options = []
    for kind, catalog in enumerate(libraries):
        for triple in rng.sample(catalog, min(48, len(catalog))):
            _, a, b = triple
            cross = [[[ident], [ident], [ident]],
                     [[ident], [a], [b]]]
            rings = [[[ident, a], [], [ident]],
                     [[], [ident, b], [ident]]]
            # Both skeletons have row totals three and column degrees two.
            # Every score below is an actual kernel cycle word.
            for topology, grid in enumerate((cross, rings)):
                rec = record(grid, "B")
                word = backbone_word(rec)
                backbone_options.append(
                    (word.bit_count(), kind, topology, grid, word))
    backbone_options.sort(key=lambda item: item[0], reverse=True)
    best_girth = backbone_options[0][0]
    choices = [item for item in backbone_options
               if item[0] == best_girth]
    preferred_kind = (composition+model+cohort) % 3
    preferred_topology = (composition+model+cohort) % 2
    preferred = [item for item in choices
                 if item[1] == preferred_kind
                 and item[2] == preferred_topology]
    if not preferred:
        preferred = [item for item in choices
                     if item[2] == preferred_topology]
    (backbone_distance, backbone_kind, backbone_topology,
     skeleton, skeleton_word) = rng.choice(
         preferred if preferred else choices)

    profiles = (
        ((2,1), (2,1), (1,2), (1,2)),
        ((3,0), (0,3), (2,1), (1,2)),
        ((3,0), (3,0), (0,3), (0,3)),
        ((3,0), (1,2), (1,2), (1,2)),
    )
    plans = ((0,0,0), (1,1,1), (2,2,2), (0,1,2), (0,1,1))
    aplan = plans[composition]
    bcounts = [
        tuple(len(skeleton[i][j]) for i in range(2))
        for j in range(3)
    ] + list(profiles[model])
    # Distinct support cards also start with different binomial orders.
    desired_orders = ((12,), (15,), (6,12,15), (5,6,12,15))[model]
    long_elements = [g for g in range(q) if orders[g] in desired_orders]
    assert long_elements
    planned_b = [[bcounts[j][i] for j in range(7)] for i in range(2)]
    assert [sum(row) for row in planned_b] == [9, 9]
    assert _grid_weights([[3,3,3]], planned_b) == 12
    assert _no_dead_lines([[3,3,3]], planned_b)

    def draw_cell(count, kind, side):
        if not count:
            return []
        if count == 1:
            return [rng.randrange(q)]
        root = rng.randrange(q)
        if count == 2:
            return [root, table[root][rng.choice(long_elements)]]
        assert count == 3
        triple = rng.choice(libraries[kind])
        if side == "A":
            return list(triple)
        return [table[root][g] for g in triple]

    def term_variant(old, side):
        """Change one affine translation/central phase, not a cell gauge.

        The affine multiplier is preserved while the independent C3 phase
        may change. Actual full generation is checked after every move.
        Spectrum recipes enrich initialization, not an exclusion claim.
        """
        if len(old) != 3:
            return None
        proposal = list(old)
        t = rng.choice((1,2)) if side == "A" else rng.randrange(3)
        shift = (rng.randrange(5), 0, rng.randrange(3))
        proposal[t] = G.idx(G.mul(shift, G.elements[proposal[t]]))
        if proposal == old or len(set(proposal)) != 3:
            return None
        root_inv = inverses[proposal[0]]
        a = table[root_inv][proposal[1]]
        b = table[root_inv][proposal[2]]
        return proposal if generates(a, b) else None

    def fresh(side):
        if side == "A":
            return [[draw_cell(3, kind, side) for kind in aplan]]
        grid = copy_grid(skeleton)
        for j, counts in enumerate(profiles[model], 3):
            for i in range(2):
                kind = (j+i+model+cohort) % 3
                grid[i].append(draw_cell(counts[i], kind, side))
        return grid

    def small_word(cols):
        """Exhaustive kernel exclusion through four; return a real witness."""
        singles = {}
        for v, col in enumerate(cols):
            word = 1 << v
            if not col:
                return word
            if col in singles:
                return word ^ singles[col]
            singles[col] = word
        pairs = {}
        for a, col in enumerate(cols):
            for b in range(a):
                value = col ^ cols[b]
                word = (1 << a) | (1 << b)
                old = singles.get(value)
                if old is not None:
                    return word ^ old
                old = pairs.get(value)
                if old is not None:
                    return word ^ old
                pairs[value] = word
        return None

    def kernel_probe(rec, side, trials, seed):
        """Genuine classical kernel witnesses, including cancellation descent."""
        local = random.Random(seed)
        cols = rec["cols"]
        width = len(cols)
        order = list(range(width))
        best_word = small_word(cols)
        retained = set(rec.get("words", ()))
        if side == "B":
            assert syndrome(cols, skeleton_word) == 0
            if best_word is None or (
                    skeleton_word.bit_count() < best_word.bit_count()):
                best_word = skeleton_word

        def observe(word):
            nonlocal best_word
            if not word:
                return
            if best_word is None or word.bit_count() < best_word.bit_count():
                best_word = word
            if word.bit_count() <= 12:
                retained.add(word)
                if len(retained) > 96:
                    keep = sorted(retained,
                                  key=lambda w: (w.bit_count(), w))[:48]
                    retained.clear()
                    retained.update(keep)

        if best_word is not None:
            observe(best_word)
        previous = rec.get("word")
        if previous is not None:
            assert syndrome(cols, previous) == 0
            observe(previous)
        for word in list(retained):
            assert syndrome(cols, word) == 0
            observe(word)

        for _ in range(trials):
            local.shuffle(order)
            pivots, basis = {}, []
            for v in order:
                value, word = cols[v], 1 << v
                while value:
                    p = value.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    observe(word)
            light = sorted(basis, key=int.bit_count)[:16]
            for word in light:
                for other in basis:
                    observe(word ^ other)
            local.shuffle(basis)
            for initial in light[:4]:
                word = initial
                for sweep in range(2):
                    changed = False
                    for other in basis:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    observe(word)
                    if not changed:
                        break
            if best_word is not None and best_word.bit_count() <= 4:
                break
        # Budgeted parity completion through ten challenges short words
        # missed by random elimination. Only verified returned words count;
        # a budget stop never certifies absence.
        width = len(cols)
        cap = min(10, best_word.bit_count())
        if cap >= 5:
            singles, pairs = {}, {}
            incidence = [0]*len(rec["rows"])
            for v, col in enumerate(cols):
                singles.setdefault(col, 1 << v)
                for r in bits(col):
                    incidence[r] |= 1 << v
                for u in range(v):
                    pairs.setdefault(col ^ cols[u],
                                     (1 << v) | (1 << u))

            priorities = [local.random() for _ in cols]
            roots = list(range(0, width, q))
            local.shuffle(roots)
            all_variables = (1 << width)-1
            budget = min(18000, max(2000, 250*trials))

            for root in roots:
                remaining = [max(1, budget//len(roots))]

                def visit(value, word, allowed):
                    if remaining[0] <= 0:
                        return
                    remaining[0] -= 1
                    if not value:
                        observe(word)
                        return
                    left = min(cap, best_word.bit_count())-word.bit_count()
                    if left <= 0 or value.bit_count() > 3*left:
                        return
                    one = singles.get(value, 0)
                    if one and one & allowed == one:
                        observe(word | one)
                    pair = pairs.get(value, 0)
                    if left >= 2 and pair and pair & allowed == pair:
                        observe(word | pair)
                    left = min(cap, best_word.bit_count())-word.bit_count()
                    if left <= 2:
                        return

                    chosen, smallest = 0, width+1
                    for r in bits(value):
                        available = incidence[r] & allowed
                        count = available.bit_count()
                        if not count:
                            return
                        if count < smallest:
                            chosen, smallest = available, count
                    candidates = list(bits(chosen))
                    candidates.sort(key=lambda v: (
                        -(cols[v] & value).bit_count(), priorities[v]))
                    live = allowed
                    for v in candidates:
                        bit = 1 << v
                        live &= ~bit
                        visit(value ^ cols[v], word | bit, live)
                        if remaining[0] <= 0:
                            break

                # Classical regular symmetry supplies one anchor per
                # component column block; no quantum symmetry is assumed.
                allowed = all_variables & ~((1 << (root+1))-1)
                visit(cols[root], 1 << root, allowed)
                if best_word.bit_count() <= 4:
                    break

        assert best_word and syndrome(cols, best_word) == 0
        observe(best_word)
        rec["words"] = sorted(
            retained | {best_word}, key=lambda w: (w.bit_count(), w))[:32]
        assert all(syndrome(cols, word) == 0 for word in rec["words"])
        return best_word

    def component_pool(side):
        if side == "A" and shared_seeds:
            # Exact module joins compete with independent alternatives.
            # Module preservation is checked on binary matrices, not on
            # a representation-level prediction of their nullity.
            bank, signatures = [], set()

            def remember(grid, options=()):
                signature = tuple(sorted(
                    tuple(sorted(cell)) for cell in grid[0]))
                if signature in signatures:
                    return None
                signatures.add(signature)
                rec = record([[list(cell) for cell in signature]], "A")
                rec["rank"] = gf2_rank(rec["rows"])
                rec["defect"] = q-rec["rank"]
                if not 0 <= rec["defect"] <= 10:
                    return None
                rec["options"] = list(options)
                bank.append(rec)
                return rec

            for signature, module in shared_seeds[:6]:
                options = module_options(module)
                assert len(options) >= 3
                remember([[list(cell) for cell in signature]], options)
                for attempt in range(8):
                    remember([[list(cell) for cell in
                               rng.sample(options, 3)]], options)
            for attempt in range(20):
                remember(fresh("A"))

            bank.sort(key=lambda r: (
                abs(r["defect"]-desired_defect), r["base"]))
            candidates = []
            for rec in bank:
                rec["free4"] = small_word(rec["cols"]) is None
                if not rec["free4"]:
                    continue
                rec["word"] = kernel_probe(
                    rec, "A", 12, diagnostic_seed+1009*len(candidates))
                rec["reading"] = rec["word"].bit_count()
                candidates.append(rec)
                if len(candidates) == 12:
                    break
            if not candidates:
                # Failure of this catalog is not evidence of exclusion;
                # the ordinary fresh-cell portfolio below remains legal.
                pass
            else:
                def merit(rec):
                    return (min(8, rec["reading"]), rec["defect"],
                            rec["reading"], -rec["base"])

                current = max(candidates, key=merit)
                for step in range(20):
                    grid = copy_grid(current["grid"])
                    words = current.get("words", [current["word"]])[:6]
                    implicated = sorted({
                        v//q for v in bits(rng.choice(words))})
                    j = rng.choice(implicated)
                    options = current["options"]
                    if options:
                        alternatives = [
                            cell for cell in options
                            if tuple(sorted(cell)) !=
                               tuple(sorted(grid[0][j]))]
                        if not alternatives:
                            continue
                        grid[0][j] = list(rng.choice(alternatives))
                    else:
                        grid[0][j] = draw_cell(3, aplan[j], "A")
                    proposed = remember(grid, options)
                    if proposed is None:
                        continue
                    proposed["free4"] = small_word(
                        proposed["cols"]) is None
                    if not proposed["free4"]:
                        continue
                    proposed["words"] = [
                        word for word in current.get("words", ())
                        if syndrome(proposed["cols"], word) == 0]
                    proposed["word"] = kernel_probe(
                        proposed, "A", 8,
                        diagnostic_seed+300007+1013*step)
                    proposed["reading"] = proposed["word"].bit_count()
                    candidates.append(proposed)
                    if merit(proposed) >= merit(current):
                        current = proposed

                candidates.sort(key=merit, reverse=True)
                finalists = candidates[:4]
                for at, rec in enumerate(finalists):
                    word = kernel_probe(
                        rec, "A", 64, diagnostic_seed+700001+at)
                    if word.bit_count() < rec["reading"]:
                        rec["word"], rec["reading"] = word, word.bit_count()
                finalists.sort(key=merit, reverse=True)
                for at, rec in enumerate(finalists[:3]):
                    word = kernel_probe(
                        rec, "A", 96, diagnostic_seed+1700009+at)
                    if word.bit_count() < rec["reading"]:
                        rec["word"], rec["reading"] = word, word.bit_count()
                for rec in finalists:
                    rec["quality"] = (
                        rec["free4"], rec["reading"],
                        -rec["base"], rec["rank"])
                finalists.sort(key=merit, reverse=True)
                return finalists[:3]

        known, screened, signatures = [], [], set()
        mutable = [(0, j) for j in range(3)] if side == "A" else [
            (i, j) for i in range(2) for j in range(3, 7)
            if bcounts[j][i]]

        def loss(rec):
            penalty = 0
            for word in known:
                if syndrome(rec["cols"], word) == 0:
                    penalty += 5000*max(1, 11-word.bit_count())**2
            return rec["base"]+penalty

        def optimize(grid, steps):
            current = record(grid, side)
            best = current
            current_cost = best_cost = loss(current)
            for step in range(steps):
                positions = mutable
                active = [word for word in known
                          if syndrome(current["cols"], word) == 0]
                if active and step % 3 != 2:
                    nc = len(grid[0])
                    occupied = {v//q for v in bits(rng.choice(active))}
                    targeted = [p for p in mutable if p[1] in occupied
                                and p[1] < nc]
                    if targeted:
                        positions = targeted
                i, j = rng.choice(positions)
                old = list(grid[i][j])
                count = len(old)
                kind = aplan[j] if side == "A" else (
                    j+i+model+cohort) % 3
                proposal = (term_variant(old, side)
                            if step % 3 != 0 else None)
                grid[i][j] = (proposal if proposal is not None
                              else draw_cell(count, kind, side))
                trial = record(grid, side)
                trial_cost = loss(trial)
                temperature = 60*(1-step/steps)**2+0.2
                accept = trial_cost <= current_cost or rng.random() < exp(
                    min(0.0, (current_cost-trial_cost)/temperature))
                if accept:
                    current, current_cost = trial, trial_cost
                    if trial_cost < best_cost:
                        best, best_cost = trial, trial_cost
                else:
                    grid[i][j] = old
            return best

        pool = [optimize(fresh(side), 80) for _ in range(4)]
        side_offset = 100003*int(side == "B")
        for stage in range(3):
            pool.sort(key=loss)
            tested = []
            for rec in pool:
                signature = tuple(tuple(sorted(cell))
                                  for row in rec["grid"] for cell in row)
                if signature in signatures:
                    continue
                signatures.add(signature)
                short = small_word(rec["cols"])
                rec["free4"] = short is None
                rec["rank"] = gf2_rank(rec["rows"])
                rec["word"] = kernel_probe(
                    rec, side, 12,
                    diagnostic_seed+side_offset+1009*stage)
                rec["reading"] = rec["word"].bit_count()
                tested.append(rec)
                screened.append(rec)
                # Repair several actual families. Immutable backbone
                # cycles are retained as readings, not repair objectives.
                for word in rec.get("words", [rec["word"]])[:8]:
                    movable = side == "A" or any(
                        v//q >= 3 for v in bits(word))
                    if (movable and word.bit_count() <= 10
                            and word not in known and len(known) < 64):
                        known.append(word)
            tested.sort(key=lambda r: (
                r["free4"], r["reading"], -loss(r), r["rank"]),
                reverse=True)
            if stage < 2:
                seeds = tested[:2] if tested else pool[:2]
                pool = [optimize(copy_grid(r["grid"]), 112) for r in seeds]
                pool += [optimize(fresh(side), 64) for _ in range(2)]
        if not screened:
            raise ValueError("empty affine-product component portfolio")
        screened.sort(key=lambda r: (
            r["free4"], r["reading"], -loss(r), r["rank"]), reverse=True)
        finalists = screened[:3]
        for rec in finalists:
            word = kernel_probe(
                rec, side, 64, diagnostic_seed+side_offset+700001)
            if word.bit_count() < rec["reading"]:
                rec["word"], rec["reading"] = word, word.bit_count()
            rec["quality"] = (
                rec["free4"], rec["reading"], -loss(rec), rec["rank"])
        finalists.sort(key=lambda r: r["quality"], reverse=True)
        return finalists

    aa, bb = component_pool("A"), component_pool("B")
    proposals = [(a, b) for a in aa for b in bb]
    proposals.sort(key=lambda ab: (
        ab[0]["free4"] and ab[1]["free4"],
        min(ab[0]["reading"], ab[1]["reading"]),
        ab[0]["reading"]+ab[1]["reading"],
        ab[0]["quality"][2]+ab[1]["quality"][2]), reverse=True)

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                lead = word.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = word
                    break
                word ^= old
        return pivots

    def outside(word, pivots):
        while word:
            old = pivots.get(word.bit_length()-1)
            if old is None:
                return True
            word ^= old
        return False

    def localized_readings(a, b, hx, hz, n):
        """Actual logical upper readings in the ORIGINAL stabilizer quotient."""
        readings = []
        for kind, rec, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            words = sorted(
                set(rec.get("words", ())) | {rec["word"]},
                key=lambda w: (w.bit_count(), w))[:4]
            candidates = set()
            for word in words:
                assert syndrome(rec["cols"], word) == 0
                for fixed in range(7 if kind == "Z" else 3):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((7*j+fixed)*q) for j in range(3))
                    else:
                        actual = word << (7*fixed*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)

            reading = n+1
            diagnostic = random.Random(
                diagnostic_seed+(4100011 if kind == "Z" else 5100011))
            reducers = list(stabilizers)
            diagnostic.shuffle(reducers)
            for actual in sorted(
                    candidates, key=lambda w: (w.bit_count(), w))[:6]:
                for sweep in range(3):
                    before = actual.bit_count()
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                    nearby = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:32]
                    best_word, best_weight = actual, actual.bit_count()
                    for i, first in enumerate(nearby):
                        partial = actual ^ first
                        for second in nearby[:i]:
                            proposed = partial ^ second
                            weight = proposed.bit_count()
                            if weight < best_weight:
                                best_word, best_weight = proposed, weight
                    actual = best_word
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            readings.append(reading)
        return tuple(readings)

    best_answer, best_key = None, None
    for a, b in proposals:
        A = [[[G.elements[g] for g in cell] for cell in row]
             for row in a["grid"]]
        B = [[[G.elements[g] for g in cell] for cell in row]
             for row in b["grid"]]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert sa == [[3,3,3]]
        assert sb == planned_b
        assert all(sum(row) == 9 for row in sb)
        assert sorted(sum(sb[i][j] for i in range(2))
                      for j in range(7)) == [2,2,2,3,3,3,3]
        assert _grid_weights(sa, sb) == 12
        assert _no_dead_lines(sa, sb)
        assert all(len({G.idx(g) for g in cell}) == len(cell)
                   for grid in (A, B) for row in grid for cell in row)

        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        assert n == 1380 and k >= 600
        if not (425*n <= 1000*k < 475*n):
            continue
        assert max(row.bit_count() for row in hx+hz) <= 12
        assert css_ok(hx, hz)

        # Compare cached components with this complete authoritative lift.
        authoritative_a = [
            sum(((hx[t] >> (j*7*q)) & mask) << (j*q)
                for j in range(3)) for t in range(q)]
        authoritative_b = [
            row & ((1 << (7*q))-1) for row in hz[:2*q]]
        assert authoritative_a == a["rows"]
        assert authoritative_b == b["rows"]
        assert b["reading"] <= backbone_distance

        # Rank gain competes only after actual logical representatives
        # have been challenged. The cap limits optimistic selection;
        # it is not a lower bound on quantum distance.
        logicals = localized_readings(a, b, hx, hz, n)
        detected = min(8, a["reading"], b["reading"], *logicals)
        quality = (
            a["free4"] and b["free4"],
            k*detected*detected, detected,
            a["reading"]+b["reading"],
            -a["base"]-b["base"],
        )
        if best_key is None or quality > best_key:
            best_key = quality
            best_answer = {
                "idx": idx, "cells": {"A": A, "B": B}, "k": k,
                "card": (composition, model, cohort),
                "component_readings": (a["reading"], b["reading"]),
                "component_kernel4_free": (a["free4"], b["free4"]),
                "component_ranks": (a["rank"], b["rank"]),
                "backbone_cycle_weight": backbone_distance,
                "backbone_spectrum": spectra[backbone_kind],
                "backbone_topology": (
                    "cross-matchings" if backbone_topology == 0
                    else "two-rings-and-bridge"),
                "backbone_counts": bcounts[:3],
                "repair_model": "complete shared modules and retained witnesses",
                "verified_localized_logical_readings": logicals,
                "requested_A_defect": desired_defect,
                "actual_A_defect": q-a["rank"],
                "shared_module_seed_dimensions": tuple(
                    len(module) for signature, module in shared_seeds),
                "pairing_quality": quality,
            }
    if best_answer is not None:
        return best_answer
    raise ValueError("affine-product pairings missed the exact 0.450 rate gate")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected matching, cubic or deliberately zero cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Cap-saturating rolled supports on Z26:Z2, (2,6)x(2,4), did not suppress small logicals: all five measured codes had d=4, with abundant sector-1 witnesses.
# REFUTED: Degree>=3 columns and local syndrome-collision selection on Z17:Z2, (2,5)x(3,7), did not eliminate weight-10 sector-1 logical families: all five measured codes returned d=10 across many Acol/Bcol blocks.
# REFUTED: AGL(1,11) all-trinomial A and order-spectrum-screened B binomials/trinomials suppressed the sector-1 weight-six bottleneck; all five measured codes instead returned six, with thousands of X witnesses and k=660.
# REFUTED: Kernel-five exclusion alone removes the affine weight-six bottleneck; the all-even parent still measured five distance-six codes with thousands of sector-1 X and Z witnesses.
# REFUTED: Independent A5 spectrum/support optimization prevents slot equivalences; the measured program still collapsed slots 21, 23, 41 and 43 to earlier isomorphic codes.
# REFUTED: Longer-order C7:C9 matching spectra and individual-term repair produce five readings >=10; all five measured selections instead returned eight at k=640, with abundant sector-1 X and Z witnesses.
# REFUTED: Competing AGL(1,5)xC3 ring/matching backbones and individual-term repair produce five readings >=10; all five measured winners instead returned eight at k=610, with abundant sector-1 X and Z witnesses.
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

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
CONCEPT = ("Q8 x S3, order 48, shapes (1,2)x(2,5): C3-invariant and C3-even physical-witness screening, crossed with four/four and three/four A allocations, should produce five fresh measured readings >=14 and exceed fitness 44.607 in cell 0.250; measured searches can refute this.")
GROUP = ("Frozen byte-for-byte under L1. This exact bed scored 38.7, with all five measured codes retaining distance readings twelve; related order-48 beds reached 43.5. The normal C3 and central C2 supply tractable restricted word spaces. Every matrix action, syndrome, assembled rank and stabilizer test remains scaffold-derived; no group-distance advantage is certified.")
SHAPES = ("Frozen (1,2)x(2,5): n=576, kb=144, guaranteed rate 1/4, q=48 against the admissible maximum 49, and unequal component variables/checks 96/48 and 240/96. Every measured parent minimum occupied sector-one Acol/Bcol blocks, frequently both A columns and Bcol1; none occupied Arow/Brow. Exact complete ranks retain 0.225<=k/n<0.275.")
FILLING = ("Five legal count architectures cross four intrinsic quotient/order policies. Three use four/four A; two use three/four A and allow degree-five B columns. Both B rows total eight and both CSS row caps are twelve. Prescribe affine quotient occupancies, mix order-4/6/12 doubletons including S3-sign-crossing pairs, and pack verified B incidences. Screen normal-fiber kernels, search ordinary/C3-even/invariant physical coordinates, retain translation-distinct directions, and repair witness-heavy cells. Equal deeper finalist budgets include exact logical-four exclusion, stabilizer-tested embeddings and both full matrices. Unequal uncoloured Tanner invariants exclude repeated outputs; restricted and sampled readings never certify distance.")
FAMILY_NAME = "r025_Q8xS3_12x25_quartic_order12_witnessrepair_v1_C3even_mixedorder_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    # (a,b,c,d) = x^a y^b r^c t^d in Q8 x S3.
    # x^4=1, y^2=x^2, y*x*y^-1=x^-1;
    # r^3=t^2=1, t*r*t=r^-1. The factors commute.
    if not valid_q(q):
        raise ValueError("Q8 x S3 requires order 48")
    els = [(a, b, c, d) for d in range(2) for c in range(3)
           for b in range(2) for a in range(4)]

    def red(g):
        return (g[0] % 4, g[1] % 2, g[2] % 3, g[3] % 2)

    def mul(g, h):
        a, b, c, d = red(g)
        u, v, w, z = red(h)
        return ((a + (u if b == 0 else -u) + 2*b*v) % 4,
                (b + v) % 2,
                (c + (w if d == 0 else -w)) % 3,
                (d + z) % 2)

    def inv(g):
        a, b, c, d = red(g)
        qa = (-a) % 4 if b == 0 else (a + 2) % 4
        return (qa, b, (-c if d == 0 else c) % 3, d)

    def idx(g):
        a, b, c, d = red(g)
        return a + 4*b + 8*c + 24*d

    return dict(elements=els, ident=(0, 0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


from functools import lru_cache
from collections import Counter


def _columns(rows, width):
    """Transpose only scaffold-generated matrices."""
    out = [0] * width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            out[bit.bit_length()-1] |= 1 << r
            source ^= bit
    return tuple(out)


def _basis(rows):
    result = {}
    for word in rows:
        while word:
            p = word.bit_length()-1
            old = result.get(p)
            if old is None:
                result[p] = word
                break
            word ^= old
    return result


def _outside(word, span):
    while word:
        p = word.bit_length()-1
        old = span.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _clean_component(cols):
    """Exactly exclude component kernel words through four."""
    seen = {0}
    for value in cols:
        if value in seen:
            return False
        seen.add(value)
    for i, value in enumerate(cols):
        for other in cols[:i]:
            pair = value ^ other
            if pair in seen:
                return False
            seen.add(pair)
    return True


def _no_logical_le4(checks, stabilizers, n):
    """Exact short-logical exclusion, allowing small stabilizers.

    Equal syndromes of supports of size <=2 cover every commuting
    support through four. Compare their exact quotient residues.
    """
    span = _basis(stabilizers)
    residues = []
    for c in range(n):
        row = span.get(c)
        if row is None:
            residues.append(1 << c)
        else:
            tail, residue = row ^ (1 << c), 0
            while tail:
                bit = tail & -tail
                residue ^= residues[bit.bit_length()-1]
                tail ^= bit
            residues.append(residue)
    cols = _columns(checks, n)
    seen = {0: 0}
    for value, residue in zip(cols, residues):
        old = seen.get(value)
        if old is None:
            seen[value] = residue
        elif old != residue:
            return False
    for i, value in enumerate(cols):
        ri = residues[i]
        for j in range(i):
            key = value ^ cols[j]
            residue = ri ^ residues[j]
            old = seen.get(key)
            if old is None:
                seen[key] = residue
            elif old != residue:
                return False
    return True


def _probe(cols, seeds, rng, trials):
    """Actual nonzero classical kernel witnesses; high misses certify nothing."""
    width = len(cols)
    memory = sorted(set(seeds), key=lambda w: (w.bit_count(), w))[:24]
    best = min((w.bit_count() for w in memory), default=width+1)
    order = list(range(width))
    for trial in range(trials):
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
            if not syndrome and word:
                words.append(word)
                best = min(best, word.bit_count())
        for _ in range(2):
            donors = sorted(set(words+memory), key=int.bit_count)[:24]
            rng.shuffle(donors)
            reduced = []
            for word in words:
                weight = word.bit_count()
                for donor in donors:
                    proposal = word ^ donor
                    pw = proposal.bit_count()
                    if 0 < pw < weight:
                        word, weight = proposal, pw
                reduced.append(word)
                best = min(best, weight)
            words = reduced
        short = sorted(set(words+memory),
                       key=lambda w: (w.bit_count(), w))[:32]
        discoveries = set(short)
        for i, word in enumerate(short):
            for other in short[:i]:
                proposal = word ^ other
                if proposal:
                    best = min(best, proposal.bit_count())
                    discoveries.add(proposal)
        if trials >= 12 and trial % 4 == 0:
            small = short[:8]
            for i, word in enumerate(small):
                for j in range(i):
                    pair = word ^ small[j]
                    for h in range(j):
                        proposal = pair ^ small[h]
                        if proposal:
                            best = min(best, proposal.bit_count())
                            discoveries.add(proposal)
        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:24]
        if best <= 6:
            break
    assert memory and best <= width
    seeds[:] = memory
    return best


def _cycle_count(cols):
    counts, total = {}, 0
    for source in cols:
        neighbors = []
        while source:
            bit = source & -source
            neighbors.append(bit.bit_length()-1)
            source ^= bit
        for i, a in enumerate(neighbors):
            for b in neighbors[:i]:
                key = (b, a)
                old = counts.get(key, 0)
                total += old
                counts[key] = old + 1
    return total


@lru_cache(maxsize=1)
def _atlas(G):
    """Compile both actions using verified single-term full LP lifts."""
    q, mask = G.q, (1 << G.q)-1
    stub_a = [[[G.e], [G.e]]]
    stub_b = [[[G.e] for _ in range(5)] for _ in range(2)]
    result = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            if side == "A":
                unit = [[[g], []]]
                hx, hz, _ = lifted_matrices(G, unit, stub_b)
                source = hx
            else:
                unit = [[[g], [], [], [], []],
                        [[], [], [], [], []]]
                hx, hz, _ = lifted_matrices(G, stub_a, unit)
                source = hz
            blocks.append(_columns([r & mask for r in source[:q]], q))
        result[side] = tuple(blocks)
    return result


def _component_columns(G, atlas, side, grid):
    """Linear superposition of verified unit blocks, not another lift."""
    q = G.q
    width = (2 if side == "A" else 5)*q
    cols = [0]*width
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            for g in cell:
                contribution = atlas[side][G.idx(g)]
                for h, value in enumerate(contribution):
                    cols[j*q+h] ^= value << (i*q)
    return tuple(cols)


def _support_cards():
    """Canonical cardinality geometries, including deliberate zero cells."""
    from itertools import combinations_with_replacement
    types = [(a, b) for a in range(5) for b in range(5)
             if 2 <= a+b <= 4]
    candidates = []
    for columns in combinations_with_replacement(types, 5):
        if sum(a for a, b in columns) != 8:
            continue
        if sum(b for a, b in columns) != 8:
            continue
        swapped = tuple(sorted((b, a) for a, b in columns))
        if columns > swapped:
            continue
        parity = tuple(sum((columns[j][i] & 1) << j
                           for j in range(5)) for i in range(2))
        # Full augmentation rank on B helps keep rank surplus in band.
        if not parity[0] or not parity[1] or parity[0] == parity[1]:
            continue
        grid = tuple(tuple(c[i] for c in columns) for i in range(2))
        quality = (min(a+b for a, b in columns),
                   sum(bool(a) and bool(b) for a, b in columns),
                   -sum((a-b)**2 for a, b in columns))
        candidates.append((quality, grid))
    candidates.sort(reverse=True)
    if len(candidates) < 5:
        raise ValueError("insufficient B support architectures")
    # Spread the five cards over occupied/zero and split-count geometries.
    positions = [i*(len(candidates)-1)//4 for i in range(5)]
    return tuple(candidates[i][1] for i in positions)


def _quotient(g):
    # Q8/centre and the S3 sign form a genuine C2^3 quotient.
    return (g[0] & 1) | (g[1] << 1) | (g[3] << 2)


def _draw_grid(local, G, counts, mode):
    central = (2, 0, 0, 0)
    order12 = [g for g in G.elements
               if g[2] != 0 and g[3] == 0
               and (g[1] == 1 or g[0] & 1)]
    grid = []
    for i, row in enumerate(counts):
        output = []
        for j, count in enumerate(row):
            if not count:
                output.append([])
                continue
            if count == 1:
                output.append([G.pick(local)])
                continue
            if count == 2:
                anchor = G.pick(local)
                relative = local.choice(order12)
                output.append([anchor, G.mul(anchor, relative)])
                continue
            for retry in range(128):
                cell = local.sample(G.elements, count)
                if any(G.mul(G.inv(g), h) == central
                       for t, g in enumerate(cell) for h in cell[:t]):
                    continue
                quotients = [_quotient(g) for g in cell]
                if mode == 1 and len(set(quotients)) != count:
                    continue
                if mode == 2 and len({g[3] for g in cell}) != 2:
                    continue
                if mode == 3 and len({g[2] for g in cell}) < 2:
                    continue
                relatives = [G.mul(G.inv(cell[0]), g) for g in cell[1:]]
                if not any(G.mul(g, h) != G.mul(h, g)
                           for t, g in enumerate(relatives)
                           for h in relatives[:t]):
                    continue
                output.append(cell)
                break
            else:
                raise ValueError("noncommuting cell policy exhausted")
        grid.append(output)
    return grid


def _component_pool(local, G, atlas, side, counts, mode, batch):
    pool, seen = [], set()

    def merit(item):
        return (item["d"], -item["cycles"], item["rank"])

    def admit(grid):
        labels = tuple(tuple(sorted(cell)) for row in grid for cell in row)
        if labels in seen:
            return
        seen.add(labels)
        cols = _component_columns(G, atlas, side, grid)
        if not _clean_component(cols):
            return
        rank = gf2_rank(cols)
        # Necessary bounds, not substitutions for the full-code rate gate.
        if side == "A" and rank < G.q-4:
            return
        if side == "B" and rank < 2*G.q-2:
            return
        seeds = []
        item = dict(grid=grid, cols=cols, rank=rank, seeds=seeds,
                    cycles=_cycle_count(cols))
        item["d"] = _probe(cols, seeds, local, 2)
        pool.append(item)
        pool.sort(key=merit, reverse=True)
        del pool[6:]

    for attempt in range(72+32*batch):
        admit(_draw_grid(local, G, counts, mode))
    # Repair actual voltages while retaining the prescribed count geometry.
    for step in range(24):
        if not pool:
            break
        parent = local.choice(pool[:min(3, len(pool))])
        grid = [[list(cell) for cell in row] for row in parent["grid"]]
        locations = [(i, j) for i, row in enumerate(counts)
                     for j, count in enumerate(row) if count]
        i, j = local.choice(locations)
        replacement = _draw_grid(local, G, counts, (mode+step) % 4)
        grid[i][j] = replacement[i][j]
        admit(grid)
    for item in pool:
        item["d"] = min(item["d"],
                         _probe(item["cols"], item["seeds"], local, 16))
    pool.sort(key=merit, reverse=True)
    pool = pool[:3]
    for item in pool:
        item["d"] = min(item["d"],
                         _probe(item["cols"], item["seeds"], local, 48))
    pool.sort(key=merit, reverse=True)
    return pool


def _dressed_reading(item, q):
    """Found quantum logicals, verified before stabilizer dressing."""
    hx, hz = item["hx"], item["hz"]
    mask, best = (1 << q)-1, item["n"]+1
    for side, seeds, checks, stabilizers in (
            ("A", item["aw"], hx, hz),
            ("B", item["bw"], hz, hx)):
        span = _basis(stabilizers)
        fixed_range = range(5) if side == "A" else range(2)
        for seed in sorted(set(seeds), key=int.bit_count)[:6]:
            for fixed in fixed_range:
                if side == "A":
                    word = sum(((seed >> (j*q)) & mask)
                               << ((j*5+fixed)*q) for j in range(2))
                else:
                    word = seed << (fixed*5*q)
                if any((word & row).bit_count() & 1 for row in checks):
                    continue
                if not _outside(word, span):
                    continue
                best = min(best, word.bit_count())
                for sweep in range(3):
                    weight, chosen = word.bit_count(), word
                    nearby = sorted(
                        ((word ^ row).bit_count(), row)
                        for row in stabilizers)[:16]
                    for _, row in nearby:
                        proposal = word ^ row
                        if 0 < proposal.bit_count() < weight:
                            chosen, weight = proposal, proposal.bit_count()
                    for i, (_, row) in enumerate(nearby):
                        for _, other in nearby[:i]:
                            proposal = word ^ row ^ other
                            if 0 < proposal.bit_count() < weight:
                                chosen, weight = proposal, proposal.bit_count()
                    if chosen == word:
                        break
                    word = chosen
                    best = min(best, weight)
    return best


def _fingerprint(hx, hz, n):
    """Uncoloured Tanner invariant: unequal values certify nonisomorphism."""
    rows = list(hx)+list(hz)
    weights = [r.bit_count() for r in rows]
    overlap = [dict() for _ in rows]
    adjacency = [0]*len(rows)
    for i, row in enumerate(rows):
        for j in range(i):
            size = (row & rows[j]).bit_count()
            if size:
                overlap[i][j] = overlap[j][i] = size
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    profiles = []
    for i, neighbors in enumerate(overlap):
        profile = Counter(
            (weights[j], size, (adjacency[i] & adjacency[j]).bit_count())
            for j, size in neighbors.items())
        profiles.append((weights[i], tuple(sorted(profile.items()))))
    incidents = [[] for _ in range(n)]
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            incidents[bit.bit_length()-1].append(weights[r])
            source ^= bit
    return (n, tuple(sorted(Counter(profiles).items())),
            tuple(sorted(Counter(tuple(sorted(v))
                                 for v in incidents).items())))


@lru_cache(maxsize=1)
def _design_bank(G):
    """Pure deterministic policy search; no duplicate or archive fallback."""
    assert G.q == 48
    q, mask = G.q, (1 << G.q)-1
    atlas, cards = _atlas(G), _support_cards()
    bank, emitted = [], set()

    for slot in range(N_SLOTS):
        architecture, mode = slot % 5, (slot // 5) % 4
        ta, tb = ((4, 4),), cards[architecture]
        assert _grid_weights(ta, tb) == 12
        assert _no_dead_lines(ta, tb)
        local = random.Random(BASE_SEED*100003 + 48025071 + 104729*slot)
        chosen = None
        for batch in range(3):
            ap = _component_pool(local, G, atlas, "A", ta, mode, batch)
            bp = _component_pool(local, G, atlas, "B", tb,
                                 (mode+architecture) % 4, batch)
            proposals = []
            for a in ap:
                for b in bp:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (225*n <= 1000*k < 275*n):
                        continue
                    # Verify the compiled component actions against full lifts.
                    ar = [sum(((row >> (5*j*q)) & mask) << (j*q)
                              for j in range(2)) for row in hx[:q]]
                    br = [row & ((1 << (5*q))-1)
                          for row in hz[:2*q]]
                    assert _columns(ar, 2*q) == a["cols"]
                    assert _columns(br, 5*q) == b["cols"]
                    item = dict(A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                                aw=list(a["seeds"]), bw=list(b["seeds"]),
                                da=a["d"], db=b["d"],
                                cycles=a["cycles"]+b["cycles"])
                    d = min(item["da"], item["db"])
                    proposals.append((k*d*d/n, item))
            proposals.sort(key=lambda x: x[0], reverse=True)

            finalists = []
            for preliminary, item in proposals[:6]:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                fp = _fingerprint(hx, hz, n)
                if fp in emitted:
                    continue
                if not _no_logical_le4(hx, hz, n):
                    continue
                if not _no_logical_le4(hz, hx, n):
                    continue
                reading = _dressed_reading(item, q)
                d = min(item["da"], item["db"], reading)
                key = (item["k"]*d*d/n, d, -item["cycles"])
                finalists.append((key, fp, item))
            finalists.sort(key=lambda x: x[0], reverse=True)
            if not finalists:
                continue
            key, fp, item = finalists[0]
            assert css_ok(item["hx"], item["hz"])
            assert item["n"] == 576 and item["k"] >= 144
            assert max(r.bit_count()
                       for r in item["hx"]+item["hz"]) <= 12
            assert all(len(cell) == len(set(cell)) <= 5
                       for grid in (item["A"], item["B"])
                       for row in grid for cell in row)
            emitted.add(fp)
            cells = {
                side: tuple(tuple(tuple(cell) for cell in row) for row in grid)
                for side, grid in (("A", item["A"]), ("B", item["B"]))}
            chosen = dict(cells=cells, k=item["k"], selection=key,
                          card=(architecture, mode), no_logical_le4=True)
            break
        bank.append(chosen)
    return tuple(bank)


@lru_cache(maxsize=4)
def NEW_CTX(rng, G, idx):
    """Pure deterministic bank; idx=-1 builds the shared filling computation."""
    if idx >= 0:
        selected = NEW_CTX(None, G, -1)[idx]
        if selected is None:
            raise ValueError("no fresh mixed-order physical-witness card passed")
        return selected

    from itertools import combinations

    assert G.q == 48
    assert A_SHAPE == (1, 2) and B_SHAPE == (2, 5)
    q, mask = G.q, (1 << G.q)-1
    atlas = _atlas(G)

    # B rows total eight. A=(3,4) permits a degree-five B column.
    # Every architecture has HX<=12 and HZ<=12, without dead lines.
    plans = (
        (((4,4),), ((2,2,1,1,2), (1,1,2,2,2))),
        (((4,4),), ((3,0,2,1,2), (0,3,1,2,2))),
        (((4,4),), ((3,1,1,2,1), (1,3,2,1,1))),
        (((3,4),), ((3,1,1,2,1), (2,2,2,1,1))),
        (((3,4),), ((4,0,1,2,1), (1,3,2,1,1))),
    )
    for sa, sb in plans:
        assert _grid_weights(sa, sb) == 12
        assert _no_dead_lines(sa, sb)

    orders = {}
    for g in G.elements:
        value = G.e
        for exponent in range(1, q+1):
            value = G.mul(value, g)
            if value == G.e:
                orders[g] = exponent
                break
        else:
            raise ValueError("finite-group order calculation failed")
    palettes = {
        order: tuple(g for g in G.elements if orders[g] == order)
        for order in (4,6,12)
    }
    crossing4 = tuple(g for g in palettes[4] if g[3])
    assert crossing4 and all(palettes.values())
    quotient_pools = {
        t: tuple(g for g in G.elements if _quotient(g) == t)
        for t in range(8)
    }

    # These normal subgroups define support spaces only, not matrix lifts.
    subgroups = (
        tuple((0,0,c,0) for c in range(3)),
        (G.e, (2,0,0,0)),
        tuple((a,0,c,0) for a in (0,2) for c in range(3)),
        tuple((0,0,c,d) for d in range(2) for c in range(3)),
        tuple((a,b,0,0) for b in range(2) for a in range(4)),
        tuple((a,0,c,d) for d in range(2)
              for c in range(3) for a in (0,2)),
    )
    fibers = {}
    translations = {}
    for side in ("A", "B"):
        partitions = []
        for subgroup in subgroups:
            unseen, partition = set(range(q)), []
            while unseen:
                g = G.elements[min(unseen)]
                orbit = {
                    G.idx(G.mul(g,h) if side == "A" else G.mul(h,g))
                    for h in subgroup
                }
                assert len(orbit) == len(subgroup)
                unseen.difference_update(orbit)
                partition.append(tuple(sorted(orbit)))
            partitions.append(tuple(sorted(partition)))
        fibers[side] = tuple(partitions)
        translations[side] = tuple(
            tuple(G.idx(G.mul(g,t) if side == "A" else G.mul(t,g))
                  for g in G.elements)
            for t in G.elements
        )

    def syndrome(cols, word):
        answer = 0
        while word:
            bit = word & -word
            answer ^= cols[bit.bit_length()-1]
            word ^= bit
        return answer

    def moved(word, side, shift):
        permutation = translations[side][shift]
        answer = 0
        while word:
            bit = word & -word
            block, position = divmod(bit.bit_length()-1, q)
            answer ^= 1 << (block*q+permutation[position])
            word ^= bit
        return answer

    def orbit_memory(cols, words, side, combine=False):
        """Remove witness-orbit ties, not label codes by transported copies."""
        canonical = {}
        for word in sorted(set(words),
                           key=lambda w: (w.bit_count(), w))[:64]:
            if not word:
                continue
            representative = min(moved(word, side, t) for t in range(q))
            canonical[representative] = representative
        reps = sorted(canonical,
                      key=lambda w: (w.bit_count(), w))[:24]
        if combine:
            discoveries = set(reps)
            base = reps[:8]
            for i, left in enumerate(base):
                for right in base[:i+1]:
                    candidates = [
                        left ^ moved(right, side, t) for t in range(q)]
                    candidates = sorted(
                        {w for w in candidates if w},
                        key=lambda w: (w.bit_count(), w))[:3]
                    for word in candidates:
                        discoveries.add(min(
                            moved(word, side, t) for t in range(q)))
            reps = sorted(discoveries,
                          key=lambda w: (w.bit_count(), w))[:24]
        assert all(syndrome(cols, w) == 0 for w in reps)
        return reps

    def norm_words(cols, side, deep=False):
        """Actual restricted witnesses; no missed witness receives credit."""
        blocks = len(cols)//q
        found = set()
        partitions = fibers[side] if deep else fibers[side][:2]
        for p, partition in enumerate(partitions):
            atoms = [
                sum(1 << (block*q+t) for t in orbit)
                for block in range(blocks) for orbit in partition]
            values = [syndrome(cols, word) for word in atoms]
            seen = {0: 0}

            def insert(value, word):
                old = seen.get(value)
                if old is None:
                    seen[value] = word
                elif old != word:
                    witness = word ^ old
                    if witness and witness.bit_count() <= 24:
                        found.add(witness)

            for value, word in zip(values, atoms):
                insert(value, word)
            for i, value in enumerate(values):
                for j in range(i):
                    insert(value ^ values[j], atoms[i] ^ atoms[j])
            if deep and p == 0:
                # Complete C3-fiber collisions through six fibers.
                # This remains a restricted, not full quantum, test.
                for i in range(2, len(values)):
                    for j in range(1, i):
                        pair, word = values[i] ^ values[j], atoms[i] ^ atoms[j]
                        for h in range(j):
                            insert(pair ^ values[h], word ^ atoms[h])
            if len(found) > 256:
                found = set(sorted(
                    found, key=lambda w: (w.bit_count(), w))[:64])
        return sorted(found, key=lambda w: (w.bit_count(), w))[:32]

    def physical_coordinates(cols, side):
        """Carry physical supports through ordinary and restricted elimination."""
        ordinary = [(value, 1 << c) for c, value in enumerate(cols)]
        invariant3, even3, invariant2 = [], [], []
        for block in range(len(cols)//q):
            for orbit in fibers[side][0]:
                a, b, c = [block*q+t for t in orbit]
                wa, wb, wc = 1 << a, 1 << b, 1 << c
                invariant3.append(
                    (cols[a] ^ cols[b] ^ cols[c], wa ^ wb ^ wc))
                even3.extend((
                    (cols[a] ^ cols[b], wa ^ wb),
                    (cols[a] ^ cols[c], wa ^ wc)))
            for orbit in fibers[side][1]:
                a, b = [block*q+t for t in orbit]
                invariant2.append(
                    (cols[a] ^ cols[b], (1 << a) ^ (1 << b)))
        return ordinary, even3, invariant3, invariant2

    def kernel_probe(cols, side, local, trials, seeds=()):
        """Bounded true-kernel search with physical, not coordinate, weights."""
        memory = sorted(
            {w for w in seeds if w and syndrome(cols, w) == 0},
            key=lambda w: (w.bit_count(), w))[:32]
        coordinates = physical_coordinates(cols, side)
        for trial in range(trials):
            active = coordinates[trial % len(coordinates)]
            order = list(range(len(active)))
            local.shuffle(order)
            pivots, dependencies = {}, []
            for c in order:
                value, word = active[c]
                while value:
                    p = value.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value and word:
                    dependencies.append(word)

            words = sorted(set(memory+dependencies),
                           key=lambda w: (w.bit_count(), w))
            discoveries = set(memory)
            discoveries.update(words[:32])
            for sweep in range(2):
                donors = words[:28]
                reduced = []
                for word in words:
                    weight = word.bit_count()
                    for donor in donors:
                        proposal = word ^ donor
                        pw = proposal.bit_count()
                        if 0 < pw < weight:
                            word, weight = proposal, pw
                    reduced.append(word)
                words = sorted(set(reduced),
                               key=lambda w: (w.bit_count(), w))
                discoveries.update(words[:32])
            short = sorted(discoveries,
                           key=lambda w: (w.bit_count(), w))[:28]
            for i, word in enumerate(short):
                for other in short[:i]:
                    proposal = word ^ other
                    if proposal:
                        discoveries.add(proposal)
            memory = sorted(discoveries,
                            key=lambda w: (w.bit_count(), w))[:32]
            if trials >= 12 and trial % 8 == 7:
                memory = orbit_memory(cols, memory, side, combine=True)
            if memory and memory[0].bit_count() <= 6:
                break
        if trials >= 12:
            memory = orbit_memory(cols, memory, side, combine=True)
        assert memory
        assert all(syndrome(cols, w) == 0 for w in memory)
        return tuple(memory)

    def affine_rank(labels):
        return gf2_rank([t ^ labels[0] for t in labels[1:]])

    def draw_cell(local, count, mode, side, i, j, batch):
        if count == 0:
            return []
        if count == 1:
            return [G.pick(local)]
        for restart in range(512):
            if count == 2:
                policy = (
                    (4,4), (12,12), (6,4), (12,4)
                )[mode]
                wanted = policy[(i+j+batch) % 2]
                palette = crossing4 if wanted == 4 else palettes[wanted]
                anchor = G.pick(local)
                cell = [anchor, G.mul(anchor, local.choice(palette))]
            else:
                if mode == 0:
                    labels = local.sample(range(8), count)
                    if count == 4 and affine_rank(labels) != 3:
                        continue
                elif mode == 1:
                    if count == 4:
                        anchor = local.randrange(8)
                        u, v = local.sample(range(1,8), 2)
                        labels = [anchor, anchor ^ u, anchor ^ v,
                                  anchor ^ u ^ v]
                    else:
                        labels = local.sample(range(8), count)
                elif mode == 2:
                    labels = local.sample(range(8), count-1)
                    labels.append(labels[0])
                    local.shuffle(labels)
                else:
                    sign = local.randrange(2)
                    labels = local.sample(range(4), count-1)
                    labels = [t | (sign << 2) for t in labels]
                    labels.append(local.randrange(4) | ((1-sign) << 2))
                cell = [local.choice(quotient_pools[t]) for t in labels]

            if len(set(cell)) != count:
                continue
            relatives = [
                G.mul(G.inv(g), h)
                for t, g in enumerate(cell) for h in cell[:t]]
            if any(orders[g] <= 2 for g in relatives):
                continue
            if count >= 3:
                generators = [G.mul(G.inv(cell[0]), g) for g in cell[1:]]
                if not any(
                    G.mul(g,h) != G.mul(h,g)
                    for t, g in enumerate(generators)
                    for h in generators[:t]):
                    continue
            return cell
        raise ValueError("mixed-order affine-quotient cell recipe exhausted")

    def generated_a(A):
        generators = [
            G.mul(G.inv(cell[0]), g)
            for cell in A[0] for g in cell[1:]]
        members, seen = [G.e], {G.e}
        for value in members:
            for g in generators:
                product = G.mul(value, g)
                if product not in seen:
                    seen.add(product)
                    members.append(product)
        return len(members) == q

    def b_column(cells):
        """XOR superposition of the scaffold-derived B unit actions."""
        columns = [0]*q
        for i, cell in enumerate(cells):
            for g in cell:
                for t, value in enumerate(atlas["B"][G.idx(g)]):
                    columns[t] ^= value << (i*q)
        return columns

    def pair_counts(cols):
        result = Counter()
        for value in cols:
            bits = []
            while value:
                bit = value & -value
                bits.append(bit)
                value ^= bit
            for i, bit in enumerate(bits):
                for other in bits[:i]:
                    result[bit | other] += 1
        return result

    def draw_grid(local, counts, mode, side, batch):
        if side == "A":
            for retry in range(64):
                grid = [[draw_cell(
                    local, count, mode, side, i, j, batch)
                    for j, count in enumerate(row)]
                    for i, row in enumerate(counts)]
                if generated_a(grid):
                    return grid
            raise ValueError("A relative supports did not generate the group")

        grid = [[[] for _ in range(5)] for _ in range(2)]
        used = Counter()
        order = list(range(5))
        local.shuffle(order)
        order.sort(key=lambda j: -sum(counts[i][j] for i in range(2)))
        for j in order:
            winner = None
            for attempt in range(4):
                cells = [draw_cell(
                    local, counts[i][j], mode, side, i, j, batch)
                    for i in range(2)]
                pairs = pair_counts(b_column(cells))
                cost = sum(v*used[p] + v*(v-1)//2
                           for p, v in pairs.items())
                key = (cost, local.random())
                if winner is None or key < winner[0]:
                    winner = key, cells, pairs
            _, cells, pairs = winner
            for i in range(2):
                grid[i][j] = cells[i]
            used.update(pairs)
        return grid

    def component_key(item):
        d = item["words"][0].bit_count()
        load = sum(w.bit_count() <= d+1 for w in item["words"])
        return (min(d,20), -load, -item["cycles"])

    def component_pool(local, side, counts, mode, batch):
        strata, seen = {}, set()

        def admit(grid):
            labels = tuple(tuple(sorted(cell))
                           for row in grid for cell in row)
            if labels in seen:
                return
            seen.add(labels)
            cols = _component_columns(G, atlas, side, grid)
            if not _clean_component(cols):
                return
            rank = gf2_rank(cols)
            if rank < (q-8 if side == "A" else 2*q-4):
                return
            seeds = norm_words(cols, side)
            words = kernel_probe(cols, side, local, 3, seeds)
            item = dict(grid=grid, cols=cols, rank=rank, words=words,
                        cycles=_cycle_count(cols))
            bucket = strata.setdefault(rank, [])
            bucket.append(item)
            bucket.sort(key=component_key, reverse=True)
            del bucket[3:]

        quota = (96 if side == "A" else 160) + 32*batch
        for attempt in range(quota):
            admit(draw_grid(local, counts, mode, side, batch))

        positions = [(i,j) for i,row in enumerate(counts)
                     for j,count in enumerate(row) if count]
        for step in range(24 if side == "A" else 48):
            if not strata:
                break
            rank = local.choice(sorted(strata))
            parent = local.choice(strata[rank])
            grid = [[list(cell) for cell in row] for row in parent["grid"]]
            pressure = [
                1+sum(((w >> (j*q)) & mask).bit_count()
                      for w in parent["words"][:10])
                for i,j in positions]
            i,j = local.choices(positions, weights=pressure, k=1)[0]
            grid[i][j] = draw_cell(
                local, counts[i][j], mode, side, i, j, batch)
            if side == "B" and step % 5 == 4:
                for r in range(2):
                    grid[r][j] = draw_cell(
                        local, counts[r][j], mode, side, r, j, batch)
            if side == "A" and not generated_a(grid):
                continue
            # Changed matrices never inherit the parent's kernel words.
            admit(grid)

        first = sorted(
            (bucket[0] for bucket in strata.values()),
            key=component_key, reverse=True)
        pool = first[:8]
        extras = sorted(
            (item for bucket in strata.values() for item in bucket[1:]),
            key=component_key, reverse=True)
        pool.extend(extras[:max(0,8-len(pool))])
        for item in pool:
            extra = norm_words(item["cols"], side, deep=True)
            item["words"] = kernel_probe(
                item["cols"], side, local, 16,
                tuple(item["words"])+tuple(extra))
        pool.sort(key=component_key, reverse=True)
        pool = pool[:4]
        for item in pool:
            item["words"] = kernel_probe(
                item["cols"], side, local, 40, item["words"])
            item["d"] = item["words"][0].bit_count()
        return sorted(pool, key=component_key, reverse=True)

    def full_probe(checks, stabilizers, n, local, trials, seeds):
        """Genuine logical witnesses in both sectors, modulo exact row span."""
        cols, span = _columns(checks, n), _basis(stabilizers)
        found, memory = set(), []
        best = n+1

        def observe(word):
            nonlocal best
            if word and _outside(word, span):
                found.add(word)
                best = min(best, word.bit_count())

        for word in seeds:
            if word and syndrome(cols, word) == 0:
                observe(word)
        order = list(range(n))
        for trial in range(trials):
            local.shuffle(order)
            # Alternate sector-first and unrestricted information sets.
            if trial % 3 != 2:
                order.sort(key=lambda c:
                           (c < 10*q) if trial % 3 == 0 else (c >= 10*q))
            pivots, dependencies = {}, []
            for c in order:
                value, word = cols[c], 1 << c
                while value:
                    p = value.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    dependencies.append(word)
                    observe(word)
            reducers = sorted(
                set(dependencies+memory),
                key=lambda w: (w.bit_count(), w))[:32]
            logicals = sorted(
                found, key=lambda w: (w.bit_count(), w))[:8]
            discoveries = set(reducers)
            for start in logicals:
                word = start
                for sweep in range(3):
                    chosen, weight = word, word.bit_count()
                    for other in reducers:
                        candidate = word ^ other
                        cw = candidate.bit_count()
                        if 0 < cw < weight and _outside(candidate, span):
                            chosen, weight = candidate, cw
                    nearby = sorted(
                        (((word ^ row).bit_count(), row)
                         for row in stabilizers),
                        key=lambda entry: entry[0])
                    for cw, row in nearby:
                        if cw < weight:
                            chosen, weight = word ^ row, cw
                    moves = [row for _, row in nearby[:12]]
                    for i, row in enumerate(moves):
                        for other in moves[:i]:
                            candidate = word ^ row ^ other
                            cw = candidate.bit_count()
                            if cw < weight:
                                chosen, weight = candidate, cw
                    if chosen == word:
                        break
                    word = chosen
                    observe(word)
                discoveries.add(word)
            for i, word in enumerate(logicals):
                for other in logicals[:i]:
                    observe(word ^ other)
            found = set(sorted(
                found, key=lambda w: (w.bit_count(), w))[:24])
            memory = sorted(
                discoveries, key=lambda w: (w.bit_count(), w))[:32]
        return best, tuple(sorted(
            found, key=lambda w: (w.bit_count(), w))[:12])

    def merit(item):
        d = min(item["da"], item["db"],
                item.get("dx",577), item.get("dz",577),
                item.get("dressed",577), 20)
        return (item["k"]*d*d/item["n"], d, -item["cycles"])

    bank, emitted = [], set()
    for slot in range(N_SLOTS):
        architecture, mode = slot % 5, (slot//5) % 4
        sa, sb = plans[architecture]
        local = random.Random(
            BASE_SEED*100003 + 48025397 + 104729*slot)
        chosen = None
        for batch in range(3):
            ap = component_pool(local, "A", sa, mode, batch)
            bp = component_pool(
                local, "B", sb, (mode+architecture) % 4, batch)
            proposals = []
            for a in ap:
                for b in bp:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (225*n <= 1000*k < 275*n):
                        continue
                    ar = [sum(
                        ((row >> (5*j*q)) & mask) << (j*q)
                        for j in range(2)) for row in hx[:q]]
                    br = [row & ((1 << (5*q))-1)
                          for row in hz[:2*q]]
                    assert _columns(ar,2*q) == a["cols"]
                    assert _columns(br,5*q) == b["cols"]
                    proposals.append(dict(
                        A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                        aw=list(a["words"]), bw=list(b["words"]),
                        da=a["d"], db=b["d"],
                        cycles=a["cycles"]+b["cycles"]))
            proposals.sort(key=merit, reverse=True)

            finalists, local_fingerprints = [], set()
            for item in proposals:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                fp = _fingerprint(hx,hz,n)
                if fp in emitted or fp in local_fingerprints:
                    continue
                if not (_no_logical_le4(hx,hz,n)
                        and _no_logical_le4(hz,hx,n)):
                    continue
                local_fingerprints.add(fp)
                aw = [
                    sum(((word >> (j*q)) & mask)
                        << ((j*5+beta)*q) for j in range(2))
                    for word in item["aw"] for beta in range(5)]
                bw = [word << (j*5*q)
                      for word in item["bw"] for j in range(2)]
                assert all(syndrome(_columns(hx,n),w) == 0 for w in aw)
                assert all(syndrome(_columns(hz,n),w) == 0 for w in bw)
                dx, wx = full_probe(hz,hx,n,local,6,bw)
                dz, wz = full_probe(hx,hz,n,local,6,aw)
                if dx > n or dz > n:
                    continue
                item.update(dx=dx,dz=dz,wx=wx,wz=wz,
                            a_seeds=tuple(aw),b_seeds=tuple(bw),
                            dressed=_dressed_reading(item,q),
                            fingerprint=fp)
                finalists.append(item)
                if len(finalists) == 3:
                    break

            # Every joint finalist gets equal fresh deeper budgets.
            for item in finalists:
                dx, wx = full_probe(
                    item["hz"],item["hx"],item["n"],local,12,
                    item["b_seeds"]+item["wx"])
                dz, wz = full_probe(
                    item["hx"],item["hz"],item["n"],local,12,
                    item["a_seeds"]+item["wz"])
                item["dx"],item["dz"] = min(item["dx"],dx),min(item["dz"],dz)
            if not finalists:
                continue
            winner = max(finalists,key=merit)
            A,B = winner["A"],winner["B"]
            hx,hz,n = lifted_matrices(G,A,B)
            assert css_ok(hx,hz)
            assert n == 576
            assert lifted_k(G,A,B) == winner["k"]
            assert 225*n <= 1000*winner["k"] < 275*n
            assert max(row.bit_count() for row in hx+hz) <= 12
            assert _grid_weights(sa,sb) <= 12
            assert _no_dead_lines(sa,sb)
            assert all(
                len(cell) == len(set(cell)) <= 5
                for grid in (A,B) for row in grid for cell in row)
            emitted.add(winner["fingerprint"])
            chosen = dict(
                cells={
                    side: tuple(tuple(tuple(cell) for cell in row)
                                for row in grid)
                    for side,grid in (("A",A),("B",B))},
                k=winner["k"],selection=merit(winner),
                card=(architecture,mode),no_logical_le4=True)
            break
        # Failed tasks never substitute an emitted or archived labeling.
        bank.append(chosen)
    return tuple(bank)


def CELL(rng, G, side, i, j, ctx):
    """Return the complete physical-witness-selected cell without resampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Uniform legal support rolling on C17, (2,5)x(2,6), did not suppress sector-1 weight-five logicals: all five measured parent codes had d30s=5.
# REFUTED: Odd-central generator deletions on S3 x C4 supply fifteen distinct improving principal designs: ten principal slots were isomorphic duplicates, and all five measured leaders remained [504,297,4] with thousands of sector-1 weight-four witnesses.
# REFUTED: Q8 x S3 noncommuting quartics, exclusively order-12 doubletons and the parent's component/dressed screening produce five readings >=14; all five measured leaders instead retained d30s=12, with sector-one minima and fitness 38.7.
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

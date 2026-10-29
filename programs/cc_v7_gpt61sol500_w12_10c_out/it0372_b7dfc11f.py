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
CONCEPT = ("Metacyclic C23 semidirect C2 with trivial action, order 46, A=(2,3), B=(2,3): primitive-order doubletons and cancellation-screened cofactor witnesses should produce five fresh distance-28 readings with k>=50 and exceed cell 0.100's fitness 64.6. This is a measured-distance hypothesis, not a certificate.")
GROUP = ("L2 parameter-only size repair within the unchanged metacyclic construction rule: (m,r,s) changes from (29,2,28) to (23,2,1). The new size factor 13 requires q<=46. This trivial-action specialization is cyclic C46, whose measured (2,3)x(2,3) bed leads cell 0.100 at 64.6; the inversion parent's five readings were eight. Keep GTYPE and the multiplication/inverse formulas unchanged.")
SHAPES = ("Change both sides to (2,3): n=598, kb=46, guaranteed rate 1/13, and equal 138-variable/92-check components. Order 46 maximizes the size window. Require exact assembled 0<k/n<0.125. Parent weight-eight minima were entirely sector-one Acol/Bcol witnesses, prominently Acol1 and Bcol1; the added independent component rows directly change these kernels, while full-code searches also examine sector two.")
FILLING = ("Twenty tasks cross five cap-legal count architectures with four intrinsic support geometries. Doubleton differences have order 46; larger cells have distinct C23 coordinates and mixed C2 occupancy. Compile component actions only from verified unit lifts, exclude component kernels through four, seed searches with syndrome-verified commuting cofactors and quotient-fiber words, recombine translated witnesses, and independently deepen finalists. Exact full ranks, logical-four exclusion, genuine full-code logical probes, stabilizer dressings, CSS checks and unequal uncoloured Tanner invariants govern emission. Found readings are upper estimates, never distance certificates.")
FAMILY_NAME = "r010f_m23r2s1_23x23_primitive46_orbitcofactor_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 46

def custom_ops(q):
    # Same metacyclic construction, resized for n=13q.
    # Trivial action: 1^2=1 mod 23; coprime factors give cyclic C46.
    if not valid_q(q):
        raise ValueError("this metacyclic specialization requires order 46")
    m, r, s = 23, 2, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


from functools import lru_cache
from collections import Counter


# Complete count grids. Their (maximum HX, maximum HZ) weights are
# respectively (12,12), (12,12), (12,12), (10,10), and (12,12).
_C46_COUNTS = (
    (((2,2,2), (2,2,3)), ((2,2,2), (2,2,3))),
    (((2,2,3), (2,3,2)), ((2,2,2), (2,2,3))),
    (((3,2,2), (1,2,3)), ((3,2,2), (1,2,3))),
    (((3,2,1), (1,2,3)), ((3,2,1), (1,2,3))),
    (((3,2,2), (1,2,3)), ((2,3,2), (2,2,3))),
)


def _c46_columns(rows, width):
    out = [0] * width
    for r, original in enumerate(rows):
        row = original
        while row:
            bit = row & -row
            out[bit.bit_length()-1] |= 1 << r
            row ^= bit
    return tuple(out)


def _c46_rows(cols, height):
    rows = [0] * height
    for c, original in enumerate(cols):
        syndrome = original
        while syndrome:
            bit = syndrome & -syndrome
            rows[bit.bit_length()-1] |= 1 << c
            syndrome ^= bit
    return rows


def _c46_basis(rows):
    span = {}
    for word in rows:
        while word:
            p = word.bit_length()-1
            old = span.get(p)
            if old is None:
                span[p] = word
                break
            word ^= old
    return span


def _c46_outside(word, span):
    while word:
        p = word.bit_length()-1
        old = span.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _c46_clean4(cols):
    """Exact constituent-kernel exclusion through four, without parity assumptions."""
    seen = {0}
    for syndrome in cols:
        if syndrome in seen:
            return False
        seen.add(syndrome)
    for i, syndrome in enumerate(cols):
        for other in cols[:i]:
            pair = syndrome ^ other
            if pair in seen:
                return False
            seen.add(pair)
    return True


def _c46_logical4(cols, stabilizers):
    """Exact logical existence through four, including both LP sectors."""
    span = _c46_basis(stabilizers)
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = word
        elif _c46_outside(word ^ old, span):
            return True
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            word = (1 << i) | (1 << j)
            old = seen.get(value)
            if old is None:
                seen[value] = word
            elif _c46_outside(word ^ old, span):
                return True
    return False


def _c46_cycles(cols):
    pairs, total = {}, 0
    for original in cols:
        syndrome, neighbors = original, []
        while syndrome:
            bit = syndrome & -syndrome
            neighbors.append(bit.bit_length()-1)
            syndrome ^= bit
        for i, a in enumerate(neighbors):
            for b in neighbors[:i]:
                key = (b, a)
                old = pairs.get(key, 0)
                total += old
                pairs[key] = old + 1
    return total


def _c46_syndrome(word, cols):
    answer = 0
    while word:
        bit = word & -word
        answer ^= cols[bit.bit_length()-1]
        word ^= bit
    return answer


def _c46_probe(cols, local, trials, seeds=(), stabilizers=None, dress=False):
    """Actual kernel/logical witnesses; no missing-witness sentinel is promoted."""
    span = None if stabilizers is None else _c46_basis(stabilizers)
    memory = sorted({w for w in seeds if w},
                    key=lambda w: (w.bit_count(), w))[:40]
    assert all(_c46_syndrome(w, cols) == 0 for w in memory)
    order = list(range(len(cols)))
    best, logical_memory = len(cols)+1, set()

    def observe(word):
        nonlocal best, logical_memory
        if not word or word.bit_count() > best:
            return
        if span is not None and not _c46_outside(word, span):
            return
        weight = word.bit_count()
        if weight < best:
            best, logical_memory = weight, {word}
        elif len(logical_memory) < 16:
            logical_memory.add(word)

    for word in memory:
        observe(word)
    for trial in range(trials):
        local.shuffle(order)
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
                observe(word)

        reducers = sorted(set(dependencies+memory),
                          key=lambda w: (w.bit_count(), w))[:32]
        for sweep in range(2):
            reduced = []
            for word in dependencies:
                weight = word.bit_count()
                for other in reducers:
                    candidate = word ^ other
                    cw = candidate.bit_count()
                    if 0 < cw < weight:
                        word, weight = candidate, cw
                reduced.append(word)
                observe(word)
            dependencies = reduced
            reducers = sorted(set(dependencies+memory),
                              key=lambda w: (w.bit_count(), w))[:32]

        short = sorted(set(reducers+memory+list(logical_memory)),
                       key=lambda w: (w.bit_count(), w))[:36]
        discoveries = set(short)
        for i, word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                if candidate:
                    observe(candidate)
                    discoveries.add(candidate)
        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:40]
        if best <= 4:
            break

    if dress and stabilizers is not None:
        # Adding stabilizers preserves the nonzero logical class.
        for original in sorted(logical_memory, key=int.bit_count)[:6]:
            word = original
            for sweep in range(3):
                chosen, weight = word, word.bit_count()
                nearby = sorted(
                    (((word ^ row).bit_count(), row)
                     for row in stabilizers),
                    key=lambda item: item[0])[:12]
                for cw, row in nearby:
                    if cw < weight:
                        chosen, weight = word ^ row, cw
                for i, (_, row) in enumerate(nearby):
                    for _, other in nearby[:i]:
                        candidate = word ^ row ^ other
                        cw = candidate.bit_count()
                        if cw < weight:
                            chosen, weight = candidate, cw
                if chosen == word:
                    break
                word = chosen
                observe(word)
        memory = sorted(set(memory) | logical_memory,
                        key=lambda w: (w.bit_count(), w))[:40]

    return best, tuple(memory), tuple(sorted(logical_memory))


def _c46_draw(local, counts, mode):
    """Primitive doubletons and intrinsic larger-support geometries."""
    units = [t for t in range(1, 46, 2) if t != 23]

    def support(size):
        if not size:
            return []
        base = local.randrange(46)
        if size == 1:
            return [base]
        if size == 2:
            delta = local.choice(units)
            return [base, (base+delta) % 46]
        if mode and size in (3, 4):
            positions = ((0,1,(3,5,9)[mode-1]) if size == 3
                         else (0,1,3,7))
            step = local.choice(units)
            return [(base+step*t) % 46 for t in positions]
        residues = local.sample(range(23), size)
        phase = local.randrange(2)
        quota = ([phase, phase, 1-phase] if size == 3
                 else [phase, phase, 1-phase, 1-phase])
        local.shuffle(quota)
        return [a if a % 2 == b else a+23
                for a, b in zip(residues, quota)]

    best = None
    for restart in range(4):
        used = [set(), set()]
        crossed = set()
        chosen_columns = [None]*3
        cost = 0
        order = list(range(3))
        local.shuffle(order)
        for j in order:
            winner = None
            for attempt in range(12):
                x, y = [support(counts[i][j]) for i in range(2)]
                anchor = x[0] if x else y[0]
                x, y = [sorted((v-anchor) % 46 for v in vals)
                        for vals in (x,y)]
                penalty, differences = 0, []
                for i, vals in enumerate((x,y)):
                    raw = [(a-b) % 46 for a in vals
                           for b in vals if a != b]
                    ds = set(raw)
                    penalty += len(raw)-len(ds)+2*len(ds & used[i])
                    differences.append(ds)
                raw_cross = [(a-b) % 46 for a in x for b in y]
                cs = set(raw_cross)
                penalty += len(raw_cross)-len(cs)+2*len(cs & crossed)
                item = (penalty, x, y, differences, cs)
                if winner is None or penalty < winner[0]:
                    winner = item
                if penalty == 0:
                    break
            penalty, x, y, differences, cs = winner
            cost += penalty
            chosen_columns[j] = (x,y)
            for i in range(2):
                used[i].update(differences[i])
            crossed.update(cs)
        grid = [
            [[(t % 23, t % 2) for t in chosen_columns[j][i]]
             for j in range(3)] for i in range(2)]
        if best is None or cost < best[0]:
            best = (cost, grid)
        if cost == 0:
            break
    grid = best[1]
    assert all(
        len(cell) == counts[i][j] == len(set(cell))
        and len({g[0] for g in cell}) == len(cell)
        and (len(cell) != 2 or cell[0][1] != cell[1][1])
        for i, row in enumerate(grid) for j, cell in enumerate(row))
    return grid


def _c46_cofactors(G, grid, cols, inverse):
    """Commuting minors, accepted only against the verified syndromes."""
    entries = [
        [set(G.inv(g) if inverse else g for g in cell) for cell in row]
        for row in grid]

    def product(left, right):
        answer = set()
        for g in left:
            for h in right:
                term = G.mul(g, h)
                if term in answer:
                    answer.remove(term)
                else:
                    answer.add(term)
        return answer

    word = 0
    for omitted in range(3):
        s, t = [j for j in range(3) if j != omitted]
        minor = (product(entries[0][s], entries[1][t])
                 ^ product(entries[0][t], entries[1][s]))
        for g in minor:
            word ^= 1 << (omitted*G.q+G.idx(g))
    return (word,) if word and _c46_syndrome(word, cols) == 0 else ()


def _c46_fibers(G, cols):
    """Exhaust the kernel restricted to six normal-C23 indicators."""
    fibers = [
        sum(1 << (j*G.q+G.idx(g)) for g in G.elements if g[1] == b)
        for j in range(3) for b in range(2)]
    syndromes = [_c46_syndrome(w, cols) for w in fibers]
    found = []
    for mask in range(1, 64):
        word, syndrome = 0, 0
        for j in range(6):
            if (mask >> j) & 1:
                word ^= fibers[j]
                syndrome ^= syndromes[j]
        if syndrome == 0:
            found.append(word)
    return tuple(sorted(found, key=lambda w: (w.bit_count(), w))[:12])


def _c46_orbits(cols, seeds, rounds=2):
    """Bounded translated-witness recombination, not a distance exclusion."""
    mask23 = (1 << 23)-1

    def rotate(value, shift):
        if not shift:
            return value
        return ((value << shift) | (value >> (23-shift))) & mask23

    def move(word, t):
        shift, flip = t % 23, t % 2
        answer = 0
        for j in range(3):
            block = (word >> (46*j)) & ((1 << 46)-1)
            lo, hi = rotate(block & mask23, shift), rotate(block >> 23, shift)
            if flip:
                lo, hi = hi, lo
            answer |= (lo | (hi << 23)) << (46*j)
        return answer

    canonical_cache = {}

    def canonical(word):
        old = canonical_cache.get(word)
        if old is not None:
            return old
        orbit = [move(word, t) for t in range(46)]
        rep = min(orbit)
        for image in orbit:
            canonical_cache[image] = rep
        return rep

    reps = sorted({canonical(w) for w in seeds if w},
                  key=lambda w: (w.bit_count(), w))[:10]
    for _ in range(rounds):
        base = reps[:8]
        discoveries = set(reps)
        for i, left in enumerate(base):
            for right in base[:i+1]:
                alignments = []
                for t in range(46):
                    word = left ^ move(right, t)
                    if word:
                        alignments.append((word.bit_count(), word))
                for _, word in sorted(alignments)[:3]:
                    discoveries.add(canonical(word))
        reps = sorted(discoveries,
                      key=lambda w: (w.bit_count(), w))[:12]
    assert all(_c46_syndrome(w, cols) == 0 for w in reps)
    return tuple(reps)


def _c46_fingerprint(hx, hz, n):
    """Uncoloured Tanner invariant; unequal values certify nonisomorphism."""
    rows = list(hx)+list(hz)
    weights = [row.bit_count() for row in rows]
    links = [dict() for _ in rows]
    adjacency = [0]*len(rows)
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                links[i][j] = links[j][i] = overlap
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    profiles = []
    for i, neighbors in enumerate(links):
        histogram = Counter(
            (weights[j], overlap,
             (adjacency[i] & adjacency[j]).bit_count())
            for j, overlap in neighbors.items())
        profiles.append((weights[i], tuple(sorted(histogram.items()))))
    qubits = [[] for _ in range(n)]
    for r, original in enumerate(rows):
        row = original
        while row:
            bit = row & -row
            qubits[bit.bit_length()-1].append(weights[r])
            row ^= bit
    return (
        tuple(sorted(Counter(profiles).items())),
        tuple(sorted(Counter(tuple(sorted(v)) for v in qubits).items())))


@lru_cache(maxsize=1)
def _c46_bank(G):
    """Pure deterministic optimization of fresh structural tasks."""
    assert G.q == 46 and A_SHAPE == B_SHAPE == (2,3)
    q, width = G.q, 3*G.q
    mask = (1 << q)-1
    zero = [[[], [], []], [[], [], []]]
    atlas = {}

    # Every action is compiled from a scaffold-generated single-term lift.
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[], [], []], [[], [], []]]
            unit[0][0] = [g]
            A, B = (unit, zero) if side == "A" else (zero, unit)
            hx, hz, _ = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            blocks.append(_c46_columns([row & mask for row in source[:q]], q))
        atlas[side] = tuple(blocks)

    def compile_component(side, grid):
        cols = [0]*width
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = atlas[side][G.idx(g)]
                    for t, syndrome in enumerate(block):
                        cols[j*q+t] ^= syndrome << (i*q)
        return tuple(cols)

    def component_key(item):
        d = min(item["d"], 34)
        load = sum(w.bit_count() <= d+1 for w in item["words"])
        return (d, -load, -item["cycles"])

    bank, emitted = [], set()
    for slot in range(N_SLOTS):
        architecture, mode = divmod(slot % 20, 4)
        sa, sb = _C46_COUNTS[architecture]
        assert _grid_weights(sa, sb) <= 12
        assert _no_dead_lines(sa, sb)
        local = random.Random(
            BASE_SEED*100003 + 46010289 + 104729*slot)

        def component_pool(side, counts, batch):
            strata, seen = {}, set()

            def admit(grid):
                labels = tuple(tuple(sorted(cell))
                               for row in grid for cell in row)
                if labels in seen:
                    return
                seen.add(labels)
                cols = compile_component(side, grid)
                if not _c46_clean4(cols):
                    return
                seeds = (_c46_cofactors(G, grid, cols, side == "B")
                         + _c46_fibers(G, cols))
                d, words, _ = _c46_probe(cols, local, 2, seeds)
                if d > width:
                    return
                rank = gf2_rank(cols)
                item = dict(
                    grid=grid, cols=cols, rank=rank, d=d, words=words,
                    cycles=_c46_cycles(cols))
                bucket = strata.setdefault(rank, [])
                bucket.append(item)
                bucket.sort(key=component_key, reverse=True)
                del bucket[3:]

            for attempt in range(96+48*batch):
                admit(_c46_draw(local, counts, mode))

            positions = [(i,j) for i,row in enumerate(counts)
                         for j,count in enumerate(row) if count]
            for step in range(28+12*batch):
                available = [item for bucket in strata.values() for item in bucket]
                if not available:
                    admit(_c46_draw(local, counts, mode))
                    continue
                parent = local.choice(available)
                grid = [[list(cell) for cell in row] for row in parent["grid"]]
                fresh = _c46_draw(local, counts, mode)
                pressure = [
                    1+sum(((w >> (j*q)) & mask).bit_count()
                          for w in parent["words"][:8])
                    for i,j in positions]
                for _ in range(2 if step % 5 == 4 else 1):
                    i,j = local.choices(positions, weights=pressure, k=1)[0]
                    grid[i][j] = fresh[i][j]
                # Changed matrices get fresh witnesses, never inherited ones.
                admit(grid)

            pool = sorted(
                (item for bucket in strata.values() for item in bucket),
                key=component_key, reverse=True)[:8]
            for item in pool:
                orbit = _c46_orbits(item["cols"], item["words"], 2)
                d, words, _ = _c46_probe(
                    item["cols"], local, 20, item["words"]+orbit)
                item["d"] = min(item["d"], d)
                item["words"] = words
            pool.sort(key=component_key, reverse=True)
            pool = pool[:4]
            for item in pool:
                orbit = _c46_orbits(item["cols"], item["words"], 2)
                d, words, _ = _c46_probe(
                    item["cols"], local, 32, item["words"]+orbit)
                item["d"] = min(item["d"], d)
                item["words"] = words
            return sorted(pool, key=component_key, reverse=True)

        def joint_key(item):
            d = min(item["a"]["d"], item["b"]["d"],
                    item.get("dx", 599), item.get("dz", 599), 34)
            return (item["k"]*d*d/item["n"], d,
                    -item["a"]["cycles"]-item["b"]["cycles"])

        winner = None
        for batch in range(2):
            ap = component_pool("A", sa, batch)
            bp = component_pool("B", sb, batch)
            proposals = []
            for a in ap:
                for b in bp:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if n != 598 or not (0 < k and 8*k < n):
                        continue
                    ar = [
                        sum(((hx[(i*3)*q+t] >> (j*3*q)) & mask) << (j*q)
                            for j in range(3))
                        for i in range(2) for t in range(q)]
                    br = [row & ((1 << width)-1) for row in hz[:2*q]]
                    assert _c46_columns(ar, width) == a["cols"]
                    assert _c46_columns(br, width) == b["cols"]
                    proposals.append(dict(
                        A=A, B=B, hx=hx, hz=hz, n=n, k=k, a=a, b=b))
            proposals.sort(key=joint_key, reverse=True)

            finalists = []
            for item in proposals:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                fp = _c46_fingerprint(hx, hz, n)
                if fp in emitted:
                    continue
                xc, zc = _c46_columns(hz, n), _c46_columns(hx, n)
                if _c46_logical4(xc, hx) or _c46_logical4(zc, hz):
                    continue

                a_seeds = [
                    sum(((word >> (j*q)) & mask) << ((j*3+beta)*q)
                        for j in range(3))
                    for word in item["a"]["words"][:12] for beta in range(3)]
                b_seeds = [
                    word << (j*3*q)
                    for word in item["b"]["words"][:12] for j in range(3)]
                assert all(_c46_syndrome(w, zc) == 0 for w in a_seeds)
                assert all(_c46_syndrome(w, xc) == 0 for w in b_seeds)
                dx, _, wx = _c46_probe(
                    xc, local, 6, b_seeds, stabilizers=hx, dress=True)
                dz, _, wz = _c46_probe(
                    zc, local, 6, a_seeds, stabilizers=hz, dress=True)
                if dx > n or dz > n:
                    continue
                item.update(
                    dx=dx, dz=dz, xc=xc, zc=zc, wx=wx, wz=wz,
                    a_seeds=tuple(a_seeds), b_seeds=tuple(b_seeds), fp=fp)
                finalists.append(item)
                if len(finalists) >= 4:
                    break

            # Every finalist receives the same independent deeper budget.
            for item in finalists:
                dx, _, _ = _c46_probe(
                    item["xc"], local, 20,
                    item["b_seeds"]+item["wx"],
                    stabilizers=item["hx"], dress=True)
                dz, _, _ = _c46_probe(
                    item["zc"], local, 20,
                    item["a_seeds"]+item["wz"],
                    stabilizers=item["hz"], dress=True)
                item["dx"], item["dz"] = min(item["dx"], dx), min(item["dz"], dz)
            if finalists:
                winner = max(finalists, key=joint_key)
                break

        # An unsuccessful task is voided, never replaced by an orbit copy.
        if winner is None:
            bank.append(None)
            continue
        A, B = winner["A"], winner["B"]
        assert css_ok(winner["hx"], winner["hz"])
        assert max(row.bit_count()
                   for row in winner["hx"]+winner["hz"]) <= 12
        assert 0 < winner["k"] and 8*winner["k"] < 598
        assert all(len(cell) == len(set(cell)) <= 5
                   for grid in (A,B) for row in grid for cell in row)
        emitted.add(winner["fp"])
        bank.append(dict(
            cells={
                side: tuple(tuple(tuple(cell) for cell in row) for row in grid)
                for side,grid in (("A",A), ("B",B))},
            k=winner["k"], card=(architecture,mode),
            selection=joint_key(winner)))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return this independently optimized structural task."""
    selected = _c46_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh screened metacyclic order-46 target-cell filling")
    return selected


def CELL(rng, G, side, i, j, ctx):
    """Return every cell of the new shapes without further sampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Parent evidence: rolled supports plus uniform labels did not produce a measured top-five distance above 4 on D29, q=58, (1,3)x(1,3); all five readings were 4 with k=232.
# REFUTED: Rotation-gap partitions plus exact short-kernel screening remove the D29 single-row distance-eight bottleneck; all five measured current-parent codes still returned d30s=8 with thousands of sector-one witnesses.
# REFUTED: The parent's sixty prescribed gap/offset tasks ensure sixty different codes; evaluator isomorphism detection retained only 33 distinct codes.
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

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
CONCEPT = ("C37, order 37, shapes (2,3)x(2,4): optimizing actual constituent kernel witnesses and then screening their non-stabilizer quantum embeddings should improve the 0.150 cell's five-code fitness beyond 43.2.")
GROUP = ("Jump from nonabelian S3 x C4 to cyclic C37. This exact bed reached 43.2, compared with 41.6 for C36 on these shapes. Prime order permits every nonzero binomial difference to have order 37. These are measured-bed and algebraic motivations, not a certified distance advantage.")
SHAPES = ("Target cell 0.150 instead of the parent's high-rate plateau: n=16*37=592, kb=2*37=74, guaranteed rate 1/8. Components have 111 and 148 variables with 74 checks each. Exact full ranks enforce 0.125<=k/n<0.175. Parent minima occupied sector-one Acol/Bcol blocks, especially Acol2 for X and Acol1 for Z; constituent witnesses are therefore screened before mixed-sector stabilizer dressings.")
FILLING = ("Enumerate legal two-row support architectures and interleave augmentation-rank strata. Cross them with unrestricted, prescribed-gap, arithmetic-progression and difference-packing labels. Optimize fresh labels and witness-heavy cell repairs using verified component matrices, syndrome-checked cyclic cofactor words and retained information-set witnesses. Give finalists equal deeper budgets, test actual logical embeddings and full-code witnesses modulo stabilizers, and conservatively reject equal uncoloured Tanner invariants. High readings remain bounded-search upper estimates.")
FAMILY_NAME = "r015_C37_23x24_cofactor_witnessrepair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 37

def custom_ops(q):
    # Cyclic C37, with a one-coordinate tuple encoding.
    if not valid_q(q):
        raise ValueError("C37 requires order 37")
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


def _quotient_columns(checks, stabilizers, n):
    """Syndromes and linear normal forms modulo the stabilizer row space."""
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            previous = pivots.get(p)
            if previous is None:
                pivots[p] = row
                break
            row ^= previous
    ordered = sorted(pivots.items(), reverse=True)
    residues = []
    for c in range(n):
        v = 1 << c
        for p, row in ordered:
            if (v >> p) & 1:
                v ^= row
        residues.append(v)

    syndromes = [0] * n
    for r, row in enumerate(checks):
        bit = 1 << r
        while row:
            low = row & -row
            syndromes[low.bit_length() - 1] |= bit
            row ^= low
    return syndromes, residues


def _quotient_short_stats(checks, stabilizers, n):
    """Exact logical rejection through weight four, allowing stabilizers.

    Equal syndromes imply a commuting difference. Its stabilizer normal
    form decides whether that difference is logical. Sentinel 5 means
    no logical support of weight <=4, not an estimated larger distance.
    """
    syndromes, residues = _quotient_columns(checks, stabilizers, n)
    if any(s == 0 and r != 0 for s, r in zip(syndromes, residues)):
        return (1, 0, 0)

    singles = {}
    for s, r in zip(syndromes, residues):
        previous = singles.get(s)
        if previous is None:
            singles[s] = (r, 1)
        elif previous[0] != r:
            return (2, 0, 0)
        else:
            singles[s] = (r, previous[1] + 1)

    # Include the empty support to detect logical pairs with syndrome 0.
    previous = singles.get(0)
    if previous is not None and previous[0] != 0:
        return (1, 0, 0)
    singles[0] = (0, 1 if previous is None else previous[1] + 1)

    totals = {}
    equal_residues = {}
    triples = 0
    quartets = 0
    for i, s in enumerate(syndromes):
        ri = residues[i]
        for j in range(i):
            syndrome = s ^ syndromes[j]
            residue = ri ^ residues[j]
            singleton = singles.get(syndrome)
            if singleton is not None and singleton[0] != residue:
                # No weight-one/two logical survives the preceding tests.
                triples += singleton[1]

            key = (syndrome, residue)
            total = totals.get(syndrome, 0)
            equal = equal_residues.get(key, 0)
            quartets += total - equal
            totals[syndrome] = total + 1
            equal_residues[key] = equal + 1

    if triples:
        # Each logical triple has three singleton/pair partitions.
        return (3, triples // 3, quartets // 3)
    if quartets:
        # Overlapping pairs cannot differ logically at weight two,
        # already excluded. Each logical quartet has three partitions.
        return (4, 0, quartets // 3)
    return (5, 0, 0)


def _quotient_incidence(rows):
    """Check-degree profile and exact Tanner four-cycle count."""
    seen = {}
    cycles = 0
    for source in rows:
        positions = []
        while source:
            low = source & -source
            positions.append(low.bit_length() - 1)
            source ^= low
        for i, a in enumerate(positions):
            for b in positions[:i]:
                key = (b, a)
                old = seen.get(key, 0)
                cycles += old
                seen[key] = old + 1
    return tuple(sorted(row.bit_count() for row in rows)), cycles


def _quotient_profiles():
    """Canonical positive supports; both single-row budgets are explicit."""
    profiles = []
    for a in combinations_with_replacement(range(1, 6), 4):
        for b in combinations_with_replacement(range(1, 6), 5):
            if sum(a) + max(b) > W_MAX:
                continue
            if sum(b) + max(a) > W_MAX:
                continue
            # Rank escape experiments first avoid the immediate
            # two-monomial component relations wherever possible.
            priority = (
                max(a.count(1), b.count(1)),
                a.count(1) + b.count(1),
                -(sum(a) + sum(b)),
                max(a + b), a, b)
            profiles.append((priority, a, b))
    profiles.sort()
    return tuple((a, b) for _, a, b in profiles)


def _c37_support_cards():
    """Canonical count architectures, interleaved by augmentation ranks."""
    from itertools import product
    types = tuple(product(range(1, 5), repeat=2))

    def library(length, row_limits, col_limits):
        result = []
        for columns in combinations_with_replacement(types, length):
            exchanged = tuple(sorted((b, a) for a, b in columns))
            if columns > exchanged:
                continue
            rows = tuple(sum(c[i] for c in columns) for i in range(2))
            cols = tuple(a + b for a, b in columns)
            if not row_limits[0] <= min(rows) <= max(rows) <= row_limits[1]:
                continue
            if not col_limits[0] <= min(cols) <= max(cols) <= col_limits[1]:
                continue
            grid = tuple(tuple(c[i] for c in columns) for i in range(2))
            parity = [sum((v & 1) << j for j, v in enumerate(row))
                      for row in grid]
            rank = gf2_rank(parity)
            result.append((grid, rows, cols, rank))
        return result

    alib = library(3, (5, 8), (3, 6))
    blib = library(4, (6, 9), (2, 5))
    buckets = {}
    for a, ar, ac, ka in alib:
        for b, br, bc, kb in blib:
            if max(ar) + max(bc) > W_MAX:
                continue
            if max(br) + max(ac) > W_MAX:
                continue
            checks = tuple(sorted(
                [x+y for x in ar for y in bc] +
                [x+y for x in br for y in ac]))
            qubits = tuple(sorted(
                [x+y for x in ac for y in bc] +
                [x+y for x in ar for y in br]))
            priority = (min(ac+bc), sum(ar)+sum(br),
                        -abs(ar[0]-ar[1])-abs(br[0]-br[1]),
                        -max(ac+bc), a, b)
            buckets.setdefault((ka, kb), []).append(
                (priority, (checks, qubits, ka, kb), a, b))
    for entries in buckets.values():
        entries.sort(reverse=True)
    strata = sorted(buckets, key=lambda s: (sum(s), max(s), s))
    positions = {s: 0 for s in strata}
    cards, seen = [], set()
    while True:
        added = False
        for s in strata:
            entries = buckets[s]
            while positions[s] < len(entries):
                _, signature, a, b = entries[positions[s]]
                positions[s] += 1
                if signature in seen:
                    continue
                seen.add(signature)
                cards.append((a, b))
                added = True
                break
        if not added:
            break
    if len(cards) < 20:
        raise ValueError("fewer than twenty support architectures")
    return tuple(cards)


def _c37_columns(rows, width):
    cols = [0] * width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            cols[bit.bit_length()-1] |= 1 << r
            source ^= bit
    return cols


def _c37_components(hx, hz, q):
    """Extract component maps only from the verified complete lift."""
    mask = (1 << q)-1
    ar = [
        sum(((hx[(i*4)*q+r] >> (j*4*q)) & mask) << (j*q)
            for j in range(3))
        for i in range(2) for r in range(q)]
    bm = (1 << (4*q))-1
    br = [hz[a*q+r] & bm for a in range(2) for r in range(q)]
    return ar, br


def _c37_syndrome(word, cols):
    value = 0
    while word:
        bit = word & -word
        value ^= cols[bit.bit_length()-1]
        word ^= bit
    return value


def _c37_cofactors(G, grid, cols, inverse=False):
    """Commutative minors construct words, never an alternative lift."""
    from itertools import combinations
    entries = [[set(G.inv(g) if inverse else g for g in cell)
                for cell in row] for row in grid]

    def multiply(a, b):
        result = set()
        for g in a:
            for h in b:
                term = G.mul(g, h)
                if term in result:
                    result.remove(term)
                else:
                    result.add(term)
        return result

    seeds = set()
    for chosen in combinations(range(len(grid[0])), 3):
        minors = []
        for omitted in range(3):
            s, t = [chosen[j] for j in range(3) if j != omitted]
            minors.append(multiply(entries[0][s], entries[1][t])
                          ^ multiply(entries[0][t], entries[1][s]))
        for shift in G.elements[::7]:
            word = 0
            for j, minor in zip(chosen, minors):
                for g in minor:
                    word ^= 1 << (j*G.q + G.idx(G.mul(g, shift)))
            if word and _c37_syndrome(word, cols) == 0:
                seeds.add(word)
    return sorted(seeds, key=lambda w: (w.bit_count(), w))[:24]


def _c37_probe(cols, memory, rng, trials):
    """Retained actual kernel witnesses; sampled misses are not proofs."""
    width = len(cols)
    memory[:] = sorted(
        {w for w in memory if w and _c37_syndrome(w, cols) == 0},
        key=lambda w: (w.bit_count(), w))[:24]
    best = min((w.bit_count() for w in memory), default=width+1)
    order = list(range(width))
    for _ in range(trials):
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
                best = min(best, word.bit_count())

        for sweep in range(2):
            donors = sorted(set(words+memory), key=int.bit_count)[:24]
            rng.shuffle(donors)
            reduced = []
            for word in words:
                for donor in donors:
                    trial = word ^ donor
                    if trial and trial.bit_count() < word.bit_count():
                        word = trial
                reduced.append(word)
                best = min(best, word.bit_count())
            words = reduced

        short = sorted(set(words+memory), key=int.bit_count)[:32]
        discovered = list(short)
        for i, word in enumerate(short):
            for other in short[:i]:
                trial = word ^ other
                if trial:
                    best = min(best, trial.bit_count())
                    if trial.bit_count() <= best+4:
                        discovered.append(trial)
        memory[:] = sorted(set(memory+discovered),
                           key=lambda w: (w.bit_count(), w))[:24]
        if best <= 8:
            break
    return best


def _c37_basis(rows):
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


def _c37_outside(word, basis):
    while word:
        p = word.bit_length()-1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _c37_logical_reading(item, rng):
    """Syndrome-checked embeddings, dressings and mixed-sector witnesses."""
    hx, hz, n, q = item["hx"], item["hz"], item["n"], 37
    mask = (1 << q)-1
    best = n+1

    def search(checks, stabilizers, seeds, side):
        nonlocal best
        cols = _c37_columns(checks, n)
        span = _c37_basis(stabilizers)

        def observe(word):
            nonlocal best
            if not word or _c37_syndrome(word, cols):
                return
            if not _c37_outside(word, span):
                return
            best = min(best, word.bit_count())
            for _ in range(3):
                chosen, weight = word, word.bit_count()
                nearby = []
                for row in stabilizers:
                    trial = word ^ row
                    tw = trial.bit_count()
                    nearby.append((tw, row))
                    if tw < weight:
                        chosen, weight = trial, tw
                nearby.sort(key=lambda p: p[0])
                directions = [row for _, row in nearby[:10]]
                for i, row in enumerate(directions):
                    for other in directions[:i]:
                        trial = word ^ row ^ other
                        tw = trial.bit_count()
                        if tw < weight:
                            chosen, weight = trial, tw
                if chosen == word:
                    break
                word = chosen
                best = min(best, weight)

        for seed in sorted(set(seeds), key=int.bit_count)[:8]:
            if side == "A":
                for beta in range(4):
                    observe(sum(((seed >> (j*q)) & mask)
                                << ((j*4+beta)*q) for j in range(3)))
            else:
                for j in range(3):
                    observe(seed << (j*4*q))

        order = list(range(n))
        for _ in range(4):
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
            logical = sorted(
                (w for w in dependencies if _c37_outside(w, span)),
                key=int.bit_count)[:8]
            donors = sorted(dependencies, key=int.bit_count)[:32]
            for word in logical:
                for _ in range(2):
                    chosen = word
                    for donor in donors:
                        trial = word ^ donor
                        if (trial and trial.bit_count() < chosen.bit_count()
                                and _c37_outside(trial, span)):
                            chosen = trial
                    if chosen == word:
                        break
                    word = chosen
                observe(word)

    search(hx, hz, item["aw"], "A")
    search(hz, hx, item["bw"], "B")
    return best


def _c37_fingerprint(hx, hz, n):
    """Uncoloured Tanner invariant; equality is rejected conservatively."""
    from collections import Counter
    rows = list(hx) + list(hz)
    weights = [row.bit_count() for row in rows]
    adjacency = [0] * len(rows)
    links = [{} for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
                links[i][j] = links[j][i] = overlap
    profiles = []
    for i, neighbors in enumerate(links):
        profile = Counter(
            (weights[j], overlap,
             (adjacency[i] & adjacency[j]).bit_count())
            for j, overlap in neighbors.items())
        profiles.append((weights[i], tuple(sorted(profile.items()))))
    degrees = [0] * n
    for source in rows:
        while source:
            bit = source & -source
            degrees[bit.bit_length()-1] += 1
            source ^= bit
    return (n, tuple(sorted(Counter(profiles).items())),
            tuple(sorted(Counter(degrees).items())))


@lru_cache(maxsize=1)
def _quotient_design_bank(G):
    """New C37 bank: deliberate support cards, no archived-cell fallback."""
    assert G.q == 37
    cards = _c37_support_cards()
    emitted, bank = set(), []

    def draw_grid(local, counts, mode):
        used, grid = set(), []
        unit = local.randrange(1, 37)
        for i, sizes in enumerate(counts):
            row = []
            for j, count in enumerate(sizes):
                best_cell, best_cost = None, None
                for attempt in range(32 if mode == 3 else 1):
                    if mode == 2:
                        base, step = local.randrange(37), local.randrange(1, 37)
                        cell = [((base+t*step) % 37,) for t in range(count)]
                    elif mode == 1 and count == 2:
                        base = local.randrange(37)
                        gap = unit * local.choice((1, 2, 4, 7, 11, 16)) % 37
                        cell = [(base,), ((base+gap) % 37,)]
                    else:
                        cell = local.sample(G.elements, count)
                    differences = {
                        G.mul(g, G.inv(h))
                        for g in cell for h in cell if g != h}
                    cost = len(differences & used)
                    if best_cost is None or cost < best_cost:
                        best_cell, best_cost = cell, cost
                    if cost == 0:
                        break
                row.append(best_cell)
                used.update(G.mul(g, G.inv(h))
                            for g in best_cell for h in best_cell if g != h)
            grid.append(row)
        return grid

    def merit(item):
        d = min(item["da"], item["db"], item.get("logical", 593))
        return (item["k"]*d*d, d, item["da"]+item["db"],
                -item["cycles"])

    for slot in range(N_SLOTS):
        # Twenty primary tasks use twenty support architectures.
        # Additional batches alter the relative-label policy and stream.
        ta, tb = cards[slot % len(cards)]
        mode = (slot % 4 + slot//20) % 4
        local = random.Random(BASE_SEED*100003 + 37015319 + 104729*slot)
        assert _grid_weights(ta, tb) <= W_MAX
        assert _no_dead_lines(ta, tb)
        finalists, seen = [], set()
        chosen = None

        def admit(A, B):
            labels = tuple(tuple(sorted(cell))
                           for grid in (A, B) for row in grid for cell in row)
            if labels in seen:
                return
            seen.add(labels)
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (125*n <= 1000*k < 175*n and k >= 74):
                return
            ar, br = _c37_components(hx, hz, G.q)
            ac, bc = _c37_columns(ar, 111), _c37_columns(br, 148)
            if not (_no_kernel_le3(ar, 111) and _no_kernel_le3(br, 148)):
                return
            aw = _c37_cofactors(G, A, ac)
            bw = _c37_cofactors(G, B, bc, inverse=True)
            item = dict(A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                        ac=ac, bc=bc, aw=aw, bw=bw,
                        da=_c37_probe(ac, aw, local, 3),
                        db=_c37_probe(bc, bw, local, 3),
                        cycles=_quotient_incidence(ar)[1]
                               + _quotient_incidence(br)[1])
            finalists.append(item)
            finalists.sort(key=merit, reverse=True)
            del finalists[6:]

        for batch in range(2):
            for attempt in range(56 + 24*batch):
                proposal_mode = mode if attempt % 7 else 0
                admit(draw_grid(local, ta, proposal_mode),
                      draw_grid(local, tb, proposal_mode))

            # Change individual voltages where retained short words live.
            # The count architecture remains fixed throughout the slot.
            for step in range(24):
                if not finalists:
                    break
                parent = local.choice(finalists[:min(3, len(finalists))])
                A = [[list(cell) for cell in row] for row in parent["A"]]
                B = [[list(cell) for cell in row] for row in parent["B"]]
                side = "A" if parent["da"] <= parent["db"] else "B"
                if step % 4 == 0:
                    side = local.choice(("A", "B"))
                grid, counts = (A, ta) if side == "A" else (B, tb)
                seeds = parent["aw"] if side == "A" else parent["bw"]
                pressure = [
                    1 + sum(((w >> (j*37)) & ((1 << 37)-1)).bit_count()
                            for w in seeds[:8])
                    for j in range(len(counts[0]))]
                j = local.choices(range(len(pressure)), weights=pressure, k=1)[0]
                i = local.randrange(2)
                replacement = draw_grid(local, counts, (mode+step//8) % 4)
                grid[i][j] = replacement[i][j]
                if step % 6 == 0:
                    other = local.randrange(len(pressure))
                    grid[1-i][other] = replacement[1-i][other]
                admit(A, B)

            # Earlier lighter witnesses survive every independent rescreen.
            for item in finalists:
                item["da"] = min(item["da"], _c37_probe(
                    item["ac"], item["aw"], local, 20))
                item["db"] = min(item["db"], _c37_probe(
                    item["bc"], item["bw"], local, 20))
            finalists.sort(key=merit, reverse=True)
            finalists = finalists[:3]
            for item in finalists:
                item["da"] = min(item["da"], _c37_probe(
                    item["ac"], item["aw"], local, 64))
                item["db"] = min(item["db"], _c37_probe(
                    item["bc"], item["bw"], local, 64))
                item["logical"] = _c37_logical_reading(item, local)

            for item in sorted(finalists, key=merit, reverse=True):
                hx, hz, n = item["hx"], item["hz"], item["n"]
                if not (_no_kernel_le3(hx, n) and _no_kernel_le3(hz, n)):
                    continue
                signature = _c37_fingerprint(hx, hz, n)
                if signature in emitted:
                    continue
                assert css_ok(hx, hz)
                assert n == 592 and item["k"] >= 74
                assert 125*n <= 1000*item["k"] < 175*n
                assert max(row.bit_count() for row in hx+hz) <= W_MAX
                assert all(1 <= len(cell) <= CELL_MAX
                           and len(cell) == len(set(cell))
                           for grid in (item["A"], item["B"])
                           for row in grid for cell in row)
                emitted.add(signature)
                chosen = (
                    tuple(tuple(tuple(cell) for cell in row)
                          for row in item["A"]),
                    tuple(tuple(tuple(cell) for cell in row)
                          for row in item["B"]),
                    merit(item), item["k"])
                break
            if chosen is not None:
                break
        # Failed cards never emit an earlier code or an orbit fallback.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return the independently optimized two-row support design."""
    chosen = _quotient_design_bank(G)[idx]
    if chosen is None:
        raise ValueError("no fresh C37 design passed target-rate screening")
    A, B, score, k = chosen
    return {"idx": idx, "cells": {"A": A, "B": B},
            "selection": score, "k": k}


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh complete cell without changing the screened matrices."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Uniform legal support rolling on C17, (2,5)x(2,6), did not suppress sector-1 weight-five logicals: all five measured parent codes had d30s=5.
# REFUTED: Fifteen prescribed odd-central deletion designs supply fifteen inequivalent codes: ten principal slots were isomorphic duplicates, and all five measured survivors remained at d30s=4.
# REFUTED: Degeneracy-aware support and rank-escape screening improves the S3 x C4 high-rate bed beyond 9.429; all five measured current codes instead had d30s=3 and fitness 5.1964.
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

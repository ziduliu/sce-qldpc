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
CONCEPT = ("C7 semidirect C6 with action 2, order 42, shapes (1,3)x(1,4): quotient-balanced singular A trinomials and bipartite B binomial subgraphs, followed by complete logical-five exclusion, should produce five fresh cell-0.500 leaders with measured distance readings at least six.")
GROUP = ("Jump from D8 to C7 semidirect C6, action 2, isomorphic to F21 x C2. Odd-parity noncentral binomial differences have orders six or fourteen, not five. The matching order-42 centre-two/ten-class (1,3)x(1,4) bed previously scored zero; this fresh construction tests an algebraic mechanism rather than presuming an untried-group advantage. All screening actions come from verified lifts.")
SHAPES = ("Target cell 0.500 with (1,3)x(1,4): n=546, kb=252, guaranteed rate 6/13, and component lengths/check counts 126/42 and 168/42. These shapes allow q<=46; the family fixes q=42. Quotient-balanced A assemblies target ranks 38..40, while connected binomials plus an odd cubic make B full-rank. Complete lifted ranks enforce 0.475<=k/n<0.525. Parent weight-four minima were entirely sector-one Acol/Bcol words, spread across both components.")
FILLING = ("Twenty policies cross five A quotient/parity architectures with four B cubic geometries. A=[3,3,3] and B=[2,2,2,3] give HX maximum 9+3=12 and HZ maximum 9+3=12. Enumerate inverse-canonical odd-parity B generators, require connected four-free binomial graphs, and optimize fresh cubic extensions. Require component exclusion through four; prefer exact B exclusion through five, with fresh through-four reserves. Independently deepen actual constituent witnesses, then exhaust complete CSS logicals through five modulo stabilizers. Rank and certified exclusion floors select candidates; high sampled readings remain heuristics. Unequal uncoloured Tanner invariants certify emitted diversity.")
FAMILY_NAME = "r050_C7semiC6_action2_13x14_balanced_bipartite_exact5_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    # x^7=y^6=1; y*x*y^-1=x^2, and 2^3=1 modulo 7.
    # y^3 is central of order two: this is F21 x C2.
    if not valid_q(q):
        raise ValueError("C7 semidirect C6 requires order 42")
    els = [(a, b) for b in range(6) for a in range(7)]
    action = [pow(2, b, 7) for b in range(6)]

    def red(g):
        return (g[0] % 7, g[1] % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + action[b] * c) % 7, (b + d) % 6)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 6
        return ((-action[bi] * a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


from functools import lru_cache

# Inactive historical D8 support helpers retained below.
# The active order-42 optimizer does not call these parent recipes.
# Its counts, labels, screens and diversity gates are defined jointly.
_RANKSHIFT_PLANS = (
    ((2, 1),),
    ((0, 3),),
    ((2, 3),),
    ((2, 1), (2, 1)),
    ((2, 1), (0, 3)),
    ((2, 1), (1, 3)),
    ((0, 3), (3, 1)),
    ((0, 1), (2, 1), (2, 1)),
    ((0, 1), (2, 3)),
    ((0, 1), (2, 1), (1, 3)),
)


def _rankshift_cell(rng, G, size, model):
    """Different relative-element orders, not translated orbit copies."""
    if size == 0:
        return []
    if size == 1:
        return [G.pick(rng)]
    base = G.pick(rng)
    if size == 2:
        # Rotation differences have order four; reflection differences
        # have order two. The two models change actual incidence cycles.
        difference = ((rng.choice((1, 3)), 0) if model == 0
                      else (rng.randrange(4), 1))
        relative = [G.e, difference]
    elif model == 0:
        relative = [(a, 0) for a in rng.sample(range(4), 3)]
    else:
        relative = [G.e, (rng.choice((1, 3)), 0),
                    (rng.randrange(4), 1)]
    cell = [G.mul(base, g) for g in relative]
    assert len({G._key(g) for g in cell}) == size
    return cell


def _rankshift_B(rng, G, counts, model):
    B = [[None] * 8 for _ in range(2)]
    relatives = list(G.elements)
    rng.shuffle(relatives)
    mono = [j for j, pair in enumerate(counts) if pair == (1, 1)]
    for j, relative in zip(mono, relatives):
        B[0][j], B[1][j] = [G.e], [G.inv(relative)]
    for j, pair in enumerate(counts):
        if pair == (1, 1):
            continue
        for i in range(2):
            B[i][j] = _rankshift_cell(rng, G, pair[i], model)
        # Remove the common right column factor instead of spending
        # candidate diversity on a monomial gauge.
        anchor = (B[0][j] or B[1][j])[0]
        inverse = G.inv(anchor)
        for i in range(2):
            B[i][j] = [G.mul(g, inverse) for g in B[i][j]]
    return B


def _rankshift_profile(rows, n):
    """Exact exclusion through weight three and weight-four census.

    After nonzero distinct singles and absence of single/pair collisions,
    equal pair syndromes have disjoint supports. Each weight-four kernel
    support contributes exactly three collisions. Kernel words need not
    be logical operators.
    """
    columns = [0] * n
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= check
            row ^= bit
    singles = set(columns)
    if 0 in singles or len(singles) != n:
        return None
    pairs = {}
    collisions = 0
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome in singles:
                return None
            previous = pairs.get(syndrome, 0)
            collisions += previous
            pairs[syndrome] = previous + 1
    histogram = {}
    for syndrome, count in pairs.items():
        key = (count, syndrome.bit_count())
        histogram[key] = histogram.get(key, 0) + 1
    return (collisions // 3, tuple(sorted(histogram.items())),
            tuple(c.bit_count() for c in columns))


def _rankshift_fingerprint(hx, hz, px, pz, k):
    """Invariant under qubit/check permutations and global X/Z exchange.

    Differing fingerprints cannot be orbit copies under those actions.
    Equal fingerprints are conservatively discarded, even when the
    underlying codes might actually differ.
    """
    types = tuple(sorted((
        (px[0], px[1], tuple(sorted(r.bit_count() for r in hx))),
        (pz[0], pz[1], tuple(sorted(r.bit_count() for r in hz))),
    )))
    joint = tuple(sorted(zip(px[2], pz[2])))
    swapped = tuple(sorted(zip(pz[2], px[2])))
    return k, types, min(joint, swapped)


@lru_cache(maxsize=1)
def _quotient42_bank(G):
    """Pure deterministic bank of fresh, fully rank-screened designs."""
    from collections import Counter
    from itertools import combinations

    assert G.q == 42
    q = G.q
    mask = (1 << q) - 1
    stub_a = [[[G.e] for _ in range(3)]]
    stub_b = [[[G.e] for _ in range(4)]]

    def columns(rows, width):
        out = [0] * width
        for r, original in enumerate(rows):
            row = original
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return tuple(out)

    def span(words):
        pivots = {}
        for original in words:
            word = original
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = word
                    break
                word ^= old
        return pivots

    def outside(word, pivots):
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def clean4_table(cols):
        """Exact exclusion of all nonzero constituent kernels through four."""
        seen = {0: 0}
        bits = [1 << i for i in range(len(cols))]
        for i, value in enumerate(cols):
            if value in seen:
                return None
            seen[value] = bits[i]
        for i, value in enumerate(cols):
            for j in range(i):
                syndrome = value ^ cols[j]
                if syndrome in seen:
                    return None
                seen[syndrome] = bits[i] | bits[j]
        return seen

    def clean5_component(cols, table):
        """Exact constituent exclusion through five, after clean4.

        Each constituent has a transitive regular action on each column
        fibre. Translate any five-support to include the identity point
        of one fibre, then exhaust its anchored triple / pair partitions.
        This argument is NOT applied to the complete nonabelian LP code.
        """
        width = len(cols)
        bits = [1 << i for i in range(width)]
        anchors = range(G.idx(G.e), width, q)
        for a in anchors:
            for b in range(width):
                if b == a:
                    continue
                value = cols[a] ^ cols[b]
                word = bits[a] | bits[b]
                for c in range(b):
                    if c == a:
                        continue
                    old = table.get(value ^ cols[c])
                    if old is not None:
                        witness = word ^ bits[c] ^ old
                        if witness:
                            assert witness.bit_count() <= 5
                            return False
        return True

    def cycles(cols):
        total = 0
        for i, value in enumerate(cols):
            for other in cols[:i]:
                overlap = (value & other).bit_count()
                total += overlap * (overlap - 1) // 2
        return total

    def probe(cols, local, trials, seeds=()):
        """Actual nonzero kernel witnesses; high readings are heuristics."""
        memory = sorted(set(seeds),
                        key=lambda w: (w.bit_count(), w))[:24]
        best = min((w.bit_count() for w in memory), default=len(cols) + 1)
        order = list(range(len(cols)))
        for _ in range(trials):
            local.shuffle(order)
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
                    best = min(best, word.bit_count())
            for sweep in range(2):
                reducers = sorted(set(words + memory),
                                  key=int.bit_count)[:24]
                local.shuffle(reducers)
                for i, word in enumerate(words):
                    weight = word.bit_count()
                    for other in reducers:
                        trial = word ^ other
                        tw = trial.bit_count()
                        if 0 < tw < weight:
                            word, weight = trial, tw
                    words[i] = word
                    best = min(best, weight)
            short = sorted(set(words + memory),
                           key=lambda w: (w.bit_count(), w))[:32]
            discoveries = list(short)
            for i, word in enumerate(short):
                for other in short[:i]:
                    trial = word ^ other
                    if trial:
                        best = min(best, trial.bit_count())
                        discoveries.append(trial)
            memory = sorted(set(discoveries + memory),
                            key=lambda w: (w.bit_count(), w))[:24]
            if best <= 5:
                break
        assert memory and 0 < best <= len(cols)
        return best, tuple(memory)

    # Compile component actions only from scaffold-generated unit lifts.
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            if side == "A":
                unit = [[[g], [], []]]
                hx, hz, _ = lifted_matrices(G, unit, stub_b)
                source = hx
            else:
                unit = [[[g], [], [], []]]
                hx, hz, _ = lifted_matrices(G, stub_a, unit)
                source = hz
            blocks.append(columns([r & mask for r in source[:q]], q))
        atlas[side] = tuple(blocks)

    def block(side, cell):
        result = [0] * q
        for g in cell:
            contribution = atlas[side][G.idx(g)]
            for h, value in enumerate(contribution):
                result[h] ^= value
        return tuple(result)

    def component(side, cells):
        return tuple(value for cell in cells for value in block(side, cell))

    # Odd b defines the C2 bipartition. Remove central involutions and
    # inverse copies before assembling the three-binomial graph.
    generators = {}
    for g in G.elements:
        if g[1] % 2 != 1 or G.mul(g, g) == G.e:
            continue
        inverse = G.inv(g)
        key = min(G.idx(g), G.idx(inverse))
        generators.setdefault(key, min((g, inverse), key=G.idx))

    graphs, graph_seen = [], set()
    for gs in combinations(generators.values(), 3):
        cells = tuple((G.e, g) for g in gs)
        cols = component("B", cells)
        incidence = tuple(sorted(cols))
        if incidence in graph_seen:
            continue
        if gf2_rank(cols) != q - 1 or clean4_table(cols) is None:
            continue
        graph_seen.add(incidence)
        normal_edges = sum(g[1] == 3 for g in gs)
        graphs.append((normal_edges, cells, cols))
    if not graphs:
        raise ValueError("no connected four-free odd-parity binomial graph")

    # Four genuinely different quotient/parity cubic geometries.
    b_models = ((0, 4, 2), (0, 1, 2), (0, 0, 4), (0, 4, 4))
    cubic_libraries = []
    for bs in b_models:
        library, seen = [], set()
        for u in range(7):
            for v in range(7):
                cell = (G.e, (u, bs[1]), (v, bs[2]))
                if len(set(cell)) != 3:
                    continue
                g, h = cell[1:]
                if G.mul(g, h) == G.mul(h, g):
                    continue
                cols = block("B", cell)
                incidence = tuple(sorted(cols))
                if incidence in seen or clean4_table(cols) is None:
                    continue
                seen.add(incidence)
                library.append((cell, cols))
        cubic_libraries.append(tuple(library))
    all_cubics = tuple(item for library in cubic_libraries for item in library)
    if not all_cubics:
        raise ValueError("no clean noncommuting cubic extension")

    a_models = ((0, 4, 2), (0, 1, 2), (0, 4, 5), (0, 1, 5))
    a_cards = ((0, 0, 0), (0, 0, 1), (0, 1, 2),
               (1, 1, 3), (1, 2, 3))

    def draw_a(local, kinds):
        cells = []
        for kind in kinds:
            bs = a_models[kind]
            for retry in range(64):
                g = (local.randrange(7), bs[1])
                h = (local.randrange(7), bs[2])
                if G.mul(g, h) != G.mul(h, g):
                    cells.append((G.e, g, h))
                    break
            else:
                raise ValueError("noncommuting balanced A recipe exhausted")
        return tuple(cells)

    def select(local, side, policy, mode, batch):
        pool, seen = [], set()
        quota = 160 + 48 * batch
        preferred = [g for g in graphs
                     if g[0] == (0 if policy in (0, 3) else 1)]
        if not preferred:
            preferred = graphs
        library = cubic_libraries[mode] or all_cubics

        for attempt in range(quota):
            if side == "A":
                cells = draw_a(local, a_cards[policy])
                cols = component("A", cells)
                rank = gf2_rank(cols)
                if not 38 <= rank <= 40:
                    continue
            else:
                graph = local.choice(
                    preferred if attempt < quota // 2 else graphs)
                cubic, extra = local.choice(library)
                cells = graph[1] + (cubic,)
                cols = graph[2] + extra
                rank = gf2_rank(cols)
                if rank != q:
                    continue
            labels = tuple(sorted(tuple(sorted(c)) for c in cells))
            if labels in seen:
                continue
            seen.add(labels)
            table = clean4_table(cols)
            if table is None:
                continue
            # A columns all have odd weight three, excluding odd kernels.
            clean5 = (True if side == "A"
                      else clean5_component(cols, table))
            reading, words = probe(cols, local, 3)
            item = dict(cells=cells, cols=cols, rank=rank,
                        clean5=clean5, d=reading, words=words,
                        cycles=cycles(cols), attempt=attempt)
            pool.append(item)
            pool.sort(key=lambda x: (
                x["clean5"], x["d"], -x["cycles"], -x["attempt"]),
                reverse=True)
            del pool[6:]

        for item in pool:
            reading, words = probe(
                item["cols"], local, 16, item["words"])
            item["d"] = min(item["d"], reading)
            item["words"] = words
        pool.sort(key=lambda x: (
            x["clean5"], x["d"], -x["cycles"]), reverse=True)
        return pool[:4]

    def logical_floor(checks, stabilizers, n):
        """Exact exclusion through four, then complete logical-five search.

        A single tail representative per syndrome suffices after every
        equal-syndrome difference of weight <=4 has been shown stabilizing.
        No transitive-action assumption is used for the complete CSS code.
        Returns 0 on a logical through four, 5 on a found five-logical,
        and 6 only after exhaustive exclusion through five.
        """
        cols = columns(checks, n)
        pivots = span(stabilizers)
        bits = [1 << i for i in range(n)]
        tails = {0: 0}
        incidence = [[] for _ in checks]
        for i, value in enumerate(cols):
            old = tails.get(value)
            if old is not None:
                if outside(bits[i] ^ old, pivots):
                    return 0
            else:
                tails[value] = bits[i]
            remaining = value
            while remaining:
                bit = remaining & -remaining
                incidence[bit.bit_length() - 1].append(i)
                remaining ^= bit
        for i, value in enumerate(cols):
            for j in range(i):
                syndrome = value ^ cols[j]
                word = bits[i] | bits[j]
                old = tails.get(syndrome)
                if old is not None:
                    if outside(word ^ old, pivots):
                        return 0
                else:
                    tails[syndrome] = word
        max_degree = max(value.bit_count() for value in cols)

        def visit(value, remaining, word):
            if not value:
                return outside(word, pivots)
            if value.bit_count() > max_degree * remaining:
                return False
            if remaining == 2:
                tail = tails.get(value)
                return tail is not None and outside(word ^ tail, pivots)
            check = (value & -value).bit_length() - 1
            for j in incidence[check]:
                if word & bits[j]:
                    continue
                if visit(value ^ cols[j], remaining - 1, word ^ bits[j]):
                    return True
            return False

        return 5 if any(visit(value, 4, bits[i])
                        for i, value in enumerate(cols)) else 6

    def fingerprint(hx, hz, n):
        """Uncoloured incidence invariant: inequality proves nonisomorphism."""
        rows = list(hx) + list(hz)
        weights = [r.bit_count() for r in rows]
        adjacency = [0] * len(rows)
        overlaps = [dict() for _ in rows]
        for i, row in enumerate(rows):
            for j in range(i):
                size = (row & rows[j]).bit_count()
                if size:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
                    overlaps[i][j] = overlaps[j][i] = size
        profiles = []
        for i, neighbors in enumerate(overlaps):
            local = Counter(
                (weights[j], size,
                 (adjacency[i] & adjacency[j]).bit_count())
                for j, size in neighbors.items())
            profiles.append((weights[i], tuple(sorted(local.items()))))
        qubits = [[] for _ in range(n)]
        for r, original in enumerate(rows):
            row = original
            while row:
                bit = row & -row
                qubits[bit.bit_length() - 1].append(weights[r])
                row ^= bit
        return (n, tuple(sorted(Counter(profiles).items())),
                tuple(sorted(Counter(
                    tuple(sorted(v)) for v in qubits).items())))

    bank, emitted = [], set()
    for slot in range(N_SLOTS):
        policy, mode = divmod(slot % 20, 4)
        winner = None
        for batch in range(3):
            local = random.Random(
                BASE_SEED * 100009 + 42050019
                + 104729 * slot + 8191 * batch)
            ap = select(local, "A", policy, mode, batch)
            bp = select(local, "B", policy, mode, batch)
            proposals = []
            for a in ap:
                for b in bp:
                    A, B = [[list(c) for c in a["cells"]]], [
                        [list(c) for c in b["cells"]]]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if not (475 * n <= 1000 * k < 525 * n):
                        continue
                    # Independently verify the atlas extraction against
                    # the assembled scaffold matrices before selection.
                    ar = [
                        sum(((row >> (4*j*q)) & mask) << (j*q)
                            for j in range(3)) for row in hx[:q]]
                    br = [row & ((1 << (4*q)) - 1) for row in hz[:q]]
                    assert columns(ar, 3*q) == a["cols"]
                    assert columns(br, 4*q) == b["cols"]
                    key = (b["clean5"], min(a["d"], b["d"]),
                           k, -a["cycles"] - b["cycles"])
                    proposals.append((key, A, B, hx, hz, n, k))
            proposals.sort(key=lambda x: x[0], reverse=True)

            tested = 0
            for key, A, B, hx, hz, n, k in proposals:
                fp = fingerprint(hx, hz, n)
                if fp in emitted:
                    continue
                fx = logical_floor(hz, hx, n)
                if fx == 0:
                    continue
                fz = logical_floor(hx, hz, n)
                if fz == 0:
                    continue
                floor = min(fx, fz)
                objective = (k * floor * floor, floor, key[1], key[3])
                if winner is None or objective > winner[0]:
                    winner = (objective, A, B, hx, hz, k, fp, floor)
                tested += 1
                if floor == 6 or tested >= 8:
                    break
            if winner is not None and winner[-1] == 6:
                break

        if winner is None:
            bank.append(None)
            continue
        objective, A, B, hx, hz, k, fp, floor = winner
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        assert _grid_weights(sa, sb) == 12
        assert _no_dead_lines(sa, sb)
        assert max(r.bit_count() for r in hx + hz) <= W_MAX
        assert css_ok(hx, hz)
        assert all(len(c) == len(set(c)) <= 5
                   for grid in (A, B) for row in grid for c in row)
        emitted.add(fp)
        bank.append({
            "cells": {
                side: tuple(tuple(tuple(c) for c in row) for row in grid)
                for side, grid in (("A", A), ("B", B))},
            "k": k, "card": slot % 20,
            "verified_exclusion_floor": floor,
            "selection": objective})
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign a fresh balanced/bipartite design, never an orbit fallback."""
    selected = _quotient42_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh order-42 half-rate design passed screening")
    return selected


def CELL(rng, G, side, i, j, ctx):
    """Return the jointly screened full cell without additional draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Distinct monomial voltages plus extra A terms on D8 (2,7)x(2,8) raised the measured distance above four; all five measured parent codes had d30s=4, with thousands of sector-1 weight-four witnesses.
# REFUTED: Moving the connected monomial component to A and making B full-rank removes the D8 weight-four plateau; all five measured rankshift leaders had k=246 and d30s=4, with thousands of sector-one weight-four witnesses.
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

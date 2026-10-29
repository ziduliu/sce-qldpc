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
W_MAX = int(os.environ.get("CM_WMAX", "9"))
CELL_MAX = 5
N_SLOTS = int(os.environ.get("CM_NSLOTS", "20"))
BASE_SEED = 20260903


# EVOLVE-BLOCK-START
CONCEPT = ("Dic3, order 12, A=(2,5), B=(3,8): independently optimized degree-two/three constituents with controlled rank deficits should supply five fresh finalists exceeding cell 0.350's current fitness of about 12.3; distance survival is a falsifiable measurement hypothesis.")
GROUP = ("Preserve the parent's dicyclic presentation and change only its order. The Dic11 order-44 bed leads cell 0.100 at 46.5, but factor 46 makes that order inadmissible. The new shapes permit q<=13; Dic3, order 12, is the largest dicyclic lift that fits. Its normal C3 and order-four cosets retain the same construction family, not a transferred distance guarantee.")
SHAPES = ("Change to (2,5)x(3,8): constituents have 60 variables/24 checks and 96 variables/36 checks, n=552, kb=180 and guaranteed rate 15/46=0.326087. Exact complete ranks retain 0.325<=k/n<0.375. Small constituent rank deficits are proposals for moving above the lower edge, not predictions of k. Parent minima mostly occupied sector-1 Acol/Bcol blocks; slot 42 also had mixed-sector weight-20 Z witnesses.")
FILLING = ("Twenty support/relative-element cards give every constituent row six terms, exactly two degree-three columns and all remaining columns degree two. BOTH CSS maxima are nine. Independently select constituents using verified lifts, exact kernel searches through five and equal-budget information-set witness refinement; weight-five discoveries are carried forward rather than hidden. Complete pairs enforce actual target-band k, CSS and exact logical-four exclusion, then refine explicitly nontrivial logical witnesses including mixed sectors. Different full uncoloured Tanner invariants conservatively separate published slots. Larger witness readings and unseen-archive novelty remain hypotheses.")
FAMILY_NAME = "r035_dicyclic_dic3_25x38_degree23_rankwitness_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 12

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic3 has order 12")
    # SAME dicyclic rule: x^(2*t)=1, y^2=x^t, y*x*y^-1=x^-1.
    # Only t changes, from 11 to 3; element encoding remains x^a y^b.
    t, m = 3, 6
    els = [(a, b) for b in range(2) for a in range(m)]

    def red(g):
        a, b = g
        return ((a + t * (b // 2)) % m, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + t * b * d) % m,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % m, 0) if b == 0 else ((a + t) % m, 1)

    def idx(g):
        a, b = red(g)
        return m * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (3, 8)               # (rb, cb)


from functools import lru_cache


@lru_cache(maxsize=1)
def _r035_catalogue(G, count):
    """Immutable deterministic portfolio; caching changes cost only."""
    from itertools import product, combinations_with_replacement, permutations
    from collections import Counter
    from hashlib import sha256

    assert G.q == 12
    cap = min(9, W_MAX)
    if cap < 9:
        raise ValueError("degree-two/three portfolio requires cap nine")
    q = G.q

    def support_library(rows, cols):
        """Enumerate incidence prescriptions modulo row/column gauges."""
        types = {
            d: [v for v in product(range(4), repeat=rows) if sum(v) == d]
            for d in (2, 3)
        }
        keys = set()
        for low in combinations_with_replacement(types[2], cols - 2):
            for high in combinations_with_replacement(types[3], 2):
                columns_ = low + high
                if any(sum(v[i] for v in columns_) != 6
                       for i in range(rows)):
                    continue
                parity_rows = [
                    sum((v[i] & 1) << j
                        for j, v in enumerate(columns_))
                    for i in range(rows)
                ]
                # Avoid forced augmentation dependencies. Rank surplus,
                # if any, must come from actual labels in the full lift.
                if gf2_rank(parity_rows) != rows:
                    continue
                key = min(
                    tuple(sorted(tuple(v[i] for i in p)
                                 for v in columns_))
                    for p in permutations(range(rows))
                )
                keys.add(key)

        def preference(key):
            loops = sum(sum(x != 0 for x in v) == 1
                        for v in key if sum(v) == 2)
            return loops, max(max(v) for v in key), key

        ordered = sorted(keys, key=preference)
        if not ordered:
            raise ValueError("no balanced incidence prescriptions")
        return [
            tuple(tuple(v[i] for v in key) for i in range(rows))
            for key in ordered
        ]

    libraries = {
        "A": support_library(*A_SHAPE),
        "B": support_library(*B_SHAPE),
    }

    def draw_grid(local, side, card):
        library = libraries[side]
        # Twenty deliberate incidence/coset prescriptions. Multiple
        # proposals per prescription are screened, never emitted as draws.
        profile = library[
            (card // 2 + (3 if side == "B" else 0))
            % min(10, len(library))
        ]
        grid = []
        for i, counts in enumerate(profile):
            row = []
            for j, terms in enumerate(counts):
                if terms == 0:
                    row.append([])
                    continue
                first = G.pick(local)
                cell = [first]
                if terms >= 2:
                    pure_column = sum(profile[r][j]
                                      for r in range(len(profile))) == terms
                    policy = (card + i + j + int(side == "B")) % 3
                    if pure_column or policy == 0:
                        # Order-six differences avoid the unavoidable
                        # order-three/four annihilator of a pure binomial.
                        relative = (local.choice((1, 5)), 0)
                    elif policy == 1:
                        relative = (local.randrange(6), 1)
                    else:
                        relative = (local.choice((1, 2, 4, 5)), 0)
                    cell.append(G.mul(first, relative))
                if terms == 3:
                    choices = [
                        h for h in G.elements
                        if h[1] == 1
                        and G.mul(first, h) not in cell
                    ]
                    cell.append(G.mul(first, local.choice(choices)))
                assert len({G._key(g) for g in cell}) == terms
                row.append(cell)
            grid.append(row)
        assert all(sum(map(len, row)) == 6 for row in grid)
        return grid

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def component_quality(rows, width):
        """Small-kernel collision count and Tanner 4/6-cycle penalties."""
        cols = columns(rows, width)
        seen = {0}
        bad = 0
        for s in cols:
            bad += int(s in seen)
            seen.add(s)
        c4 = 0
        for i, s in enumerate(cols):
            for t in cols[:i]:
                u = s ^ t
                bad += int(u in seen)
                seen.add(u)
                overlap = _popcount(s & t)
                c4 += overlap * (overlap - 1) // 2

        # Project onto check nodes. With no 4-cycles, subtracting
        # within-column triangles leaves exactly the Tanner 6-cycles.
        adj = [0] * len(rows)
        internal = 0
        for s in cols:
            degree = _popcount(s)
            internal += degree * (degree - 1) * (degree - 2) // 6
            bits = s
            while bits:
                bit = bits & -bits
                adj[bit.bit_length() - 1] |= s ^ bit
                bits ^= bit
        triangles = 0
        for u, neighbors in enumerate(adj):
            bits = neighbors >> (u + 1)
            while bits:
                bit = bits & -bits
                v = bit.bit_length() + u
                triangles += _popcount((neighbors & adj[v]) >> (v + 1))
                bits ^= bit
        return bad, c4, max(0, triangles - internal)

    def low_logical(checks, stabilizers, width):
        """Exact existence test for a nontrivial logical of weight <=4."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                if p not in pivots:
                    return True
                word ^= pivots[p]
            return False

        syndromes = columns(checks, width)
        representatives = {0: 0}
        for i, syndrome in enumerate(syndromes):
            word = 1 << i
            if syndrome in representatives:
                if nontrivial(word ^ representatives[syndrome]):
                    return True
            else:
                representatives[syndrome] = word
        for i, syndrome in enumerate(syndromes):
            bit = 1 << i
            for j in range(i):
                s = syndrome ^ syndromes[j]
                word = bit | (1 << j)
                if s in representatives:
                    if nontrivial(word ^ representatives[s]):
                        return True
                else:
                    representatives[s] = word
        # All words of weight <=4 split into two words of weight <=2.
        # One representative per syndrome suffices because the
        # stabilizer space is linear.
        return False

    def component(grid, side):
        # Zero auxiliary cells isolate exactly the scaffold constituent.
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, [[[]]])
        else:
            _, rows, _ = lifted_matrices(G, [[[]]], grid)
        width = len(grid[0]) * q
        assert len(rows) == len(grid) * q
        assert all(row.bit_length() <= width for row in rows)
        return rows, columns(rows, width)

    def kernel_five(cols):
        """Find a kernel word <=5, or exactly exclude that finite range."""
        seen = {0: 0}
        for i, syndrome in enumerate(cols):
            word = 1 << i
            old = seen.get(syndrome)
            if old is not None:
                return word ^ old
            seen[syndrome] = word
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                word = (1 << i) | (1 << j)
                old = seen.get(syndrome)
                if old is not None and old != word:
                    return word ^ old
                seen[syndrome] = word
        # Every support of weight <=5 splits into sizes <=3 and <=2.
        for i, first in enumerate(cols):
            for j in range(i):
                pair = first ^ cols[j]
                bits = (1 << i) | (1 << j)
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    old = seen.get(syndrome)
                    if old is not None:
                        return bits ^ (1 << h) ^ old
        return 0

    def span(rows):
        pivots = {}
        for word in rows:
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

    def probe(local, cols, passes, seed=0, own=None, stabilizers=()):
        """Heuristic upper bound carrying an explicitly verified witness."""
        best = seed
        weight = seed.bit_count() if seed else len(cols) + 1

        def inspect(word):
            nonlocal best, weight
            w = word.bit_count()
            if (word and w < weight
                    and (own is None or outside(word, own))):
                best, weight = word, w

        for _ in range(passes):
            order = list(range(len(cols)))
            local.shuffle(order)
            pivots, basis = {}, []
            for j in order:
                syndrome, word = cols[j], 1 << j
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    basis.append(word)

            local.shuffle(basis)
            for i, word in enumerate(basis):
                w = word.bit_count()
                for j, other in enumerate(basis):
                    if i == j:
                        continue
                    trial = word ^ other
                    tw = trial.bit_count()
                    if 0 < tw < w:
                        word, w = trial, tw
                for row in stabilizers:
                    trial = word ^ row
                    tw = trial.bit_count()
                    if 0 < tw < w:
                        word, w = trial, tw
                basis[i] = word
                inspect(word)

            for i, first in enumerate(basis):
                for second in basis[:i]:
                    inspect(first ^ second)
            short = sorted(basis, key=int.bit_count)[:28]
            for i, first in enumerate(short):
                for j in range(i):
                    pair = first ^ short[j]
                    for h in range(j):
                        inspect(pair ^ short[h])
            if best:
                for _ in range(2):
                    starting = best
                    for i, first in enumerate(short):
                        inspect(starting ^ first)
                        for second in short[:i]:
                            inspect(starting ^ first ^ second)
                    if best == starting:
                        break

        if not best:
            raise ValueError("kernel witness search returned no witness")
        syndrome = 0
        bits = best
        while bits:
            bit = bits & -bits
            syndrome ^= cols[bit.bit_length() - 1]
            bits ^= bit
        assert syndrome == 0
        assert own is None or outside(best, own)
        return weight, best

    def bank(side):
        records = []
        rows_count, cols_count = A_SHAPE if side == "A" else B_SHAPE
        width = cols_count * q
        for card in range(20):
            local = random.Random(
                BASE_SEED ^ (0xD12358 + 104729 * card
                             + (1000003 if side == "B" else 0)))
            raw, fillings = [], set()
            for trial in range(72):
                grid = draw_grid(local, side, card)
                filling = tuple(
                    tuple(sorted(G.idx(g) for g in cell))
                    for row in grid for cell in row)
                if filling in fillings:
                    continue
                fillings.add(filling)
                rows, cols = component(grid, side)
                stats = component_quality(rows, width)
                if stats[0]:
                    continue
                deficit = rows_count * q - gf2_rank(rows)
                if deficit > 4:
                    continue
                cost = (12 + 8 * (card % 3)) * stats[1] + stats[2]
                cheap = (abs(deficit - 2), cost, trial)
                raw.append(dict(
                    grid=grid, rows=rows, cols=cols, deficit=deficit,
                    cost=cost, cheap=cheap, card=card, trial=trial))
            raw.sort(key=lambda r: r["cheap"])

            shortlisted = []
            for record in raw[:8]:
                seed = kernel_five(record["cols"])
                record["distance"], record["witness"] = probe(
                    local, record["cols"], 3, seed)
                shortlisted.append(record)

            def quality(record):
                return (
                    record["distance"],
                    -abs(record["deficit"] - 2),
                    -record["cost"], -record["trial"])

            shortlisted.sort(key=quality, reverse=True)
            # All publishable constituents receive the same 3+15 passes.
            for record in shortlisted[:3]:
                record["distance"], record["witness"] = probe(
                    local, record["cols"], 15, record["witness"])
                assert all((row & record["witness"]).bit_count() % 2 == 0
                           for row in record["rows"])
                records.append(record)
        records.sort(key=lambda r: (
            -r["distance"], abs(r["deficit"] - 2),
            r["cost"], r["card"], r["trial"]))
        if not records:
            raise ValueError("no independently screened constituent")
        return records

    def graph_key(hx, hz, n):
        """Conservative invariant of the complete uncoloured Tanner graph.

        Check degrees are nine; qubit degrees are 4..6 or twelve.
        Thus vertex classes are intrinsically distinguished without
        assigning X/Z colours. Equal signatures are rejected.
        """
        rows = hx + hz
        cols = columns(rows, n)
        adjacency = [0] * len(rows)
        overlaps = {}
        for i, row in enumerate(rows):
            for j in range(i):
                shared = (row & rows[j]).bit_count()
                if shared:
                    overlaps[i, j] = shared
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        features = []
        for i, neighbors in enumerate(adjacency):
            histogram = Counter()
            bits = neighbors
            while bits:
                bit = bits & -bits
                j = bit.bit_length() - 1
                shared = overlaps.get((max(i, j), min(i, j)), 0)
                histogram[(
                    rows[j].bit_count(), shared,
                    (neighbors & adjacency[j]).bit_count()
                )] += 1
                bits ^= bit
            walk4 = sum((neighbors & other).bit_count() ** 2
                        for other in adjacency)
            features.append((
                rows[i].bit_count(), neighbors.bit_count(),
                tuple(sorted(histogram.items())), walk4))

        palette = {f: c for c, f in enumerate(sorted(set(features)))}
        colors = [palette[f] for f in features]
        incidence = []
        for col in cols:
            ids = []
            while col:
                bit = col & -col
                ids.append(bit.bit_length() - 1)
                col ^= bit
            incidence.append(ids)
        history = [tuple(sorted(Counter(features).items()))]
        for _ in range(2):
            qfeatures = [tuple(sorted(colors[r] for r in ids))
                         for ids in incidence]
            qp = {f: c for c, f in enumerate(sorted(set(qfeatures)))}
            neighborhoods = [[] for _ in rows]
            for ids, feature in zip(incidence, qfeatures):
                for r in ids:
                    neighborhoods[r].append(qp[feature])
            refined = [(colors[r], tuple(sorted(neighborhoods[r])))
                       for r in range(len(rows))]
            history.append((
                tuple(sorted(Counter(qfeatures).items())),
                tuple(sorted(Counter(refined).items()))))
            palette = {f: c for c, f in enumerate(sorted(set(refined)))}
            colors = [palette[f] for f in refined]
        value = (
            n, len(rows),
            tuple(sorted(Counter(len(ids) for ids in incidence).items())),
            tuple(history))
        return sha256(repr(value).encode("ascii")).digest()

    def quantum_reading(a, b, hx, hz, n, local):
        """Verify actual logical classes, then search mixed-sector words."""
        overall = None
        for kind, checks, stabilizers in (
                ("X", hz, hx), ("Z", hx, hz)):
            own = span(stabilizers)
            best, weight = 0, n + 1
            for block in range(A_SHAPE[1] if kind == "X"
                               else B_SHAPE[1]):
                if kind == "X":
                    word = b["witness"] << (block * B_SHAPE[1] * q)
                else:
                    word, source = 0, a["witness"]
                    while source:
                        bit = source & -source
                        j, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << ((j * B_SHAPE[1] + block) * q + h)
                        source ^= bit
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                if not outside(word, own):
                    continue
                w = word.bit_count()
                for _ in range(2):
                    starting = word
                    for i, first in enumerate(stabilizers):
                        trial = starting ^ first
                        tw = trial.bit_count()
                        if tw < w:
                            word, w = trial, tw
                        for second in stabilizers[:i]:
                            trial = starting ^ first ^ second
                            tw = trial.bit_count()
                            if tw < w:
                                word, w = trial, tw
                    if word == starting:
                        break
                if w < weight:
                    best, weight = word, w
            if not best:
                return None
            weight, best = probe(
                local, columns(checks, n), 2, best,
                own=own, stabilizers=stabilizers)
            assert outside(best, own)
            assert all((row & best).bit_count() % 2 == 0
                       for row in checks)
            overall = weight if overall is None else min(overall, weight)
        return overall

    aa, bb = bank("A"), bank("B")
    pending = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            weak = min(a["distance"], b["distance"])
            # This only orders proposals. Complete ranks replace it below.
            rank_pressure = 5 * a["deficit"] + 3 * b["deficit"]
            priority = (
                weak, min(rank_pressure, 24),
                a["distance"] + b["distance"],
                -a["cost"] - b["cost"], -ai, -bi)
            pending.append((priority, ai, bi))
    pending.sort(reverse=True)

    results, graph_keys, filling_keys = [], set(), set()
    for priority, ai, bi in pending:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if filling in filling_keys:
            continue
        filling_keys.add(filling)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _no_dead_lines(sa, sb)
        assert _grid_weights(sa, sb) == 9 <= cap

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 552 or not (325 * n <= 1000 * k < 375 * n):
            continue
        assert k >= 180
        signature = graph_key(hx, hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        if low_logical(hz, hx, n) or low_logical(hx, hz, n):
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) == 9

        local = random.Random(
            BASE_SEED ^ (0xD12359 + 1009 * ai + 65537 * bi))
        reading = quantum_reading(a, b, hx, hz, n, local)
        if reading is None:
            continue

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        quality = (k * reading * reading / float(n),
                   reading, k, priority)
        results.append((
            quality, freeze(A), freeze(B), k, reading,
            (a["distance"], b["distance"]),
            (a["card"], b["card"]), signature.hex()))
        if len(results) >= max(5, count) + 16:
            break
    results.sort(key=lambda r: r[0], reverse=True)
    if not results:
        raise ValueError("no complete target-band pair passed screening")
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Pure lookup; slot order and caller RNG do not change the portfolio."""
    catalogue = _r035_catalogue(G, N_SLOTS)
    if not 0 <= idx < len(catalogue):
        raise ValueError("invariant-separated target-band portfolio exhausted")
    quality, A, B, k, reading, components, cards, signature = catalogue[idx]
    return {
        "idx": idx, "cells": {"A": A, "B": B},
        "screen_k": k, "quality": quality,
        "component_witness_readings": components,
        "quantum_witness_reading": reading,
        "design_cards": cards, "tanner_signature": signature,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh list from the already screened complete filling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Rolled uniform Z21 fillings on (2,4)x(2,6) suffice for d30s>4: refuted by all five measured parent codes having d30s=4, with sector-1 witnesses.
# REFUTED: Component-cycle minimization plus logical-four rejection confines Dic11 minima to sector 1: measured parent slot 42 had weight-20 Z logicals touching both LP sectors.
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

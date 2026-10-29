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
CONCEPT = ("On C9 semidirect C6 with inversion, order 54, shapes (1,3)x(1,3), quotient-unbalanced noncommuting trinomials selected by direct weight-eight witness searches should produce five measured distance-ten codes at k=216 or 222 and exceed fitness 27.022 in cell 0.400.")
GROUP = ("Frozen byte-for-byte under L1. This exact order-54 bed measured five [540,228,8] codes with thousands of sector-one weight-eight witnesses. Test a different filling geometry before attributing that plateau to D9 x C3 itself; no group advantage or distance improvement is assumed.")
SHAPES = ("Frozen equal (1,3) sides: n=540, kb=216 and guaranteed rate 0.400, with equal 162-variable/54-check components. The size window permits q<=59 and the frozen group fixes q=54. Accept verified k=216 or 222, rates 0.400 or 0.411111, inside cell 0.400. Parent minima occupied only sector-one Acol/Bcol blocks, prominently Acol1 and Bcol1.")
FILLING = ("All six cells remain cubic, keeping HX/HZ row weight twelve. Twenty policies cross five C3-quotient architectures with four normal-coordinate models; at least one side includes occupancy (2,1,0), replacing universal (1,1,1) balance. Exclude component kernels through six on every admitted proposal before shortlisting, directly search through eight with explicit completion flags, and repair witness-heavy columns without reusing invalidated witnesses. Independently refine both components, probe genuine full-code logicals modulo stabilizers, verify exact ranks and CSS, and reject repeated uncoloured Tanner signatures. Bounded and randomized searches remain witness upper estimates, not quantum-distance certificates.")
FAMILY_NAME = "r040_C9semiC6_inv_13x13_unbalanced_clean6_repair8_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 54

def custom_ops(q):
    # x^9=y^6=1; y*x*y^-1=x^-1.
    # y^2 generates a central C3; this group is D9 x C3.
    if not valid_q(q):
        raise ValueError("C9 semidirect C6 requires order 54")
    els = [(a, b) for b in range(6) for a in range(9)]

    def red(g):
        return (g[0] % 9, g[1] % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (1 if b % 2 == 0 else -1) * c) % 9,
                (b + d) % 6)

    def inv(g):
        a, b = red(g)
        return ((-a if b % 2 == 0 else a) % 9, (-b) % 6)

    def idx(g):
        a, b = red(g)
        return 9 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


# Actual reflection parities, indexed by the three quotient positions.
# Every pattern mixes rotations and reflections.
_Q54_PARITIES = (
    (0, 0, 1), (0, 1, 0), (1, 0, 0),
    (0, 1, 1), (1, 0, 1), (1, 1, 0),
)

# Five arrangements of A0,A1,A2,B0,B1,B2, crossed with four
# normal-coordinate models. Final invariants, not these labels alone,
# enforce nonisomorphism between emitted slots.
_Q54_CARDS = (
    (0, 1, 2, 3, 4, 5),
    (0, 0, 3, 1, 4, 4),
    (0, 3, 4, 2, 2, 5),
    (1, 1, 5, 0, 3, 5),
    (0, 2, 4, 1, 3, 3),
)


def _screen_columns(rows, nc):
    """Transpose rows obtained from the verified construction helper."""
    cols = [0] * nc
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            cols[low.bit_length() - 1] |= bit
            row ^= low
    return cols


def _screen_four(cols, cap=64):
    """Count capped pair-syndrome collisions; zero excludes kernels <=4.

    All columns screened here have odd weight, excluding odd kernels.
    With distinct nonzero columns, colliding different pairs are disjoint.
    """
    if 0 in cols or len(set(cols)) != len(cols):
        return cap
    seen = set()
    hits = 0
    for a in range(len(cols) - 1):
        x = cols[a]
        for b in range(a + 1, len(cols)):
            syndrome = x ^ cols[b]
            if syndrome in seen:
                hits += 1
                if hits >= cap:
                    return cap
            else:
                seen.add(syndrome)
    return hits


def _screen_component(rows, q):
    cols = _screen_columns(rows, 3 * q)
    four = _screen_four(cols)
    # A representative in each fibre suffices for this overlap objective.
    # These overlaps are a selection heuristic, not a distance bound.
    cycles = 0
    for a in (0, q, 2 * q):
        x = cols[a]
        for b, y in enumerate(cols):
            if a != b:
                overlap = (x & y).bit_count()
                cycles += overlap * (overlap - 1) // 2
    return cols, four, cycles


def _screen_six(cols, q, cap=128):
    """Capped weight-6 component witnesses using anchored triple matching.

    The component lifts have a transitive group action on each fibre.
    Translate any six-support to include one of 0,q,2q, then split it
    into two disjoint triples. A zero result exhausts this test.
    """
    length = len(cols)
    bits = [1 << a for a in range(length)]
    anchored = {}
    for a in (0, q, 2 * q):
        for b in range(length - 1):
            if b == a:
                continue
            ab = cols[a] ^ cols[b]
            mask_ab = bits[a] | bits[b]
            for c in range(b + 1, length):
                if c == a:
                    continue
                key = ab ^ cols[c]
                anchored.setdefault(key, []).append(mask_ab | bits[c])

    hits = 0
    lookup = anchored.get
    for a in range(length - 2):
        for b in range(a + 1, length - 1):
            ab = cols[a] ^ cols[b]
            mask_ab = bits[a] | bits[b]
            for c in range(b + 1, length):
                mates = lookup(ab ^ cols[c])
                if mates:
                    mask = mask_ab | bits[c]
                    for other in mates:
                        if not (mask & other):
                            hits += 1
                            if hits >= cap:
                                return cap
    return hits


from functools import lru_cache
from collections import Counter


@lru_cache(maxsize=1)
def _q54_atlas(G):
    """Compile both cell actions exclusively from verified unit lifts."""
    q = G.q
    mask = (1 << q) - 1
    stub = [[[G.e] for _ in range(3)]]
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[g], [], []]]
            A, B = (unit, stub) if side == "A" else (stub, unit)
            hx, hz, _ = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            rows = [row & mask for row in source[:q]]
            blocks.append(tuple(_screen_columns(rows, q)))
        atlas[side] = tuple(blocks)
    return atlas


def _q54_columns(G, atlas, side, grid):
    """F2-linear superposition of scaffold-generated single-term blocks."""
    q = G.q
    cols = [0] * (3 * q)
    for j, cell in enumerate(grid[0]):
        for g in cell:
            block = atlas[side][G.idx(g)]
            for h, syndrome in enumerate(block):
                cols[j * q + h] ^= syndrome
    return cols


def _q54_cycles(cols):
    counts = {}
    total = 0
    for syndrome in cols:
        neighbors = []
        while syndrome:
            bit = syndrome & -syndrome
            neighbors.append(bit.bit_length() - 1)
            syndrome ^= bit
        for i, a in enumerate(neighbors):
            for b in neighbors[:i]:
                pair = (b, a)
                old = counts.get(pair, 0)
                total += old
                counts[pair] = old + 1
    return total


def _q54_draw(local, G, kinds, mode, slot, side):
    """One term per C3 coset, with prescribed parity/coordinate policies."""
    row = []
    units = (1, 2, 4, 5, 7, 8)
    for j, kind in enumerate(kinds):
        for retry in range(128):
            parity = _Q54_PARITIES[kind]
            phase = local.randrange(3)
            quotient = [(t + phase) % 3 for t in range(3)]
            bs = [b + 3 * ((parity[t] - b) % 2)
                  for t, b in enumerate(quotient)]
            if mode == 0:
                aa = local.sample(range(9), 3)
            elif mode == 1:
                base, step = local.randrange(9), local.choice(units)
                aa = [(base + step * t) % 9 for t in range(3)]
            elif mode == 2:
                base, step = local.randrange(9), local.choice(units)
                aa = [(base + step * t * t) % 9 for t in range(3)]
            else:
                aa = [local.randrange(9) for _ in range(3)]
            cell = list(zip(aa, bs))
            relatives = [G.mul(G.inv(cell[0]), g) for g in cell[1:]]
            if G.mul(relatives[0], relatives[1]) == G.mul(
                    relatives[1], relatives[0]):
                continue
            assert len(set(cell)) == 3
            assert {g[1] % 3 for g in cell} == {0, 1, 2}
            row.append(cell)
            break
        else:
            raise ValueError("noncommuting quotient recipe exhausted")
    return [row]


def _q54_probe(cols, local, trials, seeds=()):
    """Search actual classical kernel words, never a distance certificate."""
    width = len(cols)
    order = list(range(width))
    memory = sorted(set(seeds), key=lambda v: (v.bit_count(), v))[:24]
    best = min((v.bit_count() for v in memory), default=width + 1)

    for trial in range(trials):
        local.shuffle(order)
        pivots, words = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                assert word
                words.append(word)
                best = min(best, word.bit_count())

        reducers = sorted(set(words + memory),
                          key=lambda v: (v.bit_count(), v))[:32]
        for sweep in range(2):
            local.shuffle(reducers)
            for i, word in enumerate(words):
                weight = word.bit_count()
                for other in reducers:
                    combined = word ^ other
                    cw = combined.bit_count()
                    if 0 < cw < weight:
                        word, weight = combined, cw
                words[i] = word
                best = min(best, weight)
            reducers = sorted(set(words + memory),
                              key=lambda v: (v.bit_count(), v))[:32]

        short = sorted(set(words + memory),
                       key=lambda v: (v.bit_count(), v))[:40]
        discoveries = list(short)
        for i, word in enumerate(short):
            for other in short[:i]:
                combined = word ^ other
                if combined:
                    best = min(best, combined.bit_count())
                    discoveries.append(combined)
        if trials >= 12 and trial % 4 == 0:
            small = short[:10]
            for i, word in enumerate(small):
                for j in range(i):
                    pair = word ^ small[j]
                    for t in range(j):
                        combined = pair ^ small[t]
                        if combined:
                            best = min(best, combined.bit_count())
                            discoveries.append(combined)
        memory = sorted(set(discoveries + memory),
                        key=lambda v: (v.bit_count(), v))[:24]
        if best <= 6:
            break

    # With 162 columns and rank 52, every information set supplies
    # nonzero kernel words. A no-witness sentinel is not scored.
    assert memory and best <= width
    return best, tuple(memory)


def _q54_select(local, G, atlas, side, kinds, mode, slot, batch):
    """Optimize a prescribed component, then independently rescreen."""
    pool, seen = [], set()
    q = G.q
    for attempt in range(112 + 32 * batch):
        grid = _q54_draw(local, G, kinds, mode, slot, side)
        labels = tuple(tuple(sorted(cell)) for cell in grid[0])
        if labels in seen:
            continue
        seen.add(labels)
        cols = _q54_columns(G, atlas, side, grid)
        assert all(c.bit_count() == 3 for c in cols)
        if gf2_rank(cols) != q - 2:
            continue
        if _screen_four(cols, cap=1):
            continue
        cycles = _q54_cycles(cols)
        reading, words = _q54_probe(cols, local, 2)
        item = dict(grid=grid, cols=cols, cycles=cycles,
                    reading=reading, words=words, attempt=attempt)
        pool.append(item)
        pool.sort(key=lambda x: (
            -x["reading"], x["cycles"], x["attempt"]))
        del pool[8:]

    for item in pool:
        # Zero exhausts the anchored test; reaching the cap is merely
        # a capped multiplicity, not a complete six-word count.
        six = _screen_six(item["cols"], q, cap=128)
        reading, words = _q54_probe(
            item["cols"], local, 12, item["words"])
        item["six"] = six
        item["reading"] = min(item["reading"], reading,
                              6 if six else len(item["cols"]) + 1)
        item["words"] = words

    def merit(item):
        return (item["reading"], int(item["six"] == 0),
                -item["six"], -item["cycles"])

    pool.sort(key=merit, reverse=True)
    finalists = pool[:4]
    for item in finalists:
        reading, words = _q54_probe(
            item["cols"], local, 32, item["words"])
        item["reading"] = min(item["reading"], reading)
        item["words"] = words
    finalists.sort(key=merit, reverse=True)
    return finalists


def _q54_fingerprint(hx, hz, n):
    """Uncoloured Tanner invariants: inequality certifies nonisomorphism.

    Equal fingerprints are rejected conservatively. No claim that
    fingerprint equality proves graph isomorphism is used.
    """
    rows = hx + hz
    incidents = [[] for _ in range(n)]
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            incidents[bit.bit_length() - 1].append(r)
            source ^= bit
    adjacency = [Counter() for _ in rows]
    for checks in incidents:
        for i, a in enumerate(checks):
            for b in checks[:i]:
                adjacency[a][b] += 1
                adjacency[b][a] += 1

    profiles = []
    for r, neighbors in enumerate(adjacency):
        square = Counter()
        for s, multiplicity in neighbors.items():
            for t, other in adjacency[s].items():
                square[t] += multiplicity * other
        profiles.append((
            rows[r].bit_count(),
            tuple(sorted(Counter(neighbors.values()).items())),
            square.get(r, 0),
            sum(neighbors.get(t, 0) * value
                for t, value in square.items()),
            tuple(sorted(Counter(square.values()).items()))))
    return (tuple(sorted(Counter(map(len, incidents)).items())),
            tuple(sorted(Counter(profiles).items())))


@lru_cache(maxsize=1)
def _q54_bank(G):
    """Deterministic policy bank; never fall back to an emitted orbit copy."""
    assert G.q == 54 and A_SHAPE == B_SHAPE == (1, 3)
    q = G.q
    atlas = _q54_atlas(G)
    mask = (1 << q) - 1
    bank, emitted = [], set()

    for slot in range(N_SLOTS):
        architecture, mode = divmod(slot % 20, 4)
        card = _Q54_CARDS[architecture]
        lane = slot // 20
        # Extra slots change only B's relative parity architecture,
        # rather than applying a common transformation to a winner.
        ak = card[:3]
        bk = tuple((kind + lane) % len(_Q54_PARITIES)
                   for kind in card[3:])
        chosen = None

        for batch in range(3):
            local = random.Random(
                BASE_SEED * 100003 + 54022801
                + 100019 * slot + 7919 * batch)
            ap = _q54_select(
                local, G, atlas, "A", ak, mode, slot, batch)
            bp = _q54_select(
                local, G, atlas, "B", bk, mode, slot, batch)
            proposals = []
            for a in ap:
                for b in bp:
                    d = min(a["reading"], b["reading"])
                    key = (d, a["reading"] + b["reading"],
                           -a["six"] - b["six"],
                           -a["cycles"] - b["cycles"])
                    proposals.append((key, a, b))
            proposals.sort(key=lambda item: item[0], reverse=True)

            for key, a, b in proposals:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != 540 or k != 228:
                    continue

                # Check the compiled component columns against complete
                # verified lifts before trusting their selection statistics.
                ar = [
                    sum(((row >> (3 * j * q)) & mask) << (j * q)
                        for j in range(3))
                    for row in hx[:q]]
                br = [row & ((1 << (3 * q)) - 1)
                      for row in hz[:q]]
                assert _screen_columns(ar, 3 * q) == a["cols"]
                assert _screen_columns(br, 3 * q) == b["cols"]
                if _screen_four(_screen_columns(hx, n), cap=1):
                    continue
                if _screen_four(_screen_columns(hz, n), cap=1):
                    continue

                fingerprint = _q54_fingerprint(hx, hz, n)
                if fingerprint in emitted:
                    continue
                assert css_ok(hx, hz)
                assert _grid_weights([[3, 3, 3]], [[3, 3, 3]]) == 12
                assert _no_dead_lines([[3, 3, 3]], [[3, 3, 3]])
                assert max(row.bit_count() for row in hx + hz) == 12
                assert 375 * n <= 1000 * k < 425 * n
                assert all(len(cell) == len(set(cell)) == 3
                           for grid in (A, B)
                           for row in grid for cell in row)
                emitted.add(fingerprint)
                frozen_a = tuple(tuple(tuple(cell) for cell in row)
                                 for row in A)
                frozen_b = tuple(tuple(tuple(cell) for cell in row)
                                 for row in B)
                chosen = (frozen_a, frozen_b, key)
                break
            if chosen is not None:
                break

        # An unsuccessful policy is voided, not replaced by a duplicate.
        bank.append(chosen)
    return tuple(bank)


@lru_cache(maxsize=4)
def NEW_CTX(rng, G, idx):
    """Pure deterministic bank; historical rank-228 bank is not called."""
    if idx >= 0:
        selected = NEW_CTX(None, G, -1)[idx]
        if selected is None:
            raise ValueError("no fresh quotient-unbalanced filling passed")
        A, B, k, selection = selected
        return {"idx": idx, "k": k, "selection": selection,
                "cells": {
                    "A": [[list(cell) for cell in row] for row in A],
                    "B": [[list(cell) for cell in row] for row in B]}}

    assert G.q == 54 and A_SHAPE == B_SHAPE == (1, 3)
    q, width = G.q, 3 * G.q
    atlas = _q54_atlas(G)
    mask = (1 << q) - 1

    # Kind zero is balanced; kinds one/two have opposite double-coset
    # orientations. Every card changes at least one side's quotient image.
    quotient_patterns = ((0, 1, 2), (0, 0, 1), (0, 0, 2))
    cards = (
        (0, 0, 0, 1, 1, 1),
        (0, 0, 0, 0, 1, 2),
        (0, 0, 1, 0, 1, 1),
        (0, 1, 2, 1, 1, 1),
        (1, 1, 1, 1, 1, 2),
    )
    units = (1, 2, 4, 5, 7, 8)

    def draw_cell(local, kind, mode):
        for _ in range(192):
            phase = local.randrange(3)
            quotient = [(phase + t) % 3
                        for t in quotient_patterns[kind]]
            parity = local.choice(_Q54_PARITIES)
            bs = [b + 3 * ((p - b) % 2)
                  for b, p in zip(quotient, parity)]
            if mode == 0:
                aa = local.sample(range(9), 3)
            elif mode == 1:
                base, step = local.randrange(9), local.choice(units)
                aa = [(base + t * step) % 9 for t in range(3)]
                local.shuffle(aa)
            elif mode == 2:
                # Distinct residues modulo three exclude internal
                # order-three rotation differences in this policy.
                aa = [t + 3 * local.randrange(3) for t in range(3)]
                local.shuffle(aa)
            else:
                aa = [local.randrange(9) for _ in range(3)]
            cell = list(zip(aa, bs))
            if len(set(cell)) != 3:
                continue
            # No pair differing only by the central C3.
            if len({(a, b % 2) for a, b in cell}) != 3:
                continue
            relative = [G.mul(G.inv(cell[0]), g) for g in cell[1:]]
            if G.mul(relative[0], relative[1]) == G.mul(
                    relative[1], relative[0]):
                continue
            return cell
        raise ValueError("quotient-unbalanced cell recipe exhausted")

    def search_data(cols):
        """Incidence and unique tails from verified component columns."""
        bits = [1 << j for j in range(width)]
        tails = {0: 0}
        for j, syndrome in enumerate(cols):
            tails[syndrome] = bits[j]
        for j, syndrome in enumerate(cols):
            for h in range(j):
                tails[syndrome ^ cols[h]] = bits[j] | bits[h]
        # Transposing the verified syndromes is not a separate lift.
        rows = _screen_columns(cols, q)
        return bits, tails, rows

    def anchored_search(cols, data, maximum, hit_cap, node_cap):
        """Actual words through maximum, with an explicit completion flag.

        Three coordinate-zero anchors cover component support orbits.
        At an odd check, branch on the first selected available variable,
        banning earlier alternatives. Unique singleton/pair tails finish
        each branch. Exhaustion without hits excludes the requested
        component weights; any cutoff is only a bounded witness search.
        """
        bits, tails, rows = data
        found = set()
        nodes = 0
        stopped = False

        def record(word):
            nonlocal stopped
            if word:
                found.add(word)
                if len(found) >= hit_cap:
                    stopped = True

        def visit(syndrome, word, banned, weight):
            nonlocal nodes, stopped
            if stopped:
                return
            if node_cap is not None and nodes >= node_cap:
                stopped = True
                return
            nodes += 1
            if not syndrome:
                record(word)
                return
            remaining = maximum - weight
            if remaining <= 0 or syndrome.bit_count() > 3 * remaining:
                return
            if remaining <= 2:
                tail = tails.get(syndrome)
                if (tail is not None and tail.bit_count() <= remaining
                        and not (tail & banned)):
                    record(word | tail)
                return

            pending = syndrome
            choices = 0
            least = width + 1
            while pending:
                bit = pending & -pending
                options = rows[bit.bit_length() - 1] & ~banned
                count = options.bit_count()
                if not count:
                    return
                if count < least:
                    choices, least = options, count
                    if count == 1:
                        break
                pending ^= bit

            alternatives = []
            while choices:
                bit = choices & -choices
                choices ^= bit
                j = bit.bit_length() - 1
                alternatives.append(j)
            alternatives.sort(
                key=lambda j: (-(cols[j] & syndrome).bit_count(), j))
            excluded = banned
            for j in alternatives:
                if stopped:
                    break
                excluded |= bits[j]
                visit(syndrome ^ cols[j], word | bits[j],
                      excluded, weight + 1)

        for anchor in (0, q, 2 * q):
            if stopped:
                break
            visit(cols[anchor], bits[anchor], bits[anchor], 1)
        return tuple(sorted(found)), not stopped, nodes

    def select(local, side, kinds, mode, batch):
        """Screen six before shortlisting; repair actual eight witnesses."""
        pool, seen = [], set()
        quota = 192 + 64 * batch

        for attempt in range(quota + 32):
            if attempt < quota or not pool:
                grid = [[draw_cell(local, kind, mode) for kind in kinds]]
            else:
                parent = local.choice(pool[:min(3, len(pool))])
                grid = [[list(cell) for cell in parent["grid"][0]]]
                witnesses = parent["eight"][:12]
                pressure = [
                    1 + sum(((word >> (j * q)) & mask).bit_count()
                            for word in witnesses)
                    for j in range(3)]
                j = local.choices(range(3), weights=pressure, k=1)[0]
                grid[0][j] = draw_cell(local, kinds[j], mode)
                if attempt % 7 == 0:
                    h = (j + 1 + local.randrange(2)) % 3
                    grid[0][h] = draw_cell(local, kinds[h], mode)

            labels = tuple(tuple(sorted(cell)) for cell in grid[0])
            if labels in seen:
                continue
            seen.add(labels)
            cols = _q54_columns(G, atlas, side, grid)
            assert all(c.bit_count() == 3 for c in cols)
            rank = gf2_rank(cols)
            if not q - 2 <= rank <= q:
                continue
            if _screen_four(cols, cap=1):
                continue

            data = search_data(cols)
            six, complete6, _ = anchored_search(
                cols, data, 6, hit_cap=1, node_cap=None)
            if six:
                continue
            assert complete6

            eight, complete8, _ = anchored_search(
                cols, data, 8, hit_cap=24, node_cap=12000)
            cycles = _q54_cycles(cols)
            early = (int(complete8 and not eight), int(not eight),
                     -len(eight), -cycles, q - rank, -attempt)
            pool.append(dict(
                grid=grid, cols=cols, rank=rank, data=data,
                eight=eight, complete8=complete8, cycles=cycles,
                early=early))
            pool.sort(key=lambda item: item["early"], reverse=True)
            del pool[8:]

        finalists = pool[:4]
        for item in finalists:
            eight, complete8, _ = anchored_search(
                item["cols"], item["data"], 8,
                hit_cap=128, node_cap=240000)
            seeds = tuple(set(item["eight"]) | set(eight))
            reading, words = _q54_probe(
                item["cols"], local, 24, seeds)
            item["eight"] = seeds
            item["words"] = words
            item["proof8"] = complete8 and not eight
            # This is a capped actual-witness reading, not an exclusion
            # flag masquerading as a found distance-ten operator.
            item["reading"] = min(reading, 10)

        def merit(item):
            return (item["reading"], int(item["proof8"]),
                    -len(item["eight"]), -item["cycles"],
                    q - item["rank"])

        finalists.sort(key=merit, reverse=True)
        for item in finalists[:3]:
            reading, words = _q54_probe(
                item["cols"], local, 40, item["words"])
            item["reading"] = min(item["reading"], reading)
            item["words"] = words
        finalists.sort(key=merit, reverse=True)
        return finalists[:3]

    def basis(rows):
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

    def logical_reading(checks, stabilizers, n, local):
        """Found full-code non-stabilizer witnesses; upper estimates only."""
        cols = _screen_columns(checks, n)
        span = basis(stabilizers)
        order = list(range(n))
        best = n + 1

        def observe(word):
            nonlocal best
            if not word or not outside(word, span):
                return
            weight = word.bit_count()
            for _ in range(2):
                changed = False
                for row in stabilizers:
                    candidate = word ^ row
                    w = candidate.bit_count()
                    if w < weight:
                        word, weight = candidate, w
                        changed = True
                if not changed:
                    break
            best = min(best, weight)

        for _ in range(3):
            local.shuffle(order)
            pivots, words = {}, []
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
                    words.append(word)
            logical = sorted(
                (word for word in words if outside(word, span)),
                key=int.bit_count)[:12]
            reducers = sorted(words, key=int.bit_count)[:32]
            for seed in logical:
                word = seed
                for _ in range(2):
                    chosen, weight = word, word.bit_count()
                    for reducer in reducers:
                        candidate = word ^ reducer
                        w = candidate.bit_count()
                        if (candidate and w < weight
                                and outside(candidate, span)):
                            chosen, weight = candidate, w
                    if chosen == word:
                        break
                    word = chosen
                observe(word)
            for j, word in enumerate(logical):
                for other in logical[:j]:
                    observe(word ^ other)
        return best

    bank, emitted = [], set()
    for slot in range(N_SLOTS):
        architecture, mode = divmod(slot % 20, 4)
        card = cards[architecture]
        ak, bk = card[:3], list(card[3:])
        lane = slot // 20
        if lane:
            # Change a relative quotient architecture on B only.
            bk[-1] = (bk[-1] + lane) % 3
        chosen = None

        for batch in range(2):
            local = random.Random(
                BASE_SEED * 100003 + 54081037
                + 104729 * slot + 7919 * batch)
            ap = select(local, "A", ak, mode, batch)
            bp = select(local, "B", tuple(bk), mode, batch)
            assembled = []

            for a in ap:
                for b in bp:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if n != 540 or k not in (216, 222):
                        continue
                    ar = [
                        sum(((row >> (3 * j * q)) & mask) << (j * q)
                            for j in range(3))
                        for row in hx[:q]]
                    br = [row & ((1 << width) - 1) for row in hz[:q]]
                    assert _screen_columns(ar, width) == a["cols"]
                    assert _screen_columns(br, width) == b["cols"]
                    if _screen_four(_screen_columns(hx, n), cap=1):
                        continue
                    if _screen_four(_screen_columns(hz, n), cap=1):
                        continue
                    fingerprint = _q54_fingerprint(hx, hz, n)
                    if fingerprint in emitted:
                        continue
                    d = min(a["reading"], b["reading"], 10)
                    key = (k * d * d, d,
                           int(a["proof8"]) + int(b["proof8"]),
                           -len(a["eight"]) - len(b["eight"]),
                           -a["cycles"] - b["cycles"])
                    assembled.append(
                        (key, A, B, hx, hz, n, k, fingerprint, a, b))
            assembled.sort(key=lambda item: item[0], reverse=True)

            winner = None
            for item in assembled[:4]:
                _, A, B, hx, hz, n, k, fingerprint, a, b = item
                dx = logical_reading(hz, hx, n, local)
                dz = logical_reading(hx, hz, n, local)
                d = min(a["reading"], b["reading"], dx, dz, 10)
                objective = (
                    k * d * d, d,
                    int(a["proof8"]) + int(b["proof8"]),
                    -len(a["eight"]) - len(b["eight"]),
                    -a["cycles"] - b["cycles"])
                if winner is None or objective > winner[0]:
                    winner = (objective, item)

            if winner is None:
                continue
            objective, item = winner
            _, A, B, hx, hz, n, k, fingerprint, a, b = item
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) == 12
            assert _grid_weights([[3, 3, 3]], [[3, 3, 3]]) == 12
            assert _no_dead_lines([[3, 3, 3]], [[3, 3, 3]])
            assert 375 * n <= 1000 * k < 425 * n
            assert all(len(cell) == len(set(cell)) == 3
                       for grid in (A, B) for row in grid for cell in row)
            emitted.add(fingerprint)
            frozen = tuple(
                tuple(tuple(tuple(cell) for cell in row) for row in grid)
                for grid in (A, B))
            chosen = (frozen[0], frozen[1], k, objective)
            break

        # Failed cards never reuse the historical rank-228 bank.
        bank.append(chosen)
    return tuple(bank)


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened full cell without further sampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Cap-legal rolled fillings do not reliably avoid distance-2 logicals: parent slots 6, 9, 11 and many others returned d2s=2.
# REFUTED: Reflection-gap scheduling with short-component screening suffices to exceed measured distance eight on the current D29 bed; all five measured current-parent codes returned d30s=8 with thousands of sector-1 weight-eight witnesses.
# REFUTED: Quotient-balanced rank-52 trinomials with the parent's selection eliminate weight-six witnesses in every emitted code; current slots 1, 5, 6, 9, 10 and many others returned d2s=6.
# REFUTED: Twenty parity/coordinate policies plus the parent's fingerprint filtering avoid evaluator isomorphism collapse; only 29 of the current program's 60 slots were distinct.
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

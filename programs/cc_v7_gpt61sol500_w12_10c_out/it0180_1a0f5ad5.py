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
CONCEPT = ("C27 semidirect C3 with action 10, order 81, shapes (1,2)x(1,3): noncommuting cubic cells screened against central-fiber and mixed-sector witnesses should produce five measured distance readings of at least twelve in cell 0.300; this is a falsifiable search hypothesis.")
GROUP = ("Jump from cyclic C19 to x^27=y^3=1, y*x*y^-1=x^10. Since 10 has order three modulo 27, this defines a nonabelian order-81 group, with central C9 and commutator C3. Its bed is absent from BEDS. Related C7:C9 cubic readings of ten and strong C13:C9 quartic readings motivate odd-order action exploration, without asserting distance transfer.")
SHAPES = ("Target 0.300 with (1,2)x(1,3): n=567, kb=162 and guaranteed rate 2/7. Component variable/check counts are 162/81 and 243/81; the LP sectors contain 486 and 81 qubits. Parent measured weight-eight/nine minima occupied sector-one Acol/Bcol blocks across its five-column geometry. Replace that geometry and retain only verified rates in [0.275,0.325).")
FILLING = ("Twenty cards prescribe distributions of noncommuting relative-order triangles (27,27,27), (9,27,27), and (3,27,27). Every cell has three distinct terms, giving HX/HZ weights 9/12. Pack action-specific directed differences and verify component four-cycle exclusion. Reject central-C3 invariant component words through physical weight nine, seed searches with expanded folded witnesses, and independently deepen finalists. Stabilizer-tested full-kernel searches and sector-two anchored beams address mixed-sector failures. A deterministic bank conservatively rejects equal uncoloured Tanner walk fingerprints. All bounded-search readings remain upper bounds or component heuristics, never distance certificates.")
FAMILY_NAME = "r030_C27sdC3_action10_12x13_central3_joint_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 81

def custom_ops(q):
    # x^27=y^3=1, y*x*y^-1=x^10.
    # 10^2=19 and 10^3=1 modulo 27.
    if not valid_q(q):
        raise ValueError("C27 semidirect C3 requires order 81")
    els = [(a, b) for b in range(3) for a in range(27)]
    action = (1, 10, 19)

    def red(g):
        return (g[0] % 27, g[1] % 3)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + action[b]*c) % 27, (b+d) % 3)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 3
        return ((-action[bi]*a) % 27, bi)

    def idx(g):
        a, b = red(g)
        return 27*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _component_rows(G, M, side):
    """Extract the selected component from the VERIFIED quantum lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    dummy_a = [[[G.e] for _ in range(ca)] for _ in range(ra)]
    dummy_b = [[[G.e] for _ in range(cb)] for _ in range(rb)]
    A, B = (M, dummy_b) if side == "A" else (dummy_a, M)
    hx, hz, _ = lifted_matrices(G, A, B)
    q = G.q
    mask = (1 << q) - 1
    if side == "A":
        return [
            sum(((hx[(i*cb)*q+r] >> (j*cb*q)) & mask) << (j*q)
                for j in range(ca))
            for i in range(ra) for r in range(q)
        ]
    bmask = (1 << (cb*q)) - 1
    return [hz[a*q+r] & bmask
            for a in range(rb) for r in range(q)]


def _column_syndromes(rows, n):
    cols = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _four_cycles(cols):
    seen = {}
    total = 0
    for col in cols:
        neighbors = []
        while col:
            bit = col & -col
            neighbors.append(bit.bit_length() - 1)
            col ^= bit
        for a in range(len(neighbors)):
            for b in range(a):
                key = (neighbors[b], neighbors[a])
                old = seen.get(key, 0)
                total += old
                seen[key] = old + 1
    return total


def _has_kernel_le6(cols):
    """Exact collision test on all subsets of size <=3.

    Any nonempty zero-syndrome support of size <=6 partitions into
    two different subsets of size <=3 with equal syndromes.
    """
    seen = {0: 0}
    bits = [1 << i for i in range(len(cols))]
    for i, s in enumerate(cols):
        if s in seen:
            return True
        seen[s] = bits[i]
    for i in range(len(cols)):
        for j in range(i):
            s = cols[i] ^ cols[j]
            if s in seen:
                return True
            seen[s] = bits[i] | bits[j]
    for i in range(len(cols)):
        for j in range(i):
            s2 = cols[i] ^ cols[j]
            mask2 = bits[i] | bits[j]
            for k in range(j):
                s = s2 ^ cols[k]
                if s in seen:
                    return True
                seen[s] = mask2 | bits[k]
    return False


def _row_basis(rows):
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


def _outside_span(word, basis):
    while word:
        p = word.bit_length()-1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _kernel_probe(rng, cols, trials, seeds=(), stabilizers=None,
                  descent_rows=()):
    """Actual kernel words; quantum readings require a nontrivial class.

    Seeds must be true kernel words for these verified columns.
    The width+1 sentinel means no witness, not a distance certificate.
    """
    width = len(cols)
    best = width+1
    memory = sorted(set(seeds)-{0}, key=int.bit_count)[:32]
    order = list(range(width))

    def observe(word):
        nonlocal best
        if not word:
            return
        if stabilizers is not None and not _outside_span(word, stabilizers):
            return
        best = min(best, word.bit_count())

    for word in memory:
        observe(word)

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
            if not syndrome:
                words.append(word)
                observe(word)

        for sweep in range(1 + int(trials >= 12)):
            reducers = sorted(set(words+memory), key=int.bit_count)[:48]
            rng.shuffle(reducers)
            for i, word in enumerate(words):
                weight = word.bit_count()
                for other in reducers:
                    combined = word ^ other
                    cw = combined.bit_count()
                    if 0 < cw < weight:
                        word, weight = combined, cw
                words[i] = word
                observe(word)

        short = sorted(set(words+memory)-{0}, key=int.bit_count)[:40]
        discoveries = set(short)
        for i, word in enumerate(short):
            for other in short[:i]:
                combined = word ^ other
                if combined:
                    observe(combined)
                    if combined.bit_count() <= best+4:
                        discoveries.add(combined)

        if trials >= 12 and trial % 4 == 0:
            small = short[:10]
            for i, word in enumerate(small):
                for j in range(i):
                    pair = word ^ small[j]
                    for h in range(j):
                        combined = pair ^ small[h]
                        observe(combined)
                        if combined and combined.bit_count() <= best+4:
                            discoveries.add(combined)

        if descent_rows and stabilizers is not None:
            logical_seeds = [
                word for word in short
                if _outside_span(word, stabilizers)
            ][:6]
            for word in logical_seeds:
                for sweep in range(3):
                    chosen, weight = word, word.bit_count()
                    nearby = []
                    for row in descent_rows:
                        combined = word ^ row
                        cw = combined.bit_count()
                        nearby.append((cw, row))
                        if 0 < cw < weight:
                            chosen, weight = combined, cw
                    nearby.sort(key=lambda x: x[0])
                    near = [row for _, row in nearby[:10]]
                    for i, row in enumerate(near):
                        for other in near[:i]:
                            combined = word ^ row ^ other
                            cw = combined.bit_count()
                            if 0 < cw < weight:
                                chosen, weight = combined, cw
                    if chosen == word:
                        break
                    word = chosen
                    observe(word)
                    discoveries.add(word)

        memory = sorted(discoveries, key=lambda w: (w.bit_count(), w))[:32]
        if best <= 4:
            break
    return best, memory


from functools import lru_cache


@lru_cache(maxsize=1)
def _long_triangle_catalog(G):
    """Noncommuting recipes; no matrix property is inferred from labels."""
    orders = {}
    inverses = {g: G.inv(g) for g in G.elements}
    for g in G.elements:
        h = G.e
        for exponent in range(1, G.q+1):
            h = G.mul(h, g)
            if h == G.e:
                orders[g] = exponent
                break
    signatures = ((27, 27, 27), (9, 27, 27), (3, 27, 27))
    catalog = {signature: [] for signature in signatures}
    for r in G.elements:
        for s in G.elements:
            if G.mul(r, s) == G.mul(s, r):
                continue
            signature = tuple(sorted(
                (orders[r], orders[s], orders[G.mul(inverses[r], s)])))
            if signature not in catalog:
                continue
            cell = (G.e, r, s)
            ratios = {G.mul(g, inverses[h])
                      for g in cell for h in cell if g != h}
            if len(ratios) == 6:
                catalog[signature].append((r, s))
    assert all(catalog.values())
    return signatures, catalog


@lru_cache(maxsize=1)
def _central_three_orbits(G):
    z = (9, 0)
    assert G.mul(z, G.mul(z, z)) == G.e
    assert all(G.mul(z, g) == G.mul(g, z) for g in G.elements)
    covered, orbits = set(), []
    for g in G.elements:
        if G.idx(g) in covered:
            continue
        orbit, h = [], g
        for _ in range(3):
            orbit.append(G.idx(h))
            h = G.mul(h, z)
        assert h == g and len(set(orbit)) == 3
        covered.update(orbit)
        orbits.append(tuple(sorted(orbit)))
    assert len(covered) == G.q
    return tuple(orbits)


def _central_restriction(cols, G):
    """Exact restriction to central-orbit indicators of physical weight 3."""
    orbits = _central_three_orbits(G)
    masks = [sum(1 << h for h in orbit) for orbit in orbits]
    folded, indicators = [], []
    for block in range(len(cols)//G.q):
        for orbit, mask in zip(orbits, masks):
            syndrome = 0
            for h in orbit:
                syndrome ^= cols[block*G.q+h]
            compact = 0
            for r, output_mask in enumerate(masks):
                part = syndrome & output_mask
                assert part == 0 or part == output_mask
                if part:
                    compact |= 1 << r
            folded.append(compact)
            indicators.append(mask << (block*G.q))
    return folded, indicators


def _expand_indicators(word, indicators):
    full = 0
    while word:
        bit = word & -word
        full ^= indicators[bit.bit_length()-1]
        word ^= bit
    return full


def _short_floor(cols):
    """Exact kernel minimum through four; five means only unexcluded."""
    singles = set(cols)
    if 0 in singles:
        return 1
    if len(singles) != len(cols):
        return 2
    pairs, four = set(), False
    for i, col in enumerate(cols):
        for other in cols[:i]:
            syndrome = col ^ other
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                four = True
            pairs.add(syndrome)
    return 4 if four else 5


def _cubic_six_cycles(cols):
    """Exact six-cycle count when the verified cubic matrix has no C4."""
    height = max(c.bit_length() for c in cols)
    adjacency = [0]*height
    for col in cols:
        neighbors = []
        while col:
            bit = col & -col
            neighbors.append(bit.bit_length()-1)
            col ^= bit
        assert len(neighbors) == 3
        for a in neighbors:
            for b in neighbors:
                if a != b:
                    adjacency[a] |= 1 << b
    triangles = 0
    for a, neighbors in enumerate(adjacency):
        higher = neighbors >> (a+1)
        while higher:
            bit = higher & -higher
            b = a+1+bit.bit_length()-1
            triangles += ((neighbors & adjacency[b]) >> (b+1)).bit_count()
            higher ^= bit
    return triangles-len(cols)


def _small_logical(checks, span, n):
    """Exactly detect non-stabilizer logicals of weight at most four."""
    cols = _column_syndromes(checks, n)
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = word
        elif _outside_span(word ^ old, span):
            return True
    for i, col in enumerate(cols):
        for j in range(i):
            syndrome = col ^ cols[j]
            word = (1 << i) | (1 << j)
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif _outside_span(word ^ old, span):
                return True
    return False


def _mixed_beam(cols, span, anchors, rng, width=96, limit=11):
    """Heuristic anchored search returning only genuine logical witnesses."""
    height = max(c.bit_length() for c in cols)
    incident = [[] for _ in range(height)]
    bits = [1 << i for i in range(len(cols))]
    for i, col in enumerate(cols):
        while col:
            bit = col & -col
            incident[bit.bit_length()-1].append(i)
            col ^= bit
    best = len(cols)+1
    for anchor in anchors:
        states = [(bits[anchor], cols[anchor])]
        memory = {0: 0, cols[anchor]: bits[anchor]}
        for depth in range(1, limit):
            following = {}
            for word, syndrome in states:
                choices = None
                remaining = syndrome
                while remaining:
                    bit = remaining & -remaining
                    available = [i for i in incident[bit.bit_length()-1]
                                 if not word & bits[i]]
                    if choices is None or len(available) < len(choices):
                        choices = available
                    remaining ^= bit
                for i in choices or ():
                    new_word = word | bits[i]
                    new_syndrome = syndrome ^ cols[i]
                    old = memory.get(new_syndrome)
                    if old is None:
                        memory[new_syndrome] = new_word
                    else:
                        dependency = new_word ^ old
                        if dependency and _outside_span(dependency, span):
                            best = min(best, dependency.bit_count())
                    if new_syndrome:
                        following[new_word] = new_syndrome
            if not following:
                break
            ranked = list(following.items())
            rng.shuffle(ranked)
            ranked.sort(key=lambda item: item[1].bit_count())
            states = ranked[:width]
            if best <= 9:
                return best
    return best


def _joint_merit(item):
    # Component witnesses are conservative selection heuristics, not
    # automatically asserted to represent nontrivial quantum logicals.
    d = min(item["da"], item["db"], item.get("full", item["n"]+1))
    return (item["k"]*d*d/item["n"], d,
            item["da"]+item["db"], -item["six"][1], -sum(item["six"]))


def _optimize_long_card(rng, G, idx):
    signatures, catalog = _long_triangle_catalog(G)
    card = idx % 20
    a_cards = ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2))
    b_cards = ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 2))
    ap = tuple(signatures[t] for t in a_cards[card//4])
    bp = tuple(signatures[t] for t in b_cards[card%4])
    q, mask = G.q, (1 << G.q)-1
    assert _grid_weights([[3, 3]], [[3, 3, 3]]) == 12
    assert _no_dead_lines([[3, 3]], [[3, 3, 3]])

    def packed(pattern, side):
        for restart in range(16):
            used, row = set(), [None]*len(pattern)
            order = list(range(len(pattern)))
            rng.shuffle(order)
            for j in order:
                for attempt in range(128):
                    r, s = rng.choice(catalog[pattern[j]])
                    anchor = G.pick(rng)
                    cell = [anchor, G.mul(anchor, r), G.mul(anchor, s)]
                    if side == "A":
                        ratios = {G.mul(g, G.inv(h))
                                  for g in cell for h in cell if g != h}
                    else:
                        ratios = {G.mul(G.inv(g), h)
                                  for g in cell for h in cell if g != h}
                    if len(ratios) != 6 or ratios & used:
                        continue
                    used.update(ratios)
                    row[j] = cell
                    break
                if row[j] is None:
                    break
            if all(cell is not None for cell in row):
                return [row]
        return None

    raw = []
    for attempt in range(64):
        A, B = packed(ap, "A"), packed(bp, "B")
        if A is None or B is None:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        ar = [(row & mask) | (((row >> (3*q)) & mask) << q)
              for row in hx[:q]]
        br = [row & ((1 << (3*q))-1) for row in hz[:q]]
        ac = _column_syndromes(ar, 2*q)
        bc = _column_syndromes(br, 3*q)
        if _four_cycles(ac) or _four_cycles(bc):
            continue
        fa, ia = _central_restriction(ac, G)
        fb, ib = _central_restriction(bc, G)
        # A compact word of weight <=3 expands to physical weight <=9.
        if min(_short_floor(fa), _short_floor(fb)) < 4:
            continue
        six = (_cubic_six_cycles(ac), _cubic_six_cycles(bc))
        item = dict(A=A, B=B, hx=hx, hz=hz, n=n,
                    ac=ac, bc=bc, fa=fa, fb=fb, ia=ia, ib=ib,
                    six=six)
        raw.append(item)
        raw.sort(key=lambda c: (2*c["six"][1]+c["six"][0],
                                max(c["six"])))
        del raw[10:]

    pool = []
    for item in raw:
        k = item["n"]-gf2_rank(item["hx"])-gf2_rank(item["hz"])
        if not 275*item["n"] <= 1000*k < 325*item["n"]:
            continue
        item["k"] = k
        _, aw = _kernel_probe(rng, item["fa"], 8)
        _, bw = _kernel_probe(rng, item["fb"], 8)
        item["aw"] = [_expand_indicators(w, item["ia"]) for w in aw]
        item["bw"] = [_expand_indicators(w, item["ib"]) for w in bw]
        item["da"], item["aw"] = _kernel_probe(
            rng, item["ac"], 4, item["aw"])
        item["db"], item["bw"] = _kernel_probe(
            rng, item["bc"], 8, item["bw"])
        if min(item["da"], item["db"]) <= 4:
            continue
        pool.append(item)
    if not pool:
        raise ValueError("no long-order cubic card passed verified screens")

    finalists = sorted(pool, key=_joint_merit, reverse=True)[:4]
    for item in finalists:
        for side, trials in (("a", 16), ("b", 32)):
            reading, words = _kernel_probe(
                rng, item[side+"c"], trials, item[side+"w"])
            item["d"+side] = min(item["d"+side], reading)
            item[side+"w"] = words
        xs, zs = _row_basis(item["hx"]), _row_basis(item["hz"])
        xc = _column_syndromes(item["hz"], item["n"])
        zc = _column_syndromes(item["hx"], item["n"])
        dx, _ = _kernel_probe(
            rng, xc, 8, stabilizers=xs, descent_rows=item["hx"])
        dz, _ = _kernel_probe(
            rng, zc, 8, stabilizers=zs, descent_rows=item["hz"])
        # These anchors deliberately include sector two. They are a
        # bounded search, not an assertion that all translated words occur.
        anchors = (6*q, 6*q+1, 6*q+9)
        dx = min(dx, _mixed_beam(xc, xs, anchors, rng))
        dz = min(dz, _mixed_beam(zc, zs, anchors, rng))
        item["full"] = min(dx, dz)
        item["xs"], item["zs"] = xs, zs

    leaders = sorted(finalists, key=_joint_merit, reverse=True)[:2]
    for item in leaders:
        for side, trials in (("a", 32), ("b", 64)):
            reading, words = _kernel_probe(
                rng, item[side+"c"], trials, item[side+"w"])
            item["d"+side] = min(item["d"+side], reading)
            item[side+"w"] = words
    return sorted(finalists, key=_joint_merit, reverse=True)


def _walk_fingerprint(hx, hz, n):
    """Uncoloured combined-Tanner invariant; equal values are rejected.

    Different values certify nonisomorphic incidence graphs. Equality
    is only a conservative duplicate filter, not an isomorphism proof.
    """
    from collections import Counter
    rows = hx+hz
    cols = _column_syndromes(rows, n)
    neighbors = [Counter() for _ in rows]
    for col in cols:
        checks = []
        while col:
            bit = col & -col
            checks.append(bit.bit_length()-1)
            col ^= bit
        for i, a in enumerate(checks):
            for b in checks[:i]:
                neighbors[a][b] += 1
                neighbors[b][a] += 1
    profiles = []
    for i, adjacent in enumerate(neighbors):
        square = {}
        for j, multiplicity in adjacent.items():
            for h, other in neighbors[j].items():
                square[h] = square.get(h, 0)+multiplicity*other
        profiles.append((
            rows[i].bit_count(),
            tuple(sorted(Counter(adjacent.values()).items())),
            square.get(i, 0),
            sum(value*adjacent.get(h, 0) for h, value in square.items()),
            tuple(sorted(Counter(square.values()).items())),
        ))
    return (tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            tuple(sorted(Counter(profiles).items())))


@lru_cache(maxsize=1)
def _long_card_bank(G):
    """Independent optimizations with invariant rejection, never orbit copies."""
    bank, emitted = [], set()
    for slot in range(N_SLOTS):
        chosen = None
        for retry in range(3):
            local = random.Random(81030317+100019*slot+104729*retry)
            try:
                finalists = _optimize_long_card(local, G, slot)
            except ValueError:
                continue
            for item in finalists:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                if _small_logical(hz, item["xs"], n):
                    continue
                if _small_logical(hx, item["zs"], n):
                    continue
                fingerprint = _walk_fingerprint(hx, hz, n)
                if fingerprint in emitted:
                    continue
                assert css_ok(hx, hz)
                assert n == 567 and 275*n <= 1000*item["k"] < 325*n
                assert max(row.bit_count() for row in hx+hz) <= W_MAX
                assert all(len(c) == len(set(c)) == 3
                           for M in (item["A"], item["B"])
                           for row in M for c in row)
                emitted.add(fingerprint)
                chosen = (
                    tuple(tuple(tuple(c) for c in row) for row in item["A"]),
                    tuple(tuple(tuple(c) for c in row) for row in item["B"]),
                )
                break
            if chosen is not None:
                break
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return this card's independently optimized, invariant-distinct design."""
    selected = _long_card_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh long-order cubic filling for this card")
    A, B = selected
    return {"idx": idx, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Saturating w=12 with rolled supports suffices to avoid weight-6 sector-1 logicals; parent measured slots 23,25,37,44,56 all retained distance 6.
# REFUTED: Component four-cycle exclusion and short-six preference alone ensure measured readings >=10 on C19 (2,5)x(2,5); current-parent measured slots 0,1 retained weight-eight sector-one minima and slots 3,10,22 retained weight-nine minima.
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

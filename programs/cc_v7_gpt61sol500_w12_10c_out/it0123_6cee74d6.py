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
CONCEPT = ("Dic11, order 44, shapes (1,2)x(3,5): packing internal A differences and adaptively shortening verified component-kernel witnesses should produce five measured distance readings of at least 15 in cell 0.150.")
GROUP = ("Frozen under L1: retain the exact Dic11 presentation, encoding and order 44. This bed scores 29.1894 and the same group's (2,3)x(2,3) bed reaches 54.9. The C4 quotient and normal C11 fibers provide constructive filling constraints and structured witnesses, without assuming distance transfer.")
SHAPES = ("Frozen (1,2)x(3,5): n=572, kb=88, guaranteed rate 2/13, and component variable/check counts 88/44 and 220/132. The size window permits q<=46; q=44 remains fixed. Parent minimum Z witnesses weigh 13-14 and use both A columns across B columns, especially Bcol1-3, entirely in sector 1. Verified ranks retain 88<=k<=100 inside cell 0.150.")
FILLING = ("Twenty canonical sparse-B architectures replace repeated degree profiles; fifteen use A=(4,5), with five (3,5)/(4,4) controls. B columns total three, or three/four in controls, and both CSS budgets are checked explicitly. C4 occupancy models exclude involution pairs and preferentially pack distinct internal A differences. Fresh proposals and adaptive redraws are selected using verified component syndromes, basis shortening, cross-information-set recombination and normal-C11 fiber witnesses. A-focused independent finalist searches validate actual LP logical embeddings. Exact logical-four and combined-Tanner invariant screening remain mandatory. Sampled readings are upper bounds, not distance certificates.")
FAMILY_NAME = "r015_dic11_12x35_sparse_pairpack_adaptive_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 44

def custom_ops(q):
    # Dic11: x^22=1, y^2=x^11, y*x*y^-1=x^-1.
    if not valid_q(q):
        raise ValueError("Dic11 requires order 44")
    els = [(a, b) for b in range(2) for a in range(22)]

    def red(g):
        a, b = g
        return ((a + 11 * (b // 2)) % 22, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 11*b*d) % 22,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 22, 0) if b == 0 else ((a + 11) % 22, 1)

    def idx(g):
        a, b = red(g)
        return 22*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


from collections import Counter
from functools import lru_cache


@lru_cache(maxsize=1)
def _d11_sparse_cards():
    """Canonical weighted B supports, not row/column permutation copies."""
    from itertools import product, permutations, combinations_with_replacement
    types = tuple(v for v in product(range(4), repeat=3) if sum(v) == 3)
    orders = tuple(permutations(range(3)))
    canonical = set()
    for columns in combinations_with_replacement(types, 5):
        totals = tuple(sum(c[i] for c in columns) for i in range(3))
        if min(totals) < 4 or max(totals) > 7:
            continue
        image = min(
            tuple(sorted(tuple(c[i] for i in order) for c in columns))
            for order in orders)
        canonical.add(image)

    def priority(columns):
        entries = tuple(x for c in columns for x in c)
        totals = tuple(sum(c[i] for c in columns) for i in range(3))
        return (entries.count(3), max(totals)-min(totals),
                entries.count(0), max(totals), columns)

    profiles = sorted(canonical, key=priority)
    if len(profiles) < 20:
        raise ValueError("fewer than twenty canonical sparse B supports")

    cards = []
    for card, columns in enumerate(profiles[:20]):
        aw = (4, 5) if card < 15 else ((3, 5) if card < 18 else (4, 4))
        sb = [[c[i] for c in columns] for i in range(3)]
        # Controls spend the extra HX budget on distinct B columns.
        extra_columns = range(3) if 15 <= card < 18 else (
            range(5) if card >= 18 else ())
        for j in extra_columns:
            available = [i for i in range(3)
                         if sum(sb[i]) < 12-max(aw) and sb[i][j] < 5]
            if not available:
                raise ValueError("sparse control exceeded its row budget")
            i = min(available, key=lambda r: (sum(sb[r]), sb[r][j], r))
            sb[i][j] += 1
        sa = [list(aw)]
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        cards.append((tuple(aw), tuple(tuple(row) for row in sb)))
    return tuple(cards)


def _d11_support(card):
    """Assign a deliberate support architecture with checked term budgets."""
    aw, bp = _d11_sparse_cards()[card % 20]
    sa, sb = [list(aw)], [list(row) for row in bp]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    return sa, sb


def _d11_ratios(G, cell):
    """Construction filter only; cycle counts use the verified matrices."""
    return {G._key(G.mul(g, G.inv(h)))
            for g in cell for h in cell if g != h}


def _d11_a_cell(rng, G, weight, mode):
    """C4 occupancy recipes with preferentially distinct internal ratios."""
    recipes = {
        3: ((1, 1, 1, 0), (2, 1, 0, 0)),
        4: ((1, 1, 1, 1), (2, 1, 1, 0),
            (2, 2, 0, 0), (3, 1, 0, 0)),
        5: ((2, 1, 1, 1), (2, 2, 1, 0), (3, 1, 1, 0),
            (3, 2, 0, 0), (2, 0, 2, 1)),
    }
    quotas = recipes[weight]
    quota = quotas[mode % len(quotas)]
    pools = [[g for g in G.elements if (2*g[0]+g[1]) % 4 == c]
             for c in range(4)]
    best = None
    for _ in range(64):
        phase, direction = rng.randrange(4), rng.choice((-1, 1))
        cell = []
        for c, count in enumerate(quota):
            cell.extend(rng.sample(pools[(phase+direction*c) % 4], count))
        if any(G.mul(G.inv(g), h) == (11, 0)
               for i, g in enumerate(cell) for h in cell[:i]):
            continue
        ratios = _d11_ratios(G, cell)
        cost = weight*(weight-1)-len(ratios)
        if best is None or cost < best[0]:
            best = (cost, cell)
        if cost == 0:
            break
    if best is None:
        raise ValueError("C4 occupancy produced no involution-free cell")
    return best[1]


def _d11_draw(rng, G, sa, sb, mode):
    """Joint A difference packing and sparse, mixed-order B labels."""
    from math import gcd

    arow, used = [], set()
    for j, weight in enumerate(sa[0]):
        chosen = None
        for _ in range(12 if used else 1):
            cell = _d11_a_cell(rng, G, weight, mode+j)
            ratios = _d11_ratios(G, cell)
            cost = (weight*(weight-1)-len(ratios)+len(ratios & used),
                    -len(ratios))
            if chosen is None or cost < chosen[0]:
                chosen = (cost, cell, ratios)
            if cost[0] == 0:
                break
        arow.append(chosen[1])
        used.update(chosen[2])

    order22 = [(a, 0) for a in range(1, 22) if gcd(a, 22) == 1]
    order4 = [(a, 1) for a in range(22)]
    pools = (order22, order4, order22+order4)
    B = []
    for i, counts in enumerate(sb):
        row = []
        for j, count in enumerate(counts):
            if count == 0:
                cell = []
            elif count == 1:
                cell = [G.pick(rng)]
            elif count == 2:
                g = G.pick(rng)
                cell = [g, G.mul(g, rng.choice(pools[(mode+i+j) % 3]))]
            else:
                cell = _d11_a_cell(rng, G, count, mode+i+j)
            assert len(cell) == count and len(set(cell)) == count
            row.append(cell)
        B.append(row)
    return [arow], B


def _d11_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            cols[low.bit_length() - 1] |= bit
            row ^= low
    return cols


def _d11_components(hx, hz, q):
    """Extract the actual component actions from the verified LP lift."""
    mask = (1 << q) - 1
    ar = [(row & mask) | (((row >> (5*q)) & mask) << q)
          for row in hx[:q]]
    bm = (1 << (5*q)) - 1
    br = [row & bm for row in hz[:3*q]]
    return (_d11_columns(ar, 2*q), _d11_columns(br, 5*q))


def _d11_floor4(cols):
    """Exact kernel screening through four; five is only a sentinel."""
    singles = set(cols)
    if 0 in singles:
        return 1
    if len(singles) != len(cols):
        return 2
    pairs = set()
    four = False
    for i, ci in enumerate(cols):
        for cj in cols[:i]:
            syndrome = ci ^ cj
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                four = True
            pairs.add(syndrome)
    return 4 if four else 5


def _d11_cycles(cols):
    overlaps = {}
    total = 0
    for col in cols:
        checks = []
        while col:
            low = col & -col
            checks.append(low.bit_length() - 1)
            col ^= low
        for i, a in enumerate(checks):
            for b in checks[:i]:
                pair = (b, a)
                old = overlaps.get(pair, 0)
                total += old
                overlaps[pair] = old + 1
    return total


def _d11_kernel_reading(cols, rng, trials):
    """Actual kernel witnesses, with structured seeds and basis shortening.

    Every syndrome is a column of the verified lift. Higher readings
    mean that this bounded search found no smaller word, not a proof.
    """
    best = len(cols) + 1
    witnesses = set()
    order = list(range(len(cols)))

    def observe(word):
        nonlocal best
        if not word:
            return
        weight = word.bit_count()
        best = min(best, weight)
        if weight <= best + 2:
            witnesses.add(word)

    # Normal-C11 fiber indicators in the frozen Dic11 encoding.
    # Determine their dependencies using verified syndromes, not a lift.
    fibers = [
        sum(1 << (a+22*b) for b in range(2) for a in range(22)
            if (2*a+b) % 4 == c)
        for c in range(4)]
    structured_pivots, dependencies = {}, []
    for block in range(len(cols)//44):
        for fiber in fibers:
            word = fiber << (44*block)
            syndrome = 0
            remaining = word
            while remaining:
                bit = remaining & -remaining
                syndrome ^= cols[bit.bit_length()-1]
                remaining ^= bit
            while syndrome:
                p = syndrome.bit_length()-1
                old = structured_pivots.get(p)
                if old is None:
                    structured_pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                dependencies.append(word)

    # Enumerate a bounded structured subspace; also retain every generator.
    structured = [0]
    for dep in dependencies[:10]:
        structured += [word ^ dep for word in structured]
    for word in structured[1:] + dependencies:
        observe(word)
    witnesses = set(sorted(witnesses,
                           key=lambda v: (v.bit_count(), v))[:128])
    memory = sorted(witnesses, key=int.bit_count)[:24]

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
                observe(word)

        # Elementary replacements preserve the component kernel and
        # shorten reducers before pair searches and seeded descent.
        for sweep in range(2):
            rng.shuffle(words)
            changed = False
            for i in range(len(words)):
                word, weight = words[i], words[i].bit_count()
                for j, other in enumerate(words):
                    if i == j:
                        continue
                    combined = word ^ other
                    cw = combined.bit_count()
                    if 0 < cw < weight:
                        word, weight = combined, cw
                        changed = True
                words[i] = word
                observe(word)
            if not changed:
                break

        words.sort(key=int.bit_count)
        for i, word in enumerate(words):
            for other in words[:i]:
                observe(word ^ other)

        # Recombine discoveries from different information sets.
        short = sorted(set(words+memory),
                       key=lambda v: (v.bit_count(), v))[:32]
        for i, word in enumerate(short):
            for other in short[:i]:
                observe(word ^ other)

        reducers = words + memory
        for start in short[:6]:
            word = start
            for sweep in range(4):
                improved, weight = word, word.bit_count()
                for other in reducers:
                    combined = word ^ other
                    cw = combined.bit_count()
                    if 0 < cw < weight:
                        improved, weight = combined, cw
                if improved == word:
                    break
                word = improved
                observe(word)

        witnesses = set(sorted(
            witnesses, key=lambda v: (v.bit_count(), v))[:128])
        memory = sorted(witnesses,
                        key=lambda v: (v.bit_count(), v))[:24]
        if best <= 4:
            break

    short = sorted(witnesses, key=lambda v: (v.bit_count(), v))[:32]
    return best, short


def _d11_basis(rows):
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in basis:
                row ^= basis[p]
            else:
                basis[p] = row
                break
    return basis


def _d11_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        previous = basis.get(p)
        if previous is None:
            return True
        word ^= previous
    return False


def _d11_small_logical(checks, stabilizers, n):
    """Exactly detect a non-stabilizer logical of weight one through four."""
    basis = _d11_basis(stabilizers)
    cols = _d11_columns(checks, n)
    representatives = {0: 0}
    bits = [1 << i for i in range(n)]
    for i, syndrome in enumerate(cols):
        word = bits[i]
        previous = representatives.get(syndrome)
        if previous is None:
            representatives[syndrome] = word
        elif _d11_outside(word ^ previous, basis):
            return True
    for i, ci in enumerate(cols):
        for j in range(i):
            syndrome = ci ^ cols[j]
            word = bits[i] | bits[j]
            previous = representatives.get(syndrome)
            if previous is None:
                representatives[syndrome] = word
            elif _d11_outside(word ^ previous, basis):
                return True
    return False


def _d11_logical_reading(words, side, hx, hz, q, n):
    """Validate component embeddings against the full opposite checks."""
    mask = (1 << q) - 1
    checks, stabilizers = (hx, hz) if side == "A" else (hz, hx)
    basis = _d11_basis(stabilizers)
    best = n + 1
    for word in words:
        weight = word.bit_count()
        if weight >= best:
            continue
        for block in range(5 if side == "A" else 2):
            if side == "A":
                full = (((word & mask) << (block*q)) |
                        ((word >> q) << ((5 + block)*q)))
            else:
                full = word << (block*5*q)
            if any((full & row).bit_count() % 2 for row in checks):
                raise ValueError("verified component embedding failed")
            if _d11_outside(full, basis):
                best = weight
                break
    return best


def _d11_fingerprint(hx, hz, n):
    """Combined-Tanner invariant, ignoring even the X/Z check colouring.

    Different fingerprints imply nonisomorphic incidence graphs.
    Matching fingerprints conservatively reject some nonisomorphic codes.
    No label, group encoding, or protograph ordering enters this invariant.
    """
    rows = hx + hz
    incidents = [[] for _ in range(n)]
    for r, row in enumerate(rows):
        while row:
            low = row & -row
            incidents[low.bit_length() - 1].append(r)
            row ^= low

    adjacency = [Counter() for _ in rows]
    for checks in incidents:
        for i, a in enumerate(checks):
            for b in checks[:i]:
                adjacency[a][b] += 1
                adjacency[b][a] += 1

    profiles = []
    for r, neighbors in enumerate(adjacency):
        square = {}
        for s, multiplicity in neighbors.items():
            for t, other in adjacency[s].items():
                square[t] = square.get(t, 0) + multiplicity*other
        triangles = sum(neighbors.get(t, 0)*value
                        for t, value in square.items())
        profiles.append((
            rows[r].bit_count(),
            tuple(sorted(Counter(neighbors.values()).items())),
            square.get(r, 0),
            triangles,
            tuple(sorted(Counter(square.values()).items())),
        ))
    return (tuple(sorted(Counter(map(len, incidents)).items())),
            tuple(sorted(Counter(profiles).items())))


@lru_cache(maxsize=1)
def _d11_bank(G):
    """Optimize sparse cards adaptively; emit only invariant-distinct codes."""
    bank, fingerprints = [], set()

    def merit(item):
        d = min(item["da"], item["db"], 20)
        # A count of found short words is a tie-breaker, not an enumeration
        # of all minimum words or a distance certificate.
        load = sum(word.bit_count() <= d+1
                   for word in item["wa"]+item["wb"])
        return (item["k"]*d*d/item["n"], d, -load,
                min(item["da"], 24)+min(item["db"], 24),
                -item["cycles"])

    for slot in range(N_SLOTS):
        card = slot % 20
        sa, sb = _d11_support(card)
        rng = random.Random(15044029 + 100019*slot)
        chosen = None

        for batch in range(2):
            finalists, seen = [], set()

            def admit(A, B):
                labeling = tuple(tuple(sorted(cell))
                                 for grid in (A, B)
                                 for row in grid for cell in row)
                if labeling in seen:
                    return
                seen.add(labeling)
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if n != 572 or not 88 <= k <= 100:
                    return
                ac, bc = _d11_components(hx, hz, G.q)
                if min(_d11_floor4(ac), _d11_floor4(bc)) < 5:
                    return
                da, wa = _d11_kernel_reading(ac, rng, 3)
                db, wb = _d11_kernel_reading(bc, rng, 2)
                item = {
                    "A": A, "B": B, "hx": hx, "hz": hz,
                    "n": n, "k": k, "ac": ac, "bc": bc,
                    "da": da, "db": db, "wa": wa, "wb": wb,
                    "cycles": _d11_cycles(ac)+_d11_cycles(bc),
                }
                item["key"] = merit(item)
                finalists.append(item)
                finalists.sort(key=lambda x: x["key"], reverse=True)
                del finalists[8:]

            for attempt in range(48):
                # Every architecture explores several quotient recipes;
                # no recipe is obtained by relabeling a previous code.
                mode = (card+attempt+2*batch) % 5
                admit(*_d11_draw(rng, G, sa, sb, mode))

            for step in range(24):
                if not finalists:
                    admit(*_d11_draw(rng, G, sa, sb, (card+step) % 5))
                    continue
                parent = finalists[rng.randrange(min(4, len(finalists)))]
                A = [[list(cell) for cell in row] for row in parent["A"]]
                B = [[list(cell) for cell in row] for row in parent["B"]]
                proposal_A, proposal_B = _d11_draw(
                    rng, G, sa, sb, (card+step+batch+2) % 5)

                # Parent measurements identify A as the usual Z bottleneck.
                # Still explore B periodically or whenever it is weaker.
                side = ("B" if step % 5 == 0
                        or parent["db"] < parent["da"] else "A")
                if side == "A":
                    if step % 6 == 5:
                        A = proposal_A
                    else:
                        j = rng.randrange(2)
                        A[0][j] = proposal_A[0][j]
                else:
                    positions = [(i, j) for i in range(3)
                                 for j in range(5) if sb[i][j]]
                    for _ in range(2 if step % 6 == 5 else 1):
                        i, j = rng.choice(positions)
                        B[i][j] = proposal_B[i][j]
                admit(A, B)

            # Independent searches can only lower retained readings.
            for item in finalists:
                for side, trials in (("a", 24), ("b", 8)):
                    reading, words = _d11_kernel_reading(
                        item[side+"c"], rng, trials)
                    item["d"+side] = min(item["d"+side], reading)
                    item["w"+side] = sorted(
                        set(item["w"+side]+words),
                        key=lambda v: (v.bit_count(), v))[:48]
                item["key"] = merit(item)
            finalists.sort(key=lambda x: x["key"], reverse=True)

            deep = []
            for item in finalists[:4]:
                for side, trials in (("a", 80), ("b", 24)):
                    reading, words = _d11_kernel_reading(
                        item[side+"c"], rng, trials)
                    item["d"+side] = min(item["d"+side], reading)
                    item["w"+side] = sorted(
                        set(item["w"+side]+words),
                        key=lambda v: (v.bit_count(), v))[:48]

                dz = _d11_logical_reading(
                    item["wa"], "A", item["hx"], item["hz"], G.q, item["n"])
                dx = _d11_logical_reading(
                    item["wb"], "B", item["hx"], item["hz"], G.q, item["n"])
                if max(dx, dz) > item["n"]:
                    continue
                d = min(dx, dz, item["da"], item["db"], 20)
                load = sum(word.bit_count() <= d+1
                           for word in item["wa"]+item["wb"])
                key = (item["k"]*d*d/item["n"], d, -load,
                       min(dx, 28)+min(dz, 28), -item["cycles"])
                deep.append((key, item))

            deep.sort(key=lambda pair: pair[0], reverse=True)
            for key, item in deep:
                hx, hz, n = item["hx"], item["hz"], item["n"]
                if _d11_small_logical(hx, hz, n):
                    continue
                if _d11_small_logical(hz, hx, n):
                    continue
                fingerprint = _d11_fingerprint(hx, hz, n)
                if fingerprint in fingerprints:
                    continue
                assert css_ok(hx, hz)
                assert 125*n <= 1000*item["k"] < 175*n
                assert max(row.bit_count() for row in hx+hz) <= W_MAX
                assert all(len(cell) == len(set(cell)) <= 5
                           for grid in (item["A"], item["B"])
                           for row in grid for cell in row)
                chosen = (item["A"], item["B"], fingerprint)
                break
            if chosen is not None:
                break

        if chosen is None:
            bank.append(None)
        else:
            A, B, fingerprint = chosen
            fingerprints.add(fingerprint)
            bank.append((A, B))

    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign one independently optimized, incidence-distinct card."""
    selected = _d11_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh rate-legal filling passed logical-four screening")
    A, B = selected
    return {"idx": idx, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without resampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: D8 (2,7)x(2,8) distinct monomial relatives and short-syndrome selection did not break the distance-four plateau: all five measured readings were 4, with both logical types entirely in sector 1.
# REFUTED: Five degree tradeoffs crossed with four internal-difference models suffice for five measured readings >=14 on Dic11 (1,2)x(3,5); the parent measured 13,14,14,13,14, limited by sector-1 Z witnesses.
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

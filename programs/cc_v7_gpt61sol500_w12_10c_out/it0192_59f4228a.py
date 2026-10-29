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
CONCEPT = ("Dic9, order 36, shapes (2,3)x(2,4): long-rotation binomials, contrasting trinomial coset occupancies and normal-C3 witness screening should raise five-code fitness above 37.4158 in cell 0.150; the measured target is five distance readings >=18.")
GROUP = ("Frozen L1: retain the exact Dic9 presentation, encoding and order 36. This bed scores 37.4, while the related Dic11 (2,3)x(2,3) bed reaches 54.9; neither establishes distance transfer. Normal C3 and C9 cosets supply structured witnesses, whose syndromes are computed exclusively from verified lifted matrices.")
SHAPES = ("Frozen A=(2,3), B=(2,4): n=576, kb=72, guaranteed rate 1/8, and component variable/check counts 108/72 and 144/72. The size window permits q<=37; q=36 stays fixed. Four measured leaders have weight-16 X witnesses entirely in sector 1, spread over every B column and most frequently using Acol1; the best leader reaches 17. Verified ranks retain 72<=k<=100, entirely inside cell 0.150.")
FILLING = ("Twenty cards cross five A support profiles, two B profiles and two trinomial coset recipes. Every component column still has four terms, A/B row totals are six/eight, and HX/HZ weights remain 10/12. Binomials use rotation differences of order 18 or 9, with an order-18 binomial in every (2,2) column. Trinomials contrast mixed-coset long-gap supports against single-coset triples; involution pairs remain excluded. Exact C3-coset dependency collisions supply additional short kernel witnesses. B receives more proposals, witness-directed repairs and independent rescreening. Existing verified dressing, exact logical-four, target-rate and combined-Tanner invariant checks remain. Readings are upper bounds, never distance certificates.")
FAMILY_NAME = "r015_dic9_23x24_rotation18_c3_rescreen_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    # Same dicyclic rule, resized: x^18=1, y^2=x^9,
    # y*x*y^-1=x^-1; normal form x^a y^b.
    if not valid_q(q):
        raise ValueError("Dic9 requires order 36")
    els = [(a, b) for b in range(2) for a in range(18)]

    def red(g):
        a, b = g
        return ((a + 9 * (b // 2)) % 18, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 9*b*d) % 18,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 18, 0) if b == 0 else ((a + 9) % 18, 1)

    def idx(g):
        a, b = red(g)
        return 18*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


from collections import Counter
from functools import lru_cache

_DIC_N = 9
_LIFT_Q = 4 * _DIC_N


@lru_cache(maxsize=1)
def _d11_sparse_cards():
    """Ten support pairs crossed with two prescribed coset recipes.

    Repeated counts are deliberate different element models, not copies
    of one labeling. The bank also rejects equal incidence invariants.
    """
    a_profiles = (
        (2, 2, 2), (1, 2, 3), (0, 3, 3),
        (1, 1, 4), (0, 2, 4),
    )
    # These B profiles supplied the parent's strongest measured readings.
    b_profiles = ((1, 1, 3, 3), (1, 2, 2, 3))
    cards = []
    for recipe in range(2):
        for a in a_profiles:
            for b in b_profiles:
                sa = (a, tuple(4-x for x in a))
                sb = (b, tuple(4-x for x in b))
                assert tuple(map(sum, sa)) == (6, 6)
                assert tuple(map(sum, sb)) == (8, 8)
                assert _grid_weights(sa, sb) <= W_MAX
                assert _no_dead_lines(sa, sb)
                cards.append((sa, sb))
    assert len(cards) == 20
    return tuple(cards)


def _d11_support(card):
    """Complete new-shape count grids; neither side has a dead line."""
    ap, bp = _d11_sparse_cards()[card % 20]
    sa, sb = [list(row) for row in ap], [list(row) for row in bp]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert all(sum(row[j] for row in sa) == 4
               for j in range(A_SHAPE[1]))
    assert all(sum(row[j] for row in sb) == 4
               for j in range(B_SHAPE[1]))
    return sa, sb


def _d11_ratios(G, cell):
    """Construction filter only; cycle counts use the verified matrices."""
    return {G._key(G.mul(g, G.inv(h)))
            for g in cell for h in cell if g != h}


def _d11_a_cell(rng, G, weight, mode):
    """Replace order-four binomials; contrast actual coset occupancies."""
    from math import gcd

    if weight == 0:
        return []
    if weight == 1:
        return [G.pick(rng)]
    m = 2 * _DIC_N
    involution = (_DIC_N, 0)
    long_rotations = [(a, 0) for a in range(1, m)
                      if gcd(a, m) == 1]
    order9 = [(a, 0) for a in range(2, m, 2)
              if gcd(a, m) == 2]
    if weight == 2:
        # Adjacent row modes have opposite parity, so a (2,2)
        # column always includes an order-18 relative element.
        pool = long_rotations if mode % 2 == 0 else long_rotations + order9
        g = G.pick(rng)
        cell = [g, G.mul(g, rng.choice(pool))]
        assert len(set(cell)) == 2
        return cell

    pools = [[g for g in G.elements if g[1] == b] for b in range(2)]
    best = None
    for _ in range(64):
        b = rng.randrange(2)
        if weight == 3 and mode % 2 == 0:
            first = rng.choice(pools[b])
            cell = [first, G.mul(first, rng.choice(long_rotations)),
                    rng.choice(pools[1-b])]
        else:
            # Single-coset triples change relative-order structure,
            # not merely the translate of a mixed-coset triple.
            major = (3 if weight == 3 else (3, 2, 4)[mode % 3])
            cell = rng.sample(pools[b], major)
            cell += rng.sample(pools[1-b], weight-major)
        if len(set(cell)) != weight:
            continue
        differences = [G.mul(G.inv(g), h)
                       for i, g in enumerate(cell) for h in cell[:i]]
        if involution in differences:
            continue
        if not any(d[1] == 0 and gcd(d[0], m) == 1
                   for d in differences):
            continue
        ratios = _d11_ratios(G, cell)
        cost = weight*(weight-1)-len(ratios)
        if best is None or cost < best[0]:
            best = (cost, cell)
        if cost == 0:
            break
    if best is None:
        raise ValueError("long-rotation coset recipe produced no legal cell")
    assert len(best[1]) == len(set(best[1])) == weight
    return best[1]


def _d11_draw(rng, G, sa, sb, mode):
    """Pack both two-row components; counts remain fixed for the card."""
    def grid(supports, offset):
        result = []
        for i, counts in enumerate(supports):
            row, used = [], set()
            for j, weight in enumerate(counts):
                chosen = None
                for _ in range(8 if used and weight > 1 else 1):
                    cell = _d11_a_cell(
                        rng, G, weight, mode + offset + i + j)
                    ratios = _d11_ratios(G, cell)
                    cost = (weight*(weight-1)-len(ratios)
                            + len(ratios & used), -len(ratios))
                    if chosen is None or cost < chosen[0]:
                        chosen = (cost, cell, ratios)
                    if cost[0] == 0:
                        break
                row.append(chosen[1])
                used.update(chosen[2])
            result.append(row)
        return result

    A, B = grid(sa, 0), grid(sb, 1)
    assert all(len(cell) == len(set(cell)) <= 4
               for M in (A, B) for row in M for cell in row)
    return A, B


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
    """Extract both new-shape actions from the verified LP matrices."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for r in range(q):
            source = hx[(i * cb) * q + r]
            compact = 0
            for j in range(ca):
                compact |= ((source >> (j * cb * q)) & mask) << (j * q)
            ar.append(compact)
    bm = (1 << (cb * q)) - 1
    br = [row & bm for row in hz[:rb*q]]
    return (_d11_columns(ar, ca*q), _d11_columns(br, cb*q))


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


@lru_cache(maxsize=128)
def _dic9_c3_words(cols):
    """Actual component words constant on normal-C3 cosets.

    Cosets are disjoint triples. Collisions between subsets of at most
    three cosets cover every restricted kernel word using <=6 cosets.
    This is not an exclusion test for arbitrary component words.
    """
    q, m = _LIFT_Q, 2 * _DIC_N
    local = [
        sum(1 << (a+m*b) for a in range(m) if a % 6 == residue)
        for b in range(2) for residue in range(6)]
    fibers = [word << (block*q)
              for block in range(len(cols)//q) for word in local]
    syndromes = []
    for word in fibers:
        syndrome, remaining = 0, word
        while remaining:
            bit = remaining & -remaining
            syndrome ^= cols[bit.bit_length()-1]
            remaining ^= bit
        syndromes.append(syndrome)

    representatives = {0: 0}
    words = set()

    def insert(syndrome, word):
        old = representatives.get(syndrome)
        if old is None:
            representatives[syndrome] = word
        else:
            dependency = old ^ word
            if dependency:
                words.add(dependency)
        if len(words) > 256:
            short = sorted(words, key=lambda v: (v.bit_count(), v))[:64]
            words.clear()
            words.update(short)

    for i, syndrome in enumerate(syndromes):
        insert(syndrome, fibers[i])
    for i, si in enumerate(syndromes):
        for j in range(i):
            insert(si ^ syndromes[j], fibers[i] | fibers[j])
    for i, si in enumerate(syndromes):
        for j in range(i):
            pair = si ^ syndromes[j]
            word = fibers[i] | fibers[j]
            for h in range(j):
                insert(pair ^ syndromes[h], word | fibers[h])
    return tuple(sorted(words, key=lambda v: (v.bit_count(), v))[:48])


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

    # Include smaller normal-C3 cosets before the old C9 seeds.
    # Every supplied word has zero syndrome in this exact component.
    for word in _dic9_c3_words(tuple(cols)):
        observe(word)

    # Normal-C9 fibers in the resized encoding. Their dependencies
    # are determined exclusively from the verified component syndromes.
    m, q = 2 * _DIC_N, _LIFT_Q
    fibers = [
        sum(1 << (a+m*b) for b in range(2) for a in range(m)
            if (2*a+b) % 4 == c)
        for c in range(4)]
    structured_pivots, dependencies = {}, []
    for block in range(len(cols)//q):
        for fiber in fibers:
            word = fiber << (q*block)
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
    """Validate new-shape embeddings, then shorten their logical classes.

    Adding actual stabilizer rows preserves the verified logical class.
    Pair dressings can reach mixed-sector representatives. Failure to
    find a shorter representative is not a distance certificate.
    """
    ca, cb = A_SHAPE[1], B_SHAPE[1]
    mask = (1 << q) - 1
    checks, stabilizers = (hx, hz) if side == "A" else (hz, hx)
    basis = _d11_basis(stabilizers)
    best, dressed = n + 1, 0
    seeds = sorted(set(words), key=lambda v: (v.bit_count(), v))[:16]
    for word in seeds:
        if not word or word.bit_count() > best + 4:
            continue
        for block in range(cb if side == "A" else ca):
            if side == "A":
                full = 0
                for j in range(ca):
                    full |= ((word >> (j*q)) & mask) << (
                        (j*cb + block)*q)
            else:
                full = word << (block*cb*q)
            if any((full & row).bit_count() % 2 for row in checks):
                raise ValueError("verified component embedding failed")
            if not _d11_outside(full, basis):
                continue
            best = min(best, full.bit_count())
            if dressed >= 8:
                continue
            dressed += 1
            for sweep in range(3):
                chosen, weight = full, full.bit_count()
                nearby = []
                for row in stabilizers:
                    trial = full ^ row
                    tw = trial.bit_count()
                    nearby.append((tw, row))
                    if 0 < tw < weight:
                        chosen, weight = trial, tw
                nearby.sort(key=lambda item: (item[0], item[1]))
                short_rows = [row for _, row in nearby[:12]]
                for i, row in enumerate(short_rows):
                    for other in short_rows[:i]:
                        trial = full ^ row ^ other
                        tw = trial.bit_count()
                        if 0 < tw < weight:
                            chosen, weight = trial, tw
                if chosen == full:
                    break
                full = chosen
                best = min(best, weight)
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
    """Optimize long-rotation cards; emit only invariant-distinct codes."""
    bank, fingerprints = [], set()
    assert G.q == _LIFT_Q

    def merit(item):
        d = min(item["da"], item["db"], 26)
        # A count of found short words is a tie-breaker, not an enumeration
        # of all minimum words or a distance certificate.
        load = sum(word.bit_count() <= d+1
                   for word in item["wa"]+item["wb"])
        return (item["k"]*d*d/item["n"], d, -load,
                min(item["da"], 30)+min(item["db"], 30),
                -item["cycles"])

    for slot in range(N_SLOTS):
        card = slot % 20
        sa, sb = _d11_support(card)
        rng = random.Random(15036183 + 100043*slot)
        recipe = card // 10 + 2 * (slot // 20)
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
                if n != 576 or not 72 <= k <= 100:
                    return
                ac, bc = _d11_components(hx, hz, G.q)
                if min(_d11_floor4(ac), _d11_floor4(bc)) < 5:
                    return
                da, wa = _d11_kernel_reading(ac, rng, 3)
                db, wb = _d11_kernel_reading(bc, rng, 6)
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

            for attempt in range(64):
                # Keep the card's coset recipe fixed. Each proposal
                # changes actual voltages, never a common relabeling.
                admit(*_d11_draw(rng, G, sa, sb, recipe))

            for step in range(48):
                if not finalists:
                    admit(*_d11_draw(rng, G, sa, sb, recipe))
                    continue
                parent = finalists[rng.randrange(min(4, len(finalists)))]
                A = [[list(cell) for cell in row] for row in parent["A"]]
                B = [[list(cell) for cell in row] for row in parent["B"]]
                proposal_A, proposal_B = _d11_draw(
                    rng, G, sa, sb, recipe)

                # The measured parent is usually X/B limited. Repair B
                # on ties, but still repair A when it is genuinely weaker.
                side = ("B" if parent["db"] <= parent["da"] else "A")
                if step % 6 == 0:
                    side = "A" if side == "B" else "B"
                counts = sa if side == "A" else sb
                grid = A if side == "A" else B
                proposal = proposal_A if side == "A" else proposal_B
                positions = [(i, j) for i, row in enumerate(counts)
                             for j, count in enumerate(row) if count]
                seeds = parent["wa"] if side == "A" else parent["wb"]
                mask = (1 << G.q) - 1
                pressure = [
                    1 + sum(((word >> (j*G.q)) & mask).bit_count()
                            for word in seeds[:8])
                    for i, j in positions]
                for _ in range(2 if step % 6 == 5 else 1):
                    i, j = rng.choices(positions, weights=pressure, k=1)[0]
                    grid[i][j] = proposal[i][j]
                admit(A, B)

            # Independent searches can only lower retained readings.
            for item in finalists:
                for side, trials in (("a", 16), ("b", 40)):
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
                for side, trials in (("a", 48), ("b", 96)):
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
                d = min(dx, dz, item["da"], item["db"], 26)
                load = sum(word.bit_count() <= d+1
                           for word in item["wa"]+item["wb"])
                key = (item["k"]*d*d/item["n"], d, -load,
                       min(dx, 32)+min(dz, 32), -item["cycles"])
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
# REFUTED: Sparse-B architectures, internal A difference packing and adaptive component shortening on Dic11 (1,2)x(3,5) yield five measured readings >=15; all five measured parent readings remained 14, with the limiting Z families entirely in sector 1.
# REFUTED: Balanced degree-four columns and involution-free mixed binomials alone remove the sector-1 weight-16 X bottleneck on Dic9 (2,3)x(2,4); four measured leaders retained weight-16 X families, and the best measured distance reading was 17.
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

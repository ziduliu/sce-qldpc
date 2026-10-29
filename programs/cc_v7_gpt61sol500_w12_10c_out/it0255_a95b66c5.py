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
CONCEPT = ("On SL(2,3), order 24, shapes (1,3)x(2,7), common-image noncommuting trinomials and component-six exclusion should improve five-code fitness beyond 16.398 in cell 0.450. The constructive targets are verified rank surplus with retained distance-six readings, or lighter six-word families with higher measured readings; neither component screening nor sampling certifies quantum distance.")
GROUP = ("Jump from cyclic Z23 to the determinant-one 2x2 matrix group over F3, SL(2,3). It has order 24, centre two and seven conjugacy classes. The related order-24 seven-class bed scored about 15.7 on these shapes, so a group jump alone is not the hypothesis. Search actual singular block images and noncommuting relative labels. Order 24 gives n=552.")
SHAPES = ("Keep A=(1,3), B=(2,7) for cell 0.450: n=552, kb=240, guaranteed rate 10/23=0.434783. Components have 72 variables/24 checks and 168 variables/48 checks. The shape admits q<=26; this group fixes q=24. Parent minima were entirely sector-1 Acol/Bcol, especially Acol1 across many B columns. Exact full ranks retain 0.425<=k/n<0.475; component image deficits are proposals, not assumed formulas for k.")
FILLING = ("Twenty cards cross five A common-image rank preferences, 20 through 24, with four balanced B architectures. Missing image-rank strata explicitly use nearby screened strata rather than archived fillings. A row sum and both B row sums are nine; B column sums are at most three, giving HX/HZ<=12. Compile component actions only from verified single-term lifts. Exclude component words through five, test existence through six with bounded exact constituent decoding, count retained six-word families, then screen full CSS logicals through four and refine actual mixed-sector witnesses. Different combined uncoloured incidence fingerprints certify emitted nonisomorphism. Witness readings remain upper bounds.")
FAMILY_NAME = "r045_SL2F3_13x27_commonimage_component6_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    # SL(2,F3), with matrices encoded row-major.
    if not valid_q(q):
        raise ValueError("SL(2,3) requires order 24")
    els = [(a,b,c,d) for a in range(3) for b in range(3)
           for c in range(3) for d in range(3)
           if (a*d-b*c) % 3 == 1]
    positions = {g:i for i,g in enumerate(els)}

    def red(g):
        return tuple(x % 3 for x in g)

    def mul(g, h):
        a,b,c,d = red(g)
        e,f,u,v = red(h)
        return ((a*e+b*u) % 3, (a*f+b*v) % 3,
                (c*e+d*u) % 3, (c*f+d*v) % 3)

    def inv(g):
        a,b,c,d = red(g)
        return (d, (-b) % 3, (-c) % 3, a)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(1,0,0,1), mul=mul,
                inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def _q85_basis(rows):
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


def _q85_outside(v, basis):
    while v:
        p = v.bit_length() - 1
        if p not in basis:
            return True
        v ^= basis[p]
    return False


def _q85_short_screen(hx, hz, n):
    """Exact existence test for nontrivial CSS logicals of weight <=4.

    Every such support splits into two supports of size <=2.
    Equal-syndrome collisions are tested modulo the actual stabilizer
    row space. Keeping one representative per syndrome is sufficient:
    if all collisions with it are stabilizers, their pairwise differences
    are stabilizers too.
    """
    cycles = 0
    for checks, stabilizers in ((hz, hx), (hx, hz)):
        basis = _q85_basis(stabilizers)
        cols = [0] * n
        for r, row in enumerate(checks):
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= 1 << r
                row ^= bit

        # Count actual Tanner 4-cycles, not protograph coincidences.
        overlaps = {}
        for col in cols:
            ids = []
            v = col
            while v:
                bit = v & -v
                ids.append(bit.bit_length() - 1)
                v ^= bit
            for a in range(len(ids)):
                for b in range(a):
                    key = (ids[b], ids[a])
                    old = overlaps.get(key, 0)
                    cycles += old
                    overlaps[key] = old + 1

        representatives = {0: 0}
        bits = [1 << i for i in range(n)]
        for i, syndrome in enumerate(cols):
            mask = bits[i]
            if syndrome in representatives:
                v = mask ^ representatives[syndrome]
                if _q85_outside(v, basis):
                    return v.bit_count(), cycles
            else:
                representatives[syndrome] = mask

        for i in range(n):
            si, bi = cols[i], bits[i]
            for j in range(i):
                syndrome = si ^ cols[j]
                mask = bi | bits[j]
                if syndrome in representatives:
                    v = mask ^ representatives[syndrome]
                    if _q85_outside(v, basis):
                        return v.bit_count(), cycles
                else:
                    representatives[syndrome] = mask

    # Sentinel: no logical of weight <=4, not a distance estimate.
    return 5, cycles


def _z23_columns(rows, length):
    cols = [0] * length
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _z23_components(hx, hz, q):
    """Extract both components from the scaffold's exact lift."""
    ca, cb, rb = A_SHAPE[1], B_SHAPE[1], B_SHAPE[0]
    mask = (1 << q) - 1
    ar = []
    for row in hx[:q]:
        compact = 0
        for j in range(ca):
            compact |= ((row >> (j * cb * q)) & mask) << (j * q)
        ar.append(compact)
    bmask = (1 << (cb * q)) - 1
    br = [row & bmask for row in hz[:rb * q]]
    return (_z23_columns(ar, ca * q),
            _z23_columns(br, cb * q))


def _z23_short(cols, through_five=False):
    """Exact existence screening, not a heuristic distance reading.

    Singles/pairs cover weight <=4. Triples against their representatives
    cover weight <=5 without storing the much larger triple dictionary.
    The sentinel is 5 or 6 when the requested weights were excluded.
    """
    seen = {0: 0}
    bits = [1 << i for i in range(len(cols))]
    for i, syndrome in enumerate(cols):
        word = bits[i]
        old = seen.get(syndrome)
        if old is not None:
            return (old ^ word).bit_count()
        seen[syndrome] = word
    for i, ci in enumerate(cols):
        for j in range(i):
            syndrome = ci ^ cols[j]
            word = bits[i] | bits[j]
            old = seen.get(syndrome)
            if old is not None:
                return (old ^ word).bit_count()
            seen[syndrome] = word
    if not through_five:
        return 5
    for i, ci in enumerate(cols):
        for j in range(i):
            syndrome = ci ^ cols[j]
            word = bits[i] | bits[j]
            for t in range(j):
                old = seen.get(syndrome ^ cols[t])
                if old is not None:
                    return (old ^ word ^ bits[t]).bit_count()
    return 6


def _z23_reading(cols, rng, trials):
    """Upper-bound readings from actual nonzero component-kernel words."""
    best = len(cols) + 1
    order = list(range(len(cols)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots, words = {}, []
        for i in order:
            syndrome, word = cols[i], 1 << i
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
                best = min(best, word.bit_count())
        words.sort(key=int.bit_count)
        for i, word in enumerate(words):
            for other in words[:i]:
                best = min(best, (word ^ other).bit_count())
        small = words[:10]
        for i, word in enumerate(small):
            for j in range(i):
                pair = word ^ small[j]
                for t in range(j):
                    best = min(best, (pair ^ small[t]).bit_count())
        if best <= 4:
            break
    return best


def _z23_cycles(cols):
    total = 0
    for i, col in enumerate(cols):
        for other in cols[:i]:
            overlap = (col & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _z23_candidate(rng, G, sb, ap_count):
    """Fresh labels for the card, never an automorphism-copy generator."""
    q, cb = G.q, B_SHAPE[1]
    A = [[]]
    for j in range(A_SHAPE[1]):
        if j < ap_count:
            step = rng.randrange(1, q)
            offsets = (0, step, (2 * step) % q)
        else:
            while True:
                u, v = rng.sample(range(1, q), 2)
                # None of the three terms is an AP midpoint.
                if (v != (2 * u) % q and u != (2 * v) % q
                        and (u + v) % q != 0):
                    offsets = (0, u, v)
                    break
        base = G.pick(rng)
        A[0].append([G.mul(base, (a,)) for a in offsets])

    # Distinct relative labels prevent repeated all-monomial columns.
    relatives = rng.sample(range(q), cb)
    B = [[None] * cb for _ in range(2)]
    for j, relative in enumerate(relatives):
        top = G.pick(rng)
        bottom = G.mul(top, (relative,))
        B[0][j], B[1][j] = [top], [bottom]
    for i in range(2):
        for j in range(cb):
            if sb[i][j] == 2:
                delta = (rng.randrange(1, q),)
                B[i][j].append(G.mul(B[i][j][0], delta))
    return A, B


def _retired_z23_context(rng, G, idx):
    """Historical cyclic policy; the SL(2,3) builder never calls this."""
    q = G.q
    assert q == 23
    style = idx % 7
    geometry = (idx // 7) % 3
    batch = idx // 21

    # Each entry is the top term count of a degree-three B column.
    # The complementary bottom counts also sum to six. Three
    # singleton/singleton columns complete both row sums to nine.
    top_cards = ((0, 0, 3, 3), (0, 1, 2, 3), (0, 2, 2, 2))
    top_counts = top_cards[geometry]
    sa = [[3, 3, 3]]
    sb = [list(top_counts) + [1, 1, 1],
          [3 - t for t in top_counts] + [1, 1, 1]]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)

    def unsigned(t):
        t %= q
        return min(t, q - t)

    ordinary, progressions = [], []
    for u in range(1, q):
        for v in range(u + 1, q):
            offsets = (0, u, v)
            differences = frozenset(
                (unsigned(u), unsigned(v), unsigned(v - u)))
            ap = (v == 2*u % q or u == 2*v % q
                  or (u + v) % q == 0)
            record = (offsets, differences)
            (progressions if ap else ordinary).append(record)

    def draw_A():
        # Styles 0..3 prescribe exactly 0..3 repeated unsigned
        # differences across three individually non-AP trinomials.
        # Styles 4..6 prescribe one, two, or three AP cells.
        ap_count = 0 if style < 4 else style - 3
        for _ in range(4096):
            records = [
                rng.choice(progressions if j < ap_count else ordinary)
                for j in range(3)]
            union = frozenset().union(*(r[1] for r in records))
            if style < 4 and len(union) != 9 - style:
                continue
            A = [[[(a,) for a in record[0]] for record in records]]
            return A, union
        raise ValueError("could not realize prescribed A difference policy")

    def six_word_count(cols):
        """Exact number of weight-six component kernel words.

        Called only after weights 1..5 have been exhaustively excluded.
        Equal-syndrome distinct triples must then be disjoint. Every
        sextet has ten unordered 3+3 partitions, so collision count/10
        counts words, rather than duplicate partitions.
        """
        counts = {}
        collisions = 0
        for i, ci in enumerate(cols):
            for j in range(i):
                pair = ci ^ cols[j]
                for t in range(j):
                    syndrome = pair ^ cols[t]
                    old = counts.get(syndrome, 0)
                    collisions += old
                    counts[syndrome] = old + 1
        assert collisions % 10 == 0
        return collisions // 10

    # Select A policies using the scaffold's action, independently of
    # B's eventual labels. A dummy partner is used only for extraction.
    dummy_B = [[[G.e] for _ in range(7)] for _ in range(2)]
    a_pool, seen_a = [], set()
    wanted_overlap = (2, 3, 1)[batch % 3]
    for attempt in range(512):
        A, differences = draw_A()
        signature = tuple(tuple(g[0] for g in cell) for cell in A[0])
        if signature in seen_a:
            continue
        seen_a.add(signature)
        hx, hz, n = lifted_matrices(G, A, dummy_B)
        ac, _ = _z23_components(hx, hz, q)
        if _z23_short(ac) != 5:
            continue
        if _z23_short(ac, through_five=True) != 6:
            continue

        # This is a relative-geometry coupling, not a common
        # automorphism applied to an already existing filling.
        mono_pool = [
            record for record in ordinary
            if len(record[1] & differences) == wanted_overlap]
        if not mono_pool:
            continue
        a_pool.append({
            "A": A, "ac": ac, "mono": mono_pool,
            "cycles": _z23_cycles(ac), "six": None,
        })
        if len(a_pool) >= 12:
            break

    if not a_pool:
        raise ValueError("A policy found no component-clean filling")

    def draw_B(a_item):
        B = [[[] for _ in range(7)] for _ in range(2)]
        for j, top_count in enumerate(top_counts):
            if top_count == 0:
                offsets = rng.choice(ordinary)[0]
                B[1][j] = [(a,) for a in offsets]
            elif top_count == 3:
                offsets = rng.choice(ordinary)[0]
                B[0][j] = [(a,) for a in offsets]
            elif top_count == 2:
                delta = rng.randrange(1, q)
                B[0][j] = [G.e, (delta,)]
                B[1][j] = [G.pick(rng)]
            else:
                B[0][j] = [G.e]
                B[1][j] = rng.sample(G.elements, 2)

        # Distinct ordered differences of these three matching
        # relatives exclude four-cycles in their degree-two subgraph.
        # The AP-free template also enforces the slot's A/B overlap.
        relatives = rng.choice(a_item["mono"])[0]
        for j, relative in enumerate(relatives, 4):
            B[0][j] = [G.e]
            B[1][j] = [(relative,)]
        return B

    finalists, seen = [], set()
    accepted = 0
    for attempt in range(192):
        a_item = rng.choice(a_pool)
        A, B = a_item["A"], draw_B(a_item)
        signature = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if signature in seen:
            continue
        seen.add(signature)

        hx, hz, n = lifted_matrices(G, A, B)
        ac, bc = _z23_components(hx, hz, q)
        if _z23_short(bc) != 5:
            continue
        if _z23_short(bc, through_five=True) != 6:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (425*n <= 1000*k < 475*n):
            continue

        cycles = a_item["cycles"] + _z23_cycles(bc)
        item = {
            "A": A, "B": B, "hx": hx, "hz": hz, "n": n,
            "k": k, "bc": bc, "a_item": a_item, "cycles": cycles,
        }
        finalists.append(item)
        finalists.sort(
            key=lambda x: (x["k"], -x["cycles"]), reverse=True)
        del finalists[4:]
        accepted += 1
        if accepted >= 12:
            break

    if not finalists:
        raise ValueError("B policy found no component-clean filling")

    winner = None
    for item in finalists:
        low, cycles = _q85_short_screen(
            item["hx"], item["hz"], item["n"])
        if low != 5:
            continue
        a_item = item["a_item"]
        if a_item["six"] is None:
            a_item["six"] = six_word_count(a_item["ac"])
        na = a_item["six"]
        nb = six_word_count(item["bc"])
        # Exact component counts are not counts of quantum logicals.
        # Prefer rank first; do not reward a sampled reading above
        # the unavoidable weight-six A component relation.
        key = (item["k"], -(na + nb), -max(na, nb), -cycles)
        if winner is None or key > winner[0]:
            winner = (key, item, na, nb)

    if winner is None:
        raise ValueError("no finalist passed full CSS logical-four exclusion")
    key, item, na, nb = winner
    assert css_ok(item["hx"], item["hz"])
    return {
        "idx": idx, "cells": {"A": item["A"], "B": item["B"]},
        "k": item["k"], "screen": key,
        "component_six_words": (na, nb),
        "policy": (geometry, style, wanted_overlap),
    }


from functools import lru_cache


_T24_B_CARDS = (
    ((2,1), (2,1), (1,2), (1,2), (1,1), (1,1), (1,1)),
    ((3,0), (1,2), (1,2), (1,2), (1,1), (1,1), (1,1)),
    ((3,0), (0,3), (2,1), (1,2), (1,1), (1,1), (1,1)),
    ((3,0), (3,0), (0,3), (0,3), (1,1), (1,1), (1,1)),
)


def _t24_span(cols):
    """Canonical image space in the verified check coordinates."""
    basis = _q85_basis(cols)
    pivots = sorted(basis)
    for a, p in enumerate(pivots):
        for higher in pivots[a+1:]:
            if basis[higher] & (1 << p):
                basis[higher] ^= basis[p]
    return tuple((p, basis[p]) for p in pivots)


def _t24_component_floor(cols, q):
    """Exact component exclusion through six after exclusion through four.

    Every regular component action is transitive within each variable
    block. Thus coordinate-zero anchors cover every component support
    orbit. This transitivity is NOT assumed for full quantum logicals.

    Singleton/pair syndromes are unique after the four-word exclusion.
    Branching on an unsatisfied check is exhaustive: any completion
    must select a variable incident on that check.
    """
    if _z23_short(cols) != 5:
        return 4
    bits = [1 << j for j in range(len(cols))]
    tails = {0: 0}
    incidence = [[] for _ in range(max(cols).bit_length())]
    degree = max(c.bit_count() for c in cols)
    for j, syndrome in enumerate(cols):
        tails[syndrome] = bits[j]
        value = syndrome
        while value:
            bit = value & -value
            incidence[bit.bit_length()-1].append(j)
            value ^= bit
    for j, syndrome in enumerate(cols):
        for h in range(j):
            value = syndrome ^ cols[h]
            assert value not in tails
            tails[value] = bits[j] | bits[h]

    def complete(target, budget, forbidden):
        def visit(value, remaining, word):
            if not value:
                return word
            if value.bit_count() > degree*remaining:
                return None
            if remaining <= 2:
                tail = tails.get(value)
                if (tail is not None and tail.bit_count() <= remaining
                        and not tail & (word | forbidden)):
                    return word | tail
                return None
            check = (value & -value).bit_length()-1
            choices = sorted(
                incidence[check],
                key=lambda j: -(cols[j] & value).bit_count())
            for j in choices:
                if bits[j] & (word | forbidden):
                    continue
                hit = visit(value ^ cols[j], remaining-1,
                            word | bits[j])
                if hit is not None:
                    return hit
            return None
        return visit(target, budget, 0)

    # Separate budgets ensure a six-word hit never hides a five-word hit.
    for budget in (4, 5):
        for anchor in range(0, len(cols), q):
            tail = complete(cols[anchor], budget, bits[anchor])
            if tail is not None:
                word = tail | bits[anchor]
                syndrome, remaining = 0, word
                while remaining:
                    bit = remaining & -remaining
                    syndrome ^= cols[bit.bit_length()-1]
                    remaining ^= bit
                assert syndrome == 0
                assert word.bit_count() <= budget+1
                return word.bit_count()
    return 7  # Excluded through six; not a quantum-distance reading.


def _t24_six_count(cols):
    """Exact six-support census, only after exclusion through five."""
    seen, hits = {}, 0
    for i, x in enumerate(cols):
        for j in range(i):
            pair = x ^ cols[j]
            for h in range(j):
                syndrome = pair ^ cols[h]
                old = seen.get(syndrome, 0)
                hits += old
                seen[syndrome] = old+1
    assert hits % 10 == 0
    return hits//10


def _t24_fingerprint(hx, hz, n, k):
    """Combined uncoloured incidence invariant, permitting X/Z mixing."""
    rows = list(hx)+list(hz)
    weights = [r.bit_count() for r in rows]
    profiles = [{} for _ in rows]
    overlaps = {}
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            overlaps[overlap] = overlaps.get(overlap, 0)+1
            if overlap:
                a, b = (weights[j], overlap), (weights[i], overlap)
                profiles[i][a] = profiles[i].get(a, 0)+1
                profiles[j][b] = profiles[j].get(b, 0)+1
    cols = _z23_columns(rows, n)
    return (
        n, k, tuple(sorted(weights)),
        tuple(sorted(c.bit_count() for c in cols)),
        tuple(sorted(overlaps.items())),
        tuple(sorted((weights[i], tuple(sorted(p.items())))
                     for i,p in enumerate(profiles))))


def _t24_logical_probe(checks, stabilizers, n, rng, trials):
    """Actual full-code logical witnesses, with stabilizer dressing."""
    cols = _z23_columns(checks, n)
    basis = _q85_basis(stabilizers)
    order = list(range(n))
    best = n+1

    def consider(word):
        nonlocal best
        if (word and word.bit_count() < best
                and _q85_outside(word, basis)):
            best = word.bit_count()

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
                consider(word)
        small = sorted(words, key=int.bit_count)[:28]
        for i, word in enumerate(small):
            for other in small[:i]:
                consider(word ^ other)
        for seed in small[:6]:
            if not _q85_outside(seed, basis):
                continue
            word = seed
            for _ in range(3):
                changed = False
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < word.bit_count():
                        word = trial
                        changed = True
                if not changed:
                    break
            consider(word)
        if best <= 5:
            break
    return best


@lru_cache(maxsize=1)
def _t24_bank(G):
    """Pure deterministic bank; all action blocks come from verified lifts."""
    from itertools import combinations

    assert G.q == 24
    q = G.q
    zero_a = [[[] for _ in range(3)]]
    zero_b = [[[] for _ in range(7)] for _ in range(2)]
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            if side == "A":
                A, B = [[[g], [], []]], zero_b
            else:
                A = zero_a
                B = [[[g]]+[[] for _ in range(6)],
                     [[] for _ in range(7)]]
            hx, hz, _ = lifted_matrices(G, A, B)
            ac, bc = _z23_components(hx, hz, q)
            block = tuple((ac if side == "A" else bc)[:q])
            assert all(c.bit_count() == 1 for c in block)
            blocks.append(block)
        atlas[side] = tuple(blocks)

    def compile_component(side, grid):
        width = (3 if side == "A" else 7)*q
        cols = [0]*width
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = atlas[side][G.idx(g)]
                    for h, syndrome in enumerate(block):
                        cols[j*q+h] ^= syndrome << (i*q)
        return tuple(cols)

    def support_key(cell):
        # Right translation of an A cell is a column gauge.
        return min(
            tuple(sorted(G.idx(G.mul(g, G.inv(anchor))) for g in cell))
            for anchor in cell)

    local = random.Random(BASE_SEED*100003+24045131)
    records, seen_supports = [], set()
    others = [g for g in G.elements if g != G.e]
    for g,h in combinations(others, 2):
        label = support_key((G.e, g, h))
        if label in seen_supports:
            continue
        seen_supports.add(label)
        cell = tuple(G.elements[t] for t in label)
        cols = compile_component("A", [[cell, [], []]])[:q]
        if _z23_short(cols) != 5:
            continue
        records.append(dict(cell=cell, cols=cols,
                            rank=gf2_rank(cols), label=label))

    targets = {}
    for i,a in enumerate(records):
        signature = _t24_span(a["cols"])
        if 20 <= len(signature) <= 24:
            targets[signature] = dict(signature)
        for b in records[:i]:
            signature = _t24_span(a["cols"]+b["cols"])
            if 20 <= len(signature) <= 24:
                targets[signature] = dict(signature)

    ordered_targets = sorted(targets, key=lambda s: (len(s), s))
    # Keep the full-image fallback even if numerous singular images exist.
    selected_targets = ordered_targets[:64]
    for signature in ordered_targets:
        if len(signature) == q and signature not in selected_targets:
            selected_targets.append(signature)

    a_library, seen_a = [], set()
    for signature in selected_targets:
        span = targets[signature]
        members = [
            r for r in records
            if all(not _q85_outside(c, span) for c in r["cols"])]
        if len(members) < 3:
            continue
        retained, tried = [], set()
        for attempt in range(384):
            ids = tuple(sorted(local.sample(range(len(members)), 3)))
            if ids in tried:
                continue
            tried.add(ids)
            chosen = [members[t] for t in ids]
            labels = tuple(sorted(r["label"] for r in chosen))
            if labels in seen_a:
                continue
            cols = tuple(c for r in chosen for c in r["cols"])
            rank = gf2_rank(cols)
            if rank != len(signature):
                continue
            floor = _t24_component_floor(cols, q)
            if floor <= 5:
                continue
            seen_a.add(labels)
            item = dict(
                grid=[[r["cell"] for r in chosen]], cols=cols,
                rank=rank, floor=floor, cycles=_z23_cycles(cols),
                labels=labels)
            retained.append(item)
            retained.sort(key=lambda x: (
                x["floor"], -x["cycles"]), reverse=True)
            del retained[4:]
            if len(retained) == 4 and attempt >= 96:
                break
        a_library.extend(retained)

    if not a_library:
        raise ValueError("SL2(3) supplied no component-five-clean A")

    def a_priority(item):
        cap = 8 if item["floor"] == 7 else 6
        # Only a shortlist heuristic; assembled k is always verified.
        proposal_k = 240+5*(q-item["rank"])
        return (proposal_k*cap*cap, cap,
                -item["rank"], -item["cycles"])

    a_library.sort(key=a_priority, reverse=True)
    a_library = a_library[:40]
    for item in a_library:
        item["six"] = (_t24_six_count(item["cols"])
                       if item["floor"] == 6 else 0)
        item["cap"] = 8 if item["floor"] == 7 else 6

    z = (2,0,0,2)

    def fresh_cell(rng, count):
        if count <= 1:
            return rng.sample(G.elements, count)
        for _ in range(128):
            cell = rng.sample(G.elements, count)
            if any(G.mul(G.inv(g), h) == z
                   for i,g in enumerate(cell) for h in cell[:i]):
                continue
            if count == 3:
                u = G.mul(G.inv(cell[0]), cell[1])
                v = G.mul(G.inv(cell[0]), cell[2])
                if G.mul(u, v) == G.mul(v, u):
                    continue
            return cell
        raise ValueError("noncommuting cell constraints exhausted")

    def draw_b(rng, sb):
        B = [[None]*7 for _ in range(2)]
        mono = [j for j in range(7) if sb[0][j] == sb[1][j] == 1]
        relatives, cost = None, None
        for _ in range(48):
            trial = rng.sample(G.elements, len(mono))
            ratios = [G.mul(g, G.inv(h))
                      for g in trial for h in trial if g != h]
            penalty = len(ratios)-len(set(ratios))
            if cost is None or penalty < cost:
                relatives, cost = trial, penalty
            if penalty == 0:
                break
        for j, relative in zip(mono, relatives):
            top = G.pick(rng)
            B[0][j] = [top]
            B[1][j] = [G.mul(G.inv(relative), top)]
        for i in range(2):
            for j in range(7):
                if B[i][j] is None:
                    B[i][j] = fresh_cell(rng, sb[i][j])
        return B

    bank, fingerprints = [], set()
    sa = [[3,3,3]]
    for slot in range(N_SLOTS):
        policy, geometry = divmod(slot % 20, 4)
        preferred_rank = 20+policy
        columns = _T24_B_CARDS[geometry]
        sb = [[column[i] for column in columns] for i in range(2)]
        assert _grid_weights(sa, sb) == 12
        assert _no_dead_lines(sa, sb)
        assert [sum(row) for row in sb] == [9,9]
        rng = random.Random(BASE_SEED*100007+24045217+100019*slot)

        eligible = sorted(
            a_library,
            key=lambda a: (
                abs(a["rank"]-preferred_rank),
                -a_priority(a)[0], a["six"], a["cycles"]))
        # Later batches shift the intrinsic six-word/image stratum;
        # they do not transport a previously emitted labeling.
        shift = (slot//20) % min(4, len(eligible))
        choices = eligible[shift:shift+6]
        if not choices:
            choices = eligible[:6]
        chosen = None

        for batch in range(3):
            bp, seen_b = [], set()
            admitted = 0
            for attempt in range(192+96*batch):
                B = draw_b(rng, sb)
                labels = tuple(tuple(sorted(cell))
                               for row in B for cell in row)
                if labels in seen_b:
                    continue
                seen_b.add(labels)
                bc = compile_component("B", B)
                if gf2_rank(bc) != 2*q:
                    continue
                floor = _t24_component_floor(bc, q)
                if floor <= 5:
                    continue
                item = dict(
                    grid=B, cols=bc, floor=floor,
                    cap=7 if floor == 7 else 6,
                    cycles=_z23_cycles(bc), labels=labels)
                bp.append(item)
                bp.sort(key=lambda b: (
                    b["floor"], -b["cycles"]), reverse=True)
                del bp[4:]
                admitted += 1
                if admitted >= 10 and attempt >= 64:
                    break
            if not bp:
                continue

            for b in bp[:3]:
                b["six"] = (_t24_six_count(b["cols"])
                            if b["floor"] == 6 else 0)
            bp = bp[:3]
            ap = choices if batch == 0 else eligible[:10]
            proposals = []
            for a in ap:
                for b in bp:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (425*n <= 1000*k < 475*n):
                        continue
                    ac, bc = _z23_components(hx, hz, q)
                    assert tuple(ac) == a["cols"]
                    assert tuple(bc) == b["cols"]
                    cap = min(a["cap"], b["cap"])
                    six = a["six"]+b["six"]
                    key = (k*cap*cap, cap, -six,
                           -a["cycles"]-b["cycles"])
                    proposals.append(
                        (key, A, B, hx, hz, n, k, cap, six))
            proposals.sort(key=lambda x: x[0], reverse=True)

            winner, screened = None, 0
            for _,A,B,hx,hz,n,k,cap,six in proposals:
                fingerprint = _t24_fingerprint(hx, hz, n, k)
                if fingerprint in fingerprints:
                    continue
                low, cycles = _q85_short_screen(hx, hz, n)
                if low != 5:
                    continue
                dx = _t24_logical_probe(hz, hx, n, rng, 6)
                dz = _t24_logical_probe(hx, hz, n, rng, 6)
                reading = min(cap, dx, dz)
                objective = (k*reading*reading, reading,
                             k, -six, -cycles)
                if winner is None or objective > winner[0]:
                    winner = (
                        objective, A, B, hx, hz, k, fingerprint)
                screened += 1
                if screened >= 3:
                    break

            if winner is None:
                continue
            objective,A,B,hx,hz,k,fingerprint = winner
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx+hz) <= 12
            assert all(len(cell) == len(set(cell)) <= 3
                       for grid in (A,B) for row in grid for cell in row)
            fingerprints.add(fingerprint)
            A = tuple(tuple(tuple(c) for c in row) for row in A)
            B = tuple(tuple(tuple(c) for c in row) for row in B)
            chosen = (A, B, k, objective)
            break
        # Failed policies never substitute archived or transported codes.
        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return this card's fresh, verified rank/image-selected filling."""
    item = _t24_bank(G)[idx]
    if item is None:
        raise ValueError("SL2(3) card found no fresh target-rate filling")
    A, B, k, screen = item
    return {"idx": idx, "cells": {"A": A, "B": B},
            "k": k, "screen": screen}


def CELL(rng, G, side, i, j, ctx):
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's uniform rolled D14 fillings did not suppress sector-2 weight-4 logicals: all five measured codes had d30s=4.
# REFUTED: Q8 x Z5 order-20 quaternion-axis binomials plus exact-four candidate selection did not break the weight-four plateau: all five measured codes had d30s=4, with minima entirely in sector 1.
# REFUTED: Repeating the Z23 support/AP recipes with fresh sampling guarantees distinct codes: slots 20,22,23,26,27 and many later slots were marked isomorphic duplicates.
# REFUTED: Zero/trinomial B redistribution plus exact component-six counting improves the Z23 five-code score: all five current measurements remained k=230, d30s=6, fitness 15.652, with minima entirely in sector 1.
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

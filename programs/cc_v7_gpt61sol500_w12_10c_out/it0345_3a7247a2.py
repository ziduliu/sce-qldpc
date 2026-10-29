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
CONCEPT = ("On C9 semidirect C6 with inversion action, order 54, shapes (1,3)x(1,3), compatible singular cubic components should produce five fresh cell-0.450 codes with measured distance readings at least seven, breaking the parent's distance-six plateau.")
GROUP = ("L3 jump from abelian C21 to nonabelian C9:C6, y*x*y^-1=x^-1. Its centre has order three. The equal-cubic order-54 bed in FRONTIER measured five [540,228,8] codes and fitness 27.0 in cell 0.400. Those readings motivate transferring the cubic geometry to larger compatible rank deficits; they do not establish distance retention in cell 0.450.")
SHAPES = ("Change to equal A=(1,3), B=(1,3): n=540, kb=216 and guaranteed rate 0.400. Both constituents have 162 variables and 54 checks; the size window admits q<=59. Complete verified ranks must give 0.425<=k/n<0.475. Preferred component ranks are 50 and 52, but their coupling is never assigned an assumed k. Parent weight-six minima were entirely sector one, especially Acol0/Acol1 with Bcol4..6; replace the monomial-rich seven-column B geometry.")
FILLING = ("Every final cell is a distinct-element trinomial: row totals nine and column totals three give HX=HZ=12. Enumerate normalized supports and compile both actions only from verified unit lifts. Pack connected constituents with compatible rank-50/rank-52 images, exactly exclude component kernels through four, use cubic parity to exclude five, and count six exactly. Twenty policies cross five intrinsic six-load strata with four relative-order/commutation preferences. Verify complete rates, exclude full logicals through five modulo stabilizers, independently deepen actual logical witnesses, and conservatively reject equal uncoloured overlap fingerprints. Component words are not automatically quantum logicals; search readings remain upper bounds.")
FAMILY_NAME = "r045_C9sdC6_inv_13x13_singular_cubic_logical5_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 54

def custom_ops(q):
    # x^9=y^6=1; y*x*y^-1=x^-1.
    # Inversion has order two, which divides six.
    if not valid_q(q):
        raise ValueError("C9 semidirect C6 requires order 54")
    els = [(a, b) for b in range(6) for a in range(9)]

    def red(g):
        return (g[0] % 9, g[1] % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b % 2 == 0 else -c)) % 9,
                (b + d) % 6)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 6
        return ((-a if bi % 2 == 0 else a) % 9, bi)

    def idx(g):
        a, b = red(g)
        return 9*b + a

    return dict(elements=els, ident=(0, 0), mul=mul,
                inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


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
    """Extract every component check row from the VERIFIED full lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for r in range(q):
            row = hx[(i * cb) * q + r]
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


from functools import lru_cache


_Z13_B_CARDS = (
    ((2, 1), (1, 2), (1, 2)),
    ((3, 0), (1, 2), (1, 2)),
    ((3, 0), (0, 3), (2, 1)),
    ((3, 0), (0, 3), (1, 2)),
    ((2, 1), (2, 1), (0, 3)),
)


def _z13_six_count(cols):
    """Exact weight-six kernel census AFTER exclusion through five.

    Then colliding triples are disjoint, and each six-support has
    ten unordered partitions into triples.
    """
    seen = {}
    hits = 0
    for i, x in enumerate(cols):
        for j in range(i):
            pair = x ^ cols[j]
            for t in range(j):
                syndrome = pair ^ cols[t]
                old = seen.get(syndrome, 0)
                hits += old
                seen[syndrome] = old + 1
    assert hits % 10 == 0
    return hits // 10


def _z13_incidence_signature(hx, hz, n, k):
    """Invariant even if Tanner isomorphism intermixes X and Z checks.

    Different signatures certify nonisomorphism. Equal signatures
    are conservatively discarded; they do not prove isomorphism.
    """
    rows = hx + hz
    weights = [row.bit_count() for row in rows]
    local = [{} for _ in rows]
    overlap_hist = {}
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            overlap_hist[overlap] = overlap_hist.get(overlap, 0) + 1
            if overlap:
                ki, kj = (weights[j], overlap), (weights[i], overlap)
                local[i][ki] = local[i].get(ki, 0) + 1
                local[j][kj] = local[j].get(kj, 0) + 1
    profiles = tuple(sorted(
        (weights[i], tuple(sorted(profile.items())))
        for i, profile in enumerate(local)))
    cols = _z23_columns(rows, n)
    # Do not use the X/Z-colouring-dependent rank coordinate to certify
    # nonisomorphism of an otherwise uncoloured incidence graph.
    return (n, tuple(sorted(weights)),
            tuple(sorted(col.bit_count() for col in cols)),
            tuple(sorted(overlap_hist.items())), profiles)


@lru_cache(maxsize=1)
def _z13_rank_bonus_bank(G):
    """Pure deterministic bank: twenty policies, never orbit fallbacks."""
    assert G.q == 13
    q = G.q
    sa = [[3, 3, 1, 1, 1, 0],
          [0, 0, 2, 2, 2, 2]]
    stub_a = [[[G.e] for _ in range(6)] for _ in range(2)]
    stub_b = [[[G.e] for _ in range(7)] for _ in range(2)]
    bank, fingerprints = [], set()

    for slot in range(N_SLOTS):
        card = slot % 20
        geometry, mode = divmod(card, 4)
        columns = ((1, 1),) * 4 + _Z13_B_CARDS[geometry]
        sb = [[column[i] for column in columns] for i in range(2)]
        assert _grid_weights(sa, sb) == 12
        assert _no_dead_lines(sa, sb)
        rng = random.Random(BASE_SEED * 10007 + 13046001 + slot)

        def triple(ap):
            if ap:
                d = rng.randrange(1, q)
                offsets = (0, d, 2*d % q)
            else:
                while True:
                    u, v = rng.sample(range(1, q), 2)
                    if (v != 2*u % q and u != 2*v % q
                            and (u + v) % q != 0):
                        offsets = (0, u, v)
                        break
            base = rng.randrange(q)
            return [((base + a) % q,) for a in offsets]

        def difference_set(cell):
            return {(g[0] - h[0]) % q
                    for g in cell for h in cell if g != h}

        def fill(side):
            counts = sa if side == "A" else sb
            grid = [[[] for _ in row] for row in counts]
            if side == "B":
                # PG(2,3) Singer set: all twelve ordered differences
                # between these four relatives are distinct modulo 13.
                # Its translates alone are not counted as new designs.
                relatives = (0, 1, 3, 9)
                assert len({(a-b) % q for a in relatives
                            for b in relatives if a != b}) == 12
                for j, relative in enumerate(relatives):
                    base = rng.randrange(q)
                    grid[0][j] = [(base,)]
                    grid[1][j] = [((base + relative) % q,)]

            for i, row in enumerate(counts):
                for j, count in enumerate(row):
                    if not count or grid[i][j]:
                        continue
                    if count == 1:
                        grid[i][j] = [G.pick(rng)]
                    elif count == 2:
                        base = rng.randrange(q)
                        d = rng.randrange(1, q)
                        grid[i][j] = [(base,), ((base + d) % q,)]
                    else:
                        # Modes 0,1,2 prescribe zero, one, or two AP
                        # pure A triples. Mode 3 requires disjoint
                        # internal difference sets for those triples.
                        ap = (side == "A" and i == 0
                              and j < (0, 1, 2, 0)[mode])
                        if side == "B":
                            ap = (mode + i + j) % 3 == 0
                        grid[i][j] = triple(ap)

            if side == "A" and mode == 3:
                first = difference_set(grid[0][0])
                for _ in range(64):
                    second = triple(False)
                    if not first.intersection(difference_set(second)):
                        grid[0][1] = second
                        break
                else:
                    return None
            return grid

        def component_pool(side, quota):
            """Rank and syndrome tests use only scaffold-built actions."""
            pool, seen = [], set()
            wanted_rank = 25 if side == "A" else 26
            for attempt in range(quota):
                grid = fill(side)
                if grid is None:
                    continue
                signature = tuple(tuple(sorted(cell))
                                  for row in grid for cell in row)
                if signature in seen:
                    continue
                seen.add(signature)
                A, B = ((grid, stub_b) if side == "A"
                        else (stub_a, grid))
                hx, hz, _ = lifted_matrices(G, A, B)
                ac, bc = _z23_components(hx, hz, q)
                cols = ac if side == "A" else bc
                rank = gf2_rank(cols)  # column rank equals row rank
                if rank != wanted_rank:
                    continue
                if _z23_short(cols) != 5:
                    continue
                if _z23_short(cols, through_five=True) != 6:
                    continue
                cycles = _z23_cycles(cols)
                pool.append((cycles, attempt, grid, cols))
                pool.sort(key=lambda item: item[:2])
                del pool[6:]
                if len(pool) == 6 and attempt >= 64:
                    break

            scored = []
            for cycles, attempt, grid, cols in pool:
                six = _z13_six_count(cols)
                scored.append((six, cycles, attempt, grid))
            scored.sort(key=lambda item: item[:3])
            return scored[:4]

        chosen = None
        for retry in range(2):
            quota = 192 + 128 * retry
            ap = component_pool("A", quota)
            bp = component_pool("B", quota)
            pairs = sorted(
                ((a[0] + b[0], a[1] + b[1], ai, bi, a[3], b[3])
                 for ai, a in enumerate(ap)
                 for bi, b in enumerate(bp)),
                key=lambda item: item[:4])
            for six, component_cycles, _, _, A, B in pairs:
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != 598 or k != 265:
                    continue
                low, cycles = _q85_short_screen(hx, hz, n)
                if low != 5:
                    continue
                fingerprint = _z13_incidence_signature(hx, hz, n, k)
                if fingerprint in fingerprints:
                    continue
                assert css_ok(hx, hz)
                assert max(row.bit_count() for row in hx + hz) <= 12
                assert all(
                    len(cell) == len({G._key(g) for g in cell})
                    for grid in (A, B) for row in grid for cell in row)
                fingerprints.add(fingerprint)
                chosen = (A, B, k, (six, component_cycles, cycles))
                break
            if chosen is not None:
                break

        # Preserve indexing and never replace a failed policy with
        # an archived code or a relabeling of an earlier bank entry.
        bank.append(chosen)
    return tuple(bank)


# These new cards supersede the unused cyclic rank-bonus generator.
# Triple kinds are forward (2,1,0), backward (2,0,1), and (1,1,1)
# quotient occupancies, normalized before an independent cell gauge.
_F21_A_CARDS = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 2),
    (0, 2, 2),
    (2, 2, 2),
)
_F21_B_CARDS = (
    ((2, 1), (2, 1), (1, 2), (1, 2),
     (1, 1), (1, 1), (1, 1)),
    ((3, 0), (0, 3), (2, 1), (1, 2),
     (2, 0), (0, 2), (1, 1)),
    ((2, 1), (2, 1), (1, 2), (1, 2),
     (2, 0), (0, 2), (1, 1)),
    ((3, 0), (3, 0), (0, 3), (0, 3),
     (2, 0), (0, 2), (1, 1)),
)


def _f21_triple(rng, G, kind):
    """Distinct terms with genuinely noncommuting relative labels."""
    if kind < 2:
        cell = [G.e, (rng.randrange(1, 7), 0),
                (rng.randrange(7), kind + 1)]
    else:
        a = rng.randrange(7)
        # (a,1)^2=(3*a,2); exclude a single C3 subgroup norm.
        c = rng.choice([h for h in range(7) if h != 3*a % 7])
        cell = [G.e, (a, 1), (c, 2)]
    assert G.mul(cell[1], cell[2]) != G.mul(cell[2], cell[1])
    base = G.pick(rng)
    return [G.mul(base, g) for g in cell]


def _f21_draw_b(rng, G, sb, phase):
    """Cyclic replacement used by the active common-image optimizer.

    Historical F21 generators remain inactive. All emitted labels here
    use the new one-coordinate encoding and the verified lift screens.
    """
    from math import gcd

    assert G.q == 21
    q, cb = G.q, len(sb[0])
    B = [[None] * cb for _ in range(2)]
    mono = [j for j in range(cb) if sb[0][j] == sb[1][j] == 1]

    # Independent voltage optimization, not transported group-orbit copies.
    best, best_cost = None, None
    for _ in range(64):
        relatives = rng.sample(G.elements, len(mono))
        ratios = [G.mul(g, G.inv(h))
                  for g in relatives for h in relatives if g != h]
        cost = len(ratios) - len(set(ratios))
        if best_cost is None or cost < best_cost:
            best, best_cost = relatives, cost
        if cost == 0:
            break
    for j, relative in zip(mono, best):
        top = G.pick(rng)
        B[0][j] = [top]
        B[1][j] = [G.mul(G.inv(relative), top)]

    # These are intrinsic quotient occupancies, not common translations.
    models = ((0, 0, 1), (0, 0, 2), (0, 1, 2))
    long_steps = [d for d in range(1, q) if gcd(d, q) == 1]
    seven_steps = [d for d in range(1, q) if gcd(d, q) == 3]

    for i in range(2):
        for j in range(cb):
            if B[i][j] is not None:
                continue
            count = sb[i][j]
            if count == 0:
                cell = []
            elif count == 1:
                cell = [G.pick(rng)]
            elif count == 2:
                # No order-three binomial cycle is introduced.
                steps = (seven_steps if (phase + i + j) % 3 == 1
                         else long_steps)
                base = G.pick(rng)
                delta = (rng.choice(steps),)
                cell = [base, G.mul(base, delta)]
            else:
                residues = models[(phase + i + j) % len(models)]
                cell = []
                for residue in residues:
                    available = [g for g in G.elements
                                 if g[0] % 3 == residue and g not in cell]
                    cell.append(rng.choice(available))
                gauge = G.pick(rng)
                cell = [G.mul(gauge, g) for g in cell]
            assert len(cell) == count and len(cell) == len(set(cell))
            B[i][j] = cell
    return B


def _f21_six_hit(cols):
    """Exact weight-six existence AFTER exclusion through weight five.

    Distinct colliding triples are then disjoint. False excludes six;
    it is not a distance estimate or a witness at weight seven.
    """
    seen = set()
    for i, x in enumerate(cols):
        for j in range(i):
            pair = x ^ cols[j]
            for h in range(j):
                syndrome = pair ^ cols[h]
                if syndrome in seen:
                    return True
                seen.add(syndrome)
    return False


def _f21_logical_probe(checks, stabilizers, n, rng, trials):
    """Actual full-code logical witnesses, including both LP sectors."""
    cols = _z23_columns(checks, n)
    basis = _q85_basis(stabilizers)
    order = list(range(n))
    best = n + 1

    def consider(word):
        nonlocal best
        weight = word.bit_count()
        if word and weight < best and _q85_outside(word, basis):
            best = weight

    for _ in range(trials):
        rng.shuffle(order)
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
                words.append(word)
                consider(word)

        small = sorted(words, key=int.bit_count)[:32]
        for i, word in enumerate(small):
            for other in small[:i]:
                consider(word ^ other)

        # Stabilizer additions preserve a nontrivial logical class.
        for seed in small[:6]:
            if not _q85_outside(seed, basis):
                continue
            word = seed
            for _ in range(3):
                changed = False
                for row in stabilizers:
                    combined = word ^ row
                    if combined.bit_count() < word.bit_count():
                        word = combined
                        changed = True
                if not changed:
                    break
            consider(word)
        if best <= 6:
            break
    return best


@lru_cache(maxsize=1)
def _f21_design_bank(G):
    """Twenty independently optimized structural cards, using verified lifts."""
    assert G.q == 21
    q = G.q
    sa = [[3, 3, 3]]
    stub_a = [[[G.e] for _ in range(3)]]
    stub_b = [[[G.e] for _ in range(7)] for _ in range(2)]
    bank, fingerprints = [], set()

    for slot in range(N_SLOTS):
        model, geometry = divmod(slot % 20, 4)
        kinds = _F21_A_CARDS[model]
        columns = _F21_B_CARDS[geometry]
        sb = [[column[i] for column in columns] for i in range(2)]
        assert _grid_weights(sa, sb) == 12
        assert _no_dead_lines(sa, sb)
        rng = random.Random(BASE_SEED * 10007 + 21045031 + 100003*slot)

        def component_pool(side, quota):
            pool, seen = [], set()
            for attempt in range(quota):
                grid = ([[_f21_triple(rng, G, kind) for kind in kinds]]
                        if side == "A" else
                        _f21_draw_b(rng, G, sb, model))
                signature = tuple(tuple(sorted(cell))
                                  for row in grid for cell in row)
                if signature in seen:
                    continue
                seen.add(signature)
                A, B = ((grid, stub_b) if side == "A"
                        else (stub_a, grid))
                hx, hz, _ = lifted_matrices(G, A, B)
                ac, bc = _z23_components(hx, hz, q)
                cols = ac if side == "A" else bc
                if _z23_short(cols) != 5:
                    continue
                if _z23_short(cols, through_five=True) != 6:
                    continue
                hit6 = _f21_six_hit(cols)
                cycles = _z23_cycles(cols)
                rank = gf2_rank(cols)
                item = {
                    "grid": grid, "cols": cols, "hit6": hit6,
                    "cycles": cycles, "rank": rank,
                    "key": (not hit6, -cycles, rank),
                }
                pool.append(item)
                pool.sort(key=lambda x: x["key"], reverse=True)
                del pool[6:]
                if (attempt >= 64 and len(pool) == 6
                        and sum(not x["hit6"] for x in pool) >= 4):
                    break

            for item in pool:
                cols = item["cols"]
                reading = _z23_reading(cols, rng, 20)
                six = _z13_six_count(cols) if item["hit6"] else 0
                # Isolated order-seven binomials supply explicit
                # seven-term component words. The three-matching-
                # column control has a component word through six:
                # either a C3 ratio cycle or commuting C7 relatives.
                cap = 10 if side == "A" else (7 if geometry else 6)
                if item["hit6"]:
                    cap = min(cap, 6)
                item["reading"] = min(reading, cap)
                item["six"] = six
                item["key"] = (
                    not item["hit6"], item["reading"], -six,
                    -item["cycles"], item["rank"])
            pool.sort(key=lambda x: x["key"], reverse=True)
            return pool[:4]

        chosen = None
        for retry in range(2):
            quota = 128 + 128*retry
            ap = component_pool("A", quota)
            bp = component_pool("B", quota)
            candidates = []
            for a in ap:
                for b in bp:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if not (425*n <= 1000*k < 475*n):
                        continue
                    d = min(a["reading"], b["reading"], 10)
                    six = a["six"] + b["six"]
                    cycles = a["cycles"] + b["cycles"]
                    score = (k*d*d, d, -six, -cycles)
                    candidates.append(
                        (score, A, B, hx, hz, n, k, d, six, cycles))
            candidates.sort(key=lambda x: x[0], reverse=True)

            winner = None
            screened = 0
            for candidate in candidates:
                _, A, B, hx, hz, n, k, d, six, component_cycles = candidate
                low, cycles = _q85_short_screen(hx, hz, n)
                if low != 5:
                    continue
                fingerprint = _z13_incidence_signature(hx, hz, n, k)
                if fingerprint in fingerprints:
                    continue
                if d >= 7:
                    dx = _f21_logical_probe(hz, hx, n, rng, 8)
                    dz = _f21_logical_probe(hx, hz, n, rng, 8)
                    d = min(d, dx, dz)
                objective = (k*d*d, d, -six, -cycles, -component_cycles)
                if winner is None or objective > winner[0]:
                    winner = (objective, A, B, hx, hz, k, fingerprint)
                screened += 1
                if d >= 7 or screened >= 4:
                    break

            if winner is not None:
                objective, A, B, hx, hz, k, fingerprint = winner
                assert css_ok(hx, hz)
                assert max(row.bit_count() for row in hx + hz) <= 12
                assert all(
                    len(cell) <= 3 and len(cell) == len(set(cell))
                    for grid in (A, B) for row in grid for cell in row)
                fingerprints.add(fingerprint)
                chosen = (A, B, k, objective)
                break

        # Failed policies never substitute archived or transported codes.
        bank.append(chosen)
    return tuple(bank)


@lru_cache(maxsize=4)
def NEW_CTX(rng, G, idx):
    """Pure bank of common-image designs; idx=-1 shares verified compilation."""
    if idx >= 0:
        item = NEW_CTX(None, G, -1)[idx]
        if item is None:
            raise ValueError("common-image card found no fresh screened filling")
        A, B, k, screen = item
        return {"idx": idx,
                "cells": {
                    "A": [[list(c) for c in row] for row in A],
                    "B": [[list(c) for c in row] for row in B]},
                "k": k, "screen": screen}

    from itertools import combinations

    assert G.q == 21
    q = G.q
    local = random.Random(BASE_SEED * 100003 + 21045187)
    zero_a = [[[] for _ in range(3)]]
    zero_b = [[[] for _ in range(7)] for _ in range(2)]

    # F2-linear superposition of VERIFIED single-term blocks.
    # This is not an independently implemented regular representation.
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            if side == "A":
                A = [[[g], [], []]]
                B = zero_b
            else:
                A = zero_a
                B = [[[g]] + [[] for _ in range(6)],
                     [[] for _ in range(7)]]
            hx, hz, _ = lifted_matrices(G, A, B)
            ac, bc = _z23_components(hx, hz, q)
            cols = ac if side == "A" else bc
            block = tuple(cols[:q])
            assert all(c.bit_count() == 1 for c in block)
            blocks.append(block)
        atlas[side] = tuple(blocks)

    def compile_component(side, grid):
        width = (3 if side == "A" else 7) * q
        cols = [0] * width
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = atlas[side][G.idx(g)]
                    for h, syndrome in enumerate(block):
                        cols[j*q+h] ^= syndrome << (i*q)
        return cols

    def canonical_span(cols):
        """Unique reduced basis of an image in the actual check coordinates."""
        basis = _q85_basis(cols)
        pivots = sorted(basis)
        for a, p in enumerate(pivots):
            row = basis[p]
            for higher in pivots[a+1:]:
                if basis[higher] & (1 << p):
                    basis[higher] ^= row
        return tuple((p, basis[p]) for p in pivots)

    def support_key(cell):
        # A cell's right translation is an independent column gauge.
        # Quotient it out rather than count its copies as new supports.
        return min(
            tuple(sorted(G.idx(G.mul(g, G.inv(anchor))) for g in cell))
            for anchor in cell)

    # Enumerate all trinomials up to those gauges. No chiral template
    # restriction is imposed: image rank and actual short words decide.
    records, seen_supports = [], set()
    others = [g for g in G.elements if G._key(g) != G._key(G.e)]
    for g, h in combinations(others, 2):
        key = support_key([G.e, g, h])
        if key in seen_supports:
            continue
        seen_supports.add(key)
        cell = tuple(G.elements[t] for t in key)
        cols = compile_component("A", [[cell, [], []]])[:q]
        rank = gf2_rank(cols)
        if rank > 19:
            continue
        # A light word entirely inside one block cannot be repaired by
        # its other two A columns.
        if _z23_short(cols) != 5:
            continue
        records.append(dict(cell=cell, cols=tuple(cols), rank=rank,
                            label=key))

    # Rank-18 targets can be generated by one cell or by joining two
    # lower-rank cell images. Retain rank 19 as a legal backup stratum.
    targets = {}
    for i, a in enumerate(records):
        if a["rank"] in (18, 19):
            signature = canonical_span(a["cols"])
            targets[signature] = dict(signature)
        for b in records[:i]:
            signature = canonical_span(a["cols"] + b["cols"])
            if len(signature) in (18, 19):
                targets[signature] = dict(signature)

    a_pool, seen_a = [], set()
    for signature, span in sorted(
            targets.items(), key=lambda item: (len(item[0]), item[0])):
        members = [
            record for record in records
            if all(not _q85_outside(col, span) for col in record["cols"])]
        if len(members) < 3:
            continue
        triples = list(combinations(range(len(members)), 3))
        local.shuffle(triples)
        raw = []
        for attempt, ids in enumerate(triples[:1536]):
            chosen = [members[t] for t in ids]
            labels = tuple(sorted(record["label"] for record in chosen))
            if labels in seen_a:
                continue
            cols = tuple(c for record in chosen for c in record["cols"])
            rank = gf2_rank(cols)
            if rank != len(signature):
                continue
            seen_a.add(labels)
            if _z23_short(cols) != 5:
                continue
            if _z23_short(cols, through_five=True) != 6:
                continue
            cycles = _z23_cycles(cols)
            raw.append(dict(
                grid=[[record["cell"] for record in chosen]],
                cols=cols, rank=rank, cycles=cycles, labels=labels))
            raw.sort(key=lambda item: (item["cycles"], item["labels"]))
            del raw[12:]
            if len(raw) == 12 and attempt >= 192:
                break

        for item in raw:
            item["six"] = _z13_six_count(item["cols"])
        raw.sort(key=lambda item: (
            item["six"], item["cycles"], item["labels"]))
        a_pool.extend(raw[:8])

    if not a_pool:
        raise ValueError("no common-image A passed component-five exclusion")

    # Five policies select different intrinsic six-word-load regions,
    # not translates or automorphism images of one saved filling.
    preferred_rank = min(item["rank"] for item in a_pool)
    primary = sorted(
        (item for item in a_pool if item["rank"] == preferred_rank),
        key=lambda item: (item["six"], item["cycles"], item["labels"]))
    backup = sorted(
        (item for item in a_pool if item["rank"] != preferred_rank),
        key=lambda item: (item["rank"], item["six"],
                          item["cycles"], item["labels"]))

    b_cards = (
        ((2, 1), (2, 1), (1, 2), (1, 2),
         (1, 1), (1, 1), (1, 1)),
        ((3, 0), (1, 2), (1, 2), (1, 2),
         (1, 1), (1, 1), (1, 1)),
        ((3, 0), (0, 3), (2, 1), (1, 2),
         (1, 1), (1, 1), (1, 1)),
        ((3, 0), (3, 0), (0, 3), (0, 3),
         (1, 1), (1, 1), (1, 1)),
    )
    sa = [[3, 3, 3]]
    bank, fingerprints = [], set()

    for slot in range(N_SLOTS):
        model, geometry = divmod(slot % 20, 4)
        columns = b_cards[geometry]
        sb = [[column[i] for column in columns] for i in range(2)]
        assert _grid_weights(sa, sb) == 12
        assert _no_dead_lines(sa, sb)
        assert [sum(row) for row in sb] == [9, 9]
        local = random.Random(
            BASE_SEED * 100007 + 21045223 + 100019*slot)

        start = (model * len(primary)) // 5
        choices = [
            primary[(start+t) % len(primary)]
            for t in range(min(6, len(primary)))]
        # Later batches also admit the remaining primary designs.
        remaining = [a for a in primary if a not in choices]
        selected = None

        for batch in range(3):
            bp, labels_seen = [], set()
            quota = 192 + 96*batch
            for attempt in range(quota):
                B = _f21_draw_b(
                    local, G, sb, (model + slot//20 + batch) % 3)
                labels = tuple(tuple(sorted(cell))
                               for row in B for cell in row)
                if labels in labels_seen:
                    continue
                labels_seen.add(labels)
                bc = compile_component("B", B)
                if gf2_rank(bc) != 42:
                    continue
                if _z23_short(bc) != 5:
                    continue
                if _z23_short(bc, through_five=True) != 6:
                    continue
                item = dict(grid=B, cols=tuple(bc),
                            cycles=_z23_cycles(bc), labels=labels)
                bp.append(item)
                bp.sort(key=lambda x: (x["cycles"], x["labels"]))
                del bp[6:]
                if len(bp) == 6 and attempt >= 96:
                    break

            for b in bp:
                b["six"] = _z13_six_count(b["cols"])
            bp.sort(key=lambda b: (b["six"], b["cycles"], b["labels"]))
            if not bp:
                continue

            ap = list(choices)
            if batch >= 1:
                ap.extend(remaining[:8])
            if batch == 2:
                ap.extend(backup[:8])

            # Rank 18 is the constructive target, not an assumed k formula.
            # Every assembled code is measured with the verified helpers.
            pairs = sorted(
                ((a["rank"], a["six"]+b["six"],
                  a["cycles"]+b["cycles"], ai, bi, a, b)
                 for ai, a in enumerate(ap)
                 for bi, b in enumerate(bp[:4])),
                key=lambda entry: entry[:5])
            finalists = []
            for _, six, component_cycles, _, _, a, b in pairs:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (425*n <= 1000*k < 475*n):
                    continue
                if a["rank"] == 18 and k != 225:
                    continue
                ac, bc = _z23_components(hx, hz, q)
                assert tuple(ac) == a["cols"]
                assert tuple(bc) == b["cols"]
                fingerprint = _z13_incidence_signature(hx, hz, n, k)
                if fingerprint in fingerprints:
                    continue
                low, cycles = _q85_short_screen(hx, hz, n)
                if low != 5:
                    continue
                finalists.append(dict(
                    A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                    fingerprint=fingerprint, six=six,
                    cycles=cycles, component_cycles=component_cycles))
                if len(finalists) >= 4:
                    break

            winner = None
            for item in finalists:
                # Actual found logicals can lower the conservative merit.
                # A sampling miss above six is never rewarded.
                dx = _f21_logical_probe(
                    item["hz"], item["hx"], item["n"], local, 4)
                dz = _f21_logical_probe(
                    item["hx"], item["hz"], item["n"], local, 4)
                reading = min(6, dx, dz)
                if reading < 6:
                    continue
                objective = (
                    item["k"]*reading*reading, item["k"],
                    -item["six"], -item["cycles"],
                    -item["component_cycles"])
                if winner is None or objective > winner[0]:
                    winner = (objective, item)

            if winner is None:
                continue
            objective, item = winner
            A, B = item["A"], item["B"]
            assert css_ok(item["hx"], item["hz"])
            assert max(row.bit_count()
                       for row in item["hx"] + item["hz"]) <= 12
            assert all(len(cell) == len(set(cell)) <= 3
                       for grid in (A, B) for row in grid for cell in row)
            fingerprints.add(item["fingerprint"])
            A = tuple(tuple(tuple(cell) for cell in row) for row in A)
            B = tuple(tuple(tuple(cell) for cell in row) for row in B)
            selected = (A, B, item["k"], objective)
            break

        # No archived filling or transported design replaces a failed card.
        bank.append(selected)
    return tuple(bank)


def _c54_span(cols):
    """Canonical image basis in the verified check coordinates."""
    basis = _q85_basis(cols)
    pivots = sorted(basis)
    for a, p in enumerate(pivots):
        for h in pivots[a + 1:]:
            if basis[h] & (1 << p):
                basis[h] ^= basis[p]
    return tuple((p, basis[p]) for p in pivots)


def _c54_connected(G, cells):
    """Exclude constituents confined to a proper relative-label subgroup."""
    generators = [g for cell in cells for g in cell if g != G.e]
    reached, pending = {G.e}, [G.e]
    while pending:
        g = pending.pop()
        for h in generators:
            value = G.mul(g, h)
            if value not in reached:
                reached.add(value)
                pending.append(value)
    return len(reached) == G.q


def _c54_fingerprint(hx, hz, n):
    """Uncoloured Tanner invariant, including overlap-neighborhood walks.

    Unequal signatures exclude isomorphism. Equal signatures are rejected
    conservatively; equality is not an isomorphism certificate.
    """
    from collections import Counter
    rows = list(hx) + list(hz)
    weights = [r.bit_count() for r in rows]
    adjacency = [0] * len(rows)
    overlaps = [dict() for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
                overlaps[i][j] = overlaps[j][i] = overlap
    profiles = []
    for i, neighborhood in enumerate(overlaps):
        counts = Counter(
            (weights[j], overlap,
             (adjacency[i] & adjacency[j]).bit_count())
            for j, overlap in neighborhood.items())
        profiles.append((weights[i], tuple(sorted(counts.items()))))
    qubits = [[] for _ in range(n)]
    for i, original in enumerate(rows):
        word = original
        while word:
            bit = word & -word
            qubits[bit.bit_length() - 1].append(weights[i])
            word ^= bit
    return (
        n, tuple(sorted(Counter(profiles).items())),
        tuple(sorted(Counter(tuple(sorted(v)) for v in qubits).items())))


def _c54_logical_le5(checks, stabilizers, n):
    """Complete existence test through five, after logical-four exclusion.

    Anchor every qubit, branch on an unsatisfied check, and finish with
    all singleton/pair syndromes. Equal tails differ by a weight-at-most-
    four stabilizer, so one representative suffices in the quotient.
    No full-code transitivity or search-budget cutoff is assumed.
    """
    cols = _z23_columns(checks, n)
    basis = _q85_basis(stabilizers)
    bits = [1 << i for i in range(n)]
    tails = {0: 0}
    incidence = [[] for _ in checks]
    degree = max(c.bit_count() for c in cols)
    for i, syndrome in enumerate(cols):
        tails.setdefault(syndrome, bits[i])
        value = syndrome
        while value:
            bit = value & -value
            incidence[bit.bit_length() - 1].append(i)
            value ^= bit
    for i, syndrome in enumerate(cols):
        for j in range(i):
            tails.setdefault(syndrome ^ cols[j], bits[i] | bits[j])

    def visit(value, remaining, word):
        if not value:
            return _q85_outside(word, basis)
        if value.bit_count() > degree * remaining:
            return False
        if remaining <= 2:
            tail = tails.get(value)
            return (tail is not None
                    and tail.bit_count() <= remaining
                    and _q85_outside(word ^ tail, basis))
        check = (value & -value).bit_length() - 1
        for j in incidence[check]:
            if not word & bits[j]:
                if visit(value ^ cols[j], remaining - 1,
                         word ^ bits[j]):
                    return True
        return False

    return any(visit(syndrome, 4, bits[i])
               for i, syndrome in enumerate(cols))


@lru_cache(maxsize=1)
def _c54_bank(G):
    """Fresh equal-cubic designs; none of the historical banks is invoked."""
    from itertools import combinations

    assert G.q == 54 and A_SHAPE == B_SHAPE == (1, 3)
    q, mask = G.q, (1 << G.q) - 1
    zero = [[[], [], []]]
    atlas = {}

    # Each permutation contribution is extracted from the scaffold's lift.
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[g], [], []]]
            A, B = (unit, zero) if side == "A" else (zero, unit)
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 540
            source = hx if side == "A" else hz
            cols = tuple(_z23_columns(
                [row & mask for row in source[:q]], q))
            assert all(c.bit_count() == 1 for c in cols)
            blocks.append(cols)
        atlas[side] = tuple(blocks)

    def compile_cell(side, cell):
        cols = [0] * q
        for g in cell:
            block = atlas[side][G.idx(g)]
            for h, syndrome in enumerate(block):
                cols[h] ^= syndrome
        assert all(c.bit_count() == 3 for c in cols)
        return tuple(cols)

    orders = {}
    for g in G.elements:
        value = G.e
        for order in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                orders[g] = order
                break
        else:
            raise ValueError("invalid element order")

    def descriptor(cells):
        relative_orders = []
        commuting = spread = 0
        for cell in cells:
            for g, h in combinations(cell, 2):
                relative_orders.append(orders[G.mul(g, G.inv(h))])
            anchor = cell[0]
            u, v = [G.mul(g, G.inv(anchor)) for g in cell[1:]]
            commuting += int(G.mul(u, v) == G.mul(v, u))
            spread += len({g[1] for g in cell})
        return sum(relative_orders), spread, commuting

    def component_pool(side):
        classes, incidences = {}, set()
        others = [g for g in G.elements if g != G.e]
        for g, h in combinations(others, 2):
            cell = (G.e, g, h)
            cols = compile_cell(side, cell)
            rank = gf2_rank(cols)
            if rank not in (q - 4, q - 2):
                continue
            incidence = tuple(sorted(cols))
            if incidence in incidences or _z23_short(cols) != 5:
                continue
            incidences.add(incidence)
            image = _c54_span(cols)
            classes.setdefault(image, []).append((cell, cols))

        local = random.Random(
            BASE_SEED * 100003 + 54045017 + (side == "B") * 104729)
        raw, seen = [], set()
        for image, entries in sorted(classes.items()):
            if len(entries) < 3:
                continue
            accepted = []
            triples = list(combinations(range(len(entries)), 3))
            local.shuffle(triples)
            for ids in triples[:384]:
                selected = [entries[t] for t in ids]
                cells = tuple(item[0] for item in selected)
                labels = tuple(sorted(cells))
                if labels in seen:
                    continue
                seen.add(labels)
                if not _c54_connected(G, cells):
                    continue
                cols = tuple(c for item in selected for c in item[1])
                if _z23_short(cols) != 5:
                    continue
                # Cubic column parity excludes every odd kernel word,
                # including five. Six is counted below, not guessed.
                accepted.append(dict(
                    cells=(cells,), cols=cols, rank=len(image),
                    cycles=_z23_cycles(cols), labels=labels,
                    descriptor=descriptor(cells)))
                if len(accepted) >= 12:
                    break
            accepted.sort(key=lambda a: (a["cycles"], a["labels"]))
            raw.extend(accepted[:6])

        pool = []
        for rank in (q - 4, q - 2):
            entries = sorted(
                (a for a in raw if a["rank"] == rank),
                key=lambda a: (a["cycles"], a["labels"]))
            for item in entries[:24]:
                item["six"] = _z13_six_count(item["cols"])
                item["reading"] = min(
                    8, _z23_reading(item["cols"], local, 16))
                if item["six"]:
                    item["reading"] = min(item["reading"], 6)
                pool.append(item)
        pool.sort(key=lambda a: (
            -a["reading"], a["six"], a["cycles"], a["labels"]))
        return pool

    pools = {side: component_pool(side) for side in ("A", "B")}
    if not pools["A"] or not pools["B"]:
        raise ValueError("no connected compatible singular cubic constituent")

    def full_component_columns(hx, hz):
        ar = [
            sum(((row >> (j * 3 * q)) & mask) << (j * q)
                for j in range(3))
            for row in hx[:q]]
        br = [row & ((1 << (3*q)) - 1) for row in hz[:q]]
        return tuple(_z23_columns(ar, 3*q)), tuple(_z23_columns(br, 3*q))

    bank, emitted = [], set()
    for slot in range(N_SLOTS):
        stratum, model = divmod(slot % 20, 4)
        local = random.Random(
            BASE_SEED * 100007 + 54045131 + 100019 * slot)

        def choices(side, broaden):
            entries = list(pools[side])
            # Five intrinsic six-word-load regions, not saved orbit copies.
            entries.sort(key=lambda a: (
                a["six"], a["cycles"], a["rank"], a["labels"]))
            start = stratum * len(entries) // 5
            region = [
                entries[(start+t) % len(entries)]
                for t in range(min(12, len(entries)))]
            if broaden:
                region = entries

            def policy(a):
                order_sum, spread, commuting = a["descriptor"]
                preference = (
                    order_sum, -order_sum, spread, -commuting)[model]
                return (a["reading"], preference,
                        -a["six"], -a["cycles"])

            region.sort(key=policy, reverse=True)
            return region[:8 if broaden else 6]

        winner = None
        for broaden in (False, True):
            proposals = []
            for a in choices("A", broaden):
                for b in choices("B", broaden):
                    A, B = a["cells"], b["cells"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if not (425*n <= 1000*k < 475*n):
                        continue
                    ac, bc = full_component_columns(hx, hz)
                    assert ac == a["cols"] and bc == b["cols"]
                    d = min(a["reading"], b["reading"], 8)
                    proposals.append((
                        (k*d*d, d, -a["six"]-b["six"],
                         -a["cycles"]-b["cycles"]),
                        a, b, hx, hz, n, k))
            proposals.sort(key=lambda item: item[0], reverse=True)

            finalists = []
            for key, a, b, hx, hz, n, k in proposals:
                fingerprint = _c54_fingerprint(hx, hz, n)
                if fingerprint in emitted:
                    continue
                low, cycles = _q85_short_screen(hx, hz, n)
                if low != 5:
                    continue
                if _c54_logical_le5(hx, hz, n):
                    continue
                if _c54_logical_le5(hz, hx, n):
                    continue
                dx = _f21_logical_probe(hz, hx, n, local, 8)
                dz = _f21_logical_probe(hx, hz, n, local, 8)
                d = min(a["reading"], b["reading"], dx, dz, 8)
                if d < 6:
                    continue
                finalists.append(dict(
                    a=a, b=b, hx=hx, hz=hz, n=n, k=k,
                    fingerprint=fingerprint, d=d, cycles=cycles))
                if len(finalists) >= 4:
                    break

            # Independent equal deepening budgets; earlier witnesses persist.
            for item in finalists:
                da = _z23_reading(item["a"]["cols"], local, 24)
                db = _z23_reading(item["b"]["cols"], local, 24)
                dx = _f21_logical_probe(
                    item["hz"], item["hx"], item["n"], local, 12)
                dz = _f21_logical_probe(
                    item["hx"], item["hz"], item["n"], local, 12)
                item["d"] = min(item["d"], da, db, dx, dz)
                objective = (
                    item["k"] * item["d"] ** 2, item["d"],
                    -item["a"]["six"]-item["b"]["six"],
                    -item["cycles"])
                if winner is None or objective > winner[0]:
                    winner = (objective, item)
            if winner is not None:
                break

        if winner is None:
            bank.append(None)
            continue
        objective, item = winner
        A, B = item["a"]["cells"], item["b"]["cells"]
        assert css_ok(item["hx"], item["hz"])
        assert max(r.bit_count() for r in item["hx"] + item["hz"]) == 12
        assert _grid_weights([[3, 3, 3]], [[3, 3, 3]]) == 12
        assert _no_dead_lines([[3, 3, 3]], [[3, 3, 3]])
        assert all(len(cell) == len(set(cell)) == 3
                   for grid in (A, B) for row in grid for cell in row)
        emitted.add(item["fingerprint"])
        bank.append(dict(
            cells={"A": A, "B": B}, k=item["k"],
            card=slot % 20, selection=objective,
            component_ranks=(item["a"]["rank"], item["b"]["rank"]),
            component_six=(item["a"]["six"], item["b"]["six"]),
            full_logical_through_five_excluded=True))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """The new group/shape bank supersedes every historical optimizer."""
    selected = _c54_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh singular cubic design in cell 0.450")
    return selected


def CELL(rng, G, side, i, j, ctx):
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's uniform rolled D14 fillings did not suppress sector-2 weight-4 logicals: all five measured codes had d30s=4.
# REFUTED: Q8 x Z5 order-20 quaternion-axis binomials plus exact-four candidate selection did not break the weight-four plateau: all five measured codes had d30s=4, with minima entirely in sector 1.
# REFUTED: A trinomials and distinct relative labels on seven all-monomial B columns exclude weight-four logicals; parent slots 0..3 instead had d2s=4.
# REFUTED: Zero/triple B rewiring and deeper component screening improve the Z23 (1,3)x(2,7) five-code score; all five current measurements remained k=230, d30s=6, fitness 15.652.
# REFUTED: All four intrinsic A triple policies provide fresh screened Singer rank-bonus fillings: the Z13 parent emitted only modes 0/1 in geometries 1/4, leaving 48 of 60 slots void.
# REFUTED: Noncommuting A trinomials and isolated order-seven B binomials produce five distance-seven readings on F21: all five measured codes had d30s=6, and all 45 isolated-binomial cards were void.
# REFUTED: F21 common-image enumeration supplies rank-18 k=225 leaders: all 53 distinct current-parent codes instead had k=220, and the five measured readings remained six.
# REFUTED: C21 common-image enumeration raises the measured leaders to k=225: all five current measurements instead had k=220 and d30s=6, leaving fitness unchanged at 16.398.
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

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
CONCEPT = ("On C3 semidirect C8 with inversion, order 24, shapes (1,3)x(2,7), parity-correlated B rows and clean mixed-degree A constituents should produce five fresh k>=252 codes retaining measured readings at least six, exceeding fitness 16.398 in cell 0.450.")
GROUP = ("Frozen byte-for-byte under L1. The parent measured five [552,240,6] codes, while this same group-and-shape bed reached five [552,250,6] codes. These readings motivate additional verified rank deficits, not another unsupported inference that long-order voltages raise distance.")
SHAPES = ("Frozen A=(1,3), B=(2,7): n=552, kb=240, guaranteed rate 10/23 and admissible q<=26; the group fixes q=24. Components have 72 variables/24 checks and 168 variables/48 checks. Parent minimum witnesses were entirely sector one, often emphasizing Acol1 and Bcol0..2, with no Arow/Brow minimum witnesses. Require 240<=k<=262 so every emitted code remains in cell 0.450.")
FILLING = ("Twenty cards cross five support architectures with four relative-order and quotient-occupancy policies. Three architectures have identical B augmentation-row parities, giving a constructive component dependency while retaining degree-four hyperedges; mixed A profiles redistribute the row budget. Enumerate distinct clean supports and compile both component actions exclusively from verified unit lifts. Prefer lower component ranks and smaller anchored six-word loads. Verify final ranks, compiled components, CSS and row weights; require complete stabilizer-aware logical exclusion through five in both types. A seventh-distance selection reward requires full logical-six exclusion too. Unequal uncoloured Tanner fingerprints separate emitted designs; failed cards never invoke the archived historical bank.")
FAMILY_NAME = "r045_C3sdC8_13x27_paritycorrelated_mixeddegree_logical5_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    # x^3=y^8=1, y*x*y^-1=x^-1; normal form x^a y^b.
    # Inversion has order two, which divides eight.
    if not valid_q(q):
        raise ValueError("C3 semidirect C8 requires order 24")
    els = [(a, b) for b in range(8) for a in range(3)]

    def red(g):
        return (g[0] % 3, g[1] % 8)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b % 2 else c)) % 3, (b + d) % 8)

    def inv(g):
        a, b = red(g)
        return ((a if b % 2 else -a) % 3, (-b) % 8)

    def idx(g):
        a, b = red(g)
        return 3*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


from collections import Counter
from functools import lru_cache


def _d11_support(card):
    """Three degree-two and four degree-three B columns; row totals 9/9."""
    sa = [[3, 3, 3]]
    sb = [[1] * 7 for _ in range(2)]
    for j in (3, 4):
        sb[0][j] = 2
    for j in (5, 6):
        sb[1][j] = 2
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    return sa, sb


def _long_order(G, g):
    word = G.e
    for order in range(1, G.q + 1):
        word = G.mul(word, g)
        if word == G.e:
            return order
    raise ValueError("invalid finite-group element")


def _d11_floor6(cols):
    """Exact short-kernel exclusion; seven means only 'none through six'."""
    seen = {0: 0}

    def insert(syndrome, word):
        previous = seen.get(syndrome)
        if previous is not None:
            difference = previous ^ word
            if difference:
                return difference.bit_count()
        else:
            seen[syndrome] = word
        return 0

    bits = [1 << i for i in range(len(cols))]
    for i, ci in enumerate(cols):
        hit = insert(ci, bits[i])
        if hit:
            return hit
    for i, ci in enumerate(cols):
        for j in range(i):
            hit = insert(ci ^ cols[j], bits[i] | bits[j])
            if hit:
                return hit
    for i, ci in enumerate(cols):
        for j in range(i):
            syndrome = ci ^ cols[j]
            word = bits[i] | bits[j]
            for h in range(j):
                hit = insert(syndrome ^ cols[h], word | bits[h])
                if hit:
                    return hit
    return 7


def _d11_a_cell(rng, G, weight, mode):
    """Three prescribed C8 quotient positions, with fresh C3 coordinates."""
    patterns = (
        (0, 1, 2), (0, 1, 3), (0, 1, 4),
        (0, 2, 4), (0, 2, 5),
    )
    assert weight == 3
    positions = patterns[mode % len(patterns)]
    for _ in range(128):
        phase = rng.randrange(8)
        direction = rng.choice((-1, 1))
        cell = [(rng.randrange(3), (phase + direction*b) % 8)
                for b in positions]
        if any(_long_order(G, G.mul(G.inv(g), h)) == 2
               for i, g in enumerate(cell) for h in cell[:i]):
            continue
        return cell
    raise ValueError("quotient occupancy produced no legal trinomial")


def _d11_draw(rng, G, sa, sb, card):
    """Twenty occupancy multisets, not automorphism images of one labeling."""
    from itertools import combinations_with_replacement

    recipes = tuple(combinations_with_replacement(range(5), 3))[:20]
    recipe = list(recipes[card % 20])
    rng.shuffle(recipe)
    A = [[_d11_a_cell(rng, G, 3, model) for model in recipe]]

    # Three monomial columns form a cubic bipartite subgraph.
    # Pairwise relative order >=4 excludes its immediate 4/6 cycles;
    # the verified syndrome screen also detects other short relations.
    long_elements = [g for g in G.elements if _long_order(G, g) >= 4]
    relatives = None
    for _ in range(128):
        u, v = rng.sample(long_elements, 2)
        if _long_order(G, G.mul(G.inv(u), v)) >= 4:
            relatives = (G.e, u, v)
            break
    if relatives is None:
        raise ValueError("no long-order monomial triple")

    B = [[None] * 7 for _ in range(2)]
    for j, relative in enumerate(relatives):
        top = G.pick(rng)
        # top * bottom^-1 = relative, matching the verified B action.
        bottom = G.mul(G.inv(relative), top)
        B[0][j], B[1][j] = [top], [bottom]

    order8 = [g for g in long_elements if _long_order(G, g) == 8]
    order12 = [g for g in long_elements if _long_order(G, g) == 12]
    order4 = [g for g in long_elements if _long_order(G, g) == 4]
    pools = (order8, order12, order8 + order12, order4 + order8)
    for i in range(2):
        for j in range(3, 7):
            anchor = G.pick(rng)
            cell = [anchor]
            if sb[i][j] == 2:
                pool = pools[(card + i + j) % 4]
                cell.append(G.mul(anchor, rng.choice(pool)))
            assert len(cell) == sb[i][j] and len(set(cell)) == len(cell)
            B[i][j] = cell
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
    """Read both components from the scaffold's verified column ordering."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for row in hx[(i*cb)*q:(i*cb + 1)*q]:
            compact = 0
            for j in range(ca):
                compact |= ((row >> (j*cb*q)) & mask) << (j*q)
            ar.append(compact)
    bm = (1 << (cb*q)) - 1
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


def _d11_kernel_reading(cols, rng, trials):
    """Sample exact nonzero kernel words, retaining short witnesses.

    Information-set elimination, basis pairs, and greedy XOR descent
    provide upper bounds on classical distance, never certificates.
    """
    best = len(cols) + 1
    witnesses = set()
    order = list(range(len(cols)))

    def observe(word):
        nonlocal best
        if not word:
            return
        weight = word.bit_count()
        if weight < best:
            best = weight
        if weight <= best + 2:
            witnesses.add(word)

    for _ in range(trials):
        rng.shuffle(order)
        pivots, words = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                words.append(word)
                observe(word)

        words.sort(key=int.bit_count)
        for i, word in enumerate(words):
            for other in words[:i]:
                observe(word ^ other)

        for start in words[:6]:
            word = start
            for _ in range(4):
                improved = word
                weight = word.bit_count()
                for other in words:
                    trial = word ^ other
                    tw = trial.bit_count()
                    if trial and tw < weight:
                        improved, weight = trial, tw
                if improved == word:
                    break
                word = improved
                observe(word)
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
    """Check actual commutation and stabilizer membership after embedding."""
    ca, cb = A_SHAPE[1], B_SHAPE[1]
    mask = (1 << q) - 1
    checks, stabilizers = (hx, hz) if side == "A" else (hz, hx)
    basis = _d11_basis(stabilizers)
    best = n + 1
    for word in words:
        weight = word.bit_count()
        if weight >= best:
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
    """Optimize fresh quotient recipes and separate final incidence graphs."""
    bank = []
    fingerprints = set()
    assert G.q == 24

    for slot in range(N_SLOTS):
        card = slot % 20
        sa, sb = _d11_support(card)
        rng = random.Random(45024038 + 100003*slot)
        chosen = None

        for batch in range(3):
            finalists = []
            for attempt in range(40):
                A, B = _d11_draw(rng, G, sa, sb, card)
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != 552 or not 240 <= k <= 262:
                    continue
                ac, bc = _d11_components(hx, hz, G.q)
                if min(_d11_floor4(ac), _d11_floor4(bc)) < 5:
                    continue

                # The first three B columns are exactly the monomial
                # subgraph. No independent left/right lift is implemented.
                mono_floor = _d11_floor6(bc[:3*G.q])
                a_floor = _d11_floor6(ac)
                # Explore a broader fallback only in the third batch.
                if batch < 2 and min(mono_floor, a_floor) < 7:
                    continue

                da, wa = _d11_kernel_reading(ac, rng, 8)
                db, wb = _d11_kernel_reading(bc, rng, 8)
                cycles = _d11_cycles(ac) + _d11_cycles(bc)
                d = min(da, db, 12)
                key = (k*d*d/n, min(mono_floor, a_floor),
                       da + db, -cycles)
                finalists.append({
                    "key": key, "A": A, "B": B, "hx": hx, "hz": hz,
                    "n": n, "k": k, "ac": ac, "bc": bc,
                    "da": da, "db": db, "wa": wa, "wb": wb,
                    "cycles": cycles,
                })
                finalists.sort(key=lambda item: item["key"], reverse=True)
                del finalists[4:]

            deep = []
            for item in finalists:
                da, wa = _d11_kernel_reading(item["ac"], rng, 32)
                db, wb = _d11_kernel_reading(item["bc"], rng, 32)
                item["da"] = min(item["da"], da)
                item["db"] = min(item["db"], db)
                wa = sorted(set(wa + item["wa"]),
                            key=lambda v: (v.bit_count(), v))[:48]
                wb = sorted(set(wb + item["wb"]),
                            key=lambda v: (v.bit_count(), v))[:48]
                dz = _d11_logical_reading(
                    wa, "A", item["hx"], item["hz"], G.q, item["n"])
                dx = _d11_logical_reading(
                    wb, "B", item["hx"], item["hz"], G.q, item["n"])
                if max(dx, dz) > item["n"]:
                    continue

                # Exact full-component screening detects mixed
                # monomial/hyperedge words missed by the subgraph test.
                fa = _d11_floor6(item["ac"])
                fb = _d11_floor6(item["bc"])
                d = min(dx, dz, item["da"], item["db"], 12)
                if fa < 7:
                    d = min(d, fa)
                if fb < 7:
                    d = min(d, fb)
                key = (item["k"]*d*d/item["n"], min(fa, fb),
                       dx + dz, -item["cycles"])
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
                assert 425*n <= 1000*item["k"] < 475*n
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


@lru_cache(maxsize=4)
def NEW_CTX(rng, G, idx):
    """Pure shared compilation and optimization; historical bank is inactive."""
    if idx >= 0:
        selected = NEW_CTX(None, G, -1)[idx]
        if selected is None:
            raise ValueError("no fresh parity-correlated card passed logical-five screening")
        return selected

    from itertools import combinations

    assert G.q == 24
    assert A_SHAPE == (1, 3) and B_SHAPE == (2, 7)
    q = G.q
    mask = (1 << q) - 1
    za = [[[] for _ in range(3)]]
    zb = [[[] for _ in range(7)] for _ in range(2)]

    # Unit blocks use precisely the evaluator's left/right conventions.
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            A = [[list(c) for c in row] for row in za]
            B = [[list(c) for c in row] for row in zb]
            (A if side == "A" else B)[0][0] = [g]
            hx, hz, _ = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            cols = _d11_columns([row & mask for row in source[:q]], q)
            assert all(c.bit_count() == 1 for c in cols)
            blocks.append(tuple(cols))
        atlas[side] = tuple(blocks)

    def compile_grid(side, grid):
        cols = [0] * (len(grid[0]) * q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = atlas[side][G.idx(g)]
                    for h, syndrome in enumerate(block):
                        cols[j*q+h] ^= syndrome << (i*q)
        return tuple(cols)

    def canonical_span(cols):
        basis = _d11_basis(cols)
        pivots = sorted(basis)
        for a, p in enumerate(pivots):
            for higher in pivots[a+1:]:
                if basis[higher] & (1 << p):
                    basis[higher] ^= basis[p]
        return tuple((p, basis[p]) for p in pivots)

    def kernel_profile(cols, height, cap=12):
        """Complete exclusion through five; capped anchored six-word load.

        The verified component actions commute with regular translations
        on their variable fibres. Every short component word therefore
        has a translate containing one of the listed block anchors.
        Zero six-load exhausts six; a positive load is only a capped census.
        """
        if _d11_floor4(cols) != 5:
            return None
        bits = [1 << i for i in range(len(cols))]
        tails = {0: 0}
        incidence = [[] for _ in range(height)]
        degree = max(c.bit_count() for c in cols)
        for i, syndrome in enumerate(cols):
            assert syndrome not in tails
            tails[syndrome] = bits[i]
            value = syndrome
            while value:
                bit = value & -value
                incidence[bit.bit_length()-1].append(i)
                value ^= bit
        for i, syndrome in enumerate(cols):
            for j in range(i):
                value = syndrome ^ cols[j]
                assert value not in tails
                tails[value] = bits[i] | bits[j]

        def decode(target, limit, forbidden, limit_hits):
            found = set()

            def visit(value, remaining, word):
                if len(found) >= limit_hits:
                    return
                if not value:
                    found.add(word)
                    return
                if value.bit_count() > degree * remaining:
                    return
                if remaining <= 2:
                    tail = tails.get(value)
                    if (tail is not None
                            and tail.bit_count() <= remaining
                            and not tail & (word | forbidden)):
                        found.add(word | tail)
                    return
                check = (value & -value).bit_length()-1
                for j in incidence[check]:
                    if bits[j] & (word | forbidden):
                        continue
                    visit(value ^ cols[j], remaining-1, word | bits[j])
                    if len(found) >= limit_hits:
                        return

            visit(target, limit, 0)
            return found

        anchors = range(0, len(cols), q)
        for anchor in anchors:
            if decode(cols[anchor], 4, bits[anchor], 1):
                return None
        return sum(len(decode(cols[a], 5, bits[a], cap))
                   for a in anchors)

    def has_logical(checks, stabilizers, n, bound):
        """Complete logical-five/six test after logical-four exclusion.

        Every qubit is anchored: no full-code transitivity is assumed.
        Equal singleton/pair syndromes differ by a weight-at-most-four
        stabilizer, so one representative suffices in the logical quotient.
        """
        cols = _d11_columns(checks, n)
        span = _d11_basis(stabilizers)
        bits = [1 << i for i in range(n)]
        incidence = [[] for _ in checks]
        tails = {0: 0}
        degree = max(c.bit_count() for c in cols)
        for i, syndrome in enumerate(cols):
            tails.setdefault(syndrome, bits[i])
            value = syndrome
            while value:
                bit = value & -value
                incidence[bit.bit_length()-1].append(i)
                value ^= bit
        for i, syndrome in enumerate(cols):
            for j in range(i):
                tails.setdefault(syndrome ^ cols[j], bits[i] | bits[j])

        def visit(value, remaining, word):
            if not value:
                return _d11_outside(word, span)
            if value.bit_count() > degree * remaining:
                return False
            if remaining <= 2:
                tail = tails.get(value)
                return (tail is not None
                        and tail.bit_count() <= remaining
                        and _d11_outside(word ^ tail, span))
            check = (value & -value).bit_length()-1
            for j in incidence[check]:
                if word & bits[j]:
                    continue
                if visit(value ^ cols[j], remaining-1, word ^ bits[j]):
                    return True
            return False

        return any(visit(s, bound-1, bits[i])
                   for i, s in enumerate(cols))

    # Incidence equality removes independent column-gauge copies before
    # supports enter the library. Actual final graph invariants finish
    # the discrimination; support labels themselves are not certificates.
    library = {side: {w: [] for w in (2, 3, 4)}
               for side in ("A", "B")}
    others = [g for g in G.elements if g != G.e]
    orders = {g: _long_order(G, g) for g in G.elements}
    for side in ("A", "B"):
        for weight in (2, 3, 4):
            seen = set()
            for selected in combinations(others, weight-1):
                cell = (G.e,) + selected
                if weight == 2 and orders[selected[0]] < 8:
                    continue
                cols = compile_grid(side, [[cell]])
                incidence = tuple(sorted(cols))
                if incidence in seen or _d11_floor4(cols) != 5:
                    continue
                seen.add(incidence)
                span = canonical_span(cols)
                occupancy = tuple(sorted(Counter(g[1] for g in cell).values()))
                library[side][weight].append(dict(
                    cell=cell, cols=cols, rank=len(span), span=span,
                    occupancy=occupancy,
                    spread=len({g[1] for g in cell}),
                    label=tuple(G.idx(g) for g in cell)))
            library[side][weight].sort(
                key=lambda item: (item["rank"], item["label"]))

    assert all(library[side][w]
               for side in ("A", "B") for w in (2, 3, 4))

    # Each entry specifies A counts and all seven B column counts.
    # Profiles 1,2,4 have equal augmentation-row parities in B.
    # Their degree-four columns avoid an entirely degree-two constituent.
    cards = (
        ((3, 3, 3),
         ((2, 0), (0, 2), (2, 1), (1, 2),
          (1, 1), (1, 1), (1, 1))),
        ((2, 3, 3),
         ((2, 2), (2, 2), (2, 0), (0, 2),
          (1, 1), (1, 1), (1, 1))),
        ((2, 3, 3),
         ((3, 1), (1, 3), (2, 0), (0, 2),
          (1, 1), (1, 1), (1, 1))),
        ((2, 2, 4),
         ((2, 0), (0, 2), (2, 1), (1, 2),
          (1, 1), (1, 1), (1, 1))),
        ((2, 3, 3),
         ((4, 0), (0, 4), (2, 0), (0, 2),
          (1, 1), (1, 1), (1, 1))),
    )
    cubic_classes = {}
    for record in library["A"][3]:
        cubic_classes.setdefault(record["span"], []).append(record)
    cubic_classes = [bucket for bucket in cubic_classes.values()
                     if len(bucket) >= 3]
    assert cubic_classes

    # A pools are compiled once. Four policies subsequently select
    # different intrinsic six-load regions rather than translated copies.
    apools = []
    for profile, (counts, _) in enumerate(cards):
        local = random.Random(BASE_SEED*100003 + 24045317 + profile)
        raw, seen = [], set()
        for attempt in range(1536):
            if counts == (3, 3, 3):
                bucket = local.choice(cubic_classes)
                records = local.sample(bucket, 3)
            else:
                records = [local.choice(library["A"][counts[0]])]
                joined = records[0]["cols"]
                for weight in counts[1:]:
                    entries = library["A"][weight]
                    trial = local.sample(entries, min(32, len(entries)))
                    local.shuffle(trial)
                    trial.sort(key=lambda item: (
                        gf2_rank(joined + item["cols"]),
                        abs(item["spread"] - (2 + attempt % 3))))
                    record = local.choice(trial[:min(4, len(trial))])
                    records.append(record)
                    joined += record["cols"]
            labels = tuple(record["label"] for record in records)
            if labels in seen:
                continue
            seen.add(labels)
            A = [[record["cell"] for record in records]]
            ac = compile_grid("A", A)
            if _d11_floor4(ac) != 5:
                continue
            # Cubic columns exclude odd kernel words automatically.
            if counts != (3, 3, 3):
                load = kernel_profile(ac, q, cap=4)
                if load is None:
                    continue
            else:
                load = None
            item = dict(A=A, cols=ac, rank=gf2_rank(ac),
                        cycles=_d11_cycles(ac), labels=labels, six=load)
            raw.append(item)
            raw.sort(key=lambda a: (a["rank"], a["cycles"], a["labels"]))
            del raw[32:]
        pool = []
        for item in raw:
            if item["six"] is None:
                item["six"] = kernel_profile(item["cols"], q, cap=12)
            if item["six"] is not None:
                pool.append(item)
        pool.sort(key=lambda a: (
            a["rank"], a["six"], a["cycles"], a["labels"]))
        apools.append(pool)

    long8 = [g for g in others if orders[g] == 8]
    long12 = [g for g in others if orders[g] == 12]
    long4 = [g for g in others if orders[g] == 4]
    assert long8 and long12

    def draw_b(local, sb, mode, lane):
        B = [[None] * 7 for _ in range(2)]
        mono = [j for j in range(7) if sb[0][j] == sb[1][j] == 1]
        best, best_cost = None, None
        for _ in range(96):
            relatives = [G.e] + local.sample(others, len(mono)-1)
            ratios = [G.mul(g, G.inv(h))
                      for g in relatives for h in relatives if g != h]
            cost = (sum(orders[g] <= 3 for g in ratios),
                    len(ratios)-len(set(ratios)))
            if best_cost is None or cost < best_cost:
                best, best_cost = relatives, cost
            if cost == (0, 0):
                break
        for j, relative in zip(mono, best):
            top = G.pick(local)
            B[0][j] = [top]
            B[1][j] = [G.mul(G.inv(relative), top)]

        for i in range(2):
            for j in range(7):
                if B[i][j] is not None:
                    continue
                count = sb[i][j]
                if count == 0:
                    cell = []
                elif count == 1:
                    cell = [G.pick(local)]
                elif count == 2:
                    isolated = sb[1-i][j] == 0
                    pools = (long8, long12, long8+long12,
                             long8+long12 if isolated
                             else long4+long8+long12)
                    base = G.pick(local)
                    cell = [base, G.mul(base, local.choice(pools[mode]))]
                else:
                    entries = library["B"][count]
                    choices = local.sample(entries, min(24, len(entries)))
                    target = 2 + ((mode + lane + i + j) % (count-1))
                    choices.sort(key=lambda item: (
                        abs(item["spread"]-target),
                        item["rank"] if mode % 2 else -item["rank"]))
                    support = local.choice(
                        choices[:min(6, len(choices))])["cell"]
                    gauge = G.pick(local)
                    cell = [G.mul(gauge, g) for g in support]
                assert len(cell) == count == len(set(cell))
                B[i][j] = cell
        return B

    bank, emitted = [], set()
    for slot in range(N_SLOTS):
        profile, mode = divmod(slot % 20, 4)
        counts, columns = cards[profile]
        sa = [list(counts)]
        sb = [[column[i] for column in columns] for i in range(2)]
        assert _grid_weights(sa, sb) <= 12
        assert _no_dead_lines(sa, sb)
        if profile in (1, 2, 4):
            assert all((sb[0][j] ^ sb[1][j]) % 2 == 0
                       for j in range(7))

        pool = apools[profile]
        if not pool:
            bank.append(None)
            continue
        primary_rank = min(a["rank"] for a in pool)
        primary = sorted(
            (a for a in pool if a["rank"] == primary_rank),
            key=lambda a: (a["six"], a["cycles"], a["labels"]))
        start = mode * len(primary) // 4
        achoices = [primary[(start+t) % len(primary)]
                    for t in range(min(6, len(primary)))]
        remaining = [a for a in pool if a not in achoices]
        local = random.Random(
            BASE_SEED*100007 + 24045443 + 100019*slot)
        selected = None

        for batch in range(3):
            bp, seen_b = [], set()
            for attempt in range(512 + 256*batch):
                B = draw_b(local, sb, mode, slot//20 + batch)
                labels = tuple(tuple(sorted(cell))
                               for row in B for cell in row)
                if labels in seen_b:
                    continue
                seen_b.add(labels)
                bc = compile_grid("B", B)
                if _d11_floor4(bc) != 5:
                    continue
                load = kernel_profile(bc, 2*q, cap=8)
                if load is None:
                    continue
                bp.append(dict(
                    B=B, cols=bc, rank=gf2_rank(bc), six=load,
                    cycles=_d11_cycles(bc), labels=labels))
                bp.sort(key=lambda b: (
                    b["rank"], b["six"], b["cycles"], b["labels"]))
                del bp[8:]
                if len(bp) == 8 and attempt >= 160:
                    break
            if not bp:
                continue

            choices = list(achoices)
            if batch:
                choices.extend(remaining[:12])
            proposals = []
            for a in choices:
                for b in bp[:6]:
                    A, B = a["A"], b["B"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if n != 552 or not 240 <= k <= 262:
                        continue
                    ac, bc = _d11_components(hx, hz, q)
                    assert tuple(ac) == a["cols"]
                    assert tuple(bc) == b["cols"]
                    possible = 7 if a["six"] == b["six"] == 0 else 6
                    key = (k*possible*possible, k,
                           -a["six"]-b["six"],
                           -a["cycles"]-b["cycles"])
                    proposals.append((key, a, b, hx, hz, n, k))
            proposals.sort(key=lambda item: item[0], reverse=True)

            winner, screened = None, 0
            for _, a, b, hx, hz, n, k in proposals:
                fingerprint = _d11_fingerprint(hx, hz, n)
                if fingerprint in emitted:
                    continue
                if _d11_small_logical(hx, hz, n):
                    continue
                if _d11_small_logical(hz, hx, n):
                    continue
                if has_logical(hx, hz, n, 5):
                    continue
                if has_logical(hz, hx, n, 5):
                    continue
                floor = 6
                if a["six"] == b["six"] == 0:
                    if (not has_logical(hx, hz, n, 6)
                            and not has_logical(hz, hx, n, 6)):
                        floor = 7
                objective = (
                    k*floor*floor, k, -a["six"]-b["six"],
                    -a["cycles"]-b["cycles"])
                if winner is None or objective > winner[0]:
                    winner = (objective, a, b, hx, hz, k, fingerprint, floor)
                screened += 1
                if floor == 7 or screened >= 6:
                    break

            if winner is None:
                continue
            objective, a, b, hx, hz, k, fingerprint, floor = winner
            A, B = a["A"], b["B"]
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= 12
            assert all(len(cell) == len(set(cell)) <= 5
                       for grid in (A, B) for row in grid for cell in row)
            assert 425*552 <= 1000*k < 475*552
            emitted.add(fingerprint)
            selected = {
                "idx": slot, "card": slot % 20, "k": k,
                "selection": objective,
                "verified_exclusion_floor": floor,
                "cells": {
                    side: tuple(tuple(tuple(cell) for cell in row)
                                for row in grid)
                    for side, grid in (("A", A), ("B", B))}}
            break

        # Failed policies never substitute historical or transported codes.
        bank.append(selected)
    return tuple(bank)


def CELL(rng, G, side, i, j, ctx):
    """Return the jointly rank-verified, logical-five-screened full cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: D8 (2,7)x(2,8) distinct monomial relatives and short-syndrome selection did not break the distance-four plateau: all five measured readings were 4, with both logical types entirely in sector 1.
# REFUTED: Dic11 (1,2)x(3,5) degree tradeoffs and logical-four screening yield five d30s readings >=14: two measured readings were 13, with limiting Z witnesses entirely in sector 1.
# REFUTED: Long-order cell differences and cubic monomial-subgraph screening yield five distance-eight readings on C3 semidirect C8 (1,3)x(2,7): all five measured parent codes instead had k=240 and d30s=6, with minimum witnesses entirely in sector 1.
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

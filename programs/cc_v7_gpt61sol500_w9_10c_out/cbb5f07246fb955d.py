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
CONCEPT = ("C7 semidirect_2 C15, order 105, A=B=(1,2): independently optimized noncommuting trinomial spectra and reduction of explicit full-CSS logical witnesses should supply five measured distance readings >=14 in cell 0.200.")
GROUP = ("Jump from C11:C5 to the different odd-order group C7:C15(s=2). Since 2^3=1 mod 7, the action defines a valid group of order 105, with normal C7 and central C5. The related C13:C9 order-117 inspiration measured four distance-18 finalists and one distance-15 finalist. This motivates testing an odd central extension with longer quotient orders, not transferring a distance certificate. This order-105 bed is absent from the displayed BEDS.")
SHAPES = ("Equal (1,2) sides have 210 variables and 105 checks per constituent. They give n=525, kb=105 and guaranteed rate 0.200; the size window permits q<=119. Actual full ranks retain [0.175,0.225). Parent weight-five minima occupied sector-1 Acol/Bcol blocks across its three columns, sometimes all A columns at Bcol2; the earlier weight-four failure also entered Arow0/Brow0. Both constituents and complete centralizers are therefore screened.")
FILLING = ("Every cell contains three distinct elements, with noncommuting normalized relative generators of triangle spectra (15,15,15), (7,15,15) or (15,15,35). Each component row totals six and every column totals three, so BOTH CSS maxima are nine, below nine and twelve. Twenty pairs of constituent-spectrum prescriptions guide independent optimized banks. Verified lifts exclude constituent kernels through four, screen one/two normal-C7 cosets and refine actual kernel witnesses equally. Full matrices enforce rank, CSS and complete kernel-four exclusion; actual nontrivial logical witnesses undergo stabilizer-pair descent. Different intrinsic full-Tanner signatures separate emitted slots conservatively. Larger readings remain upper bounds; unseen archive novelty is evaluator-tested.")
FAMILY_NAME = "r020_c7sd_c15_s2_q105_12x12_independent_triangle_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 105

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C7:C15(s=2) has order 105")
    # x^7=y^15=1, y*x*y^-1=x^2; 2^3=1 modulo 7.
    powers = [pow(2, b, 7) for b in range(15)]
    els = [(a, b) for b in range(15) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 15)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b] * c) % 7, (b + d) % 15)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 15
        return ((-powers[bi] * a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 2)               # (rb, cb)


def _kernel_profile(rows, n):
    """Exact low-weight kernel screen, not a distance-search estimate.

    A reading of five means only that weights one through four were
    excluded. Equal syndromes of words of weight <=2 detect all such
    kernel words. Collision counts are only a secondary selection feature.
    """
    columns = [0] * n
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= check
            row ^= low
    seen = {0: 0}
    lightest, collisions = 5, 0

    def inspect(syndrome, word):
        nonlocal lightest, collisions
        if syndrome in seen:
            difference = word ^ seen[syndrome]
            if difference:
                lightest = min(lightest, difference.bit_count())
                collisions += 1
        else:
            seen[syndrome] = word

    for j, syndrome in enumerate(columns):
        inspect(syndrome, 1 << j)
    for j, syndrome in enumerate(columns):
        for i in range(j):
            inspect(syndrome ^ columns[i], (1 << j) | (1 << i))
    return lightest, collisions


def _component_rows(G, side, grid):
    """Isolate the actual scaffold constituent with a zero auxiliary cell."""
    dummy = [[[]]]
    width = len(grid[0]) * G.q
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, dummy)
    else:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    assert len(rows) == G.q
    assert all(row.bit_length() <= width for row in rows)
    return rows


def _syndrome_columns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= tag
            row ^= bit
    return cols


def _clean_four(rows, width):
    """Exactly exclude every nonzero kernel support of weight <=4."""
    cols = _syndrome_columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return False
    pairs = set()
    for i, first in enumerate(cols):
        for j in range(i):
            syndrome = first ^ cols[j]
            if syndrome in singles or syndrome in pairs:
                return False
            pairs.add(syndrome)
    return True


def _row_basis(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                pivots[p] = word
                break
            word ^= pivots[p]
    return pivots


def _outside_span(word, pivots):
    while word:
        p = word.bit_length() - 1
        if p not in pivots:
            return True
        word ^= pivots[p]
    return False


def _kernel_probe(rows, width, rng, passes, seed):
    """Actual kernel witness, not an exhaustive distance certificate."""
    assert seed and all((row & seed).bit_count() % 2 == 0 for row in rows)
    cols = _syndrome_columns(rows, width)
    best, weight = seed, seed.bit_count()
    order = list(range(width))

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if 0 < w < weight:
            best, weight = word, w

    for _ in range(passes):
        rng.shuffle(order)
        pivots, kernel = {}, []
        for j in order:
            syndrome, word = cols[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = syndrome, word
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                kernel.append(word)
                inspect(word)

        # Elementary exchanges preserve independence and the full kernel.
        rng.shuffle(kernel)
        for i in range(len(kernel)):
            for j in range(len(kernel)):
                if i != j:
                    trial = kernel[i] ^ kernel[j]
                    if 0 < trial.bit_count() < kernel[i].bit_count():
                        kernel[i] = trial
            inspect(kernel[i])

        for i, first in enumerate(kernel):
            for second in kernel[:i]:
                inspect(first ^ second)
        ranked = sorted(kernel, key=int.bit_count)
        short = ranked[:28]
        tail = ranked[28:]
        short += rng.sample(tail, min(8, len(tail)))
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                for h in range(j):
                    inspect(pair ^ short[h])
        for _ in range(2):
            starting = best
            for i, first in enumerate(short):
                inspect(starting ^ first)
                for second in short[:i]:
                    inspect(starting ^ first ^ second)
            if best == starting:
                break

    assert best and all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _tanner_signature(hx, hz):
    """An uncoloured Tanner-isomorphism invariant, not an isomorphism test.

    Check intersections define two adjacency relations. Each check's
    profile counts common neighbours in both relations for every other
    check. Different signatures certify non-isomorphic Tanner graphs.
    """
    rows = hx + hz
    m = len(rows)
    one, two = [0] * m, [0] * m
    for i in range(m):
        for j in range(i):
            c = (rows[i] & rows[j]).bit_count()
            if c:
                adj = one if c == 1 else two
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    profiles = [{} for _ in rows]
    for i in range(m):
        for j in range(i):
            kind = ((one[i] >> j) & 1) + 2 * ((two[i] >> j) & 1)
            key = (kind,
                   (one[i] & one[j]).bit_count(),
                   (one[i] & two[j]).bit_count()
                   + (two[i] & one[j]).bit_count(),
                   (two[i] & two[j]).bit_count())
            for p in (profiles[i], profiles[j]):
                p[key] = p.get(key, 0) + 1
    return tuple(sorted(tuple(sorted(p.items())) for p in profiles))


from functools import lru_cache


@lru_cache(maxsize=1)
def _odd15_portfolio(G, count):
    """Deterministic optimization; cached publication cells are immutable."""
    from itertools import combinations
    from hashlib import sha256

    assert G.q == 105
    q, width = G.q, 2 * G.q
    support = ((3, 3),)
    assert _no_dead_lines(support, support)
    assert _grid_weights(support, support) == 9 <= min(W_MAX, 9, 12)

    def order(g):
        value = G.e
        for length in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                return length
        raise ValueError("invalid element order")

    orders = {g: order(g) for g in G.elements}
    spectra = ((15, 15, 15), (7, 15, 15), (15, 15, 35))
    triangles = [[] for _ in spectra]
    nonidentity = [g for g in G.elements if g != G.e]
    for g, h in combinations(nonidentity, 2):
        if G.mul(g, h) == G.mul(h, g):
            continue
        spectrum = tuple(sorted((
            orders[g], orders[h], orders[G.mul(G.inv(g), h)])))
        if spectrum in spectra:
            triangles[spectra.index(spectrum)].append((g, h))
    if any(not pool for pool in triangles):
        raise ValueError("requested noncommuting triangle spectrum absent")

    profiles = tuple((a, b) for a in range(3) for b in range(a, 3))
    # Twenty structural cards, excluding the all-type-zero pairing.
    cards = tuple((a, b) for a in range(6) for b in range(a, 6)
                  if (a, b) != (0, 0))
    card_number = {pair: i for i, pair in enumerate(cards)}

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def digest(value):
        return sha256(repr(value).encode("ascii")).digest()

    def intrinsic(rows, other=()):
        """Uncoloured check-intersection and degree invariants.

        Inequality certifies nonisomorphism for these codes: check
        degrees are six/nine, while qubit degrees are three/six.
        Equality is conservatively merged, not treated as an iso proof.
        """
        first, second = list(rows), list(other)
        all_rows = first + second
        n = max((row.bit_length() for row in all_rows), default=0)
        cols = _syndrome_columns(all_rows, n)
        return digest((
            tuple(sorted(row.bit_count() for row in all_rows)),
            tuple(sorted(col.bit_count() for col in cols)),
            _tanner_signature(first, second),
        ))

    cosets = tuple(
        tuple(G.idx((a, b)) for a in range(7))
        for b in range(15))

    def component_stats(rows):
        """Exact short screen and one/two normal-C7-coset screen."""
        if not _clean_four(rows, width):
            return None
        cols = _syndrome_columns(rows, width)
        seed = (1 << width) - 1
        assert all((row & seed).bit_count() % 2 == 0 for row in rows)
        best = seed
        seen = {0: 0}
        for block in range(2):
            for coset in cosets:
                word = syndrome = 0
                for t in coset:
                    j = block * q + t
                    word |= 1 << j
                    syndrome ^= cols[j]
                if syndrome in seen:
                    candidate = word ^ seen[syndrome]
                    if candidate and candidate.bit_count() < best.bit_count():
                        best = candidate
                else:
                    seen[syndrome] = word
        penalty = 0
        for i, row in enumerate(rows):
            for other in rows[:i]:
                overlap = (row & other).bit_count()
                penalty += overlap * (overlap - 1) // 2
        return best, penalty

    bank, component_keys = [], set()
    for profile, prescription in enumerate(profiles):
        local = random.Random(BASE_SEED ^ (0x105701 + 104729 * profile))
        proposals, seen = [], set()
        # Independent optimization of every constituent prescription.
        for trial in range(128):
            cells = []
            for kind in prescription:
                g, h = local.choice(triangles[kind])
                if local.randrange(2):
                    g, h = h, g
                cells.append([G.e, g, h])
            grid = [cells]
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for cell in cells)
            if filling in seen:
                continue
            seen.add(filling)
            rows = _component_rows(G, "A", grid)
            rank = gf2_rank(rows)
            if rank < q - 4:
                continue
            stats = component_stats(rows)
            if stats is None:
                continue
            seed, penalty = stats
            cheap = (int(seed.bit_count() <= 14), penalty,
                     -seed.bit_count(), q - rank)
            proposals.append(dict(
                grid=grid, rows=rows, seed=seed, penalty=penalty,
                profile=profile, trial=trial, cheap=cheap))

        proposals.sort(key=lambda item: (item["cheap"], item["trial"]))
        shortlist = []
        for record in proposals[:20]:
            witness = _kernel_probe(
                record["rows"], width, local, 3, record["seed"])
            record["left"] = witness
            record["quality"] = (witness.bit_count(), -record["penalty"])
            shortlist.append(record)
        shortlist.sort(key=lambda item: item["quality"], reverse=True)

        # Every constituent that can be published gets 3+12 passes.
        finalists = shortlist[:12]
        for record in finalists:
            record["left"] = _kernel_probe(
                record["rows"], width, local, 12, record["left"])
            record["quality"] = (
                record["left"].bit_count(), -record["penalty"])
        finalists.sort(key=lambda item: item["quality"], reverse=True)

        kept = 0
        for record in finalists:
            signature = intrinsic(record["rows"])
            if signature in component_keys:
                continue
            right_rows = _component_rows(G, "B", record["grid"])
            if not _clean_four(right_rows, width):
                continue
            # Inversion transfers left constituent supports to right.
            # This transfer is checked against the scaffold's matrix.
            right, source = 0, record["left"]
            while source:
                bit = source & -source
                block, t = divmod(bit.bit_length() - 1, q)
                right |= 1 << (
                    block * q + G.idx(G.inv(G.elements[t])))
                source ^= bit
            if any((row & right).bit_count() & 1 for row in right_rows):
                raise ValueError("verified constituent inversion transfer failed")
            component_keys.add(signature)
            record["right"] = right
            record["reading"] = record["left"].bit_count()
            record["grid"] = freeze(record["grid"])
            bank.append(record)
            kept += 1
            if kept >= 6:
                break
    if len(bank) < 6:
        raise ValueError("too few independently screened constituents")

    def embedded(record, kind, checks, own_rows):
        """Verify logical class, then reduce actual logical witnesses."""
        own = _row_basis(own_rows)
        best = 0
        for block in range(2):
            if kind == "X":
                word = record["right"] << (block * width)
            else:
                word, source = 0, record["left"]
                while source:
                    bit = source & -source
                    j, t = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * 2 + block) * q + t)
                    source ^= bit
            if any((row & word).bit_count() & 1 for row in checks):
                raise ValueError("verified quantum witness embedding failed")
            if not _outside_span(word, own):
                continue
            for _ in range(2):
                starting = word
                for row in own_rows:
                    trial = word ^ row
                    if trial.bit_count() < word.bit_count():
                        word = trial
                if word == starting:
                    break
            if not best or word.bit_count() < best.bit_count():
                best = word
        if not best:
            return 0

        # Same full stabilizer-pair budget for every candidate.
        starting, weight = best, best.bit_count()
        for i, first in enumerate(own_rows):
            for second in own_rows[:i]:
                trial = starting ^ first ^ second
                if trial.bit_count() < weight:
                    best, weight = trial, trial.bit_count()
        assert best and _outside_span(best, own)
        assert all((row & best).bit_count() % 2 == 0 for row in checks)
        return best

    plans = []
    for ai, a in enumerate(bank):
        for bi in range(ai, len(bank)):
            b = bank[bi]
            prescription = tuple(sorted((a["profile"], b["profile"])))
            if prescription not in card_number:
                continue
            priority = (
                min(a["reading"], b["reading"]),
                a["reading"] + b["reading"],
                -a["penalty"] - b["penalty"])
            plans.append((priority, ai, bi, card_number[prescription]))
    plans.sort(key=lambda item: (item[0], -item[1], -item[2]), reverse=True)

    graph_keys, survivors = set(), []
    for preliminary, ai, bi, card in plans:
        a, b = bank[ai], bank[bi]
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 525 or not (k > 0 and 175*n <= 1000*k < 225*n):
            continue
        if not _clean_four(hx, n) or not _clean_four(hz, n):
            continue
        signature = intrinsic(hx, hz)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)

        xword = embedded(b, "X", hz, hx)
        zword = embedded(a, "Z", hx, hz)
        if not xword or not zword:
            continue
        reading = min(xword.bit_count(), zword.bit_count())
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) == 9
        assert lifted_k(G, A, B) == k
        quality = (k * reading * reading / n, reading, k, preliminary)
        survivors.append((
            quality, A, B, k, reading, card,
            (a["profile"], b["profile"]),
            (a["reading"], b["reading"])))
        if len(survivors) >= max(4 * count, 100):
            break
    if not survivors:
        raise ValueError("no complete target-rate design passed full screens")

    survivors.sort(key=lambda item: item[0], reverse=True)
    # Reserve the best available representative of each deliberate card,
    # then fill remaining positions with other invariant-distinct winners.
    chosen, used_cards = [], set()
    remainder = []
    for record in survivors:
        if record[5] not in used_cards and len(chosen) < count:
            chosen.append(record)
            used_cards.add(record[5])
        else:
            remainder.append(record)
    chosen.extend(remainder[:max(0, count - len(chosen))])
    chosen.sort(key=lambda item: item[0], reverse=True)
    return tuple(chosen)


def NEW_CTX(rng, G, idx):
    """Pure publication from an immutable independently optimized portfolio."""
    catalogue = _odd15_portfolio(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("invariant-distinct odd-central-extension portfolio exhausted")
    quality, A, B, k, reading, card, profiles, components = catalogue[idx]
    return {"A": A, "B": B, "screen_k": k,
            "logical_witness_upper_bound": reading,
            "exact_kernel_floor": 5,
            "design_card": card, "constituent_profiles": profiles,
            "constituent_witness_upper_bounds": components,
            "selection_quality": quality}


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell, with no additional sampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: Constituent screens and the two-second proxy suffice to protect the Frobenius-bed finalists from weight-four logicals; slot 30 had k=226 and d2s=5 but d30s=4, with both logical types touching both LP sectors.
# REFUTED: Cross-Cayley pairing supplies five inequivalent k=226 finalists: the measured current program supplied only three; slot 3 collapsed to slot 0 and the remaining two measured finalists had k=222.
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

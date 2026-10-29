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
CONCEPT = ("A5, order 60, A=(1,2), B=(1,3): difference-packed trinomials with component exclusion through weight six and mixed-sector logical screening should improve the parent's measured distance-9..10 plateau in cell 0.300; this is a falsifiable search hypothesis.")
GROUP = ("Jump from Dic21 to the simple alternating group A5, an untried bed in BEDS. Its nonidentity element orders are 2,3,5; prescribed triples exclude involution-separated terms. Order 60 gives n=420. No distance advantage is assumed from simplicity or from the smaller lift.")
SHAPES = ("Keep (1,2)x(1,3) for cell 0.300: n=420, kb=120 and guaranteed rate 2/7. Component variable/check counts are 120/60 and 180/60. Parent weight-nine X witnesses touched both sectors, including Arow0/Brow0; weight-ten witnesses occupied sector-one blocks across all B columns, often emphasizing Acol1. Exact ranks retain only 0.275<=k/n<0.325.")
FILLING = ("Only trinomials are emitted: HX/HZ weights are 9/12, eliminating the parent's distance-five binomial recipes. Twenty primary cards prescribe distinct relative pair-order architectures using 335,355,555 triples. Pack ordered differences and verify component four-cycle exclusion using the scaffold lift. Finalists undergo component-kernel exclusion through weight six, deeper kernel searches and full-code logical searches with stabilizer membership. Invariant fingerprints prevent emitted orbit copies. Search readings are heuristic upper bounds, not distance certificates.")
FAMILY_NAME = "r030_A5_12x13_sidon_pairtypes_full_logical_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 60

def custom_ops(q):
    # A5: even permutations of five points, composed as g after h.
    from itertools import permutations
    if not valid_q(q):
        raise ValueError("A5 has order 60")
    els = [
        p for p in permutations(range(5))
        if sum(p[i] > p[j] for i in range(5)
               for j in range(i + 1, 5)) % 2 == 0
    ]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(g)

    def mul(g, h):
        return tuple(g[h[i]] for i in range(5))

    def inv(g):
        out = [0] * 5
        for i, image in enumerate(g):
            out[image] = i
        return tuple(out)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=tuple(range(5)),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


from functools import lru_cache

# Every cell has three terms: wx=6+3=9, wz=9+3=12.
# Model numbers name sorted relative pair orders 335, 355, 555.
_A5_A_PATTERNS = (
    (0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2),
)
_A5_B_PATTERNS = (
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1),
    (0, 1, 2), (0, 2, 2), (1, 1, 1), (1, 1, 2),
    (1, 2, 2), (2, 2, 2),
)
_A5_CARD_PAIRS = (
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
    (0, 6), (1, 7), (2, 8), (3, 9), (4, 0), (5, 1),
    (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7),
    (2, 9), (4, 8),
)
# If the harness requests sixty slots, use all sixty architecture pairs.
_A5_CARD_PAIRS += tuple(
    (a, b) for a in range(6) for b in range(10)
    if (a, b) not in _A5_CARD_PAIRS
)


def _a5_order(G, g):
    product = G.e
    for order in range(1, G.q + 1):
        product = G.mul(product, g)
        if product == G.e:
            return order
    raise ValueError("invalid element order")


@lru_cache(maxsize=1)
def _a5_triple_catalog(G):
    """Relative architectures, not pre-existing code labelings."""
    orders = {g: _a5_order(G, g) for g in G.elements}
    models = {(3, 3, 5): 0, (3, 5, 5): 1, (5, 5, 5): 2}
    catalog = [[], [], []]
    nonidentity = [g for g in G.elements if g != G.e]
    for i, g in enumerate(nonidentity):
        for h in nonidentity[i + 1:]:
            kind = models.get(tuple(sorted((
                orders[g], orders[h], orders[G.mul(G.inv(g), h)]
            ))))
            if kind is None:
                continue
            cell = (G.e, g, h)
            differences = {
                G.mul(x, G.inv(y))
                for x in cell for y in cell if x != y
            }
            # Excludes cyclic triples and within-cell four-cycles.
            if len(differences) == 6:
                catalog[kind].append(cell)
    if any(not pool for pool in catalog):
        raise ValueError("empty A5 relative-triple model")
    return tuple(tuple(pool) for pool in catalog)


def _a5_packed_grid(rng, G, models):
    """Pack difference orbits; actual lifts subsequently verify girth."""
    catalog = _a5_triple_catalog(G)
    order = sorted(range(len(models)), key=lambda j: -models[j])
    for restart in range(48):
        used = set()
        row = [None] * len(models)
        for j in order:
            for attempt in range(96):
                relative = rng.choice(catalog[models[j]])
                base = G.pick(rng)
                cell = [G.mul(base, g) for g in relative]
                differences = {
                    G.mul(x, G.inv(y))
                    for x in cell for y in cell if x != y
                }
                if len(differences) != 6 or differences & used:
                    continue
                used.update(differences)
                row[j] = cell
                break
            if row[j] is None:
                break
        if all(cell is not None for cell in row):
            return [row]
    return None


def _columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _short_floor(columns):
    """Exact component-kernel minimum through four; five means unexcluded."""
    singles = set(columns)
    if 0 in singles:
        return 1
    if len(singles) != len(columns):
        return 2
    pairs = set()
    four = False
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                four = True
            pairs.add(syndrome)
    return 4 if four else 5


def _component_cycles(rows):
    counts = {}
    cycles = 0
    for row in rows:
        positions = []
        while row:
            bit = row & -row
            positions.append(bit.bit_length() - 1)
            row ^= bit
        for i, a in enumerate(positions):
            for b in positions[:i]:
                key = (b, a)
                old = counts.get(key, 0)
                cycles += old
                counts[key] = old + 1
    return cycles


def _kernel_reading(columns, rng, trials):
    """Find actual classical kernel words; high readings are not proofs."""
    best = len(columns) + 1
    order = list(range(len(columns)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
        for c in order:
            syndrome, word = columns[c], 1 << c
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
        for i, word in enumerate(words):
            for j in range(i):
                best = min(best, (word ^ words[j]).bit_count())

        # A few short basis triples supplement pairwise information sets.
        small = sorted(words, key=int.bit_count)[:10]
        for i, word in enumerate(small):
            for j in range(i):
                pair = word ^ small[j]
                for h in range(j):
                    best = min(best, (pair ^ small[h]).bit_count())
        if best <= 4:
            break
    return best


def _component_data(hx, hz, q):
    """Extract both actions from the scaffold's verified LP construction."""
    mask = (1 << q) - 1
    # HX beta=0: the A blocks are sector-one blocks 0 and 3.
    ar = [
        (row & mask) | (((row >> (3*q)) & mask) << q)
        for row in hx[:q]
    ]
    # HZ A-column=0: B occupies the first three sector-one blocks.
    bm = (1 << (3*q)) - 1
    br = [row & bm for row in hz[:q]]
    return ar, br, _columns(ar, 2*q), _columns(br, 3*q)


def _a5_no_kernel_le6(columns, q):
    """Exact constituent exclusion using regular-action anchors.

    Any kernel support can be translated so one occupied coordinate is
    coordinate zero in its protograph block. A support of size <=6
    partitions into two subsets of size <=3, one containing that anchor.
    Columns themselves come only from the verified LP construction.
    """
    width = len(columns)
    bits = [1 << i for i in range(width)]
    seen = {0: 0}

    def insert(syndrome, word):
        old = seen.get(syndrome)
        if old is not None:
            return old == word
        seen[syndrome] = word
        return True

    for i, syndrome in enumerate(columns):
        if not insert(syndrome, bits[i]):
            return False
    for i, syndrome in enumerate(columns):
        for j in range(i):
            if not insert(syndrome ^ columns[j], bits[i] | bits[j]):
                return False
    for anchor in range(0, width, q):
        for i in range(width):
            if i == anchor:
                continue
            pair_s = columns[anchor] ^ columns[i]
            pair_w = bits[anchor] | bits[i]
            for j in range(i):
                if j == anchor:
                    continue
                if not insert(pair_s ^ columns[j], pair_w | bits[j]):
                    return False

    for i in range(width):
        for j in range(i):
            syndrome = columns[i] ^ columns[j]
            for h in range(j):
                old = seen.get(syndrome ^ columns[h])
                if old is not None and old != (bits[i] | bits[j] | bits[h]):
                    return False
    return True


def _a5_basis(rows):
    basis = {}
    for row in rows:
        while row:
            pivot = row.bit_length() - 1
            old = basis.get(pivot)
            if old is None:
                basis[pivot] = row
                break
            row ^= old
    return basis


def _a5_outside(word, basis):
    while word:
        pivot = word.bit_length() - 1
        old = basis.get(pivot)
        if old is None:
            return True
        word ^= old
    return False


def _a5_logical_reading(columns, stabilizers, rng, trials, best):
    """Actual full-code logical witnesses, including both LP sectors.

    A word lowers the reading only after zero syndrome and nonmembership
    in the same-type stabilizer span. Failure to find a lighter word
    provides no positive distance certificate.
    """
    order = list(range(len(columns)))

    def consider(word):
        nonlocal best
        if word and word.bit_count() < best and _a5_outside(word, stabilizers):
            best = word.bit_count()

    for _ in range(trials):
        rng.shuffle(order)
        pivots, words = {}, []
        for j in order:
            syndrome, word = columns[j], 1 << j
            while syndrome:
                pivot = syndrome.bit_length() - 1
                old = pivots.get(pivot)
                if old is None:
                    pivots[pivot] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                words.append(word)
                consider(word)

        short = sorted(words, key=int.bit_count)[:24]
        for i, word in enumerate(short):
            for other in short[:i]:
                consider(word ^ other)
        small = short[:10]
        for i, word in enumerate(small):
            for j in range(i):
                pair = word ^ small[j]
                for h in range(j):
                    consider(pair ^ small[h])

        for seed in short[:6]:
            word = seed
            for _ in range(3):
                chosen, weight = word, word.bit_count()
                for direction in words:
                    trial = word ^ direction
                    if trial and trial.bit_count() < weight:
                        chosen, weight = trial, trial.bit_count()
                if chosen == word:
                    break
                word = chosen
                consider(word)
    return best


def _a5_check_fingerprint(rows):
    """Permutation invariant of the check-intersection graph.

    Unequal fingerprints prove nonisomorphism; equal fingerprints are
    conservatively rejected, without claiming that equality proves it.
    """
    size = len(rows)
    adjacency = [0] * size
    overlaps = [[0] * size for _ in range(size)]
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            overlaps[i][j] = overlaps[j][i] = overlap
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i

    signatures = []
    for i, row in enumerate(rows):
        histogram = {}
        for j in range(size):
            if i == j:
                continue
            common = (adjacency[i] & adjacency[j]).bit_count()
            key = (overlaps[i][j], common)
            histogram[key] = histogram.get(key, 0) + 1
        signatures.append((row.bit_count(), tuple(sorted(histogram.items()))))
    return tuple(sorted(signatures))


def _a5_fingerprint(item):
    # Canonicalize a possible global X/Z swap.
    sides = sorted((_a5_check_fingerprint(item["hx"]),
                    _a5_check_fingerprint(item["hz"])))
    return item["n"], item["k"], tuple(sides)


def _a5_merit(item):
    # Constituent readings are conservative selection heuristics; only
    # dx/dz below are explicitly checked full-code logical witnesses.
    reading = min(item["da"], item["db"], item["dx"], item["dz"])
    return (int(item["clean"]),
            item["k"] * reading * reading / item["n"],
            reading, item["da"] + item["db"], -(item["k"] - 120))


def _a5_deepen(item, rng, trials):
    item["dx"] = _a5_logical_reading(
        item["xc"], item["zs"], rng, trials, item["dx"])
    item["dz"] = _a5_logical_reading(
        item["zc"], item["xs"], rng, trials, item["dz"])


@lru_cache(maxsize=1)
def _a5_design_bank(G):
    """Deterministic independent optimization of prescribed slot cards.

    The bank couples slots only to reject invariant-equal outputs.
    No selected code is translated, permuted or automorphism-copied
    into another slot.
    """
    sa, sb = [[3, 3]], [[3, 3, 3]]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    fingerprints, bank = set(), []

    for idx in range(N_SLOTS):
        rng = random.Random(BASE_SEED * 10007 + idx)
        a_card, b_card = _A5_CARD_PAIRS[idx % len(_A5_CARD_PAIRS)]
        a_models = _A5_A_PATTERNS[a_card]
        b_models = _A5_B_PATTERNS[b_card]
        winner = None

        for batch in range(2):
            pool = []
            for attempt in range(32):
                A = _a5_packed_grid(rng, G, a_models)
                B = _a5_packed_grid(rng, G, b_models)
                if A is None or B is None:
                    continue
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (275*n <= 1000*k < 325*n):
                    continue

                ar, br, ac, bc = _component_data(hx, hz, G.q)
                # Verify the difference-packing claim on the actual actions.
                if _component_cycles(ar) or _component_cycles(br):
                    continue
                if min(_short_floor(ac), _short_floor(bc)) < 5:
                    continue
                da = _kernel_reading(ac, rng, 4)
                db = _kernel_reading(bc, rng, 4)
                item = {
                    "A": A, "B": B, "hx": hx, "hz": hz,
                    "n": n, "k": k, "ac": ac, "bc": bc,
                    "da": da, "db": db,
                    "coarse": (min(da, db), da + db, -(k - 120)),
                }
                pool.append(item)
                pool.sort(key=lambda x: x["coarse"], reverse=True)
                del pool[6:]

            finalists = pool[:4]
            for item in finalists:
                clean_a = _a5_no_kernel_le6(item["ac"], G.q)
                clean_b = _a5_no_kernel_le6(item["bc"], G.q)
                item["clean"] = clean_a and clean_b
                # All constituent columns have odd weight three.
                # With weights <=4 already excluded, a failed <=6 test
                # witnesses weight six, rather than an invented reading.
                if not clean_a:
                    item["da"] = min(item["da"], 6)
                if not clean_b:
                    item["db"] = min(item["db"], 6)
                item["da"] = min(
                    item["da"], _kernel_reading(item["ac"], rng, 24))
                item["db"] = min(
                    item["db"], _kernel_reading(item["bc"], rng, 24))
                item["xc"] = _columns(item["hx"], item["n"])
                item["zc"] = _columns(item["hz"], item["n"])
                item["xs"] = _a5_basis(item["hx"])
                item["zs"] = _a5_basis(item["hz"])
                item["dx"] = item["dz"] = item["n"] + 1
                _a5_deepen(item, rng, 12)

            finalists.sort(key=_a5_merit, reverse=True)
            for item in finalists[:2]:
                _a5_deepen(item, rng, 28)

            for item in sorted(finalists, key=_a5_merit, reverse=True):
                fingerprint = _a5_fingerprint(item)
                if fingerprint in fingerprints:
                    continue
                objective = _a5_merit(item)
                if winner is None or objective > winner[0]:
                    winner = objective, item, fingerprint
            if winner is not None and winner[0][0] and winner[0][2] >= 10:
                break

        if winner is None:
            bank.append(None)
            continue
        _, item, fingerprint = winner
        assert css_ok(item["hx"], item["hz"])
        assert 275*item["n"] <= 1000*item["k"] < 325*item["n"]
        fingerprints.add(fingerprint)
        bank.append((item["A"], item["B"]))

    if sum(item is not None for item in bank) < min(5, N_SLOTS):
        raise ValueError("fewer than five invariant-distinct A5 designs")
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return the independently optimized, invariant-distinct slot design."""
    bank = _a5_design_bank(G)
    if idx >= len(bank) or bank[idx] is None:
        raise ValueError("no fresh screened A5 filling for this card")
    A, B = bank[idx]
    return {"idx": idx, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """All labels and counts were selected jointly; do not randomize again."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Distinct monomial relative labels and short-syndrome screening on D8 at (2,7)x(2,8) would lift the distance-four plateau; all five measured d30s readings remained 4, with both types supported entirely in sector 1.
# REFUTED: High-order Dic21 binomial recipes can match the all-trinomial leaders; distinct slots 4..18 had d2s=5, whereas the measured all-trinomial leaders reached d30s=9..10.
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

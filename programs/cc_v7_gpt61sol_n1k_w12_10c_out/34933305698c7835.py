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
CONCEPT = ("PSL(2,7), order 168, shapes (1,2)x(1,3): full-group-generating trinomials selected against single-cell and multi-cell kernel witnesses should yield five selected distance readings >=14 in cell 0.300. Failure to reach that five-code target refutes the hypothesis; readings remain provisional upper bounds.")
GROUP = ("Keep the frozen projective matrix group PSL(2,7), q=168. This exact bed measured fitness 44.6084 and a best cubic reading of 14. All 48 binomial-profile slots read seven. These measurements motivate filling-level exploration on the existing bed, not a certified distance claim.")
SHAPES = ("Keep A=(1,2), B=(1,3): n=1176, k>=336, guaranteed rate 2/7, targeting cell 0.300. Components have unequal lengths 336 and 504 with 168 checks each. Parent minimum witnesses occupied sector-1 Acol/Bcol blocks, frequently Acol1 across all three B columns; none of the listed minima occupied sector 2. Exact ranks retain [0.275,0.325).")
FILLING = ("Every submitted cell is a trinomial: HX rows weigh nine and HZ rows twelve. Twenty cards cross four relative-order enrichment profiles with five cycle penalties. Relative generators span the full group and exclude involutory pair ratios. Verified monomial lifts supply all screening matrices by GF2 linearity. Prefer invertible single-cell blocks when available; otherwise rank actual single-cell kernel witnesses. Optimize A-left and B-right components independently using accumulated surviving witnesses and witness-directed whole-cell replacements, then independently deepen finalists. Exact full ranks and CSS checks select the final pair. Sampled kernel readings are not quantum-distance certificates.")
HOW_AB = FILLING
FAMILY_NAME = "r030_PSL2F7_12x13_cubic_cellkernel_witness_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 168

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("PSL(2,7) requires order 168")

    # SL(2,F7) modulo its central subgroup {I,-I}.
    # Elements encode a matrix (a,b;c,d), choosing min(M,-M).
    def red(g):
        v = tuple(int(x) % 7 for x in g)
        return min(v, tuple((-x) % 7 for x in v))

    els = []
    for a in range(7):
        for b in range(7):
            for c in range(7):
                for d in range(7):
                    g = (a, b, c, d)
                    if (a*d - b*c) % 7 == 1 and red(g) == g:
                        els.append(g)
    assert len(els) == q
    pos = {g: i for i, g in enumerate(els)}

    # Cache the exact quotient law once. Subsequent verified lifts
    # still use the scaffold, not an independently implemented lift.
    table = []
    for a, b, c, d in els:
        table.append([
            red((a*e + b*h, a*f + b*k,
                 c*e + d*h, c*f + d*k))
            for e, f, h, k in els
        ])
    inverses = {
        g: red((g[3], -g[1], -g[2], g[0])) for g in els
    }

    def mul(g, h):
        return table[pos[g]][pos[h]]

    def inv(g):
        return inverses[g]

    def idx(g):
        return pos[red(g)]

    return dict(elements=els, ident=(1, 0, 0, 1),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


# Five density allocations, crossed with four within-cell order spectra.
# Maximum HX/HZ weights: (9,12), (11,12), (10,12), (10,12), (9,11).
_PSL_COUNTS = (
    ((3, 3), (3, 3, 3)),
    ((4, 4), (2, 3, 3)),
    ((3, 4), (2, 3, 3)),
    ((3, 3), (2, 3, 4)),
    ((3, 3), (2, 3, 3)),
)


def _psl_bits(value):
    while value:
        bit = value & -value
        yield bit.bit_length() - 1
        value ^= bit


def _psl_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        for c in _psl_bits(row):
            columns[c] |= 1 << r
    return columns


def _psl_no_kernel4(columns):
    """Exact exclusion of nonzero kernel vectors of weights one to four."""
    singles = set(columns)
    if 0 in singles or len(singles) != len(columns):
        return False
    seen = singles.copy()
    for i, a in enumerate(columns):
        for j in range(i):
            value = a ^ columns[j]
            if value in seen:
                return False
            seen.add(value)
    return True


def _psl_components(hx, hz, q):
    """Project components from the scaffold's actual LP column ordering."""
    mask = (1 << q) - 1
    ar = []
    for t in range(q):
        source = hx[t]  # Arow 0, Bcol 0.
        ar.append((source & mask)
                  | (((source >> (3*q)) & mask) << q))
    bmask = (1 << (3*q)) - 1
    br = [hz[t] & bmask for t in range(q)]  # Acol 0, Brow 0.
    return ar, br


def _psl_quality(rows, width):
    """Actual component C4 count and a projection-triangle heuristic."""
    columns = _psl_columns(rows, width)
    if 0 in columns or len(set(columns)) != width:
        return None
    adjacency = [0] * len(rows)
    uses = {}
    four = forced = 0
    for signature in columns:
        ns = list(_psl_bits(signature))
        degree = len(ns)
        forced += degree * (degree - 1) * (degree - 2) // 6
        for i, u in enumerate(ns):
            adjacency[u] |= signature ^ (1 << u)
            for v in ns[:i]:
                key = v * len(rows) + u
                previous = uses.get(key, 0)
                four += previous
                uses[key] = previous + 1

    reached = frontier = 1
    while frontier:
        nxt = 0
        for u in _psl_bits(frontier):
            nxt |= adjacency[u]
        frontier = nxt & ~reached
        reached |= frontier
    if reached != (1 << len(rows)) - 1:
        return None

    triangles = 0
    for u, neighbors in enumerate(adjacency):
        for v in _psl_bits(neighbors & (-1 << (u + 1))):
            triangles += (neighbors & adjacency[v]
                          & (-1 << (v + 1))).bit_count()
    return (four, max(0, triangles - forced)), columns


def _psl_circuit_reading(rng, columns, trials):
    """Find actual classical kernel words; readings are upper bounds only."""
    reading = len(columns) + 1
    order = list(range(len(columns)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        basis = []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                lead = syndrome.bit_length() - 1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                basis.append(word)
                reading = min(reading, word.bit_count())
        light = sorted(basis, key=int.bit_count)[:20]
        for a in light:
            for b in basis:
                if a != b:
                    reading = min(reading, (a ^ b).bit_count())
        if len(light) >= 3:
            for _ in range(80):
                a, b, c = rng.sample(light, 3)
                reading = min(reading, (a ^ b ^ c).bit_count())
        if reading <= 6:
            break
    return reading


def NEW_CTX(rng, G, idx):
    """Cubic cells, verified single-cell kernels, and witness-directed search."""
    from itertools import combinations

    q = G.q
    mask = (1 << q) - 1
    card, cohort = idx % 20, idx // 20
    model, objective = card // 5, card % 5
    cycle_penalty = (200, 500, 1000, 2000, 4000)[objective]
    diagnostic_seed = rng.randrange(1 << 62)
    assert _grid_weights([[3, 3]], [[3, 3, 3]]) == 12
    assert W_MAX >= 12

    # These are screening-only dummy protographs. Every monomial block
    # comes from the verified constructor, including its left/right
    # conventions. Subsequent XORs and disjoint column concatenations
    # use only GF2 linearity; no group-action lift is implemented here.
    hx, _, _ = lifted_matrices(
        G, [[[g] for g in G.elements]], [[[]]])
    _, hz, _ = lifted_matrices(
        G, [[[]]], [[[g] for g in G.elements]])
    left = [
        [(row >> (j*q)) & mask for row in hx]
        for j in range(q)
    ]
    right = [
        [(row >> (j*q)) & mask for row in hz]
        for j in range(q)
    ]
    del hx, hz

    orders = {}
    for g in G.elements:
        value = G.e
        for order in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                orders[g] = order
                break
        else:
            raise ValueError("invalid projective group law")

    def generates_full(g, h):
        reached = {G.e}
        frontier = [G.e]
        for x in frontier:
            for y in (g, h):
                z = G.mul(x, y)
                if z not in reached:
                    reached.add(z)
                    frontier.append(z)
        return len(reached) == q

    def cell_signature(cell):
        # Remove within-cell ordering and normalized column-gauge copies.
        # This does not claim to remove every Tanner-graph isomorphism.
        return min(
            tuple(sorted(G.idx(G.mul(G.inv(root), g)) for g in cell))
            for root in cell
        )

    def single_geometry(rows):
        """Actual single-block kernel reading and exact rank.

        Enumerate the entire kernel when its dimension is <=12.
        Larger kernels receive basis/pair searches only. A full-rank
        block is genuinely invertible; other readings are upper bounds.
        """
        columns = _psl_columns(rows, q)
        pivots, basis = {}, []
        for c, value in enumerate(columns):
            word = 1 << c
            while value:
                p = value.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (value, word)
                    break
                value ^= old[0]
                word ^= old[1]
            if not value:
                basis.append(word)

        if not basis:
            return q + 1, q
        reading = min(word.bit_count() for word in basis)
        if len(basis) <= 12:
            word = 0
            for t in range(1, 1 << len(basis)):
                word ^= basis[(t & -t).bit_length() - 1]
                reading = min(reading, word.bit_count())
        else:
            for i, word in enumerate(basis):
                for other in basis[:i]:
                    reading = min(reading, (word ^ other).bit_count())
        return reading, len(pivots)

    targets = ((4, 4, 7), (4, 7, 7), (3, 4, 7), None)
    target = targets[(model + cohort) % 4]
    nonidentity = [g for g in G.elements if g != G.e]
    library, signatures = [], set()

    # Enrich, rather than hard-gate, the requested spectrum. General
    # full-generating triples provide a fallback and additional geometry.
    for attempt in range(24000):
        if len(library) >= 96:
            break
        g, h = rng.sample(nonidentity, 2)
        spectrum = tuple(sorted((
            orders[g], orders[h], orders[G.mul(G.inv(g), h)]
        )))
        if 2 in spectrum:
            continue
        error = (0 if target is None else
                 sum(abs(a-b) for a, b in zip(spectrum, target)))
        if len(library) < 64 and attempt < 12000 and error:
            continue
        cell = (G.e, g, h)
        signature = cell_signature(cell)
        if signature in signatures or not generates_full(g, h):
            continue
        signatures.add(signature)
        ids = [G.idx(x) for x in cell]
        ar = [left[ids[0]][r] ^ left[ids[1]][r] ^ left[ids[2]][r]
              for r in range(q)]
        br = [right[ids[0]][r] ^ right[ids[1]][r] ^ right[ids[2]][r]
              for r in range(q)]
        da, ra = single_geometry(ar)
        db, rb = single_geometry(br)
        library.append({
            "cell": cell, "A": ar, "B": br,
            "floorA": da, "floorB": db,
            "rankA": ra, "rankB": rb,
            "error": error, "spectrum": spectrum
        })

    if len(library) < 8:
        raise ValueError("insufficient full-generating cubic cell geometry")

    def syndrome(columns, word):
        value = 0
        while word:
            bit = word & -word
            value ^= columns[bit.bit_length()-1]
            word ^= bit
        return value

    def kernel_search(columns, seed, trials):
        """Genuine component words from independent systematic bases."""
        local = random.Random(seed)
        width = len(columns)
        order = list(range(width))
        reading, witness = width + 1, None
        for _ in range(trials):
            local.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    p = value.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    weight = word.bit_count()
                    if weight < reading:
                        reading, witness = weight, word

            light = sorted(basis, key=int.bit_count)[:16]
            for a in light:
                for b in basis:
                    word = a ^ b
                    if word:
                        weight = word.bit_count()
                        if weight < reading:
                            reading, witness = weight, word
            for a, b, c in combinations(light[:8], 3):
                word = a ^ b ^ c
                if word:
                    weight = word.bit_count()
                    if weight < reading:
                        reading, witness = weight, word
            if reading <= 10:
                break
        if witness is None or syndrome(columns, witness):
            raise ValueError("component search failed kernel verification")
        return reading, witness

    def optimize_component(side, count):
        floor_key, rank_key = "floor"+side, "rank"+side
        eligible = sorted(
            range(len(library)),
            key=lambda j: (
                library[j][floor_key], library[j][rank_key],
                -library[j]["error"]
            ), reverse=True
        )[:36]
        width = count*q
        seed = diagnostic_seed + (0 if side == "A" else 1000003)
        bank = set()
        pool = {}

        def assembled(ids):
            return [
                sum(library[t][side][r] << (j*q)
                    for j, t in enumerate(ids))
                for r in range(q)
            ]

        def rating(record):
            reading = record["reading"]
            for word in bank:
                if word.bit_count() < reading:
                    if syndrome(record["columns"], word) == 0:
                        reading = word.bit_count()
            return (reading, -record["cost"], -record["error"])

        def remember_word(word):
            if word is not None:
                bank.add(word)
                if len(bank) > 64:
                    keep = sorted(bank, key=int.bit_count)[:48]
                    bank.clear()
                    bank.update(keep)

        def probe(ids, trials=6):
            rows = assembled(ids)
            quality = _psl_quality(rows, width)
            if quality is None:
                return None
            stats, columns = quality
            single = min(library[t][floor_key] for t in ids)
            reading, witness = kernel_search(columns, seed, trials)
            remember_word(witness)
            return {
                "ids": tuple(ids), "columns": columns,
                "reading": min(single, reading), "word": witness,
                "cost": cycle_penalty*stats[0] + stats[1],
                "stats": stats,
                "error": sum(library[t]["error"] for t in ids)
            }

        def retain(record):
            if record is None:
                return
            key = record["ids"]
            old = pool.get(key)
            if old is None or record["reading"] < old["reading"]:
                pool[key] = record
            if len(pool) > 8:
                worst = min(pool, key=lambda k: rating(pool[k]))
                del pool[worst]

        def deepen(record, trials, offset):
            reading, witness = kernel_search(
                record["columns"], seed + offset, trials)
            record["reading"] = min(record["reading"], reading)
            if witness.bit_count() <= record["word"].bit_count():
                record["word"] = witness
            remember_word(witness)

        for _ in range(14):
            retain(probe(rng.sample(eligible, count)))
        if not pool:
            raise ValueError("no connected cubic component")

        initial = sorted(pool.values(), key=rating, reverse=True)[:4]
        for record in initial:
            deepen(record, 40, 17)
        current = max(initial, key=rating)

        # Replace an entire relative-label geometry, preserving cubic
        # supports and full generation. Prefer blocks actually used by
        # a surviving short word, rather than arbitrary matrix entries.
        for step in range(40):
            surviving = [
                w for w in bank
                if syndrome(current["columns"], w) == 0
            ]
            witness = (min(surviving, key=int.bit_count)
                       if surviving else current["word"])
            active = sorted({c//q for c in _psl_bits(witness)})
            ids = list(current["ids"])
            j = (rng.choice(active) if active and rng.random() < 0.85
                 else rng.randrange(count))
            choices = [t for t in eligible if t not in ids]
            if not choices:
                continue
            ids[j] = rng.choice(choices)
            proposed = probe(ids, 6)
            if proposed is None:
                continue
            pk, ck = rating(proposed), rating(current)
            accept = pk > ck
            if not accept and pk[0] == ck[0]:
                accept = (
                    proposed["cost"] <= current["cost"] + cycle_penalty
                    and rng.random() < 0.08
                )
            retain(proposed)
            if accept:
                current = proposed

        finalists = sorted(pool.values(), key=rating, reverse=True)[:3]
        # Only independently deepened records may leave this optimizer.
        for record in finalists:
            deepen(record, 160 + 16*objective, 1009)
            deepen(record, 64, 100003)
        finalists.sort(key=rating, reverse=True)
        for record in finalists:
            record["effective"] = rating(record)[0]
            record["cells"] = [[
                list(library[t]["cell"]) for t in record["ids"]
            ]]
        return finalists

    afinal = optimize_component("A", 2)
    bfinal = optimize_component("B", 3)
    proposals = []
    for a in afinal:
        for b in bfinal:
            key = (
                min(a["effective"], b["effective"]),
                b["effective"], a["effective"] + b["effective"],
                -(a["cost"] + b["cost"])
            )
            proposals.append((key, a, b))
    proposals.sort(key=lambda item: item[0], reverse=True)

    for quality, a, b in proposals:
        A, B = a["cells"], b["cells"]
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 1176
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 275*n <= 1000*k < 325*n):
            continue
        # Check that cached linear screening blocks are exactly the
        # components in the final authoritative construction.
        ar, br = _psl_components(hx, hz, q)
        assert ar == [
            sum(library[t]["A"][r] << (j*q)
                for j, t in enumerate(a["ids"])) for r in range(q)
        ]
        assert br == [
            sum(library[t]["B"][r] << (j*q)
                for j, t in enumerate(b["ids"])) for r in range(q)
        ]
        assert max(row.bit_count() for row in hx + hz) <= 12
        assert css_ok(hx, hz)
        return {
            "idx": idx, "A": A, "B": B, "k": k,
            "card": (model, objective, cohort),
            "component_readings": (a["effective"], b["effective"]),
            "cycle_statistics": (a["stats"], b["stats"]),
            "single_cell_readings": (
                [library[t]["floorA"] for t in a["ids"]],
                [library[t]["floorB"] for t in b["ids"]]
            ),
            "quality": quality
        }
    raise ValueError("no cubic witness-screened pair in target rate band")


def CELL(rng, G, side, i, j, ctx):
    """Return the entire selected cell without subsequent randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Saturating the row-weight cap suffices to eliminate sub-12 sector-1 logicals; parent D298 slots 4 and 5 had w=12 but measured distance eight, with X minima entirely in sector 1.
# REFUTED: Order-seven binomial density cards improve the cubic PSL(2,7) bed: all 48 binomial-profile parent slots read d2s=7, while every measured top-five slot used five trinomials and read 12 or 14.
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

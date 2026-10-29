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
CONCEPT = ("On C67 semidirect C3 with action 29, order 201, A=(1,2), B=(1,3), cubic fillings selected against verified cyclic-module cofactor kernels and translated cancellations should yield five measured distance readings >=16 in cell 0.300; these readings remain provisional upper bounds.")
GROUP = ("Keep the frozen C67:C3 group and order 201. This bed leads cell 0.300 at fitness 47.6996, versus 44.6084 on PSL(2,7) and 41.1430 on Dic53 with the same shapes. Its normal C67 subgroup permits efficient cyclic-module witness generation, but neither the algebra nor these measurements certifies distance.")
SHAPES = ("Keep A=(1,2), B=(1,3): n=1407, k>=402, guaranteed rate 2/7=0.285714. Components have 402/201 and 603/201 variables/checks. Parent X minima of 12 or 14 were entirely sector 1, predominantly Acol1 across all B columns; Z minima were 18..22. Prioritize the longer B component while retaining both kernels and an exact full-rank target-cell gate.")
FILLING = ("Twenty cards cross four B balanced/repeated-coset profiles with five A geometry/selection prescriptions. Every cell has three distinct terms: HX=9 and HZ=12. Extract cyclic-module coefficients from verified component matrices, generate three-row cofactor kernels, and search translated pair cancellations. Verify retained words in the original component rows and independently refine randomized kernel readings. Embed short component witnesses in the actual quantum matrices and exclude original stabilizer spans. Fresh labels, exact ranks and CSS checks gate submission; these screens are not distance certificates.")
HOW_AB = FILLING
FAMILY_NAME = "r030_frobenius67x3_12x13_cubic_module_cancellations_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 201

def custom_ops(q):
    # Frobenius C_67 semidirect C_3:
    # y*x*y^-1 = x^29, with 29^2=37 and 29^3=1 mod 67.
    if not valid_q(q):
        raise ValueError("this Frobenius group requires order 201")
    action = (1, 29, 37)
    els = [(a, b) for b in range(3) for a in range(67)]

    def red(g):
        return (g[0] % 67, g[1] % 3)

    def mul(g, h):
        return ((g[0] + action[g[1]] * h[0]) % 67,
                (g[1] + h[1]) % 3)

    def inv(g):
        bi = (-g[1]) % 3
        return ((-action[bi] * g[0]) % 67, bi)

    def idx(g):
        a, b = red(g)
        return 67 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


# Four genuinely different term-count allocations, each with five
# prescribed coset models. Only B has a binomial, never two binomials
# that could both normalize into commuting normal-subgroup polynomials.
_DENSITY_CARDS = (
    ((3, 3), (3, 3, 3)),
    ((3, 4), (2, 3, 3)),
    ((4, 4), (2, 3, 3)),
    ((3, 3), (2, 3, 4)),
)


def _draw_frobenius(rng, counts, side, model, cohort):
    """Independent labels, mixed larger cells and long-order binomials."""
    cells = []
    for j, count in enumerate(counts):
        if count == 2:
            occupancy = (2, 0, 0)
        else:
            if model == 0:
                kind = 0
            elif model == 1:
                kind = 1
            elif model == 2:
                kind = int(side == "A")
            elif model == 3:
                kind = int(side == "B")
            else:
                kind = 2
            patterns = (
                {3: (1, 1, 1), 4: (2, 1, 1)},
                {3: (2, 1, 0), 4: (2, 2, 0)},
                {3: (2, 1, 0), 4: (3, 1, 0)},
            )
            occupancy = patterns[kind][count]

        # Coset shifts and orientations are independently selected per
        # cell; final exponent geometry is independently optimized too.
        shift = rng.randrange(3)
        orientation = (1, 2)[rng.randrange(2)]
        cell = []
        for t, size in enumerate(occupancy):
            if not size:
                continue
            b = (shift + orientation * t) % 3
            for _ in range(1000):
                exponents = rng.sample(range(67), size)
                differences = [
                    (a - c) % 67 for a in exponents
                    for c in exponents if a != c
                ]
                spacing = 1 + 2 * (cohort % 3)
                if (len(set(differences)) == len(differences)
                        and all(min(d, 67 - d) >= spacing
                                for d in differences)):
                    break
            else:
                raise ValueError("Sidon exponent construction failed")
            cell.extend((a, b) for a in exponents)
        assert len(cell) == count and len(set(cell)) == count
        if count >= 3:
            assert len({b for a, b in cell}) >= 2
        cells.append(cell)
    return [cells]


def _columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _kernel4_free(columns):
    """Exact component exclusion through weight four."""
    singles = set(columns)
    if 0 in singles or len(singles) != len(columns):
        return False
    pairs = set()
    for i, a in enumerate(columns):
        for b in columns[:i]:
            syndrome = a ^ b
            if syndrome in singles or syndrome in pairs:
                return False
            pairs.add(syndrome)
    return True


def _component_rows(hx, hz, q, side):
    """Extract the actual left/right components from verified matrices."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    if side == "B":
        mask = (1 << (cb * q)) - 1
        return [hz[a * q + r] & mask
                for a in range(rb) for r in range(q)]
    mask = (1 << q) - 1
    rows = []
    for i in range(ra):
        for r in range(q):
            source = hx[(i * cb) * q + r]
            row = 0
            for j in range(ca):
                row |= ((source >> (j * cb * q)) & mask) << (j * q)
            rows.append(row)
    return rows


def _cycle_cost(rows, penalty):
    """Count component four/six cycles, excluding reused variable nodes."""
    h = len(rows)
    common = [[0] * h for _ in range(h)]
    upper = [[] for _ in range(h)]
    c4 = 0
    for a in range(h):
        for b in range(a + 1, h):
            overlap = (rows[a] & rows[b]).bit_count()
            if overlap:
                common[a][b] = overlap
                upper[a].append(b)
                c4 += overlap * (overlap - 1) // 2
    c6 = 0
    for a in range(h):
        ns = upper[a]
        for t, b in enumerate(ns):
            ab = common[a][b]
            for c in ns[t + 1:]:
                bc = common[b][c]
                if bc:
                    ac = common[a][c]
                    triple = (rows[a] & rows[b] & rows[c]).bit_count()
                    c6 += ab * ac * bc - triple * (ab + ac + bc - 2)
    return penalty * c4 + c6


def _connected_component(rows, columns):
    """Connectedness of the actual component Tanner graph."""
    adjacency = [0] * len(rows)
    for syndrome in columns:
        word = syndrome
        while word:
            bit = word & -word
            r = bit.bit_length() - 1
            adjacency[r] |= syndrome ^ bit
            word ^= bit
    visited = frontier = 1
    while frontier:
        reached = 0
        word = frontier
        while word:
            bit = word & -word
            reached |= adjacency[bit.bit_length() - 1]
            word ^= bit
        frontier = reached & ~visited
        visited |= frontier
    return visited == (1 << len(rows)) - 1


def _span_pivots(rows):
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old
    return pivots


def _span_signature(word, pivots):
    """Linear canonical remainder, including ALL missing-pivot bits."""
    remainder = 0
    while word:
        p = word.bit_length() - 1
        old = pivots.get(p)
        if old is None:
            bit = 1 << p
            remainder |= bit
            word ^= bit
        else:
            word ^= old
    return remainder


def _kernel_probe(rng, columns, trials, signatures=None, stabilizers=()):
    """Sample verified kernel circuits and cancellations.

    With signatures, zero signature means membership in the ORIGINAL
    stabilizer span. Descents use only original stabilizer rows.
    The returned minimum is a sampled upper reading, never a certificate.
    """
    width = len(columns)
    best = width + 1
    best_word = None
    order = list(range(width))
    descent = list(stabilizers)

    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        circuits = []
        for c in order:
            syndrome = columns[c]
            word = 1 << c
            tag = signatures[c] if signatures is not None else word
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word, tag)
                    break
                syndrome ^= old[0]
                word ^= old[1]
                tag ^= old[2]
            if not syndrome:
                circuits.append((word, tag))
                if tag and word.bit_count() < best:
                    best, best_word = word.bit_count(), word

        circuits.sort(key=lambda item: item[0].bit_count())
        light = circuits[:12]
        if signatures is not None:
            light += [item for item in circuits if item[1]][:4]
        for a, ta in light:
            for b, tb in circuits:
                if ta ^ tb:
                    candidate = a ^ b
                    weight = candidate.bit_count()
                    if weight < best:
                        best, best_word = weight, candidate

        if descent:
            rng.shuffle(descent)
            seeds = [word for word, tag in circuits if tag][:3]
            if best_word is not None:
                seeds.append(best_word)
            for word in seeds:
                for sweep in range(2):
                    changed = False
                    weight = word.bit_count()
                    for row in descent:
                        candidate = word ^ row
                        new_weight = candidate.bit_count()
                        if new_weight < weight:
                            word, weight = candidate, new_weight
                            changed = True
                    if not changed:
                        break
                if word.bit_count() < best:
                    best, best_word = word.bit_count(), word

    return best, best_word


def _component_pool(rng, G, counts, side, model, cohort):
    """Independent actual-lift component geometry and kernel screening."""
    unit = [[[G.e]]]
    width = len(counts) * G.q
    mask = (1 << width) - 1
    records = []

    for trial in range(28):
        grid = _draw_frobenius(rng, counts, side, model, cohort)
        if side == "A":
            hx, _, _ = lifted_matrices(G, grid, unit)
            rows = [row & mask for row in hx]
        else:
            _, hz, _ = lifted_matrices(G, unit, grid)
            rows = [row & mask for row in hz]
        columns = _columns(rows, width)
        if (0 in columns or len(set(columns)) != width
                or not _connected_component(rows, columns)):
            continue
        rank = gf2_rank(rows)
        if rank < len(rows) - 16:
            continue
        cost = _cycle_cost(rows, 160)
        records.append({"grid": grid, "rows": rows, "cols": columns,
                        "rank": rank, "cost": cost, "trial": trial})

    records.sort(key=lambda r: (r["cost"], -r["rank"], r["trial"]))
    screened = []
    for record in records:
        if not _kernel4_free(record["cols"]):
            continue
        record["d"] = _kernel_probe(rng, record["cols"], 8)[0]
        screened.append(record)
        if len(screened) == 5:
            break
    screened.sort(key=lambda r: (r["d"], -r["cost"], r["rank"]),
                  reverse=True)

    finalists = screened[:2]
    for record in finalists:
        # Fresh elimination orders independently check an optimistic
        # preliminary component reading before pairing the components.
        record["d"] = min(record["d"],
                          _kernel_probe(rng, record["cols"], 24)[0])
    finalists.sort(key=lambda r: (r["d"], -r["cost"]), reverse=True)
    return finalists


def _quantum_probe(rng, hx, hz, n, trials):
    """Actual logical upper readings, not projected component homology."""
    reading = n + 1
    witnesses = []
    for opposite, stabilizers in ((hz, hx), (hx, hz)):
        columns = _columns(opposite, n)
        pivots = _span_pivots(stabilizers)
        signatures = [_span_signature(1 << c, pivots) for c in range(n)]
        d, word = _kernel_probe(rng, columns, trials,
                                signatures, stabilizers)
        if word is not None:
            # Check every retained witness in the original matrices.
            assert all((word & row).bit_count() % 2 == 0
                       for row in opposite)
            assert _span_signature(word, pivots) != 0
            witnesses.append(word.bit_count())
            reading = min(reading, d)
    return reading, tuple(witnesses)


def NEW_CTX(rng, G, idx):
    """Select cubic fillings using explicit, verified module cancellations."""
    from itertools import combinations, permutations

    card, cohort = idx % 20, idx // 20
    model, balanced_b = card % 5, card // 5
    m, q = 67, G.q
    mask = (1 << m) - 1
    actions = [
        G.mul((0, b), (1, 0))[0] for b in range(3)
    ]
    sa, sb = [[3, 3]], [[3, 3, 3]]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    # T spans all three quotient cosets; D has occupancy (2,1,0).
    # The fifth prescription changes both gap geometry and the objective.
    a_patterns = (
        ("T", "T"), ("T", "D"),
        ("D", "D"), ("D", "D"), ("D", "D")
    )[model]
    b_patterns = ("T",) * balanced_b + ("D",) * (3 - balanced_b)

    def fresh(patterns, side):
        cells = []
        spacing = (1, 2, 4, 6, 9)[model] + cohort % 3
        for j, kind in enumerate(patterns):
            shift = rng.randrange(3)
            if kind == "T":
                cell = [(rng.randrange(m), b) for b in range(3)]
            else:
                if model == 2:
                    direction = 1
                elif model == 3:
                    direction = 1 if j % 2 == 0 else 2
                else:
                    direction = rng.choice((1, 2))
                for _ in range(256):
                    a, c = rng.sample(range(m), 2)
                    gap = (a - c) % m
                    if min(gap, m - gap) >= spacing:
                        break
                else:
                    raise ValueError("could not fill cubic gap prescription")
                cell = [(a, shift), (c, shift),
                        (rng.randrange(m), (shift + direction) % 3)]
            assert len(cell) == len(set(cell)) == 3
            cells.append(cell)
        return [cells]

    def multiply(a, b):
        """Multiplication in F2[z]/(z^67-1), used only for witnesses."""
        if a.bit_count() > b.bit_count():
            a, b = b, a
        answer = 0
        while a:
            bit = a & -a
            shift = bit.bit_length() - 1
            answer ^= ((b << shift) | (b >> (m - shift))) & mask
            a ^= bit
        return answer

    def physical(word, side, channels):
        """Undo a coordinate permutation; this does not construct a lift."""
        answer = 0
        for channel in range(channels):
            value = (word >> (channel * m)) & mask
            factor = actions[channel % 3] if side == "A" else 1
            while value:
                bit = value & -value
                a = bit.bit_length() - 1
                answer |= 1 << (channel * m + factor * a % m)
                value ^= bit
        return answer

    def cofactor_words(rows, columns, side, channels):
        """Read the cyclic module from VERIFIED syndrome columns.

        A is invariant under normal-subgroup right translations, requiring
        coset-dependent coordinate rescaling. B uses left translations.
        All retained witnesses are checked against the original bit rows.
        """
        polynomial = [[0] * channels for _ in range(3)]
        for channel in range(channels):
            syndrome = columns[channel * m]
            for b in range(3):
                factor = actions[b] if side == "A" else 1
                value = 0
                for t in range(m):
                    if syndrome & (1 << (b * m + factor * t % m)):
                        value |= 1 << t
                polynomial[b][channel] = value

        minors = {}
        perms = tuple(permutations(range(3)))
        for triple in combinations(range(channels), 3):
            value = 0
            for p in perms:
                value ^= multiply(
                    multiply(polynomial[0][triple[p[0]]],
                             polynomial[1][triple[p[1]]]),
                    polynomial[2][triple[p[2]]])
            minors[triple] = value

        words = set()
        for quartet in combinations(range(channels), 4):
            word = 0
            for omitted in quartet:
                triple = tuple(c for c in quartet if c != omitted)
                word |= minors[triple] << (omitted * m)
            if word:
                words.add(word)
        words = sorted(words, key=lambda w: (w.bit_count(), w))
        if words:
            witness = physical(words[0], side, channels)
            assert all((witness & row).bit_count() % 2 == 0
                       for row in rows)
        return words

    def rotate(word, shift, channels):
        answer = 0
        for channel in range(channels):
            value = (word >> (channel * m)) & mask
            value = ((value << shift) | (value >> (m - shift))) & mask
            answer |= value << (channel * m)
        return answer

    def cancellation_reading(record, light_count):
        """Search actual cofactor words and translated pair cancellations."""
        seeds = record["seeds"]
        if not seeds:
            return record["width"] + 1, None
        best_word = seeds[0]
        best = best_word.bit_count()
        light = seeds[:light_count]
        for other in seeds:
            for shift in range(m):
                translated = rotate(other, shift, record["channels"])
                for word in light:
                    candidate = word ^ translated
                    if candidate:
                        weight = candidate.bit_count()
                        if weight < best:
                            best, best_word = weight, candidate
        witness = physical(best_word, record["side"], record["channels"])
        assert witness and witness.bit_count() == best
        assert all((witness & row).bit_count() % 2 == 0
                   for row in record["rows"])
        return best, witness

    def refine(record, trials, light_count):
        d, word = cancellation_reading(record, light_count)
        if word is not None and d < record["d"]:
            record["d"], record["word"] = d, word
        d, word = _kernel_probe(rng, record["cols"], trials)
        if word is not None:
            assert all((word & row).bit_count() % 2 == 0
                       for row in record["rows"])
            if d < record["d"]:
                record["d"], record["word"] = d, word

    def component_pool(side, patterns):
        unit = [[[G.e]]]
        channels = 3 * len(patterns)
        width = channels * m
        records = []
        # More independent B starts address the measured X bottleneck.
        starts = 28 if side == "A" else 64
        for trial in range(starts):
            grid = fresh(patterns, side)
            if side == "A":
                hx, _, _ = lifted_matrices(G, grid, unit)
                rows = [row & ((1 << width) - 1) for row in hx]
            else:
                _, hz, _ = lifted_matrices(G, unit, grid)
                rows = [row & ((1 << width) - 1) for row in hz]
            columns = _columns(rows, width)
            if not _connected_component(rows, columns):
                continue
            rank = gf2_rank(rows)
            if rank < q - 16:
                continue
            seeds = cofactor_words(rows, columns, side, channels)
            floor = seeds[0].bit_count() if seeds else width + 1
            multiplicity = sum(w.bit_count() == floor for w in seeds)
            records.append({
                "grid": grid, "rows": rows, "cols": columns,
                "rank": rank, "seeds": seeds, "side": side,
                "channels": channels, "width": width,
                "raw": floor, "multiplicity": multiplicity,
                "cost": _cycle_cost(rows, 160), "trial": trial,
                "d": width + 1, "word": None
            })

        records.sort(
            key=lambda r: (r["raw"], -r["multiplicity"],
                           -r["cost"], r["rank"]), reverse=True)
        screened = []
        for record in records:
            if not _kernel4_free(record["cols"]):
                continue
            refine(record, 8, 8)
            screened.append(record)
            if len(screened) == 8:
                break
        screened.sort(
            key=lambda r: (r["d"], -r["cost"], r["rank"]), reverse=True)
        finalists = screened[:4]
        for record in finalists:
            # Independent orders accumulate, never replace, upper readings.
            refine(record, 32, 16)
        finalists.sort(
            key=lambda r: (r["d"], -r["cost"]), reverse=True)
        return finalists[:3]

    def embedded_readings(a, b, hx, hz, n):
        """Test component witnesses in ORIGINAL quantum stabilizer spans."""
        reading = n + 1
        witnesses = []
        for side, record, opposite, stabilizers in (
                ("A", a, hx, hz), ("B", b, hz, hx)):
            source = record["word"]
            if source is None:
                continue
            pivots = _span_pivots(stabilizers)
            placements = range(3) if side == "A" else range(2)
            for place in placements:
                word = 0
                remaining = source
                while remaining:
                    bit = remaining & -remaining
                    c = bit.bit_length() - 1
                    block, g = divmod(c, q)
                    target = ((block * 3 + place) * q + g
                              if side == "A"
                              else (place * 3 + block) * q + g)
                    word |= 1 << target
                    remaining ^= bit
                assert all((word & row).bit_count() % 2 == 0
                           for row in opposite)
                if _span_signature(word, pivots):
                    witnesses.append(word.bit_count())
                    reading = min(reading, word.bit_count())
        return reading, tuple(witnesses)

    for attempt in range(2):
        aa = component_pool("A", a_patterns)
        bb = component_pool("B", b_patterns)
        pairs = []
        for a in aa:
            for b in bb:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (n == 1407 and k >= 402
                        and 11 * n <= 40 * k < 13 * n):
                    continue
                d, witnesses = embedded_readings(a, b, hx, hz, n)
                if model == 4:
                    merit = (min(a["d"], b["d"], d), b["d"], a["d"],
                             -b["cost"], -a["cost"])
                else:
                    merit = (min(a["d"], b["d"], d),
                             a["d"] + b["d"], b["d"],
                             -a["cost"] - b["cost"])
                pairs.append({
                    "A": A, "B": B, "hx": hx, "hz": hz,
                    "n": n, "k": k, "merit": merit,
                    "component_d": (a["d"], b["d"]),
                    "logical_d": d, "witnesses": witnesses
                })
        pairs.sort(key=lambda r: r["merit"], reverse=True)
        finalists = pairs[:3]
        if not finalists:
            continue

        for record in finalists:
            # Also search mixed-sector circuits, using original spans.
            d, witnesses = _quantum_probe(
                rng, record["hx"], record["hz"], record["n"], 4)
            record["logical_d"] = min(record["logical_d"], d)
            record["witnesses"] += witnesses

        chosen = max(
            finalists,
            key=lambda r: (
                min(*r["component_d"], r["logical_d"]),
                r["logical_d"], r["merit"], -(r["k"] - 402)))
        hx, hz = chosen["hx"], chosen["hz"]
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) == 12
        return {
            "idx": idx, "card": (model, balanced_b, cohort),
            "cells": {"A": chosen["A"], "B": chosen["B"]},
            "screen_k": chosen["k"],
            "component_readings": chosen["component_d"],
            "logical_reading": chosen["logical_d"],
            "verified_witness_weights": chosen["witnesses"]
        }
    raise ValueError("no cubic module-screened pair passed target-cell ranks")


def CELL(rng, G, side, i, j, ctx):
    """Return the independently screened full cell without new randomness."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolling and uniform labels alone suppress very short logicals in the order-52 (2,6)x(2,4) bed: all five measured parent codes had distance 4, with many sector-1 witnesses.
# REFUTED: Trading a B trinomial for a binomial to increase A density improves the C67:C3 (1,2)x(1,3) bottleneck: every such parent slot had d2s<=10, while all five measured all-cubic slots retained readings 12 or 14.
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

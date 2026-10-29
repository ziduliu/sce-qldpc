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
CONCEPT = ("F25-additive semidirect C6, order 150, A=(1,2), B=(2,3): primitive-multiplier A supports and independently retested component witnesses should produce five measured readings above 40 in cell 0.150. Failure refutes this selection hypothesis; readings remain provisional upper bounds.")
GROUP = ("Jump from C37:C3, order 111, to the irreducible affine group F25-additive:C6, order 150. With alpha^2=alpha-1, alpha^3=-1 and alpha has order six; discriminant 2 is nonsquare in F5, so its action on the translation plane is irreducible. The related F25-additive:C8 bed reached 73.1 at rate 0.300, motivating an affine experiment without claiming distance transfer. Different orders prove nonisomorphism to the parent.")
SHAPES = ("A=(1,2), B=(2,3), q=150 give n=1200, k>=150 and guaranteed rate 0.125, the included lower boundary of cell 0.150. These shapes permit q<=187. Components have variables/checks 300/150 and 450/300, nominal rates 1/2 and 1/3. Parent measured minima were sector-1 Z witnesses of weight 22..24 across both Acols within individual Bcols, while X readings were 74..85. Increase A length and lower B's nominal rate; authoritative full ranks enforce [0.125,0.175).")
FILLING = ("Twenty cards cross four A degree/multiplier profiles with five B topologies. Degree-eight A rows pair with degree-four B columns, and degree-nine rows with degree-three columns; both quantum row sums remain <=12. Verified monomial lifts supply every component matrix. Exact component kernel-through-four screening, localized A-cell witnesses, independent elimination/descent retests and witness-directed cell repair select fresh fillings. Final localized quantum witnesses are checked modulo original stabilizer spans. Component readings and search failures are not quantum-distance certificates.")
HOW_AB = FILLING
FAMILY_NAME = "r015_F25add_C6_12x23_irreducible_witness_retest_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 150

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("F25-additive:C6 requires order 150")
    # F25 = F5[alpha]/(alpha^2-alpha+1).
    # (a,b,t) acts as z -> alpha^t*z + a+b*alpha.
    # alpha^3=-1, and the six powers below are distinct.
    powers = ((1,0), (0,1), (4,1),
              (4,0), (0,4), (1,4))
    els = [(a,b,t) for t in range(6)
           for b in range(5) for a in range(5)]

    def red(g):
        return (int(g[0]) % 5, int(g[1]) % 5, int(g[2]) % 6)

    def mul(g, h):
        a,b,t = red(g)
        c,d,u = red(h)
        v,w = powers[t]
        return ((a+v*c-w*d) % 5,
                (b+v*d+w*c+w*d) % 5, (t+u) % 6)

    def inv(g):
        a,b,t = red(g)
        ti = (-t) % 6
        v,w = powers[ti]
        return ((-v*a+w*b) % 5,
                (-v*b-w*a-w*b) % 5, ti)

    def idx(g):
        a,b,t = red(g)
        return 25*t+5*b+a

    return dict(elements=els, ident=(0,0,0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


# Profiles specify actual multiplier multiplicities, not orbit images.
# Repeated fibers supply nonzero translations; primitive differences
# act irreducibly and generate the full translation plane.
_A_PLANS = (
    ((0,0,1,2), (0,0,1,2)),
    ((0,0,0,1), (0,1,2,2)),
    ((0,0,1), (0,0,1,1,2)),
    ((0,0,1,2), (0,0,0,1,2)),
)
_B_FOUR = (
    ((2,2,2), (2,2,2)),
    ((3,2,1), (1,2,3)),
    ((3,3,1), (1,1,3)),
    ((4,2,0), (0,2,4)),
    ((4,1,1), (0,3,3)),
)
_B_THREE = (
    ((2,1,2), (1,2,1)),
    ((3,1,1), (0,2,2)),
    ((3,2,0), (0,1,3)),
    ((2,2,2), (1,1,1)),
    ((3,2,1), (0,1,2)),
)


def _affine_cell(rng, profile):
    """Distinct F25 translations with a prescribed multiplier profile."""
    phase = rng.randrange(6)
    fibers = [(t+phase) % 6 for t in profile]
    cell = []
    for t in sorted(set(fibers)):
        coordinates = rng.sample(range(25), fibers.count(t))
        cell.extend((a % 5, a // 5, t) for a in coordinates)
    assert len(cell) == len(set(cell)) == len(profile)
    return cell


def _aff_bits(word):
    while word:
        bit = word & -word
        yield bit.bit_length()-1
        word ^= bit


def _aff_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        for c in _aff_bits(row):
            columns[c] |= 1 << r
    return columns


def _aff_kernel_probe(columns, seed, trials, retained=()):
    """Genuine classical kernel witnesses; sampled upper readings only."""
    diagnostic = random.Random(seed)
    width = len(columns)
    order = list(range(width))
    retained = sorted(set(retained), key=int.bit_count)[:32]
    reading = min((w.bit_count() for w in retained), default=width+1)

    def observe(word, found):
        nonlocal reading
        if not word:
            return
        weight = word.bit_count()
        reading = min(reading, weight)
        if weight <= reading+6:
            found.append(word)

    for _ in range(trials):
        diagnostic.shuffle(order)
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
        if not basis:
            break
        found = []
        for word in basis:
            observe(word, found)
        light = sorted(set(retained+basis), key=int.bit_count)[:20]
        for i, word in enumerate(basis):
            others = basis[:i] if width == 300 else light
            for other in others:
                observe(word ^ other, found)
            if len(found) > 256:
                found = sorted(set(found), key=int.bit_count)[:48]
        diagnostic.shuffle(basis)
        for initial in light[:6]:
            word, weight = initial, initial.bit_count()
            for sweep in range(3):
                changed = False
                for other in basis+retained:
                    candidate = word ^ other
                    if candidate and candidate.bit_count() < weight:
                        word, weight = candidate, candidate.bit_count()
                        changed = True
                if not changed:
                    break
            observe(word, found)
        if len(light) >= 3:
            for _ in range(96):
                a,b,c = diagnostic.sample(light, 3)
                observe(a ^ b ^ c, found)
        retained = sorted(set(retained+light+found),
                          key=int.bit_count)[:32]
        reading = min(reading, retained[0].bit_count())
        if reading <= 6:
            break
    return reading, retained


def _aff_localized(columns, seed):
    """Exact single-cell minimum for nullity<=12, sampled otherwise."""
    pivots, basis = {}, []
    for c, source in enumerate(columns):
        value, word = source, 1 << c
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
    if not basis:
        return len(columns)+1, []
    if len(basis) > 12:
        return _aff_kernel_probe(columns, seed, 12)
    word, best = 0, None
    for t in range(1, 1 << len(basis)):
        word ^= basis[(t & -t).bit_length()-1]
        if best is None or word.bit_count() < best.bit_count():
            best = word
    return best.bit_count(), [best]


def _aff_span(rows):
    pivots = {}
    for source in rows:
        word = source
        while word:
            p = word.bit_length()-1
            old = pivots.get(p)
            if old is None:
                pivots[p] = word
                break
            word ^= old
    return pivots


def _aff_nontrivial(word, pivots):
    """Original stabilizer-span membership, not a distance certificate."""
    while word:
        p = word.bit_length()-1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _component_stats(rows):
    """Count small classical kernels and four-cycles, not CSS distance."""
    columns = {}
    for check, row in enumerate(rows):
        while row:
            bit = row & -row
            pos = bit.bit_length() - 1
            columns[pos] = columns.get(pos, 0) | (1 << check)
            row ^= bit
    syndromes = list(columns.values())
    singles = {}
    n2 = 0
    four_cycles = 0
    check_pairs = {}
    for syndrome in syndromes:
        n2 += singles.get(syndrome, 0)
        singles[syndrome] = singles.get(syndrome, 0) + 1
        neighbors = []
        x = syndrome
        while x:
            bit = x & -x
            neighbors.append(bit.bit_length() - 1)
            x ^= bit
        for a in range(len(neighbors)):
            for b in range(a):
                pair = (neighbors[b], neighbors[a])
                previous = check_pairs.get(pair, 0)
                four_cycles += previous
                check_pairs[pair] = previous + 1

    # With distinct nonzero columns, pair collisions count weight-4
    # kernels three times; membership in singles counts weight-3
    # kernels three times. Duplicate columns are separately penalized.
    pairs = {}
    n3 = 0
    n4 = 0
    for a, x in enumerate(syndromes):
        for b in range(a):
            syndrome = x ^ syndromes[b]
            n3 += singles.get(syndrome, 0)
            previous = pairs.get(syndrome, 0)
            n4 += previous
            pairs[syndrome] = previous + 1
    bad = 10000 * n2 + 100 * (n3 // 3) + n4 // 3
    return bad, four_cycles, len(syndromes)


def NEW_CTX(rng, G, idx):
    """Twenty affine cards, independently screened and witness-repaired."""
    q = G.q
    assert q == 150
    card, cohort = idx % 20, idx // 20
    model, topology = card % 4, card // 4
    seed = rng.randrange(1 << 62)
    mask = (1 << q)-1
    sa = [[len(p) for p in _A_PLANS[model]]]
    source = _B_THREE if model == 3 else _B_FOUR
    sb = [list(row) for row in source[topology]]
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    # Cache native group operations only. Both monomial conventions
    # below are obtained from the authoritative construction helper.
    table = [[G.idx(G.mul(g,h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(G.e)

    class Probe:
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = identity

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverses[g]

    probe = Probe()
    blocks = {"A": [], "B": []}
    for start in range(0, q, 16):
        ids = list(range(start, min(start+16, q)))
        monomials = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(probe, monomials, [[[]]])
        _, zz, _ = lifted_matrices(probe, [[[]]], monomials)
        for j in range(len(ids)):
            blocks["A"].append([(row >> (j*q)) & mask for row in xx])
            blocks["B"].append([(row >> (j*q)) & mask for row in zz])

    def profile(side, i, j, count):
        if side == "A":
            return _A_PLANS[model][j]
        recipes = {
            0: (), 1: (0,),
            2: ((0,1), (0,0), (0,2), (0,1))[model],
            3: ((0,0,1), (0,1,2), (0,0,1), (0,1,2))[model],
            4: (0,0,1,2),
            5: (0,0,1,1,2),
        }
        answer = recipes[count]
        # Extra harness cohorts change selected relative orders, not
        # a common group automorphism applied to one design.
        if count == 2 and cohort % 3 and (i+j+topology) % 2 == 0:
            answer = (0,0) if cohort % 3 == 1 else (0,2)
        return answer

    def fresh(side):
        support = sa if side == "A" else sb
        return [[_affine_cell(rng, profile(side, i, j, count))
                 for j, count in enumerate(row)]
                for i, row in enumerate(support)]

    def inspect(grid, side):
        width = len(grid[0])*q
        rows = [0]*(len(grid)*q)
        for i, base_row in enumerate(grid):
            for j, cell in enumerate(base_row):
                for g in cell:
                    block = blocks[side][G.idx(g)]
                    for r, word in enumerate(block):
                        rows[i*q+r] ^= word << (j*q)
        columns = _aff_columns(rows, width)
        if not all(columns) or len(set(columns)) != width:
            return None
        uses, four = {}, 0
        adjacency = [0]*len(rows)
        for syndrome in columns:
            checks = list(_aff_bits(syndrome))
            for k, a in enumerate(checks):
                adjacency[a] |= syndrome ^ (1 << a)
                for b in checks[:k]:
                    key = (b,a)
                    old = uses.get(key, 0)
                    four += old
                    uses[key] = old+1
        reached = frontier = 1
        while frontier:
            neighbors = 0
            for r in _aff_bits(frontier):
                neighbors |= adjacency[r]
            frontier = neighbors & ~reached
            reached |= frontier
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in grid for cell in row)
        return {
            "grid": grid, "rows": rows, "columns": columns,
            "cycles": four, "rank": gf2_rank(rows),
            "disconnected": reached != (1 << len(rows))-1,
            "signature": signature,
        }

    def diagnose(record, side, trials, offset):
        words = record.get("words", [])
        if side == "A" and "local" not in record:
            local = []
            for j in range(2):
                d, found = _aff_localized(
                    record["columns"][j*q:(j+1)*q],
                    seed+300007+17*j)
                local.extend(word << (j*q) for word in found)
            record["local"] = local
            words = words+local
        d, found = _aff_kernel_probe(
            record["columns"],
            seed+offset+(100003 if side == "B" else 0),
            trials, words)
        record["d"] = min(record.get("d", d), d)
        record["words"] = sorted(set(words+found),
                                 key=int.bit_count)[:32]
        for word in record["words"]:
            syndrome = 0
            for c in _aff_bits(word):
                syndrome ^= record["columns"][c]
            assert word and syndrome == 0

    def quality(record):
        return (-record["small"], record["d"],
                -int(record["disconnected"]),
                -record["cycles"], record["rank"])

    def pool(side):
        records, seen = [], set()
        for _ in range(48):
            record = inspect(fresh(side), side)
            if record is None or record["signature"] in seen:
                continue
            seen.add(record["signature"])
            records.append(record)
        records.sort(key=lambda r: (
            r["disconnected"], r["cycles"],
            len(r["rows"])-r["rank"]))
        survivors = []
        for record in records:
            bad, cycles, width = _component_stats(record["rows"])
            assert width == len(record["columns"])
            record["small"] = bad
            if bad:
                continue
            diagnose(record, side, 8, 0)
            survivors.append(record)
            if len(survivors) >= 8:
                break
        if not survivors:
            # Small-kernel exclusion is preferred, never a source of
            # automatic slot loss. Fallbacks remain weight/rank legal.
            survivors = records[:4]
            for record in survivors:
                if "small" not in record:
                    record["small"] = _component_stats(record["rows"])[0]
                diagnose(record, side, 12, 0)
        if not survivors:
            raise ValueError("no distinct affine component proposals")
        survivors.sort(key=quality, reverse=True)
        finalists = survivors[:3]
        for record in finalists:
            diagnose(record, side, 24, 700001)
            diagnose(record, side, 24, 1700009)
        finalists.sort(key=quality, reverse=True)

        current = finalists[0]
        support = sa if side == "A" else sb
        for move in range(8):
            grid = [[cell[:] for cell in row]
                    for row in current["grid"]]
            active = sorted({c//q for c in _aff_bits(current["words"][0])})
            j = rng.choice(active) if active else rng.randrange(len(grid[0]))
            occupied = [i for i in range(len(grid)) if support[i][j]]
            i = rng.choice(occupied)
            grid[i][j] = _affine_cell(
                rng, profile(side, i, j, support[i][j]))
            proposed = inspect(grid, side)
            if proposed is None or proposed["signature"] in seen:
                continue
            seen.add(proposed["signature"])
            proposed["small"] = _component_stats(proposed["rows"])[0]
            if proposed["small"] > current["small"]:
                continue
            diagnose(proposed, side, 10, 0)
            if quality(proposed) <= quality(current):
                continue
            diagnose(proposed, side, 24, 700001)
            diagnose(proposed, side, 24, 1700009)
            if quality(proposed) > quality(current):
                current = proposed
        if all(current["signature"] != r["signature"] for r in finalists):
            finalists.append(current)
        finalists.sort(key=quality, reverse=True)
        finalists = finalists[:3]
        for record in finalists:
            diagnose(record, side, 48 if side == "A" else 32, 2700011)
        finalists.sort(key=quality, reverse=True)
        return finalists[:2]

    def quantum_reading(hx, hz, a, b, n):
        """Check localized sampled words modulo the original spans."""
        zspan, xspan = _aff_span(hz), _aff_span(hx)
        reading = n+1
        for word in a["words"][:12]:
            for beta in range(3):
                actual = sum(((word >> (j*q)) & mask)
                             << ((3*j+beta)*q) for j in range(2))
                if (all((actual & row).bit_count() % 2 == 0 for row in hx)
                        and _aff_nontrivial(actual, zspan)):
                    reading = min(reading, actual.bit_count())
        for word in b["words"][:12]:
            for j in range(2):
                actual = word << (3*j*q)
                if (all((actual & row).bit_count() % 2 == 0 for row in hz)
                        and _aff_nontrivial(actual, xspan)):
                    reading = min(reading, actual.bit_count())
        return reading

    best = None
    for batch in range(3):
        aa, bb = pool("A"), pool("B")
        for a in aa:
            for b in bb:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if not (n == 1200 and k >= 150
                        and 125*n <= 1000*k < 175*n):
                    continue
                # Independently confirm the cached linear assemblies.
                actual_a = [
                    sum(((row >> (3*j*q)) & mask) << (j*q)
                        for j in range(2))
                    for row in hx[:q]]
                actual_b = [row & ((1 << (3*q))-1)
                            for row in hz[:2*q]]
                assert actual_a == a["rows"] and actual_b == b["rows"]
                dq = quantum_reading(hx, hz, a, b, n)
                detected = min(a["d"], b["d"], dq)
                merit = (-a["small"]-b["small"], detected,
                         k*detected*detected,
                         a["d"]+b["d"],
                         -a["cycles"]-b["cycles"])
                if best is None or merit > best[0]:
                    best = (merit, A, B, hx, hz, k,
                            (a["d"], b["d"], dq),
                            (a["rank"], b["rank"]))
        if best is not None:
            break
    if best is None:
        raise ValueError("affine pairings missed the exact 0.150 rate gate")
    merit, A, B, hx, hz, k, readings, ranks = best
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx+hz) <= min(12, W_MAX)
    return {
        "idx": idx, "cells": {"A": A, "B": B}, "k": k,
        "card": (model, topology, cohort),
        "component_readings": readings,
        "component_ranks": ranks, "quality": merit,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the independently retested full cell without new draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted by parent measurements: rolled positive-cell supports on Z19:Z3 (4,2)x(5,3) reliably avoid small sector-2 logicals; measured slots 2 and 9 both contain X logicals of weight 6.
# REFUTED: Component kernel-through-four and cycle screening suffices to remove the sector-1 A bottleneck on Z37:C3 (1,2)x(3,5); all five parent measured minima remained Z-type weight 22..24 across both Acols at individual Bcols, while X readings were 74..85.
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

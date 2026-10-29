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
CONCEPT = ("Hypothesis: on split-metacyclic C97:C3(s=35), order291, A=B=(1,2), low-boundary eight-variable intercolumn suppression plus accumulated verified kernel challenges raises five-code fitness above982.5 in cell0.200. Measurement can refute this experiment; every finite distance reading remains a provisional upper bound.")
GROUP = ("Preserve the split-metacyclic cyclic-by-cyclic construction family, GTYPE and A0; resize its parameters to m=97,r=3,s=35,q=291 for the new factor-five shape. The action is valid since35^3=1 modulo97. This exact order/shape bed leads cell0.200 at982.5 with measured readings68..72. That evidence motivates the bed, not a certified distance claim or permission to replay its archived labels.")
SHAPES = ("Change B from(2,3) to(1,2), retaining A=(1,2). Equal half-rate components have582 variables and291 checks each; n=1455,k>=291 and guaranteed rate0.200, with unrestricted q_max=299. Parent minimum witnesses occupied sector-one Acol/Bcol strips: Z minima crossed both Acols within Bcol1/2, while X minima could occupy one Acol across all three Bcols. No reported parent minimum used Arow/Brow sector-two blocks. Removing the unequal cubic B component permits four terms in every cell at HX=HZ=12. Exact full ranks enforce0.175<=k/n<0.225.")
FILLING = ("Twenty deliberate cards cross five asymmetric assignments of mixed quartic quotient occupancies with four C6-versus-eight-variable-boundary tradeoffs. Every cell has four distinct elements; native left and right directed differences are unique across each component's two cells. All screening blocks come from lifted_matrices. For actual component blocks M0,M1, explicitly evaluate the syndromes of weight-eight words(M1*e_h,M0*e_k); penalize small boundaries and reject genuine zero-boundary words. This is a selection heuristic, not a distance exclusion. Exact normal-fixed enumeration, component kernel-four gates and ranks, accumulated C3-fixed/unrestricted probes, witness-directed relative-label repair and equally challenged finalists remain binding. Final native assembly, original-span logical-strip reduction, actual rate, weights and CSS are checked. Fresh relative labels are generated independently; no slot is an intentional orbit copy.")
HOW_AB = FILLING
FAMILY_NAME = "r020_metacyclic_m97r3s35_12x12_swapboundary_holdout_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 291

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("resized split-metacyclic bed requires order 291")
    # Same construction rule: (a,b)=x^a*y^b, y*x*y^-1=x^s.
    m, r, s = 97, 3, 35
    powers = tuple(pow(s, b, m) for b in range(r))
    assert powers == (1, 35, 61) and pow(s, r, m) == 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, int(g[1]) % r)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b]*c) % m, (b+d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-powers[bi]*a) % m, bi)

    def idx(g):
        a, b = red(g)
        return m*b+a

    return dict(elements=els, ident=(0, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 2)               # (rb, cb)


def _copy_cells(grid):
    return [[list(cell) for cell in row] for row in grid]


def _cubic_support(rng, cols, concentrated, target):
    """Select a balanced prescription with exactly three terms per column."""
    import itertools
    order = list(range(cols))
    rng.shuffle(order)
    sparse = set(order[:concentrated])
    options = [(0, 3) if j in sparse else (1, 2)
               for j in range(cols)]
    choices = [v for v in itertools.product(*options) if sum(v) == target]
    first = list(rng.choice(choices))
    grid = [first, [3 - v for v in first]]
    assert all(sum(row) > 0 for row in grid)
    return grid


def _coset_pattern(rng, supports):
    """Two split columns carry different coset holonomies."""
    cols = len(supports[0])
    bits = [[rng.randrange(2) for _ in range(cols)] for _ in range(2)]
    split = [j for j in range(cols)
             if supports[0][j] and supports[1][j]]
    assert len(split) >= 2
    j, k = split[:2]
    bits[0][j] = bits[1][j] = 0
    bits[0][k], bits[1][k] = 0, 1
    return bits


def _legal_exponents(exponents, m):
    import math
    if len(set(exponents)) != len(exponents):
        return False
    if len(exponents) < 2:
        return True
    gaps = []
    divisor = m
    for a in range(len(exponents)):
        for b in range(a + 1, len(exponents)):
            d = (exponents[a] - exponents[b]) % m
            if 2 * d == m:
                return False
            gaps.append(min(d, m - d))
            divisor = math.gcd(divisor, d)
    # Distinct signed difference classes prevent internal four-cycles.
    return divisor == 1 and len(set(gaps)) == len(gaps)


def _new_coset_cell(rng, m, count, bit):
    if count == 0:
        return []
    while True:
        exponents = rng.sample(range(m), count)
        if _legal_exponents(exponents, m):
            return [(a, bit) for a in exponents]


def _initial_cubic_cells(rng, G, supports, bits):
    m = G.q // 2
    return [[_new_coset_cell(rng, m, count, bits[i][j])
             for j, count in enumerate(row)]
            for i, row in enumerate(supports)]


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


def _verified_costs(G, A, B, penalty, side=None):
    hx, hz, _ = lifted_matrices(G, A, B)
    sides = (side,) if side is not None else ("A", "B")
    costs = [_cycle_cost(_component_rows(hx, hz, G.q, s), penalty)
             for s in sides]
    return costs[0] if side is not None else costs


def _small_kernel_penalty(rows, width):
    """Exact <=4 veto plus a connected-triple weight-six witness screen."""
    import itertools
    columns = [0] * width
    adjacency = [set() for _ in range(width)]
    for r, row in enumerate(rows):
        remaining = row
        incident = []
        while remaining:
            bit = remaining & -remaining
            j = bit.bit_length() - 1
            columns[j] |= 1 << r
            incident.append(j)
            remaining ^= bit
        for j in incident:
            adjacency[j].update(k for k in incident if k != j)

    singles = set(columns)
    if 0 in singles or len(singles) != width:
        return 1000000

    pairs = set()
    short = 0
    for a in range(width):
        for b in range(a + 1, width):
            syndrome = columns[a] ^ columns[b]
            if syndrome in singles or syndrome in pairs:
                short += 1
            pairs.add(syndrome)
    if short:
        return 100000 + short

    # Enumerate triples connected through shared checks. A collision
    # verifies a six-column kernel word; this is not an exhaustive
    # general weight-six search.
    triples = set()
    for b in range(width):
        for a, c in itertools.combinations(sorted(adjacency[b]), 2):
            triples.add(tuple(sorted((a, b, c))))
    seen = {}
    six = 0
    for a, b, c in sorted(triples):
        syndrome = columns[a] ^ columns[b] ^ columns[c]
        mask = (1 << a) | (1 << b) | (1 << c)
        old = seen.get(syndrome)
        if old is not None:
            if (old ^ mask).bit_count() == 6:
                six += 1
        else:
            seen[syndrome] = mask
    return six


def _optimize_labels(rng, G, A, B, penalty):
    import math
    costs = _verified_costs(G, A, B, penalty)
    current = sum(costs)
    best = current
    best_A, best_B = _copy_cells(A), _copy_cells(B)
    positions = [[(i, j) for i, row in enumerate(grid)
                  for j, cell in enumerate(row) if cell]
                 for grid in (A, B)]
    m = G.q // 2
    steps = 240
    for step in range(steps):
        which = rng.randrange(2)
        grid = A if which == 0 else B
        i, j = rng.choice(positions[which])
        old = list(grid[i][j])
        bit = old[0][1]
        if step % 9 == 0:
            new = _new_coset_cell(rng, m, len(old), bit)
        else:
            new = None
            for _ in range(40):
                exponents = [g[0] for g in old]
                exponents[rng.randrange(len(old))] = rng.randrange(m)
                if _legal_exponents(exponents, m):
                    new = [(a, bit) for a in exponents]
                    break
            if new is None:
                continue
        grid[i][j] = new
        side = ("A", "B")[which]
        proposed_cost = _verified_costs(G, A, B, penalty, side)
        proposed = current - costs[which] + proposed_cost
        temperature = G.q * (18.0 * (1.0 - step / steps) ** 2 + 0.25)
        accept = proposed <= current
        if not accept:
            accept = rng.random() < math.exp((current - proposed) /
                                             temperature)
        if accept:
            current = proposed
            costs[which] = proposed_cost
            if current < best:
                best = current
                best_A, best_B = _copy_cells(A), _copy_cells(B)
        else:
            grid[i][j] = old
    return best_A, best_B, best


def NEW_CTX(rng, G, idx):
    """Equal quartic components selected by verified swap-boundary probes."""
    q = G.q
    assert q == 291 and A_SHAPE == B_SHAPE == (1, 2)
    card, cohort = idx % 20, idx // 20
    model, policy = divmod(card, 4)
    seed = rng.randrange(1 << 62) ^ (0x97C320B8 + 15485863*idx)
    rng = random.Random(seed ^ rng.getrandbits(128))
    mask = (1 << q)-1
    support = [[4, 4]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)

    # Five ASYMMETRIC assignments, not matched copies of a side.
    p, d, t = (2, 1, 1), (2, 2, 0), (3, 1, 0)
    profiles = (
        ((p, p), (p, p)),
        ((p, d), (p, p)),
        ((d, d), (p, d)),
        ((t, p), (d, p)),
        ((t, d), (t, p)),
    )
    plans = dict(zip(("A", "B"), profiles[model]))
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(G.e)

    class Probe:
        def __init__(self):
            self.q, self.e = q, identity
            self.elements = list(range(q))

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverse[g]

    P = Probe()
    # Extract and transpose AUTHORITATIVE sector-two monomials.
    # Tall galleries avoid q^2 rows containing very high bit positions.
    gallery = [[[g]] for g in range(q)]
    _, at, _ = lifted_matrices(P, gallery, [[[]]])
    bt, _, _ = lifted_matrices(P, [[[]]], gallery)
    blocks = {}
    for side, raw in (("A", at), ("B", bt)):
        monos = []
        for g in range(q):
            rows = [0]*q
            for r, source in enumerate(raw):
                value = (source >> ((g+1)*q)) & mask
                assert value.bit_count() == 1
                rows[value.bit_length()-1] |= 1 << r
            monos.append(rows)
        blocks[side] = monos
    del gallery, at, bt

    def differences(cell):
        left, right = set(), set()
        for a in cell:
            for b in cell:
                if a != b:
                    left.add(table[inverse[a]][b])
                    right.add(table[a][inverse[b]])
        return left, right

    def legal_cell(cell):
        if len(cell) != 4 or len(set(cell)) != 4:
            return False
        left, right = differences(cell)
        if len(left) != 12 or len(right) != 12:
            return False
        ratios = [table[inverse[cell[0]]][g] for g in cell[1:]]
        return any(table[a][b] != table[b][a]
                   for i, a in enumerate(ratios) for b in ratios[:i])

    def pair_free(grid):
        first, second = grid[0]
        if not legal_cell(first) or not legal_cell(second):
            return False
        a, b = differences(first)
        c, d_ = differences(second)
        return a.isdisjoint(c) and b.isdisjoint(d_)

    def draw(side, j):
        occupancy = plans[side][j]
        for _ in range(2048):
            fibers = list(range(3))
            rng.shuffle(fibers)
            cell = []
            for b, count in zip(fibers, occupancy):
                cell.extend(G.idx((a, b))
                            for a in rng.sample(range(97), count))
            if legal_cell(cell):
                return cell
        raise ValueError("mixed quartic Sidon recipe exhausted")

    def fresh(side):
        for attempt in range(2048):
            if attempt % 32 == 0:
                first = draw(side, 0)
            grid = [[first, draw(side, 1)]]
            if pair_free(grid):
                return grid
        raise ValueError("bidirectional quartic pair exhausted")

    def dependencies(columns, order):
        pivots, basis = {}, []
        for c in order:
            value, word = columns[c], 1 << c
            while value:
                lead = value.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = (value, word)
                    break
                value ^= old[0]
                word ^= old[1]
            if not value:
                basis.append(word)
        return basis

    groups = {"A": {}, "B": {}}
    for side in groups:
        for order, subgroup in (
                (3, [G.idx((0, b)) for b in range(3)]),
                (97, [G.idx((a, 0)) for a in range(97)])):
            remaining, orbits = set(range(q)), []
            while remaining:
                root = min(remaining)
                orbit = {
                    table[root][h] if side == "A" else table[h][root]
                    for h in subgroup
                }
                assert len(orbit) == order and orbit <= remaining
                remaining.difference_update(orbit)
                orbits.append(tuple(sorted(orbit)))
            groups[side][order] = [
                sum(1 << (j*q+c) for c in orbit)
                for j in range(2) for orbit in orbits
            ]

    def expand(pattern, partition):
        word = 0
        for c in _mp_bits(pattern):
            word ^= partition[c]
        return word

    def inspect(grid, side):
        if not pair_free(grid):
            return None
        rows = [0]*q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                for r, value in enumerate(blocks[side][g]):
                    rows[r] ^= value << (j*q)
        columns = [0]*(2*q)
        for r, row in enumerate(rows):
            for c in _mp_bits(row):
                columns[c] |= 1 << r
        if not all(columns) or len(set(columns)) != 2*q:
            return None
        assert all(c.bit_count() == 4 for c in columns)

        # Genuine eight-variable precursors:
        # H*(M1*e_h, M0*e_k)=M0*M1*e_h + M1*M0*e_k.
        # Regular symmetry makes h=identity cover every relative k.
        c0, c1 = columns[:q], columns[q:]
        first = _mp_syndrome(c0, c1[identity])
        boundaries = [
            (first ^ _mp_syndrome(c1, c0[k])).bit_count()
            for k in range(q)
        ]
        if min(boundaries) == 0:
            k = boundaries.index(0)
            word = c1[identity] | (c0[k] << q)
            assert word.bit_count() == 8
            assert _mp_syndrome(columns, word) == 0
            return None

        six = _cycle_cost(rows, 0)
        boundary_energy = sum(max(0, 24-v)**2 for v in boundaries)
        boundary_low = sum(v <= 16 for v in boundaries)
        cost = ((16, 48, 144, 432)[policy]*six
                + (512, 256, 128, 64)[policy]*boundary_energy
                + 20000*boundary_low)

        partition = groups[side][97]
        normal = [_mp_syndrome(columns, w) for w in partition]
        basis = dependencies(normal, range(len(normal)))
        assert len(partition) == 6 and len(basis) <= 6
        words, pattern = [], 0
        for turn in range(1, 1 << len(basis)):
            pattern ^= basis[(turn & -turn).bit_length()-1]
            words.append(expand(pattern, partition))
        words = sorted(set(words), key=lambda w: (w.bit_count(), w))[:24]
        if not words or words[0].bit_count() < 80:
            return None
        assert all(_mp_syndrome(columns, w) == 0 for w in words)
        fixed_groups = groups[side][3]
        fixed = [_mp_syndrome(columns, w) for w in fixed_groups]
        return dict(grid=_copy_cells(grid), rows=rows, columns=columns,
                    signature=tuple(tuple(sorted(c)) for c in grid[0]),
                    cost=cost, six=six,
                    boundary=(min(boundaries), boundary_low, boundary_energy),
                    normal=words[0].bit_count(), beam=words,
                    fixed=fixed, fixed_groups=fixed_groups, fixed_beam=[],
                    fixed_d=2*q+1, d=words[0].bit_count())

    def exact_gate(rec):
        rows, columns = rec["rows"], rec["columns"]
        rank = gf2_rank(rows)
        if rank < q-6:
            return False
        cell_ranks = (gf2_rank([r & mask for r in rows]),
                      gf2_rank([r >> q for r in rows]))
        if min(cell_ranks) < q-9:
            return False
        singles = set(columns)
        seen = set()
        for i, value in enumerate(columns):
            for other in columns[:i]:
                pair = value ^ other
                if pair in singles or pair in seen:
                    return False
                seen.add(pair)
        rec["rank"], rec["cell_ranks"] = rank, cell_ranks
        return True

    def search(columns, trials, offset, initial=()):
        """Accumulated genuine words; finite exhaustion excludes nothing."""
        local = random.Random(seed+offset)
        beam = sorted({w for w in initial if w},
                      key=lambda w: (w.bit_count(), w))[:24]
        assert all(_mp_syndrome(columns, w) == 0 for w in beam)
        order = list(range(len(columns)))
        for turn in range(trials):
            local.shuffle(order)
            if beam and turn % 4 == 3:
                active = list(_mp_bits(local.choice(beam[:6])))
                preferred = set(local.sample(active, max(1, len(active)//2)))
                order.sort(key=lambda c: c not in preferred)
            basis = dependencies(columns, order)
            if not basis:
                continue
            light = sorted(set(beam+basis),
                           key=lambda w: (w.bit_count(), w))[:16]
            moves, grown = basis+beam, list(light)+beam
            best = light[0].bit_count()

            def observe(word):
                nonlocal best, grown
                if not word:
                    return
                weight = word.bit_count()
                best = min(best, weight)
                if weight <= best+6:
                    grown.append(word)
                if len(grown) > 384:
                    grown = sorted(set(grown),
                                   key=lambda w: (w.bit_count(), w))[:48]

            if len(columns) <= 2*q//3:
                # Complete pairs of each sampled restricted basis.
                for i, word in enumerate(basis):
                    for other in basis[:i]:
                        observe(word ^ other)
            for word in light:
                for other in moves:
                    observe(word ^ other)
            local.shuffle(moves)
            for initial_word in light[:6]:
                word = initial_word
                for _ in range(3):
                    before = word.bit_count()
                    for move in moves:
                        proposed = word ^ move
                        if proposed and proposed.bit_count() < word.bit_count():
                            word = proposed
                    if word.bit_count() == before:
                        break
                observe(word)
            beam = sorted({w for w in grown if w},
                          key=lambda w: (w.bit_count(), w))[:24]
            if beam[0].bit_count() <= 8:
                break
        if not beam:
            raise ValueError("no actual equal-component kernel witness")
        assert _mp_syndrome(columns, beam[0]) == 0
        return beam

    def diagnose(rec, full_trials, fixed_trials, offset):
        rec["fixed_beam"] = search(
            rec["fixed"], fixed_trials, offset+5000011,
            rec["fixed_beam"])
        actuals = [expand(w, rec["fixed_groups"])
                   for w in rec["fixed_beam"][:8]]
        assert all(_mp_syndrome(rec["columns"], w) == 0 for w in actuals)
        rec["fixed_d"] = min(
            rec["fixed_d"], actuals[0].bit_count())
        rec["beam"] = search(
            rec["columns"], full_trials, offset,
            rec["beam"]+actuals)
        rec["d"] = min(rec["d"], rec["beam"][0].bit_count())
        assert rec["beam"][0].bit_count() == rec["d"]

    def merit(rec):
        # Seventy-four is only a selection cap, never a lower bound.
        return (min(74, rec["d"]), -rec["cost"], rec["d"], rec["rank"])

    def optimize(side):
        offset = 100003*int(side == "B")
        records, seen = [], set()
        for _ in range(96):
            rec = inspect(fresh(side), side)
            if rec is None or rec["signature"] in seen:
                continue
            seen.add(rec["signature"])
            records.append(rec)
        records.sort(key=lambda r: r["cost"])
        selected = []
        for rec in records:
            if not exact_gate(rec):
                continue
            diagnose(rec, 4, 12, offset+700001)
            selected.append(rec)
            if len(selected) == 8:
                break
        if not selected:
            raise ValueError("no equal quartic component passed exact gates")
        current = max(selected, key=merit)
        for step in range(28):
            grid = _copy_cells(current["grid"])
            word = rng.choice(current["beam"][:8])
            occupied = sorted({c//q for c in _mp_bits(word)})
            j = rng.choice(occupied)
            if step % 7 == 6:
                grid = fresh(side)
            elif step % 3 == 0:
                grid[0][j] = draw(side, j)
            else:
                at_ = rng.randrange(4)
                old = G.elements[grid[0][j][at_]]
                options = [G.idx((a, old[1])) for a in range(97)
                           if G.idx((a, old[1])) not in grid[0][j]]
                grid[0][j][at_] = rng.choice(options)
            rec = inspect(grid, side)
            if rec is None or rec["signature"] in seen:
                continue
            seen.add(rec["signature"])
            if not exact_gate(rec):
                continue
            known = [w for w in current["beam"]
                     if _mp_syndrome(rec["columns"], w) == 0]
            rec["beam"] = sorted(set(rec["beam"]+known),
                                 key=lambda w: (w.bit_count(), w))[:24]
            rec["d"] = rec["beam"][0].bit_count()
            diagnose(rec, 3, 10, offset+1000003+7919*step)
            selected.append(rec)
            if merit(rec) >= merit(current):
                current = rec
            elif rec["d"] >= current["d"]-2 and rng.random() < 0.08:
                current = rec

        finalists = sorted(selected, key=merit, reverse=True)[:4]
        # EVERY eligible finalist gets both independent challenge stages.
        for at_, rec in enumerate(finalists):
            diagnose(rec, 24, 64, offset+2700011+1013*at_)
            diagnose(rec, 16, 64, offset+6700061+1009*at_)
        finalists.sort(key=merit, reverse=True)
        return finalists[:2]

    def strip_readings(a, b, hx, hz, n):
        """Nontrivial actual logical strips modulo ORIGINAL stabilizer spans."""
        readings = []
        for kind, rec, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = _mp_span(stabilizers)
            candidates = set()
            for word in rec["beam"][:6]:
                for location in range(2):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask) << ((2*j+location)*q)
                            for j in range(2))
                    else:
                        actual = word << (2*location*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if _mp_nontrivial(actual, pivots):
                        candidates.add(actual)
            reading = n+1
            local = random.Random(seed+(3100009 if kind == "Z" else 4100011))
            reducers = list(stabilizers)
            local.shuffle(reducers)
            for actual in sorted(candidates, key=int.bit_count)[:8]:
                for _ in range(3):
                    before = actual.bit_count()
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                    nearby = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:32]
                    best = actual
                    for i, first in enumerate(nearby):
                        for second in nearby[:i]:
                            proposed = actual ^ first ^ second
                            if proposed.bit_count() < best.bit_count():
                                best = proposed
                    actual = best
                    if actual.bit_count() == before:
                        break
                assert actual and _mp_nontrivial(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            readings.append(reading)
        return tuple(readings)

    aa, bb = optimize("A"), optimize("B")
    best = None
    for a in aa:
        for b in bb:
            hx, hz, n = lifted_matrices(P, a["grid"], b["grid"])
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (n == 1455 and k >= q
                    and 175*n <= 1000*k < 225*n):
                continue
            assert _component_rows(hx, hz, q, "A") == a["rows"]
            assert _component_rows(hx, hz, q, "B") == b["rows"]
            logicals = strip_readings(a, b, hx, hz, n)
            detected = min(a["d"], b["d"], *logicals)
            capped = min(74, detected)
            key = (k*capped*capped, capped, -a["cost"]-b["cost"],
                   detected, a["d"]+b["d"])
            if best is None or key > best[0]:
                best = key, a, b, hx, hz, k, logicals
    if best is None:
        raise ValueError("equal-component pairings missed cell 0.200")
    key, a, b, cached_hx, cached_hz, k, logicals = best
    A, B = [
        [[[G.elements[g] for g in cell] for cell in row]
         for row in rec["grid"]]
        for rec in (a, b)
    ]
    hx, hz, n = lifted_matrices(G, A, B)
    assert hx == cached_hx and hz == cached_hz and n == 1455
    assert max(row.bit_count() for row in hx+hz) == 12
    assert css_ok(hx, hz)
    assert all(len(cell) == len(set(cell)) == 4
               for grid in (A, B) for row in grid for cell in row)
    return {"idx": idx, "cells": {"A": A, "B": B}, "screen_k": k,
            "card": (model, policy, cohort), "target_cell": 0.200,
            "component_readings": (a["d"], b["d"]),
            "C3_fixed_readings": (a["fixed_d"], b["fixed_d"]),
            "normal_fixed_exact_minima": (a["normal"], b["normal"]),
            "component_ranks": (a["rank"], b["rank"]),
            "component_cell_ranks": (a["cell_ranks"], b["cell_ranks"]),
            "component_six_cycles": (a["six"], b["six"]),
            "eight_variable_boundary_profiles": (a["boundary"], b["boundary"]),
            "verified_localized_logical_readings": logicals,
            "quality": key}


def _mp_bits(word):
    while word:
        bit = word & -word
        yield bit.bit_length() - 1
        word ^= bit


def _mp_syndrome(columns, word):
    result = 0
    for v in _mp_bits(word):
        result ^= columns[v]
    return result


def _mp_span(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = word
                break
            word ^= old
    return pivots


def _mp_nontrivial(word, pivots):
    """Exact membership test in the original stabilizer span."""
    while word:
        p = word.bit_length() - 1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _retired_faithful_six_ctx(rng, G, idx):
    """Inactive unequal-component generator; NEW_CTX defines the new bed."""
    q = G.q
    assert q == 186
    assert A_SHAPE == (1, 2) and B_SHAPE == (2, 3)
    card, cohort = idx % 20, idx // 20
    mode, topology = card % 5, card // 5
    seed = rng.randrange(1 << 62) ^ 0x186C316B2
    # Fresh relative coordinates, never a stored code or an orbit copy.
    rng = random.Random(
        rng.getrandbits(128) ^ seed ^ (1000003*(idx+1)))
    sa = [[4, 5]]
    # Four cards change both support balance and the number of long-order
    # binomials. No card is manufactured by relabeling another filling.
    bplans = (
        ((1, 1, 2), (2, 2, 1)),
        ((1, 1, 2), (2, 2, 1)),
        ((1, 1, 1), (2, 2, 2)),
        ((1, 1, 1), (2, 2, 2)),
    )
    sb = [list(row) for row in bplans[topology]]
    # At most three occupied quotient fibers, never an all-even cell.
    # These five pairs change internal degree/relative-label geometry.
    amodels = (
        ((2, 1, 1), (2, 2, 1)),
        ((2, 1, 1), (3, 1, 1)),
        ((3, 1, 0), (2, 2, 1)),
        ((3, 1, 0), (3, 1, 1)),
        ((3, 1, 0), (3, 2, 0)),
    )
    assert all(any(size & 1 for size in occupancy)
               for pair in amodels for occupancy in pair)
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert all(count > 0 for row in sb for count in row)

    # Cache native operations, never an independently implemented lift.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(G.e)

    # Prescribe actual native ratio orders, not quotient names.
    orders = []
    for g in range(q):
        value = identity
        for order in range(1, q+1):
            value = table[value][g]
            if value == identity:
                orders.append(order)
                break
        else:
            raise ValueError("invalid finite group")
    ratio_pools = {
        order: [g for g in range(q) if orders[g] == order]
        for order in (3, 6, 31)
    }
    assert all(ratio_pools.values())

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

    P = Probe()
    mask = (1 << q) - 1
    monomials = [[[g] for g in range(q)]]
    source, _, _ = lifted_matrices(P, monomials, [[[]]])
    ablocks = [[(row >> (g*q)) & mask for row in source]
               for g in range(q)]
    _, source, _ = lifted_matrices(P, [[[]]], monomials)
    bblocks = [[(row >> (g*q)) & mask for row in source]
               for g in range(q)]

    # Coordinate partitions for the actual regular component symmetries:
    # A-left components commute with right multiplication; B-right
    # components commute with left multiplication. Restrictions below
    # use ONLY sums of authoritative component columns.
    subgroup = [G.idx((0, 2*b)) for b in range(3)]
    fixed_masks = {}
    normal_masks = {
        side: [
            sum(1 << (block*q + G.idx((a, b))) for a in range(31))
            for block in range(count) for b in range(6)
        ]
        for side, count in (("A", 2), ("B", 3))
    }
    for side, blocks in (("A", 2), ("B", 3)):
        remaining, orbits = set(range(q)), []
        while remaining:
            root = min(remaining)
            orbit = {
                table[root][h] if side == "A" else table[h][root]
                for h in subgroup
            }
            assert len(orbit) == 3 and orbit <= remaining
            remaining.difference_update(orbit)
            orbits.append(tuple(sorted(orbit)))
        fixed_masks[side] = [
            sum(1 << (block*q + v) for v in orbit)
            for block in range(blocks) for orbit in orbits
        ]

    def normal_witnesses(columns, side):
        """Exact normal-fixed minimum at nullity<=14; witnesses otherwise.

        All compressed columns are sums of authoritative physical columns.
        Even an exact restricted result says nothing about unrestricted
        logical distance or whether every classical word embeds nontrivially.
        """
        masks = normal_masks[side]
        compressed = [_mp_syndrome(columns, word) for word in masks]
        pivots, basis = {}, []
        for c, source in enumerate(compressed):
            value, word = source, 1 << c
            while value:
                lead = value.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = (value, word)
                    break
                value ^= old[0]
                word ^= old[1]
            if not value:
                basis.append(word)

        kept = []
        best_weight = len(masks)+1

        def observe(pattern):
            nonlocal best_weight, kept
            if not pattern:
                return
            weight = pattern.bit_count()
            if weight < best_weight:
                best_weight = weight
                kept = []
            if weight == best_weight:
                kept.append(pattern)
                if len(kept) > 16:
                    kept = sorted(set(kept))[:8]

        if len(basis) <= 14:
            pattern = 0
            # Gray-code enumeration visits every basis combination.
            for t in range(1, 1 << len(basis)):
                pattern ^= basis[(t & -t).bit_length()-1]
                observe(pattern)
        else:
            for t, first in enumerate(basis):
                observe(first)
                for second in basis[:t]:
                    observe(first ^ second)

        actuals = []
        for pattern in sorted(set(kept))[:8]:
            actual = 0
            for c in _mp_bits(pattern):
                actual ^= masks[c]
            assert actual.bit_count() == 31*pattern.bit_count()
            assert _mp_syndrome(columns, actual) == 0
            actuals.append(actual)
        return actuals, len(basis) <= 14

    def whole_sidon(cell):
        if len(cell) != len(set(cell)):
            return False
        if len(cell) < 2:
            return True
        left, right = set(), set()
        for a in cell:
            for b in cell:
                if a != b:
                    left.add(table[inverses[a]][b])
                    right.add(table[a][inverses[b]])
        wanted = len(cell)*(len(cell)-1)
        if len(left) != wanted or len(right) != wanted:
            return False
        if len(cell) >= 3:
            ratios = [table[inverses[cell[0]]][g] for g in cell[1:]]
            if not any(table[a][b] != table[b][a]
                       for t, a in enumerate(ratios) for b in ratios[:t]):
                return False
        return True

    def draw(occupancy):
        sizes = [size for size in occupancy if size]
        assert 2 <= len(sizes) <= 3
        assert any(size & 1 for size in sizes)
        for _ in range(4096):
            # A quotient difference of three gives an involution:
            # reversing that pair repeats its directed difference.
            # Consecutive fibers avoid this impossible Sidon condition.
            origin = rng.randrange(6)
            direction = rng.choice((1, -1))
            rng.shuffle(sizes)
            cell = []
            for t, size in enumerate(sizes):
                b = (origin + direction*t) % 6
                cell.extend(G.idx((a, b))
                            for a in rng.sample(range(31), size))
            if whole_sidon(cell):
                return cell
        raise ValueError("involution-free Sidon occupancy exhausted")

    def cell_for(side, i, j):
        if side == "A":
            return draw(amodels[mode][j])
        count = sb[i][j]
        if count == 1:
            return [rng.randrange(q)]
        assert count == 2
        # Binding order allocations on two positive incidence graphs.
        # 0/2: all order-six ratios.
        # 1: one order31 and two order-three ratios.
        # 3: one each of orders31, three and six.
        if topology in (0, 2):
            wanted = 6
        elif topology == 1:
            wanted = 31 if (i, j) == (1, 0) else 3
        else:
            wanted = (31, 3, 6)[j]
        root = rng.randrange(q)
        ratio = rng.choice(ratio_pools[wanted])
        cell = [root, table[root][ratio]]
        assert whole_sidon(cell)
        assert orders[table[inverses[cell[0]]][cell[1]]] == wanted
        return cell

    def right_differences(cell):
        return {table[a][inverses[b]]
                for a in cell for b in cell if a != b}

    def fresh(side):
        if side == "A":
            # Within-cell Sidon plus disjoint right differences excludes
            # A-left cross-column C4. inspect independently verifies this
            # against authoritative blocks before accepting the candidate.
            for attempt in range(1024):
                if attempt % 24 == 0:
                    first = cell_for("A", 0, 0)
                    differences = right_differences(first)
                second = cell_for("A", 0, 1)
                if differences.isdisjoint(right_differences(second)):
                    return [[first, second]]
            raise ValueError("mixed zero-four-cycle A pair exhausted")
        return [[cell_for(side, i, j) for j in range(len(row))]
                for i, row in enumerate(sb)]

    def eight_cycles(rows, columns):
        """Exact simple Tanner C8 count, including cycles with chords.

        Every component column block has a transitive regular action.
        Pair internally disjoint length-four paths from each orbit root.
        This statistic is geometry, not a distance exclusion certificate.
        """
        checks = [list(_mp_bits(row)) for row in rows]
        neighbors = [list(_mp_bits(col)) for col in columns]
        rooted = 0
        for root in range(0, len(columns), q):
            paths = {}
            for c1 in neighbors[root]:
                for middle in checks[c1]:
                    if middle == root:
                        continue
                    for c2 in neighbors[middle]:
                        if c2 == c1:
                            continue
                        for end in checks[c2]:
                            if end == root or end == middle:
                                continue
                            previous = paths.setdefault(end, [])
                            for d1, other, d2 in previous:
                                if (other != middle
                                        and c1 != d1 and c1 != d2
                                        and c2 != d1 and c2 != d2):
                                    rooted += 1
                            previous.append((c1, middle, c2))
        assert q * rooted % 4 == 0
        return q * rooted // 4

    def inspect(grid, side):
        """GF2 sums of authoritative scaffold monomial blocks."""
        blocks = ablocks if side == "A" else bblocks
        width = len(grid[0])*q
        rows = []
        for cells in grid:
            for r in range(q):
                row = 0
                for j, cell in enumerate(cells):
                    block = 0
                    for g in cell:
                        block ^= blocks[g][r]
                    row |= block << (j*q)
                rows.append(row)
        columns = [0]*width
        for r, row in enumerate(rows):
            for v in _mp_bits(row):
                columns[v] |= 1 << r
        if not all(columns) or len(set(columns)) != width:
            return None
        normal_words, normal_exact = normal_witnesses(columns, side)
        normal_reading = min((word.bit_count() for word in normal_words),
                             default=width+1)
        if normal_reading < 50:
            return None

        adjacency = [0]*len(rows)
        pairs = {}
        four = forced = 0
        for column in columns:
            ns = list(_mp_bits(column))
            d = len(ns)
            forced += d*(d-1)*(d-2)//6
            for t, a in enumerate(ns):
                adjacency[a] |= column ^ (1 << a)
                for b in ns[:t]:
                    key = (b, a)
                    old = pairs.get(key, 0)
                    four += old
                    pairs[key] = old + 1
        reached = frontier = 1
        while frontier:
            neighbors = 0
            for r in _mp_bits(frontier):
                neighbors |= adjacency[r]
            frontier = neighbors & ~reached
            reached |= frontier
        if reached != (1 << len(rows))-1:
            return None

        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in _mp_bits(neighbors & (-1 << (a+1))):
                triangles += (
                    neighbors & adjacency[b] & (-1 << (b+1))
                ).bit_count()
        excess = max(0, triangles-forced)
        eight = low_six = low_eight = 0
        if side == "A":
            if four:
                return None
            # With no C4, projection triangle excess counts actual C6.
            # Also measure the induced degree-four-variable subgraph:
            # its cycles have less extrinsic degree than mixed 4/5 cycles.
            low_rows = [row & mask for row in rows]
            low_columns = columns[:q]
            eight = eight_cycles(rows, columns)
            low_six = _cycle_cost(low_rows, 0)
            low_eight = eight_cycles(low_rows, low_columns)
            cost = ((60, 120, 240, 480, 960)[mode] * excess
                    + eight
                    + (120, 180, 240, 360, 480)[mode] * low_six
                    + (8, 12, 20, 32, 48)[mode] * low_eight)
        else:
            cost = (400, 700, 1100, 1600)[mode % 4]*four + excess
        signature = tuple(tuple(sorted(cell))
                          for row in grid for cell in row)
        return dict(grid=_copy_cells(grid), rows=rows, columns=columns,
                    cost=cost,
                    cycles=(four, excess, eight, low_six, low_eight),
                    signature=signature, d=normal_reading,
                    words=list(normal_words),
                    normal_fixed_reading=normal_reading,
                    normal_fixed_exact=normal_exact)

    def note(rec, word):
        if not word:
            return
        weight = word.bit_count()
        rec["d"] = min(rec["d"], weight)
        if weight <= rec["d"] + 6:
            rec["words"].append(word)
            if len(rec["words"]) > 80:
                rec["words"] = sorted(set(rec["words"]),
                                      key=int.bit_count)[:20]

    def fixed_challenge(rec, side, trials, offset):
        """Actual C3-fixed kernel witnesses, not an exclusion certificate."""
        masks = fixed_masks[side]
        if "fixed_columns" not in rec:
            rec["fixed_columns"] = [
                _mp_syndrome(rec["columns"], support)
                for support in masks
            ]
            rec["fixed_words"] = []
        columns = rec["fixed_columns"]
        width = len(columns)
        diagnostic = random.Random(seed + offset)
        order = list(range(width))
        retained = list(rec["fixed_words"])
        best = min((w.bit_count() for w in retained), default=width+1)

        def expand(word):
            actual = 0
            for c in _mp_bits(word):
                actual ^= masks[c]
            return actual

        def observe(word, found):
            nonlocal best
            if not word:
                return
            weight = word.bit_count()
            best = min(best, weight)
            if weight <= best+4:
                found.append(word)
            if 3*weight <= rec["d"]+6:
                note(rec, expand(word))

        for word in retained:
            note(rec, expand(word))
        for trial in range(trials):
            diagnostic.shuffle(order)
            pivots, basis, independent = {}, [], []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    p = value.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
                        independent.append(c)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
            if not basis:
                raise ValueError("fixed component restriction has no kernel")

            found = []
            for word in basis:
                observe(word, found)
            light = sorted(set(retained+basis), key=int.bit_count)[:16]
            for t, word in enumerate(basis):
                others = basis[:t] if side == "A" else light
                for other in others:
                    observe(word ^ other, found)
                if len(found) > 160:
                    found = sorted(set(found), key=int.bit_count)[:32]

            # Stern-style finite list challenge: pair words that agree
            # on sampled independent coordinates. Their XOR is a genuine
            # kernel word involving up to four fundamental circuits.
            # Hash misses make this a witness search, not an exclusion.
            if len(basis) >= 4 and independent:
                sample = diagnostic.sample(
                    independent, min(8, len(independent)))
                hash_mask = sum(1 << c for c in sample)
                buckets = {}
                for t, first in enumerate(basis):
                    for second in basis[:t]:
                        pair = first ^ second
                        key = pair & hash_mask
                        bucket = buckets.setdefault(key, [])
                        for other in bucket:
                            observe(pair ^ other, found)
                        bucket.append(pair)
                if len(found) > 256:
                    found = sorted(set(found), key=int.bit_count)[:32]

            if side == "A" and trials >= 24 and trial == trials-1:
                # Complete triples of ONE sampled restricted basis.
                # This exhausts these combinations, not the full kernel.
                for t, first in enumerate(basis):
                    for u in range(t):
                        pair = first ^ basis[u]
                        for third in basis[:u]:
                            observe(pair ^ third, found)
                    if len(found) > 256:
                        found = sorted(set(found), key=int.bit_count)[:32]

            moves = retained+basis
            diagnostic.shuffle(moves)
            for initial in light[:6]:
                word = initial
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    if not changed:
                        break
                observe(word, found)
            if len(basis) >= 3:
                for _ in range(64):
                    a, b, c = diagnostic.sample(basis, 3)
                    observe(a ^ b ^ c, found)
            retained = sorted(set(retained+light+found),
                              key=int.bit_count)[:24]
            if best <= 8:
                break

        rec["fixed_words"] = retained
        actual = expand(retained[0])
        assert actual.bit_count() == 3*retained[0].bit_count()
        assert _mp_syndrome(rec["columns"], actual) == 0
        assert all((actual & row).bit_count() % 2 == 0
                   for row in rec["rows"])
        note(rec, actual)
        rec["fixed_d"] = actual.bit_count()
        rec["words"] = sorted(set(rec["words"]), key=int.bit_count)[:20]

    def screen(rec, side):
        """Exact ranks and exclusion of all component words through four."""
        rank = gf2_rank(rec["rows"])
        if rank < len(rec["rows"]) - (12 if side == "A" else 18):
            return False
        if side == "A":
            # Faithful-six representation blocks can have larger deficits.
            # Actual full ranks, not this preference, decide the rate cell.
            if min(gf2_rank([row & mask for row in rec["rows"]]),
                   gf2_rank([row >> q for row in rec["rows"]])) < q-12:
                return False
        singles = set(rec["columns"])
        seen = set()
        for i, col in enumerate(rec["columns"]):
            for other in rec["columns"][:i]:
                value = col ^ other
                if value in singles or value in seen:
                    return False
                seen.add(value)
        rec["rank"] = rank
        return True

    def query(rec, side, trials, offset):
        """Retain fixed-subgroup and unrestricted genuine kernel witnesses."""
        fixed_challenge(rec, side, max(6, min(48, trials)),
                        offset+5000011)
        diagnostic = random.Random(seed + offset)
        columns = rec["columns"]
        order = list(range(len(columns)))
        retained = list(rec["words"])
        for trial in range(trials):
            diagnostic.shuffle(order)
            if side == "A" and retained and trial % 3 == 1:
                # Bias an information set toward a previously verified
                # word without changing any matrix coordinates.
                active = list(_mp_bits(diagnostic.choice(retained[:8])))
                front = diagnostic.sample(active, max(1, len(active)//2))
                selected = set(front)
                order = front + [v for v in order if v not in selected]
            pivots, basis = {}, []
            for v in order:
                value, word = columns[v], 1 << v
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
                    note(rec, word)
            if not basis:
                raise ValueError("component unexpectedly has no kernel")
            light = sorted(set(retained+basis), key=int.bit_count)[:16]
            for t, word in enumerate(basis):
                # A's half-rate kernel receives complete circuit-pair
                # cancellation, rather than short-vs-short cancellation.
                others = basis[:t] if side == "A" else light
                for other in others:
                    candidate = word ^ other
                    if candidate and candidate.bit_count() <= rec["d"]+2:
                        note(rec, candidate)
            moves = retained+basis
            diagnostic.shuffle(moves)
            for word in light[:6]:
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    if not changed:
                        break
                note(rec, word)
            rec["words"] = sorted(set(rec["words"]),
                                  key=int.bit_count)[:20]
            retained = rec["words"]
            if rec["d"] <= 8:
                break
        for word in rec["words"][:3]:
            assert _mp_syndrome(columns, word) == 0

    def merit(rec):
        multiplicity = sum(w.bit_count() == rec["d"] for w in rec["words"])
        return (rec["d"], -multiplicity, -rec["cost"], rec["rank"])

    def pool(side):
        offset = 100003 if side == "A" else 200003
        records, signatures = [], set()
        for _ in range(144 if side == "A" else 88):
            rec = inspect(fresh(side), side)
            if rec is None or rec["signature"] in signatures:
                continue
            signatures.add(rec["signature"])
            records.append(rec)
        records.sort(key=lambda r: r["cost"])
        survivors = []
        for rec in records:
            if not screen(rec, side):
                continue
            query(rec, side, 8 if side == "A" else 6, offset)
            survivors.append(rec)
            if len(survivors) == (16 if side == "A" else 8):
                break
        if not survivors:
            raise ValueError("no connected mixed component passed exact gates")
        survivors.sort(key=merit, reverse=True)
        finalists = survivors[:4]
        for rec in finalists:
            query(rec, side, 24 if side == "A" else 20, offset+700001)
        finalists.sort(key=merit, reverse=True)
        current = finalists[0]
        keepers = finalists[:]

        for step in range(28 if side == "A" else 16):
            grid = _copy_cells(current["grid"])
            light = current["words"][:8]
            word = rng.choice(light)
            occupied = sorted({v//q for v in _mp_bits(word)})
            j = rng.choice(occupied)
            active = [i for i in range(len(grid)) if grid[i][j]]
            i = rng.choice(active)
            if step % 5 == 0:
                grid[i][j] = cell_for(side, i, j)
            else:
                t = rng.randrange(len(grid[i][j]))
                old = G.elements[grid[i][j][t]]
                options = [G.idx((a, old[1])) for a in range(31)
                           if G.idx((a, old[1])) not in grid[i][j]]
                grid[i][j][t] = rng.choice(options)
                if not whole_sidon(grid[i][j]):
                    continue
                if side == "B" and len(grid[i][j]) == 2:
                    previous = current["grid"][i][j]
                    before = orders[
                        table[inverses[previous[0]]][previous[1]]]
                    after = orders[
                        table[inverses[grid[i][j][0]]][grid[i][j][1]]]
                    if after != before:
                        continue
            rec = inspect(grid, side)
            if rec is None or rec["signature"] in signatures:
                continue
            signatures.add(rec["signature"])
            if not screen(rec, side):
                continue
            # Historical words matter only if they remain genuine kernels.
            for known in current["words"]:
                if _mp_syndrome(rec["columns"], known) == 0:
                    note(rec, known)
            query(rec, side, 6, offset+1000003+step*7919)
            if merit(rec) <= merit(current):
                continue
            query(rec, side, 20, offset+1300007+step*7919)
            if merit(rec) > merit(current):
                current = rec
                keepers.append(rec)

        unique = {rec["signature"]: rec for rec in keepers}
        finalists = sorted(unique.values(), key=merit, reverse=True)[:4]
        for rec in finalists:
            query(rec, side, 64 if side == "A" else 40, offset+2700011)
        finalists.sort(key=merit, reverse=True)
        return finalists[:2]

    aa, bb = pool("A"), pool("B")

    def localized(hx, hz, a, b, n):
        """Verified nontrivial logicals, reduced with ORIGINAL stabilizers."""
        readings = []
        for kind, opposite, stabilizers, rec in (
                ("Z", hx, hz, a), ("X", hz, hx, b)):
            span = _mp_span(stabilizers)
            diagnostic = random.Random(seed + (3100009 if kind == "Z"
                                               else 4100011))
            candidates = []
            for word in rec["words"][:5]:
                for block in range(3 if kind == "Z" else 2):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask) << ((3*j+block)*q)
                            for j in range(2))
                    else:
                        actual = word << (3*block*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if _mp_nontrivial(actual, span):
                        candidates.append(actual)
            found = n+1
            for actual in sorted(candidates, key=int.bit_count)[:8]:
                reducers = list(stabilizers)
                diagnostic.shuffle(reducers)
                for sweep in range(3):
                    changed = False
                    for row in reducers:
                        candidate = actual ^ row
                        if candidate.bit_count() < actual.bit_count():
                            actual, changed = candidate, True
                    if not changed:
                        break
                neighborhood = sorted(
                    stabilizers,
                    key=lambda row: (row & actual).bit_count(),
                    reverse=True)[:40]
                for _ in range(160):
                    first, second = diagnostic.sample(neighborhood, 2)
                    candidate = actual ^ first ^ second
                    if candidate.bit_count() < actual.bit_count():
                        actual = candidate
                assert actual and _mp_nontrivial(actual, span)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                found = min(found, actual.bit_count())
            readings.append(found)
        return tuple(readings)

    best = None
    for a in aa:
        for b in bb:
            A = [[[G.elements[g] for g in cell] for cell in row]
                 for row in a["grid"]]
            B = [[[G.elements[g] for g in cell] for cell in row]
                 for row in b["grid"]]
            hx, hz, n = lifted_matrices(P, a["grid"], b["grid"])
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 1488 and k >= 186
                    and 125*n <= 1000*k < 175*n):
                continue
            assert _component_rows(hx, hz, q, "A") == a["rows"]
            assert _component_rows(hx, hz, q, "B") == b["rows"]
            logicals = localized(hx, hz, a, b, n)
            detected = min(a["d"], b["d"], *logicals)
            key = (detected, k*detected*detected,
                   a["d"]+b["d"], -a["cost"]-b["cost"])
            if best is None or key > best[0]:
                best = (key, A, B, hx, hz, k,
                        (a["d"], b["d"]), logicals,
                        (a["fixed_d"], b["fixed_d"]),
                        (a["normal_fixed_reading"],
                         b["normal_fixed_reading"]),
                        (a["normal_fixed_exact"], b["normal_fixed_exact"]))
    if best is None:
        raise ValueError("involution-free faithful-six pairs missed cell 0.150")
    (key, A, B, cached_hx, cached_hz, k, components,
     logicals, fixed, normal, normal_exact) = best
    hx, hz, n = lifted_matrices(G, A, B)
    assert hx == cached_hx and hz == cached_hz and n == 1488
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx+hz) <= 12
    assert [[len(cell) for cell in row] for row in A] == sa
    assert [[len(cell) for cell in row] for row in B] == sb
    assert all(len(set(cell)) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    return {"idx": idx, "card": (mode, topology, cohort),
            "cells": {"A": A, "B": B}, "screen_k": k,
            "component_readings": components,
            "complement_fixed_upper_readings": fixed,
            "normal_fixed_detected_readings": normal,
            "normal_fixed_enumeration_complete": normal_exact,
            "verified_localized_logical_readings": logicals,
            "quality": key}


def _legacy_cubic_ctx(rng, G, idx):
    """Unused historical two-row optimizer; NEW_CTX above replaces it."""
    design = idx % 20
    sa = _cubic_support(rng, A_SHAPE[1], design % 4, 7)
    sb = _cubic_support(rng, B_SHAPE[1], design // 4, 9)
    assert _grid_weights(sa, sb) == 12
    bits_a = _coset_pattern(rng, sa)
    bits_b = _coset_pattern(rng, sb)
    penalty = 320 + 40 * (idx // 20)
    fallback = None

    for attempt in range(5):
        starts = []
        for _ in range(2):
            A = _initial_cubic_cells(rng, G, sa, bits_a)
            B = _initial_cubic_cells(rng, G, sb, bits_b)
            cost = sum(_verified_costs(G, A, B, penalty))
            starts.append((cost, A, B))
        _, A, B = min(starts, key=lambda item: item[0])
        A, B, cost = _optimize_labels(rng, G, A, B, penalty)

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 0.325 * n <= k < 0.375 * n):
            continue
        ar = _component_rows(hx, hz, G.q, "A")
        br = _component_rows(hx, hz, G.q, "B")
        short = (_small_kernel_penalty(ar, A_SHAPE[1] * G.q) +
                 _small_kernel_penalty(br, B_SHAPE[1] * G.q))
        key = (short, cost)
        if fallback is None or key < fallback[0]:
            fallback = (key, A, B, k)
        if short == 0:
            break

    if fallback is None:
        raise ValueError("no exact-rank candidate in target rate band")
    key, A, B, k = fallback
    hx, hz, _ = lifted_matrices(G, A, B)
    if not css_ok(hx, hz):
        raise ValueError("verified construction failed CSS check")
    assert max(row.bit_count() for row in hx + hz) <= 12
    return {"idx": idx, "cells": {"A": A, "B": B},
            "screen_k": k, "structural_penalty": key}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected equal-component quartic cell."""
    cell = list(ctx["cells"][side][i][j])
    assert len(cell) <= 5
    assert len({G.idx(g) for g in cell}) == len(cell)
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolling and uniform labels alone suppress very short logicals in the order-52 (2,6)x(2,4) bed: all five measured parent codes had distance 4, with many sector-1 witnesses.
# REFUTED: Adding a third cubic A check-block row on C15:C2 (3,7)x(2,6) produces five measured distances above ten: the measured selections instead returned 8,8,10,8,8, with minimum witnesses entirely in sector 1 and fitness 24.0532.
# REFUTED: Zero-cell cubic B cards supply competitive screening readings >=50 on C61:C3 (1,2)x(2,3): all 36 current zero-cell slots returned d2s<=42, while all five measured winners had positive B cells and d2s=50..53.
# REFUTED: Positive cubic B supports and complement-fixed challenges make a short reading >=51 exclude logicals below44 on C61:C3 (1,2)x(2,3): current slot 52 fell from 51 to 42, with a sector-one Z witness crossing both Acols at Bcol1.
# REFUTED: Arbitrarily shuffled six-fiber A occupancies are uniformly feasible under bidirectional Sidon on faithful C31:C6: the shown order-186 program voided 36 of 60 slots with six-fiber Sidon occupancy failures and measured fitness250.0050.
# REFUTED: Involution-free occupancy repair and normal-fixed challenges make a screening reading53 exclude logicals below44 on C31:C6 (1,2)x(2,3): current slot30 measured35, crossing both Acols at Bcol1; the five-code score was231.7998.
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

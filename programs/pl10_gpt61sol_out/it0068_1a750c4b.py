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
CONCEPT = ("On metacyclic Z7:Z2(s=6), order 14, shapes (1,4)x(2,7), strengthening B to degree-two/three columns and making its doubleton directions reflection-heavy should reduce circuit failures by separating ancilla-spread pairs from A's short rotation cycles, despite permitting controlled weight-three A logicals.")
GROUP = ("Keep the frozen order-14 group and multiplication rule byte-for-byte. This bed scored 3.2001 and the related (2,4)x(2,5) bed scored 4.7; neither reading isolates a group defect. This L1 experiment changes filling only.")
SHAPES = ("Keep (1,4)x(2,7): n=420, kb=210, guaranteed rate 0.500, component matrices 14x56 and 28x98; the size factor permits q<=19 but the frozen group remains q=14. Parent weight-four witnesses were entirely sector-1 Acol/Bcol operators. Failures affected all 213 logicals, with mostly 1.0-1.2x block enrichment and no consistent Brow hotspot. Require exact k=210 rather than rewarding rank deficits.")
FILLING = ("Twenty cards cross four B support architectures with five relative-direction policies. A counts [2,2,2,3] use three distinct rotation binomials and a mixed trinomial. B has eighteen terms, nine per row, and column totals [2,2,2,3,3,3,3]; paired zero cells distinguish the architectures. HX weights are 11/12 and HZ weights 11/12, giving 14364 six-round CNOTs versus the parent's 13440. Start with a Fano-like three-column monomial core; broaden its voltage geometry only if needed. Verified lifts screen rank and exact stabilizer-aware logicals through weight four, prioritizing X weight-three removal and shared-check exposure. A weighted full-Tanner four-cycle invariant assigns each slot a different residue, proving within-program nonisomorphism without manufacturing automorphism copies. These are pL hypotheses: the parent still failed 107-133/250 shots despite almost universal first-pass settlement.")
FAMILY_NAME = "r050_m7r2s6_14x27_reflection_orthogonal_B3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 14

def custom_ops(q):
    # metacyclic Z_7 x| Z_2: y x y^-1 = x^6 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 7, 2, 6
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def _distinct_terms(rng, G, count):
    terms = []
    seen = set()
    while len(terms) < count:
        g = G.pick(rng)
        key = G._key(g)
        if key not in seen:
            seen.add(key)
            terms.append(g)
    return terms


def _b_profile(rng, extra, zeros):
    """Seven live columns; fixed total terms and fixed zero-cell count."""
    for _ in range(2000):
        triple = set(rng.sample(range(7), extra))
        sparse = set(rng.sample(range(7), zeros))
        sb = [[0] * 7 for _ in range(2)]
        for j in range(7):
            degree = 3 if j in triple else 2
            if j in sparse:
                a, b = 0, degree
            else:
                a, b = 1, degree - 1
            if rng.randrange(2):
                a, b = b, a
            sb[0][j], sb[1][j] = a, b
        if 0 < sum(sb[0]) <= 9 and 0 < sum(sb[1]) <= 9:
            return sb
    raise ValueError("could not realize deliberate B profile")


def _syndrome_risk(rows, n):
    """Ambiguity counts, not a distance certificate or circuit score."""
    columns = [0] * n
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= check
            row ^= bit
    if 0 in columns or len(set(columns)) != n:
        return None
    singles = set(columns)
    pairs = {}
    triples = 0
    for a in range(n):
        x = columns[a]
        for b in range(a):
            syndrome = x ^ columns[b]
            triples += syndrome in singles
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    collisions = sum(c * (c - 1) // 2 for c in pairs.values())
    return triples // 3, collisions


def _component_class_reference(rng, G, idx):
    """Unused reference model; NEW_CTX below builds the reinforced design."""
    from itertools import combinations

    q = G.q
    els = G.elements
    e = G.e
    mask = (1 << q) - 1
    mul = [[G.idx(G.mul(g, h)) for h in els] for g in els]
    autos = [
        [G.idx(((u * g[0] + v * g[1]) % 7, g[1]))
         for g in els]
        for u in range(1, 7) for v in range(7)
    ]

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def four_cycles(cols):
        overlaps = {}
        for syndrome in cols:
            checks = []
            while syndrome:
                bit = syndrome & -syndrome
                checks.append(bit.bit_length() - 1)
                syndrome ^= bit
            for pair in combinations(checks, 2):
                overlaps[pair] = overlaps.get(pair, 0) + 1
        return sum(t * (t - 1) // 2 for t in overlaps.values())

    def component_fours(cols, weights):
        """Exact weight-four words; ordered pairs count each support once."""
        pairs = {}
        count = exposure = 0
        participation = [0] * len(cols)
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                syndrome = cols[a] ^ cols[b]
                bucket = pairs.setdefault(syndrome, [])
                for c, d in bucket:
                    if d < a:
                        count += 1
                        ids = (c, d, a, b)
                        exposure += sum(weights[x] * weights[y]
                                        for x, y in combinations(ids, 2))
                        for x in ids:
                            participation[x] += 1
                bucket.append((a, b))
        return exposure, count, max(participation)

    # Right translations are column gauges. Enumerate their trinomial
    # orbits rather than repeatedly drawing relabeled copies.
    tri_type = {}
    representatives = []
    for triple in combinations(range(q), 3):
        if triple in tri_type:
            continue
        orbit = {
            tuple(sorted(mul[g][h] for g in triple))
            for h in range(q)
        }
        t = len(representatives)
        representatives.append(min(orbit))
        for image in orbit:
            tri_type[image] = t

    type_actions = [
        [tri_type[tuple(sorted(aut[g] for g in triple))]
         for triple in representatives]
        for aut in autos
    ]

    # Obtain every component column from the verified full construction.
    # Only extraction/concatenation is performed here, not a separate lift.
    tri_columns = []
    B_probe = [[[e] for _ in range(7)] for _ in range(2)]
    unit_columns = None
    for triple in representatives:
        A_probe = [[[e], [els[g] for g in triple], [e], [e]]]
        hx, hz, n = lifted_matrices(G, A_probe, B_probe)
        ar = [((hx[t] >> (7 * q)) & mask) for t in range(q)]
        tri_columns.append(columns(ar, q))
        if unit_columns is None:
            unit_columns = columns([hx[t] & mask for t in range(q)], q)

    A_book = []
    seen_A = set()
    for triple_types in combinations(range(len(representatives)), 3):
        canonical = min(
            tuple(sorted(action[t] for t in triple_types))
            for action in type_actions
        )
        if canonical in seen_A:
            continue
        seen_A.add(canonical)
        ac = list(unit_columns)
        for t in canonical:
            ac.extend(tri_columns[t])
        assert len(set(ac)) == 4 * q and all(s.bit_count() % 2 for s in ac)
        # Sector-1 gate degrees are 3 for the unit block and 5 otherwise.
        risk = component_fours(ac, [3] * q + [5] * (3 * q))
        A_book.append((risk + (four_cycles(ac),), canonical))
    A_book.sort()

    # Classify B subsets under left translation and group automorphisms.
    # These generators also cover right translation through conjugation.
    generators = [
        mul[G.idx((1, 0))],
        mul[G.idx((0, 1))],
        autos[14],                 # u=3, v=0
        autos[1],                  # u=1, v=1
    ]

    def image_mask(bits, permutation):
        image = 0
        while bits:
            bit = bits & -bits
            image |= 1 << permutation[bit.bit_length() - 1]
            bits ^= bit
        return image

    unseen = {sum(1 << g for g in subset)
              for subset in combinations(range(q), 7)}
    B_book = []
    A_probe = [[[e] for _ in range(4)]]
    bmask = (1 << (7 * q)) - 1
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        orbit = [seed]
        todo = [seed]
        while todo:
            current = todo.pop()
            for permutation in generators:
                image = image_mask(current, permutation)
                if image in unseen:
                    unseen.remove(image)
                    orbit.append(image)
                    todo.append(image)
        canonical = min(orbit)
        labels = tuple(g for g in range(q) if canonical & (1 << g))
        B_probe = [
            [[e] for _ in range(7)],
            [[els[g]] for g in labels],
        ]
        hx, hz, n = lifted_matrices(G, A_probe, B_probe)
        bc = columns([row & bmask for row in hz[:2 * q]], 7 * q)
        assert len(set(bc)) == 7 * q and all(s.bit_count() == 2 for s in bc)
        risk = component_fours(bc, [1] * (7 * q))
        B_book.append(((risk[1], risk[2]), labels))
    B_book.sort()

    # Twenty slots use four strong A classes crossed with five B classes.
    # Larger harnesses continue to additional A classes, not orbit copies.
    nb = min(5, len(B_book))
    if not nb or idx >= len(A_book) * nb:
        raise ValueError("distinct component design book exhausted")
    a_types = A_book[idx // nb][1]
    b_labels = B_book[idx % nb][1]
    A = [[[e]] + [[els[g] for g in representatives[t]] for t in a_types]]
    assert _grid_weights([[1, 3, 3, 3]], [[1] * 7, [1] * 7]) <= W_MAX

    def span(rows):
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

    def logical_risk(check_cols, stab_cols, stab_rows, degrees):
        """Exact nontrivial logicals through weight four, plus hook proxy."""
        pivots = span(stab_rows)

        def nontrivial(v):
            while v:
                p = v.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    return True
                v ^= old
            return False

        width = len(check_cols)
        bits = [1 << a for a in range(width)]
        singles = {}
        counts = [0, 0, 0, 0]
        hooks = exposure = 0
        for a, syndrome in enumerate(check_cols):
            bucket = singles.setdefault(syndrome, [])
            if syndrome == 0 and nontrivial(bits[a]):
                counts[0] += 1
            for b in bucket:
                if nontrivial(bits[a] | bits[b]):
                    counts[1] += 1
            bucket.append(a)

        pairs = {}
        for a in range(width):
            for b in range(a + 1, width):
                syndrome = check_cols[a] ^ check_cols[b]
                support = bits[a] | bits[b]
                for c in singles.get(syndrome, ()):
                    if c > b and nontrivial(support | bits[c]):
                        counts[2] += 1
                bucket = pairs.setdefault(syndrome, [])
                for c, d, previous in bucket:
                    if d >= a or not nontrivial(previous | support):
                        continue
                    counts[3] += 1
                    ids = (c, d, a, b)
                    # Common same-error-basis checks can propagate pairs.
                    # This is pessimistic: the actual ASC ordering is not
                    # available here, so it is not a circuit certificate.
                    for x, y in combinations(ids, 2):
                        hooks += (stab_cols[x] & stab_cols[y]).bit_count()
                        exposure += degrees[x] * degrees[y]
                bucket.append((a, b, support))
        return tuple(counts), hooks, exposure

    order = list(range(len(autos)))
    rng.shuffle(order)
    order = [0] + [t for t in order if t != 0][:11]
    best = None
    best_key = None
    for t in order:
        aut = autos[t]
        B = [
            [[e] for _ in range(7)],
            [[els[aut[g]]] for g in b_labels],
        ]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 0.475 <= k / n < 0.525):
            continue
        xc, zc = columns(hx, n), columns(hz, n)
        degrees = [x.bit_count() + z.bit_count() for x, z in zip(xc, zc)]
        zrisk = logical_risk(xc, zc, hz, degrees)
        xrisk = logical_risk(zc, xc, hx, degrees)
        if any(zrisk[0][:3]) or any(xrisk[0][:3]):
            continue
        cx, cz = four_cycles(xc), four_cycles(zc)
        key = (
            max(xrisk[1], zrisk[1]), xrisk[1] + zrisk[1],
            max(xrisk[2], zrisk[2]), xrisk[2] + zrisk[2],
            max(xrisk[0][3], zrisk[0][3]),
            xrisk[0][3] + zrisk[0][3],
            max(cx / len(hx), cz / len(hz)), cx + cz,
            k - 210,
        )
        if best_key is None or key < best_key:
            best_key = key
            best = {"A": A, "B": B}
    if best is None:
        raise ValueError("component brief failed verified short-logical screen")
    hx, hz, n = lifted_matrices(G, best["A"], best["B"])
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    return best


def NEW_CTX(rng, G, idx):
    """Reinforce B, screen actual CSS logicals, and enforce graph diversity."""
    from itertools import combinations
    from math import isqrt

    card = idx % 20
    architecture, policy = divmod(card, 5)
    q = G.q
    e = G.e

    # Every card has A total 9, B row totals 9, and B columns <=3.
    # Architectures introduce balanced pairs of zero cells without
    # changing either row budget or the seven column totals.
    sa = [[2, 2, 2, 3]]
    sb = [[1, 1, 1, 2, 2, 1, 1],
          [1, 1, 1, 1, 1, 2, 2]]
    if architecture in (1, 3):
        sb[0][0], sb[1][0] = 2, 0
        sb[0][1], sb[1][1] = 0, 2
    if architecture in (2, 3):
        sb[0][3], sb[1][3] = 3, 0
        sb[0][5], sb[1][5] = 0, 3
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    # Different residues of an isomorphism invariant guarantee that
    # no two final slots are relabelings of the same Tanner graph.
    # This does not claim to know the evaluator's historical archive.
    modulus = max(23, N_SLOTS + 1)
    while any(modulus % d == 0
              for d in range(2, isqrt(modulus) + 1)):
        modulus += 1
    target = idx

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            marker = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= marker
                row ^= bit
        return out

    def graph_tag(rows):
        """Invariant of the combined X/Z check Tanner graph.

        Participation counts include every four-cycle incident on each
        check. Weighting by check degree and summing a cubic retains more
        structural information than the total cycle count alone.
        Floor division is only a deterministic hash, not an assumption
        that a nonabelian lifted product has a free global group action.
        """
        participation = [0] * len(rows)
        for a, row in enumerate(rows):
            for b in range(a):
                overlap = (row & rows[b]).bit_count()
                cycles = overlap * (overlap - 1) // 2
                participation[a] += cycles
                participation[b] += cycles
        return sum((row.bit_count() + 3) * (v + 1) ** 3
                   for row, v in zip(rows, participation)) // q

    def component_profile(cols):
        """Exact weight-three/four kernel counts when columns are distinct."""
        if 0 in cols or len(set(cols)) != len(cols):
            return None
        singles = set(cols)
        pairs = {}
        triples = collisions = 0
        for a, x in enumerate(cols):
            for b in range(a):
                syndrome = x ^ cols[b]
                triples += int(syndrome in singles)
                previous = pairs.get(syndrome, 0)
                collisions += previous
                pairs[syndrome] = previous + 1
        return triples // 3, collisions // 3

    def draw(trial):
        # The three rotation steps exhaust the distinct undirected
        # order-seven binomial directions, avoiding parallel A columns.
        # A mixed trinomial connects the two rotation cosets.
        A = [[[e, (1, 0)], [e, (2, 0)], [e, (3, 0)],
              [e, (rng.randrange(1, 7), 0), (rng.randrange(7), 1)]]]

        relaxation = trial // 1024
        if relaxation == 0:
            core = [(0, 0), (1, 0), (3, 0)]
        elif relaxation == 1:
            core = rng.sample([(a, 0) for a in range(7)], 3)
        else:
            core = rng.sample(G.elements, 3)
        remaining = [g for g in G.elements if g not in core]
        voltages = core + rng.sample(remaining, 4)
        B = [[[] for _ in range(7)] for _ in range(2)]

        for i in range(2):
            for j in range(7):
                count = sb[i][j]
                if not count:
                    continue
                base = e if i == 0 else voltages[j]
                if count == 1:
                    B[i][j] = [base]
                    continue

                if sb[1 - i][j] == 0:
                    # An isolated degree-two column must not have an
                    # involutory ratio: that would duplicate syndromes.
                    step = rng.choice((1, 2, 3))
                    if count == 2:
                        B[i][j] = [e, (step, 0)]
                    elif policy == 0:
                        offsets = rng.choice(((0, 1, 2), (0, 1, 3)))
                        B[i][j] = [((step * a) % 7, 0)
                                   for a in offsets]
                    else:
                        B[i][j] = [e, (step, 0), (rng.randrange(7), 1)]
                    continue

                # Reflection-heavy doubletons are intended to avoid
                # matching the rotation differences in A's triangles.
                if policy == 0:
                    reflected = False
                elif policy == 1:
                    reflected = True
                elif policy == 2:
                    reflected = i == 0
                elif policy == 3:
                    reflected = bool((i + j) % 2)
                else:
                    reflected = True
                if relaxation >= 3:
                    reflected = rng.randrange(6) <= policy

                if reflected:
                    axis = j % 7 if policy == 4 and relaxation < 3 \
                        else rng.randrange(7)
                    direction = (axis, 1)
                else:
                    direction = (rng.choice((1, 2, 3)), 0)
                B[i][j] = [base, G.mul(base, direction)]
        return A, B

    def residues(stabilizers, n):
        """Linear normal forms modulo the actual stabilizer row space."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        bits = [1 << c for c in range(n)]
        out = list(bits)
        for p in sorted(pivots):
            tail = pivots[p] ^ bits[p]
            remainder = 0
            while tail:
                bit = tail & -tail
                remainder ^= out[bit.bit_length() - 1]
                tail ^= bit
            out[p] = remainder
        return out

    def logical_profile(checks, stabilizers, degrees, n):
        """Exact logical supports through weight four and exposure proxies.

        Shared opposite-basis checks mark possible spread pairs; their
        realization by the ASC gate order is not assumed. Sorted-support
        partitions count each triple and quartet exactly once.
        """
        syndrome = columns(checks, n)
        shared = columns(stabilizers, n)
        normal = residues(stabilizers, n)
        singles = {}
        counts = [0, 0, 0, 0]
        hook3 = hook4 = exposure = 0
        hits = [0] * n

        def record(ids):
            nonlocal exposure
            for a, b in combinations(ids, 2):
                exposure += degrees[a] * degrees[b]
            for a in ids:
                hits[a] += 1

        for a, s in enumerate(syndrome):
            bucket = singles.setdefault(s, [])
            if s == 0 and normal[a]:
                counts[0] += 1
            for b in bucket:
                if normal[a] ^ normal[b]:
                    counts[1] += 1
            bucket.append(a)

        pairs = {}
        for a in range(n):
            for b in range(a + 1, n):
                s = syndrome[a] ^ syndrome[b]
                remainder = normal[a] ^ normal[b]
                for c in singles.get(s, ()):
                    if c > b and remainder ^ normal[c]:
                        counts[2] += 1
                        ids = (a, b, c)
                        hook3 += sum((shared[x] & shared[y]).bit_count()
                                     for x, y in combinations(ids, 2))
                        record(ids)
                bucket = pairs.setdefault(s, [])
                for c, d, previous in bucket:
                    if d < a and previous ^ remainder:
                        counts[3] += 1
                        ids = (c, d, a, b)
                        hook4 += sum((shared[x] & shared[y]).bit_count()
                                     for x, y in combinations(ids, 2))
                        record(ids)
                bucket.append((a, b, remainder))
        return tuple(counts), hook3, hook4, exposure, max(hits)

    pool = []
    seen = set()
    qualified = 0
    bmask = (1 << (7 * q)) - 1
    for trial in range(4096):
        A, B = draw(trial)
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for grid in (A, B) for row in grid for cell in row)
        if signature in seen:
            continue
        seen.add(signature)
        hx, hz, n = lifted_matrices(G, A, B)

        # Extract B directly from verified HZ, retaining its right action.
        br = [row & bmask for row in hz[:2 * q]]
        bp = component_profile(columns(br, 7 * q))
        if bp is None:
            continue
        tag = graph_tag(hx + hz)
        if tag % modulus != target:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 210:
            continue

        pool.append((bp, trial, A, B, hx, hz, n, tag))
        pool.sort(key=lambda item: (item[0], item[1]))
        del pool[8:]
        qualified += 1
        if qualified >= 24:
            break

    best = None
    for coarse, trial, A, B, hx, hz, n, tag in pool:
        cx, cz = columns(hx, n), columns(hz, n)
        degrees = [x.bit_count() + z.bit_count()
                   for x, z in zip(cx, cz)]
        px = logical_profile(hz, hx, degrees, n)
        pz = logical_profile(hx, hz, degrees, n)
        if any(px[0][:2]) or any(pz[0][:2]):
            continue
        # A permits controlled triangles. First avoid making B equally
        # vulnerable, then minimize short-logical shared-check exposure.
        score = (
            px[0][2],
            max(px[1], pz[1]), px[1] + pz[1],
            max(px[2], pz[2]), px[2] + pz[2],
            max(px[3], pz[3]), px[3] + pz[3],
            max(px[0][3], pz[0][3]), px[0][3] + pz[0][3],
            max(px[4], pz[4]),
        )
        if best is None or score < best[0]:
            best = (score, A, B, hx, hz, tag, px, pz)

    if best is None:
        raise ValueError("no rank-210, graph-quota filling passed logical screening")
    score, A, B, hx, hz, tag, px, pz = best
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert tag % modulus == idx
    return {"A": A, "B": B, "card": card, "k": 210,
            "graph_tag": tag, "graph_modulus": modulus,
            "logical_profile": (px, pz), "screen": score}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete reinforced, screened cell without new draws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone did not give reliable memory on the parent D14 (6,2)x(6,2) bed: fitness 3.4511, ubiquitous weight-four sector-2 witnesses, and frequent failures in both bases.
# REFUTED: First-pass syndrome settlement certifies reliable memory on the immediate parent: slot 0 X settled 246/4/0 yet failed 120/250 shots, affecting all 213 logical qubits.
# REFUTED: A catalog of component translation/automorphism classes guarantees many inequivalent full codes: the current parent produced only six distinct codes from sixty slots.
# REFUTED: Removing weight-three logicals and minimizing weight-four hook surrogates suffices for reliable q14 memory: the current parent scored 3.2001 and its measured codes still failed 107-133/250 shots with failures spread over all 213 logicals.
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

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
CONCEPT = ("Cyclic C23, order 23, A=B=(1,4): enumerated unit-pivot/inverse-distinct-binomial pairs should supply at least twenty fresh, intrinsically separated high-rate designs with certified distance three. The five-code target is 81/17, matching the present distance-three high-rate benchmark; strict improvement is not asserted.")
GROUP = ("Keep the cyclic construction and order 23. The new factor seventeen gives n=391, inside the size window without an order change. The live archive lists no C23 (1,4)x(1,4) bed. The parent's distance-23 readings belong to a different, low-rate constituent mechanism and are not transferred.")
SHAPES = ("Change both sides from (3,4) to (1,4). Each constituent has 23 checks and 92 coordinates; n=391, kb=207 and guaranteed rate 9/17=0.529412, strictly inside open-ended cell 0.550. Unit pivots make both complete check matrices full row rank, hence k=207 exactly. Parent minima occupied sector-1 Acol/Bcol blocks broadly; measured slots 12 and 13 also exposed lighter mixed-sector representatives.")
FILLING = ("Both supports are [1,2,2,2], so each constituent row has seven terms and BOTH CSS maxima are nine. Exhaustively enumerate inverse-distinct cyclic direction triples and remove common-automorphism and side-swap copies before building. An immutable portfolio conservatively rejects equal intrinsic full-Tanner signatures, including degree-conditioned rooted spheres. Every published lift verifies actual ranks, CSS, complete commuting-kernel exclusion through weight two and an explicit non-stabilizer weight-three witness. These establish distance three, not a larger heuristic reading; novelty against the unseen archive remains evaluator-controlled.")
FAMILY_NAME = "r055_C23_14x14_unitbinomial_certified3_portfolio_v4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 23

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C23 requires order 23")
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def mul(g, h):
        return ((g[0] + h[0]) % q,)

    def inv(g):
        return ((-g[0]) % q,)

    def idx(g):
        return int(g[0]) % q

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


def _ev_columns(rows, n):
    """Column syndromes of an already constructed binary matrix."""
    cols = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _ev_geometry(rows, n):
    """Exact short kernel collisions and component 4/6-cycle counts."""
    cols = _ev_columns(rows, n)
    seen = {0}
    hits = 0
    for s in cols:
        if s in seen:
            hits += 1
        seen.add(s)
    for i, x in enumerate(cols):
        for j in range(i):
            s = x ^ cols[j]
            if s in seen:
                hits += 1
            seen.add(s)

    # Edge labels here are coordinate masks shared by two checks.
    edges = {}
    adj = [set() for _ in rows]
    for j, s in enumerate(cols):
        incident = []
        while s:
            bit = s & -s
            incident.append(bit.bit_length() - 1)
            s ^= bit
        for a, u in enumerate(incident):
            for v in incident[a + 1:]:
                key = (u, v)
                edges[key] = edges.get(key, 0) | (1 << j)
                adj[u].add(v)
                adj[v].add(u)

    c4 = sum(m.bit_count() * (m.bit_count() - 1) // 2
             for m in edges.values())
    c6 = 0
    for u in range(len(rows)):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[u].intersection(adj[v]):
                if w <= v:
                    continue
                a, b, c = edges[u, v], edges[u, w], edges[v, w]
                na, nb, nc = a.bit_count(), b.bit_count(), c.bit_count()
                # Three distinct coordinates are required for a 6-cycle.
                c6 += (na * nb * nc
                       - (a & b).bit_count() * nc
                       - (a & c).bit_count() * nb
                       - (b & c).bit_count() * na
                       + 2 * (a & b & c).bit_count())
    return hits, c4, c6


def _ev_lp4_bad(checks, stabilizers, n):
    """Find any nontrivial logical of weight <=4 in the exact LP lift."""
    basis = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p in basis:
                row ^= basis[p]
            else:
                basis[p] = row
                break

    def nontrivial(row):
        while row:
            p = row.bit_length() - 1
            if p not in basis:
                return True
            row ^= basis[p]
        return False

    cols = _ev_columns(checks, n)
    seen = {0: 0}
    # Every weight <=4 vector is a difference of two weight <=2 masks.
    # One representative per syndrome suffices modulo stabilizer space.
    for i, s in enumerate(cols):
        mask = 1 << i
        if s in seen:
            if nontrivial(mask ^ seen[s]):
                return True
        else:
            seen[s] = mask
    for i, x in enumerate(cols):
        bit = 1 << i
        for j in range(i):
            s = x ^ cols[j]
            mask = bit | (1 << j)
            if s in seen:
                if nontrivial(mask ^ seen[s]):
                    return True
            else:
                seen[s] = mask
    return False


def _legacy_C23_saturated_ctx(rng, G, idx):
    """Inactive reference for the old shapes; never publishes cells."""
    from itertools import permutations
    try:
        import numpy as np
    except ImportError:
        np = None

    q, width = G.q, 4 * G.q
    maskq = (1 << q) - 1
    card, epoch = idx % 20, idx // 20

    # All three structures have row totals five, column totals
    # (3,4,4,4), and augmentation rank one. The even submatrix
    # consists respectively of a six-cycle, a four-cycle plus a
    # doubled edge, or three doubled edges: not permutation copies.
    profiles = (
        ((1, 0, 2, 2), (1, 2, 0, 2), (1, 2, 2, 0)),
        ((1, 4, 0, 0), (1, 0, 2, 2), (1, 0, 2, 2)),
        ((1, 4, 0, 0), (1, 0, 4, 0), (1, 0, 0, 4)),
    )
    pairs = tuple((a, b) for a in range(3) for b in range(a, 3))
    cards = tuple((a, b, style) for style in range(3)
                  for a, b in pairs) + ((0, 0, 3), (0, 1, 3))
    pa, pb, mode = cards[card]
    sa, sb = profiles[pa], profiles[pb]
    assert _grid_weights(sa, sb) == 9
    assert _grid_weights(sa, sb) <= min(W_MAX, 9)
    assert _no_dead_lines(sa, sb)

    def draw_component(side):
        support = sa if side == "A" else sb
        C = [[[] for _ in range(4)] for _ in range(3)]
        first_binomial = True
        for i in range(3):
            # Row translations gauge-fix the monomial first column.
            C[i][0] = [G.e]
            used_gaps = set()
            for j in range(1, 4):
                count = support[i][j]
                if not count:
                    continue
                if count == 2:
                    available = [d for d in range(1, q)
                                 if min(d, q - d) not in used_gaps]
                    if mode == 3:
                        short = [d for d in available
                                 if min(d, q - d) <= 5]
                        if short:
                            available = short
                    if mode == 1 and first_binomial:
                        desired = (1 if side == "A" else
                                   2 + (card + epoch) % (q - 2))
                        gap = desired if desired in available else rng.choice(available)
                    else:
                        gap = rng.choice(available)
                    first_binomial = False
                    used_gaps.add(min(gap, q - gap))
                    anchor = rng.randrange(q)
                    labels = (anchor, (anchor + gap) % q)
                else:
                    # Sidon quartets remove within-cell repeated
                    # differences; the unrestricted card tests whether
                    # that restriction sacrifices kernel quality.
                    for attempt in range(256):
                        labels = rng.sample(range(q), 4)
                        differences = [
                            (a - b) % q for a in labels for b in labels
                            if a != b]
                        if mode == 0 or len(set(differences)) == 12:
                            break
                    else:
                        raise ValueError("failed to draw a Sidon quartet")
                C[i][j] = [G.elements[a] for a in labels]
                assert len({G._key(g) for g in C[i][j]}) == count
        return C

    def verified_component(C, side):
        # Zero auxiliary blocks isolate the exact scaffold component.
        if side == "A":
            rows, _, _ = lifted_matrices(G, C, [[[]]])
        else:
            _, rows, _ = lifted_matrices(G, [[[]]], C)
        return [row & ((1 << width) - 1) for row in rows]

    def polynomial_product(a, b):
        value = 0
        while a:
            bit = a & -a
            shift = bit.bit_length() - 1
            value ^= ((b << shift) | (b >> (q - shift))) & maskq
            a ^= bit
        return value

    def rotate_word(word, shift):
        value = 0
        for j in range(4):
            block = (word >> (j * q)) & maskq
            block = ((block << shift) |
                     (block >> (q - shift))) & maskq
            value |= block << (j * q)
        return value

    def cofactor_probe(rows):
        """Actual kernel witnesses derived from VERIFIED column images."""
        cols = _ev_columns(rows, width)
        polynomials = [
            [(cols[j * q] >> (i * q)) & maskq for j in range(4)]
            for i in range(3)]
        word = 0
        for omitted in range(4):
            remaining = [j for j in range(4) if j != omitted]
            minor = 0
            for perm in permutations(remaining):
                product = polynomial_product(
                    polynomials[0][perm[0]], polynomials[1][perm[1]])
                minor ^= polynomial_product(product, polynomials[2][perm[2]])
            word |= minor << (omitted * q)
        # An even column annihilates its complete-group support.
        # This explicit word establishes ceiling 23 and seeds every
        # probe, including when the cofactor vanishes.
        witness = maskq << q
        best = witness.bit_count()
        assert all((row & witness).bit_count() % 2 == 0 for row in rows)
        if not word:
            return best, witness
        assert all((row & word).bit_count() % 2 == 0 for row in rows)
        if word.bit_count() < best:
            best, witness = word.bit_count(), word

        orbit = [rotate_word(word, s) for s in range(q)]
        # Translation symmetry makes one anchored representative
        # sufficient for all cofactor pairs and triples.
        for a in range(1, q):
            pair = word ^ orbit[a]
            if pair and pair.bit_count() < best:
                best, witness = pair.bit_count(), pair
            for b in range(1, a):
                triple = pair ^ orbit[b]
                if triple and triple.bit_count() < best:
                    best, witness = triple.bit_count(), triple
        return best, witness

    def kernel_basis(rows, order=None):
        """Complete binary kernel basis, returned in original coordinates."""
        if order is None:
            order = list(range(width))
        image = [0] * width
        for position, original in enumerate(order):
            image[original] = 1 << position
        pivots = {}
        for original_row in rows:
            row = 0
            while original_row:
                bit = original_row & -original_row
                row ^= image[bit.bit_length() - 1]
                original_row ^= bit
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        ascending = sorted(pivots)
        basis = []
        for free in range(width):
            if free in pivots:
                continue
            word = 1 << free
            for p in ascending:
                if (pivots[p] & word).bit_count() & 1:
                    word ^= 1 << p
            original_word = 0
            while word:
                bit = word & -word
                original_word |= 1 << order[bit.bit_length() - 1]
                word ^= bit
            basis.append(original_word)
        return basis

    def information_probe(record, passes):
        """Find explicit words; a high probe reading proves nothing."""
        best, witness = record["upper"], record["witness"]
        for _ in range(passes):
            order = list(range(width))
            rng.shuffle(order)
            basis = kernel_basis(record["rows"], order)
            for i, first in enumerate(basis):
                weight = first.bit_count()
                if weight < best:
                    best, witness = weight, first
                for j in range(i):
                    pair = first ^ basis[j]
                    weight = pair.bit_count()
                    if weight < best:
                        best, witness = weight, pair
                    for h in range(j):
                        triple = pair ^ basis[h]
                        weight = triple.bit_count()
                        if weight < best:
                            best, witness = weight, triple

            for _ in range(3):
                starting = witness
                for i, first in enumerate(basis):
                    trial = starting ^ first
                    if trial and trial.bit_count() < best:
                        best, witness = trial.bit_count(), trial
                    for j in range(i):
                        trial = starting ^ first ^ basis[j]
                        if trial and trial.bit_count() < best:
                            best, witness = trial.bit_count(), trial
                if witness == starting:
                    break
            if best < 23:
                break
        record["upper"], record["witness"] = best, witness
        assert witness
        assert all((row & witness).bit_count() % 2 == 0
                   for row in record["rows"])

    def exact_kernel(record, floor):
        """Enumerate the ENTIRE kernel, or stop on a floor counterexample.

        Rank 67 gives dimension 25. Optional NumPy batches accelerate
        enumeration; the standard-Python fallback performs the same test.
        This certifies only the classical component admission floor.
        """
        basis = record["basis"]
        best, witness, multiplicity = width + 1, 0, 0

        if np is None:
            word = 0
            for number in range(1, 1 << len(basis)):
                changed = (number & -number).bit_length() - 1
                word ^= basis[changed]
                weight = word.bit_count()
                if weight < floor:
                    return weight, word, False, 0
                if weight < best:
                    best, witness, multiplicity = weight, word, 1
                elif weight == best:
                    multiplicity += 1
            return best, witness, True, multiplicity

        def span_words(vectors):
            words = [0]
            for vector in vectors:
                words += [word ^ vector for word in words]
            return words

        split = min(15, len(basis))
        lower = span_words(basis[:split])
        upper = span_words(basis[split:])
        mask64 = (1 << 64) - 1
        lo = np.fromiter((word & mask64 for word in lower),
                         dtype=np.uint64, count=len(lower))
        hi = np.fromiter((word >> 64 for word in lower),
                         dtype=np.uint64, count=len(lower))
        u64 = np.uint64

        def popcounts(values):
            values = values - ((values >> 1) & u64(0x5555555555555555))
            values = ((values & u64(0x3333333333333333)) +
                      ((values >> 2) & u64(0x3333333333333333)))
            values = (values + (values >> 4)) & u64(0x0f0f0f0f0f0f0f0f)
            return (values * u64(0x0101010101010101)) >> 56

        for high_word in upper:
            weights = (popcounts(lo ^ u64(high_word & mask64)) +
                       popcounts(hi ^ u64(high_word >> 64)))
            if high_word == 0:
                weights[0] = 255
            position = int(weights.argmin())
            weight = int(weights[position])
            word = high_word ^ lower[position]
            if weight < floor:
                return weight, word, False, 0
            count = int(np.count_nonzero(weights == weight))
            if weight < best:
                best, witness, multiplicity = weight, word, count
            elif weight == best:
                multiplicity += count
        return best, witness, True, multiplicity

    def ranking(record):
        hits, c4, c6 = record["geometry"]
        return (min(record["upper"], 23), -hits,
                -((16 + 8 * mode) * c4 + c6), -record["trial"])

    def final_component_key(record):
        return (record["floor"], record["upper"],
                -record["multiplicity"], ranking(record))

    def select_component(side):
        pool = []
        fingerprints = set()
        for trial in range(192):
            C = draw_component(side)
            fingerprint = tuple(tuple(sorted(G.idx(g) for g in cell))
                                for row in C for cell in row)
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            rows = verified_component(C, side)
            geometry = _ev_geometry(rows, width)
            if geometry[0]:
                continue
            upper, witness = cofactor_probe(rows)
            record = dict(C=C, rows=rows, geometry=geometry,
                          upper=upper, witness=witness, trial=trial,
                          floor=0, complete=False, multiplicity=0)
            pool.append(record)
            pool.sort(key=ranking, reverse=True)
            del pool[16:]

        survivors = []
        for record in pool:
            # The new augmentation matrix also has rank one.
            # Full nontrivial-character rank gives binary rank 67.
            if gf2_rank(record["rows"]) != 67:
                continue
            record["basis"] = kernel_basis(record["rows"])
            assert len(record["basis"]) == 25
            information_probe(record, 6)
            survivors.append(record)
        if not survivors:
            raise ValueError("no saturated rank-67 component survivor")

        survivors.sort(key=ranking, reverse=True)
        for record in survivors[:8]:
            information_probe(record, 18)

        # Do not return the first floor-22 survivor. Compare the same
        # number of finalists at ceiling 23 first, carrying every
        # counterexample forward. Successful exhaustive enumerations
        # also supply minimum-word multiplicities.
        for floor in (23, 22):
            candidates = sorted(survivors, key=ranking, reverse=True)[:8]
            for record in candidates:
                if record["complete"] or record["upper"] < floor:
                    continue
                distance, witness, complete, count = exact_kernel(record, floor)
                record["upper"] = min(record["upper"], distance)
                record["witness"] = witness
                if complete:
                    record.update(floor=distance, complete=True,
                                  multiplicity=count)

        survivors.sort(key=final_component_key, reverse=True)
        return survivors

    def tanner_bucket(hx, hz, n):
        """Intrinsic uncoloured Tanner invariant, not an isomorphism test.

        Isomorphic graphs necessarily receive the same bucket.
        Different slot buckets therefore exclude cross-slot orbit
        copies; collisions can only reject additional candidates.
        """
        from collections import Counter
        import hashlib

        rows = hx + hz
        cols = _ev_columns(rows, n)
        adj = [dict() for _ in rows]
        for syndrome in cols:
            incident = []
            while syndrome:
                bit = syndrome & -syndrome
                incident.append(bit.bit_length() - 1)
                syndrome ^= bit
            for position, a in enumerate(incident):
                for b in incident[:position]:
                    adj[a][b] = adj[a].get(b, 0) + 1
                    adj[b][a] = adj[b].get(a, 0) + 1

        masks = [sum(1 << j for j in neighbors) for neighbors in adj]
        weights = [row.bit_count() for row in rows]
        signatures = []
        for i, neighbors in enumerate(adj):
            histogram = Counter(
                (multiplicity, weights[j], len(adj[j]),
                 (masks[i] & masks[j]).bit_count())
                for j, multiplicity in neighbors.items())
            signatures.append((weights[i], len(neighbors),
                               tuple(sorted(histogram.items()))))
        signature = (tuple(sorted(Counter(signatures).items())),
                     tuple(sorted(Counter(col.bit_count()
                                          for col in cols).items())))
        digest = hashlib.sha256(repr(signature).encode("ascii")).digest()
        return int.from_bytes(digest[:8], "big") % N_SLOTS

    def scaled_b(record, unit):
        """An exact constituent coordinate permutation, not a new probe."""
        if unit == 1:
            return record
        transformed = dict(record)
        transformed["C"] = [
            [[G.elements[(unit * G.idx(g)) % q] for g in cell]
             for cell in row] for row in record["C"]]
        word = record["witness"]
        transformed_word = 0
        while word:
            bit = word & -word
            block, h = divmod(bit.bit_length() - 1, q)
            transformed_word |= 1 << (block * q + (unit * h) % q)
            word ^= bit
        transformed["witness"] = transformed_word
        # Floors and multiplicities survive this permutation.
        return transformed

    def quantum_witness_reading(a, b, hx, hz, n):
        """Verified nontrivial logicals reduced by stabilizer moves.

        These are upper readings. Failure to find a lighter mixed-sector
        representative is not a quantum-distance lower bound.
        """
        best = n + 1
        found = False
        for kind, checks, stabilizers, component in (
                ("X", hz, hx, b), ("Z", hx, hz, a)):
            pivots = {}
            for row in stabilizers:
                while row:
                    p = row.bit_length() - 1
                    if p not in pivots:
                        pivots[p] = row
                        break
                    row ^= pivots[p]

            def outside(word):
                while word:
                    p = word.bit_length() - 1
                    previous = pivots.get(p)
                    if previous is None:
                        return True
                    word ^= previous
                return False

            moves = list(stabilizers)
            moves.extend(first ^ stabilizers[j]
                         for i, first in enumerate(stabilizers)
                         for j in range(i))
            for block in range(4):
                if kind == "X":
                    word = component["witness"] << (block * width)
                else:
                    word = 0
                    bits = component["witness"]
                    while bits:
                        bit = bits & -bits
                        j, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << ((j * 4 + block) * q + h)
                        bits ^= bit
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                if not outside(word):
                    continue
                found = True
                weight = word.bit_count()
                for descent in range(2):
                    starting = word
                    for move in moves:
                        trial = starting ^ move
                        trial_weight = trial.bit_count()
                        if trial_weight < weight:
                            word, weight = trial, trial_weight
                    if word == starting:
                        break
                assert word and outside(word)
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                best = min(best, weight)
        return best if found else None

    # Every slot has a different intrinsic bucket, including additional
    # epochs when the harness requests more than twenty slots.
    # Relative B scaling can change the full LP graph while preserving
    # its constituent distance. Isomorphic scaling images cannot reach
    # different buckets and are never used as a diversity certificate.
    for batch in range(2):
        aa, bb = select_component("A"), select_component("B")
        pairings = [(a, b) for a in aa for b in bb]
        pairings.sort(key=lambda pair: (
            min(pair[0]["floor"], pair[1]["floor"]),
            min(pair[0]["upper"], pair[1]["upper"]),
            pair[0]["upper"] + pair[1]["upper"],
            -(pair[0]["multiplicity"] + pair[1]["multiplicity"]),
            ranking(pair[0]), ranking(pair[1])), reverse=True)
        winners = []
        for position, (a, original_b) in enumerate(pairings):
            units = range(1, q) if position < 12 else (1,)
            for unit in units:
                b = scaled_b(original_b, unit)
                cells = {"A": a["C"], "B": b["C"]}
                hx, hz, n = lifted_matrices(G, cells["A"], cells["B"])
                if tanner_bucket(hx, hz, n) != idx:
                    continue
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if k != 35 or not (8 * k < n):
                    continue
                if max(row.bit_count() for row in hx + hz) > min(9, W_MAX):
                    continue
                if _ev_lp4_bad(hx, hz, n) or _ev_lp4_bad(hz, hx, n):
                    continue
                assert css_ok(hx, hz)
                reading = quantum_witness_reading(a, b, hx, hz, n)
                if reading is None:
                    continue
                reading = min(reading, a["upper"], b["upper"])
                quality = (
                    min(a["floor"], b["floor"]),
                    k * reading * reading / float(n),
                    reading,
                    -(a["multiplicity"] + b["multiplicity"]),
                    ranking(a), ranking(b))
                winners.append((quality, cells, a, b, k, reading, unit))
                if len(winners) >= 4:
                    break
            if len(winners) >= 4:
                break

        if winners:
            _, cells, a, b, k, reading, unit = max(
                winners, key=lambda item: item[0])
            return {
                "idx": idx, "cells": cells, "k": k,
                "component_floor": (a["floor"], b["floor"]),
                "component_upper": (a["upper"], b["upper"]),
                "component_exact": (a["complete"], b["complete"]),
                "quantum_witness_upper": reading,
                "relative_B_unit": unit,
                "intrinsic_bucket": idx,
                "support_card": (pa, pb, mode),
            }
    raise ValueError("no screened saturated pair reached this intrinsic bucket")


def _hi_no_kernel_two(rows, width):
    """Exactly exclude all nonzero commuting words of weights one/two."""
    cols = _ev_columns(rows, width)
    return 0 not in cols and len(set(cols)) == width


def _hi_outside(word, rows):
    """Exact nonmembership in the supplied stabilizer row space."""
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old
    while word:
        p = word.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _hi_signature(hx, hz, width):
    """Intrinsic full-Tanner invariant; equality is rejected conservatively.

    Check degrees are eight/nine; qubit degrees are two/three/four/fourteen.
    Thus the two vertex classes are intrinsically distinguishable here.
    Degree-conditioned projections and rooted spheres preserve arbitrary
    Tanner relabelings. Different signatures certify nonisomorphism.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    cols = _ev_columns(rows, width)
    m = len(rows)
    relations = {}
    overlaps = [dict() for _ in rows]
    for col in cols:
        degree = col.bit_count()
        relation = relations.setdefault(degree, [0] * m)
        bits = col
        while bits:
            bit = bits & -bits
            i = bit.bit_length() - 1
            relation[i] |= col ^ bit
            rest = col ^ bit
            while rest:
                other = rest & -rest
                j = other.bit_length() - 1
                overlaps[i][j] = overlaps[i].get(j, 0) + 1
                rest ^= other
            bits ^= bit

    all_adj = [0] * m
    for relation in relations.values():
        for i, mask in enumerate(relation):
            all_adj[i] |= mask

    def spheres(root, relation):
        visited = frontier = 1 << root
        profile = []
        while frontier:
            following = 0
            bits = frontier
            while bits:
                bit = bits & -bits
                following |= relation[bit.bit_length() - 1]
                bits ^= bit
            frontier = following & ~visited
            if frontier:
                profile.append(frontier.bit_count())
                visited |= frontier
        return tuple(profile)

    degrees = [row.bit_count() for row in rows]
    features = []
    for i in range(m):
        local = Counter(
            (multiplicity, degrees[j], all_adj[j].bit_count(),
             (all_adj[i] & all_adj[j]).bit_count())
            for j, multiplicity in overlaps[i].items())
        features.append((
            degrees[i], tuple(sorted(local.items())),
            spheres(i, all_adj),
            tuple((degree, spheres(i, relations[degree]))
                  for degree in sorted(relations))))
    signature = (
        width, tuple(sorted(Counter(col.bit_count()
                                    for col in cols).items())),
        tuple(sorted(Counter(features).items())))
    return sha256(repr(signature).encode("ascii")).digest()


from functools import lru_cache


@lru_cache(maxsize=1)
def _hi_portfolio(G, count):
    """Pure enumerated portfolio; cached outputs are immutable."""
    from itertools import combinations

    q = G.q
    if q != 23 or A_SHAPE != (1, 4) or B_SHAPE != (1, 4):
        raise ValueError("high-rate portfolio requires C23 and (1,4) sides")
    support = [[1, 2, 2, 2]]
    assert _no_dead_lines(support, support)
    assert _grid_weights(support, support) == 9
    assert _grid_weights(support, support) <= W_MAX

    triples = tuple(combinations(range(1, (q + 1) // 2), 3))

    def image(triple, unit):
        return tuple(sorted(min((unit * d) % q, (-unit * d) % q)
                            for d in triple))

    actions = {
        triple: tuple(image(triple, unit) for unit in range(1, q))
        for triple in triples}
    representatives = sorted({
        min(actions[triple]) for triple in triples})

    def cells(triple):
        return [[[G.e]] + [[G.e, G.elements[d]] for d in triple]]

    # Enumerate structural pairings, rather than draw twenty labelings.
    # One side can be normalized by a common group automorphism.
    plans = [(a, b) for a in representatives for b in triples]
    local = random.Random(BASE_SEED ^ 0x2314C3)
    local.shuffle(plans)
    orbit_keys, graph_keys, results = set(), set(), []

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for a, b in plans:
        orbit = min(
            tuple(sorted((aa, bb)))
            for aa, bb in zip(actions[a], actions[b]))
        if orbit in orbit_keys:
            continue
        orbit_keys.add(orbit)
        A, B = cells(a), cells(b)
        hx, hz, n = lifted_matrices(G, A, B)
        if n != 391:
            raise ValueError("unexpected high-rate lift size")
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 207 or 1000 * k <= 525 * n:
            continue
        if max(row.bit_count() for row in hx + hz) > min(9, W_MAX):
            continue
        if not (_hi_no_kernel_two(hx, n)
                and _hi_no_kernel_two(hz, n)):
            continue

        signature = _hi_signature(hx, hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)

        # Obtain a B-kernel three-word from VERIFIED column images.
        # Its binomial column is corrected using the identity block.
        _, br, _ = lifted_matrices(G, [[[]]], B)
        columns = _ev_columns(br, 5 * q)
        position = q + G.idx(G.e)
        component_word = (1 << position) ^ columns[position]
        if component_word.bit_count() != 3:
            continue
        assert all((row & component_word).bit_count() % 2 == 0
                   for row in br)

        # Embed in A-column one, leaving the A-column-zero pivot empty.
        # Verify commutation AND nontriviality in the complete matrices.
        witness = component_word << (4 * q)
        if any((row & witness).bit_count() & 1 for row in hz):
            raise ValueError("verified three-word embedding failed")
        if not _hi_outside(witness, hx):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified high-rate lift failed CSS")

        results.append((
            freeze(A), freeze(B), k, signature.hex(), (a, b)))
        if len(results) >= count:
            break
    return tuple(results)


def NEW_CTX(rng, G, idx):
    """Publish an invariant-separated, distance-three high-rate design."""
    portfolio = _hi_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("intrinsically distinct high-rate portfolio exhausted")
    A, B, k, signature, directions = portfolio[idx]
    return {
        "idx": idx, "cells": {"A": A, "B": B}, "k": k,
        "distance_certificate": 3,
        "tanner_signature": signature,
        "direction_prescription": directions,
    }


def CELL(rng, G, side, i, j, ctx):
    """Publish the selected complete cell without additional label draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled supports with uniform labels on D58, (1,3)x(1,3), do not avoid weight-2 logicals: all 55 distinct slots had d2s=2 and all five measured slots had d30s=2.
# REFUTED: Twenty support/relative-voltage prescriptions alone supply twenty distinct C23 codes: the measured odd-cofactor parent retained only eight classes across sixty slots, with repeated isomorphism duplicates.
# REFUTED: One-sided matching trinomials with monomial B remove the weight-22 sector-1 plateau: four of the five measured odd-cofactor parent codes instead had d30s=22.
# REFUTED: Exact constituent-floor targeting protects five distance-23 quantum readings on saturated C23 (3,4)x(3,4): current measured slots 12 and 13 instead reached mixed-sector minima of weights 18 and 20.
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

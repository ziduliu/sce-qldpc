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
CONCEPT = ("C13:C9(s=3), order 117, shapes (1,2)x(1,2): recombining strong balanced constituents through independent relative automorphisms should yield five new codes with k>=119 and d30s=18, exceeding fitness 64.8 in cell 0.200. Full-kernel mixed-sector searches test, but do not certify, retention of distance.")
GROUP = ("Keep the frozen order-117 C13:C9(s=3): this bed has reached fitness 64.8. The maps x->x^u, y->y^w with u nonzero modulo 13 and w in {1,4,7} are automorphisms. Applied to only one constituent they preserve its classical code while changing the relative LP coupling; inner or globally applied automorphisms alone would merely waste archive-equivalent slots.")
SHAPES = ("Keep equal (1,2) sides: each constituent has 234 variables and 117 checks, n=585, and rank counting guarantees k>=117, rate 0.200; q<=119 fits the size window. Verified ranks retain cell 0.200 and favor k>=119. Parent minima repeatedly occupied sector-1 Acol/Bcol blocks, but slot 46 had a weight-17 mixed-sector X witness touching Arow0/Brow0, motivating full-CSS quotient-tagged screening.")
FILLING = ("All four cells remain distinct-element trinomials, so both CSS row maxima are nine. Four balanced A constituents come from measured weight-18 codes. B uses those same element lists, with its own verified right-action construction, or four measured strong normal/noncommuting constituents. Twelve relative outer-automorphism representatives change the coupling without weakening the constituent screening budget. Twenty preference cards prioritize different seed pairs, while intrinsic Tanner buckets constrain final graph diversity. Verified lifts require rank gains, CSS and logical-four exclusion. Embedded witnesses receive stabilizer descent, followed by sixteen-pass full-kernel searches with exact stabilizer-quotient tags. Recombination and buckets do not certify archive novelty; all distance readings remain upper bounds.")
FAMILY_NAME = "r020_c13sd_c9_s3_q117_12x12_rankgain_relative_outer_guard_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 117

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C13:C9(s=3) has order 117")
    # x^13=y^9=1, y*x*y^-1=x^3; 3^3=1 modulo 13.
    powers = [pow(3, b, 13) for b in range(9)]
    els = [(a, b) for b in range(9) for a in range(13)]

    def red(g):
        return (int(g[0]) % 13, int(g[1]) % 9)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b] * c) % 13, (b + d) % 9)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 9
        return ((-powers[bi] * a) % 13, bi)

    def idx(g):
        a, b = red(g)
        return 13 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 2)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Recombine strong constituents and screen their relative coupling."""
    from hashlib import sha256

    sa = sb = ((3, 3),)
    assert G.q == 117
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 9 <= min(W_MAX, 9)

    # These are constituent warm starts, NOT complete archived codes.
    # Each balanced A was part of a measured code with sector-1
    # Z witnesses at weight 18. The normal B records likewise had
    # measured sector-1 X witnesses at weight 18.
    balanced = (
        (((9, 2), (9, 3), (10, 7)),
         ((3, 0), (3, 2), (0, 7))),
        (((10, 3), (12, 5), (1, 7)),
         ((8, 3), (9, 4), (2, 5))),
        (((0, 0), (11, 4), (10, 5)),
         ((3, 1), (10, 2), (3, 3))),
        (((6, 0), (2, 4), (10, 8)),
         ((12, 0), (3, 5), (5, 7))),
    )
    normal_b = (
        (((11, 2), (12, 5), (1, 8)),
         ((4, 0), (6, 2), (0, 4))),
        (((3, 0), (8, 3), (12, 3)),
         ((11, 4), (11, 6), (12, 6))),
        (((3, 2), (2, 5), (8, 5)),
         ((8, 0), (5, 3), (6, 7))),
        (((12, 2), (0, 5), (5, 8)),
         ((1, 2), (2, 7), (3, 7))),
    )
    references = {"A": balanced, "B": balanced + normal_b}
    card = idx % 20
    preferred_a = card % 4
    preferred_b = (card // 4 + preferred_a) % 8
    prescriptions = {
        "A": ("balanced", preferred_a),
        "B": ("relative-coupling", preferred_b),
    }

    # Modding out the inner x-scaling subgroup <3> leaves four
    # representatives in F13*. The three central scalings give
    # twelve relative outer representatives, not inner orbit copies.
    outer = tuple((u, w) for u in (1, 2, 4, 7)
                  for w in (1, 4, 7))

    def automorphism(g, u, w):
        return ((u * g[0]) % 13, (w * g[1]) % 9)

    for u, w in outer:
        assert len({automorphism(g, u, w) for g in G.elements}) == G.q
        for g in ((1, 0), (0, 1), (2, 4)):
            for h in ((3, 0), (0, 2), (5, 7)):
                assert automorphism(G.mul(g, h), u, w) == G.mul(
                    automorphism(g, u, w), automorphism(h, u, w))

    # <x> is normal C13; its nine cosets are indexed by b.
    cosets = tuple(
        tuple(G.idx(g) for g in G.elements if g[1] == b)
        for b in range(9)
    )
    assert all(len(orbit) == 13 for orbit in cosets)

    # Unlike the nine normal-C13 cosets above, these are the 39
    # three-element orbits of the CENTRAL subgroup <y^3>.
    central_orbits = set()
    for g in G.elements:
        orbit = tuple(sorted(
            G.idx(G.mul(g, z)) for z in ((0, 0), (0, 3), (0, 6))
        ))
        central_orbits.add(orbit)
    central_orbits = tuple(sorted(central_orbits))
    assert len(central_orbits) == 39
    assert all(len(set(orbit)) == 3 for orbit in central_orbits)

    def transform_word(word, images):
        """Apply the same group automorphism in each constituent block."""
        transformed = 0
        while word:
            bit = word & -word
            block, h = divmod(bit.bit_length() - 1, G.q)
            transformed |= 1 << (block * G.q + images[h])
            word ^= bit
        return transformed

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def component_quality(rows, width):
        """Small-kernel collision count and Tanner 4/6-cycle penalties."""
        cols = columns(rows, width)
        seen = {0}
        bad = 0
        for s in cols:
            bad += int(s in seen)
            seen.add(s)
        c4 = 0
        for i, s in enumerate(cols):
            for t in cols[:i]:
                u = s ^ t
                bad += int(u in seen)
                seen.add(u)
                overlap = (s & t).bit_count()
                c4 += overlap * (overlap - 1) // 2

        # Project onto check nodes. With no 4-cycles, subtracting
        # within-column triangles leaves exactly the Tanner 6-cycles.
        adj = [0] * len(rows)
        internal = 0
        for s in cols:
            degree = s.bit_count()
            internal += degree * (degree - 1) * (degree - 2) // 6
            bits = s
            while bits:
                bit = bits & -bits
                adj[bit.bit_length() - 1] |= s ^ bit
                bits ^= bit
        triangles = 0
        for u, neighbors in enumerate(adj):
            bits = neighbors >> (u + 1)
            while bits:
                bit = bits & -bits
                v = bit.bit_length() + u
                triangles += ((neighbors & adj[v]) >> (v + 1)).bit_count()
                bits ^= bit
        # Exact search of one/two complete normal-C13 cosets.
        # This is NOT enumeration of the entire invariant subspace.
        # Uniform support on both odd-weight columns is always a
        # constituent kernel word and supplies a verified fallback seed.
        orbit_witness = (1 << width) - 1
        assert all((row & orbit_witness).bit_count() % 2 == 0
                   for row in rows)
        orbit_distance = width
        representatives = {0: 0}
        for j in range(2):
            for orbit in cosets:
                word = syndrome = 0
                for h in orbit:
                    position = j * G.q + h
                    word |= 1 << position
                    syndrome ^= cols[position]
                if syndrome in representatives:
                    candidate = word ^ representatives[syndrome]
                    weight = candidate.bit_count()
                    if 0 < weight < orbit_distance:
                        orbit_distance, orbit_witness = weight, candidate
                else:
                    representatives[syndrome] = word
        return (bad, c4, max(0, triangles - internal),
                orbit_distance, orbit_witness)

    def kernel_probe(rows, width, passes, seed):
        """Heuristic component-kernel upper bound and explicit witness.

        Random information sets expose kernel bases. Search every
        basis singleton, pair and triple, then descend by pairs.
        All returned witnesses are actual kernel words; the search
        is not exhaustive over the whole kernel.
        """
        best_word = seed
        best_weight = seed.bit_count() if seed else width + 1
        syndromes = columns(rows, width)
        for _ in range(passes):
            order = list(range(width))
            rng.shuffle(order)
            # Column elimination carries supports in the ORIGINAL
            # coordinates. Dependent columns give a systematic kernel
            # basis without rebuilding a permuted row matrix.
            pivots = {}
            basis = []
            for original in order:
                syndrome = syndromes[original]
                word = 1 << original
                while syndrome:
                    p = syndrome.bit_length() - 1
                    previous = pivots.get(p)
                    if previous is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= previous[0]
                    word ^= previous[1]
                if syndrome == 0:
                    basis.append(word)

            # Elementary basis exchanges preserve the entire kernel.
            # They expose different low-weight combinations before the
            # exhaustive singleton/pair/triple scan.
            rng.shuffle(basis)
            weights = [word.bit_count() for word in basis]
            for i in range(len(basis)):
                for j in range(len(basis)):
                    if i == j:
                        continue
                    trial = basis[i] ^ basis[j]
                    weight = trial.bit_count()
                    if 0 < weight < weights[i]:
                        basis[i], weights[i] = trial, weight
                if weights[i] < best_weight:
                    best_weight, best_word = weights[i], basis[i]

            # Search every basis pair. Restrict triples to a mixture of
            # short vectors and randomly selected remaining vectors:
            # dimension is now about 117 rather than about 44.
            for i, first in enumerate(basis):
                for second in basis[:i]:
                    pair = first ^ second
                    weight = pair.bit_count()
                    if 0 < weight < best_weight:
                        best_weight, best_word = weight, pair

            ranked = sorted(basis, key=int.bit_count)
            short = ranked[:36]
            tail = ranked[36:]
            short += rng.sample(tail, min(12, len(tail)))
            for i, first in enumerate(short):
                for j in range(i):
                    pair = first ^ short[j]
                    for h in range(j):
                        triple = pair ^ short[h]
                        weight = triple.bit_count()
                        if 0 < weight < best_weight:
                            best_weight, best_word = weight, triple

            # All moves are combinations of actual kernel words.
            for descent in range(3):
                starting = best_word
                improved = False
                for i, first in enumerate(short):
                    word = starting ^ first
                    weight = word.bit_count()
                    if 0 < weight < best_weight:
                        best_weight, best_word = weight, word
                        improved = True
                    for second in short[:i]:
                        word = starting ^ first ^ second
                        weight = word.bit_count()
                        if 0 < weight < best_weight:
                            best_weight, best_word = weight, word
                            improved = True
                if not improved:
                    break
            if best_weight <= (6 if width < 100 else 12):
                break

        assert best_word
        assert all((row & best_word).bit_count() % 2 == 0
                   for row in rows)
        return best_weight, best_word

    def central_probe(rows):
        """Screen the central-C3-invariant subcode of VERIFIED rows.

        A compressed bit means all three physical coordinates in its
        orbit. No independent left/right lift convention is introduced.
        If the heuristic reading exceeds six, exhaustive collisions of
        supports of size <=3 test existence of any quotient word <=6.
        """
        from itertools import combinations

        masks = tuple(
            sum(1 << (j * G.q + h) for h in orbit)
            for j in range(2) for orbit in central_orbits
        )
        folded = []
        for row in rows:
            compressed_row = 0
            for i, mask in enumerate(masks):
                if (row & mask).bit_count() & 1:
                    compressed_row |= 1 << i
            if compressed_row:
                folded.append(compressed_row)
        folded = list(dict.fromkeys(folded))
        compressed_width = len(masks)
        seed = (1 << compressed_width) - 1
        assert all((row & seed).bit_count() % 2 == 0 for row in folded)
        reading, compressed_word = kernel_probe(
            folded, compressed_width, 4, seed)

        if reading > 6:
            syndromes = columns(folded, compressed_width)
            representatives = {0: 0}
            collision = None
            for size in (1, 2, 3):
                for support in combinations(range(compressed_width), size):
                    syndrome = word = 0
                    for position in support:
                        syndrome ^= syndromes[position]
                        word |= 1 << position
                    previous = representatives.get(syndrome)
                    if previous is not None:
                        candidate = word ^ previous
                        if candidate:
                            collision = candidate
                            break
                    else:
                        representatives[syndrome] = word
                if collision is not None:
                    break
            if collision is not None:
                compressed_word = collision
                reading = collision.bit_count()
                assert 0 < reading <= 6

        lifted_word = 0
        remaining = compressed_word
        while remaining:
            bit = remaining & -remaining
            lifted_word |= masks[bit.bit_length() - 1]
            remaining ^= bit
        assert lifted_word.bit_count() == 3 * reading
        assert all((row & lifted_word).bit_count() % 2 == 0
                   for row in rows)
        return 3 * reading, lifted_word

    def low_logical(checks, stabilizers, width):
        """Exact existence test for a nontrivial logical of weight <=4."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                if p not in pivots:
                    return True
                word ^= pivots[p]
            return False

        syndromes = columns(checks, width)
        representatives = {0: 0}
        for i, syndrome in enumerate(syndromes):
            word = 1 << i
            if syndrome in representatives:
                if nontrivial(word ^ representatives[syndrome]):
                    return True
            else:
                representatives[syndrome] = word
        for i, syndrome in enumerate(syndromes):
            bit = 1 << i
            for j in range(i):
                s = syndrome ^ syndromes[j]
                word = bit | (1 << j)
                if s in representatives:
                    if nontrivial(word ^ representatives[s]):
                        return True
                else:
                    representatives[s] = word
        # All words of weight <=4 split into two words of weight <=2.
        # One representative per syndrome suffices because the
        # stabilizer space is linear.
        return False

    q = G.q
    width = 2 * q

    def verified_component(grid, side):
        """A zero auxiliary cell isolates the scaffold constituent."""
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, [[[]]])
        else:
            _, rows, _ = lifted_matrices(G, [[[]]], grid)
        assert len(rows) == q
        assert all(row.bit_length() <= width for row in rows)
        return rows

    def constituent_finalists(profile, side):
        assert profile == sa
        finalists = []
        preferred = preferred_a if side == "A" else preferred_b
        for seed_id, reference in enumerate(references[side]):
            base_grid = [[list(cell) for cell in reference]]
            base_rows = verified_component(base_grid, side)
            rank = gf2_rank(base_rows)
            if side == "A" and rank != q - 2:
                continue
            stats = component_quality(base_rows, width)
            if stats[0]:
                continue
            central_distance, central_word = central_probe(base_rows)
            seed = min((stats[4], central_word), key=int.bit_count)

            # A measured constituent is only a warm start: independently
            # check it in the precise convention of the side using it.
            distance, witness = kernel_probe(base_rows, width, 3, seed)
            distance, witness = kernel_probe(
                base_rows, width, 16, witness)
            if distance < 18:
                continue

            # Do not invert the B labels. Coordinate inversion relates
            # left and right constituents using the SAME element lists.
            # Every transformed record is nevertheless checked against
            # rows built by the verified scaffold helper.
            variants = ((1, 1),) if side == "A" else outer
            for u, w in variants:
                grid = [[[automorphism(g, u, w) for g in cell]
                         for cell in reference]]
                rows = (base_rows if (u, w) == (1, 1)
                        else verified_component(grid, side))
                images = [
                    G.idx(automorphism(g, u, w)) for g in G.elements
                ]
                moved_witness = transform_word(witness, images)
                moved_central = transform_word(central_word, images)
                assert moved_witness.bit_count() == distance
                assert all((row & moved_witness).bit_count() % 2 == 0
                           for row in rows)
                assert all((row & moved_central).bit_count() % 2 == 0
                           for row in rows)
                quality = (
                    distance, q - rank,
                    -stats[1], -stats[2],
                    int(seed_id == preferred))
                finalists.append(dict(
                    grid=grid, rows=rows, stats=stats,
                    distance=distance, witness=moved_witness,
                    central_distance=central_distance,
                    central_witness=moved_central,
                    quality=quality, rank=rank,
                    seed_id=seed_id, relative_auto=(u, w)))
        if not finalists:
            raise ValueError("No reference constituent retained reading 18")
        finalists.sort(key=lambda item: item["quality"], reverse=True)
        return finalists

    def intrinsic_bucket(hx, hz, n):
        """Permutation-invariant uncoloured Tanner features.

        Different buckets certify different Tanner graphs. Equal
        signatures are not claimed to certify isomorphism. No element
        indices, row ordering or CSS colouring enter the signature.
        """
        rows = list(hx) + list(hz)
        weights = [row.bit_count() for row in rows]
        adjacency = [[] for _ in rows]
        masks = [0] * len(rows)
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                if overlap:
                    adjacency[i].append((j, overlap))
                    adjacency[j].append((i, overlap))
                    masks[i] |= 1 << j
                    masks[j] |= 1 << i
        features = []
        for i, neighbors in enumerate(adjacency):
            histogram = {}
            for j, overlap in neighbors:
                key = (weights[j], overlap, len(adjacency[j]),
                       (masks[i] & masks[j]).bit_count())
                histogram[key] = histogram.get(key, 0) + 1
            features.append((
                weights[i], len(neighbors),
                tuple(sorted(histogram.items())),
            ))
        signature = (n, len(rows), tuple(sorted(features)))
        digest = sha256(repr(signature).encode("ascii")).digest()
        return int.from_bytes(digest[:8], "big") % N_SLOTS

    pairs = []
    aa = constituent_finalists(sa, "A")
    bb = constituent_finalists(sb, "B")
    pending = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            weak = min(a["distance"], b["distance"])
            priority = (
                weak, 2 * q - a["rank"] - b["rank"],
                int(a["seed_id"] == preferred_a),
                int(b["seed_id"] == preferred_b),
                -(a["stats"][1] + b["stats"][1]), -ai, -bi)
            pending.append((priority, ai, bi))
    pending.sort(reverse=True)

    for _, ai, bi in pending:
        a, b = aa[ai], bb[bi]
        hx, hz, n = lifted_matrices(G, a["grid"], b["grid"])
        assert n == 585
        if intrinsic_bucket(hx, hz, n) != idx:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k >= 119 and 7 * n <= 40 * k < 9 * n):
            continue
        weak = min(a["distance"], b["distance"])

        # Relative coupling is invisible to the separate constituent
        # distances. Large cross-colour overlaps are a secondary
        # heuristic only, never a logical-distance certificate.
        coupling = sum(
            max(0, (x & z).bit_count() - 2)
            for x in hx for z in hz)
        quality = (
            k * weak * weak / float(n), weak, -coupling,
            -(a["stats"][1] + b["stats"][1]))
        pairs.append(dict(a=a, b=b, hx=hx, hz=hz, n=n,
                          k=k, quality=quality))
        if len(pairs) >= 10:
            break

    pairs.sort(key=lambda item: item["quality"], reverse=True)
    if not pairs:
        raise ValueError("No cell-0.200 pair in the intrinsic slot bucket")

    def mixed_logical_probe(checks, stabilizers, n, seed):
        """Heuristic full-kernel search with exact logical-class tags."""
        stab_pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                previous = stab_pivots.get(p)
                if previous is None:
                    stab_pivots[p] = row
                    break
                row ^= previous

        def quotient(word):
            remainder = 0
            while word:
                p = word.bit_length() - 1
                previous = stab_pivots.get(p)
                if previous is None:
                    bit = 1 << p
                    remainder |= bit
                    word ^= bit
                else:
                    word ^= previous
            return remainder

        assert seed and quotient(seed)
        syndromes = columns(checks, n)
        best_word, best_weight = seed, seed.bit_count()
        for attempt in range(16):
            order = list(range(n))
            rng.shuffle(order)
            pivots = {}
            basis = []
            for original in order:
                syndrome = syndromes[original]
                word = 1 << original
                while syndrome:
                    p = syndrome.bit_length() - 1
                    previous = pivots.get(p)
                    if previous is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= previous[0]
                    word ^= previous[1]
                if syndrome == 0:
                    basis.append(word)

            rng.shuffle(basis)
            tags = [quotient(word) for word in basis]
            weights = [word.bit_count() for word in basis]
            for i in range(len(basis)):
                word, tag, weight = basis[i], tags[i], weights[i]
                for j in range(len(basis)):
                    if i == j:
                        continue
                    trial = word ^ basis[j]
                    trial_weight = trial.bit_count()
                    if 0 < trial_weight < weight:
                        word, weight = trial, trial_weight
                        tag ^= tags[j]
                basis[i], tags[i], weights[i] = word, tag, weight
                if tag and weight < best_weight:
                    best_word, best_weight = word, weight

            logical = sorted(
                (i for i, tag in enumerate(tags) if tag),
                key=lambda i: weights[i])
            trivial = sorted(
                (i for i, tag in enumerate(tags) if not tag),
                key=lambda i: weights[i])
            short = logical[:28] + trivial[:12]
            tail = logical[28:]
            short += rng.sample(tail, min(8, len(tail)))

            # Tags are propagated linearly, so neither short checks
            # nor arbitrary stabilizer combinations can win as logicals.
            for ii, i in enumerate(short):
                for jj in range(ii):
                    j = short[jj]
                    pair = basis[i] ^ basis[j]
                    pair_tag = tags[i] ^ tags[j]
                    weight = pair.bit_count()
                    if pair_tag and 0 < weight < best_weight:
                        best_word, best_weight = pair, weight
                    for kk in range(jj):
                        h = short[kk]
                        if not (pair_tag ^ tags[h]):
                            continue
                        trial = pair ^ basis[h]
                        weight = trial.bit_count()
                        if 0 < weight < best_weight:
                            best_word, best_weight = trial, weight

            starting = best_word
            starting_tag = quotient(starting)
            for ii, i in enumerate(short):
                trial = starting ^ basis[i]
                weight = trial.bit_count()
                if (starting_tag ^ tags[i]) and 0 < weight < best_weight:
                    best_word, best_weight = trial, weight
                for jj in range(ii):
                    j = short[jj]
                    if not (starting_tag ^ tags[i] ^ tags[j]):
                        continue
                    trial = starting ^ basis[i] ^ basis[j]
                    weight = trial.bit_count()
                    if 0 < weight < best_weight:
                        best_word, best_weight = trial, weight

            if best_weight < 18:
                break

        assert quotient(best_word)
        assert all((row & best_word).bit_count() % 2 == 0
                   for row in checks)
        return best_weight, best_word

    def quantum_witness_reading(record):
        """Actual nontrivial logical witnesses, not a distance certificate.

        Embed the component witnesses using the verified sector-1 column
        order. Check commutation and nontriviality explicitly before
        reducing their weights by one- and two-stabilizer moves.
        """
        best = record["n"] + 1
        found = False
        for kind, checks, stabilizers in (
                ("X", record["hz"], record["hx"]),
                ("Z", record["hx"], record["hz"])):
            pivots = {}
            for row in stabilizers:
                while row:
                    p = row.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = row
                        break
                    row ^= old

            def nontrivial(word):
                while word:
                    p = word.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        return True
                    word ^= old
                return False

            kind_seed = None
            # A longer central-orbit witness can reduce to a lighter
            # mixed-sector representative than the shortest probe word.
            for block, source in (
                    (block, source)
                    for block in range(2)
                    for source in ("witness", "central_witness")):
                if kind == "X":
                    word = record["b"][source] << (block * width)
                else:
                    word = 0
                    component_word = record["a"][source]
                    while component_word:
                        bit = component_word & -component_word
                        j, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << ((j * 2 + block) * q + h)
                        component_word ^= bit
                if any((row & word).bit_count() & 1 for row in checks):
                    raise ValueError("Component witness embedding failed")
                if not nontrivial(word):
                    continue
                found = True
                weight = word.bit_count()
                # Adding stabilizers preserves both commutation and
                # logical class, including when the support enters sector 2.
                for descent in range(2):
                    starting = word
                    for i, first in enumerate(stabilizers):
                        trial = starting ^ first
                        trial_weight = trial.bit_count()
                        if trial_weight < weight:
                            word, weight = trial, trial_weight
                        for j in range(i):
                            trial = starting ^ first ^ stabilizers[j]
                            trial_weight = trial.bit_count()
                            if trial_weight < weight:
                                word, weight = trial, trial_weight
                    if word == starting:
                        break
                assert word and nontrivial(word)
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                best = min(best, weight)
                if kind_seed is None or weight < kind_seed.bit_count():
                    kind_seed = word
            if kind_seed is not None:
                mixed_weight, mixed_word = mixed_logical_probe(
                    checks, stabilizers, record["n"], kind_seed)
                assert nontrivial(mixed_word)
                best = min(best, mixed_weight)
        return best if found else None

    # Compare equally refined pairs with the same quantum witness budget.
    # Screening additional pairs supplies fallbacks without allowing
    # unrefined constituents to win through a shorter search.
    survivors = []
    for record in pairs:
        hx, hz, n = record["hx"], record["hz"], record["n"]
        if low_logical(hz, hx, n) or low_logical(hx, hz, n):
            continue
        assert css_ok(hx, hz)
        reading = quantum_witness_reading(record)
        record["quantum_reading"] = reading
        if reading is None or reading < 18:
            continue
        weak = (reading if reading is not None else
                min(record["a"]["distance"], record["b"]["distance"]))
        record["final_quality"] = (
            int(reading is not None),
            record["k"] * weak * weak / float(n),
            weak, record["quality"])
        survivors.append(record)
        if len(survivors) >= 6:
            break
    if not survivors:
        raise ValueError("No selected pair passed the weight-four logical screen")
    winner = max(survivors, key=lambda item: item["final_quality"])
    return {
        "idx": idx,
        "cells": {"A": winner["a"]["grid"], "B": winner["b"]["grid"]},
        "component_probe": (
            winner["a"]["distance"], winner["b"]["distance"]),
        "quantum_witness_probe": winner["quantum_reading"],
        "central_component_probe": (
            winner["a"]["central_distance"],
            winner["b"]["central_distance"]),
        "k": winner["k"],
        "card": card,
        "triangle_types": prescriptions,
        "constituent_sources": (
            winner["a"]["seed_id"], winner["b"]["seed_id"]),
        "relative_automorphism": winner["b"]["relative_auto"],
        "intrinsic_bucket": idx,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh list from the already screened complete filling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Rolled uniform Z21 fillings on (2,4)x(2,6) suffice for d30s>4: refuted by all five measured parent codes having d30s=4, with sector-1 witnesses.
# REFUTED: Weight-four rejection plus component cycle minimization excludes weight-20 mixed-sector logicals on Dic11 (2,3)x(2,3): measured slot 42 had d30s=20 with Z witnesses in both sectors.
# REFUTED: The v2 mixed-coset/component-kernel policy yields five measured readings >=24: its measured slots 8 and 9 instead had d30s=23, with both logical types entirely in sector 1.
# REFUTED: The v3 independently optimized saturated-constituent policy yields five measured readings >=25: all five measured codes instead had d30s=24, with minimum witnesses entirely in sector 1.
# REFUTED: A d2s=18 reading is always stable under the longer search on the order-117 bed: parent slot 7 fell to d30s=15, with X witnesses using both LP sectors.
# REFUTED: The normal-C39/noncommuting central-orbit policy lifts measured distances beyond 18: its five measured codes read 18,18,18,17,18, and slot 46 exposed a mixed-sector X witness.
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

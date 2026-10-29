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
CONCEPT = ("C13 semidirect C9 with action 3, order 117, shapes (1,2)x(1,2): crossovers and genuine support edits around nine independently measured weight-20 constituent references, with known complete-code signatures excluded, should produce five new measured readings >=20 and fitness >=80 in cell 0.200. This is a falsifiable search target, not a distance certificate.")
GROUP = ("Keep the frozen C13:C9(s=3), order 117. This bed reached fitness 75.56 with four measured distance-20 codes, versus the current parent's five readings of 18. The action is valid since 3^3=1 modulo 13; central C3 supplies one structured mutation direction. Neither this algebra nor a constituent's measured upper bound certifies quantum distance.")
SHAPES = ("Keep the frozen equal (1,2) sides: each constituent has 234 variables and 117 checks, n=585, and rank counting guarantees k>=117 and rate>=0.200. The size window permits q<=119. Parent weight-18 minima occupied sector-1 Acol/Bcol blocks across both columns; earlier weight-15 and weight-17 mixed-sector failures also touched Arow0/Brow0. Improve both constituents while retaining verified ranks, logical-four exclusion and full-CSS quotient-tag screening.")
FILLING = ("All four cells are distinct-element noncommuting trinomials of spectrum (9,9,39), keeping BOTH CSS maximum row weights nine. Twenty cards combine five mutation geometries with four cycle objectives. Each side searches nine measured constituent references, new column crossovers, one/two genuine support edits and independent proposals. Group-automorphism/right-column-gauge classes are deduplicated; cheap and exploratory classes plus references receive kernel probes, with equal final budgets and neutral basis exchanges. Verified singleton lifts supply cached blocks. Complete-code Tanner invariants reject the five shown distance-20 reference codes before bucket selection. One-sided automorphisms are not counted as novelty by themselves. Full-CSS finalists receive stronger screening; readings remain upper bounds and novelty against the remaining archive is not assumed.")
FAMILY_NAME = "r020_c13sd_c9_s3_q117_12x12_reference_crossover_exclusion_v5"

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
    """Cross measured constituents, edit supports, and exclude known codes."""
    from hashlib import sha256

    sa = sb = ((3, 3),)
    assert G.q == 117
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 9 <= min(W_MAX, 9)

    spectra = ((9, 9, 9), (9, 9, 13), (9, 9, 39))
    prescriptions = {"A": (2, 2), "B": (2, 2)}
    cycle_policies = ((64, 0), (16, 1), (4, 1), (1, 2))
    plans = tuple((mode, policy) for mode in range(5)
                  for policy in range(4))
    card = idx % len(plans)
    mode, policy = plans[card]
    mutation_modes = {"A": mode, "B": (mode + 2) % 5}
    side_policies = {
        "A": cycle_policies[policy],
        "B": cycle_policies[(policy + mode + 1) % 4],
    }
    draw_counts = {"A": 0, "B": 0}

    # Every reference occurred as a constituent of a measured code whose
    # search reached weight 20 for the corresponding logical type.
    # This does not establish its exact classical or quantum distance.
    references = (
        (((8, 0), (9, 3), (12, 8)),
         ((2, 0), (8, 1), (7, 6))),
        (((1, 0), (10, 2), (7, 6)),
         ((7, 2), (1, 4), (5, 7))),
        (((10, 1), (1, 4), (11, 6)),
         ((6, 0), (7, 2), (0, 5))),
        (((5, 1), (4, 4), (12, 6)),
         ((3, 0), (10, 2), (0, 5))),
        (((10, 1), (9, 5), (3, 7)),
         ((7, 0), (9, 3), (4, 5))),
        (((5, 1), (7, 4), (12, 6)),
         ((3, 0), (10, 2), (0, 5))),
        (((10, 1), (8, 4), (11, 6)),
         ((6, 0), (7, 2), (0, 5))),
        (((7, 0), (10, 6), (10, 8)),
         ((0, 4), (0, 5), (2, 7))),
        (((4, 0), (6, 6), (9, 7)),
         ((0, 0), (10, 6), (4, 8))),
    )
    relative_u = pow(2, card % 12, 13)
    relative_w = (1, 4, 7)[(card // 4) % 3]

    def element_order(g):
        value = G.e
        for order in range(1, G.q + 1):
            value = G.mul(value, g)
            if value == G.e:
                return order
        raise ValueError("invalid group element order")

    orders = {g: element_order(g) for g in G.elements}
    banks = [[] for _ in spectra]
    nonidentity = [g for g in G.elements if g != G.e]
    for i, first in enumerate(nonidentity):
        for second in nonidentity[:i]:
            if G.mul(first, second) == G.mul(second, first):
                continue
            spectrum = tuple(sorted((
                orders[first], orders[second],
                orders[G.mul(G.inv(first), second)],
            )))
            if spectrum in spectra:
                banks[spectra.index(spectrum)].append((first, second))
    if any(not bank for bank in banks):
        raise ValueError("noncommuting triangle spectrum absent")

    # <x> is normal C13; its nine cosets are indexed by b.
    cosets = tuple(
        tuple(G.idx(g) for g in G.elements if g[1] == b)
        for b in range(9)
    )
    assert all(len(orbit) == 13 for orbit in cosets)

    def all39(cell):
        if len(set(cell)) != 3:
            return False
        first, second, third = cell
        return tuple(sorted((
            orders[G.mul(G.inv(first), second)],
            orders[G.mul(G.inv(first), third)],
            orders[G.mul(G.inv(second), third)],
        ))) == (9, 9, 39)

    assert all(all39(cell) for row in references for cell in row)

    # These valid automorphisms are used to avoid repeatedly probing
    # elementary copies of one classical constituent. They are not a
    # complete arbitrary Tanner-isomorphism or archive test.
    automorphisms = tuple(
        tuple(G.idx((
            (u * a + v * (0, 1, 4)[b % 3]) % 13,
            (w * b) % 9,
        )) for a, b in G.elements)
        for u in range(1, 13)
        for v in range(13)
        for w in (1, 4, 7)
    )
    cell_image_cache = {}

    def cell_images(cell):
        # Independent right multiplication permutes a constituent's
        # corresponding variable block under both verified conventions.
        normalized = tuple(sorted({
            tuple(sorted(G.idx(G.mul(g, G.inv(anchor)))
                         for g in cell))
            for anchor in cell
        }))
        cached = cell_image_cache.get(normalized)
        if cached is None:
            cached = tuple(
                min(tuple(sorted(permutation[g] for g in choice))
                    for choice in normalized)
                for permutation in automorphisms
            )
            cell_image_cache[normalized] = cached
        return cached

    def constituent_class(grid):
        first, second = (cell_images(cell) for cell in grid[0])
        return min(
            (a, b) if a <= b else (b, a)
            for a, b in zip(first, second)
        )

    reference_classes = {
        constituent_class([[list(cell) for cell in row]])
        for row in references
    }

    def fresh_triangle():
        anchor = G.pick(rng)
        first, second = rng.choice(banks[2])
        if rng.randrange(2):
            first, second = second, first
        cell = [anchor, G.mul(anchor, first), G.mul(anchor, second)]
        assert all39(cell)
        return cell

    def draw_grid(profile, side):
        assert profile == sa
        trial = draw_counts[side]
        draw_counts[side] += 1
        mutation_mode = mutation_modes[side]

        if trial < len(references):
            row = [list(cell) for cell in references[trial]]
        elif trial % 5 == 0:
            row = [fresh_triangle(), fresh_triangle()]
        else:
            row = [list(cell) for cell in rng.choice(references)]
            # Cross columns INSIDE a constituent, not merely swap the
            # two complete sides of an already measured quantum code.
            if trial % 5 in (1, 2) or mutation_mode == 4:
                row[1] = list(rng.choice(references)[1])
            edits = 0 if trial % 5 == 1 else 1 + int(trial % 5 == 4)
            for edit in range(edits):
                column = (trial + edit + card) % 2
                position = rng.randrange(3)
                original = row[column][position]
                for attempt in range(192):
                    a, b = original
                    if mutation_mode == 0:
                        replacement = (a, (b + 3 * rng.choice((1, 2))) % 9)
                    elif mutation_mode == 1:
                        replacement = ((a + rng.randrange(1, 13)) % 13, b)
                    elif mutation_mode == 2:
                        replacement = (a, (b + rng.randrange(1, 9)) % 9)
                    else:
                        replacement = G.pick(rng)
                    if replacement == original:
                        continue
                    proposal = list(row[column])
                    proposal[position] = replacement
                    if all39(proposal):
                        row[column] = proposal
                        break
                else:
                    row[column] = fresh_triangle()

        if side == "B":
            # A relative coupling proposal, NOT a novelty certificate.
            row = [[((relative_u * a) % 13, (relative_w * b) % 9)
                    for a, b in cell] for cell in row]
        assert all(all39(cell) for cell in row)
        return [row]

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
        # Congestion proxy only: projected walks may reuse variables,
        # so this is not an exact Tanner eight-cycle count.
        projected_four = 0
        for i, neighbors in enumerate(adj):
            for other in adj[:i]:
                common = (neighbors & other).bit_count()
                projected_four += common * (common - 1) // 2
        projected_four //= 2
        return (bad, c4, max(0, triangles - internal),
                orbit_distance, orbit_witness, projected_four)

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

            # Elementary exchanges preserve the entire kernel basis.
            # Neutral moves explore a weight plateau without replacing
            # a verified witness reading by an optimistic estimate.
            rng.shuffle(basis)
            weights = [word.bit_count() for word in basis]
            indices = list(range(len(basis)))
            for sweep in range(2):
                rng.shuffle(indices)
                for i in indices:
                    for j in indices:
                        if i == j:
                            continue
                        trial = basis[i] ^ basis[j]
                        weight = trial.bit_count()
                        if (0 < weight < weights[i] or
                                (weight == weights[i] and
                                 rng.randrange(8) == 0)):
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
            # A verified sub-20 word already defeats this constituent
            # target. Further search cannot repair that obstruction.
            if best_weight < 20:
                break

        assert best_word
        assert all((row & best_word).bit_count() % 2 == 0
                   for row in rows)
        return best_weight, best_word

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

    component_lifts = {}

    def verified_component(grid, side):
        """XOR singleton blocks extracted only from the verified helper.

        The temporary catalogue is a lookup, not a submitted shape.
        Finalists are also checked against a fresh direct helper call.
        """
        if side not in component_lifts:
            catalogue = [[[g] for g in G.elements]]
            if side == "A":
                master, _, _ = lifted_matrices(G, catalogue, [[[]]])
            else:
                _, master, _ = lifted_matrices(G, [[[]]], catalogue)
            assert len(master) == q
            mask = (1 << q) - 1
            component_lifts[side] = tuple(
                tuple((row >> (j * q)) & mask for row in master)
                for j in range(q)
            )
        rows = [0] * q
        blocks = component_lifts[side]
        for j, cell in enumerate(grid[0]):
            for g in cell:
                for r, block_row in enumerate(blocks[G.idx(g)]):
                    rows[r] ^= block_row << (j * q)
        assert all(row.bit_length() <= width for row in rows)
        return rows

    def constituent_finalists(profile, side):
        pool = []
        seen_classes = set()
        draw_counts[side] = 0
        six_bias, eight_bias = side_policies[side]
        for trial in range(1536):
            grid = draw_grid(profile, side)
            orbit_key = constituent_class(grid)
            if orbit_key in seen_classes:
                continue
            seen_classes.add(orbit_key)
            rows = verified_component(grid, side)
            stats = component_quality(rows, width)
            if stats[0]:
                continue
            cheap = (
                int(stats[3] < 20),
                8192 * stats[1] +
                six_bias * stats[2] + eight_bias * stats[5],
                -stats[3], stats[1], stats[2], stats[5])
            pool.append(dict(
                grid=grid, rows=rows, stats=stats, cheap=cheap,
                trial=trial, orbit_key=orbit_key,
                novel_component_class=orbit_key not in reference_classes,
                reference=trial < len(references)))
        pool.sort(key=lambda item: (item["cheap"], item["trial"]))
        if not pool:
            raise ValueError("No short-dependency-free crossover constituent")

        # Cycle pressure is a proposal heuristic, not permission to
        # discard every strong reference or every exploratory class.
        selected = pool[:32]
        tail = pool[32:]
        selected += rng.sample(tail, min(24, len(tail)))
        reference_records = [item for item in pool if item["reference"]]
        for record in reference_records:
            if record not in selected:
                selected.append(record)

        def quality(record):
            return (
                record["distance"], -record["cheap"][1],
                int(record["novel_component_class"]),
                -record["stats"][1])

        shortlist = []
        for record in selected:
            record["distance"], record["witness"] = kernel_probe(
                record["rows"], width, 4, record["stats"][4])
            record["quality"] = quality(record)
            shortlist.append(record)
        shortlist.sort(key=lambda item: item["quality"], reverse=True)

        # Preserve the distinct reference classes and twelve independently
        # proposed classes. All publishable records get the same budget;
        # early termination follows only a verified sub-20 obstruction.
        finalists = list(reference_records)
        for record in shortlist:
            if record["novel_component_class"]:
                finalists.append(record)
            if sum(item["novel_component_class"]
                   for item in finalists) >= 12:
                break
        for record in shortlist:
            if len(finalists) >= 16:
                break
            if record not in finalists:
                finalists.append(record)

        for record in finalists:
            record["distance"], record["witness"] = kernel_probe(
                record["rows"], width, 32, record["witness"])
            record["quality"] = quality(record)
            if side == "A":
                direct, _, _ = lifted_matrices(
                    G, record["grid"], [[[]]])
            else:
                _, direct, _ = lifted_matrices(
                    G, [[[]]], record["grid"])
            assert direct == record["rows"]
        finalists.sort(key=lambda item: item["quality"], reverse=True)
        return finalists

    def intrinsic_signature(hx, hz, n):
        """Permutation-invariant uncoloured Tanner features.

        Different signatures certify different Tanner graphs. Equal
        signatures do not certify isomorphism. No element indices,
        row ordering or CSS colouring enter the signature.
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
        return sha256(
            b"reference-crossover-exclusion-v5:" +
            repr(signature).encode("ascii")).digest()

    # Reject the complete measured distance-20 reference codes, including
    # any image sharing their Tanner invariant. This is deliberately
    # conservative and is NOT a claim to know the remaining archive.
    forbidden_signatures = set()
    known_pairs = ((1, 0), (0, 2), (3, 4), (5, 6), (7, 8))
    for ai, bi in known_pairs:
        a_grid = [[list(cell) for cell in references[ai]]]
        b_grid = [[list(cell) for cell in references[bi]]]
        old_hx, old_hz, old_n = lifted_matrices(G, a_grid, b_grid)
        forbidden_signatures.add(
            intrinsic_signature(old_hx, old_hz, old_n))

    pairs = []
    for restart in range(3):
        aa = constituent_finalists(sa, "A")
        bb = constituent_finalists(sb, "B")
        pending = []
        for ai, a in enumerate(aa):
            for bi, b in enumerate(bb):
                weak = min(a["distance"], b["distance"])
                priority = (
                    weak, a["distance"] + b["distance"],
                    -(a["stats"][1] + b["stats"][1]), -ai, -bi)
                pending.append((priority, ai, bi))
        pending.sort(reverse=True)

        for _, ai, bi in pending:
            a, b = aa[ai], bb[bi]
            hx, hz, n = lifted_matrices(G, a["grid"], b["grid"])
            assert n == 585
            signature = intrinsic_signature(hx, hz, n)
            if signature in forbidden_signatures:
                continue
            bucket = int.from_bytes(signature[:8], "big") % N_SLOTS
            if bucket != idx:
                continue
            rx, rz = gf2_rank(hx), gf2_rank(hz)
            k = n - rx - rz
            if not (k > 0 and 7 * n <= 40 * k < 9 * n):
                continue
            assert k >= 117
            weak = min(a["distance"], b["distance"])
            quality = (
                k * weak * weak / float(n), weak,
                a["distance"] + b["distance"],
                -(a["stats"][1] + b["stats"][1]))
            pairs.append(dict(a=a, b=b, hx=hx, hz=hz, n=n,
                              k=k, quality=quality))
            if len(pairs) >= 8:
                break
        if len(pairs) >= 4:
            break

    pairs.sort(key=lambda item: item["quality"], reverse=True)
    if not pairs:
        raise ValueError("No cell-0.200 pair in the intrinsic slot bucket")

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

            for block in range(2):
                if kind == "X":
                    word = record["b"]["witness"] << (block * width)
                else:
                    word = 0
                    component_word = record["a"]["witness"]
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
        return best if found else None

    def mixed_kernel_reading(record, passes):
        """Find actual logicals in the full CSS kernel, in both sectors.

        A quotient tag is the exact canonical remainder modulo the
        stabilizer row space. Tags XOR linearly during kernel exchanges.
        Nonzero tags therefore distinguish logicals from stabilizers.
        This randomized search is not a lower bound or certificate.
        """
        n = record["n"]
        best_weight = n + 1
        best_word = None
        best_checks = best_stabilizers = None

        for checks, stabilizers in (
                (record["hz"], record["hx"]),
                (record["hx"], record["hz"])):
            stabilizer_pivots = {}
            for row in stabilizers:
                while row:
                    p = row.bit_length() - 1
                    previous = stabilizer_pivots.get(p)
                    if previous is None:
                        stabilizer_pivots[p] = row
                        break
                    row ^= previous

            def quotient_tag(word):
                remainder = 0
                while word:
                    p = word.bit_length() - 1
                    previous = stabilizer_pivots.get(p)
                    if previous is None:
                        bit = 1 << p
                        remainder |= bit
                        word ^= bit
                    else:
                        word ^= previous
                return remainder

            syndromes = columns(checks, n)
            fixed = list(stabilizers)
            fixed_count = len(fixed)

            for trial in range(passes):
                order = list(range(n))
                rng.shuffle(order)
                pivots = {}
                basis = []
                for position in order:
                    syndrome = syndromes[position]
                    word = 1 << position
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

                words = fixed + basis
                tags = [0] * fixed_count
                tags.extend(quotient_tag(word) for word in basis)
                weights = [word.bit_count() for word in words]

                for i in range(fixed_count, len(words)):
                    if tags[i] and weights[i] < best_weight:
                        best_weight, best_word = weights[i], words[i]
                        best_checks = checks
                        best_stabilizers = stabilizers

                # Retaining the original sparse stabilizers makes these
                # exchanges sensitive to mixed-sector representatives.
                # Replacing one kernel generator by its XOR with another
                # preserves the generated space together with fixed rows.
                for sweep in range(3):
                    movable = list(range(fixed_count, len(words)))
                    reducers = list(range(len(words)))
                    rng.shuffle(movable)
                    rng.shuffle(reducers)
                    for i in movable:
                        word, tag, weight = words[i], tags[i], weights[i]
                        for j in reducers:
                            if i == j or not words[j]:
                                continue
                            candidate = word ^ words[j]
                            candidate_weight = candidate.bit_count()
                            candidate_tag = tag ^ tags[j]
                            if candidate_tag and candidate_weight < best_weight:
                                best_weight = candidate_weight
                                best_word = candidate
                                best_checks = checks
                                best_stabilizers = stabilizers
                            if (candidate_weight < weight or
                                    (candidate_weight == weight and weight
                                     and rng.randrange(8) == 0)):
                                word, tag, weight = (
                                    candidate, candidate_tag, candidate_weight)
                                if weight == 0:
                                    break
                        words[i], tags[i], weights[i] = word, tag, weight

                # A short tagged generator plus two short kernel
                # generators can expose a logical missed by pair descent.
                logical_indices = sorted(
                    (i for i in range(fixed_count, len(words)) if tags[i]),
                    key=lambda i: weights[i])[:16]
                other_indices = sorted(
                    (i for i in range(len(words)) if words[i]),
                    key=lambda i: weights[i])[:32]
                for i in logical_indices:
                    for a, j in enumerate(other_indices):
                        if i == j:
                            continue
                        pair = words[i] ^ words[j]
                        pair_tag = tags[i] ^ tags[j]
                        for h in other_indices[:a]:
                            if h == i:
                                continue
                            tag = pair_tag ^ tags[h]
                            if not tag:
                                continue
                            candidate = pair ^ words[h]
                            weight = candidate.bit_count()
                            if weight < best_weight:
                                best_weight, best_word = weight, candidate
                                best_checks = checks
                                best_stabilizers = stabilizers

                # A discovered sub-20 logical already defeats this
                # candidate's target; extra work cannot repair it.
                if best_weight < 20:
                    break
            if best_weight < 20:
                break

        if best_word is None:
            return None
        assert best_word and all(
            (row & best_word).bit_count() % 2 == 0
            for row in best_checks)
        # Independently verify nontriviality for the returned witness.
        assert gf2_rank(best_stabilizers + [best_word]) > gf2_rank(
            best_stabilizers)
        return best_weight

    def update_quality(record, reading):
        previous = record.get("quantum_reading")
        if previous is not None:
            reading = previous if reading is None else min(previous, reading)
        record["quantum_reading"] = reading
        weak = (reading if reading is not None else
                min(record["a"]["distance"], record["b"]["distance"]))
        record["final_quality"] = (
            int(reading is not None),
            record["k"] * weak * weak / float(record["n"]),
            weak, record["quality"])

    # Every pair gets the same preliminary budget. A sub-20 witness
    # permits early termination because it cannot be undone by searching.
    survivors = []
    for record in pairs:
        hx, hz, n = record["hx"], record["hz"], record["n"]
        if low_logical(hz, hx, n) or low_logical(hx, hz, n):
            continue
        assert css_ok(hx, hz)
        update_quality(record, quantum_witness_reading(record))
        update_quality(record, mixed_kernel_reading(record, 4))
        survivors.append(record)
        if len(survivors) >= 8:
            break
    if not survivors:
        raise ValueError("No selected pair passed the weight-four logical screen")

    # Only equally refined finalists are eligible for publication.
    survivors.sort(key=lambda item: item["final_quality"], reverse=True)
    finalists = survivors[:4]
    for record in finalists:
        update_quality(record, mixed_kernel_reading(record, 24))
    winner = max(finalists, key=lambda item: item["final_quality"])
    return {
        "idx": idx,
        "cells": {"A": winner["a"]["grid"], "B": winner["b"]["grid"]},
        "component_probe": (
            winner["a"]["distance"], winner["b"]["distance"]),
        "quantum_witness_probe": winner["quantum_reading"],
        "k": winner["k"],
        "card": card,
        "triangle_types": prescriptions,
        "mutation_modes": mutation_modes,
        "cycle_policies": side_policies,
        "relative_automorphism": (relative_u, relative_w),
        "novel_component_classes": (
            winner["a"]["novel_component_class"],
            winner["b"]["novel_component_class"]),
        "known_distance20_signatures_excluded": len(forbidden_signatures),
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
# REFUTED: On C13:C9 q117 (1,2)x(1,2), logical-four exclusion plus embedded-constituent stabilizer descent excludes sub-18 mixed-sector logicals: parent slot 7 had d30s=15 with an X logical touching both sectors.
# REFUTED: Different triangle-spectrum cards ensure inequivalent complete codes: current slots 12 and 13 were iso_dup_of_10, and slot 25 was iso_dup_of_3.
# REFUTED: The all39 profile alone excludes sub-20 mixed-sector logicals: the best shown all39 program measured slot 2 at d30s=17 with both logical types touching both sectors.
# REFUTED: Excluding central-C3-fixed constituent words through weight 18 suffices for five measured readings >=20: the shown central-quotient-voltage program instead measured 18,18,18,18,16.
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

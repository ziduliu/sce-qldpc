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
CONCEPT = ("C13 semidirect C9 with action 3, order 117, shapes (1,2)x(1,2): cross-pairing seven measured weight-20 constituent anchors, followed by multi-witness-seeded full-CSS screening, should yield five new measured readings >=20 and fitness >=80 in cell 0.200, exceeding 75.5600. All search readings remain upper bounds.")
GROUP = ("Keep the frozen C13:C9(s=3), order 117: this bed now achieved fitness 75.5600 with four measured readings of 20. The action satisfies 3^3=1 modulo 13. Normal C13 and central C3 support structured probes, but neither subgroup structure nor noncommutativity guarantees distance.")
SHAPES = ("Keep the frozen equal (1,2) sides: each constituent has 234 variables and 117 checks, n=585, and q<=119 is allowed. Rank counting guarantees k>=117 and rate>=0.200. All five measured parent codes had k=117. Four retained weight-20 sector-1 Acol/Bcol minima; slot 2 instead exposed weight-17 mixed-sector logicals touching Arow0/Brow0. Improve constituent diversity and mixed-sector screening rather than changing this bed.")
FILLING = ("Four distinct-element trinomials retain triangle spectrum (9,9,39) and BOTH CSS row maxima nine. Seven constituent anchors come from measured distance-20 codes, replacing the old two-anchor panel. Twenty cycle cards search 1024 proposals per side using independent triangles, crossovers, and genuine a-coordinate or central-coset mutations. Same-source anchor pairs and two observed archived anchor pairings are excluded conservatively. Preserve all surviving anchor proposals alongside cycle-selected candidates, refine up to 24 constituents, and retain several explicit short kernel words. Embed separated witnesses into both full-CSS kernels; tagged exchanges preserve nonzero logical tags and occasionally traverse equal-weight same-tag plateaus. Verified singleton caching accelerates construction without changing lift conventions. Intrinsic buckets and exact logical-four rejection remain; neither the source exclusions nor the heuristic probes certify archive novelty or distance.")
FAMILY_NAME = "r020_c13sd_c9_s3_q117_12x12_panel7_seeded_quotient_v4"

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
    """Cross-pair a broader measured panel and screen seeded CSS kernels."""
    from hashlib import sha256

    # Local acceleration only: the scaffold and group remain unchanged.
    _popcount = int.bit_count
    witness_banks = {}
    sa = sb = ((3, 3),)
    assert G.q == 117
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 9 <= min(W_MAX, 9)

    spectra = ((9, 9, 9), (9, 9, 13), (9, 9, 39))
    cycle_policies = ((64, 0), (32, 1), (16, 1),
                      (8, 1), (4, 1), (2, 1))
    plans = tuple((a, b) for a in range(6) for b in range(a, 6))[:20]
    card = idx % len(plans)
    pa, pb = plans[card]
    side_policies = {"A": cycle_policies[pa], "B": cycle_policies[pb]}
    prescriptions = {"A": (2, 2), "B": (2, 2)}

    # Constituent anchors from measured distance-20 complete codes.
    # Their opposite-side use is always screened with verified lifts.
    # These are not permission to reproduce their archived joint codes.
    references = (
        (((1, 0), (10, 2), (7, 6)),
         ((7, 2), (1, 4), (5, 7))),
        (((10, 1), (1, 4), (11, 6)),
         ((6, 0), (7, 2), (0, 5))),
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
    panel_size = 2 * len(references)
    archived_source_pairs = {(3, 4), (5, 6)}
    draw_counts = {"A": 0, "B": 0}
    # x -> x^u, y -> y^v is an automorphism for these parameters.
    # Applying it only to B changes the relative LP coupling, rather
    # than applying one automorphism to the entire code.
    relative_u = pow(2, card % 4, 13)
    relative_v = (1, 4, 7)[(card // 4) % 3]

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

    def triangle_kind(cell):
        if len({G._key(g) for g in cell}) != 3:
            return None
        first, second, third = cell
        spectrum = tuple(sorted((
            orders[G.mul(G.inv(first), second)],
            orders[G.mul(G.inv(first), third)],
            orders[G.mul(G.inv(second), third)],
        )))
        return spectra.index(spectrum) if spectrum in spectra else None

    def draw_grid(profile, side):
        assert profile == sa
        trial = draw_counts[side]
        draw_counts[side] += 1

        def fresh_triangle():
            first = G.pick(rng)
            r, s = rng.choice(banks[2])
            return [first, G.mul(first, r), G.mul(first, s)]

        if trial < panel_size or trial % 5 != 4:
            source = (trial % len(references) if trial < panel_size
                      else rng.randrange(len(references)))
            variant = (trial // len(references) if trial < panel_size
                       else rng.randrange(12))
            u = pow(2, (source + variant) % 4, 13)
            v = (1, 4, 7)[(source // 2 + variant // 4) % 3]
            if side == "B":
                u = u * relative_u % 13
                v = v * relative_v % 9

            def image(cell):
                return [((u * a) % 13, (v * b) % 9) for a, b in cell]

            row = [image(cell) for cell in references[source]]
            assert all(triangle_kind(cell) == 2 for cell in row)
            original_key = tuple(
                tuple(sorted(G.idx(g) for g in cell)) for cell in row)

            if trial >= panel_size:
                # Crossover changes a constituent, not just its labels.
                if trial % 5 == 0:
                    other = rng.choice(references)
                    row[rng.randrange(2)] = image(other[rng.randrange(2)])
                for mutation in range(1 + (trial // 5) % 2):
                    column = rng.randrange(2)
                    position = rng.randrange(3)
                    original = row[column][position]
                    for attempt in range(128):
                        a, b = original
                        if rng.randrange(3):
                            replacement = ((a + rng.randrange(1, 13)) % 13, b)
                        else:
                            replacement = (a, (b + rng.choice((3, 6))) % 9)
                        proposal = list(row[column])
                        proposal[position] = replacement
                        if triangle_kind(proposal) == 2:
                            row[column] = proposal
                            break
                    else:
                        row[column] = fresh_triangle()

                # Do not call an unchanged anchor a mutant.
                current_key = tuple(
                    tuple(sorted(G.idx(g) for g in cell)) for cell in row)
                while current_key == original_key:
                    row[rng.randrange(2)] = fresh_triangle()
                    current_key = tuple(
                        tuple(sorted(G.idx(g) for g in cell)) for cell in row)
            return [row]

        row = [fresh_triangle(), fresh_triangle()]
        assert all(triangle_kind(cell) == 2 for cell in row)
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
                overlap = _popcount(s & t)
                c4 += overlap * (overlap - 1) // 2

        # Project onto check nodes. With no 4-cycles, subtracting
        # within-column triangles leaves exactly the Tanner 6-cycles.
        adj = [0] * len(rows)
        internal = 0
        for s in cols:
            degree = _popcount(s)
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
                triangles += _popcount((neighbors & adj[v]) >> (v + 1))
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
        # Projected-check-graph four-cycles are only an eight-cycle
        # congestion proxy: repeated variable nodes are not excluded.
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
        bank = witness_banks.setdefault(tuple(rows), set())
        if seed:
            bank.add(seed)
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
            # Retain multiple actual kernel words for later full-CSS
            # searches, rather than publishing only one witness seed.
            bank.add(best_word)
            bank.update(word for word in basis
                        if 0 < word.bit_count() <= 24)
            if len(bank) > 160:
                retained = sorted(
                    bank, key=lambda word: (word.bit_count(), word))[:96]
                bank.clear()
                bank.update(retained)
            # A verified sub-20 component word defeats the constituent
            # target; spending more search cannot increase its distance.
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
        """XOR singleton blocks extracted from the verified helper.

        The temporary catalogue is computational only, not a submitted
        shape. No independent group-action convention is introduced.
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
                for j in range(q))
        rows = [0] * q
        blocks = component_lifts[side]
        for j, cell in enumerate(grid[0]):
            for g in cell:
                for r, block_row in enumerate(blocks[G.idx(g)]):
                    rows[r] ^= block_row << (j * q)
        assert all(row.bit_length() <= width for row in rows)
        return rows

    def column_gauge_key(grid, side):
        """Remove elementary independent column-translation copies.

        This is only a sufficient duplicate test, not a complete
        constituent-isomorphism test. The joint Tanner bucket remains
        responsible for permutation-invariant slot separation.
        """
        normalized = []
        for cell in grid[0]:
            choices = []
            for anchor in cell:
                inverse = G.inv(anchor)
                if side == "A":
                    image = [G.mul(g, inverse) for g in cell]
                else:
                    image = [G.mul(inverse, g) for g in cell]
                choices.append(tuple(sorted(G.idx(g) for g in image)))
            normalized.append(min(choices))
        return tuple(sorted(normalized))

    def orbit_probe(rows, side, seed):
        """Search pairs/triples of regular translates of a kernel word.

        These are explicit kernel combinations, hence upper bounds.
        The search is not exhaustive over the constituent kernel.
        """
        best_word = seed
        best_weight = seed.bit_count()
        if best_weight < 20:
            return best_weight, best_word
        for refinement in range(2):
            orbit = set()
            for shift in G.elements:
                translated = 0
                bits = best_word
                while bits:
                    bit = bits & -bits
                    block, h = divmod(bit.bit_length() - 1, q)
                    element = G.elements[h]
                    image = (G.mul(element, shift) if side == "A"
                             else G.mul(shift, element))
                    translated |= 1 << (block * q + G.idx(image))
                    bits ^= bit
                orbit.add(translated)
            words = sorted(orbit)
            previous = best_weight
            for i, first in enumerate(words):
                for j in range(i):
                    pair = first ^ words[j]
                    weight = pair.bit_count()
                    if 0 < weight < best_weight:
                        best_weight, best_word = weight, pair
                    for h in range(j):
                        candidate = pair ^ words[h]
                        weight = candidate.bit_count()
                        if 0 < weight < best_weight:
                            best_weight, best_word = weight, candidate
            if best_weight == previous or best_weight < 20:
                break
        assert best_word
        assert all((row & best_word).bit_count() % 2 == 0
                   for row in rows)
        return best_weight, best_word

    def constituent_finalists(profile, side):
        pool = []
        seen = set()
        draw_counts[side] = 0
        six_bias, eight_bias = side_policies[side]
        for trial in range(1024):
            grid = draw_grid(profile, side)
            key = column_gauge_key(grid, side)
            if key in seen:
                continue
            seen.add(key)
            rows = verified_component(grid, side)
            stats = component_quality(rows, width)
            if stats[0]:
                continue
            cheap = (
                int(stats[3] < 20),
                4096 * stats[1] +
                six_bias * stats[2] + eight_bias * stats[5],
                -stats[3], stats[1], stats[2], stats[5])
            source = trial % len(references) if trial < panel_size else -1
            pool.append(dict(grid=grid, rows=rows, stats=stats,
                             cheap=cheap, trial=trial, source=source))
        pool.sort(key=lambda item: (item["cheap"], item["trial"]))
        if not pool:
            raise ValueError("No short-dependency-free all39 constituent")

        # Cycle objectives must not silently discard the anchor panel.
        references_in_pool = [
            item for item in pool if item["trial"] < panel_size]
        selected = pool[:32]
        for record in references_in_pool:
            if record not in selected:
                selected.append(record)

        shortlist = []
        for record in selected:
            record["distance"], record["witness"] = kernel_probe(
                record["rows"], width, 6, record["stats"][4])
            record["quality"] = (
                record["distance"], -record["cheap"][1],
                -record["stats"][1])
            shortlist.append(record)
        shortlist.sort(key=lambda item: item["quality"], reverse=True)

        # Retain the panel, but give anchors exactly the same actual
        # kernel-search budget as other publishable constituents.
        finalists = shortlist[:8]
        for record in references_in_pool:
            if record not in finalists:
                finalists.append(record)
        for record in shortlist:
            if len(finalists) >= 24:
                break
            if record not in finalists:
                finalists.append(record)

        # All eligible constituents get six preliminary plus 36
        # additional passes, except early rejection by a sub-20 witness.
        for record in finalists:
            record["distance"], record["witness"] = kernel_probe(
                record["rows"], width, 36, record["witness"])
            record["distance"], record["witness"] = orbit_probe(
                record["rows"], side, record["witness"])
            record["quality"] = (
                record["distance"], -record["cheap"][1],
                -record["stats"][1])
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
        digest = sha256(
            b"panel7-seeded-quotient-v4:" +
            repr(signature).encode("ascii")).digest()
        return int.from_bytes(digest[:8], "big") % N_SLOTS

    pairs = []
    seen_joint = set()
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
            sources = tuple(sorted((a["source"], b["source"])))
            if sources[0] >= 0:
                # Conservative exclusions: avoid diagonal anchor
                # pairings and both observed complete panel pairings,
                # even when a relative automorphism might change them.
                if sources[0] == sources[1] or sources in archived_source_pairs:
                    continue
            joint_key = tuple(
                tuple(tuple(sorted(G.idx(g) for g in cell))
                      for cell in component["grid"][0])
                for component in (a, b))
            if joint_key in seen_joint:
                continue
            seen_joint.add(joint_key)
            hx, hz, n = lifted_matrices(G, a["grid"], b["grid"])
            assert n == 585
            if intrinsic_bucket(hx, hz, n) != idx:
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
        # One bucket-compatible pair is not enough evidence to abandon
        # alternative constituent pools, particularly with many slots.
        if len(pairs) >= 4:
            break

    pairs.sort(key=lambda item: item["quality"], reverse=True)
    if not pairs:
        raise ValueError("No cell-0.200 pair in the intrinsic slot bucket")

    def embedded_words(record, kind):
        """Embed several separated, explicitly verified component words."""
        component = record["b"] if kind == "X" else record["a"]
        bank = set(witness_banks.get(tuple(component["rows"]), ()))
        bank.add(component["witness"])
        candidates = sorted(
            bank, key=lambda word: (word.bit_count(), word))
        least = candidates[0].bit_count()
        candidates = [
            word for word in candidates if word.bit_count() <= least + 2]
        seeds = candidates[:1]
        while len(seeds) < min(4, len(candidates)):
            remaining = [word for word in candidates if word not in seeds]
            seeds.append(max(
                remaining,
                key=lambda word: (
                    min((word ^ seed).bit_count() for seed in seeds),
                    -word.bit_count(), -word)))
        words = []
        for seed in seeds:
            assert all((row & seed).bit_count() % 2 == 0
                       for row in component["rows"])
            for block in range(2):
                if kind == "X":
                    word = seed << (block * width)
                else:
                    word = 0
                    bits = seed
                    while bits:
                        bit = bits & -bits
                        j, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << ((j * 2 + block) * q + h)
                        bits ^= bit
                words.append(word)
        return list(dict.fromkeys(words))

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

            for word in embedded_words(record, kind):
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

        for kind, checks, stabilizers in (
                ("X", record["hz"], record["hx"]),
                ("Z", record["hx"], record["hz"])):
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
                # Short constituent logicals expose useful mixed-sector
                # neighborhoods that a random information set can miss.
                basis = embedded_words(record, kind)
                assert all(
                    (row & word).bit_count() % 2 == 0
                    for word in basis for row in checks)
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
                for sweep in range(2):
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
                            # Do not collapse a logical generator into a
                            # stabilizer. Equal-weight same-tag moves can
                            # escape a greedy stabilizer-descent plateau.
                            accept = (
                                candidate_weight < weight
                                and (not tag or candidate_tag))
                            if (candidate_weight == weight and tag
                                    and candidate_tag == tag
                                    and rng.randrange(8) == 0):
                                accept = True
                            if accept:
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
        update_quality(record, mixed_kernel_reading(record, 8))
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
        "cycle_policies": side_policies,
        "relative_automorphism": (relative_u, relative_v),
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
# REFUTED: Different triangle-spectrum cards alone ensure inequivalent complete codes: the current table marks slots 12 and 13 iso_dup_of_10, and slot 25 iso_dup_of_3.
# REFUTED: The all39 two-anchor mutant/orbit policy yields five measured readings >=20: its slot 2 fell from d2s=18 to d30s=17, with both X and Z minimum witnesses touching both sectors.
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

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
CONCEPT = ("C13:C9(s=3), order 117, shapes (1,2)x(1,2): screening balanced trinomial constituents in BOTH the central invariant and trace-zero subcodes should remove the parent's sector-1 weight-16 obstruction and yield five k=125 codes retaining d30s=18, improving fitness beyond 66.325 in cell 0.200. Searches provide upper bounds, not distance certificates.")
GROUP = ("Keep the frozen order-117 C13:C9(s=3). The live bed reached fitness 67.7; this parent measured four readings of 18 and one of 16 at k=125. Since <y^3> is central C3 and characteristic two does not divide three, F2[C3] splits as F2 x F4. The parent's orbit-constant enumerator addresses the invariant summand; paired zero-parity orbit coordinates expose the complementary trace-zero summand using verified rows.")
SHAPES = ("Keep equal (1,2) sides: both constituents have 234 variables and 117 checks; n=585, guaranteed k>=117 and rate 0.200, with q<=119 permitted. Retain verified rank 115 on both constituents and full k=125, rate about 0.214. All five parental measured minima were in sector-1 Acol/Bcol blocks; the weight-16 Z family touched both A columns and both B columns, motivating complementary constituent screening rather than a shape change.")
FILLING = ("Four distinct-element balanced noncommuting trinomials retain both CSS row maxima nine. Twenty cards now include correlated central-voltage planes as well as cycle/enumerator penalties. Keep exact invariant weight-six enumeration, and add paired-orbit information-set searches of the trace-zero subcode: physical zero-parity orbit masks are eliminated against VERIFIED rows, and projected two-plus-two free-orbit collisions search for physical words of weight <=16. Enumerated candidates receive twelve primitive passes; finalists receive eighty more unless already rejected by an explicit sub-18 witness, plus the existing 3+32 full-width budget. Full lifts retain k=125, CSS, logical-four exclusion and translated stabilizer descent, now including primitive witnesses. Novelty remains a downstream test.")
FAMILY_NAME = "r020_c13sd_c9_s3_q117_12x12_rank125_c3primitive_stern_v5"

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
    """Screen both central summands before pairing balanced constituents."""
    from hashlib import sha256

    sa = sb = ((3, 3),)
    assert G.q == 117
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 9 <= min(W_MAX, 9)

    # These cards change the objective used to select complete
    # constituents, not just their anchors or an automorphism image.
    spectra = ((9, 9, 9),)
    card = idx % 20
    card_cycle_bias = (8, 16, 32, 64, 128)[card % 5]
    enumerator_penalty = (1, 4, 16, 64)[card // 5]
    prescriptions = {"A": (0, 0), "B": (0, 0)}

    # Central voltages of the two cells explore a correlated affine
    # plane. This changes the proposal distribution, not merely anchors.
    voltage_planes = tuple(
        (a, b, c, d)
        for a in range(3) for b in range(3)
        for c in range(3) for d in range(3)
        if (a * d - b * c) % 3
    )
    assert len(voltage_planes) == 48
    voltage_plan = {
        "A": voltage_planes[(7 * card + 5) % 48],
        "B": voltage_planes[(11 * card + 19) % 48],
    }

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
            spectrum = tuple(sorted((
                orders[first], orders[second],
                orders[G.mul(G.inv(first), second)],
            )))
            if spectrum not in spectra:
                continue
            kind = spectra.index(spectrum)
            if G.mul(first, second) == G.mul(second, first):
                continue
            banks[kind].append((first, second))
    if any(not bank for bank in banks):
        raise ValueError("balanced noncommuting triangle absent")

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

    def draw_grid(profile, side):
        assert profile == sa
        a, b, c, d = voltage_plan[side]
        u, v = rng.randrange(3), rng.randrange(3)
        offset = (card // 5 + int(side == "B")) % 3
        phases = (
            (u, v),
            ((a * u + b * v + offset) % 3,
             (c * u + d * v + card % 3) % 3),
        )
        row = []
        for kind, (ru, sv) in zip(prescriptions[side], phases):
            first = G.pick(rng)
            r, s = rng.choice(banks[kind])
            # Normalize the two nonzero abelianization classes, then
            # replace only their central coordinates. Noncommutation
            # depends on b modulo three and is preserved by these twists.
            if r[1] % 3 == 2:
                r, s = s, r
            assert r[1] % 3 == 1 and s[1] % 3 == 2
            r = (r[0], 1 + 3 * ru)
            s = (s[0], 2 + 3 * sv)
            assert G.mul(r, s) != G.mul(s, r)
            cell = [first, G.mul(first, r), G.mul(first, s)]
            assert len({G._key(g) for g in cell}) == 3
            assert {g[1] % 3 for g in cell} == {0, 1, 2}
            row.append(cell)
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

    def primitive_probe(rows, passes, seed=None):
        """Heuristic search in the central trace-zero constituent subcode.

        Each orbit contributes two physical even-parity masks. Their
        syndromes come exclusively from the verified constituent rows.
        Processing whole orbit pairs makes each rank increment zero or
        two: the processed space is closed under the central action.

        For each information set, projected collisions combine up to
        two free orbits from each half, with all three nonzero symbols
        per free orbit. Projection and information-set choices make
        this a heuristic search, not an exact distance exclusion.
        """
        cols = columns(rows, 2 * G.q)
        orbit_masks = []
        atoms = []
        for column in range(2):
            for orbit in central_orbits:
                positions = [column * G.q + h for h in orbit]
                bits = [1 << position for position in positions]
                orbit_masks.append(bits[0] | bits[1] | bits[2])
                atoms.append((
                    (bits[0] | bits[1],
                     cols[positions[0]] ^ cols[positions[1]]),
                    (bits[0] | bits[2],
                     cols[positions[0]] ^ cols[positions[2]]),
                ))

        best_word = seed
        best = seed.bit_count() if seed is not None else 2 * G.q + 1

        def combinations_up_to_two(groups):
            yield 0
            for group in groups:
                yield from group
            for i, first in enumerate(groups):
                for second in groups[:i]:
                    for x in first:
                        for y in second:
                            yield x ^ y

        for _ in range(passes):
            # Once an explicit sub-18 word exists, additional work
            # cannot rescue this constituent for the retention target.
            if best <= 16:
                break
            order = list(range(len(atoms)))
            rng.shuffle(order)
            pivots = {}
            free_groups = []
            independent = []
            for original in order:
                dependent = []
                for support, syndrome in atoms[original]:
                    word = support
                    while syndrome:
                        p = syndrome.bit_length() - 1
                        old = pivots.get(p)
                        if old is None:
                            pivots[p] = (syndrome, word)
                            break
                        syndrome ^= old[0]
                        word ^= old[1]
                    if not syndrome:
                        dependent.append(word)
                assert len(dependent) in (0, 2)
                if dependent:
                    first, second = dependent
                    group = (first, second, first ^ second)
                    free_groups.append(group)
                    for word in group:
                        weight = word.bit_count()
                        if weight < best:
                            best, best_word = weight, word
                else:
                    independent.append(original)

            if best <= 16:
                break
            assert free_groups and independent

            # Requiring zero on six pivot orbits is only a sampling
            # restriction. Full physical weights decide every collision.
            projection = 0
            for original in rng.sample(
                    independent, min(6, len(independent))):
                projection |= orbit_masks[original]
            rng.shuffle(free_groups)
            split = len(free_groups) // 2
            left = {}
            for word in combinations_up_to_two(free_groups[:split]):
                signature = word & projection
                left.setdefault(signature, []).append(word)

            defeated = False
            for word in combinations_up_to_two(free_groups[split:]):
                for old in left.get(word & projection, ()):
                    candidate = word ^ old
                    if not candidate:
                        continue
                    weight = candidate.bit_count()
                    if weight < best:
                        best, best_word = weight, candidate
                        if best <= 16:
                            defeated = True
                            break
                if defeated:
                    break
            if defeated:
                break

        assert best_word is not None and best_word
        assert best_word.bit_count() == best
        assert all((row & best_word).bit_count() % 2 == 0
                   for row in rows)
        assert all((mask & best_word).bit_count() % 2 == 0
                   for mask in orbit_masks)
        return best, best_word

    def central_probe(rows):
        """Exact small central enumerator, plus explicit physical words.

        First exclude quotient words of weight <=4. If none exist,
        equal-syndrome triples must be disjoint and give weight-six
        words. Keeping ALL triple representatives enumerates every such
        word; a set removes its ten unordered 3+3 decompositions.
        Exactness applies only to this invariant subcode.
        """
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
        syndromes = columns(folded, compressed_width)
        bits = [1 << i for i in range(compressed_width)]

        def expand(word):
            physical = 0
            while word:
                bit = word & -word
                physical |= masks[bit.bit_length() - 1]
                word ^= bit
            assert all((row & physical).bit_count() % 2 == 0
                       for row in rows)
            return physical

        representatives = {0: 0}
        short = None
        for i, syndrome in enumerate(syndromes):
            word = bits[i]
            previous = representatives.get(syndrome)
            if previous is not None:
                short = word ^ previous
                break
            representatives[syndrome] = word
        if short is None:
            for i in range(compressed_width):
                for j in range(i):
                    syndrome = syndromes[i] ^ syndromes[j]
                    word = bits[i] | bits[j]
                    previous = representatives.get(syndrome)
                    if previous is not None:
                        short = word ^ previous
                        break
                    representatives[syndrome] = word
                if short is not None:
                    break
        if short is not None:
            physical = expand(short)
            assert 0 < physical.bit_count() <= 12
            return physical.bit_count(), physical, 10**9, (physical,)

        triples = {}
        words6 = set()
        for i in range(compressed_width):
            for j in range(i):
                syndrome2 = syndromes[i] ^ syndromes[j]
                word2 = bits[i] | bits[j]
                for h in range(j):
                    syndrome = syndrome2 ^ syndromes[h]
                    word = word2 | bits[h]
                    previous = triples.get(syndrome)
                    if previous is None:
                        triples[syndrome] = [word]
                    else:
                        for old in previous:
                            candidate = word ^ old
                            assert candidate.bit_count() == 6
                            words6.add(candidate)
                        previous.append(word)

        if words6:
            # Enumeration is complete before witness sampling.
            multiplicity = len(words6)
            ordered = sorted(words6)
            chosen = rng.sample(ordered, min(16, len(ordered)))
            physical_words = tuple(expand(word) for word in chosen)
            assert all(word.bit_count() == 18 for word in physical_words)
            return 18, physical_words[0], multiplicity, physical_words

        seed = (1 << compressed_width) - 1
        assert all((row & seed).bit_count() % 2 == 0 for row in folded)
        reading, compressed_word = kernel_probe(
            folded, compressed_width, 4, seed)
        assert reading > 6
        physical = expand(compressed_word)
        assert physical.bit_count() == 3 * reading
        return 3 * reading, physical, 0, (physical,)

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
        pool = []
        seen = set()
        cycle_bias = card_cycle_bias + 4 * int(side == "B")
        for trial in range(256):
            grid = draw_grid(profile, side)
            key = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for cell in grid[0])
            if key in seen:
                continue
            seen.add(key)
            rows = verified_component(grid, side)
            if gf2_rank(rows) != q - 2:
                continue
            stats = component_quality(rows, width)
            if stats[0]:
                continue
            cheap = (int(stats[3] <= 13),
                     cycle_bias * stats[1] + stats[2],
                     -stats[3], stats[1])
            pool.append(dict(grid=grid, rows=rows, stats=stats,
                             cheap=cheap, trial=trial))
        pool.sort(key=lambda item: (item["cheap"], item["trial"]))
        if not pool:
            raise ValueError("No rank-115 balanced noncommuting constituent")

        enumerated = []
        for record in pool[:64]:
            distance, word, count6, words = central_probe(record["rows"])
            record["central_distance"] = distance
            record["central_witness"] = word
            record["central_count6"] = count6
            record["central_words"] = words
            primitive_distance, primitive_word = primitive_probe(
                record["rows"], 12)
            record["primitive_distance"] = primitive_distance
            record["primitive_witness"] = primitive_word
            record["enum_cost"] = (
                enumerator_penalty * count6 + record["cheap"][1])
            enumerated.append(record)

        # An invariant reading of 18 says nothing about the other
        # central summand. Prefer candidates not explicitly defeated
        # in either summand before spending the full-width budget.
        clean = [
            record for record in enumerated
            if min(record["central_distance"],
                   record["primitive_distance"]) >= 18
        ]
        if clean:
            enumerated = clean
        enumerated.sort(key=lambda item: (
            -min(item["central_distance"], item["primitive_distance"]),
            item["enum_cost"], item["cheap"], item["trial"]))

        shortlist = []
        for record in enumerated[:24]:
            seed = min((
                record["stats"][4], record["central_witness"],
                record["primitive_witness"]), key=int.bit_count)
            record["distance"], record["witness"] = kernel_probe(
                record["rows"], width, 3, seed)
            record["quality"] = (
                record["distance"], -record["enum_cost"],
                -record["central_count6"], -record["stats"][1])
            shortlist.append(record)
        shortlist.sort(key=lambda item: item["quality"], reverse=True)

        # Keep the same 3+32 full-width refinement. Complement it with
        # 12+80 primitive information sets, stopping primitive searches
        # early only when an actual sub-18 witness already defeats them.
        finalists = shortlist[:12]
        for record in finalists:
            primitive_distance, primitive_word = primitive_probe(
                record["rows"], 80, record["primitive_witness"])
            record["primitive_distance"] = primitive_distance
            record["primitive_witness"] = primitive_word
            seed = min(
                (record["witness"], primitive_word), key=int.bit_count)
            record["distance"], record["witness"] = kernel_probe(
                record["rows"], width, 32, seed)
            record["quality"] = (
                record["distance"], -record["enum_cost"],
                -record["central_count6"], -record["stats"][1])
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
    for restart in range(3):
        aa = constituent_finalists(sa, "A")
        bb = constituent_finalists(sb, "B")
        pending = []
        for ai, a in enumerate(aa):
            for bi, b in enumerate(bb):
                weak = min(a["distance"], b["distance"])
                priority = (
                    weak, a["distance"] + b["distance"],
                    -(a["enum_cost"] + b["enum_cost"]),
                    -(a["central_count6"] + b["central_count6"]),
                    -ai, -bi)
                pending.append((priority, ai, bi))
        pending.sort(reverse=True)

        for _, ai, bi in pending:
            a, b = aa[ai], bb[bi]
            hx, hz, n = lifted_matrices(G, a["grid"], b["grid"])
            assert n == 585
            rx, rz = gf2_rank(hx), gf2_rank(hz)
            k = n - rx - rz
            if k != 125:
                continue
            assert 7 * n <= 40 * k < 9 * n
            bucket = intrinsic_bucket(hx, hz, n)
            weak = min(a["distance"], b["distance"])
            quality = (
                k * weak * weak / float(n), weak,
                a["distance"] + b["distance"],
                -(a["enum_cost"] + b["enum_cost"]),
                int(bucket == idx))
            pairs.append(dict(a=a, b=b, hx=hx, hz=hz, n=n,
                              k=k, quality=quality, bucket=bucket))
            if len(pairs) >= 8:
                break
        if pairs:
            break

    pairs.sort(key=lambda item: item["quality"], reverse=True)
    if not pairs:
        raise ValueError("No verified rank-125 pair in cell 0.200")

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

            component = record["b"] if kind == "X" else record["a"]
            sources = list(dict.fromkeys(
                [component["witness"], component["central_witness"],
                 component["primitive_witness"]]
                + list(component["central_words"])))
            if len(sources) > 8:
                # Never discard the complementary-summand witness
                # when sampling the invariant minimum family.
                sources = sources[:3] + rng.sample(sources[3:], 5)

            # Constituent regular translations are not presumed to be
            # symmetries of the full quantum code. Verify every embedding.
            orbit_words = set()
            for translate in G.elements:
                permutation = [
                    G.idx(G.mul(translate, h) if kind == "X"
                          else G.mul(h, translate))
                    for h in G.elements
                ]
                for source in sources:
                    translated = 0
                    remaining = source
                    while remaining:
                        bit = remaining & -remaining
                        column, h = divmod(bit.bit_length() - 1, q)
                        translated |= 1 << (
                            column * q + permutation[h])
                        remaining ^= bit
                    orbit_words.add(translated)

            seeds = {}
            for component_word in sorted(orbit_words):
                for block in range(2):
                    if kind == "X":
                        word = component_word << (block * width)
                    else:
                        word = 0
                        remaining = component_word
                        while remaining:
                            bit = remaining & -remaining
                            column, h = divmod(bit.bit_length() - 1, q)
                            word |= 1 << ((column * 2 + block) * q + h)
                            remaining ^= bit
                    if any((row & word).bit_count() & 1 for row in checks):
                        raise ValueError("Translated witness embedding failed")
                    if not nontrivial(word):
                        continue
                    weight = word.bit_count()
                    for _ in range(2):
                        starting = word
                        for stabilizer in stabilizers:
                            trial = starting ^ stabilizer
                            trial_weight = trial.bit_count()
                            if trial_weight < weight:
                                word, weight = trial, trial_weight
                        if word == starting:
                            break
                    seeds[word] = weight

            selected = sorted(
                seeds, key=lambda word: (seeds[word], word))[:8]
            for word in selected:
                assert nontrivial(word)
                found = True
                weight = word.bit_count()
                # These moves preserve the exact logical class.
                # Higher-order moves are deliberately budgeted, not
                # claimed to exhaust all stabilizer combinations.
                from itertools import combinations
                for descent in range(3):
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
                    hot = sorted(
                        stabilizers,
                        key=lambda row: (
                            2 * (starting & row).bit_count()
                            - row.bit_count()),
                        reverse=True)[:32]
                    for first, second, third in combinations(hot, 3):
                        trial = starting ^ first ^ second ^ third
                        trial_weight = trial.bit_count()
                        if trial_weight < weight:
                            word, weight = trial, trial_weight
                    for first, second, third, fourth in combinations(
                            hot[:18], 4):
                        trial = starting ^ first ^ second ^ third ^ fourth
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
        weak = (reading if reading is not None else
                min(record["a"]["distance"], record["b"]["distance"]))
        record["final_quality"] = (
            int(reading is not None),
            record["k"] * weak * weak / float(n),
            weak, record["quality"])
        survivors.append(record)
        if len(survivors) >= 8:
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
        "primitive_component_probe": (
            winner["a"]["primitive_distance"],
            winner["b"]["primitive_distance"]),
        "central_voltage_planes": voltage_plan,
        "k": winner["k"],
        "card": card,
        "triangle_types": prescriptions,
        "central_weight6_multiplicities": (
            winner["a"]["central_count6"],
            winner["b"]["central_count6"]),
        "enumerator_penalty": enumerator_penalty,
        "cycle_penalty": card_cycle_bias,
        "intrinsic_bucket": winner["bucket"],
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
# REFUTED: The normal-C39/central-C3 v2 model raises measured minima beyond 18: its measured readings were 18,18,18,18,17, all at k=117, and many other distinct slots screened at six.
# REFUTED: Balanced rank-125 constituents plus long kernel probing and local witness descent suffice to retain five readings of 18: D1 measured 16,16,9,18,17, including mixed-sector failures.
# REFUTED: The v4 central weight-six enumerator and translated-witness policy retains five measured readings of 18: this parent measured 16,18,18,18,18 at k=125; its weight-16 Z witnesses were entirely in sector 1.
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

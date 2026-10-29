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
CONCEPT = ("SD32=C16 semidirect_7 C2, order 32, A=(1,2), B=(3,7): canonical isolated/split cubic incidence patterns and verified two-cubic/core witness screening should supply five measured readings >=10 in cell 0.250; this is falsifiable, not a distance certificate.")
GROUP = ("Jump from Dic8 to semidihedral SD32. The action 7 satisfies 7^2=1 modulo 16; y has order two and the group has multiple involutions, unlike Dic8's unique involution. The order preserves n=544. Related order-32 beds reached fitness 21.4/22.0 on other shapes, while this parent's five finalists read 9,9,9,8,8. No group-distance advantage is assumed.")
SHAPES = ("Retain A=(1,2), B=(3,7): constituents have 64 variables/32 checks and 224 variables/96 checks. At q=32, n=544, kb=128 and guaranteed rate 4/17=0.235294, inside cell 0.250; q<=35 is allowed by size. Exact full ranks retain k=128. Parent minima predominantly used sector-1 Acol0/1 across Bcols, with occasional mixed-sector X witnesses involving Brow1/2; retain the complete logical-four gate.")
FILLING = ("Twenty cards enumerate canonical four-column cubic count patterns around a three-matching core, including isolated trinomials and split binomials absent from the parent's two-profile catalogue. A has two mixed-coset trinomials; B rows total six and columns have degree two/three, so BOTH check maxima remain nine, also below twelve. Verified B columns explicitly produce core-cycle and two-cubic T-join witnesses. Their actual supports seed equally budgeted kernel refinement before full CSS rank, logical-four and stabilizer-pair screens. Intrinsic full-Tanner exclusions remain conservative; neither larger distance nor unseen-archive novelty is certified.")
FAMILY_NAME = "r025_sd32_q32_12x37_canonical_cubic_tjoin_witness_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("SD32 has order 32")
    # x^16=y^2=1, y*x*y^-1=x^7.
    # 7^2=1 modulo 16, so this is C16 semidirect_7 C2.
    els = [(a, b) for b in range(2) for a in range(16)]

    def red(g):
        return (int(g[0]) % 16, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (7 if b else 1) * c) % 16, (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-(7 if b else 1) * a) % 16, b)

    def idx(g):
        a, b = red(g)
        return 16 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (3, 7)               # (rb, cb)


from functools import lru_cache


@lru_cache(maxsize=1)
def _r025_catalogue(G, count):
    """Pure deterministic portfolio; cached published cells are immutable."""
    from collections import Counter

    from itertools import combinations_with_replacement, permutations, product

    assert G.q == 32
    assert A_SHAPE == (1, 2) and B_SHAPE == (3, 7)
    rng = random.Random(BASE_SEED ^ 0x1237532)
    q = G.q
    core_counts = ((1, 0, 1), (1, 1, 0), (0, 1, 1))

    # Each tail column has three terms and each tail row four.
    # Canonicalize row and tail-column permutations before making cards.
    triples = [v for v in product(range(4), repeat=3) if sum(v) == 3]
    tails = set()
    for cols in combinations_with_replacement(triples, 4):
        if any(sum(col[i] for col in cols) != 4 for i in range(3)):
            continue
        key = min(
            tuple(sorted(tuple(col[i] for i in perm) for col in cols))
            for perm in permutations(range(3)))
        tails.add(key)
    tails = sorted(tails, key=lambda cols: (
        -sum(v == 3 for col in cols for v in col),
        -sum(v == 0 for col in cols for v in col), cols))
    profiles = tuple(
        tuple(core_counts[i] + tuple(col[i] for col in tail)
              for i in range(3))
        for tail in tails)
    assert len(profiles) >= 3

    sa = ((3, 3),)
    for sb in profiles:
        assert _no_dead_lines(sa, sb)
        assert all(sum(row) == 6 for row in sb)
        assert tuple(sum(sb[i][j] for i in range(3))
                     for j in range(7)) == (2, 2, 2, 3, 3, 3, 3)
        wx = max(sum(sa[0]) + sum(sb[i][j] for i in range(3))
                 for j in range(7))
        wz = max(sum(row) + sa[0][j]
                 for row in sb for j in range(2))
        assert wx == wz == 9
        assert max(wx, wz) <= min(W_MAX, 9, 12)

    rotations = {
        16: [a for a in range(1, 16, 2)],
        8: [a for a in range(2, 16, 4)],
    }
    coset_pools = (
        [g for g in G.elements if g[1] == 0],
        [g for g in G.elements if g[1] == 1],
    )
    # In SD32, (a,1)^2=(8*a,0): odd a gives order four.
    cross4 = [(a, 1) for a in range(1, 16, 2)]

    def draw_cell(count, order, phase):
        if not count:
            return []
        first = G.pick(rng)
        if count == 1:
            return [rng.choice(coset_pools[phase % 2])]
        if count == 2:
            relative = (rng.choice(cross4) if phase % 3 == 0 else
                        (rng.choice(rotations[order]), 0))
            cell = [first, G.mul(first, relative)]
        else:
            # Mixed, noncommuting trinomials, including isolated
            # cubic columns absent from the parent's support model.
            cell = [first,
                    G.mul(first, (rng.choice(rotations[order]), 0)),
                    G.mul(first, rng.choice(cross4))]
        assert len({G._key(g) for g in cell}) == count
        return cell

    def draw_grid(side, card):
        profile = card % len(profiles)
        policy = card // len(profiles)
        if side == "A":
            return [[
                draw_cell(3, 8 if (policy + j) % 3 == 1 else 16,
                          card + j)
                for j in range(2)]]

        C = [[[] for _ in range(7)] for _ in range(3)]
        order = 16 if policy % 2 == 0 else 8
        voltage = (rng.choice(rotations[order]), 0)
        C[0][0] = C[1][0] = [G.e]
        C[1][1] = C[2][1] = [G.e]
        C[2][2], C[0][2] = [G.e], [voltage]
        for i in range(3):
            for j in range(3, 7):
                count_here = profiles[profile][i][j]
                relative_order = 16 if (card + i + j) % 2 == 0 else 8
                C[i][j] = draw_cell(
                    count_here, relative_order, card + i + 2 * j)
        assert tuple(tuple(len(cell) for cell in row)
                     for row in C) == profiles[profile]
        return C

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
        return bad, c4, max(0, triangles - internal)

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

    def row_basis(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = word
                    break
                word ^= previous
        return pivots

    def outside(word, pivots):
        while word:
            p = word.bit_length() - 1
            previous = pivots.get(p)
            if previous is None:
                return True
            word ^= previous
        return False

    def kernel_probe(rows, width, passes, seed=0):
        """Actual kernel witnesses; failure to improve is not a lower bound."""
        cols = columns(rows, width)
        best_word = seed
        best = seed.bit_count() if seed else width + 1

        def inspect(word):
            nonlocal best, best_word
            if word and word.bit_count() < best:
                best, best_word = word.bit_count(), word

        order = list(range(width))
        for _ in range(passes):
            rng.shuffle(order)
            pivots, basis = {}, []
            for j in order:
                syndrome, word = cols[j], 1 << j
                while syndrome:
                    p = syndrome.bit_length() - 1
                    previous = pivots.get(p)
                    if previous is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= previous[0]
                    word ^= previous[1]
                if not syndrome:
                    basis.append(word)
                    inspect(word)

            # Elementary exchanges preserve the complete kernel basis.
            rng.shuffle(basis)
            for i in range(len(basis)):
                weight = basis[i].bit_count()
                for j in range(len(basis)):
                    if i == j:
                        continue
                    trial = basis[i] ^ basis[j]
                    trial_weight = trial.bit_count()
                    if 0 < trial_weight < weight:
                        basis[i], weight = trial, trial_weight
                inspect(basis[i])

            for i, first in enumerate(basis):
                for second in basis[:i]:
                    inspect(first ^ second)
            short = sorted(basis, key=int.bit_count)[:32]
            for i, first in enumerate(short):
                for j in range(i):
                    pair = first ^ short[j]
                    for h in range(j):
                        inspect(pair ^ short[h])
            for descent in range(2):
                starting = best_word
                for i, first in enumerate(short):
                    inspect(starting ^ first)
                    for second in short[:i]:
                        inspect(starting ^ first ^ second)
                if best_word == starting:
                    break
            if best <= 6:
                break

        assert best_word
        assert all((row & best_word).bit_count() % 2 == 0 for row in rows)
        return best, best_word

    def verified_component(grid, side):
        # Zero auxiliary cells expose exactly the scaffold's action.
        zero = [[[]]]
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, zero)
            width = A_SHAPE[1] * q
        else:
            _, rows, _ = lifted_matrices(G, zero, grid)
            width = B_SHAPE[1] * q
        assert all(row.bit_length() <= width for row in rows)
        return rows, width

    def core_pair_witness(rows, width):
        """Actual B words using core edges and zero/two cubic variables.

        All incidences come from verified rows. Minimum boundary joins
        are obtained by shortest-path perfect matching. This restricted
        witness search is not a complete component-distance calculation.
        """
        cols = columns(rows, width)
        checks = len(rows)
        adj = [[] for _ in range(checks)]
        cubic = []
        for j, syndrome in enumerate(cols):
            degree = syndrome.bit_count()
            if degree == 3:
                cubic.append(j)
                continue
            assert degree == 2
            bit = syndrome & -syndrome
            a = bit.bit_length() - 1
            b = (syndrome ^ bit).bit_length() - 1
            adj[a].append((b, j))
            adj[b].append((a, j))

        best, best_word = width + 1, 0

        def inspect(word):
            nonlocal best, best_word
            weight = word.bit_count()
            if word and weight < best:
                best, best_word = weight, word

        distances, paths = [], []
        for root in range(checks):
            distance = [10000] * checks
            parent_edge = [-1] * checks
            path = [0] * checks
            distance[root] = 0
            queue = [root]
            for a in queue:
                for b, edge in adj[a]:
                    if distance[b] == 10000:
                        distance[b] = distance[a] + 1
                        parent_edge[b] = edge
                        path[b] = path[a] ^ (1 << edge)
                        queue.append(b)
                    elif (edge != parent_edge[a]
                          and edge != parent_edge[b]):
                        inspect(path[a] ^ path[b] ^ (1 << edge))
            distances.append(distance)
            paths.append(path)

        memo = {0: (0, 0)}

        def join(boundary):
            old = memo.get(boundary)
            if old is not None:
                return old
            bit = boundary & -boundary
            a = bit.bit_length() - 1
            rest = boundary ^ bit
            choices = rest
            answer = (10000, 0)
            while choices:
                other = choices & -choices
                b = other.bit_length() - 1
                distance = distances[a][b]
                if distance < 10000:
                    cost, word = join(rest ^ other)
                    candidate = (distance + cost, paths[a][b] ^ word)
                    if candidate[0] < answer[0]:
                        answer = candidate
                choices ^= other
            memo[boundary] = answer
            return answer

        # Left regular translations preserve the verified right-action
        # component, so these anchors cover all two-cubic support orbits.
        anchors = [
            j * q + G.idx(G.e) for j in range(B_SHAPE[1])
            if cols[j * q + G.idx(G.e)].bit_count() == 3]
        for anchor in anchors:
            for other in cubic:
                if anchor == other:
                    continue
                boundary = cols[anchor] ^ cols[other]
                assert boundary.bit_count() % 2 == 0
                cost, core_word = join(boundary)
                if cost < 10000:
                    inspect(core_word ^ (1 << anchor) ^ (1 << other))

        assert best_word
        assert best == best_word.bit_count()
        assert all((row & best_word).bit_count() % 2 == 0
                   for row in rows)
        return best, best_word

    def select_components(side, card):
        pool = []
        cycle_bias = 16 + 8 * ((card // 2) % 3)
        trials = 40 if side == "A" else 96
        for trial in range(trials):
            grid = draw_grid(side, card)
            rows, width = verified_component(grid, side)
            stats = component_quality(rows, width)
            if stats[0]:
                continue
            cost = cycle_bias * stats[1] + stats[2]
            pair_reading, pair_word = (
                core_pair_witness(rows, width) if side == "B"
                else (width + 1, 0))
            pool.append(dict(
                grid=grid, rows=rows, width=width,
                cost=cost, cycles=stats[1], trial=trial,
                pair_reading=pair_reading, pair_word=pair_word))
        # Prefer fewer exposed light words before cycle tie-breaks.
        # A high restricted reading is not a distance lower bound.
        pool.sort(key=lambda r: (
            -r["pair_reading"], r["cost"], r["trial"]))

        shortlist = []
        for record in pool:
            if gf2_rank(record["rows"]) != len(record["rows"]):
                continue
            record["reading"], record["witness"] = kernel_probe(
                record["rows"], record["width"], 4, record["pair_word"])
            shortlist.append(record)
            if len(shortlist) >= 8:
                break
        shortlist.sort(key=lambda r: (
            r["reading"], -r["cost"], -r["trial"]), reverse=True)

        # Every retained constituent gets the same 4+16-pass budget.
        # Explicit witnesses are carried forward, so readings cannot rise.
        finalists = shortlist[:4]
        for record in finalists:
            record["reading"], record["witness"] = kernel_probe(
                record["rows"], record["width"], 16, record["witness"])
        finalists.sort(key=lambda r: (
            r["reading"], -r["cost"], -r["trial"]), reverse=True)
        return finalists

    def graph_signature(hx, hz, width):
        """Conservative invariant of the complete uncoloured Tanner graph.

        Different signatures separate Tanner isomorphisms. Equality
        is inconclusive, so equal signatures are conservatively rejected.
        No claim about the evaluator's unseen archive is made.
        """
        rows = list(hx) + list(hz)
        cols = columns(rows, width)
        incidence, shared = [], Counter()
        degrees = [row.bit_count() for row in rows]
        for col in cols:
            checks = []
            while col:
                bit = col & -col
                checks.append(bit.bit_length() - 1)
                col ^= bit
            incidence.append(checks)
            for i, a in enumerate(checks):
                for b in checks[:i]:
                    shared[(min(a, b), max(a, b))] += 1
        adjacency = [{} for _ in rows]
        for (a, b), multiplicity in shared.items():
            adjacency[a][b] = multiplicity
            adjacency[b][a] = multiplicity
        masks = [sum(1 << j for j in neighbors)
                 for neighbors in adjacency]

        features = []
        for i, neighbors in enumerate(adjacency):
            profile = Counter(
                (degrees[j], multiplicity,
                 (masks[i] & masks[j]).bit_count())
                for j, multiplicity in neighbors.items())
            walk4 = sum((masks[i] & other).bit_count() ** 2
                        for other in masks)
            features.append(
                (degrees[i], tuple(sorted(profile.items())), walk4))
        palette = {feature: c for c, feature in
                   enumerate(sorted(set(features)))}
        colors = [palette[feature] for feature in features]
        qfeatures = [
            tuple(sorted(colors[r] for r in checks))
            for checks in incidence]
        qpalette = {feature: c for c, feature in
                    enumerate(sorted(set(qfeatures)))}
        check_profiles = [[] for _ in rows]
        for checks, feature in zip(incidence, qfeatures):
            for r in checks:
                check_profiles[r].append(qpalette[feature])
        refined = [
            (colors[r], tuple(sorted(profile)))
            for r, profile in enumerate(check_profiles)]
        return (
            width, len(rows),
            tuple(sorted(Counter(features).items())),
            tuple(sorted(Counter(qfeatures).items())),
            tuple(sorted(Counter(refined).items())))

    def logical_reading(a, b, hx, hz, bx, bz, n):
        """Embed, verify and reduce explicit nontrivial X/Z witnesses."""
        best = n + 1
        found = False
        for kind, checks, stabilizers, own in (
                ("X", hz, hx, bx), ("Z", hx, hz, bz)):
            blocks = range(A_SHAPE[1]) if kind == "X" else range(B_SHAPE[1])
            for block in blocks:
                if kind == "X":
                    word = b["witness"] << (block * B_SHAPE[1] * q)
                else:
                    source, word = a["witness"], 0
                    while source:
                        bit = source & -source
                        j, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << ((j * B_SHAPE[1] + block) * q + h)
                        source ^= bit
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                if not outside(word, own):
                    continue
                found = True
                weight = word.bit_count()
                for descent in range(2):
                    starting = word
                    for i, first in enumerate(stabilizers):
                        trial = starting ^ first
                        trial_weight = trial.bit_count()
                        if trial_weight < weight:
                            word, weight = trial, trial_weight
                        for second in stabilizers[:i]:
                            trial = starting ^ first ^ second
                            trial_weight = trial.bit_count()
                            if trial_weight < weight:
                                word, weight = trial, trial_weight
                    if word == starting:
                        break
                assert word and outside(word, own)
                assert all((row & word).bit_count() % 2 == 0
                           for row in checks)
                best = min(best, weight)
        return best if found else None

    results, used = [], set()
    # Twenty structural cards, extended with independently optimized
    # unused fillings if the harness requests more slots. Failed cards
    # do not invalidate already constructed catalogue entries.
    for design in range(max(80, 4 * count)):
        card = design % 20
        winner = None
        examined = set()
        for batch in range(2):
            aa = select_components("A", card)
            bb = select_components("B", card)
            plans = []
            for ai, a in enumerate(aa):
                for bi, b in enumerate(bb):
                    quality = (
                        min(a["reading"], b["reading"]),
                        a["reading"] + b["reading"],
                        -a["cost"] - b["cost"])
                    plans.append((quality, ai, bi))
            plans.sort(reverse=True)
            admitted = 0
            for preliminary, ai, bi in plans:
                a, b = aa[ai], bb[bi]
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                # Admission always uses the evaluator's exact ranks.
                if n != 544:
                    raise ValueError("new shape has incorrect size")
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if k != 4 * q or not (225 * n <= 1000 * k < 275 * n):
                    continue
                if max(row.bit_count() for row in hx + hz) > min(W_MAX, 9):
                    continue
                if low_logical(hz, hx, n) or low_logical(hx, hz, n):
                    continue
                signature = graph_signature(hx, hz, n)
                if signature in used or signature in examined:
                    continue
                examined.add(signature)
                bx, bz = row_basis(hx), row_basis(hz)
                reading = logical_reading(a, b, hx, hz, bx, bz, n)
                if reading is None:
                    continue
                quality = (
                    k * reading * reading / n,
                    reading, preliminary, -a["cycles"] - b["cycles"])
                record = dict(
                    quality=quality, A=A, B=B, hx=hx, hz=hz,
                    k=k, reading=reading, signature=signature, card=card)
                if winner is None or quality > winner["quality"]:
                    winner = record
                admitted += 1
                if admitted >= 3:
                    break
            if winner is not None and winner["reading"] >= 9:
                break

        if winner is None:
            continue
        assert css_ok(winner["hx"], winner["hz"])
        assert lifted_k(G, winner["A"], winner["B"]) == winner["k"]
        used.add(winner["signature"])
        frozen_A = tuple(tuple(tuple(cell) for cell in row)
                         for row in winner["A"])
        frozen_B = tuple(tuple(tuple(cell) for cell in row)
                         for row in winner["B"])
        results.append((
            frozen_A, frozen_B, winner["k"], winner["reading"],
            winner["card"], winner["quality"]))
        if len(results) >= count:
            break
    return tuple(results)


def NEW_CTX(rng, G, idx):
    """Publish one immutable screened design; no post-screen resampling."""
    catalogue = _r025_catalogue(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("distinct target-rate catalogue exhausted")
    A, B, k, reading, card, quality = catalogue[idx]
    return {
        "idx": idx, "cells": {"A": A, "B": B},
        "screen_k": k, "logical_witness_upper_bound": reading,
        "design_card": card, "screen_quality": quality,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh list from the already screened complete filling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Rolled uniform Z21 fillings on (2,4)x(2,6) suffice for d30s>4: refuted by all five measured parent codes having d30s=4, with sector-1 witnesses.
# REFUTED: Weight-four logical rejection and component cycle minimization eliminate weight-20 mixed-sector logicals on Dic11 (2,3)x(2,3): measured parent slot 42 had d30s=20 with Z witnesses touching both sectors.
# REFUTED: Dic8 long-core/cubic selection supplies five measured readings >=9: current measured slots 14 and 15 instead had d30s=8, with sector-1 Z witnesses.
# REFUTED: The Dic8 two-profile portfolio supplies twenty inequivalent designs: sixty submitted slots collapsed to five distinct codes under evaluator isomorphism deduplication.
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

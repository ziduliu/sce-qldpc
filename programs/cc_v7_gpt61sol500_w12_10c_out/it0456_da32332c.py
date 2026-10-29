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
CONCEPT = ("C7 semidirect C6 with action 3, order 42, A=(1,3), B=(1,4): compatible mixed cubic/quartic A images should produce fresh half-rate codes with k>272 and no logical below weight four, improving the measured five [546,272,4] plateau.")
GROUP = ("Frozen byte-for-byte under L1. This exact bed measured fitness 7.9710, k=272 and distance-four readings in all five leaders. Its demonstrated rank surplus motivates testing larger compatible cokernels, not assuming a group-driven distance improvement.")
SHAPES = ("Frozen (1,3)x(1,4): n=546, kb=252, guaranteed rate 6/13, and components with 126/168 variables and 42 checks each. The size window permits q<=46; the frozen group fixes q=42. Parent minimum witnesses occupied sector-one Acol/Bcol blocks broadly in both types, with no reported Arow/Brow minima. Exact complete ranks retain 0.475<=k/n<0.525.")
FILLING = ("Compile normalized binomial, cubic and quartic A blocks exclusively from verified unit lifts. Assemble compatible [3,3,4], [2,3,4], [2,4,4] and [2,3,3] profiles, never the archived parent's all-cubic architecture. Enumerate connected four-binomial B geometries using order-six and order-seven relatives. Twenty cards cross five A support/quotient preferences with four B quartet-load preferences. Exact stabilizer-aware syndrome collisions exclude logicals through three and classify all quartets; reward a five-weight exclusion floor only when every quartet collision is a stabilizer. Optimize actual k and reject repeated uncoloured Tanner invariants. All row caps remain twelve; no parent-bank fallback is called.")
FAMILY_NAME = "r050_C7C6s3_13x14_mixed_quartic_images_logical4_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    # x^7=y^6=1, y*x*y^-1=x^3; 3 has order six modulo seven.
    if not valid_q(q):
        raise ValueError("C7:C6 with action 3 requires order 42")
    els = [(a, b) for b in range(6) for a in range(7)]
    powers = [pow(3, b, 7) for b in range(6)]

    def red(g):
        return (g[0] % 7, g[1] % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b]*c) % 7, (b+d) % 6)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 6
        return ((-powers[bi]*a) % 7, bi)

    def idx(g):
        a, b = red(g)
        return 7*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


from functools import lru_cache

# Quotient occupancy is fixed, but relative rotation patterns differ.
_A_PATTERNS = ((1,1,1), (1,1,2), (1,1,3), (1,2,3), (1,2,4))
_B_TRIPLES = ((0,2,4), (0,1,2), (0,1,5), (0,0,1))


def _qcolumns(rows, width):
    cols = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length()-1] |= marker
            row ^= bit
    return cols


def _qcomponents(hx, hz, q):
    """Extract both actions only from the scaffold's verified lift."""
    mask = (1 << q)-1
    ar = [
        sum(((row >> (j*4*q)) & mask) << (j*q)
            for j in range(3))
        for row in hx[:q]
    ]
    br = [row & ((1 << (4*q))-1) for row in hz[:q]]
    return ar, br, _qcolumns(ar, 3*q), _qcolumns(br, 4*q)


def _qfloor(cols):
    """Exact kernel exclusion through four; five means not excluded."""
    singles = set(cols)
    if 0 in singles:
        return 1
    if len(singles) != len(cols):
        return 2
    pairs = set()
    four = False
    for i, x in enumerate(cols):
        for j in range(i):
            syndrome = x ^ cols[j]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                four = True
            pairs.add(syndrome)
    return 4 if four else 5


def _qfive(cols):
    """Exact weight-five test after kernel weights <=4 were excluded."""
    pairs = {cols[i] ^ cols[j]
             for i in range(len(cols)) for j in range(i)}
    for i, x in enumerate(cols):
        for j in range(i):
            pair = x ^ cols[j]
            for h in range(j):
                if pair ^ cols[h] in pairs:
                    return False
    return True


def _qcycles(rows):
    seen = {}
    cycles = 0
    for row in rows:
        bits = []
        while row:
            bit = row & -row
            bits.append(bit.bit_length()-1)
            row ^= bit
        for i, a in enumerate(bits):
            for b in bits[:i]:
                key = (b, a)
                old = seen.get(key, 0)
                cycles += old
                seen[key] = old+1
    return cycles


def _qprobe(cols, rng, trials, memory=()):
    """Actual kernel witnesses, recombined across information sets."""
    memory = sorted(set(memory), key=int.bit_count)[:24]
    order = list(range(len(cols)))
    for trial in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = list(memory)
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                words.append(word)

        words = sorted(set(words), key=int.bit_count)
        for sweep in range(2):
            changed = False
            for i, original in enumerate(words):
                word, weight = original, original.bit_count()
                for reducer in words[:i]:
                    combined = word ^ reducer
                    cw = combined.bit_count()
                    if combined and cw < weight:
                        word, weight = combined, cw
                        changed = True
                words[i] = word
            words.sort(key=int.bit_count)
            if not changed:
                break

        short = sorted(set(words+memory), key=int.bit_count)[:32]
        discoveries = list(short)
        for i, word in enumerate(short):
            for other in short[:i]:
                combined = word ^ other
                if combined:
                    discoveries.append(combined)
        memory = sorted(set(discoveries), key=int.bit_count)[:24]
        if memory and memory[0].bit_count() <= 4:
            break
    return tuple(memory)


def _qpair_profile(rows, n):
    """Complete-matrix kernel census and permutation-invariant signature.

    Small kernel words are conservatively rejected even if stabilizers.
    With weights <=3 absent, quartet supports give three pair collisions.
    These are kernel counts, not a logical-distance certification.
    """
    cols = _qcolumns(rows, n)
    singles = set(cols)
    if 0 in singles or len(singles) != n:
        return None
    pairs = {}
    collisions = 0
    for i, x in enumerate(cols):
        for j in range(i):
            syndrome = x ^ cols[j]
            if syndrome in singles:
                return None
            old = pairs.get(syndrome, 0)
            collisions += old
            pairs[syndrome] = old+1
    histogram = {}
    for syndrome, count in pairs.items():
        key = (count, syndrome.bit_count())
        histogram[key] = histogram.get(key, 0)+1
    return (collisions//3, tuple(sorted(histogram.items())),
            tuple(sorted(row.bit_count() for row in rows)))


def _qbasis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length()-1
            old = basis.get(p)
            if old is None:
                basis[p] = word
                break
            word ^= old
    return basis


def _qoutside(word, basis):
    while word:
        p = word.bit_length()-1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _qdressed(item, q):
    """Verify logical embeddings and search lighter stabilizer dressings."""
    hx, hz, n = item["hx"], item["hz"], item["n"]
    mask = (1 << q)-1
    best = n+1
    for side, seeds, checks, stabilizers in (
            ("A", item["aw"], hx, hz),
            ("B", item["bw"], hz, hx)):
        basis = _qbasis(stabilizers)
        for seed in seeds[:6]:
            for fixed in range(4 if side == "A" else 3):
                if side == "A":
                    word = sum(
                        ((seed >> (j*q)) & mask) << ((j*4+fixed)*q)
                        for j in range(3))
                else:
                    word = seed << (fixed*4*q)
                if any((word & row).bit_count() & 1 for row in checks):
                    continue
                if not _qoutside(word, basis):
                    continue
                best = min(best, word.bit_count())
                for sweep in range(3):
                    chosen, weight = word, word.bit_count()
                    nearby = []
                    for row in stabilizers:
                        cw = (word ^ row).bit_count()
                        nearby.append((cw, row))
                        if 0 < cw < weight:
                            chosen, weight = word ^ row, cw
                    nearby.sort(key=lambda entry: entry[0])
                    near = [row for _, row in nearby[:12]]
                    for i, row in enumerate(near):
                        for other in near[:i]:
                            combined = word ^ row ^ other
                            cw = combined.bit_count()
                            if 0 < cw < weight:
                                chosen, weight = combined, cw
                    if chosen == word:
                        break
                    word = chosen
                    best = min(best, weight)
    return best


def _qdraw(rng, G, card):
    """Five A relative patterns crossed with four B term architectures."""
    pattern = _A_PATTERNS[card//4]
    scale = rng.randrange(1, 7)
    A = [[]]
    for delta in pattern:
        u = rng.randrange(7)
        v = (3*u + scale*delta) % 7
        # v != 3*u excludes the commuting C3 subgroup-sum triple.
        A[0].append([G.e, (u, 2), (v, 4)])

    # Three distinct inverse-pair representatives of elements of order six.
    B = [[[G.e, (a, 1)] for a in rng.sample(range(7), 3)]]
    quota = _B_TRIPLES[card % 4]
    for attempt in range(96):
        cell = [G.e, (rng.randrange(7), quota[1]),
                (rng.randrange(7), quota[2])]
        if len(set(cell)) != 3:
            continue
        # Avoid every relative involution; term distinctness alone is weaker.
        if any(
            G.mul(G.mul(G.inv(g), h), G.mul(G.inv(g), h)) == G.e
            for i, g in enumerate(cell) for h in cell[:i]
        ):
            continue
        B[0].append(cell)
        return A, B
    raise ValueError("B trinomial relative-order recipe exhausted")


def _qmerit(item):
    da = min((word.bit_count() for word in item["aw"]),
             default=len(item["ac"])+1)
    db = min((word.bit_count() for word in item["bw"]),
             default=len(item["bc"])+1)
    reading = min(da, db, item.get("logical", item["n"]+1))
    # Actual witnesses are required, so a no-witness sentinel earns nothing.
    if da > len(item["ac"]) or db > len(item["bc"]):
        reading = 0
    return (item["floor"], item.get("clean5", 0),
            item["k"]*reading*reading/item["n"],
            reading, da+db, -item["cycles"])


@lru_cache(maxsize=1)
def _qbank(G):
    """Enumerated A compatibility, even B augmentation, verified ranks."""
    from collections import Counter
    from itertools import combinations

    q = G.q
    assert q == 42
    mask = (1 << q)-1
    sa, sb = [[3,3,3]], [[2,2,2,2]]
    assert _grid_weights(sa, sb) == 11
    assert _no_dead_lines(sa, sb)
    stub_a = [[[G.e] for _ in range(3)]]
    stub_b = [[[G.e] for _ in range(4)]]

    def span(words):
        """Canonical binary span of verified component columns."""
        basis = _qbasis(words)
        pivots = sorted(basis)
        for p in pivots:
            for r in pivots:
                if r > p and ((basis[r] >> p) & 1):
                    basis[r] ^= basis[p]
        return tuple(basis[p] for p in pivots)

    # Compile both component actions exclusively from scaffold unit lifts.
    # XOR and concatenation below reuse these verified contributions.
    atlas = {}
    for side in ("A", "B"):
        units = []
        for g in G.elements:
            if side == "A":
                A = [[[g], [G.e], [G.e]]]
                hx, hz, _ = lifted_matrices(G, A, stub_b)
                source = hx
            else:
                B = [[[g], [G.e], [G.e], [G.e]]]
                hx, hz, _ = lifted_matrices(G, stub_a, B)
                source = hz
            units.append(tuple(row & mask for row in source[:q]))
        atlas[side] = tuple(units)

    def block(cell, side):
        rows = [0]*q
        for g in cell:
            unit = atlas[side][G.idx(g)]
            for r in range(q):
                rows[r] ^= unit[r]
        return tuple(rows), tuple(_qcolumns(rows, q))

    # Right normalization removes a common A column factor. Equal sorted
    # column incidences are discarded before compatibility assembly.
    classes = {}
    nonidentity = [g for g in G.elements if g != G.e]
    for g, h in combinations(nonidentity, 2):
        cell = (G.e, g, h)
        rows, cols = block(cell, "A")
        if 0 in cols or len(set(cols)) != q:
            continue
        space = span(cols)
        if q-len(space) < 4:
            continue
        incidence = tuple(sorted(cols))
        classes.setdefault(space, {}).setdefault(
            incidence, (cell, rows, cols))

    # A sum of column spaces corresponds to an intersection of row
    # cokernels. This also admits unequal individual block deficits.
    spaces = tuple(classes)
    common = {
        s for s in spaces if 4 <= q-len(s) <= 10
    }
    for i, first in enumerate(spaces):
        for second in spaces[:i]:
            dimension = gf2_rank(first+second)
            if q-10 <= dimension <= q-4:
                common.add(span(first+second))

    packed = []
    for target in sorted(common, key=lambda s: (len(s), s)):
        entries = {}
        for space, bucket in classes.items():
            if len(space) > len(target):
                continue
            if gf2_rank(target+space) != len(target):
                continue
            for incidence, entry in bucket.items():
                entries.setdefault(incidence, entry)
        if len(entries) >= 3:
            packed.append((q-len(target), tuple(entries.values())))
    if not packed:
        raise ValueError("no three-block singular A compatibility class")

    def fingerprint(hx, hz, n):
        """Invariant of the combined Tanner graph, without X/Z colouring.

        Unequal signatures prove nonisomorphism. Equal signatures are
        rejected conservatively; they need not prove isomorphism.
        """
        rows = list(hx)+list(hz)
        incidents = [[] for _ in range(n)]
        for r, original in enumerate(rows):
            row = original
            while row:
                bit = row & -row
                incidents[bit.bit_length()-1].append(r)
                row ^= bit
        adjacency = [Counter() for _ in rows]
        for checks in incidents:
            for i, a in enumerate(checks):
                for b in checks[:i]:
                    adjacency[a][b] += 1
                    adjacency[b][a] += 1
        profiles = []
        for r, neighbors in enumerate(adjacency):
            square = {}
            for s, multiplicity in neighbors.items():
                for t, other in adjacency[s].items():
                    square[t] = square.get(t, 0)+multiplicity*other
            triangles = sum(
                neighbors.get(t, 0)*value for t, value in square.items())
            profiles.append((
                rows[r].bit_count(),
                tuple(sorted(Counter(
                    (rows[s].bit_count(), multiplicity)
                    for s, multiplicity in neighbors.items()).items())),
                square.get(r, 0), triangles,
                tuple(sorted(Counter(square.values()).items())),
            ))
        return (
            tuple(sorted(Counter(map(len, incidents)).items())),
            tuple(sorted(Counter(profiles).items())),
        )

    # Prescribed preferences, not twenty translations of one selected code.
    desired_deficit = (10, 8, 6, 4, 4)
    desired_spread = (3, 6, 9, 6, 9)
    b_geometries = (
        (0,1,2,3), (0,1,2,4), (0,1,2,5), (0,1,3,5),
    )
    bank, used = [], set()

    for slot in range(N_SLOTS):
        policy, mode = divmod(slot % 20, 4)
        local = random.Random(42266173+100003*slot)
        preferred = [
            item for item in packed
            if item[0] >= desired_deficit[policy]
        ]
        if not preferred:
            preferred = list(packed)
        winner = None
        aseen, bseen = set(), set()

        for batch in range(2):
            choices = preferred if batch == 0 else packed
            apool = []
            for attempt in range(160+96*batch):
                _, entries = local.choice(choices)
                selected = local.sample(entries, 3)
                signature = tuple(sorted(entry[0] for entry in selected))
                if signature in aseen:
                    continue
                aseen.add(signature)
                ac = tuple(c for entry in selected for c in entry[2])
                # Cubic columns have odd parity, excluding odd kernel
                # weights. Distinct columns additionally exclude weight two.
                if len(set(ac)) != 3*q:
                    continue
                deficit = q-gf2_rank(ac)
                if not 4 <= deficit <= 10:
                    continue
                ar = [
                    sum(entry[1][r] << (j*q)
                        for j, entry in enumerate(selected))
                    for r in range(q)
                ]
                A = [[list(entry[0]) for entry in selected]]
                spread = sum(len({g[1] for g in cell}) for cell in A[0])
                cycles = _qcycles(ar)
                key = (deficit,
                       -abs(spread-desired_spread[policy]), -cycles)
                apool.append((key, A, cycles))
                apool.sort(key=lambda item: item[0], reverse=True)
                del apool[12:]
            if not apool:
                continue

            bpool = []
            for attempt in range(24):
                scale, shift = local.randrange(1, 7), local.randrange(7)
                parameters = tuple(sorted(
                    (scale*a+shift) % 7 for a in b_geometries[mode]))
                if parameters in bseen:
                    continue
                bseen.add(parameters)
                B = [[[G.e, (a,1)] for a in parameters]]
                entries = [block(cell, "B") for cell in B[0]]
                bc = tuple(c for _, cols in entries for c in cols)
                br = [
                    sum(rows[r] << (j*q)
                        for j, (rows, _) in enumerate(entries))
                    for r in range(q)
                ]
                # Four distinct complement edges form a connected
                # bipartite component. Verify its rank and short kernels.
                if gf2_rank(br) != q-1 or _qfloor(bc) < 4:
                    continue
                bpool.append((_qcycles(br), B))
                if len(bpool) >= 6:
                    break
            if not bpool:
                continue

            proposals = []
            for ak, A, acycles in apool:
                for bcycles, B in bpool:
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (k >= 266 and 475*n <= 1000*k < 525*n):
                        continue
                    key = (k, ak[1], -acycles-bcycles)
                    proposals.append((key, A, B, hx, hz, n))
            proposals.sort(key=lambda item: item[0], reverse=True)

            for key, A, B, hx, hz, n in proposals:
                # These exact complete-matrix tests exclude all kernel
                # supports through three, including mixed-sector supports.
                px = _qpair_profile(hx, n)
                if px is None:
                    continue
                pz = _qpair_profile(hz, n)
                if pz is None:
                    continue
                fp = fingerprint(hx, hz, n)
                if fp in used:
                    continue
                assert css_ok(hx, hz)
                assert max(row.bit_count() for row in hx+hz) == 11
                assert all(len(cell) == len(set(cell)) <= 5
                           for grid in (A, B)
                           for row in grid for cell in row)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                # Four is the measured-plateau hypothesis. Kernel quartet
                # counts are not asserted to count logical operators.
                score = (16*k/n, k, -max(px[0], pz[0]),
                         -px[0]-pz[0], key[1], key[2])
                winner = (
                    tuple(tuple(tuple(cell) for cell in row) for row in A),
                    tuple(tuple(tuple(cell) for cell in row) for row in B),
                    k, score, fp,
                )
                break
            if winner is not None:
                break

        if winner is None:
            bank.append(None)
        else:
            A, B, k, score, fp = winner
            used.add(fp)
            bank.append((A, B, k, score))
    return tuple(bank)


@lru_cache(maxsize=4)
def NEW_CTX(rng, G, idx):
    """Pure cached mixed-support optimization; -1 builds the shared bank."""
    if idx >= 0:
        selected = NEW_CTX(None, G, -1)[idx]
        if selected is None:
            raise ValueError("no fresh mixed-support half-rate design passed screening")
        return selected

    from collections import Counter
    from itertools import combinations

    q = G.q
    assert q == 42
    mask = (1 << q) - 1
    local = random.Random(BASE_SEED * 100003 + 42334191)
    stub_a = [[[G.e] for _ in range(3)]]
    stub_b = [[[G.e] for _ in range(4)]]

    def canonical(words):
        basis = _qbasis(words)
        pivots = sorted(basis)
        for p in pivots:
            for h in pivots:
                if h > p and ((basis[h] >> p) & 1):
                    basis[h] ^= basis[p]
        return tuple(basis[p] for p in pivots)

    # Every component action is extracted from a scaffold-generated lift.
    atlas = {}
    for side in ("A", "B"):
        units = []
        for g in G.elements:
            if side == "A":
                unit = [[[g], [G.e], [G.e]]]
                hx, hz, _ = lifted_matrices(G, unit, stub_b)
                source = hx
            else:
                unit = [[[g], [G.e], [G.e], [G.e]]]
                hx, hz, _ = lifted_matrices(G, stub_a, unit)
                source = hz
            units.append(tuple(row & mask for row in source[:q]))
        atlas[side] = tuple(units)

    def block(cell, side):
        rows = [0] * q
        for g in cell:
            unit = atlas[side][G.idx(g)]
            for r, value in enumerate(unit):
                rows[r] ^= value
        return tuple(rows), tuple(_qcolumns(rows, q))

    def joined_rows(entries):
        return [
            sum(entry["rows"][r] << (j * q)
                for j, entry in enumerate(entries))
            for r in range(q)]

    # Retain distinct column incidences, not independent cell gauges.
    # Reservoirs preserve label diversity within each exact image stratum.
    classes = {}
    others = [g for g in G.elements if g != G.e]
    for size in (2, 3, 4):
        for tail in combinations(others, size - 1):
            cell = (G.e,) + tail
            rows, cols = block(cell, "A")
            if 0 in cols or len(set(cols)) != q:
                continue
            # Uniform column parity excludes odd kernel weights for
            # cubic/even cells separately; binomials still need a test.
            if size == 2 and _qfloor(cols) < 4:
                continue
            space = canonical(cols)
            if q - len(space) < 4:
                continue
            incidence = tuple(sorted(cols))
            group = classes.setdefault(
                space, {"seen": set(), "entries": {}, "counts": {}})
            if incidence in group["seen"]:
                continue
            group["seen"].add(incidence)
            entry = {
                "cell": cell, "rows": rows, "cols": cols,
                "labels": tuple(G.idx(g) for g in cell),
                "spread": len({g[1] for g in cell})}
            bucket = group["entries"].setdefault(size, [])
            count = group["counts"].get(size, 0) + 1
            group["counts"][size] = count
            if len(bucket) < 32:
                bucket.append(entry)
            else:
                position = local.randrange(count)
                if position < 32:
                    bucket[position] = entry

    spaces = tuple(classes)
    # Originals cover binomial/quartic containing images. Joins of
    # cubic/binomial images also cover unequal-image first-two blocks.
    targets = {s for s in spaces if 4 <= q - len(s) <= 10}
    small_spaces = [
        s for s in spaces
        if any(t in classes[s]["entries"] for t in (2, 3))]
    for i, first in enumerate(small_spaces):
        for second in small_spaces[:i]:
            rank = gf2_rank(first + second)
            if q - 10 <= rank <= q - 4:
                targets.add(canonical(first + second))

    packed = []
    for target in sorted(targets, key=lambda s: (len(s), s)):
        basis = {word.bit_length() - 1: word for word in target}
        buckets = {2: [], 3: [], 4: []}
        for space, group in classes.items():
            if len(space) > len(target):
                continue
            if any(_qoutside(word, basis) for word in space):
                continue
            for size, entries in group["entries"].items():
                buckets[size].extend(entries)
        if sum(map(len, buckets.values())) >= 3:
            packed.append((q - len(target), buckets))
    if not packed:
        raise ValueError("no compatible mixed cubic/quartic image library")

    profiles = ((3, 3, 4), (3, 3, 4), (2, 3, 4),
                (2, 4, 4), (2, 3, 3))
    spread_targets = (5, 9, 7, 6, 8)
    apools = []
    for policy, profile in enumerate(profiles):
        assert _grid_weights([list(profile)], [[2] * 4]) <= W_MAX
        needed = Counter(profile)
        eligible = [
            item for item in packed
            if all(len(item[1][size]) >= count
                   for size, count in needed.items())]
        local.shuffle(eligible)
        eligible.sort(key=lambda item: -item[0])
        retained, seen = {}, set()

        for target_deficit, buckets in eligible[:32]:
            for attempt in range(128):
                selected = []
                for size, count in sorted(needed.items()):
                    selected.extend(local.sample(buckets[size], count))
                labels = tuple(sorted(entry["labels"] for entry in selected))
                if labels in seen:
                    continue
                seen.add(labels)
                cols = tuple(value for entry in selected
                             for value in entry["cols"])
                # Mixed column parities require the actual union test:
                # constituent cleanliness alone does not exclude triples.
                if _qfloor(cols) < 4:
                    continue
                deficit = q - gf2_rank(cols)
                if not 4 <= deficit <= 10:
                    continue
                rows = joined_rows(selected)
                A = [[list(entry["cell"]) for entry in selected]]
                spread = sum(entry["spread"] for entry in selected)
                cycles = _qcycles(rows)
                key = (-abs(spread - spread_targets[policy]),
                       -cycles, labels)
                bucket = retained.setdefault(deficit, [])
                bucket.append((key, A, deficit, cycles))
                bucket.sort(key=lambda item: item[0], reverse=True)
                del bucket[8:]

        pool = [item for bucket in retained.values() for item in bucket]
        pool.sort(key=lambda item: (
            item[2], item[0][0], -item[3]), reverse=True)
        apools.append(tuple(pool))

    # Four-binomial geometries are enumerated, not obtained by affine
    # transport of the parent's prescribed complement parameter sets.
    bins = {}
    for g in others:
        rows, cols = block((G.e, g), "B")
        if _qfloor(cols) < 4:
            continue
        bins.setdefault(tuple(sorted(cols)), {
            "cell": (G.e, g), "rows": rows, "cols": cols})
    bin_entries = list(bins.values())
    bpools, bseen = [], set()
    for entries in combinations(bin_entries, 4):
        cols = tuple(value for entry in entries for value in entry["cols"])
        incidence = tuple(sorted(cols))
        if incidence in bseen:
            continue
        bseen.add(incidence)
        if _qfloor(cols) < 4:
            continue
        rows = joined_rows(entries)
        if gf2_rank(rows) != q - 1:
            continue
        stats = _qpair_profile(rows, 4 * q)
        if stats is None:
            continue
        B = [[list(entry["cell"]) for entry in entries]]
        rotations = sum(entry["cell"][1][1] == 0 for entry in entries)
        labels = tuple(sorted(
            tuple(sorted(G.idx(g) for g in entry["cell"]))
            for entry in entries))
        bpools.append({
            "B": B, "quartets": stats[0], "rotations": rotations,
            "cycles": _qcycles(rows), "labels": labels})
    if not bpools:
        raise ValueError("no connected four-binomial B passed triple exclusion")

    def logical_four(checks, stabilizers, n):
        """Exact short logical test, retaining one syndrome representative.

        Equal-syndrome differences of singleton/pair supports cover every
        kernel support through four. Testing all collisions against one
        representative suffices modulo the stabilizer row space.
        """
        cols = _qcolumns(checks, n)
        basis = _qbasis(stabilizers)
        seen = {0: 0}
        found_four = False
        quartet_collisions = 0

        for i, syndrome in enumerate(cols):
            word = 1 << i
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            else:
                difference = word ^ old
                if difference and _qoutside(difference, basis):
                    return None

        for i, syndrome in enumerate(cols):
            first = 1 << i
            for j in range(i):
                value = syndrome ^ cols[j]
                word = first | (1 << j)
                old = seen.get(value)
                if old is None:
                    seen[value] = word
                    continue
                difference = word ^ old
                if not difference:
                    continue
                weight = difference.bit_count()
                if weight == 4:
                    quartet_collisions += 1
                if not _qoutside(difference, basis):
                    continue
                if weight <= 3:
                    return None
                found_four = True

        return (4 if found_four else 5, quartet_collisions)

    def fingerprint(hx, hz, n):
        """Uncoloured incidence invariant; unequal values exclude isomorphism."""
        rows = list(hx) + list(hz)
        weights = [row.bit_count() for row in rows]
        incidents = [[] for _ in range(n)]
        for r, original in enumerate(rows):
            row = original
            while row:
                bit = row & -row
                incidents[bit.bit_length() - 1].append(r)
                row ^= bit
        overlaps = [Counter() for _ in rows]
        for checks in incidents:
            for i, a in enumerate(checks):
                for b in checks[:i]:
                    overlaps[a][b] += 1
                    overlaps[b][a] += 1
        adjacency = [
            sum(1 << s for s in neighbors) for neighbors in overlaps]
        profiles = []
        for r, neighbors in enumerate(overlaps):
            square = Counter()
            for s, multiplicity in neighbors.items():
                for t, other in overlaps[s].items():
                    square[t] += multiplicity * other
            profiles.append((
                weights[r],
                tuple(sorted(Counter(
                    (weights[s], multiplicity,
                     (adjacency[r] & adjacency[s]).bit_count())
                    for s, multiplicity in neighbors.items()).items())),
                tuple(sorted(Counter(square.values()).items()))))
        return (
            n, tuple(sorted(Counter(weights).items())),
            tuple(sorted(Counter(profiles).items())),
            tuple(sorted(Counter(
                tuple(sorted(weights[r] for r in checks))
                for checks in incidents).items())))

    emitted, bank = set(), []
    for slot in range(N_SLOTS):
        policy, mode = divmod(slot % 20, 4)
        choice_rng = random.Random(42334273 + 100019 * slot)
        ap = list(apools[policy])
        # Broaden only within the new support architectures.
        if not ap:
            ap = [item for pool in apools for item in pool]
        choice_rng.shuffle(ap)
        ap.sort(key=lambda item: (
            -item[2], -item[0][0], item[3]))
        if slot >= 20 and ap:
            shift = (4 * (slot // 20)) % len(ap)
            ap = ap[shift:] + ap[:shift]

        preferred_rotations = 0 if mode < 2 else 1
        region = [
            item["quartets"] for item in bpools
            if item["rotations"] == preferred_rotations]
        if not region:
            region = [item["quartets"] for item in bpools]
        target_load = min(region) if mode % 2 == 0 else max(region)
        bp = list(bpools)
        choice_rng.shuffle(bp)
        bp.sort(key=lambda item: (
            abs(item["rotations"] - preferred_rotations),
            abs(item["quartets"] - target_load), item["cycles"]))

        proposals, exact_seen = [], set()
        for ak, A, deficit, acycles in ap[:12]:
            for b in bp[:12]:
                B = b["B"]
                labels = tuple(
                    tuple(sorted(G.idx(g) for g in cell))
                    for grid in (A, B) for row in grid for cell in row)
                if labels in exact_seen:
                    continue
                exact_seen.add(labels)
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (475 * n <= 1000 * k < 525 * n):
                    continue
                # Rank gates use complete verified matrices, not a
                # conjectured formula involving constituent deficits.
                key = (k, ak[0], -acycles - b["cycles"])
                proposals.append((key, A, B, hx, hz, n, k))
        choice_rng.shuffle(proposals)
        proposals.sort(key=lambda item: item[0], reverse=True)

        winner, tested = None, 0
        for key, A, B, hx, hz, n, k in proposals:
            fp = fingerprint(hx, hz, n)
            if fp in emitted:
                continue
            px = logical_four(hx, hz, n)
            if px is None:
                continue
            pz = logical_four(hz, hx, n)
            if pz is None:
                continue
            floor = min(px[0], pz[0])
            objective = (
                k * floor * floor / n, floor, k,
                -px[1] - pz[1], key[1], key[2])
            if winner is None or objective > winner[0]:
                winner = (objective, A, B, hx, hz, k, fp, floor)
            tested += 1
            if floor == 5 or tested >= 6:
                break

        if winner is None:
            bank.append(None)
            continue
        objective, A, B, hx, hz, k, fp, floor = winner
        assert css_ok(hx, hz)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        assert tuple(sorted(sa[0])) != (3, 3, 3)
        assert all(len(cell) == len(set(cell)) <= 5
                   for grid in (A, B) for row in grid for cell in row)
        emitted.add(fp)
        bank.append({
            "idx": slot,
            "cells": {
                side: tuple(tuple(tuple(cell) for cell in row) for row in grid)
                for side, grid in (("A", A), ("B", B))},
            "k": k, "card": (policy, mode),
            "verified_exclusion_floor": floor,
            "selection_score": objective})
    return tuple(bank)


def CELL(rng, G, side, i, j, ctx):
    """Return the full jointly screened cell without further labeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Adding a second check row and screened distinct relative labels on D8 (2,7)x(2,8) breaks the distance-four plateau; all five measured parent leaders still had d30s=4 and thousands of sector-one weight-four witnesses.
# REFUTED: Even-quotient A trinomials and three order-six B binomials plus a cubic augmenter yield five distance-six readings on C7:C6 (1,3)x(1,4); all five measured leaders instead had k=264 and d30s=4, with hundreds of sector-one X witnesses.
# REFUTED: All-cubic compatible A images and four even B binomials remove the distance-four bottleneck or supply twenty distinct designs; the measured parent had five [546,272,4] leaders, only fourteen distinct codes, repeated isomorphic slots and exhausted policies.
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

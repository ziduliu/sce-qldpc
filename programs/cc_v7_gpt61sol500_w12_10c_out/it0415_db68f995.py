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
CONCEPT = ("Dic21, order 84, A=(1,2), B=(1,3): exact constituent exclusion through ten and component-seeded logical dressing should supply five fresh measured readings >=12 and improve fitness above 37.9538 in cell 0.300; measurement can refute this.")
GROUP = ("Frozen L1 Dic21 presentation and order, kept byte-for-byte. This bed produced measured [588,176] codes with readings twelve in slots 26,36,38, while slots 20,22 fell to nine and eleven. These results motivate stronger filling acceptance, not a certified group advantage.")
SHAPES = ("Frozen (1,2)x(1,3): n=588, kb=168, guaranteed rate 2/7, maximum admissible lift order 85, and unequal constituent variable/check counts 168/84 and 252/84 at q=84. Parent twelve-weight minima occupied sector-one Acol/Bcol blocks; the lighter X witnesses in slots 20,22 also touched Arow0/Brow0. Gate actual assembled rates to [0.275,0.325).")
FILLING = ("Twenty cards cross five A rotation-pair architectures with four B architectures. Cubic cells keep HX/HZ weights 9/12. Compile constituents exclusively from verified unit lifts, exactly exclude constituent dependencies through ten, retain true kernel supports, and independently deepen survivors. Broader bounded decoding screens sector-two singletons and selected pairs. Explicitly syndrome-check component embeddings, classify them modulo stabilizers, and attempt single/pair check dressings before full-code searches. Reject found logical witnesses below twelve and repeated incidence fingerprints. Constituent exclusions do not certify quantum distance; found logical readings remain upper bounds.")
FAMILY_NAME = "r030_dic21_12x13_cubic_exact10_seeded_dressing_v4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 84

def custom_ops(q):
    # Dic21: x^42=1, y^2=x^21, y*x*y^-1=x^-1.
    if q != 84:
        raise ValueError("Dic21 has order 84")
    els = [(a, b) for b in range(2) for a in range(42)]

    def red(g):
        a, b = g
        return ((a + 21 * (b // 2)) % 42, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 21*b*d) % 42,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 42, 0) if b == 0 else ((a + 21) % 42, 1)

    def idx(g):
        a, b = red(g)
        return 42*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


# Five different term architectures, crossed with four pair-order recipes.
# For single-row sides: wx=sum(A)+max(B), wz=sum(B)+max(A).
# Their respective (wx,wz) maxima are (9,12), (10,12), (10,12),
# (11,12), and (8,12). No profile contains an empty or singleton cell.
_SUPPORT_PLANS = (
    ((3, 3), (3, 3, 3)),
    ((3, 3), (2, 3, 4)),
    ((3, 4), (2, 3, 3)),
    ((4, 4), (2, 3, 3)),
    ((2, 3), (3, 3, 3)),
)


def _designed_grid(rng, G, counts, mode, phase):
    """Constrain internal relative orders, not just translate one labeling."""
    from math import gcd
    divisor = (1, 2, 3, 6)[mode]
    pools = [[g for g in G.elements if g[1] == b] for b in range(2)]
    row = []
    for j, weight in enumerate(counts):
        b = (phase + j) & 1
        # A binomial with an order-four relative element would itself have
        # a weight-four kernel word. Use order 42 for every binomial.
        wanted = 1 if weight == 2 else divisor
        steps = [d for d in range(1, 42) if gcd(d, 42) == wanted]
        first = rng.choice(pools[b])
        second = ((first[0] + rng.choice(steps)) % 42, b)
        cell = [first, second]
        if weight == 3:
            remaining = [1 - b]
        elif weight == 4:
            remaining = [1 - b, 1 - b if mode & 1 else b]
        else:
            remaining = []
        for coset in remaining:
            legal = [
                g for g in pools[coset]
                if g not in cell
                and all(G.mul(G.inv(h), g) != (21, 0) for h in cell)
            ]
            cell.append(rng.choice(legal))
        assert len(cell) == weight and len(set(cell)) == weight
        row.append(cell)
    return [row]


def _columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _short_floor(columns):
    """Exact component-kernel minimum through four; five means unexcluded."""
    singles = set(columns)
    if 0 in singles:
        return 1
    if len(singles) != len(columns):
        return 2
    pairs = set()
    four = False
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                four = True
            pairs.add(syndrome)
    return 4 if four else 5


def _component_cycles(rows):
    counts = {}
    cycles = 0
    for row in rows:
        positions = []
        while row:
            bit = row & -row
            positions.append(bit.bit_length() - 1)
            row ^= bit
        for i, a in enumerate(positions):
            for b in positions[:i]:
                key = (b, a)
                old = counts.get(key, 0)
                cycles += old
                counts[key] = old + 1
    return cycles


def _kernel_reading(columns, rng, trials):
    """Find actual classical kernel words; high readings are not proofs."""
    best = len(columns) + 1
    order = list(range(len(columns)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                words.append(word)
                best = min(best, word.bit_count())
        for i, word in enumerate(words):
            for j in range(i):
                best = min(best, (word ^ words[j]).bit_count())

        # A few short basis triples supplement pairwise information sets.
        small = sorted(words, key=int.bit_count)[:10]
        for i, word in enumerate(small):
            for j in range(i):
                pair = word ^ small[j]
                for h in range(j):
                    best = min(best, (pair ^ small[h]).bit_count())
        if best <= 4:
            break
    return best


def _component_data(hx, hz, q):
    """Extract both actions from the scaffold's verified LP construction."""
    mask = (1 << q) - 1
    # HX beta=0: the A blocks are sector-one blocks 0 and 3.
    ar = [
        (row & mask) | (((row >> (3*q)) & mask) << q)
        for row in hx[:q]
    ]
    # HZ A-column=0: B occupies the first three sector-one blocks.
    bm = (1 << (3*q)) - 1
    br = [row & bm for row in hz[:q]]
    return ar, br, _columns(ar, 2*q), _columns(br, 3*q)


from functools import lru_cache


def _dic_packed(rng, G, divisors):
    """Fresh cubic labels with prescribed rotation-pair orders."""
    from math import gcd
    steps = {
        d: [v for v in range(1, 42) if gcd(v, 42) == d]
        for d in (1, 2, 3)
    }
    for restart in range(16):
        used = set()
        row = [None] * len(divisors)
        order = list(range(len(divisors)))
        rng.shuffle(order)
        for j in order:
            for attempt in range(160):
                a, b = rng.randrange(42), rng.randrange(2)
                delta = rng.choice(steps[divisors[j]])
                cell = [(a, b), ((a + delta) % 42, b),
                        (rng.randrange(42), 1 - b)]
                differences = {
                    G.mul(g, G.inv(h))
                    for g in cell for h in cell if g != h
                }
                if len(differences) != 6 or differences & used:
                    continue
                if (21, 0) in differences:
                    continue
                row[j] = cell
                used.update(differences)
                break
            if row[j] is None:
                break
        if all(cell is not None for cell in row):
            return [row]
    return None


def _dic_basis(rows):
    basis = {}
    for word in rows:
        while word:
            pivot = word.bit_length() - 1
            previous = basis.get(pivot)
            if previous is None:
                basis[pivot] = word
                break
            word ^= previous
    return basis


def _dic_outside(word, basis):
    while word:
        pivot = word.bit_length() - 1
        previous = basis.get(pivot)
        if previous is None:
            return True
        word ^= previous
    return False


def _dic_probe(cols, rng, trials, best=None, stabilizers=None):
    """Actual kernel witnesses; full-code witnesses exclude stabilizers."""
    best = len(cols) + 1 if best is None else best
    basis = None if stabilizers is None else _dic_basis(stabilizers)
    generators = [] if stabilizers is None else list(stabilizers)
    order = list(range(len(cols)))
    memory = []

    def consider(word):
        nonlocal best
        if not word or word.bit_count() >= best:
            return
        if basis is None or _dic_outside(word, basis):
            best = word.bit_count()

    for trial in range(trials):
        rng.shuffle(order)
        pivots, dependencies = {}, []
        for j in order:
            syndrome, word = cols[j], 1 << j
            while syndrome:
                pivot = syndrome.bit_length() - 1
                previous = pivots.get(pivot)
                if previous is None:
                    pivots[pivot] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                dependencies.append(word)
                consider(word)

        dependencies.sort(key=int.bit_count)
        words = sorted(set(dependencies[:40] + memory),
                       key=int.bit_count)[:48]
        # Kernel additions are permitted even when a direction is a
        # stabilizer. Only a non-stabilizer word changes a logical reading.
        for sweep in range(2):
            changed = False
            for i, word in enumerate(words):
                weight = word.bit_count()
                for j, direction in enumerate(words):
                    if i == j:
                        continue
                    combined = word ^ direction
                    if combined and combined.bit_count() < weight:
                        word, weight = combined, combined.bit_count()
                        changed = True
                words[i] = word
                consider(word)
            if not changed:
                break

        discoveries = set(words)
        for i, word in enumerate(words):
            for direction in words[:i]:
                combined = word ^ direction
                consider(combined)
                if combined and combined.bit_count() <= best + 4:
                    discoveries.add(combined)

        # Use the entire information-set basis as descent directions.
        for seed in sorted(discoveries, key=int.bit_count)[:8]:
            word = seed
            for sweep in range(2):
                old_weight = word.bit_count()
                for direction in dependencies:
                    combined = word ^ direction
                    if combined and combined.bit_count() < word.bit_count():
                        word = combined
                if word.bit_count() == old_weight:
                    break
            consider(word)
            discoveries.add(word)

            if generators:
                for sweep in range(2):
                    old_weight = word.bit_count()
                    for direction in generators:
                        combined = word ^ direction
                        if combined and combined.bit_count() < word.bit_count():
                            word = combined
                    if word.bit_count() == old_weight:
                        break
                consider(word)
                discoveries.add(word)

        memory = sorted(discoveries, key=int.bit_count)[:32]
        if best <= 4:
            break
    return best


def _dic_decoder(cols, height):
    """Bounded cubic decoding with canonical first-choice branching.

    With cap=1 this is an exhaustive existence test, stopping only on a
    witness. Finishing tables are unique after component-four exclusion.
    Earlier alternatives are banned in later branches, avoiding repeated
    permutations of the same completion. No node/time cutoff is used.
    """
    bits = [1 << i for i in range(len(cols))]
    incidence = [0] * height
    tails = {0: 0}
    for i, syndrome in enumerate(cols):
        assert syndrome.bit_count() == 3 and syndrome not in tails
        tails[syndrome] = bits[i]
        value = syndrome
        while value:
            bit = value & -value
            incidence[bit.bit_length() - 1] |= bits[i]
            value ^= bit
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            assert value not in tails
            tails[value] = bits[i] | bits[j]

    def decode(syndrome, limit, forbidden=0, cap=0):
        found = set()

        def visit(value, remaining, word, banned):
            if cap and len(found) >= cap:
                return
            if not value:
                found.add(word)
                return
            # Every cubic column has odd parity.
            if (value.bit_count() ^ remaining) & 1:
                remaining -= 1
            if remaining <= 0 or value.bit_count() > 3 * remaining:
                return
            if remaining <= 2:
                tail = tails.get(value)
                if (tail is not None and tail.bit_count() <= remaining
                        and not (tail & banned)):
                    found.add(word | tail)
                return

            pending, choices, least = value, 0, len(cols) + 1
            while pending:
                bit = pending & -pending
                available = incidence[bit.bit_length() - 1] & ~banned
                count = available.bit_count()
                if not count:
                    return
                if count < least:
                    choices, least = available, count
                    if count == 1:
                        break
                pending ^= bit

            alternatives = []
            while choices:
                bit = choices & -choices
                choices ^= bit
                alternatives.append(bit.bit_length() - 1)
            alternatives.sort(
                key=lambda j: (-(cols[j] & value).bit_count(), j))
            excluded = banned
            for j in alternatives:
                excluded |= bits[j]
                visit(value ^ cols[j], remaining - 1,
                      word | bits[j], excluded)
                if cap and len(found) >= cap:
                    return

        visit(syndrome, limit, 0, forbidden)
        return sorted(found, key=lambda word: (word.bit_count(), word))

    return decode


@lru_cache(maxsize=1)
def _dic_unit_atlas(G):
    """Component actions extracted only from scaffold-generated unit lifts."""
    q, mask = G.q, (1 << G.q) - 1
    zero_a, zero_b = [[[], []]], [[[], [], []]]
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            A = [[[g], []]] if side == "A" else zero_a
            B = [[[g], [], []]] if side == "B" else zero_b
            hx, hz, _ = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            blocks.append(tuple(_columns(
                [row & mask for row in source[:q]], q)))
        atlas[side] = tuple(blocks)
    return atlas


def _dic_compile_component(G, side, cells):
    """F2-linear superposition of verified unit actions."""
    q = G.q
    atlas = _dic_unit_atlas(G)[side]
    cols = [0] * (len(cells[0]) * q)
    for j, cell in enumerate(cells[0]):
        for g in cell:
            for t, syndrome in enumerate(atlas[G.idx(g)]):
                cols[j*q+t] ^= syndrome
    rows = [0] * q
    for c, syndrome in enumerate(cols):
        while syndrome:
            bit = syndrome & -syndrome
            rows[bit.bit_length()-1] |= 1 << c
            syndrome ^= bit
    return rows, cols


def _dic_retained_words(cols, rng, trials, seeds=()):
    """Retain actual nonzero kernel supports, never a no-witness estimate."""
    memory = sorted({w for w in seeds if w}, key=int.bit_count)[:32]
    order = list(range(len(cols)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots, dependencies = {}, []
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
            if not syndrome and word:
                dependencies.append(word)
        words = sorted(set(memory + dependencies), key=int.bit_count)
        discoveries = set(memory)
        for sweep in range(2):
            reducers = words[:32]
            shortened = []
            for word in words:
                weight = word.bit_count()
                for other in reducers:
                    candidate = word ^ other
                    cw = candidate.bit_count()
                    if 0 < cw < weight:
                        word, weight = candidate, cw
                shortened.append(word)
            words = sorted(set(shortened), key=int.bit_count)
            discoveries.update(words[:40])
        short = sorted(discoveries, key=int.bit_count)[:28]
        for i, word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                if candidate:
                    discoveries.add(candidate)
        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:32]
        if memory and memory[0].bit_count() <= 10:
            break
    return tuple(memory)


def _dic_component_pool(rng, G, side, divisors, quota):
    """Admit only constituents with no dependency of weight <=10."""
    raw, labels_seen = [], set()
    keep = 12 if side == "A" else 24

    for attempt in range(quota):
        cells = _dic_packed(rng, G, divisors)
        if cells is None:
            continue
        labels = tuple(tuple(sorted(cell)) for cell in cells[0])
        if labels in labels_seen:
            continue
        labels_seen.add(labels)
        rows, cols = _dic_compile_component(G, side, cells)
        if _component_cycles(rows) or _short_floor(cols) < 5:
            continue
        reading = _kernel_reading(cols, rng, 3)
        if reading <= 10:
            continue
        item = dict(cells=cells, cols=cols, d=reading,
                    rank=gf2_rank(rows), attempt=attempt)
        raw.append(item)
        raw.sort(key=lambda x: (
            min(x["d"], 16), -x["rank"], -x["attempt"]), reverse=True)
        del raw[keep:]

    pool = []
    for item in raw:
        cols = item["cols"]
        words = _dic_retained_words(cols, rng, 6)
        if not words or words[0].bit_count() <= 10:
            continue
        decode = _dic_decoder(cols, G.q)

        # The regular action is transitive within each constituent block.
        # Every dependency through ten can therefore be moved to contain
        # one of these anchors. No full-code anchor reduction is claimed.
        clean = True
        for bound in (6, 8, 10):
            for anchor in range(0, len(cols), G.q):
                if decode(cols[anchor], bound-1, 1 << anchor, cap=1):
                    clean = False
                    break
            if not clean:
                break
        if not clean:
            continue

        words = _dic_retained_words(
            cols, rng, 16 if side == "A" else 28, words)
        assert words and words[0].bit_count() >= 12
        item.update(decode=decode, words=words,
                    d=min(item["d"], words[0].bit_count()), clean10=True)
        # A high reading stays a retained-witness upper estimate.
        item["d"] = _dic_probe(
            cols, rng, 24 if side == "A" else 40, item["d"])
        assert item["d"] >= 12
        pool.append(item)

    pool.sort(key=lambda x: (
        min(x["d"], 16), -x["rank"],
        -sum(w.bit_count() == x["d"] for w in x["words"]),
        -x["attempt"]), reverse=True)
    return pool[:3 if side == "A" else 4]


def _dic_singleton_readings(item, aa, bb, G):
    """Find logicals <=11 with one or selected two sector-2 coordinates.

    All singletons are considered. Pair supports use four intrinsic
    rotation gaps, not an assumed transitive action on the full code.
    Minimum completion weights are exhaustively queried; at most eight
    completions per target are combined. This bounded multiplicity and
    restricted pair family do NOT certify full logical exclusion.
    """
    from itertools import product
    q, n = G.q, item["n"]
    mask, n1 = (1 << q)-1, 6*q
    caches = {"A": {}, "B": {}}
    readings = [n+1, n+1]

    def options(side, component, targets, budget):
        decode, cache = component["decode"], caches[side]
        lower = []
        for target in targets:
            if not target:
                lower.append(0)
                continue
            weight = max(1, (target.bit_count()+2)//3)
            if (weight ^ target.bit_count()) & 1:
                weight += 1
            lower.append(weight)
        if sum(lower) > budget:
            return None

        minima = []
        for j, target in enumerate(targets):
            if not target:
                minima.append(0)
                continue
            ceiling = budget-sum(lower)+lower[j]
            minimum = None
            for bound in range(lower[j], ceiling+1, 2):
                key = (target, bound, 1)
                if key not in cache:
                    cache[key] = tuple(decode(target, bound, cap=1))
                if cache[key]:
                    minimum = cache[key][0].bit_count()
                    break
            if minimum is None:
                return None
            minima.append(minimum)
        if sum(minima) > budget:
            return None

        result = []
        for j, target in enumerate(targets):
            if not target:
                # clean10 excludes any additional kernel completion here
                # within the budgets used by this screen.
                result.append((0,))
                continue
            bound = budget-sum(minima)+minima[j]
            key = (target, bound, 8)
            if key not in cache:
                cache[key] = tuple(decode(target, bound, cap=8))
            if not cache[key]:
                return None
            result.append(cache[key])
        return result

    def inspect(positions):
        sector_word = sum(1 << (n1+t) for t in positions)
        budget = 11-len(positions)
        for kind, side, component, cols, span, count in (
                (0, "B", bb, item["xc"], item["xs"], 2),
                (1, "A", aa, item["zc"], item["zs"], 3)):
            syndrome = 0
            for t in positions:
                syndrome ^= cols[n1+t]
            targets = [(syndrome >> (j*q)) & mask for j in range(count)]
            choices = options(side, component, targets, budget)
            if choices is None:
                continue
            for pieces in product(*choices):
                if sum(w.bit_count() for w in pieces) > budget:
                    continue
                word = sector_word
                for j, compact in enumerate(pieces):
                    if side == "B":
                        word |= compact << (j*3*q)
                    else:
                        word |= (compact & mask) << (j*q)
                        word |= ((compact >> q) & mask) << ((3+j)*q)
                if not _dic_outside(word, span):
                    continue
                residual, remaining = 0, word
                while remaining:
                    bit = remaining & -remaining
                    residual ^= cols[bit.bit_length()-1]
                    remaining ^= bit
                assert residual == 0
                readings[kind] = min(readings[kind], word.bit_count())
                return True
        return False

    for t in range(q):
        if inspect((t,)):
            return tuple(readings)

    pairs = set()
    for gap in (6, 7, 14, 21):
        for t, g in enumerate(G.elements):
            u = G.idx(G.mul(g, (gap, 0)))
            if t != u:
                pairs.add(tuple(sorted((t, u))))
    for positions in sorted(pairs):
        if inspect(positions):
            return tuple(readings)
    return tuple(readings)


def _dic_component_dressings(item, aa, bb, q):
    """Actual embedded logical witnesses and sparse stabilizer dressings.

    Both syndromes and logical classes are checked against the assembled
    matrices. Single/pair check additions may reach mixed-sector words.
    The returned sentinel means no logical was found, not a certificate.
    """
    mask = (1 << q)-1
    readings = [item["n"]+1, item["n"]+1]
    for kind, component, checks, stabilizers, span, fixed_count in (
            (0, bb, item["hz"], item["hx"], item["xs"], 2),
            (1, aa, item["hx"], item["hz"], item["zs"], 3)):
        for seed in component["words"][:12]:
            for fixed in range(fixed_count):
                if kind == 0:
                    word = seed << (fixed*3*q)
                else:
                    word = sum(
                        ((seed >> (j*q)) & mask)
                        << ((3*j+fixed)*q) for j in range(2))
                assert not any((word & row).bit_count() & 1 for row in checks)
                if not _dic_outside(word, span):
                    continue
                readings[kind] = min(readings[kind], word.bit_count())
                for sweep in range(3):
                    chosen, weight = word, word.bit_count()
                    nearby = sorted(
                        (((word ^ row).bit_count(), row)
                         for row in stabilizers),
                        key=lambda entry: entry[0])
                    for cw, row in nearby:
                        if cw < weight:
                            chosen, weight = word ^ row, cw
                    moves = [row for _, row in nearby[:16]]
                    for i, row in enumerate(moves):
                        for other in moves[:i]:
                            candidate = word ^ row ^ other
                            cw = candidate.bit_count()
                            if cw < weight:
                                chosen, weight = candidate, cw
                    if chosen == word:
                        break
                    word = chosen
                    readings[kind] = min(readings[kind], weight)
                if readings[kind] < 12:
                    return tuple(readings)
    return tuple(readings)


def _dic_check_signature(rows):
    """Permutation-invariant check-intersection fingerprint.

    Equality is conservatively rejected; it is not asserted to prove
    isomorphism. Unequal signatures exclude a check/qubit relabeling.
    """
    size = len(rows)
    adjacency = [0] * size
    overlap = [[0] * size for _ in range(size)]
    for i, row in enumerate(rows):
        for j in range(i):
            value = (row & rows[j]).bit_count()
            overlap[i][j] = overlap[j][i] = value
            if value:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    signatures = []
    for i, row in enumerate(rows):
        histogram = {}
        for j in range(size):
            if i == j:
                continue
            key = (overlap[i][j],
                   (adjacency[i] & adjacency[j]).bit_count())
            histogram[key] = histogram.get(key, 0) + 1
        signatures.append(
            (row.bit_count(), tuple(sorted(histogram.items()))))
    return tuple(sorted(signatures))


def _dic_fingerprint(item):
    return (
        item["n"], item["k"],
        tuple(sorted((_dic_check_signature(item["hx"]),
                      _dic_check_signature(item["hz"])))),
    )


def _dic_merit(item):
    # Constituent witnesses guide selection but are not automatically
    # claimed to be nontrivial quantum logical representatives.
    d = min(item["da"], item["db"], item["dx"], item["dz"])
    return (item["k"] * d * d / item["n"], d,
            item["da"] + item["db"])


@lru_cache(maxsize=1)
def _dic_design_bank(G):
    """Five A architectures crossed with four B architectures.

    Later lanes use new B pair-order mixtures and fresh optimization.
    No labeling is transported between cards; incidence rejection
    still decides whether their assembled outputs are distinct.
    """
    a_cards = ((1, 1), (1, 2), (2, 2), (3, 3), (1, 3))
    b_cards = (
        ((1, 1, 3), (1, 2, 2), (1, 3, 3), (3, 3, 3)),
        ((1, 2, 3), (2, 2, 3), (2, 3, 3), (1, 1, 2)),
        ((1, 1, 1), (1, 2, 2), (1, 2, 3), (2, 2, 2)),
    )
    sa, sb = [[3, 3]], [[3, 3, 3]]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    bank, fingerprints = [], set()

    for idx in range(N_SLOTS):
        # A changed stream accompanies the changed construction; no
        # archived labeling is used as an initial code or a template.
        rng = random.Random(
            BASE_SEED * 100003 + 840301019 + 104729 * idx)
        card, batch = idx % 20, idx // 20
        architecture, model = divmod(card, 4)
        ad = a_cards[architecture]
        bd = b_cards[batch % len(b_cards)][model]
        winner = None

        for retry in range(2):
            ap = _dic_component_pool(
                rng, G, "A", ad, 128 + 64 * retry)
            bp = _dic_component_pool(
                rng, G, "B", bd, 320 + 160 * retry)
            candidates = []
            for aa in ap:
                for bb in bp:
                    A, B = aa["cells"], bb["cells"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n - gf2_rank(hx) - gf2_rank(hz)
                    if not (275 * n <= 1000 * k < 325 * n):
                        continue
                    d = min(aa["d"], bb["d"])
                    candidates.append((
                        (k * d * d / n, d), aa, bb, hx, hz, n, k))
            candidates.sort(key=lambda c: c[0], reverse=True)

            finalists = []
            for _, aa, bb, hx, hz, n, k in candidates:
                item = {
                    "A": aa["cells"], "B": bb["cells"],
                    "hx": hx, "hz": hz, "n": n, "k": k,
                    "da": aa["d"], "db": bb["d"],
                }
                fingerprint = _dic_fingerprint(item)
                if fingerprint in fingerprints:
                    continue
                item["fingerprint"] = fingerprint
                item["xc"] = _columns(hz, n)
                item["zc"] = _columns(hx, n)
                item["xs"] = _dic_basis(hx)
                item["zs"] = _dic_basis(hz)
                # Confirm the compiled actions against this complete lift.
                _, _, ac, bc = _component_data(hx, hz, G.q)
                assert ac == aa["cols"] and bc == bb["cols"]

                item["dx"], item["dz"] = _dic_singleton_readings(
                    item, aa, bb, G)
                if min(item["dx"], item["dz"]) < 12:
                    continue
                dressed_x, dressed_z = _dic_component_dressings(
                    item, aa, bb, G.q)
                item["dx"] = min(item["dx"], dressed_x)
                item["dz"] = min(item["dz"], dressed_z)
                if min(item["dx"], item["dz"]) < 12:
                    continue
                # Require actual non-stabilizer seeds in both types.
                if max(item["dx"], item["dz"]) > n:
                    continue
                item["dx"] = _dic_probe(
                    item["xc"], rng, 16, item["dx"], hx)
                item["dz"] = _dic_probe(
                    item["zc"], rng, 16, item["dz"], hz)
                if min(item["dx"], item["dz"]) < 12:
                    continue
                finalists.append(item)
                if len(finalists) >= 4:
                    break

            # Give every finalist the same additional full-code budget;
            # retain all earlier lighter witnesses.
            for item in finalists:
                item["dx"] = _dic_probe(
                    item["xc"], rng, 32, item["dx"], item["hx"])
                item["dz"] = _dic_probe(
                    item["zc"], rng, 32, item["dz"], item["hz"])
                if min(item["dx"], item["dz"]) < 12:
                    continue
                if winner is None or _dic_merit(item) > _dic_merit(winner):
                    winner = item
            if winner is not None and _dic_merit(winner)[1] >= 12:
                break

        if winner is None:
            bank.append(None)
            continue
        assert css_ok(winner["hx"], winner["hz"])
        assert max(row.bit_count()
                   for row in winner["hx"] + winner["hz"]) <= W_MAX
        assert 275 * winner["n"] <= 1000 * winner["k"] < 325 * winner["n"]
        assert all(len(cell) == 3 and len(set(cell)) == 3
                   for M in (winner["A"], winner["B"])
                   for row in M for cell in row)
        fingerprints.add(winner["fingerprint"])
        bank.append((
            tuple(tuple(tuple(cell) for cell in row) for row in winner["A"]),
            tuple(tuple(tuple(cell) for cell in row) for row in winner["B"]),
            winner["k"], _dic_merit(winner),
        ))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return the independently optimized slot, without relabeling it."""
    selected = _dic_design_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh rank-legal cubic design for this card")
    A, B, k, score = selected
    return {"idx": idx, "cells": {"A": A, "B": B},
            "k": k, "screen": score}


def CELL(rng, G, side, i, j, ctx):
    """Return the full jointly selected cell without additional draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform D58 labels at (1,3)x(1,3) did not produce a top-five measured reading above 4; all five d30s readings were 4.
# REFUTED: Distinct monomial relative labels and short-syndrome screening on D8 at (2,7)x(2,8) would lift the distance-four plateau; all five measured d30s readings remained 4, with both types supported entirely in sector 1.
# REFUTED: High-order binomial support profiles preserve the cubic Dic21 distance advantage; every distinct nonuniform-count parent slot read d2s=5, while all five measured parent codes used five trinomials.
# REFUTED: Component-kernel selection alone removes mixed-sector failures on Dic21; measured parent slots 0 and 3 retained weight-nine X logicals touching Arow0/Brow0.
# REFUTED: Constituent-through-eight tests, singleton-sector-2 decoding through five/three and the existing full probes ensure five readings >=12; current measured slots 20,22 instead retained mixed-sector X readings nine and eleven.
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

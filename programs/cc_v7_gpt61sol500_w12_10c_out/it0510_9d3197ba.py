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
CONCEPT = ("Frozen Dic2=Q8, order 8, A=(2,8), B=(2,8): duplicate-free degree-two components with augmentation-even degree-four replacement columns should supply five fresh k>=308 codes excluding logical weights below four and raise fitness above 8.6118; measured ranks and distances can refute this.")
GROUP = ("Frozen byte-for-byte under L1. This exact bed measured one [544,288,4] and four [544,294,4] leaders, while several higher-k slots had d2s=2. Q8 has three order-four axes and one central involution, permitting intrinsically different hypercolumn supports. These facts motivate rank-preserving filling changes, not a group-distance claim.")
SHAPES = ("Frozen equal (2,8) components: n=544, kb=288, guaranteed rate 9/17, and 64 variables with 16 checks on each side. The maximum admissible lift order is eight. Parent weight-four minima were entirely in sector-one Acol/Bcol blocks, broadly involving Acol3..6 and Bcol2..7; complete logical screens nevertheless include both sectors.")
FILLING = ("Twenty cards cross five even-cardinality hypercolumn geometries with four Q8 direction policies. Begin with duplicate-free monomial or character-split degree-two components, replace selected columns by balanced (2,2) or unbalanced (3,1)/(1,3) supports, and permit central differences. Even column cardinality preserves a constant component row dependency, but verified complete ranks decide k. Exclude component kernels and full logicals through three, classify logical-four existence modulo stabilizers, blacklist reconstructed parent matrices, and retain unequal uncoloured Tanner invariants strengthened by exact modular Gram moments. Sentinel five means exclusion through four, never a sampled distance reading.")
FAMILY_NAME = "r055_dic2_28x28_evenhypercolumns_rank308_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    # Same dicyclic family: Dic2=Q8.
    # x^4=1, y^2=x^2, y*x*y^-1=x^-1.
    if not valid_q(q):
        raise ValueError("Dic2 has order 8")
    m, half = 4, 2
    els = [(a, b) for b in range(2) for a in range(m)]

    def red(g):
        a, b = g
        return ((a + half * (b // 2)) % m, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + half*b*d) % m,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % m, 0) if b == 0 else ((a + half) % m, 1)

    def idx(g):
        a, b = red(g)
        return m*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 8)               # (ra, ca)
B_SHAPE = (2, 8)               # (rb, cb)


def _support_designs():
    """Enumerated geometries, separated by full Tanner degree multisets."""
    from itertools import combinations_with_replacement
    types = ((1, 1), (2, 0), (0, 2), (2, 1),
             (1, 2), (2, 2), (1, 0), (0, 1))
    representatives = {}
    for columns in combinations_with_replacement(types, 8):
        columns = tuple(sorted(columns))
        exchanged = tuple(sorted((b, a) for a, b in columns))
        if columns != min(columns, exchanged):
            continue
        rows = tuple(sum(c[i] for c in columns) for i in range(2))
        cols = tuple(a+b for a, b in columns)
        if not (6 <= min(rows) and max(rows) <= 10):
            continue
        if not (14 <= sum(rows) <= 20):
            continue
        if cols.count(1) > 2 or cols.count(4) > 2:
            continue
        grid = tuple(tuple(c[i] for c in columns) for i in range(2))
        key = (tuple(sorted(rows)), tuple(sorted(cols)))
        # Prefer more occupied cells within a given degree architecture.
        quality = (sum(bool(v) for row in grid for v in row), grid)
        old = representatives.get(key)
        if old is None or quality > old[0]:
            representatives[key] = (quality, grid, rows, cols)

    profiles = [(grid, rows, cols)
                for _, grid, rows, cols in representatives.values()]
    profiles.sort()
    pairs = []
    for a, (ta, ar, ac) in enumerate(profiles):
        for tb, br, bc in profiles[a:]:
            if max(ar)+max(bc) > 12 or max(br)+max(ac) > 12:
                continue
            checks = tuple(sorted(
                [x+y for x in ar for y in bc] +
                [x+y for x in br for y in ac]))
            qubits = tuple(sorted(
                [x+y for x in ac for y in bc] +
                [x+y for x in ar for y in br]))
            priority = (
                min(ac+bc), min(ar+br),
                sum(ar)+sum(br), -max(ac+bc),
                -abs(ar[0]-ar[1])-abs(br[0]-br[1]))
            pairs.append((priority, (checks, qubits), ta, tb))
    pairs.sort(reverse=True)
    seen, designs = set(), []
    for _, fingerprint, ta, tb in pairs:
        if fingerprint not in seen:
            seen.add(fingerprint)
            designs.append((ta, tb))
    if len(designs) < max(20, N_SLOTS):
        raise ValueError("insufficient degree-distinguished slot designs")
    return tuple(designs)


def _columns(rows, width):
    """Transpose only matrices obtained from the verified construction."""
    cols = [0]*width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length()-1] |= 1 << r
            row ^= bit
    return cols


def _row_basis(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length()-1
            old = pivots.get(p)
            if old is None:
                pivots[p] = word
                break
            word ^= old
    return pivots


def _outside(word, pivots):
    while word:
        p = word.bit_length()-1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _short_logical(cols, stabilizer_basis, through):
    """Exhaustive logical existence through three or four.

    Returns a genuine logical witness weight, or through+1 as an
    exclusion sentinel. Equal-syndrome supports are classified modulo
    the exact stabilizer span. This includes both LP sectors.
    """
    seen = {0: 0}
    bits = [1 << i for i in range(len(cols))]
    for i, syndrome in enumerate(cols):
        word = bits[i]
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = word
        else:
            witness = word ^ old
            if witness and _outside(witness, stabilizer_basis):
                return witness.bit_count()

    # With through=3, singles versus pairs cover every remaining
    # support through three. Pair representatives are added only for
    # the through-four test.
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            word = bits[i] | bits[j]
            old = seen.get(value)
            if old is None:
                if through == 4:
                    seen[value] = word
                continue
            witness = word ^ old
            weight = witness.bit_count()
            if (0 < weight <= through
                    and _outside(witness, stabilizer_basis)):
                return weight
    return through+1


_SUPPORT_DESIGNS = _support_designs()


def _four_cycles(rows):
    counts = {}
    cycles = 0
    for row in rows:
        bits = []
        while row:
            bit = row & -row
            bits.append(bit.bit_length()-1)
            row ^= bit
        for a, x in enumerate(bits):
            for y in bits[a+1:]:
                key = (x, y)
                old = counts.get(key, 0)
                cycles += old
                counts[key] = old + 1
    return cycles


def _kernel_probe(rows, width, rng):
    """Upper readings on classical kernel distance, used only as heuristics."""
    best = width + 1
    for _ in range(4):
        perm = list(range(width))
        rng.shuffle(perm)
        pivots = {}
        for original in rows:
            row = 0
            while original:
                bit = original & -original
                row ^= 1 << perm[bit.bit_length()-1]
                original ^= bit
            while row:
                p = row.bit_length()-1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        ordered = sorted(pivots.items())
        basis = []
        for f in range(width):
            if f in pivots:
                continue
            v = 1 << f
            for p, row in ordered:
                if (row & v).bit_count() & 1:
                    v ^= 1 << p
            best = min(best, v.bit_count())
            basis.append(v)
        for a, v in enumerate(basis):
            for u in basis[a+1:]:
                best = min(best, (v ^ u).bit_count())
    return best


def _component_checks(hx, hz, q):
    """Extract both component maps from the verified LP column order."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q)-1
    ar = []
    for i in range(ra):
        for r in range(q):
            source = hx[(i*cb)*q+r]
            ar.append(sum(
                ((source >> (j*cb*q)) & mask) << (j*q)
                for j in range(ca)))
    bm = (1 << (cb*q))-1
    br = [hz[a*q+r] & bm
          for a in range(rb) for r in range(q)]
    return ar, br


def _label_grid(rng, G, counts, model=0):
    """Fresh labels; cells never contain a central-involution pair."""
    involution = (G.q//4, 0)
    grid = []
    for i, sizes in enumerate(counts):
        row = []
        for j, count in enumerate(sizes):
            cell = []
            while len(cell) < count:
                available = [
                    g for g in G.elements if g not in cell
                    and all(G.mul(G.inv(h), g) != involution
                            for h in cell)]
                if cell and model:
                    # The three nonzero Q8/centre classes distinguish
                    # relative directions, not a common relabeling.
                    wanted = 1+(i+2*j+model) % 3
                    preferred = []
                    for g in available:
                        d = G.mul(G.inv(cell[0]), g)
                        axis = (d[0] % 2)+2*d[1]
                        if axis == wanted:
                            preferred.append(g)
                    if preferred:
                        available = preferred
                cell.append(rng.choice(available))
            row.append(cell)
        grid.append(row)
    return grid


def _parent_element_ctx(rng, G, idx):
    """Historical filling, reconstructed only for archive exclusion."""
    ta, tb = _SUPPORT_DESIGNS[idx]
    sa, sb = [list(row) for row in ta], [list(row) for row in tb]
    assert G.q == 8
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    q, width = G.q, A_SHAPE[1]*G.q
    finalists, seen = [], set()

    def preliminary(item):
        # Genuine classical witnesses guide selection, but are not
        # automatically claimed to be nontrivial quantum logicals.
        d = min(item["da"], item["db"], 6)
        return (item["k"]*d*d, d, -item["cycles"])

    def admit(A, B):
        key = tuple(tuple(sorted(cell))
                    for grid in (A, B) for row in grid for cell in row)
        if key in seen:
            return
        seen.add(key)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if k <= 0 or 40*k <= 21*n:
            return
        ar, br = _component_checks(hx, hz, q)
        item = dict(
            A=A, B=B, hx=hx, hz=hz, n=n, k=k,
            ar=ar, br=br,
            da=_kernel_probe(ar, width, rng),
            db=_kernel_probe(br, width, rng),
            cycles=_four_cycles(ar)+_four_cycles(br))
        finalists.append(item)
        finalists.sort(key=preliminary, reverse=True)
        del finalists[10:]

    for attempt in range(96):
        model = (idx+attempt//24) % 4
        admit(_label_grid(rng, G, ta, model),
              _label_grid(rng, G, tb, (model+1) % 4))

    for step in range(48):
        if not finalists:
            admit(_label_grid(rng, G, ta),
                  _label_grid(rng, G, tb))
            continue
        parent = finalists[rng.randrange(min(4, len(finalists)))]
        A = [[list(cell) for cell in row] for row in parent["A"]]
        B = [[list(cell) for cell in row] for row in parent["B"]]
        side = ("A" if parent["da"] < parent["db"] else "B")
        if step % 4 == 0:
            side = rng.choice(("A", "B"))
        grid, counts = (A, ta) if side == "A" else (B, tb)
        positions = [(i, j) for i, row in enumerate(counts)
                     for j, count in enumerate(row) if count]
        i, j = rng.choice(positions)
        proposal = _label_grid(rng, G, counts, (idx+step) % 4)
        grid[i][j] = proposal[i][j]
        if step % 6 == 0:
            ii, jj = rng.choice(positions)
            grid[ii][jj] = proposal[ii][jj]
        admit(A, B)

    if not finalists:
        raise ValueError("no legal high-rate labeling")

    # Independent component rescreens can only lower witness readings.
    for item in finalists:
        for _ in range(4):
            item["da"] = min(
                item["da"], _kernel_probe(item["ar"], width, rng))
            item["db"] = min(
                item["db"], _kernel_probe(item["br"], width, rng))
    finalists.sort(key=preliminary, reverse=True)

    winner, best_score = None, None
    for item in finalists:
        hx, hz, n = item["hx"], item["hz"], item["n"]
        xc, zc = _columns(hz, n), _columns(hx, n)
        xb, zb = _row_basis(hx), _row_basis(hz)
        dx = _short_logical(xc, xb, 3)
        dz = _short_logical(zc, zb, 3)
        if dx == 4:
            dx = _short_logical(xc, xb, 4)
        if dz == 4:
            dz = _short_logical(zc, zb, 4)
        floor = min(dx, dz)
        # Five is an exact exclusion sentinel, not a fabricated
        # distance witness or an upper-bound distance estimate.
        score = (item["k"]*floor*floor/n, floor,
                 min(item["da"], item["db"]), -item["cycles"])
        if best_score is None or score > best_score:
            winner, best_score = item, score

    A, B = winner["A"], winner["B"]
    assert css_ok(winner["hx"], winner["hz"])
    assert max(row.bit_count()
               for row in winner["hx"]+winner["hz"]) <= 12
    assert winner["n"] == 544 and winner["k"] >= 288
    assert 40*winner["k"] > 21*winner["n"]
    assert all(len(cell) == len(set(cell)) and len(cell) <= 5
               for grid in (A, B) for row in grid for cell in row)
    return {"A": A, "B": B, "selection": best_score}


@__import__("functools").lru_cache(maxsize=2)
def NEW_CTX(rng, G, idx):
    """Pure shared architecture search; negative idx memoizes its bank."""
    if idx >= 0:
        bank = NEW_CTX(None, G, -1)
        if idx >= len(bank):
            raise ValueError("no further fresh even-hypercolumn design")
        A, B, score = bank[idx]
        return {"A": A, "B": B, "selection": score}

    from collections import Counter
    assert G.q == 8
    q, width = G.q, 8*G.q
    elements = tuple(G.elements)
    central = (2, 0)
    noncentral = tuple(g for g in elements
                       if g != G.e and g != central)
    local = random.Random(BASE_SEED*100003 + 80550317)

    try:
        import numpy as np
    except ImportError:
        np = None

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def mono():
        # Every relative occurs exactly once, unlike independent draws.
        return [[[G.e] for _ in elements],
                [[g] for g in elements]]

    def split(character):
        """A character-bipartite, complete degree-two component."""
        u, v = character

        def parity(g):
            return (u*(g[0] & 1) + v*g[1]) & 1

        odd = [g for g in elements if parity(g)]
        relatives = [g for g in odd
                     if G.idx(g) < G.idx(G.inv(g))]
        assert len(odd) == 4 and len(relatives) == 2
        row0 = [[G.e, relatives[0]], [G.e, relatives[1]],
                [], []] + [[G.e] for _ in odd]
        row1 = [[], [], [G.e, relatives[0]],
                [G.e, relatives[1]]] + [[g] for g in odd]
        return [row0, row1]

    characters = ((1, 0), (0, 1), (1, 1))
    cores = [mono()] + [split(c) for c in characters]

    def fingerprint(hx, hz, n):
        """Invariant of the combined, uncoloured Tanner graph.

        Unequal signatures certify nonisomorphism. Equal signatures
        are discarded conservatively, not called an isomorphism proof.
        """
        rows = list(hx) + list(hz)
        weights = [r.bit_count() for r in rows]
        size = len(rows)
        links = [dict() for _ in rows]
        adjacency = [0]*size
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                if overlap:
                    links[i][j] = links[j][i] = overlap
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i

        keys = []
        for i, neighbors in enumerate(links):
            profile = Counter(
                (weights[j], overlap,
                 (adjacency[i] & adjacency[j]).bit_count())
                for j, overlap in neighbors.items())
            keys.append((weights[i], tuple(sorted(profile.items()))))

        history, colours = [], []
        for _ in range(3):
            histogram = Counter(keys)
            history.append(tuple(sorted(histogram.items())))
            palette = {key: c for c, key in enumerate(sorted(histogram))}
            colours = [palette[key] for key in keys]
            keys = [
                (colours[i], tuple(sorted(Counter(
                    (colours[j], overlap)
                    for j, overlap in links[i].items()).items())))
                for i in range(size)]

        incidents = [[] for _ in range(n)]
        for colour, source in zip(colours, rows):
            while source:
                bit = source & -source
                incidents[bit.bit_length()-1].append(colour)
                source ^= bit
        coarse = (
            n, tuple(history),
            tuple(sorted(Counter(
                tuple(sorted(v)) for v in incidents).items())))

        if np is None:
            return coarse

        # Integer overlap Gram matrices are conjugated by a check
        # permutation under an uncoloured Tanner isomorphism.
        # These are exact finite-field invariants, not floating spectra.
        prime = 65521
        gram = np.zeros((size, size), dtype=np.int64)
        for i in range(size):
            gram[i, i] = weights[i]
            for j, overlap in links[i].items():
                gram[i, j] = overlap
        square = (gram @ gram) % prime
        cube = (square @ gram) % prime
        diagonal = Counter(
            (weights[i], colours[i],
             int(square[i, i]), int(cube[i, i]))
            for i in range(size))
        # Gram is symmetric: these sums are traces of powers four/six.
        # At this size all products and sums fit safely in int64.
        moments = (
            int(np.sum(square* square) % prime),
            int(np.sum(cube* cube) % prime),
            tuple(sorted(diagonal.items())))
        return coarse + (moments,)

    archived = set()
    # This bed has one recorded program. Reconstruct its sixty displayed
    # outputs, even when this invocation requests only twenty slots.
    old_count = min(len(_SUPPORT_DESIGNS), max(60, N_SLOTS))
    for slot in range(old_count):
        old_rng = random.Random(BASE_SEED*10007 + slot)
        old = _parent_element_ctx(old_rng, G, slot)
        hx, hz, n = lifted_matrices(G, old["A"], old["B"])
        archived.add(fingerprint(hx, hz, n))

    kept = {}

    def admit(A, B):
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        if _grid_weights(sa, sb) > W_MAX:
            return False
        if not _no_dead_lines(sa, sb):
            return False
        if any(len(c) != len(set(c)) or len(c) > CELL_MAX
               for grid in (A, B) for row in grid for c in row):
            return False

        hx, hz, n = lifted_matrices(G, A, B)
        ar, br = _component_checks(hx, hz, q)
        # Conservative component tests are cheap and reject actual
        # dependencies rather than trusting a support geometry.
        if _short_logical(_columns(ar, width), {}, 3) != 4:
            return False
        if _short_logical(_columns(br, width), {}, 3) != 4:
            return False

        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k < 294:
            return False
        fp = fingerprint(hx, hz, n)
        if fp in archived or fp in kept:
            return False

        xc, zc = _columns(hz, n), _columns(hx, n)
        xb, zb = _row_basis(hx), _row_basis(hz)
        if _short_logical(xc, xb, 3) != 4:
            return False
        if _short_logical(zc, zb, 3) != 4:
            return False
        dx = _short_logical(xc, xb, 4)
        dz = _short_logical(zc, zb, 4)
        floor = min(dx, dz)
        assert floor in (4, 5)
        assert n == 544 and css_ok(hx, hz)
        assert max(r.bit_count() for r in hx+hz) <= W_MAX
        score = (k*floor*floor/n, floor, k,
                 -_four_cycles(ar)-_four_cycles(br))
        kept[fp] = (freeze(A), freeze(B), score)
        return True

    # Controls change the equivariant component architecture, not just
    # the labels of one matrix. Invariants collapse any equivalent pair.
    for A in cores:
        for B in cores:
            admit(A, B)

    def axis(g):
        return (g[0] & 1) + 2*g[1]

    def extend(cell, amount, policy, row, preferred):
        """Add distinct terms; central differences are allowed on purpose."""
        anchor = cell[0]
        if amount == 1:
            if policy == 0:
                step = preferred
            elif policy == 1:
                pool = [g for g in noncentral
                        if axis(g) != axis(preferred)]
                step = local.choice(pool)
            elif policy == 2:
                step = central if row == 0 else preferred
            else:
                step = local.choice(
                    [g for g in elements if g != G.e])
            added = [G.mul(anchor, step)]
        else:
            if policy == 0:
                steps = (preferred, G.inv(preferred))
            elif policy == 1:
                other = local.choice(
                    [g for g in noncentral
                     if axis(g) != axis(preferred)])
                steps = (preferred, other)
            elif policy == 2:
                steps = (central, preferred)
            else:
                steps = tuple(local.sample(
                    [g for g in elements if g != G.e], 2))
            added = [G.mul(anchor, step) for step in steps]
        result = list(cell) + added
        assert len(result) == len(set(result))
        return result

    # Five cardinality geometries times four direction/action policies:
    # 0: one balanced degree-four column;
    # 1: one (3,1) degree-four column;
    # 2: two balanced degree-four columns;
    # 3: alternating (3,1),(1,3) degree-four columns;
    # 4: one balanced cross-column in a character-split component.
    #
    # A always has row sums eight and column sums two. B row sums
    # never exceed ten and column sums never exceed four. Thus both
    # CSS inequalities are satisfied, including the extremal 8+4=12.
    for card in range(20):
        geometry, policy = divmod(card, 4)
        A = cores[policy]
        successes = 0
        labels_seen = set()
        for attempt in range(96):
            if geometry == 4:
                character = characters[(policy + attempt//32) % 3]
                B = split(character)
                available = list(range(4, 8))
            else:
                B = mono()
                available = list(range(8))

            # Relative placement of two hypercolumns is optimized, not
            # generated by permuting a previously emitted complete code.
            first = available[0]
            second = local.choice(available[1:])
            preferred = local.choice(noncentral)
            if geometry in (0, 4):
                changes = ((0, first, 1), (1, first, 1))
            elif geometry == 1:
                changes = ((0, first, 2),)
            elif geometry == 2:
                changes = ((0, first, 1), (1, first, 1),
                           (0, second, 1), (1, second, 1))
            else:
                changes = ((0, first, 2), (1, second, 2))

            for row, column, amount in changes:
                B[row][column] = extend(
                    B[row][column], amount, policy, row, preferred)
                if column == second:
                    preferred = local.choice(noncentral)

            # Each column remains even, so the component has the
            # constant row dependency. Complete ranks are still measured.
            assert all(sum(len(B[i][j]) for i in range(2)) % 2 == 0
                       for j in range(8))
            label = tuple(tuple(sorted(c)) for row in B for c in row)
            if label in labels_seen:
                continue
            labels_seen.add(label)
            if admit(A, B):
                successes += 1
                if successes >= max(3, (N_SLOTS+19)//20):
                    break

    selected = sorted(kept.values(), key=lambda item: item[2],
                      reverse=True)[:N_SLOTS]
    # Cache an exhausted bank too; never rerun it for every void slot
    # and never pad it with archived codes or another slot's orbit copy.
    return tuple(selected)


def CELL(rng, G, side, i, j, ctx):
    """Return the complete jointly screened cell without further draws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Degree-distinguished Q8 supports and retained component probes exceed the 0.550 frontier fitness 9.429: the measured program scored 8.6118, all five leaders retained weight-four sector-one logicals, and several higher-k slots had d2s=2.
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

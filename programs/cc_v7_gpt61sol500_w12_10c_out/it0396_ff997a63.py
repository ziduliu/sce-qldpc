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
CONCEPT = ("C9:C6 with faithful action 2, order 54, A=(1,3), B=(1,3): independently short-eight-screened cubic constituents and verified full logical searches should supply five fresh readings >=10 and exceed the 0.400 frontier fitness 34.2; measured searches can refute this.")
GROUP = ("Jump from S3 x C4 to the full holomorph C9:C6. Since 2 has order six modulo nine, the presentation is associative and has trivial centre. Recorded order-54 beds have centres three or six and are different groups. Order 54 gives n=540; the faithful action supplies new relative-label geometry, not a certified distance advantage.")
SHAPES = ("Retarget cell 0.400: equal (1,3) sides give n=540, kb=216, guaranteed rate 2/5 and equal 162-variable/54-check components. The shape admits q<=59; this group has fixed order 54. Six trinomials give HX=HZ=12. Parent minimum weight-four witnesses occupied sector-one Acol/Bcol blocks broadly, so replace both constituent architectures rather than permute their labels.")
FILLING = ("Twenty structural cards cross five quotient-occupancy arrangements with four rotation-residue multiplicity policies. Pack relative differences, reject actual constituent four-cycles, exactly test constituent kernels through eight using block transitivity, and retain true kernel witnesses under deeper searches. Assemble verified ranks in [0.375,0.425), exclude full kernels through four, and search both complete logical quotients with stabilizer dressings. Unequal uncoloured Tanner signatures distinguish outputs; equal signatures are conservatively rejected. All sampled readings remain upper witnesses, not distance certificates.")
FAMILY_NAME = "r040_HolC9_13x13_cubic_short8_jointlogical_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 54

def custom_ops(q):
    # Hol(C9): x^9=y^6=1, y*x*y^-1=x^2.
    # 2 has order six modulo nine.
    if not valid_q(q):
        raise ValueError("Hol(C9) requires order 54")
    els = [(a, b) for b in range(6) for a in range(9)]
    action = [pow(2, b, 9) for b in range(6)]

    def red(g):
        return (g[0] % 9, g[1] % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + action[b]*c) % 9, (b+d) % 6)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 6
        return ((-action[bi]*a) % 9, bi)

    def idx(g):
        a, b = red(g)
        return 9*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


from functools import lru_cache
from collections import Counter


_H54_PATTERNS = (
    (0, 1, 2), (0, 0, 1), (0, 0, 2),
    (0, 2, 4), (0, 1, 1), (0, 2, 2),
)
_H54_CARDS = (
    (0, 0, 0), (0, 1, 2), (1, 1, 4),
    (2, 3, 5), (0, 3, 4),
)


def _h54_columns(rows, width):
    """Transpose only scaffold-derived matrices."""
    cols = [0]*width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            cols[bit.bit_length()-1] |= 1 << r
            source ^= bit
    return tuple(cols)


def _h54_basis(rows):
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


def _h54_outside(word, basis):
    while word:
        p = word.bit_length()-1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _h54_syndrome(cols, word):
    value = 0
    while word:
        bit = word & -word
        value ^= cols[bit.bit_length()-1]
        word ^= bit
    return value


def _h54_clean4(cols):
    """Exact conservative kernel exclusion through four."""
    seen = {0}
    for syndrome in cols:
        if syndrome in seen:
            return False
        seen.add(syndrome)
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            if value in seen:
                return False
            seen.add(value)
    return True


@lru_cache(maxsize=1)
def _h54_atlas(G):
    """F2-linear blocks compiled from verified single-term lifts."""
    q, mask = G.q, (1 << G.q)-1
    stub = [[[G.e], [G.e], [G.e]]]
    atlas = {}
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[g], [], []]]
            A, B = (unit, stub) if side == "A" else (stub, unit)
            hx, hz, _ = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            blocks.append(_h54_columns(
                [row & mask for row in source[:q]], q))
        atlas[side] = tuple(blocks)
    return atlas


def _h54_component(G, atlas, side, grid):
    cols = [0]*(3*G.q)
    for j, cell in enumerate(grid[0]):
        for g in cell:
            for h, syndrome in enumerate(atlas[side][G.idx(g)]):
                cols[j*G.q+h] ^= syndrome
    return tuple(cols)


def _h54_cycles(cols, height):
    """Actual four- and six-cycle counts for cubic constituents."""
    pairs, four = {}, 0
    adjacency = [0]*height
    for syndrome in cols:
        checks, remaining = [], syndrome
        while remaining:
            bit = remaining & -remaining
            checks.append(bit.bit_length()-1)
            remaining ^= bit
        assert len(checks) == 3
        for i, a in enumerate(checks):
            for b in checks[:i]:
                key = tuple(sorted((a, b)))
                old = pairs.get(key, 0)
                four += old
                pairs[key] = old+1
                adjacency[a] |= 1 << b
                adjacency[b] |= 1 << a
    if four:
        return four, 0
    triangles = 0
    for a, neighbours in enumerate(adjacency):
        remaining = neighbours
        while remaining:
            bit = remaining & -remaining
            b = bit.bit_length()-1
            triangles += (neighbours & adjacency[b]).bit_count()
            remaining ^= bit
    # Each variable contributes one same-variable check triangle.
    return 0, triangles//6-len(cols)


def _h54_probe(cols, rng, trials, seeds=(), stabilizers=None):
    """Actual kernel witnesses; logical readings exclude stabilizers."""
    span = None if stabilizers is None else _h54_basis(stabilizers)
    generators = [] if stabilizers is None else list(stabilizers)
    best = len(cols)+1
    memory = sorted({w for w in seeds if w}, key=int.bit_count)[:40]
    order = list(range(len(cols)))

    def observe(word):
        nonlocal best
        if word and word.bit_count() < best:
            if span is None or _h54_outside(word, span):
                best = word.bit_count()

    for word in memory:
        assert _h54_syndrome(cols, word) == 0
        observe(word)

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
            if not syndrome:
                dependencies.append(word)
                observe(word)

        words = sorted(set(dependencies+memory), key=int.bit_count)
        for sweep in range(2):
            reducers = sorted(set(words+memory), key=int.bit_count)[:32]
            reduced = []
            for word in words[:80]:
                weight = word.bit_count()
                for reducer in reducers:
                    candidate = word ^ reducer
                    cw = candidate.bit_count()
                    if 0 < cw < weight:
                        word, weight = candidate, cw
                reduced.append(word)
                observe(word)
            words = sorted(set(reduced+memory), key=int.bit_count)

        discoveries = set(words)
        short = words[:32]
        for i, word in enumerate(short):
            for other in short[:i]:
                candidate = word ^ other
                if candidate:
                    observe(candidate)
                    if candidate.bit_count() <= best+4:
                        discoveries.add(candidate)

        if span is not None:
            logicals = sorted(
                (w for w in discoveries if _h54_outside(w, span)),
                key=int.bit_count)[:4]
            for word in logicals:
                for sweep in range(2):
                    weight, chosen = word.bit_count(), word
                    nearby = sorted(
                        (((word ^ row).bit_count(), row)
                         for row in generators),
                        key=lambda entry: entry[0])[:12]
                    for _, row in nearby:
                        candidate = word ^ row
                        if candidate.bit_count() < weight:
                            chosen, weight = candidate, candidate.bit_count()
                    for i, (_, row) in enumerate(nearby):
                        for _, other in nearby[:i]:
                            candidate = word ^ row ^ other
                            if candidate.bit_count() < weight:
                                chosen = candidate
                                weight = candidate.bit_count()
                    if chosen == word:
                        break
                    word = chosen
                    observe(word)
                discoveries.add(word)
            # Preserve logical directions separately from short checks.
            memory = sorted(discoveries, key=int.bit_count)[:28]
            memory += sorted(
                (w for w in discoveries if _h54_outside(w, span)),
                key=int.bit_count)[:12]
        else:
            memory = sorted(discoveries, key=int.bit_count)[:40]
        if best <= 4:
            break
    return best, tuple(sorted(set(memory), key=int.bit_count))


def _h54_short8(cols, q):
    """Exact constituent-kernel test through eight.

    Cubic columns force even kernel weights. Regular actions are
    transitive within each constituent block, so one anchor per block
    suffices here; no such reduction is used for full-code logicals.
    Nine is an exclusion sentinel, not a witnessed distance.
    """
    bits = [1 << i for i in range(len(cols))]
    incidence = [[] for _ in range(q)]
    tails = {0: 0}
    for i, syndrome in enumerate(cols):
        assert syndrome not in tails
        tails[syndrome] = bits[i]
        remaining = syndrome
        while remaining:
            bit = remaining & -remaining
            incidence[bit.bit_length()-1].append(i)
            remaining ^= bit
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            assert value not in tails
            tails[value] = bits[i] | bits[j]

    def decode(target, limit, forbidden):
        answer = None

        def visit(value, remaining, word):
            nonlocal answer
            if answer is not None:
                return
            if not value:
                answer = word
                return
            if value.bit_count() > 3*remaining:
                return
            if remaining <= 2:
                tail = tails.get(value)
                if (tail is not None and tail.bit_count() <= remaining
                        and not tail & (word | forbidden)):
                    answer = word | tail
                return
            check = (value & -value).bit_length()-1
            choices = sorted(
                incidence[check],
                key=lambda c: -(cols[c] & value).bit_count())
            for c in choices:
                if not bits[c] & (word | forbidden):
                    visit(value ^ cols[c], remaining-1, word | bits[c])
                    if answer is not None:
                        return

        visit(target, limit, 0)
        return answer

    for bound in (6, 8):
        for anchor in range(0, len(cols), q):
            completion = decode(cols[anchor], bound-1, bits[anchor])
            if completion is not None:
                word = completion | bits[anchor]
                assert _h54_syndrome(cols, word) == 0
                return word.bit_count(), word
    return 9, 0


def _h54_draw(rng, G, side, recipe, mode, orders):
    """Pack fresh relative geometry within a structural occupancy card."""
    for restart in range(12):
        used, row = set(), []
        for j, kind in enumerate(recipe):
            chosen = None
            for attempt in range(192):
                phase = rng.randrange(6)
                direction = rng.choice((1, 5))
                bs = [(phase+direction*b) % 6
                      for b in _H54_PATTERNS[kind]]
                if mode == 0:
                    aa = [rng.randrange(9) for _ in range(3)]
                elif mode == 1:
                    residues = rng.sample(range(3), 3)
                    aa = [r+3*rng.randrange(3) for r in residues]
                elif mode == 2:
                    residues = [rng.randrange(3)]*2
                    residues.append((residues[0]+rng.choice((1, 2))) % 3)
                    rng.shuffle(residues)
                    aa = [r+3*rng.randrange(3) for r in residues]
                else:
                    base, step = rng.randrange(9), rng.choice((1, 2, 4, 5, 7, 8))
                    aa = [(base+t*step) % 9 for t in range(3)]
                    rng.shuffle(aa)
                cell = tuple(zip(aa, bs))
                if len(set(cell)) != 3:
                    continue
                differences = {
                    (G.mul(g, G.inv(h)) if side == "A"
                     else G.mul(G.inv(g), h))
                    for g in cell for h in cell if g != h}
                if (len(differences) != 6 or differences & used
                        or any(orders[g] <= 3 for g in differences)):
                    continue
                relatives = [G.mul(G.inv(cell[0]), g) for g in cell[1:]]
                if G.mul(*relatives) == G.mul(*reversed(relatives)):
                    continue
                chosen = cell
                used.update(differences)
                break
            if chosen is None:
                break
            row.append(chosen)
        if len(row) == 3:
            return (tuple(row),)
    return None


def _h54_pool(rng, G, atlas, side, recipe, mode, orders, quota):
    """Independent structural search followed by equal deeper budgets."""
    raw, seen = [], set()
    for attempt in range(quota):
        grid = _h54_draw(rng, G, side, recipe, mode, orders)
        if grid is None:
            continue
        labels = tuple(tuple(sorted(cell)) for cell in grid[0])
        if labels in seen:
            continue
        seen.add(labels)
        cols = _h54_component(G, atlas, side, grid)
        four, six = _h54_cycles(cols, G.q)
        if four:
            continue
        raw.append(dict(grid=grid, cols=cols, six=six, attempt=attempt))
        raw.sort(key=lambda item: (item["six"], item["attempt"]))
        del raw[18:]

    tested = []
    for item in raw:
        if not _h54_clean4(item["cols"]):
            continue
        item["d"], item["words"] = _h54_probe(item["cols"], rng, 4)
        tested.append(item)
    tested.sort(key=lambda item: (item["d"], -item["six"]), reverse=True)
    tested = tested[:8]

    for item in tested:
        floor, witness = _h54_short8(item["cols"], G.q)
        seeds = item["words"] + ((witness,) if witness else ())
        d, words = _h54_probe(item["cols"], rng, 32, seeds)
        item["d"] = min(item["d"], d,
                        witness.bit_count() if witness else len(item["cols"])+1)
        item["words"] = words
        item["clean8"] = floor == 9
    tested.sort(key=lambda item: (
        min(item["d"], 12), item["clean8"], -item["six"]), reverse=True)
    return tested[:3]


def _h54_signature(hx, hz, n):
    """Uncoloured Tanner invariant; equality is rejected conservatively."""
    rows = list(hx)+list(hz)
    weights = [row.bit_count() for row in rows]
    links = [dict() for _ in rows]
    adjacency = [0]*len(rows)
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                links[i][j] = links[j][i] = overlap
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    keys = [
        (weights[i], tuple(sorted(Counter(
            (weights[j], overlap,
             (adjacency[i] & adjacency[j]).bit_count())
            for j, overlap in links[i].items()).items())))
        for i in range(len(rows))]
    history = []
    colours = []
    for _ in range(3):
        histogram = Counter(keys)
        history.append(tuple(sorted(histogram.items())))
        palette = {key: c for c, key in enumerate(sorted(histogram))}
        colours = [palette[key] for key in keys]
        keys = [
            (colours[i], tuple(sorted(Counter(
                (colours[j], overlap)
                for j, overlap in links[i].items()).items())))
            for i in range(len(rows))]
    qubits = [[] for _ in range(n)]
    for colour, source in zip(colours, rows):
        while source:
            bit = source & -source
            qubits[bit.bit_length()-1].append(colour)
            source ^= bit
    return (n, tuple(history), tuple(sorted(Counter(
        tuple(sorted(v)) for v in qubits).items())))


def _h54_embeddings(words, side, q):
    mask = (1 << q)-1
    if side == "B":
        return [word << (j*3*q) for word in words[:12] for j in range(3)]
    return [
        sum(((word >> (j*q)) & mask) << ((j*3+beta)*q)
            for j in range(3))
        for word in words[:12] for beta in range(3)]


@lru_cache(maxsize=1)
def _h54_bank(G):
    """Pure deterministic bank: optimize each card, never emit orbit padding."""
    assert G.q == 54 and A_SHAPE == B_SHAPE == (1, 3)
    assert _grid_weights([[3, 3, 3]], [[3, 3, 3]]) == 12
    atlas = _h54_atlas(G)
    orders = {}
    for g in G.elements:
        value = G.e
        for order in range(1, G.q+1):
            value = G.mul(value, g)
            if value == G.e:
                orders[g] = order
                break
        assert g in orders

    bank, emitted = [], set()
    q, mask = G.q, (1 << G.q)-1

    def merit(item):
        d = min(item["da"], item["db"], item["dx"], item["dz"], 12)
        return (item["k"]*d*d/item["n"], d,
                item["da"]+item["db"], -item["six"])

    for slot in range(N_SLOTS):
        card, mode = divmod(slot % 20, 4)
        batch = slot//20
        ad = _H54_CARDS[card]
        bd = _H54_CARDS[(card+1+batch) % len(_H54_CARDS)]
        local = random.Random(BASE_SEED*100003+54040017+104729*slot)
        winner = None

        for retry in range(2):
            ap = _h54_pool(local, G, atlas, "A", ad, mode, orders,
                           112+48*retry)
            bp = _h54_pool(local, G, atlas, "B", bd, (mode+1) % 4,
                           orders, 160+64*retry)
            proposals = []
            for aa in ap:
                for bb in bp:
                    proposals.append((
                        (min(aa["d"], bb["d"]),
                         aa["d"]+bb["d"], -aa["six"]-bb["six"]), aa, bb))
            proposals.sort(key=lambda entry: entry[0], reverse=True)
            finalists = []

            for _, aa, bb in proposals:
                A, B = aa["grid"], bb["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if not (n == 540 and 375*n <= 1000*k < 425*n):
                    continue

                ar = [
                    sum(((row >> (j*3*q)) & mask) << (j*q)
                        for j in range(3)) for row in hx[:q]]
                br = [row & ((1 << (3*q))-1) for row in hz[:q]]
                assert _h54_columns(ar, 3*q) == aa["cols"]
                assert _h54_columns(br, 3*q) == bb["cols"]

                xc, zc = _h54_columns(hz, n), _h54_columns(hx, n)
                if not (_h54_clean4(xc) and _h54_clean4(zc)):
                    continue
                fp = _h54_signature(hx, hz, n)
                if fp in emitted:
                    continue

                xs, zs = _h54_basis(hx), _h54_basis(hz)
                xseeds = [
                    w for w in _h54_embeddings(bb["words"], "B", q)
                    if _h54_syndrome(xc, w) == 0 and _h54_outside(w, xs)]
                zseeds = [
                    w for w in _h54_embeddings(aa["words"], "A", q)
                    if _h54_syndrome(zc, w) == 0 and _h54_outside(w, zs)]
                dx, xwords = _h54_probe(xc, local, 8, xseeds, hx)
                dz, zwords = _h54_probe(zc, local, 8, zseeds, hz)
                if dx > n or dz > n:
                    continue
                finalists.append(dict(
                    A=A, B=B, hx=hx, hz=hz, n=n, k=k, fp=fp,
                    xc=xc, zc=zc, dx=dx, dz=dz,
                    xwords=xwords, zwords=zwords,
                    da=aa["d"], db=bb["d"], six=aa["six"]+bb["six"]))
                if len(finalists) >= 3:
                    break

            for item in finalists:
                dx, _ = _h54_probe(
                    item["xc"], local, 16, item["xwords"], item["hx"])
                dz, _ = _h54_probe(
                    item["zc"], local, 16, item["zwords"], item["hz"])
                item["dx"] = min(item["dx"], dx)
                item["dz"] = min(item["dz"], dz)
                if winner is None or merit(item) > merit(winner):
                    winner = item
            if winner is not None and merit(winner)[1] >= 10:
                break

        if winner is None:
            bank.append(None)
            continue
        hx, hz = winner["hx"], winner["hz"]
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx+hz) == 12
        assert winner["k"] >= 216
        assert all(len(cell) == len(set(cell)) == 3
                   for grid in (winner["A"], winner["B"])
                   for row in grid for cell in row)
        emitted.add(winner["fp"])
        bank.append(dict(
            cells={"A": winner["A"], "B": winner["B"]},
            k=winner["k"], card=(ad, bd, mode), screen=merit(winner)))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign the jointly screened fresh design for this structural card."""
    selected = _h54_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh target-rate cubic design passed this card")
    return selected


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without further random labeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Uniform legal support rolling on C17, (2,5)x(2,6), did not suppress sector-1 weight-five logicals: all five measured parent codes had d30s=5.
# REFUTED: Odd-central S3 x C4 generator deletions overcome the high-rate weight-four bottleneck: all five measured parent leaders retained d30s=4, and fifteen principal slots collapsed to five isomorphism classes.
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

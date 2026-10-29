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
CONCEPT = ("C44, cyclic order 44, shapes (2,3)x(2,3): primitive doubletons, controlled heavier cells and independently deepened witness searches should supply five fresh low-rate codes exceeding fitness 60.3; measured evaluation can refute this.")
GROUP = ("Jump from nonabelian S3 x C4 to genuinely different cyclic C44. The neighboring C45 and C46 beds scored 59.2 and 60.3, while cyclic C44 is absent from BEDS. Order 44 gives n=572 and distinct order-4/order-11 subgroup structure. These comparisons motivate an experiment, not a certified distance advantage.")
SHAPES = ("Target cell 0.100 with equal (2,3) components: n=572, kb=44, guaranteed rate 1/13, and 132 variables/88 checks on each side. The shape permits q<=46; this program fixes q=44. Exact assembled ranks enforce k/n<0.125. Parent weight-four minima occupied sector-one Acol/Bcol blocks broadly; full-code probes and stabilizer dressings also inspect the second sector.")
FILLING = ("Enumerate support geometries with unequal combined Tanner-degree fingerprints and both LP weight sums <=12. Optimize primitive order-44 doubletons and mixed higher-weight supports independently by component rank strata. Retain syndrome-verified commuting cofactors and information-set kernel words, repair witness-heavy cells, screen cyclic-fiber witnesses, and give joint finalists equal deeper budgets. Full logical probes classify witnesses modulo exact stabilizer spans. All randomized readings remain upper estimates, not distance certificates.")
FAMILY_NAME = "r010_C44_23x23_primitive_cofactor_rankrepair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 44

def custom_ops(q):
    # Cyclic C44, equivalently C11 x C4.
    if not valid_q(q):
        raise ValueError("this cyclic family requires order 44")
    els = [(a,) for a in range(q)]

    def red(g):
        return (g[0] % q,)

    def mul(g, h):
        return ((g[0] + h[0]) % q,)

    def inv(g):
        return ((-g[0]) % q,)

    def idx(g):
        return g[0] % q

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


from functools import lru_cache


def _c44_supports():
    """Distinct full Tanner-degree architectures, not permutation copies."""
    from itertools import product, permutations
    permutations3 = tuple(permutations(range(3)))
    profiles = []
    for t in product(range(1, 5), repeat=6):
        rows = (sum(t[:3]), sum(t[3:]))
        cols = tuple(t[j] + t[j+3] for j in range(3))
        if min(rows) < 5 or max(rows) > 8:
            continue
        if min(cols) < 3 or max(cols) > 6:
            continue
        if t != min(
                tuple(t[3*r+c] for r in order for c in perm)
                for order in ((0,1), (1,0))
                for perm in permutations3):
            continue
        parity = tuple(sum((t[3*i+j] & 1) << j
                           for j in range(3)) for i in range(2))
        rank = int(bool(parity[0] or parity[1]))
        rank += int(bool(parity[0] and parity[1]
                         and parity[0] != parity[1]))
        profiles.append((t, rows, cols, rank))

    buckets = {}
    for a, (ta, ar, ac, ka) in enumerate(profiles):
        for tb, br, bc, kb in profiles[a:]:
            if max(ar)+max(bc) > W_MAX:
                continue
            if max(br)+max(ac) > W_MAX:
                continue
            checks = tuple(sorted(
                [x+y for x in ar for y in bc] +
                [x+y for x in br for y in ac]))
            qubits = tuple(sorted(
                [x+y for x in ac for y in bc] +
                [x+y for x in ar for y in br]))
            quality = (min(ac+bc), min(ar+br), sum(ta)+sum(tb),
                       -max(ac+bc), -max(ar+br))
            buckets.setdefault(tuple(sorted((ka,kb))), []).append(
                (quality, (checks,qubits), ta,tb))
    for bucket in buckets.values():
        bucket.sort(reverse=True)

    strata = ((0,0), (0,1), (1,1), (0,2), (1,2), (2,2))
    positions = {s: 0 for s in strata}
    seen, designs = set(), []
    while True:
        added = False
        for s in strata:
            bucket = buckets.get(s, ())
            while positions[s] < len(bucket):
                _, signature, ta, tb = bucket[positions[s]]
                positions[s] += 1
                if signature in seen:
                    continue
                seen.add(signature)
                designs.append((ta,tb,signature))
                added = True
                break
        if not added:
            break
    return tuple(designs)


def _c44_columns(rows, width):
    cols = [0]*width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            cols[bit.bit_length()-1] |= 1 << r
            source ^= bit
    return tuple(cols)


def _c44_component(hx, hz, q, side):
    """Extract ONLY the component convention built by the scaffold."""
    mask = (1 << q)-1
    if side == "A":
        rows = [
            sum(((hx[(i*3)*q+r] >> (j*3*q)) & mask) << (j*q)
                for j in range(3))
            for i in range(2) for r in range(q)]
    else:
        rows = [row & ((1 << (3*q))-1) for row in hz[:2*q]]
    return rows, _c44_columns(rows, 3*q)


def _c44_basis(rows):
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


def _c44_outside(word, basis):
    while word:
        p = word.bit_length()-1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _c44_clean4(cols):
    """Exact component-kernel exclusion through four; not quantum distance."""
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


def _c44_low_logical4(checks, stabilizers, n):
    """Exact logical existence through four, allowing small stabilizers."""
    cols = _c44_columns(checks, n)
    basis = _c44_basis(stabilizers)
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = word
        elif _c44_outside(word ^ old, basis):
            return True
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            word = (1 << i) | (1 << j)
            old = seen.get(value)
            if old is None:
                seen[value] = word
            elif _c44_outside(word ^ old, basis):
                return True
    return False


def _c44_words(cols, rng, trials, seeds=()):
    """Actual kernel words with cross-information-set recombination."""
    def valid(word):
        syndrome, remaining = 0, word
        while remaining:
            bit = remaining & -remaining
            syndrome ^= cols[bit.bit_length()-1]
            remaining ^= bit
        return word != 0 and syndrome == 0

    memory = sorted({w for w in seeds if valid(w)},
                    key=lambda w: (w.bit_count(), w))[:32]
    order = list(range(len(cols)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots, deps = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome,word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                deps.append(word)
        words = sorted(set(memory+deps),
                       key=lambda w: (w.bit_count(), w))
        discoveries = set(memory)
        for sweep in range(2):
            reducers = words[:32]
            reduced = []
            for word in words:
                weight = word.bit_count()
                for other in reducers:
                    candidate = word ^ other
                    cw = candidate.bit_count()
                    if 0 < cw < weight:
                        word, weight = candidate, cw
                reduced.append(word)
            words = sorted(set(reduced),
                           key=lambda w: (w.bit_count(), w))
            discoveries.update(words[:32])
        short = sorted(discoveries,
                       key=lambda w: (w.bit_count(), w))[:28]
        for i, word in enumerate(short):
            for other in short[:i]:
                if word ^ other:
                    discoveries.add(word ^ other)
        memory = sorted(discoveries,
                        key=lambda w: (w.bit_count(), w))[:32]
    return tuple(memory)


def _c44_cycles(rows):
    pairs, count = {}, 0
    for source in rows:
        positions = []
        while source:
            bit = source & -source
            positions.append(bit.bit_length()-1)
            source ^= bit
        for i, x in enumerate(positions):
            for y in positions[:i]:
                key = (y,x)
                old = pairs.get(key, 0)
                count += old
                pairs[key] = old+1
    return count


def _c44_draw(rng, counts, model):
    """Primitive binomials and intrinsically different larger supports."""
    from math import gcd
    primitive = [d for d in range(1,44) if gcd(d,44) == 1]
    grid = []
    for i in range(2):
        row = []
        for j in range(3):
            count = counts[3*i+j]
            best = None
            for attempt in range(32):
                anchor = rng.randrange(44)
                if count == 1:
                    residues = [anchor]
                elif count == 2:
                    residues = [anchor,
                                (anchor+rng.choice(primitive)) % 44]
                elif model == 1:
                    step = rng.choice(primitive)
                    offsets = (0,1,3,7)[:count]
                    residues = [(anchor+step*t) % 44 for t in offsets]
                elif model == 2:
                    phase = rng.randrange(2)
                    quota = [(phase+t) % 2 for t in range(count)]
                    if count == 4 and (i+j) % 2:
                        quota = [phase]*3+[1-phase]
                    rng.shuffle(quota)
                    residues = []
                    for parity in quota:
                        pool = [a for a in range(parity,44,2)
                                if a not in residues
                                and all((a-b) % 44 != 22
                                        for b in residues)]
                        residues.append(rng.choice(pool))
                else:
                    residues = rng.sample(range(44), count)
                if len(set(residues)) != count:
                    continue
                if any((a-b) % 44 == 22 for a in residues
                       for b in residues if a != b):
                    continue
                differences = [(a-b) % 44 for a in residues
                               for b in residues if a != b]
                energy = len(differences)-len(set(differences))
                if best is None or energy < best[0]:
                    best = (energy,residues)
                if model != 3 or energy == 0:
                    break
            if best is None:
                raise ValueError("cyclic support recipe exhausted")
            row.append([(a,) for a in best[1]])
        grid.append(row)
    return grid


def _c44_cofactors(G, grid, cols, inverse):
    """Commuting minors construct words, never an alternative matrix lift."""
    entries = [[set(G.inv(g) if inverse else g for g in cell)
                for cell in row] for row in grid]

    def product(left, right):
        answer = set()
        for g in left:
            for h in right:
                term = G.mul(g,h)
                if term in answer:
                    answer.remove(term)
                else:
                    answer.add(term)
        return answer

    minors = []
    for omitted in range(3):
        s,t = [j for j in range(3) if j != omitted]
        minors.append(
            product(entries[0][s],entries[1][t]) ^
            product(entries[0][t],entries[1][s]))
    found = []
    for shift in G.elements[::11]:
        word = sum(
            1 << (j*G.q+G.idx(G.mul(g,shift)))
            for j,minor in enumerate(minors) for g in minor)
        remaining, syndrome = word, 0
        while remaining:
            bit = remaining & -remaining
            syndrome ^= cols[bit.bit_length()-1]
            remaining ^= bit
        if word and syndrome == 0:
            found.append(word)
    return tuple(found)


def _c44_fiber_words(cols, rng):
    """Restricted cyclic-subgroup witnesses, with verified syndromes."""
    q, found = 44, set()
    for size in (2,4,11,22):
        step = q//size
        fibers = [sum(1 << (r+t*step) for t in range(size))
                  for r in range(step)]
        words = [fiber << (j*q) for j in range(3) for fiber in fibers]
        syndromes = []
        for word in words:
            value, syndrome = word, 0
            while value:
                bit = value & -value
                syndrome ^= cols[bit.bit_length()-1]
                value ^= bit
            syndromes.append(syndrome)
        for coefficients in _c44_words(tuple(syndromes), rng, 3)[:12]:
            expanded = 0
            while coefficients:
                bit = coefficients & -coefficients
                expanded ^= words[bit.bit_length()-1]
                coefficients ^= bit
            if expanded:
                found.add(expanded)
    return tuple(sorted(found, key=lambda w: (w.bit_count(),w))[:24])


def _c44_logical_probe(checks, stabilizers, n, rng, trials, seeds):
    """Actual full-code logicals, including stabilizer-dressed mixed sectors."""
    cols = _c44_columns(checks,n)
    basis = _c44_basis(stabilizers)
    words = _c44_words(cols,rng,trials,seeds)
    logicals = [w for w in words if _c44_outside(w,basis)]
    # Kernel reduction may favor stabilizers. Keep every genuine seed too.
    for word in seeds:
        if not word or not _c44_outside(word,basis):
            continue
        if not any((word & row).bit_count() & 1 for row in checks):
            logicals.append(word)
    best, retained = n+1, []
    for word in sorted(set(logicals), key=int.bit_count)[:12]:
        best = min(best,word.bit_count())
        for sweep in range(3):
            nearby = sorted(
                (((word ^ row).bit_count(),row) for row in stabilizers),
                key=lambda p: p[0])
            chosen, weight = word, word.bit_count()
            for cw,row in nearby:
                if cw < weight:
                    chosen,weight = word ^ row,cw
            moves = [row for _,row in nearby[:16]]
            for i,row in enumerate(moves):
                for other in moves[:i]:
                    candidate = word ^ row ^ other
                    cw = candidate.bit_count()
                    if cw < weight:
                        chosen,weight = candidate,cw
            if chosen == word:
                break
            word = chosen
            best = min(best,weight)
        retained.append(word)
    return best,tuple(retained)


@lru_cache(maxsize=1)
def _c44_bank(G):
    """Transparent deterministic cache of independently optimized geometries."""
    assert G.q == 44
    designs = _c44_supports()
    zero = [[[] for _ in range(3)] for _ in range(2)]
    bank = []
    for slot in range(N_SLOTS):
        if slot >= len(designs):
            bank.append(None)
            continue
        ta,tb,signature = designs[slot]
        sa = [list(ta[:3]),list(ta[3:])]
        sb = [list(tb[:3]),list(tb[3:])]
        assert _grid_weights(sa,sb) <= W_MAX
        assert _no_dead_lines(sa,sb)
        local = random.Random(BASE_SEED*100003+44010071+104729*slot)
        mode = slot % 4

        def component_key(item):
            d = item["words"][0].bit_count()
            return (min(d,32), -sum(w.bit_count() <= d+1
                                   for w in item["words"]),
                    -item["cycles"])

        def component_pool(side, counts, batch):
            strata,seen = {},set()

            def admit(grid):
                key = tuple(tuple(sorted(cell))
                            for row in grid for cell in row)
                if key in seen:
                    return
                seen.add(key)
                A,B = (grid,zero) if side == "A" else (zero,grid)
                hx,hz,_ = lifted_matrices(G,A,B)
                rows,cols = _c44_component(hx,hz,G.q,side)
                if not _c44_clean4(cols):
                    return
                seeds = _c44_cofactors(G,grid,cols,side == "B")
                words = _c44_words(cols,local,3,seeds)
                if not words:
                    return
                item = dict(grid=grid,rows=rows,cols=cols,words=words,
                            rank=gf2_rank(rows),cycles=_c44_cycles(rows))
                bucket = strata.setdefault(item["rank"],[])
                bucket.append(item)
                bucket.sort(key=component_key,reverse=True)
                del bucket[2:]

            for attempt in range(48+24*batch):
                admit(_c44_draw(local,counts,(mode+attempt//16) % 4))
            positions = [(i,j) for i in range(2) for j in range(3)]
            mask = (1 << 44)-1
            for step in range(24+12*batch):
                if not strata:
                    admit(_c44_draw(local,counts,mode))
                    continue
                rank = local.choice(sorted(strata))
                parent = local.choice(strata[rank])
                grid = [[list(cell) for cell in row]
                        for row in parent["grid"]]
                pressure = [
                    1+sum(((w >> (j*44)) & mask).bit_count()
                          for w in parent["words"][:8])
                    for i,j in positions]
                fresh = _c44_draw(local,counts,(mode+step//8) % 4)
                for _ in range(1+int(step % 5 == 0)):
                    i,j = local.choices(positions,weights=pressure,k=1)[0]
                    grid[i][j] = fresh[i][j]
                admit(grid)

            first = sorted((b[0] for b in strata.values()),
                           key=component_key,reverse=True)
            extras = sorted((x for b in strata.values() for x in b[1:]),
                            key=component_key,reverse=True)
            pool = first[:6]
            pool += extras[:max(0,6-len(pool))]
            for item in pool:
                seeds = item["words"]+_c44_fiber_words(item["cols"],local)
                item["words"] = _c44_words(item["cols"],local,20,seeds)
            return sorted(pool,key=component_key,reverse=True)[:4]

        def joint_key(item):
            d = min(item["a"]["words"][0].bit_count(),
                    item["b"]["words"][0].bit_count(),
                    item.get("dx",573),item.get("dz",573))
            return (item["k"]*d*d/572,d,
                    -item["a"]["cycles"]-item["b"]["cycles"])

        winner = None
        for batch in range(2):
            ap = component_pool("A",ta,batch)
            bp = component_pool("B",tb,batch)
            proposals = []
            for a in ap:
                for b in bp:
                    hx,hz,n = lifted_matrices(G,a["grid"],b["grid"])
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if k < 44 or 8*k >= n:
                        continue
                    proposals.append(dict(
                        A=a["grid"],B=b["grid"],hx=hx,hz=hz,n=n,k=k,
                        a=a,b=b))
            proposals.sort(key=joint_key,reverse=True)
            finalists = []
            mask = (1 << 44)-1
            for item in proposals[:8]:
                hx,hz,n = item["hx"],item["hz"],item["n"]
                if (_c44_low_logical4(hx,hz,n)
                        or _c44_low_logical4(hz,hx,n)):
                    continue
                aw = [
                    sum(((word >> (j*44)) & mask)
                        << ((j*3+beta)*44) for j in range(3))
                    for word in item["a"]["words"] for beta in range(3)]
                bw = [
                    word << (j*3*44)
                    for word in item["b"]["words"] for j in range(3)]
                assert all(not any((w & r).bit_count() & 1 for r in hx)
                           for w in aw)
                assert all(not any((w & r).bit_count() & 1 for r in hz)
                           for w in bw)
                dx,wx = _c44_logical_probe(hz,hx,n,local,6,bw)
                dz,wz = _c44_logical_probe(hx,hz,n,local,6,aw)
                if dx > n or dz > n:
                    continue
                item.update(dx=dx,dz=dz,wx=wx,wz=wz,aw=aw,bw=bw)
                finalists.append(item)
                if len(finalists) == 4:
                    break

            # Every finalist receives the same independent deeper budgets.
            for item in finalists:
                for side in ("a","b"):
                    component = item[side]
                    component["words"] = _c44_words(
                        component["cols"],local,48,component["words"])
                dx,wx = _c44_logical_probe(
                    item["hz"],item["hx"],572,local,16,
                    tuple(item["bw"])+item["wx"])
                dz,wz = _c44_logical_probe(
                    item["hx"],item["hz"],572,local,16,
                    tuple(item["aw"])+item["wz"])
                item["dx"],item["dz"] = min(item["dx"],dx),min(item["dz"],dz)
                if winner is None or joint_key(item) > joint_key(winner):
                    winner = item
            if winner is not None:
                break

        if winner is None:
            bank.append(None)
            continue
        assert css_ok(winner["hx"],winner["hz"])
        assert winner["n"] == 572 and 44 <= winner["k"]
        assert 8*winner["k"] < 572
        assert max(r.bit_count() for r in winner["hx"]+winner["hz"]) <= W_MAX
        assert all(len(cell) == len(set(cell)) <= CELL_MAX
                   for grid in (winner["A"],winner["B"])
                   for row in grid for cell in row)
        bank.append(dict(
            cells={side: tuple(tuple(tuple(cell) for cell in row)
                               for row in winner[side])
                   for side in ("A","B")},
            k=winner["k"],architecture=signature,
            selection=joint_key(winner)))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign a unique degree architecture; never refill with an orbit copy."""
    item = _c44_bank(G)[idx]
    if item is None:
        raise ValueError("no screened low-rate filling for this architecture")
    return item


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell without additional random labeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Uniform legal support rolling on C17, (2,5)x(2,6), did not suppress sector-1 weight-five logicals: all five measured parent codes had d30s=5.
# REFUTED: Fifteen S3 x C4 generator-deletion cards provide fifteen distinct principal codes: slots 0-14 collapsed to five isomorphism classes, and all five measured leaders retained sector-one weight-four logicals.
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

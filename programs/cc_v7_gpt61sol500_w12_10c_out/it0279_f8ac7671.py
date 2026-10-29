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
CONCEPT = ("On C13 semidirect C4 with faithful action 5, order 52, A=(1,3) and B=(1,3), noncommuting trinomials using order-four conjugation should produce five measured readings above the parent's distance-eight plateau in cell 0.400.")
GROUP = ("Jump from D29 of order 58 to the genuinely different centreless C13:C4 of order 52. Since 5^2=-1 and 5^4=1 modulo 13, the action is faithful. BEDS lists an order-52 group with centre 2, not this centreless group; this bed is untried. Avoiding inversion-only relative actions is a testable filling hypothesis, not evidence of greater distance.")
SHAPES = ("Keep equal (1,3) sides: n=520, kb=208 and guaranteed rate 0.400, with equal 156-variable/52-check components. Parent measured minima were exclusively sector-1 Acol/Bcol witnesses, often concentrated around Acol1 or Bcol1. Screen both components and actual assembled logical spaces; require verified 0.375<=k/n<0.425.")
FILLING = ("Twenty policies cross five quotient-support architectures with four normal-coordinate models. Every cell contains three distinct terms, so HX/HZ weights are exactly 12. Compile cell actions exclusively from verified unit lifts. Exclude component kernels through six before shortlisting, search weight eight with explicit budget flags, and independently probe genuine kernel words. Assemble target-rate codes, exclude full logicals through four, deepen actual logical probes, and reject equal uncoloured Tanner fingerprints across slots. Failed policies never substitute orbit copies. Search readings remain upper estimates, not distance certificates.")
FAMILY_NAME = "r040_C13semiC4_action5_13x13_preclean6_chiral_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 52

def custom_ops(q):
    # x^13=y^4=1; y*x*y^-1=x^5.
    # 5^2=-1, 5^4=1 modulo 13: a faithful order-four action.
    if not valid_q(q):
        raise ValueError("C13:C4 requires order 52")
    els = [(a, b) for b in range(4) for a in range(13)]

    def red(g):
        return (g[0] % 13, g[1] % 4)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + pow(5, b, 13)*c) % 13, (b+d) % 4)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 4
        return ((-pow(5, bi, 13)*a) % 13, bi)

    def idx(g):
        a, b = red(g)
        return 13*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _retired_D29_context(rng, G, idx):
    """Historical D29 bucket builder; the new group builder never calls it."""
    from collections import Counter
    from hashlib import blake2b

    q = G.q
    n1 = 9 * q
    mask = (1 << q) - 1
    # Representatives of the four non-AP affine triple classes over F29.
    geometries = (3, 4, 5, 9)

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            bit = 1 << r
            while row:
                low = row & -row
                out[low.bit_length() - 1] |= bit
                row ^= low
        return out

    def free_four(cols):
        # Every column here has odd weight, excluding odd kernels.
        if 0 in cols or len(set(cols)) != len(cols):
            return False
        seen = set()
        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                if syndrome in seen:
                    return False
                seen.add(syndrome)
        return True

    def cycles(cols):
        total = 0
        for i, x in enumerate(cols):
            for j in range(i):
                overlap = (x & cols[j]).bit_count()
                total += overlap * (overlap - 1) // 2
        return total

    def bucket(hx, hz, xc, zc, n):
        """Uncolored union-Tanner invariant, including local refinements.

        Isomorphic graphs have equal fingerprints and therefore equal
        buckets. Different emitted slot buckets cannot be isomorphic.
        """
        rows = hx + hz
        uc = [x | (z << len(hx)) for x, z in zip(xc, zc)]
        pair_counts = {}
        incidences = []
        row_keys = []
        for row in rows:
            bits = []
            while row:
                low = row & -row
                bits.append(low.bit_length() - 1)
                row ^= low
            incidences.append(bits)
            keys = []
            for a, x in enumerate(bits):
                for y in bits[a + 1:]:
                    key = x * n + y
                    keys.append(key)
                    pair_counts[key] = pair_counts.get(key, 0) + 1
            row_keys.append(keys)

        rc = [sum(pair_counts[key] - 1 for key in keys)
              for keys in row_keys]
        qc = [0] * n
        for key, count in pair_counts.items():
            if count > 1:
                x, y = divmod(key, n)
                value = count * (count - 1) // 2
                qc[x] += value
                qc[y] += value

        qlabels = [(uc[c].bit_count(), qc[c]) for c in range(n)]
        rlabels = [
            (len(bits), rc[r],
             tuple(sorted(Counter(qlabels[c] for c in bits).items())))
            for r, bits in enumerate(incidences)
        ]
        qrefined = []
        for c, col in enumerate(uc):
            neighbours = []
            while col:
                low = col & -col
                r = low.bit_length() - 1
                neighbours.append((len(incidences[r]), rc[r]))
                col ^= low
            qrefined.append(
                (qlabels[c], tuple(sorted(Counter(neighbours).items()))))
        fingerprint = (
            tuple(sorted(Counter(rlabels).items())),
            tuple(sorted(Counter(qrefined).items())),
        )
        digest = blake2b(repr(fingerprint).encode(),
                         digest_size=8).digest()
        return int.from_bytes(digest, "little") % N_SLOTS

    def has_six(cols):
        """Exact component weight-six existence after free_four.

        The verified component has a transitive action on each fibre.
        Translate a six-support to contain 0, q or 2q, then match
        disjoint triples. Only the component extraction uses LP conventions.
        """
        length = len(cols)
        bits = [1 << c for c in range(length)]
        anchored = {}
        for a in (0, q, 2 * q):
            for b in range(length):
                if b == a:
                    continue
                ab = cols[a] ^ cols[b]
                support = bits[a] | bits[b]
                for c in range(b + 1, length):
                    if c == a:
                        continue
                    anchored.setdefault(ab ^ cols[c], []).append(
                        support | bits[c])
        lookup = anchored.get
        for a in range(length):
            for b in range(a):
                ab = cols[a] ^ cols[b]
                support = bits[a] | bits[b]
                for c in range(b):
                    mates = lookup(ab ^ cols[c])
                    if mates:
                        triple = support | bits[c]
                        if any(not (triple & other) for other in mates):
                            return True
        return False

    def mixed_six(cols, x_checks):
        """Exact mixed-sector six-support test when all cells are units.

        An odd number of tail qubits requires odd sector-1 occupancy
        in each of the three check blocks. At total weight six this
        leaves only tail/sector-1 splits 1+5 and 3+3.
        An even nonzero tail support cannot vanish in an empty check
        block because each tail block map is invertible.
        """
        singles = [{} for _ in range(3)]
        for c in range(n1):
            block = (c // q) % 3 if x_checks else (c // q) // 3
            syndrome = (cols[c] >> (block * q)) & mask
            singles[block][syndrome] = 1 << c

        pair_maps = [None] * 3
        tail = [
            tuple((s >> (b * q)) & mask for b in range(3))
            for s in cols[n1:]
        ]

        # One tail qubit, with three sector-1 qubits in one block
        # and one in each of the other two.
        for segments in tail:
            for heavy in range(3):
                others = [b for b in range(3) if b != heavy]
                if any(segments[b] not in singles[b] for b in others):
                    continue
                if pair_maps[heavy] is None:
                    items = list(singles[heavy].items())
                    pairs = {}
                    for a, (s, v) in enumerate(items):
                        for t, w in items[:a]:
                            pairs[s ^ t] = v | w
                    pair_maps[heavy] = pairs
                pairs = pair_maps[heavy]
                for syndrome, support in singles[heavy].items():
                    mate = pairs.get(segments[heavy] ^ syndrome)
                    if mate is not None and not (mate & support):
                        return True

        # Three tail qubits and one sector-1 qubit in each block.
        for a in range(q):
            for b in range(a):
                ab = tuple(tail[a][h] ^ tail[b][h] for h in range(3))
                for c in range(b):
                    if all((ab[h] ^ tail[c][h]) in singles[h]
                           for h in range(3)):
                        return True
        return False

    pool = []
    admitted = 0
    attempts = max(256, 20 * N_SLOTS)
    for attempt in range(attempts):
        # Change triangle classes as well as gaps during optimization.
        # The invariant bucket, not random offsets, defines slot identity.
        ta = geometries[(idx + attempt // 48) % 4]
        tb = geometries[(idx // 4 + attempt // 24) % 4]
        scale_b = rng.randrange(1, 29)
        gaps = rng.sample(range(1, 15), 4)
        cells = {}
        for side, triangle, scale, ds in (
                ("A", ta, 1, gaps[:2]),
                ("B", tb, scale_b, gaps[2:])):
            row = [[(0, 0), (scale, 0),
                    ((scale * triangle) % 29, 0)]]
            for gap in ds:
                row.append([(0, 0), (gap, 0),
                            (rng.randrange(29), 1)])
            cells[side] = [row]

        hx, hz, n = lifted_matrices(G, cells["A"], cells["B"])
        xc, zc = columns(hx, n), columns(hz, n)
        ac = [xc[3 * j * q + h]
              for j in range(3) for h in range(q)]
        bc = zc[:3 * q]
        if not free_four(ac) or not free_four(bc):
            continue
        if bucket(hx, hz, xc, zc, n) != idx:
            continue
        if n - gf2_rank(hx) - gf2_rank(hz) != 4 * q:
            continue

        ca, cb = cycles(ac), cycles(bc)
        pool.append((max(ca, cb), ca + cb, attempt, cells,
                     hx, hz, xc, zc, ac, bc))
        admitted += 1
        if admitted >= 4:
            break

    if not pool:
        raise ValueError("no exact-rate filling in prescribed Tanner bucket")
    pool.sort(key=lambda item: item[:3])

    best = None
    best_key = None
    for ca, total, attempt, cells, hx, hz, xc, zc, ac, bc in pool:
        if not free_four(xc) or not free_four(zc):
            continue

        # Verify invertibility of every individual cell map by extracting
        # its columns from the scaffold's component matrices.
        unit_cells = all(
            gf2_rank(columns(component[j*q:(j+1)*q], q)) == q
            for component in (ac, bc) for j in range(3))
        a6, b6 = has_six(ac), has_six(bc)
        if unit_cells:
            z6 = mixed_six(xc, True)
            x6 = mixed_six(zc, False)
        else:
            # Do not claim exhaustive mixed-sector screening without units.
            x6 = z6 = True
        key = (int(a6) + int(b6) + int(x6) + int(z6),
               ca, total, attempt)
        if best_key is None or key < best_key:
            best_key, best = key, (cells, hx, hz, unit_cells)
        if key[0] == 0:
            break

    if best is None:
        # Legal, distinct-bucket fallback; distance is measured downstream.
        _, _, _, cells, hx, hz, _, _, _, _ = pool[0]
        screened = False
    else:
        cells, hx, hz, units = best
        screened = units and best_key[0] == 0
    assert css_ok(hx, hz)
    assert 580 - gf2_rank(hx) - gf2_rank(hz) == 232
    return {"cells": cells, "screened_through_six": screened,
            "selection": best_key}


from functools import lru_cache
from collections import Counter
from hashlib import blake2b


_F52_CARDS = (
    ((0, 0, 0), (0, 0, 0)),
    ((0, 0, 2), (0, 2, 2)),
    ((0, 2, 2), (2, 2, 2)),
    ((2, 2, 2), (2, 2, 2)),
    ((0, 1, 2), (0, 0, 2)),
)


def _f52_columns(rows, width):
    cols = [0]*width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length()-1] |= 1 << r
            row ^= bit
    return cols


def _f52_basis(rows):
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length()-1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old
    return basis


def _f52_outside(word, basis):
    while word:
        p = word.bit_length()-1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _f52_four(cols, stabilizers=()):
    """Exact logical/kernel existence test through four.

    A return value of True excludes those weights only. Singles and
    pairs include odd supports, so this also works on full-code columns.
    """
    basis = _f52_basis(stabilizers)
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        old = seen.get(syndrome)
        if old is not None:
            if _f52_outside(old ^ word, basis):
                return False
        else:
            seen[syndrome] = word
    for i, syndrome in enumerate(cols):
        for j in range(i):
            key = syndrome ^ cols[j]
            word = (1 << i) | (1 << j)
            old = seen.get(key)
            if old is not None:
                if _f52_outside(old ^ word, basis):
                    return False
            else:
                seen[key] = word
    return True


@lru_cache(maxsize=1)
def _f52_atlas(G):
    """Both actions are extracted from scaffold-generated unit lifts."""
    atlas = {}
    mask = (1 << G.q)-1
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            A, B = ([[[g]]], [[[]]]) if side == "A" else (
                [[[]]], [[[g]]])
            hx, hz, _ = lifted_matrices(G, A, B)
            rows = hx if side == "A" else hz
            block = tuple(_f52_columns(
                [row & mask for row in rows], G.q))
            assert all(c.bit_count() == 1 for c in block)
            blocks.append(block)
        atlas[side] = tuple(blocks)
    return atlas


def _f52_component(G, atlas, side, grid):
    cols = [0]*(3*G.q)
    for j, cell in enumerate(grid[0]):
        for g in cell:
            block = atlas[side][G.idx(g)]
            for h, syndrome in enumerate(block):
                cols[j*G.q+h] ^= syndrome
    assert all(c.bit_count() == 3 for c in cols)
    return tuple(cols)


def _f52_draw(local, G, kinds, mode, lane):
    """Three intrinsic quotient occupancy types, with relative models."""
    row = []
    # Adjacent repeated coset, opposite repeated coset, three cosets.
    quotient_pairs = ((0, 1), (0, 2), (1, 2))
    squares = (1, 3, 4, 9, 10, 12)
    nonsquares = (2, 5, 6, 7, 8, 11)
    for j, kind in enumerate(kinds):
        b, d = quotient_pairs[kind]
        if kind == 2 and (lane+j) % 2:
            b, d = 1, 3
        for _ in range(96):
            u = local.randrange(1, 13)
            if mode == 0:
                v = local.randrange(13)
            elif mode == 1:
                ratio = (1, 2, 5, 7)[(j+lane) % 4]
                v = ratio*u % 13
            elif mode == 2:
                v = local.choice(squares)*u % 13
            else:
                v = local.choice(nonsquares)*u % 13
            g, h = (u, b), (v, d)
            if g == h or h == G.e:
                continue
            if G.mul(g, h) == G.mul(h, g):
                continue
            cell = [G.e, g, h]
            assert len(set(cell)) == 3
            row.append(cell)
            break
        else:
            raise ValueError("faithful-action cell policy exhausted")
    return [row]


def _f52_short(cols, q, maximum, node_cap=None, hit_cap=1):
    """Anchored component decoding with explicit completion reporting.

    Through six there is no node cutoff. A regular component action
    moves a support element to coordinate zero in its variable block.
    A complete singleton/pair table terminates each search branch.
    """
    width = len(cols)
    bits = [1 << i for i in range(width)]
    tails = {0: 0}
    incidence = [0]*q
    pair_pressure = 0
    for i, syndrome in enumerate(cols):
        assert syndrome not in tails
        tails[syndrome] = bits[i]
        source = syndrome
        while source:
            bit = source & -source
            incidence[bit.bit_length()-1] |= bits[i]
            source ^= bit
    for i, syndrome in enumerate(cols):
        for j in range(i):
            key = syndrome ^ cols[j]
            assert key not in tails
            tails[key] = bits[i] | bits[j]
            pair_pressure += 8*(key.bit_count() == 2)
            pair_pressure += key.bit_count() == 4

    found = set()
    nodes = 0
    complete = True
    stop = False

    def record(word):
        nonlocal stop, complete
        if word:
            found.add(word)
            if len(found) >= hit_cap:
                complete = False
                stop = True

    def visit(value, remaining, word):
        nonlocal nodes, complete, stop
        if stop:
            return
        if node_cap is not None and nodes >= node_cap:
            complete = False
            stop = True
            return
        nodes += 1
        if value == 0:
            record(word)
            return
        if remaining <= 0 or value.bit_count() > 3*remaining:
            return
        if remaining <= 2:
            tail = tails.get(value)
            if (tail is not None and tail.bit_count() <= remaining
                    and not (tail & word)):
                record(word | tail)
            return

        source = value
        choices = None
        least = width+1
        while source:
            bit = source & -source
            available = incidence[bit.bit_length()-1] & ~word
            count = available.bit_count()
            if count == 0:
                return
            if count < least:
                choices, least = available, count
            source ^= bit
        options = []
        while choices:
            bit = choices & -choices
            choices ^= bit
            j = bit.bit_length()-1
            options.append((-(cols[j] & value).bit_count(), j))
        for _, j in sorted(options):
            visit(value ^ cols[j], remaining-1, word | bits[j])
            if stop:
                break

    for anchor in range(0, width, q):
        visit(cols[anchor], maximum-1, bits[anchor])
        if stop:
            break
    return tuple(found), complete, nodes, pair_pressure


def _f52_probe(cols, local, trials, stabilizers=()):
    """Upper estimates from genuine kernel/logical witnesses only."""
    basis = _f52_basis(stabilizers)
    order = list(range(len(cols)))
    memory = []
    best = len(cols)+1

    def consider(word):
        nonlocal best
        if word and word.bit_count() < best:
            if _f52_outside(word, basis):
                best = word.bit_count()

    for _ in range(trials):
        local.shuffle(order)
        pivots, words = {}, []
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
            if syndrome == 0:
                words.append(word)
                consider(word)

        reducers = sorted(set(words+memory),
                          key=lambda v: (v.bit_count(), v))[:32]
        for i, word in enumerate(words):
            for other in reducers:
                combined = word ^ other
                if combined and combined.bit_count() < word.bit_count():
                    word = combined
            words[i] = word
            consider(word)
        short = sorted(set(words+memory),
                       key=lambda v: (v.bit_count(), v))[:32]
        discoveries = list(short)
        for i, word in enumerate(short):
            for other in short[:i]:
                combined = word ^ other
                if combined:
                    consider(combined)
                    discoveries.append(combined)
        if stabilizers:
            for word in short[:6]:
                if not _f52_outside(word, basis):
                    continue
                for sweep in range(2):
                    changed = False
                    for row in stabilizers:
                        combined = word ^ row
                        if combined.bit_count() < word.bit_count():
                            word = combined
                            changed = True
                    if not changed:
                        break
                consider(word)
        memory = sorted(set(discoveries),
                        key=lambda v: (v.bit_count(), v))[:24]
        if best <= 6:
            break
    return best


def _f52_pool(local, G, atlas, side, kinds, mode, lane, quota):
    """Screen six on every admitted proposal, BEFORE shortlisting."""
    retained, seen = [], set()
    admitted = 0
    for attempt in range(quota):
        grid = _f52_draw(local, G, kinds, mode, lane)
        labels = tuple(tuple(sorted(cell)) for cell in grid[0])
        if labels in seen:
            continue
        seen.add(labels)
        cols = _f52_component(G, atlas, side, grid)
        rank = gf2_rank(cols)
        if rank < G.q-4 or not _f52_four(cols):
            continue
        words, complete, _, pressure = _f52_short(
            cols, G.q, 6, node_cap=None, hit_cap=1)
        if words:
            continue
        assert complete
        retained.append(dict(
            grid=grid, cols=cols, rank=rank,
            pressure=pressure, attempt=attempt))
        retained.sort(key=lambda item: (
            item["pressure"], item["attempt"]))
        del retained[6:]
        admitted += 1
        if admitted >= 14 and attempt >= 64:
            break

    for item in retained:
        words, complete, nodes, _ = _f52_short(
            item["cols"], G.q, 8, node_cap=18000, hit_cap=1)
        reading = _f52_probe(item["cols"], local, 12)
        if words:
            reading = min(reading, min(w.bit_count() for w in words))
        item["reading"] = reading
        item["eight_complete"] = complete
        item["eight_hit"] = bool(words)
        item["nodes"] = nodes

    retained.sort(key=lambda item: (
        item["reading"],
        int(item["eight_complete"] and not item["eight_hit"]),
        -item["pressure"], -item["attempt"]), reverse=True)
    # Fresh independent information sets can only lower the merit.
    for item in retained[:4]:
        item["reading"] = min(
            item["reading"], _f52_probe(item["cols"], local, 24))
    retained.sort(key=lambda item: (
        item["reading"], -item["pressure"], -item["attempt"]),
        reverse=True)
    return retained[:4]


def _f52_signature(hx, hz, n):
    """Permutation invariants of the union Tanner graph, without X/Z colours.

    Equal signatures are conservatively rejected. Inequality rules out
    isomorphism; hash collisions can only cause extra rejection.
    """
    rows = hx+hz
    neighbors = []
    incidents = [[] for _ in range(n)]
    for r, source in enumerate(rows):
        ids = []
        while source:
            bit = source & -source
            c = bit.bit_length()-1
            ids.append(c)
            incidents[c].append(r)
            source ^= bit
        neighbors.append(ids)

    adjacency = [Counter() for _ in rows]
    for checks in incidents:
        for i, a in enumerate(checks):
            for b in checks[:i]:
                adjacency[a][b] += 1
                adjacency[b][a] += 1

    def digest(value):
        return blake2b(repr(value).encode(), digest_size=16).digest()

    rc = [
        digest(("check", len(neighbors[r]),
                tuple(sorted(Counter(adj.values()).items()))))
        for r, adj in enumerate(adjacency)]
    qc = []
    for checks in incidents:
        cycle_load = sum(
            adjacency[a][b]-1
            for i, a in enumerate(checks) for b in checks[:i])
        qc.append(digest(("qubit", len(checks), cycle_load)))

    history = []
    for _ in range(3):
        history.append((
            tuple(sorted(Counter(rc).items())),
            tuple(sorted(Counter(qc).items()))))
        new_r = [digest((rc[r], tuple(sorted(qc[c] for c in ids))))
                 for r, ids in enumerate(neighbors)]
        new_q = [digest((qc[c], tuple(sorted(rc[r] for r in checks))))
                 for c, checks in enumerate(incidents)]
        rc, qc = new_r, new_q
    history.append((
        tuple(sorted(Counter(rc).items())),
        tuple(sorted(Counter(qc).items()))))
    return tuple(history)


@lru_cache(maxsize=1)
def _f52_bank(G):
    """Deterministic policy bank; no archived or transported fallbacks."""
    assert G.q == 52 and A_SHAPE == B_SHAPE == (1, 3)
    atlas = _f52_atlas(G)
    bank, emitted = [], set()
    mask = (1 << G.q)-1

    for slot in range(N_SLOTS):
        architecture, mode = divmod(slot % 20, 4)
        ak, bk = _F52_CARDS[architecture]
        lane = slot//20
        # Extra harness lanes alter relative quotient geometry, not a
        # common automorphism applied to a previously emitted code.
        if lane:
            bk = tuple((kind+lane) % 3 for kind in bk)
        chosen = None

        for batch in range(3):
            local = random.Random(
                BASE_SEED*100003 + 52040113
                + 100019*slot + 7919*batch)
            quota = 160+96*batch
            ap = _f52_pool(
                local, G, atlas, "A", ak, mode, lane, quota)
            bp = _f52_pool(
                local, G, atlas, "B", bk, mode, lane+1, quota)
            proposals = []
            for a in ap:
                for b in bp:
                    A, B = a["grid"], b["grid"]
                    hx, hz, n = lifted_matrices(G, A, B)
                    k = n-gf2_rank(hx)-gf2_rank(hz)
                    if not (375*n <= 1000*k < 425*n):
                        continue
                    d = min(a["reading"], b["reading"])
                    key = (k*d*d, d, -a["pressure"]-b["pressure"])
                    proposals.append((key, a, b, hx, hz, n, k))
            proposals.sort(key=lambda item: item[0], reverse=True)

            finalists = []
            for _, a, b, hx, hz, n, k in proposals:
                xc, zc = _f52_columns(hx, n), _f52_columns(hz, n)
                if not _f52_four(xc, hz) or not _f52_four(zc, hx):
                    continue
                fingerprint = _f52_signature(hx, hz, n)
                if fingerprint in emitted:
                    continue
                dx = _f52_probe(zc, local, 4, hx)
                dz = _f52_probe(xc, local, 4, hz)
                d = min(a["reading"], b["reading"], dx, dz)
                finalists.append(dict(
                    a=a, b=b, hx=hx, hz=hz, xc=xc, zc=zc,
                    n=n, k=k, dx=dx, dz=dz, d=d,
                    fingerprint=fingerprint))
                if len(finalists) >= 4:
                    break

            for item in finalists:
                item["dx"] = min(item["dx"], _f52_probe(
                    item["zc"], local, 12, item["hx"]))
                item["dz"] = min(item["dz"], _f52_probe(
                    item["xc"], local, 12, item["hz"]))
                item["d"] = min(
                    item["d"], item["dx"], item["dz"])
            finalists.sort(key=lambda item: (
                item["k"]*item["d"]**2, item["d"],
                -item["a"]["pressure"]-item["b"]["pressure"]),
                reverse=True)

            if not finalists:
                continue
            winner = finalists[0]
            A, B = winner["a"]["grid"], winner["b"]["grid"]
            hx, hz = winner["hx"], winner["hz"]

            # Independently verify atlas compilation against full lifts.
            ar = [
                sum(((row >> (3*j*G.q)) & mask) << (j*G.q)
                    for j in range(3))
                for row in hx[:G.q]]
            br = [row & ((1 << (3*G.q))-1) for row in hz[:G.q]]
            assert tuple(_f52_columns(ar, 3*G.q)) == winner["a"]["cols"]
            assert tuple(_f52_columns(br, 3*G.q)) == winner["b"]["cols"]
            assert winner["n"] == 520 and winner["k"] >= 208
            assert css_ok(hx, hz)
            assert _grid_weights([[3, 3, 3]], [[3, 3, 3]]) == 12
            assert _no_dead_lines([[3, 3, 3]], [[3, 3, 3]])
            assert max(row.bit_count() for row in hx+hz) == 12
            assert all(len(cell) == len(set(cell)) == 3
                       for grid in (A, B) for row in grid for cell in row)
            emitted.add(winner["fingerprint"])
            frozen = [
                tuple(tuple(tuple(cell) for cell in row) for row in grid)
                for grid in (A, B)]
            chosen = (frozen[0], frozen[1], winner["k"],
                      (winner["d"], winner["dx"], winner["dz"]))
            break

        bank.append(chosen)
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return fresh copies of the selected faithful-action policy."""
    selected = _f52_bank(G)[idx]
    if selected is None:
        raise ValueError("faithful-action policy found no fresh screened code")
    A, B, k, selection = selected
    return {"idx": idx, "k": k, "selection": selection,
            "cells": {
                "A": [[list(cell) for cell in row] for row in A],
                "B": [[list(cell) for cell in row] for row in B]}}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without additional sampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled counts with uniform labels reliably suppress short logicals on this bed; many seed slots returned d2s=2 and all five measured seed codes returned d30s=4, with sector-1 minima.
# REFUTED: The shown all-mixed trinomial gap screens reliably exceed measured distance eight; all three shown best programs scored 25.6 with five d30s=8 readings, motivating a different pure/mixed architecture.
# REFUTED: One asymmetric rotation-only trinomial per side plus prescribed invariant buckets escapes the D29 weight-eight plateau; all five measured current-parent codes still returned d30s=8 with sector-1 minima, and slots 0 and 7 found no exact-rate filling in their buckets.
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

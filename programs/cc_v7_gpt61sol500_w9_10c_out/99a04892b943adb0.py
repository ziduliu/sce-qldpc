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
CONCEPT = ("Metacyclic Z35:Z1(s=1), order 35, A=(2,3), B=(2,4): deliberately imposed seventh-root rank deficits and cofactor-aware constituent optimization should supply five novel target-cell codes exceeding fitness 25.290; distance survival is a falsifiable measurement hypothesis.")
GROUP = ("Keep the parent's cyclic specialization of the metacyclic construction rule, resizing only its order. Factor 16 permits q<=37; q=35 gives n=560. Related cyclic (2,3)x(2,4) beds reached approximately 25.3 at q=31 and 25.2 at q=37. Since 35=5*7 and ord_7(2)=3, seventh-root evaluation constraints can expose degree-three rank losses. Verified lifts determine the actual losses; no distance advantage is assumed.")
SHAPES = ("A=(2,3), B=(2,4) give 105-variable/70-check and 140-variable/70-check constituents. With q=35, n=560, kb=70 and guaranteed rate 0.125, verified complete ranks restrict publication to [0.125,0.175), targeting cell 0.150. Parent measured minima occupied sector-1 Acol/Bcol blocks, especially Acol1 with Bcol5..7, rather than sector-2 blocks. Both new sides use two-row cubic-column components.")
FILLING = ("Optimize balanced and exclusive cubic-column supports under free and two reciprocal GF8 seventh-root prescriptions. A rows total four/five and B rows six; every column totals three, so HX<=8 and HZ<=9. Exact constituent ranks validate prescribed degree-three deficits without accepting accidental additional losses. Verified cyclic cofactors, translated differences and equal-budget kernel probes supply actual witness upper bounds. Independently selected pairs receive actual-rate, CSS, complete kernel-four, embedded-logical and mixed-sector witness screens. An immutable portfolio conservatively rejects repeated intrinsic full-Tanner signatures. Larger witness readings and unseen-archive novelty remain hypotheses.")
FAMILY_NAME = "r015_m35r1s1_23x24_seventhroot_cofactor_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 35

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("the resized cyclic metacyclic group has order 35")
    # Same metacyclic rule and r=1,s=1 cyclic specialization.
    # Factor 16 admits order 35; seventh roots have degree three.
    m, r, s = 35, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _r035_columns(rows, width):
    """Syndromes extracted only from verified lifted matrices."""
    out = [0] * width
    mask = (1 << width) - 1
    for r, row in enumerate(rows):
        row &= mask
        tag = 1 << r
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= tag
            row ^= bit
    return out


def _r035_floor(cols, through_five=True):
    """Exact kernel screening; return 6 when weights <=5 are excluded.

    With through_five=False, return 5 when weights <=4 are excluded.
    This conservatively excludes short stabilizers as well.
    """
    singles = set(cols)
    if 0 in singles:
        return 1
    if len(singles) != len(cols):
        return 2
    pairs = set()
    four = False
    for i, x in enumerate(cols):
        for j in range(i):
            s = x ^ cols[j]
            if s in singles:
                return 3
            if s in pairs:
                four = True
            pairs.add(s)
    if four:
        return 4
    if not through_five:
        return 5
    # Smaller kernels have already been excluded, so a triple/pair
    # collision necessarily has disjoint support and weight five.
    for i, x in enumerate(cols):
        for j in range(i):
            s = x ^ cols[j]
            for h in range(j):
                if (s ^ cols[h]) in pairs:
                    return 5
    return 6


def _r035_signature(hx, hz):
    """An uncoloured Tanner invariant, not a label fingerprint.

    Checks are pooled without X/Z colours. Their intersection relations
    and common-neighbour profiles are preserved by Tanner isomorphisms.
    Different signatures therefore certify different Tanner graphs.
    """
    rows = hx + hz
    m = len(rows)
    one, many = [0] * m, [0] * m
    for i in range(m):
        for j in range(i):
            overlap = (rows[i] & rows[j]).bit_count()
            if overlap:
                adj = one if overlap == 1 else many
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    profiles = [{} for _ in rows]
    for i in range(m):
        for j in range(i):
            key = (
                ((one[i] >> j) & 1) + 2 * ((many[i] >> j) & 1),
                (one[i] & one[j]).bit_count(),
                (one[i] & many[j]).bit_count()
                + (many[i] & one[j]).bit_count(),
                (many[i] & many[j]).bit_count(),
            )
            for p in (profiles[i], profiles[j]):
                p[key] = p.get(key, 0) + 1
    return tuple(sorted(
        (row.bit_count(), tuple(sorted(p.items())))
        for row, p in zip(rows, profiles)
    ))


def _r015_basis(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = word
                break
            word ^= old
    return pivots


def _r015_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        old = pivots.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _r015_rows(G, grid, side):
    """Constituent action extracted only from the verified construction."""
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    dummy = [[[]]]
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, dummy)
    else:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in rows]


def _r015_probe(local, rows, width, passes, seed=0, own=None):
    """Return an explicit kernel/logical witness, never an unresolved cap."""
    cols = _r035_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _r015_outside(word, own):
            return
        best, weight = word, w

    order = list(range(width))
    for turn in range(passes):
        local.shuffle(order)
        if own is not None and turn % 2:
            n1 = A_SHAPE[1] * B_SHAPE[1] * 35
            order.sort(key=lambda j: j < n1)
        pivots, kernel = {}, []
        for j in order:
            syndrome, word = cols[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                kernel.append(word)
                inspect(word)
        local.shuffle(kernel)
        for i in range(len(kernel)):
            word = kernel[i]
            for j, other in enumerate(kernel):
                if i != j:
                    trial = word ^ other
                    if 0 < trial.bit_count() < word.bit_count():
                        word = trial
            kernel[i] = word
            inspect(word)
        short = sorted(kernel, key=int.bit_count)[:24]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                inspect(pair)
                if i < 16:
                    for h in range(j):
                        inspect(pair ^ short[h])
        if best:
            starting = best
            for i, first in enumerate(short):
                inspect(starting ^ first)
                for second in short[:i]:
                    inspect(starting ^ first ^ second)
        if weight <= 10:
            break
    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        assert own is None or _r015_outside(best, own)
    return best


def _r015_cofactor(rows, count, q):
    """Cofactors and translated differences from verified column syndromes."""
    from itertools import combinations
    mask = (1 << q) - 1
    cols = _r035_columns(rows, count * q)
    polys = [[(cols[j * q] >> (i * q)) & mask
              for j in range(count)] for i in range(2)]

    def rotate(poly, shift):
        return poly if shift == 0 else (
            ((poly << shift) | (poly >> (q - shift))) & mask)

    def multiply(a, b):
        out = 0
        while a:
            bit = a & -a
            out ^= rotate(b, bit.bit_length() - 1)
            a ^= bit
        return out

    best = 0
    for js in combinations(range(count), 3):
        pieces = [0] * count
        for omitted, j in enumerate(js):
            a, b = js[:omitted] + js[omitted + 1:]
            pieces[j] = (multiply(polys[0][a], polys[1][b])
                         ^ multiply(polys[0][b], polys[1][a]))
        word = sum(poly << (j * q) for j, poly in enumerate(pieces))
        if not word:
            continue
        assert all((row & word).bit_count() % 2 == 0 for row in rows)
        if not best or word.bit_count() < best.bit_count():
            best = word
        for shift in range(1, q):
            moved = sum(rotate(poly, shift) << (j * q)
                        for j, poly in enumerate(pieces))
            trial = word ^ moved
            if trial and trial.bit_count() < best.bit_count():
                best = trial
    if best:
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return best


def _r015_draw(local, G, profile, policy, root):
    """Sparse evaluation constraints, not an independently implemented lift."""
    q = G.q
    extent = (5, 9, 13, 17)[policy]

    def multiply(a, b):
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & 8:
                a ^= 0xB
        return result

    beta = 1
    for _ in range(root or 1):
        beta = multiply(beta, 2)
    powers = [1]
    for _ in range(6):
        powers.append(multiply(powers[-1], beta))
    assert len(set(powers)) == 7
    logarithm = {v: a for a, v in enumerate(powers)}
    grid = [[[] for _ in profile[0]] for _ in range(2)]
    used = set()
    for j in range(len(profile[0])):
        counts = (profile[0][j], profile[1][j])
        assert sum(counts) == 3
        offsets, values = [], []
        for count in counts:
            if count == 0:
                terms = []
            elif count == 1:
                terms = [0]
            elif count == 2:
                choices = [d for d in range(1, extent + 1)
                           if d not in used and (root is None or d % 7)]
                if not choices:
                    choices = [d for d in range(1, 18)
                               if root is None or d % 7]
                d = local.choice(choices)
                used.add(d)
                terms = [0, d if local.randrange(2) else (-d) % q]
            elif root is not None:
                a = local.randrange(1, 7)
                b = logarithm[1 ^ powers[a]]
                terms = [0, a + 7 * local.randrange(5),
                         b + 7 * local.randrange(5)]
            else:
                terms = [0] + local.sample(range(1, q), 2)
            offsets.append(terms)
            value = 0
            for exponent in terms:
                value ^= powers[exponent % 7]
            values.append(value)

        anchors = [local.randrange(q), local.randrange(q)]
        if root is not None:
            if counts[0] and counts[1]:
                assert values[0] and values[1]
                residue = (anchors[0] + logarithm[values[0]]
                           - logarithm[values[1]]) % 7
                anchors[1] = residue + 7 * local.randrange(5)
            else:
                assert values == [0, 0]
        evaluations = []
        for i in range(2):
            labels = sorted((anchors[i] + d) % q for d in offsets[i])
            assert len(set(labels)) == counts[i]
            grid[i][j] = [G.elements[a] for a in labels]
            value = 0
            for a in labels:
                value ^= powers[a % 7]
            evaluations.append(value)
        if root is not None:
            assert evaluations[0] == evaluations[1]
    return grid


def _r015_bank(G, side):
    """Optimize support, spectral-factor and relative-gap prescriptions."""
    profiles = (
        (((1, 1, 2), (2, 2, 1)),
         ((0, 2, 2), (3, 1, 1)))
        if side == "A" else
        (((1, 1, 2, 2), (2, 2, 1, 1)),
         ((0, 1, 2, 3), (3, 2, 1, 0)))
    )
    bank, seen = [], set()
    for card in range(24):
        pi, rest = divmod(card, 12)
        policy, ri = divmod(rest, 3)
        root = (1, 3, None)[ri]
        profile = profiles[pi]
        augmentation = gf2_rank([
            sum((c & 1) << j for j, c in enumerate(row))
            for row in profile])
        expected = 2 * (G.q - 1) + augmentation - (
            3 if root is not None else 0)
        local = random.Random(
            BASE_SEED ^ (0x351507 + 104729 * card
                         + 1000003 * int(side == "B")))
        width = len(profile[0]) * G.q
        raw = []
        for trial in range(128):
            grid = _r015_draw(local, G, profile, policy, root)
            filling = tuple(tuple(G.idx(g) for g in cell)
                            for row in grid for cell in row)
            if filling in seen:
                continue
            seen.add(filling)
            rows = _r015_rows(G, grid, side)
            rank = gf2_rank(rows)
            if rank != expected:
                continue
            if _r035_floor(_r035_columns(rows, width), False) != 5:
                continue
            seed = _r015_cofactor(rows, len(profile[0]), G.q)
            if not seed or seed.bit_count() < 12:
                continue
            cycles = sum(
                (rows[i] & rows[j]).bit_count()
                * ((rows[i] & rows[j]).bit_count() - 1) // 2
                for i in range(len(rows)) for j in range(i))
            raw.append((seed.bit_count(), cycles, trial,
                        grid, rows, seed, rank))
        raw.sort(key=lambda r: (-r[0], r[1], r[2]))
        refined = []
        for _, cycles, trial, grid, rows, seed, rank in raw[:12]:
            word = _r015_probe(local, rows, width, 4, seed)
            refined.append((word.bit_count(), cycles, trial,
                            grid, rows, word, rank))
        refined.sort(key=lambda r: (-r[0], r[1], r[2]))
        # All retained proposals receive the same deeper witness budget.
        for _, cycles, trial, grid, rows, seed, rank in refined[:4]:
            word = _r015_probe(local, rows, width, 24, seed)
            if not word or word.bit_count() < 12:
                continue
            bank.append(dict(
                grid=grid, word=word, reading=word.bit_count(),
                rank=rank, root=root, augmentation=augmentation,
                cycles=cycles, card=card, trial=trial))
    bank.sort(key=lambda r: (
        -r["reading"], r["cycles"], r["card"], r["trial"]))
    return bank


def _r015_logical(G, a, b, hx, hz, n, local):
    """Verified nontrivial witnesses, including mixed-sector probes."""
    q, best = G.q, n + 1
    for kind, checks, stabilizers in (
            ("X", hz, hx), ("Z", hx, hz)):
        own = _r015_basis(stabilizers)
        source = b["word"] if kind == "X" else a["word"]
        seeds = []
        for block in range(3 if kind == "X" else 4):
            if kind == "X":
                word = source << (block * 4 * q)
            else:
                word, bits = 0, source
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * 4 + block) * q + h)
                    bits ^= bit
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            if not _r015_outside(word, own):
                continue
            for _ in range(2):
                previous = word
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < word.bit_count():
                        word = trial
                starting = word
                moves = sorted(
                    stabilizers,
                    key=lambda row: (starting ^ row).bit_count())[:24]
                for i, first in enumerate(moves):
                    for second in moves[:i]:
                        trial = starting ^ first ^ second
                        if trial.bit_count() < word.bit_count():
                            word = trial
                if previous == word:
                    break
            assert word and _r015_outside(word, own)
            seeds.append(word)
        if not seeds:
            return None
        word = _r015_probe(
            local, checks, n, 3, min(seeds, key=int.bit_count), own)
        if not word:
            return None
        best = min(best, word.bit_count())
    return best


def _r015_full_key(hx, hz, n):
    """Intrinsic full-Tanner invariant; equality is rejected conservatively."""
    from collections import Counter
    rows = hx + hz
    cols = _r035_columns(rows, n)
    incidence = []
    for col in cols:
        neighbors = []
        while col:
            bit = col & -col
            neighbors.append(bit.bit_length() - 1)
            col ^= bit
        incidence.append(neighbors)
    colors = [row.bit_count() for row in rows]
    history = []
    for _ in range(4):
        qfeatures = [tuple(sorted(colors[r] for r in ns))
                     for ns in incidence]
        palette = {f: i for i, f in enumerate(sorted(set(qfeatures)))}
        profiles = [[] for _ in rows]
        for ns, feature in zip(incidence, qfeatures):
            for r in ns:
                profiles[r].append(palette[feature])
        refined = [(colors[r], tuple(sorted(profiles[r])))
                   for r in range(len(rows))]
        history.append((
            tuple(sorted(Counter(qfeatures).items())),
            tuple(sorted(Counter(refined).items()))))
        palette = {f: i for i, f in enumerate(sorted(set(refined)))}
        colors = [palette[f] for f in refined]
    return (_r035_signature(hx, hz),
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            tuple(history))


from functools import lru_cache


@lru_cache(maxsize=1)
def _r015_portfolio(G, count):
    """Pure deterministic portfolio with immutable publication data."""
    assert G.q == 35
    aa, bb = _r015_bank(G, "A"), _r015_bank(G, "B")
    if not aa or not bb:
        raise ValueError("no screened seventh-root constituent bank")
    plans = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            weak = min(a["reading"], b["reading"], 14)
            deficits = (70 - a["rank"], 70 - b["rank"])
            # This optimistic ordering tests both shared and reciprocal
            # alignments. Only complete verified ranks determine k.
            potential = 70 + 2 * deficits[0] + deficits[1]
            if a["root"] is not None and b["root"] is not None:
                potential += 6
            priority = (
                potential * weak * weak, weak,
                -a["cycles"] - b["cycles"],
                int(a["card"] != b["card"]))
            plans.append((priority, ai, bi))
    plans.sort(key=lambda p: (p[0], -p[1], -p[2]), reverse=True)

    results, graph_keys = [], set()
    cap = min(9, W_MAX)
    tested = 0
    for preliminary, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        wx = max(sum(row) + sum(sb[i][j] for i in range(2))
                 for row in sa for j in range(4))
        wz = max(sum(row) + sum(sa[i][j] for i in range(2))
                 for row in sb for j in range(3))
        assert wx <= 8 and wz <= 9
        assert _no_dead_lines(sa, sb)
        if max(wx, wz) > cap:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 560 and k > 0
                and 125 * n <= 1000 * k < 175 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        if (_r035_floor(_r035_columns(hx, n), False) != 5
                or _r035_floor(_r035_columns(hz, n), False) != 5):
            continue
        signature = _r015_full_key(hx, hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        tested += 1
        if not css_ok(hx, hz):
            raise ValueError("verified seventh-root pair failed CSS")
        local = random.Random(
            BASE_SEED ^ (0x3515CC + 65537 * ai + 1009 * bi))
        reading = _r015_logical(G, a, b, hx, hz, n, local)
        if reading is None:
            continue

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        quality = (k * min(reading, 14) ** 2 / n,
                   min(reading, 14), k,
                   -a["cycles"] - b["cycles"])
        screen = (
            ("k", k), ("logical_witness_upper_bound", reading),
            ("component_witness_readings", (a["reading"], b["reading"])),
            ("component_ranks", (a["rank"], b["rank"])),
            ("seventh_root_cards", (a["root"], b["root"])),
            ("design_cards", (a["card"], b["card"])),
            ("complete_kernel_exclusion_through", 4),
        )
        results.append((quality, freeze(A), freeze(B), screen))
        if len(results) >= count + max(32, count // 2):
            break
        if tested >= max(240, 6 * count) and len(results) >= count:
            break
    results.sort(key=lambda r: r[0], reverse=True)
    return tuple((A, B, screen) for _, A, B, screen in results[:count])


def NEW_CTX(rng, G, idx):
    """Publish a distinct screened design without mutable cross-slot state."""
    portfolio = _r015_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("distinct target-rate seventh-root portfolio exhausted")
    A, B, screen = portfolio[idx]
    return {"idx": idx, "cells": {"A": A, "B": B},
            "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete screened cell without post-screen randomness."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled dense supports and uniform labels on Z17 (2,5)x(2,6) suffice for d30s>4: all five measured parent readings were 4, with sector-1 logical families spread across Acol/Bcol blocks.
# REFUTED: The parent Tanner hash-bin implementation prevents downstream isomorphic collapse: sixty evaluated slots retained forty-seven distinct codes and thirteen iso_dup slots.
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

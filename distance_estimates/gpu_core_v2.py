"""Batched GPU kernels of the directed-decoder distance estimator: min-sum BP,
GF(2) RREF with OSD, and stabilizer-coset weight descent.

Layout
  - Check-node tables are dense [C, dmax] with padding pointing at a dummy
    variable slot n whose LLR is pinned to +LARGE, so padded edges never win
    a min and never flip a sign.
  - The H table is shared across the batch (index_add_ with a shared index
    vector); the single logical row varies per batch element (scatter_add_).
  - Packed rows use torch.int32 words, 32 bits per word, LSB first.
  - The syndrome rides along as one extra column at position n, so after
    full reduction the OSD-0 solution is literally that column's bits on the
    pivot rows. It is excluded from pivot selection.
"""
import numpy as np
import torch

LARGE = 1e6
BITS = 32  # bits per packed int32 word; see module docstring


# ---------------------------------------------------------------------------
# host-side table construction
# ---------------------------------------------------------------------------

def check_table(H, pad_to=None):
    """Dense [C, dmax] variable-index table for check matrix H, padded with n.

    pad_to: optional minimum table width; the extra slots hold the dummy
    index n, like the natural per-row padding. bp_minsum masks slots equal
    to n and sends exact zeros through them, so wider padding does not change
    its output; it lets codes with different max row weights share one
    CUDA-graph shape key (see estimator_gpu_v2.search_side_gpu).
    """
    m, n = H.shape
    dmax = int(H.sum(1).max())
    if pad_to is not None:
        dmax = max(dmax, int(pad_to))
    tab = np.full((m, dmax), n, dtype=np.int64)
    for r in range(m):
        cols = np.flatnonzero(H[r])
        tab[r, :len(cols)] = cols
    return tab


def logical_table(L, pad_to=None):
    """Dense [k, Lmax] variable-index table for the logical rows, padded with n.

    pad_to: optional minimum table width, as in check_table.
    """
    k, n = L.shape
    lmax = int(L.sum(1).max())
    if pad_to is not None:
        lmax = max(lmax, int(pad_to))
    tab = np.full((k, lmax), n, dtype=np.int64)
    for r in range(k):
        cols = np.flatnonzero(L[r])
        tab[r, :len(cols)] = cols
    return tab


# ---------------------------------------------------------------------------
# batched normalized min-sum BP
# ---------------------------------------------------------------------------

def _check_update(msg, pad, sflip, alpha):
    """msg, pad: [B, C, d]. sflip: broadcastable +1/-1 (syndrome sign).
    Returns check-to-variable messages with the min taken excluding self."""
    a = torch.where(pad, torch.full_like(msg, LARGE), msg.abs())
    v2, i2 = torch.topk(a, 2, dim=-1, largest=False)          # ascending
    min1, min2 = v2[..., 0:1], v2[..., 1:2]
    arg1 = i2[..., 0:1]
    ar = torch.arange(msg.shape[-1], device=msg.device).view(1, 1, -1)
    min_excl = torch.where(ar == arg1, min2, min1)
    neg = (msg < 0) & (~pad)
    tot = neg.sum(-1, keepdim=True)
    sign_excl = 1.0 - 2.0 * ((tot - neg.long()) % 2).to(msg.dtype)
    out = alpha * sign_excl * min_excl * sflip
    return torch.where(pad, torch.zeros_like(out), out)


def bp_minsum(Htab, Ltab_b, prior, n, iters=30, alpha=0.8):
    """Batched min-sum syndrome BP for the stacked matrices [H ; L_i].

    Htab   : [C, dH] long, shared across batch (device tensor)
    Ltab_b : [B, dL] long, per-batch logical row support (device tensor)
    prior  : [B, n+1] float LLRs; slot n is the dummy and must be +LARGE
    Returns posterior LLRs [B, n+1]. Syndrome is 0 on all H rows and 1 on the
    logical row, so only the logical row gets the sign flip.
    """
    B = prior.shape[0]
    C, dH = Htab.shape
    dev = prior.device
    Hflat = Htab.reshape(-1)                                   # [C*dH]
    padH = (Htab == n).view(1, C, dH)
    padL = (Ltab_b == n).unsqueeze(1)                          # [B,1,dL]

    m_cv_H = torch.zeros(B, C, dH, device=dev)
    m_cv_L = torch.zeros(B, 1, Ltab_b.shape[1], device=dev)

    for _ in range(iters):
        tot = prior.clone()
        tot.index_add_(1, Hflat, m_cv_H.reshape(B, -1))
        tot.scatter_add_(1, Ltab_b, m_cv_L.squeeze(1))

        m_vc_H = tot[:, Hflat].view(B, C, dH) - m_cv_H
        m_vc_H = torch.where(padH, torch.full_like(m_vc_H, LARGE), m_vc_H)
        m_vc_L = tot.gather(1, Ltab_b).unsqueeze(1) - m_cv_L
        m_vc_L = torch.where(padL, torch.full_like(m_vc_L, LARGE), m_vc_L)

        m_cv_H = _check_update(m_vc_H, padH.expand(B, -1, -1), 1.0, alpha)
        m_cv_L = _check_update(m_vc_L, padL, -1.0, alpha)      # syndrome bit 1

    tot = prior.clone()
    tot.index_add_(1, Hflat, m_cv_H.reshape(B, -1))
    tot.scatter_add_(1, Ltab_b, m_cv_L.squeeze(1))
    return tot


# ---------------------------------------------------------------------------
# batched GF(2) RREF + OSD
# ---------------------------------------------------------------------------

def _pack(bits):
    """bits: [B, m, W*BITS] uint8 -> packed int32 [B, m, W].

    Accumulates one bit position at a time in int64 (bit 31 is the int32
    sign bit), then casts to int32, which keeps the low 32 bits.
    """
    B, m, tot = bits.shape
    W = tot // BITS
    x = bits.view(B, m, W, BITS)
    acc = torch.zeros(B, m, W, dtype=torch.int64, device=bits.device)
    for j in range(BITS):
        acc |= x[..., j].to(torch.int64) << j
    return acc.to(torch.int32)


def _getbit(packed, c):
    # >> on int32 is an arithmetic shift (sign-fills for words with bit 31
    # set), but the & 1 keeps only the true data bit.
    return (packed[:, :, c // BITS] >> (c % BITS)) & 1


def _rref_build(M, order, syn):
    """Permute columns by `order`, append the syndrome column, bit-pack.

    Shared by the eager and CUDA-graph paths (the only branch, `if pad:`,
    depends on static shapes, so the function is capture-safe).
    """
    B, m, n = M.shape
    dev = M.device
    Mp = torch.gather(M, 2, order.unsqueeze(1).expand(B, m, n))
    aug = torch.cat([Mp, syn.unsqueeze(2)], dim=2)             # [B, m, n+1]
    W = (n + 1 + BITS - 1) // BITS
    pad = W * BITS - (n + 1)
    if pad:
        aug = torch.cat([aug, torch.zeros(B, m, pad, dtype=torch.uint8,
                                          device=dev)], dim=2)
    return _pack(aug)


def _rref_column_loop(packed, n, ncols, check_every=64):
    """Eliminate columns 0..ncols-1 in place; returns (rank, is_pivot).

    The row swap is one gather + one index_put_ over the stacked [r; first]
    index pair; piv is the pre-swap row `first`, which is the post-swap row
    r. The elimination mask uses the pre-swap column bits with rows r and
    first zeroed: after the swap row first holds the old row r, whose bit c
    is 0 whenever first != r. pivot_col is rebuilt after the loop
    (_rref_pivot_cols): row r's pivot column is the r-th smallest is_pivot
    index.

    check_every: run the host-synced rank-saturation break every that many
    columns (0/None disables it, which leaves the loop without host syncs
    and CUDA-graph capturable, gpu_graphs.py); columns after saturation are
    no-ops, so both settings return identical results.
    """
    B, m, W = packed.shape
    dev = packed.device
    rank = torch.zeros(B, dtype=torch.long, device=dev)
    is_pivot = torch.zeros(B, n, dtype=torch.bool, device=dev)
    # loop invariants; sh32/one_w are 0-dim int32 constants, so the shift
    # and mask need no per-column scalar conversion
    bidx2 = torch.arange(B, device=dev).view(B, 1).expand(B, 2)
    rows = torch.arange(m, device=dev).view(1, m)
    zero_w = torch.zeros((), dtype=torch.int32, device=dev)     # 0-dim int32
    one_w = torch.ones((), dtype=torch.int32, device=dev)
    sh32 = [torch.full((), j, dtype=torch.int32, device=dev)
            for j in range(BITS)]

    for c in range(ncols):
        # candidate rows: bit c set, at or below the pivot row (int32 bits
        # stay int32 -- any/argmax take them directly, no bool round trip)
        col = (packed[:, :, c // BITS] >> sh32[c % BITS]) & one_w  # [B, m]
        cand = torch.where(rows >= rank.view(B, 1), col, zero_w)
        has = cand.any(1)                                       # [B] bool
        # argmax runs on uint8 on purpose: with two or more candidates all
        # maxima tie, and CUDA's tie-breaking depends on the input dtype
        first = torch.argmax(cand.to(torch.uint8), dim=1)       # first 1-row
        r = torch.clamp(rank, max=m - 1)
        idx2 = torch.stack((r, first), dim=1)                   # [B, 2]

        # row swap: ONE gather of both rows, ONE index_put_ of both rows.
        # No-pivot elements write back what they read (value no-op); a
        # duplicate r == first writes identical values into both slots.
        both = packed[bidx2, idx2]                              # [B, 2, W]
        sel3 = has.view(B, 1, 1)
        packed[bidx2, idx2] = torch.where(sel3, both.flip(1), both)

        # eliminate bit c from every other row carrying it
        piv = torch.where(has.view(B, 1), both[:, 1], zero_w)   # [B, W]
        emask = torch.where(has.view(B, 1), col, zero_w)        # [B, m]
        emask[bidx2, idx2] = zero_w                             # keep r, first
        packed ^= emask.unsqueeze(2) * piv.unsqueeze(1)

        is_pivot[:, c] = has
        rank.add_(has)

        # Rank-saturation early-out (host sync), eager paths only. Once
        # rank saturates every later column is a no-op (cand is empty), so
        # a delayed -- or absent -- break returns the identical state.
        if check_every and c % check_every == check_every - 1 \
                and bool((rank >= m).all()):
            break
    return rank, is_pivot


def _rref_pivot_cols(is_pivot, m):
    """Rebuild pivot_col [B, m] from is_pivot: pivots are assigned to rows
    0,1,2,... as columns are scanned left to right, so row r's pivot column
    is the r-th smallest True index of is_pivot (rows >= rank stay -1).
    Runs once per rref call, off the hot loop; capture-safe (sort/where)."""
    B, n = is_pivot.shape
    dev = is_pivot.device
    idx = torch.arange(n, device=dev).view(1, n).expand(B, n)
    big = torch.where(is_pivot, idx, torch.full_like(idx, n))   # n = sentinel
    srt = torch.sort(big, dim=1).values
    pivot_col = torch.full((B, m), -1, dtype=torch.long, device=dev)
    t = min(m, n)
    vals = srt[:, :t]
    pivot_col[:, :t] = torch.where(vals < n, vals,
                                   torch.full_like(vals, -1))
    return pivot_col


def batched_rref(M, order, syn, ncols, check_every=64):
    """Column-permuted full RREF over GF(2), batched.

    M     : [B, m, n] uint8 device tensor (the stacked check matrix per batch)
    order : [B, n] long, column priority (most-likely-error first)
    syn   : [B, m] uint8 syndrome
    Returns (packed, rank, pivot_col, is_pivot) where packed holds the reduced
    matrix with the syndrome as column `ncols`, pivot_col[b, r] is the permuted
    column index of pivot row r, and is_pivot[b, c] marks pivot columns.

    Split into build + column loop + pivot_col reconstruction so the eager
    and CUDA-graph paths (gpu_graphs.py, check_every=0) share the same code.
    """
    B, m, n = M.shape
    packed = _rref_build(M, order, syn)
    rank, is_pivot = _rref_column_loop(packed, n, ncols,
                                       check_every=check_every)
    pivot_col = _rref_pivot_cols(is_pivot, m)
    return packed, rank, pivot_col, is_pivot


def osd_solutions(packed, rank, pivot_col, is_pivot, order, n, m,
                  lam=24, pairs=True):
    """OSD-0 plus a combination sweep over the earliest non-pivot columns.

    Returns (best_weight [B], best_vec [B, n] uint8) in ORIGINAL coordinates.

    When a batch element has fewer than `lam` free (non-pivot) columns, the
    padded candidate slots get weight BIGW in both the single- and pair-flip
    searches and are never selected. All best-so-far updates are masked
    torch.where ops, so there are no host syncs and the pair loop runs a
    fixed lam-1 iterations.
    """
    B = packed.shape[0]
    dev = packed.device
    rows = torch.arange(m, device=dev).view(1, m)
    live = (rows < rank.view(B, 1))                             # pivot rows

    x0 = (_getbit(packed, n).to(torch.uint8)) * live.to(torch.uint8)  # [B, m]
    best_w = x0.sum(1).to(torch.long)
    best_x = x0.clone()
    best_flip = torch.full((B, 2), -1, dtype=torch.long, device=dev)

    # earliest non-pivot columns in permuted order = least reliable free bits
    free = (~is_pivot)
    idx = torch.arange(n, device=dev).view(1, n).expand(B, n)
    big = torch.where(free, idx, torch.full_like(idx, n + 1))
    cand_raw = torch.sort(big, dim=1).values[:, :lam]           # [B, lam']
    valid = cand_raw < n                                        # padded slots
    cand = torch.clamp(cand_raw, max=n - 1)
    lam_eff = cand.shape[1]
    BIGW = n + m + 1                                # unreachable flip weight

    cols = []
    for j in range(lam_eff):
        cj = cand[:, j]                                         # [B]
        word = cj // BITS
        bit = cj % BITS
        v = (torch.gather(packed, 2, word.view(B, 1, 1).expand(B, m, 1)
                          ).squeeze(2) >> bit.view(B, 1)) & 1
        cols.append((v.to(torch.uint8) * live.to(torch.uint8)))
    A = torch.stack(cols, dim=2)                                # [B, m, lam']

    # single flips
    w1 = (x0.unsqueeze(2) ^ A).sum(1).to(torch.long) + 1        # [B, lam']
    w1 = torch.where(valid, w1, torch.full_like(w1, BIGW))
    v1, i1 = w1.min(1)
    upd = v1 < best_w
    sel = torch.gather(A, 2, i1.view(B, 1, 1).expand(B, m, 1)).squeeze(2)
    best_x = torch.where(upd.view(B, 1), x0 ^ sel, best_x)
    best_flip[:, 0] = torch.where(upd, torch.gather(cand, 1, i1.view(B, 1)
                                                    ).squeeze(1),
                                  best_flip[:, 0])
    best_flip[:, 1] = torch.where(upd, torch.full_like(best_flip[:, 1], -1),
                                  best_flip[:, 1])
    best_w = torch.minimum(best_w, v1)

    # pair flips (chunked over the first index to bound memory); fixed
    # lam'-1 iterations, updates masked per element, no host branch
    if pairs:
        for a in range(lam_eff - 1):
            xa = x0 ^ A[:, :, a]                                # [B, m]
            wp = (xa.unsqueeze(2) ^ A[:, :, a + 1:]).sum(1).to(torch.long) + 2
            ok = valid[:, a:a + 1] & valid[:, a + 1:]
            wp = torch.where(ok, wp, torch.full_like(wp, BIGW))
            vp, ip = wp.min(1)
            upd = vp < best_w
            selb = torch.gather(A[:, :, a + 1:], 2,
                                ip.view(B, 1, 1).expand(B, m, 1)).squeeze(2)
            best_x = torch.where(upd.view(B, 1), xa ^ selb, best_x)
            best_flip[:, 0] = torch.where(upd, cand[:, a], best_flip[:, 0])
            best_flip[:, 1] = torch.where(
                upd, torch.gather(cand, 1, (ip + a + 1).view(B, 1)).squeeze(1),
                best_flip[:, 1])
            best_w = torch.minimum(best_w, vp)

    # scatter back to original coordinates
    V = torch.zeros(B, n + 1, dtype=torch.uint8, device=dev)
    pc = torch.clamp(pivot_col, min=0)
    origin = torch.gather(order, 1, pc)                         # [B, m]
    origin = torch.where(pivot_col >= 0, origin,
                         torch.full_like(origin, n))
    V.scatter_(1, origin, best_x)
    for t in range(2):
        f = best_flip[:, t]
        ok = f >= 0
        oc = torch.gather(order, 1, torch.clamp(f, min=0).view(B, 1)).squeeze(1)
        oc = torch.where(ok, oc, torch.full_like(oc, n))
        V.scatter_(1, oc.view(B, 1), torch.ones(B, 1, dtype=torch.uint8,
                                                device=dev))
    V = V[:, :n]
    return V.sum(1).to(torch.long), V


# ---------------------------------------------------------------------------
# batched stabilizer-coset weight descent
# ---------------------------------------------------------------------------
#
# The 0/1 matmuls run in fp16 on CUDA when n <= 2048: every partial sum is an
# integer <= n, which fp16 represents exactly. Outputs are cast to fp32 at
# once; the delta arithmetic is always done in fp32.

def _mm_01(a, bt_native, bt_half):
    """a @ bt, exact: fp16 inputs when a half operand was prepared."""
    if bt_half is not None:
        return (a.to(torch.float16) @ bt_half).to(torch.float32)
    return a @ bt_native


def gpu_descent(V, S, iters=40):
    """V: [B, n] uint8 logical vectors; S: [ms, n] float stabilizer rows.
    Greedy single-row XOR descent; stays inside the logical coset."""
    Vf = V.to(torch.float32)
    n = Vf.shape[1]
    use_half = Vf.is_cuda and n <= 2048
    St = S.t()
    St_h = St.to(torch.float16) if use_half else None
    Srw = S.sum(1).view(1, -1)
    for _ in range(iters):
        ov = _mm_01(Vf, St, St_h)
        delta = Srw - 2.0 * ov
        best, idx = delta.min(1)
        upd = best < -0.5
        if not bool(upd.any()):
            break
        Vf = torch.where(upd.view(-1, 1), torch.remainder(Vf + S[idx], 2.0), Vf)
    return Vf.to(torch.uint8)


def gpu_descent_pairs(V, S, G=None, rounds=6, inner=40, chunk=None):
    """Descent with single and pair stabilizer moves, fully batched.

    Weight change for XOR-ing rows a and c into v, in closed form:

        |v ^ Sa ^ Sc| - |v| = (|Sa| + |Sc| - 2 G_ac)
                              - 2 (<v,Sa> + <v,Sc> - 2 T_ac)
        where G = S S^T   (batch-independent, precomputed)
              T_ac = <v, Sa AND Sc> = (S diag(v) S^T)_ac

    T is one batched matmul, so all ms^2 pair moves are scored at once. The
    inner single-move loop runs `inner` fixed iterations, and the chunk loop
    accumulates a device-side `moved` flag that is read once per round.
    """
    dev = V.device
    Vf = V.to(torch.float32)
    B, n = Vf.shape
    ms = S.shape[0]
    Sw = S.sum(1)
    if G is None:
        G = S @ S.t()
    eye = torch.eye(ms, device=dev, dtype=torch.bool)
    if chunk is None:
        chunk = max(1, int(6e7 // max(1, ms * ms)))
    use_half = Vf.is_cuda and n <= 2048
    St = S.t()
    St_h = St.to(torch.float16) if use_half else None
    S_h = S.to(torch.float16) if use_half else None
    base = (Sw.view(1, -1, 1) + Sw.view(1, 1, -1) - 2.0 * G.unsqueeze(0))

    def single(Vf):
        for _ in range(inner):
            ov = _mm_01(Vf, St, St_h)
            delta = Sw.view(1, -1) - 2.0 * ov
            best, idx = delta.min(1)
            upd = best < -0.5
            Vf = torch.where(upd.view(-1, 1),
                             torch.remainder(Vf + S[idx], 2.0), Vf)
        return Vf

    Vf = single(Vf)
    for _ in range(rounds):
        moved = torch.zeros((), dtype=torch.bool, device=dev)
        for s in range(0, B, chunk):
            vb = Vf[s:s + chunk]
            b = vb.shape[0]
            if use_half:
                vh = vb.to(torch.float16)
                ov = (vh @ St_h).to(torch.float32)              # [b, ms]
                T = torch.bmm(S_h.unsqueeze(0) * vh.unsqueeze(1),
                              St_h.unsqueeze(0).expand(b, -1, -1)
                              ).to(torch.float32)               # [b, ms, ms]
            else:
                ov = vb @ St                                    # [b, ms]
                T = torch.bmm(S.unsqueeze(0) * vb.unsqueeze(1),
                              St.unsqueeze(0).expand(b, -1, -1))  # [b, ms, ms]
            d2 = base - 2.0 * (ov.unsqueeze(2) + ov.unsqueeze(1) - 2.0 * T)
            d2 = d2.masked_fill(eye.unsqueeze(0), 0.0)
            flat = d2.view(b, -1)
            best, pos = flat.min(1)
            upd = best < -0.5
            moved = moved | upd.any()
            ia, ic = pos // ms, pos % ms
            add = torch.remainder(S[ia] + S[ic], 2.0)
            Vf[s:s + chunk] = torch.where(
                upd.view(-1, 1), torch.remainder(vb + add, 2.0), vb)
        if not bool(moved):                # the ONE host sync per round
            break
        Vf = single(Vf)
    return Vf.to(torch.uint8)

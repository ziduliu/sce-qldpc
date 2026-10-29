"""M4RI-style blocked (panel) GF(2) RREF, a drop-in for gpu_core_v2.batched_rref.

Pivots are selected column by column on a private copy of one 32-bit word
column; the elimination of the whole panel is then applied to the full packed
matrix in one pass as a 0/1 matmul (fp16 on CUDA, fp32 on CPU; counts never
exceed 32, so both are exact). For a non-pivot row with original content x0
and bits b_p at the panel's pivot columns, the result is
x0 XOR sum_p b_p * P_p, with P_p the fully reduced pivot rows.

Outputs (reduced matrix including the syndrome column, rank, pivot_col,
is_pivot) are bit-identical to batched_rref; packed words are torch.int32,
32 bits per word, LSB first, syndrome at bit position n.
"""
import torch

BITS = 32  # payload bits per packed int32 word

# Byte budget for the unpacked 0/1 float temporaries of one apply GEMM.
# Rows are processed in chunks that keep B * chunk * (W*BITS) under this
# budget.
MAX_APPLY_BYTES = 256 << 20


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
    """Bit c of int32-packed rows. Right shift on negative int32 is
    arithmetic, but & 1 keeps only the wanted bit, so the sign extension
    is harmless (bit 31 included)."""
    return (packed[:, :, c // BITS] >> (c % BITS)) & 1


def batched_rref_panel(M, order, syn, ncols, panel=32):
    """Column-permuted full RREF over GF(2), batched and panel-blocked.

    M     : [B, m, n] uint8 device tensor (stacked check matrix per batch)
    order : [B, n] long, column priority (most-likely-error first)
    syn   : [B, m] uint8 syndrome
    Returns (packed, rank, pivot_col, is_pivot) with the same values as
    gpu_core_v2.batched_rref; packed is int32 [B, m, W]. `panel` must
    divide 32 so a panel never straddles a packed word.
    """
    if panel < 1 or BITS % panel:
        raise ValueError("panel must divide %d" % BITS)
    B, m, n = M.shape
    dev = M.device
    Mp = torch.gather(M, 2, order.unsqueeze(1).expand(B, m, n))
    aug = torch.cat([Mp, syn.unsqueeze(2)], dim=2)              # [B, m, n+1]
    W = (n + 1 + BITS - 1) // BITS
    pad = W * BITS - (n + 1)
    if pad:
        aug = torch.cat([aug, torch.zeros(B, m, pad, dtype=torch.uint8,
                                          device=dev)], dim=2)
    packed = _pack(aug)                                          # [B, m, W]
    # One all-zero dummy row at index m: scatter target for panel slots
    # that found no pivot, so step 4 never needs data-dependent boolean
    # indexing (which would host-sync). It is invisible to pivot selection
    # (row masks stop at m) and stays zero forever: its combination row is
    # zero and only zeros are ever scattered into it. Sliced off on return.
    packed = torch.cat([packed, torch.zeros(B, 1, W, dtype=torch.int32,
                                            device=dev)], dim=1)

    rank = torch.zeros(B, dtype=torch.long, device=dev)
    pivot_col = torch.full((B, m), -1, dtype=torch.long, device=dev)
    is_pivot = torch.zeros(B, n, dtype=torch.bool, device=dev)
    bidx = torch.arange(B, device=dev)
    rows = torch.arange(m, device=dev).view(1, m)
    sh = torch.arange(BITS, device=dev, dtype=torch.int64)
    # fp16 GEMM is exact here (counts <= 32, well under 2048), but torch
    # 2.1 CPU half matmul is unreliable, so gate it on CUDA.
    fdt = torch.float16 if packed.is_cuda else torch.float32

    for c0 in range(0, ncols, panel):
        # The one host sync per panel: rank-saturation early exit
        # (saturated batches yield no further pivots or eliminations).
        if bool((rank >= m).all()):
            break
        w = c0 // BITS
        nb = min(panel, ncols - c0)       # pivot-eligible bits this panel

        # ---- step 1: pivot selection + elimination on word w only ------
        # `packed` gets the row swaps but keeps its pre-elimination bits.
        wcol = packed[:, :m, w].clone()                          # [B, m]
        prow = torch.full((B, panel), -1, dtype=torch.long, device=dev)
        for j in range(nb):
            bit = (c0 + j) % BITS
            col = ((wcol >> bit) & 1).bool()
            cand = col & (rows >= rank.view(B, 1)) & (rank < m).view(B, 1)
            has = cand.any(1)
            first = torch.argmax(cand.to(torch.uint8), dim=1)
            r = torch.clamp(rank, max=m - 1)
            sel = has.view(B, 1)

            # swap rows r and first where this column yields a pivot, in
            # the word copy and in the full packed rows ([B, W], cheap)
            wr = wcol[bidx, r].clone()
            wf = wcol[bidx, first].clone()
            wcol[bidx, r] = torch.where(has, wf, wr)
            wcol[bidx, first] = torch.where(has, wr, wf)
            rowr = packed[bidx, r].clone()
            rowf = packed[bidx, first].clone()
            packed[bidx, r] = torch.where(sel, rowf, rowr)
            packed[bidx, first] = torch.where(sel, rowr, rowf)

            # eliminate this bit from other rows in the word copy ONLY
            pivw = wcol[bidx, r]                                 # [B]
            colb = ((wcol >> bit) & 1).bool() & sel
            colb[bidx, r] = False                    # keep pivot row
            wcol = wcol ^ (colb.to(torch.int32) * pivw.view(B, 1))

            prow[:, j] = torch.where(has, r, torch.full_like(r, -1))
            pivot_col[bidx, r] = torch.where(
                has, torch.full_like(r, c0 + j), pivot_col[bidx, r])
            is_pivot[:, c0 + j] = has
            rank = rank + has.long()

        # ---- step 2: Gauss-Jordan among the pivot rows on full words ---
        # Gathered rows are the pivot rows' original contents (eliminations
        # were confined to wcol); visiting the slots in column order leaves
        # them fully reduced.
        valid = prow >= 0                                        # [B, panel]
        gidx = torch.clamp(prow, min=0).unsqueeze(-1).expand(B, panel, W)
        P0 = torch.gather(packed, 1, gidx)                     # [B, panel, W]
        P0 = torch.where(valid.unsqueeze(-1), P0, torch.zeros_like(P0))
        for j in range(nb):
            bit = (c0 + j) % BITS
            tb = ((P0[:, :, w] >> bit) & 1).clone()              # [B, panel]
            tb[:, j] = 0                              # keep own row
            # invalid slot j holds a zero row, so this XOR is a no-op there
            P0 = P0 ^ (tb.unsqueeze(-1) * P0[:, j:j + 1, :])

        # ---- step 3: combination matrix from PRE-elimination bits ------
        # C[b, i, p] = row i's original bit at pivot column c_p; reduced
        # pivot rows turn that into the exact sequential net update.
        shifts = (c0 % BITS) + torch.arange(nb, device=dev)
        colw = packed[:, :, w]                                   # [B, m+1]
        C = (colw.unsqueeze(-1) >> shifts.view(1, 1, nb)) & 1
        C = C * valid[:, :nb].to(C.dtype).view(B, 1, nb)
        scat = torch.where(valid, prow, torch.full_like(prow, m))
        rowsel = torch.zeros(B, m + 1, dtype=torch.bool, device=dev)
        rowsel.scatter_(1, scat, True)
        C = C * (~rowsel).to(C.dtype).unsqueeze(-1)  # pivot rows: no XOR

        # ---- step 4: one-pass apply, GF(2) combine as a matmul ---------
        wb = W * BITS
        pbits = (P0[:, :nb].to(torch.int64).unsqueeze(-1) >> sh) & 1
        Pf = pbits.view(B, nb, wb).to(fdt)
        Cf = C.to(fdt)
        mc = max(1, int(MAX_APPLY_BYTES // max(1, B * wb * 8)))
        for s in range(0, m + 1, mc):
            e = min(m + 1, s + mc)
            counts = Cf[:, s:e] @ Pf              # integer-exact, <= 32
            ub = counts.to(torch.int64) & 1
            upd = (ub.view(B, e - s, W, BITS) << sh).sum(-1).to(torch.int32)
            packed[:, s:e] ^= upd
        # scatter the reduced pivot rows into their slots; invalid slots
        # write zeros into the dummy row m (duplicates all carry zeros,
        # so scatter's duplicate-index freedom cannot change the result)
        src = torch.where(valid.unsqueeze(-1), P0, torch.zeros_like(P0))
        packed.scatter_(1, scat.unsqueeze(-1).expand(B, panel, W), src)
        # No separate fix-up for word w: the closed-form update plus the
        # scatter leave it bit-identical to wcol's final state.

    return packed[:, :m].contiguous(), rank, pivot_col, is_pivot

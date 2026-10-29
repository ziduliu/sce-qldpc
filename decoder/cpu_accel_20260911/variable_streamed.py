"""Ordered F32 variable updates with looped, bounded-size Triton code.

Prefixes are stored in Q during the forward pass and read back during the
reverse pass. This preserves the Rust summation order without keeping a
separate register tensor for every edge. Launch with enable_fp_fusion=False.
"""
import triton as tr
import triton.language as tl


@tr.jit
def variable_streamed(Q, CV, POST, PRIOR, PTR, MAP, VARS, DONE, HARD, UNSAFE,
                      NV: tl.constexpr, NB, D: tl.constexpr,
                      BS: tl.constexpr, VS: tl.constexpr,
                      G: tl.constexpr, IG: tl.constexpr):
    vi = tl.program_id(0) * VS + tl.arange(0, VS)
    v = tl.load(VARS + vi, vi < NV, other=0)
    b = tl.program_id(1) * BS + tl.arange(0, BS)
    live = ((vi[:, None] < NV) & (b[None, :] < NB)
            & (tl.load(DONE + b, b < NB, other=1)[None, :] == 0))
    start = tl.load(PTR + v)
    end = tl.load(PTR + v + 1)
    prior = tl.load(PRIOR + v)[:, None]
    old = tl.load(POST + v[:, None] * NB + b[None, :], live, other=0.)
    acc = prior * IG + old * G
    limit: tl.constexpr = 3.4028234663852886e38
    acc = tl.where(prior == limit, prior, acc)
    nf = live & (~(tl.abs(acc) <= limit) | ~(tl.abs(old) <= limit))

    # Deliberately use ordinary loops, not tl.static_range: the compiled
    # instruction count should not grow linearly with the degree bound D.
    for j in range(D):
        idx = start + j
        m = live & (idx[:, None] < end[:, None])
        edge = tl.load(MAP + idx, (vi < NV) & (idx < end), other=0)
        cv = tl.load(CV + edge[:, None] * NB + b[None, :], m, other=0.)
        tl.store(Q + edge[:, None] * NB + b[None, :], acc, m)
        acc = tl.where(m, acc + cv, acc)
        nf |= m & (~(tl.abs(cv) <= limit) | ~(tl.abs(acc) <= limit))

    tl.store(POST + v[:, None] * NB + b[None, :], acc, live)
    tl.store(HARD + v[:, None] * NB + b[None, :], acc <= 0., live)
    suffix = tl.full((VS, BS), 0., tl.float32)
    for jj in range(D):
        idx = start + D - 1 - jj
        m = live & (idx[:, None] < end[:, None])
        edge = tl.load(MAP + idx, (vi < NV) & (idx < end), other=0)
        cv = tl.load(CV + edge[:, None] * NB + b[None, :], m, other=0.)
        prefix = tl.load(Q + edge[:, None] * NB + b[None, :], m, other=0.)
        outgoing = prefix + suffix
        tl.store(Q + edge[:, None] * NB + b[None, :], outgoing, m)
        suffix = tl.where(m, suffix + cv, suffix)
        nf |= m & (~(tl.abs(outgoing) <= limit) | ~(tl.abs(suffix) <= limit))

    tl.atomic_or(UNSAFE + b, 1,
                 mask=(b < NB) & (tl.sum(nf.to(tl.int32), 0) > 0),
                 sem="relaxed")

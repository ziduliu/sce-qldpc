"""CUDA-graph wrappers for the batched kernels of gpu_core_v2 (CUDA only at call time).

GraphedRREF, GraphedBP and GraphedOSD capture the rref build and column loop
(check_every=0), bp_minsum and osd_solutions once per input shape and replay
them. Every call copies the inputs into the graph's static buffers; the
returned tensors live in static buffers and are valid only until the next call
with the same shape. `pool` is passed to torch.cuda.graph (None: a private
pool per graph). Importing the module does not touch CUDA.
"""
import torch

import gpu_core_v2 as core

WARMUP_ITERS = 3


class _Graphed:
    """Shape-keyed CUDA-graph cache around one branch-free tensor function.

    Subclasses implement _run(*tensors, *args) -- the exact eager body --
    and __call__, which builds the cache key and forwards to _call. First
    call for a key: clone the inputs as static buffers, warm up eagerly on
    a side stream (cuDNN/cuBLAS workspaces, allocator priming), capture one
    run into a torch.cuda.CUDAGraph. Every call (first included): copy the
    inputs into the static buffers, replay, return the static outputs.
    """

    def __init__(self, pool=None, warmup=WARMUP_ITERS):
        self.pool = pool
        self.warmup = warmup
        self._cache = {}

    def _run(self, *a):
        raise NotImplementedError

    def _capture(self, tensors, args):
        statics = tuple(t.clone() for t in tensors)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.warmup):
                self._run(*statics, *args)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            outs = self._run(*statics, *args)
        return g, statics, outs

    def _call(self, key, tensors, args=()):
        if not tensors[0].is_cuda:
            raise RuntimeError(
                "%s requires CUDA tensors; route CPU work through the eager "
                "path (estimator_gpu_v2 forces use_graphs=False on cpu)"
                % type(self).__name__)
        ent = self._cache.get(key)
        if ent is None:
            ent = self._capture(tensors, args)
            self._cache[key] = ent
        g, statics, outs = ent
        for dst, src in zip(statics, tensors):
            dst.copy_(src)
        g.replay()
        return outs


class GraphedRREF(_Graphed):
    """Graphed gpu_core_v2 rref: build + column loop (check_every=0) only.

    _rref_pivot_cols (a torch.sort) runs eagerly after each replay, outside
    the graph. Returns (packed, rank, pivot_col, is_pivot); packed, rank and
    is_pivot live in static buffers valid until the next call with the same
    (shape, ncols) key.
    """

    def _run(self, M, order, syn, ncols):
        packed = core._rref_build(M, order, syn)
        rank, is_pivot = core._rref_column_loop(packed, M.shape[2], ncols,
                                                check_every=0)
        return packed, rank, is_pivot

    def __call__(self, M, order, syn, ncols):
        key = (tuple(M.shape), int(ncols))
        packed, rank, is_pivot = self._call(key, (M, order, syn),
                                            (int(ncols),))
        pivot_col = core._rref_pivot_cols(is_pivot, M.shape[1])
        return packed, rank, pivot_col, is_pivot


class GraphedBP(_Graphed):
    """Graphed gpu_core_v2.bp_minsum (fixed `iters`, no host branches)."""

    def _run(self, Htab, Ltab_b, prior, n, iters, alpha):
        return core.bp_minsum(Htab, Ltab_b, prior, n, iters=iters,
                              alpha=alpha)

    def __call__(self, Htab, Ltab_b, prior, n, iters=30, alpha=0.8):
        key = (tuple(Htab.shape), tuple(Ltab_b.shape), tuple(prior.shape),
               int(n), int(iters), float(alpha))
        return self._call(key, (Htab, Ltab_b, prior),
                          (int(n), int(iters), float(alpha)))


class GraphedOSD(_Graphed):
    """Graphed gpu_core_v2.osd_solutions (fixed lam_eff-1 pair iterations;
    the empty-wc guard lives in the caller's harvest, outside the graph)."""

    def _run(self, packed, rank, pivot_col, is_pivot, order, n, m, lam,
             pairs):
        return core.osd_solutions(packed, rank, pivot_col, is_pivot, order,
                                  n, m, lam=lam, pairs=pairs)

    def __call__(self, packed, rank, pivot_col, is_pivot, order, n, m,
                 lam=24, pairs=True):
        key = (tuple(packed.shape), tuple(order.shape), int(n), int(m),
               int(lam), bool(pairs))
        return self._call(key, (packed, rank, pivot_col, is_pivot, order),
                          (int(n), int(m), int(lam), bool(pairs)))

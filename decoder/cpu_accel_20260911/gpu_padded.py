"""Experimental first-leg implementation with aligned shot strides.

Arithmetic and convergence checks match gpu_prepass.ExactFirstLeg. Physical
shot strides are multiples of 32; extra columns remain permanently inactive.
This module is independent of the validated implementation.
"""
import numpy as np
import torch
import triton as tr
import triton.language as tl

from gpu_prepass import ExactFirstLeg as _BaseFirstLeg, _check, _syndrome, _logical
from screen_contract import binary_array, detectors_fingerprint, make_screen_protocol


@tr.jit
def _variable_padded(Q, CV, POST, PRIOR, PTR, MAP, VARS, DONE, HARD, UNSAFE,
                     NV: tl.constexpr, NB, D: tl.constexpr,
                     BS: tl.constexpr, VS: tl.constexpr,
                     G: tl.constexpr, IG: tl.constexpr):
    stride = tl.multiple_of(NB, 32)
    vi = tl.program_id(0) * VS + tl.arange(0, VS)
    v = tl.load(VARS + vi, vi < NV, other=0)
    b = tl.program_id(1) * BS + tl.arange(0, BS)
    live = ((vi[:, None] < NV) & (b[None, :] < stride)
            & (tl.load(DONE + b, b < stride, other=1)[None, :] == 0))
    start = tl.load(PTR + v)
    end = tl.load(PTR + v + 1)
    prior = tl.load(PRIOR + v)[:, None]
    old = tl.load(POST + v[:, None] * stride + b[None, :], live, other=0.)
    acc = prior * IG + old * G
    limit: tl.constexpr = 3.4028234663852886e38
    acc = tl.where(prior == limit, prior, acc)
    nf = live & (~(tl.abs(acc) <= limit) | ~(tl.abs(old) <= limit))
    for j in range(D):
        idx = start + j
        m = live & (idx[:, None] < end[:, None])
        edge = tl.load(MAP + idx, (vi < NV) & (idx < end), other=0)
        cv = tl.load(CV + edge[:, None] * stride + b[None, :], m, other=0.)
        tl.store(Q + edge[:, None] * stride + b[None, :], acc, m)
        acc = tl.where(m, acc + cv, acc)
        nf |= m & (~(tl.abs(cv) <= limit) | ~(tl.abs(acc) <= limit))
    tl.store(POST + v[:, None] * stride + b[None, :], acc, live)
    tl.store(HARD + v[:, None] * stride + b[None, :], acc <= 0., live)
    suffix = tl.full((VS, BS), 0., tl.float32)
    for jj in range(D):
        idx = start + D - 1 - jj
        m = live & (idx[:, None] < end[:, None])
        edge = tl.load(MAP + idx, (vi < NV) & (idx < end), other=0)
        cv = tl.load(CV + edge[:, None] * stride + b[None, :], m, other=0.)
        prefix = tl.load(Q + edge[:, None] * stride + b[None, :], m, other=0.)
        outgoing = prefix + suffix
        tl.store(Q + edge[:, None] * stride + b[None, :], outgoing, m)
        suffix = tl.where(m, suffix + cv, suffix)
        nf |= m & (~(tl.abs(outgoing) <= limit) | ~(tl.abs(suffix) <= limit))
    tl.atomic_or(UNSAFE + b, 1,
                 mask=(b < stride) & (tl.sum(nf.to(tl.int32), 0) > 0), sem='relaxed')


@tr.jit
def _gather_padded(SRC, DST, KEEP, ROWS, COUNT, OLD_STRIDE, NEW_STRIDE,
                   BLOCK: tl.constexpr):
    old_stride = tl.multiple_of(OLD_STRIDE, 32)
    new_stride = tl.multiple_of(NEW_STRIDE, 32)
    flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = flat // new_stride
    col = flat % new_stride
    live = (row < ROWS) & (col < COUNT)
    source_col = tl.load(KEEP + col, col < COUNT, other=0)
    value = tl.load(SRC + row.to(tl.int64) * old_stride + source_col, live, other=0)
    # Padding is explicitly initialized, though inactive kernels never use it.
    tl.store(DST + flat.to(tl.int64), value, row < ROWS)


def _padded_columns(source, keep, stride):
    rows = source.shape[0]
    result = torch.empty((rows, stride), dtype=source.dtype, device=source.device)
    if rows:
        _gather_padded[(tr.cdiv(rows * stride, 1024),)](
            source, result, keep, rows, len(keep), source.shape[1], stride, 1024,
            num_warps=4)
    return result


class ExactFirstLeg(_BaseFirstLeg):
    def __init__(self, H, pri, gamma0=.1, device='cuda', L=None,
                 check_warps=4, variable_warps=8, syndrome_warps=4,
                 variable_bs=32, variable_vs=16):
        super().__init__(H, pri, gamma0=gamma0, device=device, L=L,
                         check_warps=check_warps, variable_warps=variable_warps,
                         syndrome_warps=syndrome_warps)
        if variable_bs not in (16, 32) or variable_vs not in (0, 8, 16, 32):
            raise ValueError('variable_bs must be 16/32 and variable_vs must be 0/8/16/32')
        self.variable_bs = variable_bs
        self.variable_vs = variable_vs

    def variable_kernel(self):
        return _variable_padded

    def decode(self, det, pre_iter=200, check_every=8, return_details=True):
        protocol = make_screen_protocol(self.model_id, pre_iter=pre_iter, gamma0=self.gamma0)
        if not isinstance(check_every, int) or check_every < 1:
            raise ValueError('check_every must be a positive integer')
        det = binary_array(det, 'detectors')
        if det.ndim != 2 or det.shape[1] != self.C:
            raise ValueError('Invalid detectors')
        B = len(det)
        original_B = B
        if not B:
            raise ValueError('Empty batch')
        if not return_details and self.logical is None:
            raise ValueError('Logical matrix is required for compact output')
        stride = tr.cdiv(B, 32) * 32
        syn = torch.zeros((self.C, stride), dtype=torch.uint8, device=self.dev)
        syn[:, :B] = torch.tensor(det.T.copy(), device=self.dev)
        q = self.prior[self.ind.long(), None].expand(-1, stride).contiguous()
        cv = torch.empty_like(q)
        post = self.prior[:, None].expand(-1, stride).contiguous()
        hard = torch.zeros((self.N, stride), dtype=torch.uint8, device=self.dev)
        done = torch.ones(stride, dtype=torch.bool, device=self.dev)
        done[:B] = False
        bad = torch.empty((self.C, stride), dtype=torch.bool, device=self.dev)
        iterations = torch.zeros(stride, dtype=torch.int32, device=self.dev)
        unsafe = torch.zeros(stride, dtype=torch.int32, device=self.dev)
        ids = torch.arange(B, device=self.dev)
        out_hard = torch.zeros((self.N, B), dtype=torch.uint8, device=self.dev)
        out_post = self.prior[:, None].expand(-1, B).clone() if return_details else None
        out_success = torch.zeros(B, dtype=torch.bool, device=self.dev)
        out_unsafe = torch.zeros(B, dtype=torch.bool, device=self.dev)
        out_iterations = torch.full((B,), pre_iter, dtype=torch.int32, device=self.dev)
        check_bs = 16
        variable_kernel = self.variable_kernel()
        for it in range(pre_iter):
            for k, rows in self.rows:
                _check[(len(rows), tr.cdiv(stride, check_bs))](
                    q, cv, self.ptr, rows, syn, done, stride, len(rows), k, check_bs,
                    num_warps=self.check_warps, enable_fp_fusion=False)
            for degree, variables in self.variables:
                vs = self.variable_vs or (32 if degree <= 16 else 16 if degree <= 32 else 8)
                variable_kernel[(tr.cdiv(len(variables), vs),
                                  tr.cdiv(stride, self.variable_bs))](
                    q, cv, post, self.prior, self.vptr, self.mapping, variables, done,
                    hard, unsafe, len(variables), stride, degree,
                    self.variable_bs, vs,
                    float(self.gamma), float(self.igamma),
                    num_warps=self.variable_warps, enable_fp_fusion=False)
            for k, rows in self.rows:
                _syndrome[(len(rows), tr.cdiv(stride, check_bs))](
                    hard, self.ptr, self.ind, rows, syn, bad, done, stride, k, check_bs,
                    num_warps=self.syndrome_warps, enable_fp_fusion=False)
            ok = ~bad.any(dim=0)
            iterations.masked_fill_(ok & ~done, it + 1)
            done |= ok
            if (it + 1) % check_every == 0 or it + 1 == pre_iter:
                good = done[:B].nonzero().flatten()
                if good.numel():
                    dst = ids.index_select(0, good)
                    out_hard.index_copy_(1, dst, hard.index_select(1, good))
                    if return_details:
                        out_post.index_copy_(1, dst, post.index_select(1, good))
                    out_iterations.index_copy_(0, dst, iterations.index_select(0, good))
                    out_success[dst] = unsafe.index_select(0, good) == 0
                    out_unsafe[dst] = unsafe.index_select(0, good) != 0
                    remain = (~done[:B]).nonzero().flatten()
                    ids = ids.index_select(0, remain)
                    if not ids.numel():
                        break
                    next_B = int(ids.numel())
                    next_stride = tr.cdiv(next_B, 32) * 32
                    q = _padded_columns(q, remain, next_stride)
                    post = _padded_columns(post, remain, next_stride)
                    hard = _padded_columns(hard, remain, next_stride)
                    syn = _padded_columns(syn, remain, next_stride)
                    # Check messages are overwritten before their next read.
                    # Reuse their storage without copying obsolete messages.
                    cv = cv.reshape(-1)[:q.numel()].view_as(q)
                    next_unsafe = torch.zeros(next_stride, dtype=torch.int32, device=self.dev)
                    next_unsafe[:next_B] = unsafe.index_select(0, remain)
                    unsafe = next_unsafe
                    B, stride = next_B, next_stride
                    done = torch.ones(stride, dtype=torch.bool, device=self.dev)
                    done[:B] = False
                    bad = torch.empty((self.C, stride), dtype=torch.bool, device=self.dev)
                    iterations = torch.zeros(stride, dtype=torch.int32, device=self.dev)
        if ids.numel():
            out_hard.index_copy_(1, ids, hard[:, :B])
            if return_details:
                out_post.index_copy_(1, ids, post[:, :B])
            out_unsafe[ids] = unsafe[:B] != 0
        result = dict(success=out_success.cpu().numpy(), iterations=out_iterations.cpu().numpy(),
                      unsafe_mask=out_unsafe.cpu().numpy(), protocol=protocol,
                      detectors_fingerprint=detectors_fingerprint(det))
        if return_details:
            result.update(decoding=out_hard.T.cpu().numpy(), posterior=out_post.T.cpu().numpy())
        if self.logical is not None:
            count, indices, rows, starts, ends = self.logical
            predictions = torch.zeros((count, original_B), dtype=torch.int32, device=self.dev)
            if len(rows):
                _logical[(len(rows), tr.cdiv(original_B, 16))](
                    out_hard, indices, rows, starts, ends, predictions, original_B, 256, 16,
                    num_warps=4)
            result.update(observables=predictions.T.to(torch.uint8).cpu().numpy(),
                          observables_fingerprint=self.logical_id)
        return result


def main():
    import argparse
    import json
    from pathlib import Path
    from profile_kernel_cost import load_fixture, sample, assert_same, timed_compact
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path,
                        default=Path(__file__).with_name('fixture_1480_Z.npz'))
    parser.add_argument('--shots', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--variable-bs', type=int, choices=(16, 32), default=32)
    parser.add_argument('--variable-vs', type=int, choices=(0, 8, 16, 32), default=16,
                        help='0 retains the baseline variable tiles for each degree')
    args = parser.parse_args()
    if args.shots < 1 or args.repeats < 1:
        parser.error('shots and repeats must be positive')
    H, L, pri, circuit = load_fixture(args.fixture)
    det = sample(circuit, args.shots)
    config = dict(check_warps=16, variable_warps=8, syndrome_warps=8)
    baseline = _BaseFirstLeg(H, pri, L=L, **config)
    expected = baseline.decode(det, return_details=True)
    baseline_timing = timed_compact(baseline, det, expected, args.repeats, 8)
    print(json.dumps(dict(event='baseline', shots=args.shots, timing=baseline_timing)), flush=True)
    candidate = ExactFirstLeg(H, pri, L=L, variable_bs=args.variable_bs,
                             variable_vs=args.variable_vs, **config)
    actual = candidate.decode(det, return_details=True)
    assert_same(expected, actual)
    timing = timed_compact(candidate, det, expected, args.repeats, 8)
    print(json.dumps(dict(event='padded', shots=args.shots, full_output_byte_equal=True,
                          variable_bs=args.variable_bs, variable_vs=args.variable_vs,
                          timing=timing,
                          speedup=baseline_timing['median_seconds'] / timing['median_seconds'])),
          flush=True)


if __name__ == '__main__':
    main()

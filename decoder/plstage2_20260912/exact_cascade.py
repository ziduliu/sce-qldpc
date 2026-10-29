"""exact_cascade with one extra mode, "gpu2": "hoist" with the stage-2 relay-BP
(and, with PLG_STAGE1=gpu, also stage 1) run on the GPU by gpu_relay2.GpuRelay,
using the gamma stream reproduced by chacha12. Modes orig, ref and hoist are
forwarded to the module at PLG_ORIG_CASCADE.

A relay decoder's gammas depend on its RNG stream position, which restarts at
zero in every cascade() call; _GpuDriver, which owns the position, is therefore
built per call, while the GpuRelay behind it holds no stream state and is
cached per process.
"""
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.environ.get("PLG_ORIG_CASCADE",
                      "/u/zdliu/lp/plcost_20260912/exact_cascade.py")
ACC = "/u/zdliu/lp/circmean/cpu_accel_20260911"
CIRCMEAN = "/u/zdliu/lp/circmean"
for _p in (HERE, CIRCMEAN, ACC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location("_exact_cascade_orig", ORIG)
_orig = importlib.util.module_from_spec(_spec)
sys.modules["_exact_cascade_orig"] = _orig
_spec.loader.exec_module(_orig)

import pl_at_p as PL  # noqa: E402
from screen_contract import (binary_array, model_fingerprint,  # noqa: E402
                             observables_fingerprint,
                             validate_screen_contract)

STAGE3 = _orig.STAGE3
build_stage3 = _orig.build_stage3
cached_stage3 = _orig.cached_stage3
stage3_stats = _orig.stage3_stats

GPU_STAGE1 = os.environ.get("PLG_STAGE1", "cpu") == "gpu"

_GPU_CACHE = {}
_GPU_BUILD_SECONDS = [0.0]


def gpu_relay(H, pri, cfg, tag):
    """One GpuRelay per (model, config) per process.  Holds no stream state."""
    from gpu_relay2 import GpuRelay
    key = (model_fingerprint(H, pri), tag)
    g = _GPU_CACHE.get(key)
    if g is None:
        t = time.perf_counter()
        g = GpuRelay(H, pri, cfg)
        g.warmup()
        _GPU_BUILD_SECONDS[0] += time.perf_counter() - t
        _GPU_CACHE[key] = g
    return g


def gpu_stats():
    return dict(builds=len(_GPU_CACHE),
                build_seconds=round(_GPU_BUILD_SECONDS[0], 2))


class _GpuDriver:
    """A stand-in for one RelayDecoderF32 instance: owns the stream position."""

    def __init__(self, relay, cfg):
        from chacha12 import GammaStream
        self.relay = relay
        self.cfg = cfg
        self.n = relay.N
        self.num_sets = int(cfg["num_sets"])
        lo, hi = cfg["gamma_dist_interval"]
        self.stream = GammaStream(seed=0, low=lo, high=hi)
        self.offset = 0
        self.gamma_seconds = 0.0
        self.decode_seconds = 0.0

    def decode(self, d):
        from chacha12 import chunked_gammas
        t = time.perf_counter()
        gam = (chunked_gammas(self.stream, self.offset, self.num_sets, self.n)
               if self.num_sets else None)
        self.gamma_seconds += time.perf_counter() - t
        t = time.perf_counter()
        out = self.relay.decode(d, gammas=gam)
        self.decode_seconds += time.perf_counter() - t
        self.offset += out["draws"]
        return out["decoding"]


def cascade(H, L, pri, det, obs, screen, mode="ref"):
    if mode != "gpu2":
        return _orig.cascade(H, L, pri, det, obs, screen, mode=mode)

    t0 = time.perf_counter()
    validate_screen_contract(screen, H, pri, det, PL.STAGE1)
    det = binary_array(det, 'detectors')
    if L.shape[1] != H.shape[1]:
        raise ValueError('Logical matrix and check matrix have different column counts')
    obs = binary_array(obs, 'observed logical flips', (len(det), L.shape[0]))
    good = np.asarray(screen['success'], bool)
    if 'observables' in screen:
        if screen.get('observables_fingerprint') != observables_fingerprint(L):
            raise ValueError('Screen observable predictions use a different logical matrix')
        predictions = binary_array(screen['observables'], 'screen observables', obs.shape).copy()
    else:
        decoding = binary_array(screen.get('decoding'), 'screen decoding', (len(det), H.shape[1]))
        predictions = ((L.astype(np.uint8) @ decoding.T).T & 1).astype(np.uint8)
    dec1 = dec2 = dec3 = None
    stages = [int(good.sum()), 0, 0]
    unsat = 0
    pri = np.asarray(pri, np.float64)
    builds_here = 0
    per_shot = []
    gamma_s = [0.0]
    gpu_s = [0.0]

    def ok(e, d):
        return not (((H @ e) % 2) != d).any()

    for i in np.flatnonzero(~good):
        d = det[i]
        t_shot = time.perf_counter()
        reached, t1, t2, t3, tb = 1, 0.0, 0.0, 0.0, 0.0
        if dec1 is None:
            if GPU_STAGE1:
                dec1 = _GpuDriver(gpu_relay(H, pri, PL.STAGE1, "STAGE1"),
                                  PL.STAGE1)
            else:
                from relay_bp import RelayDecoderF32
                dec1 = RelayDecoderF32(check_matrix=H, error_priors=pri,
                                       **PL.STAGE1)
        ta = time.perf_counter()
        e = np.asarray(dec1.decode(d), np.int64)
        t1 = time.perf_counter() - ta
        if ok(e, d):
            stages[0] += 1
        elif PL.CASCADE:
            reached = 2
            if dec2 is None:
                dec2 = _GpuDriver(gpu_relay(H, pri, PL.STAGE2, "STAGE2"),
                                  PL.STAGE2)
            ta = time.perf_counter()
            e = np.asarray(dec2.decode(d), np.int64)
            t2 = time.perf_counter() - ta
            if ok(e, d):
                stages[1] += 1
            else:
                reached = 3
                if dec3 is None:
                    before = _orig._DEC3_BUILDS[0]
                    ta = time.perf_counter()
                    dec3 = cached_stage3(H, pri)
                    tb = time.perf_counter() - ta
                    builds_here += _orig._DEC3_BUILDS[0] - before
                ta = time.perf_counter()
                e = np.asarray(dec3.decode(d), np.int64)
                t3 = time.perf_counter() - ta
                if ok(e, d):
                    stages[2] += 1
                else:
                    unsat += 1
        else:
            unsat += 1
        predictions[i] = np.asarray((L @ e) % 2, np.uint8)
        per_shot.append(dict(i=int(i), reached=reached,
                             total=round(time.perf_counter() - t_shot, 3),
                             s1=round(t1, 3), s2=round(t2, 3), s3=round(t3, 3),
                             s3_build=round(tb, 3)))
    for drv in (dec1, dec2):
        if isinstance(drv, _GpuDriver):
            gamma_s[0] += drv.gamma_seconds
            gpu_s[0] += drv.decode_seconds
    mismatch = predictions != obs
    out = (int(mismatch.any(axis=1).sum()), mismatch.sum(axis=0).astype(np.int64),
           unsat, stages, len(det))
    return out, dict(seconds=round(time.perf_counter() - t0, 3), shots=per_shot,
                     dec3_builds_in_call=builds_here,
                     gamma_seconds=round(gamma_s[0], 3),
                     gpu_relay_seconds=round(gpu_s[0], 3),
                     # which shots failed, for the block-level failure map
                     failed_idx=[int(i) for i in np.flatnonzero(mismatch.any(axis=1))])

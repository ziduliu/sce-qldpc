"""Decoder cascade for the shots the GPU screen leaves unconverged: relay-BP,
a second relay-BP, then BP-OSD (or BP-LSD with PLC_STAGE3=lsd).

  mode="ref"   -- re-implementation of continue_chunk; every decoder is built
                  inside the call.
  mode="hoist" -- as "ref" except that the stage-3 decoder comes from a
                  process-wide cache.

Only stage 3 is cached: a RelayDecoderF32 advances its RNG stream with every
relay set it runs, so stages 1 and 2 must be built fresh per call, while the
stage-3 decoder draws no random numbers. Diagnostics are returned in a separate
dict next to the count tuple.
"""
import os
import sys
import time

import numpy as np

ACC = "/u/zdliu/lp/circmean/cpu_accel_20260911"
CIRCMEAN = "/u/zdliu/lp/circmean"
for _p in (CIRCMEAN, ACC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pl_at_p as PL  # noqa: E402
from screen_contract import (binary_array, model_fingerprint,  # noqa: E402
                             observables_fingerprint,
                             validate_screen_contract)

# stage-3 decoder: "osd" (BP-OSD, as in continue_chunk) or "lsd" (BP-LSD order 1)
STAGE3 = os.environ.get("PLC_STAGE3", "osd")

_DEC3_CACHE = {}
_DEC3_BUILDS = [0]
_DEC3_BUILD_SECONDS = [0.0]


def build_stage3(H, pri):
    """Stage-3 decoder: continue_chunk's BP-OSD, or BP-LSD if STAGE3 is lsd."""
    if STAGE3 == "lsd":
        from ldpc.bplsd_decoder import BpLsdDecoder
        return BpLsdDecoder(H, error_channel=list(pri), max_iter=PL.OSD_ITERS,
                            bp_method="minimum_sum", ms_scaling_factor=0.,
                            lsd_method="lsd_cs", lsd_order=1)
    from ldpc import BpOsdDecoder
    return BpOsdDecoder(H, error_channel=list(pri), max_iter=PL.OSD_ITERS,
                        bp_method='minimum_sum', ms_scaling_factor=0.,
                        osd_method='osd_cs', osd_order=PL.OSD_ORDER)


def cached_stage3(H, pri):
    """One stage-3 decoder per (check matrix, priors) per process."""
    key = (model_fingerprint(H, pri), STAGE3)
    dec = _DEC3_CACHE.get(key)
    if dec is None:
        t = time.perf_counter()
        dec = build_stage3(H, pri)
        _DEC3_BUILD_SECONDS[0] += time.perf_counter() - t
        _DEC3_BUILDS[0] += 1
        _DEC3_CACHE[key] = dec
    return dec


def stage3_stats():
    return dict(builds=_DEC3_BUILDS[0], build_seconds=round(_DEC3_BUILD_SECONDS[0], 2))


def cascade(H, L, pri, det, obs, screen, mode="ref"):
    """Returns ((shot_fail, per_obs, unsat, stages, shots), diagnostics)."""
    from relay_bp import RelayDecoderF32

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

    def ok(e, d):
        return not (((H @ e) % 2) != d).any()

    for i in np.flatnonzero(~good):
        d = det[i]
        t_shot = time.perf_counter()
        reached, t1, t2, t3, tb = 1, 0.0, 0.0, 0.0, 0.0
        if dec1 is None:
            dec1 = RelayDecoderF32(check_matrix=H, error_priors=pri, **PL.STAGE1)
        ta = time.perf_counter()
        e = np.asarray(dec1.decode(d), np.int64)
        t1 = time.perf_counter() - ta
        if ok(e, d):
            stages[0] += 1
        elif PL.CASCADE:
            reached = 2
            if dec2 is None:
                dec2 = RelayDecoderF32(check_matrix=H, error_priors=pri, **PL.STAGE2)
            ta = time.perf_counter()
            e = np.asarray(dec2.decode(d), np.int64)
            t2 = time.perf_counter() - ta
            if ok(e, d):
                stages[1] += 1
            else:
                reached = 3
                if dec3 is None:
                    if mode == "hoist":
                        before = _DEC3_BUILDS[0]
                        ta = time.perf_counter()
                        dec3 = cached_stage3(H, pri)
                        tb = time.perf_counter() - ta
                        builds_here += _DEC3_BUILDS[0] - before
                    else:
                        ta = time.perf_counter()
                        dec3 = build_stage3(H, pri)
                        tb = time.perf_counter() - ta
                        builds_here += 1
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
        per_shot.append(dict(i=int(i), reached=reached, total=round(time.perf_counter() - t_shot, 3),
                             s1=round(t1, 3), s2=round(t2, 3), s3=round(t3, 3),
                             s3_build=round(tb, 3)))
    mismatch = predictions != obs
    out = (int(mismatch.any(axis=1).sum()), mismatch.sum(axis=0).astype(np.int64),
           unsat, stages, len(det))
    return out, dict(seconds=round(time.perf_counter() - t0, 3), shots=per_shot,
                     dec3_builds_in_call=builds_here)

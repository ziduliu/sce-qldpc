"""Bind a deterministic GPU screen to its exact CPU inputs and parameters.

Fingerprints are computed once per model or chunk, never once per shot.
This catches accidental reuse or parameter drift; it is not a substitute for
the CPU/GPU arithmetic regression tests.
"""
import hashlib
import numbers

import numpy as np
import scipy.sparse as sp


def _binary_csr(matrix, name):
    matrix = sp.csr_matrix(matrix, copy=False)
    if not matrix.has_canonical_format or not np.all(matrix.data == 1):
        raise ValueError(f'{name} must be a canonical binary CSR matrix')
    return matrix


def _hash_array(hasher, array, dtype):
    array = np.ascontiguousarray(array, dtype=dtype)
    hasher.update(np.asarray(array.shape, dtype='<i8').tobytes())
    hasher.update(array.tobytes())


def _hash_matrix(hasher, matrix):
    _hash_array(hasher, matrix.shape, '<i8')
    _hash_array(hasher, matrix.indptr, '<i8')
    _hash_array(hasher, matrix.indices, '<i8')


def model_fingerprint(H, pri):
    H = _binary_csr(H, 'H')
    pri = np.asarray(pri, dtype=np.float64)
    if pri.shape != (H.shape[1],) or not np.isfinite(pri).all():
        raise ValueError('Priors must be a finite vector with one value per column')
    if ((pri < 0) | (pri > 1)).any():
        raise ValueError('Priors must be probabilities between zero and one')
    hasher = hashlib.blake2b(digest_size=24, person=b'relay-screen-v1')
    _hash_matrix(hasher, H)
    _hash_array(hasher, pri, '<f8')
    return hasher.hexdigest()


def binary_array(value, name, shape=None):
    """Check before casting so values such as 0.5 or 256 cannot be truncated."""
    value = np.asarray(value)
    if shape is not None and value.shape != tuple(shape):
        raise ValueError(f'{name} has shape {value.shape}; expected {tuple(shape)}')
    if not np.isin(value, (0, 1)).all():
        raise ValueError(f'{name} must contain only zero and one')
    return np.asarray(value, dtype=np.uint8)


def detectors_fingerprint(det):
    det = binary_array(det, 'detectors')
    if det.ndim != 2:
        raise ValueError('Detectors must be a two-dimensional batch')
    hasher = hashlib.blake2b(digest_size=24, person=b'relay-det-v1')
    _hash_array(hasher, det, 'u1')
    return hasher.hexdigest()


def observables_fingerprint(L):
    L = _binary_csr(L, 'L')
    hasher = hashlib.blake2b(digest_size=24, person=b'relay-obs-v1')
    _hash_matrix(hasher, L)
    return hasher.hexdigest()


def _iteration_budget(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


def make_screen_protocol(model_id, *, pre_iter, gamma0=.1, alpha=1.):
    if gamma0 != .1 or alpha != 1.:
        raise ValueError('Only gamma0=0.1 and fixed alpha=1 are supported')
    if not isinstance(model_id, str) or not model_id:
        raise ValueError('A model fingerprint is required')
    return dict(model_fingerprint=model_id, gamma0=.1, alpha=1.,
                pre_iter=_iteration_budget(pre_iter, 'GPU pre_iter'))


def validate_screen_contract(screen, H, pri, det, stage1):
    """Fail before decoding if a screen cannot safely skip CPU first legs."""
    allowed = {'gamma0', 'pre_iter', 'num_sets', 'set_max_iter', 'stop_nconv',
               'gamma_dist_interval', 'seed', 'explicit_gammas', 'alpha',
               'alpha_iteration_scaling_factor', 'data_scale_value',
               'max_data_value', 'logging'}
    if set(stage1) - allowed:
        raise ValueError(f'Unsupported stage-1 options: {sorted(set(stage1) - allowed)}')
    if stage1.get('stop_nconv') != 1 or stage1.get('gamma0') != .1:
        raise ValueError('Exact first-leg screening requires stop_nconv=1 and gamma0=0.1')
    if stage1.get('alpha') not in (None, 1.):
        raise ValueError('Exact first-leg screening requires fixed alpha=1')
    if stage1.get('alpha_iteration_scaling_factor', 1.) != 1.:
        raise ValueError('Nondefault alpha iteration scaling is unsupported')
    if stage1.get('data_scale_value') is not None or stage1.get('max_data_value') is not None:
        raise ValueError('Data scaling and magnitude clipping are unsupported')
    if stage1.get('logging', False):
        raise ValueError('Skipping CPU calls would change relay logging output')
    original_budget = _iteration_budget(stage1.get('pre_iter'), 'CPU pre_iter')
    protocol = screen.get('protocol')
    if not isinstance(protocol, dict):
        raise ValueError('Screen protocol metadata is missing')
    if protocol.get('gamma0') != .1 or protocol.get('alpha') != 1.:
        raise ValueError('Screen gamma or alpha does not match the CPU protocol')
    budget = _iteration_budget(protocol.get('pre_iter'), 'GPU pre_iter')
    if budget > original_budget:
        raise ValueError('GPU first-leg budget must not exceed the CPU first-leg budget')
    if protocol.get('model_fingerprint') != model_fingerprint(H, pri):
        raise ValueError('Screen check matrix or priors do not match the CPU inputs')
    det = binary_array(det, 'detectors')
    if det.ndim != 2 or det.shape[1] != H.shape[0]:
        raise ValueError('Detector batch does not match the check matrix')
    if screen.get('detectors_fingerprint') != detectors_fingerprint(det):
        raise ValueError('Screen detector samples or sample order do not match')
    good = binary_array(screen.get('success'), 'screen success', (len(det),)).astype(bool)
    if budget == 0 and good.any():
        raise ValueError('A zero-iteration CPU first leg cannot report convergence')
    if 'unsafe_mask' not in screen:
        raise ValueError('Screen non-finite arithmetic safety mask is missing')
    unsafe = binary_array(screen['unsafe_mask'], 'screen unsafe mask', (len(det),)).astype(bool)
    if (good & unsafe).any():
        raise ValueError('Unsafe GPU samples must be sent to the CPU')
    if 'iterations' in screen:
        iterations = np.asarray(screen['iterations'])
        if iterations.shape != (len(det),) or not np.issubdtype(iterations.dtype, np.integer):
            raise ValueError('Screen iterations must be an integer vector')
        if ((iterations < 0) | (iterations > budget)).any() or (iterations[good] < 1).any():
            raise ValueError('Screen iteration counts do not match its budget')

"""Isomorphism-aware dedup key for CSS codes from Tanner-graph neighbourhood
fingerprints (every variable node is a root, so isomorphic codes always share
a key). The key is a necessary condition only; is_isomorphic_exact confirms a
collision with VF2++ on the typed Tanner graph.
"""
import hashlib
import os
from collections import Counter

import numpy as np

RADIUS = int(os.environ.get("CM_ISO_RADIUS", "2"))


def _matrix_fp(H, radius):
    H = np.asarray(H, dtype=np.uint8) % 2
    m, n = H.shape
    var_to_checks = [np.flatnonzero(H[:, j]).astype(np.int32) for j in range(n)]
    check_to_vars = [np.flatnonzero(H[i, :]).astype(np.int32) for i in range(m)]
    fp = Counter()
    for root_v in range(n):                       # ALL roots: exact invariant
        frontier = [("v", root_v)]
        seen_v = {root_v}
        seen_c = set()
        sig = []
        for r in range(1, radius + 1):
            nxt = []
            degrees = []
            revisits = 0
            for kind, idx in frontier:
                if kind == "v":
                    for c0 in var_to_checks[idx]:
                        c = int(c0)
                        degrees.append(len(check_to_vars[c]))
                        if c in seen_c:
                            revisits += 1
                        else:
                            seen_c.add(c)
                            nxt.append(("c", c))
                else:
                    for v0 in check_to_vars[idx]:
                        v = int(v0)
                        degrees.append(len(var_to_checks[v]))
                        if v in seen_v:
                            revisits += 1
                        else:
                            seen_v.add(v)
                            nxt.append(("v", v))
            hist = tuple(sorted(Counter(degrees).items())) if degrees else ()
            sig.append((r, hist, revisits, len(nxt), len(seen_v), len(seen_c)))
            frontier = nxt
        fp[repr(tuple(sig))] += 1
    return tuple(sorted(fp.items()))


def iso_key(HX, HZ, radius=None):
    """Isomorphism-invariant dedup key for the CSS pair (HX, HZ)."""
    radius = RADIUS if radius is None else radius
    blob = repr((_matrix_fp(HX, radius), _matrix_fp(HZ, radius)))
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def tanner_graph(HX, HZ):
    """Typed bipartite Tanner graph: qubits (t=0), X checks (1), Z (2)."""
    import networkx as nx
    HX = np.asarray(HX, np.uint8) % 2
    HZ = np.asarray(HZ, np.uint8) % 2
    G = nx.Graph()
    for j in range(HX.shape[1]):
        G.add_node(("q", j), t=0)
    for i in range(HX.shape[0]):
        G.add_node(("x", i), t=1)
    for i in range(HZ.shape[0]):
        G.add_node(("z", i), t=2)
    xi, xj = np.nonzero(HX)
    G.add_edges_from(zip([("x", int(a)) for a in xi], [("q", int(b)) for b in xj]))
    zi, zj = np.nonzero(HZ)
    G.add_edges_from(zip([("z", int(a)) for a in zi], [("q", int(b)) for b in zj]))
    return G


def is_isomorphic_exact(HX1, HZ1, HX2, HZ2, timeout=None):
    """Exact test (VF2++). Returns True/False; on timeout returns False
    (treat as distinct - never delete on an unproven claim)."""
    import signal
    import networkx as nx
    timeout = int(os.environ.get("CM_ISO_TIMEOUT", "30")) if timeout is None else timeout

    class _TO(Exception):
        pass

    def _h(sig, frm):
        raise _TO()
    old = signal.signal(signal.SIGALRM, _h)
    signal.alarm(timeout)
    try:
        return bool(nx.vf2pp_is_isomorphic(tanner_graph(HX1, HZ1),
                                           tanner_graph(HX2, HZ2),
                                           node_label="t"))
    except _TO:
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

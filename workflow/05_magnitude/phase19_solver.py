"""Exact solver for the phase19 station-term inversion (decision D-B091).

phase19 fits station ML_ijp = M_i + C_jp to the Hutton-Boore station magnitudes within
150 km, with two gauge rows (sum of the P terms = 0, sum of the S terms = 0, row weight
GAUGE_W = 1000) and damping DAMP = 1e-3 on every unknown, i.e. it minimises

    sum_k (m_k - M_e(k) - C_j(k))^2 + GAUGE_W^2 [(sum_P C)^2 + (sum_S C)^2]
        + DAMP^2 (|M|^2 + |C|^2).

Until D-B091 this was solved with scipy LSQR stopped at iter_lim 3000, which ends at
istop 7 (iteration limit) far from the minimum: the directions in which a whole
ocean-bottom deployment moves against its events cost little misfit, and a Krylov
solver resolves them last. Where it stops depends on rounding, i.e. on the BLAS thread
count.

method "exact" (default) solves the same objective directly. The normal equations are

    [ D    B ] [M]   [r_M]      D = diag(picks per event) + DAMP^2
    [ B'   H ] [C] = [r_C]      B = event x station-phase pick counts
                                H = diag(picks per term) + DAMP^2 + GAUGE_W^2 G G'

Eliminating the events (D is diagonal) leaves the Schur complement
S = H - B' D^-1 B, of size n_terms x n_terms (1,338 for v4.1), which is solved by a
symmetric eigendecomposition; then M = D^-1 (r_M - B C). Two steps of iterative
refinement on the full sparse system follow. The solve stops with NotConverged unless
the last refinement step is below REFINE_TOL and the relative normal-equation residual
|A'(b - Ax) - DAMP^2 x| / (|A|_F |b - Ax|) is below NORMAL_TOL. The result does not
depend on the starting point or on where an iteration stops.

method "lsqr" runs scipy LSQR on the same objective with the columns scaled by their
data-row norms (Jacobi preconditioning), iter_lim LSQR_ITER and atol = btol = LSQR_TOL,
and raises NotConverged unless istop is 1 or 2. It is the independent check of "exact"
(v4.1 pass 1: 332 iterations, max difference 2e-6). istop 2 bounds the normal-equation
residual, not the error: in a weak direction the error is that residual divided by a
small eigenvalue, so the tolerance must be tight (with atol = btol = 1e-8 the scaled
LSQR reports istop 2 after 176 iterations while 0.5 away from the minimum).

method "legacy3000" is the pre-D-B091 solver (LSQR, iter_lim 3000, atol = btol = 1e-8,
no convergence check); only for reproducing the stored v4.1 / v5.1 values.

Thread count: the exact solve is deterministic up to rounding (~1e-12). Pin BLAS threads
with OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS before numpy is imported
(the drivers set 12).

info (returned by solve): method, n_picks, n_events, n_terms, n_components (connected
components of the event / station-phase graph), terms_outside_main (station-phase terms
in a component not linked to the main network; their level is set by the damping only),
near_events_outside_main (events with >= 3 picks in such a component; must be 0),
r_norm (residual norm of the data rows), normal_res_rel, refine_step, istop, itn,
eig_min (three smallest eigenvalues of S).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr

DAMP = 1e-3
GAUGE_W = 1000.0
LEGACY_ITER = 3000
LSQR_ITER = 200_000
LSQR_TOL = 1e-12
REFINE_TOL = 1e-8
NORMAL_TOL = 1e-10


class NotConverged(RuntimeError):
    """The station-term inversion did not reach its convergence test."""


def build(d):
    """Factorise events and station-phase terms of the pick table d (columns event_id,
    sp, m) and build the phase19 design matrix with its gauge rows."""
    e, ev_ids = pd.factorize(d.event_id)
    j, sp_ids = pd.factorize(d.sp)
    n, ne, ns = len(d), len(ev_ids), len(sp_ids)
    m = d.m.to_numpy(dtype=float)
    ph = np.array([k.split("|")[1] for k in sp_ids])
    rows, cols, vals = [np.arange(n)] * 2, [e, ne + j], [np.ones(n)] * 2
    for i, p in enumerate(("P", "S")):
        sel = np.flatnonzero(ph == p)
        rows.append(np.full(len(sel), n + i))
        cols.append(ne + sel)
        vals.append(np.full(len(sel), GAUGE_W))
    A = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n + 2, ne + ns)).tocsr()
    b = np.concatenate([m, [0.0, 0.0]])
    return dict(e=e, j=j, ev_ids=ev_ids, sp_ids=sp_ids, ph=ph, n=n, ne=ne, ns=ns, m=m, A=A, b=b)


def components(s):
    """Connected components of the bipartite event / station-phase graph."""
    n, ne, ns = s["n"], s["ne"], s["ns"]
    g = sparse.coo_matrix((np.ones(n), (s["e"], ne + s["j"])), shape=(ne + ns, ne + ns)).tocsr()
    nc, lab = connected_components(g, directed=False)
    main = np.argmax(np.bincount(lab))
    out_ev = lab[:ne] != main
    out_sp = lab[ne:] != main
    cnt = np.bincount(s["e"], minlength=ne)
    return dict(n_components=int(nc), terms_outside_main=list(s["sp_ids"][out_sp]),
                near_events_outside_main=int((out_ev & (cnt >= 3)).sum()))


def schur(s, damp=DAMP):
    """Normal-equation blocks with the events eliminated: S (dense), its right-hand side
    and the pieces to recover M."""
    lam = damp ** 2
    ne, ns, e, j, m = s["ne"], s["ns"], s["e"], s["j"], s["m"]
    D = np.bincount(e, minlength=ne).astype(float) + lam
    B = sparse.coo_matrix((np.ones(s["n"]), (e, j)), shape=(ne, ns)).tocsr()
    G = np.zeros((ns, 2))
    G[s["ph"] == "P", 0] = GAUGE_W
    G[s["ph"] == "S", 1] = GAUGE_W
    S = (np.diag(np.bincount(j, minlength=ns).astype(float) + lam)
         - (B.T @ sparse.diags(1.0 / D) @ B).toarray() + G @ G.T)
    return S, D, B


def solve_exact(s, damp=DAMP, refine=2):
    lam = damp ** 2
    A, b, ne = s["A"], s["b"], s["ne"]
    S, D, B = schur(s, damp)
    w, V = np.linalg.eigh(S)
    if w[0] <= 0:
        raise NotConverged(f"Schur complement not positive definite (smallest eigenvalue {w[0]:.3e})")

    def step(gM, gC):
        dC = V @ ((V.T @ (gC - B.T @ (gM / D))) / w)
        return np.concatenate([(gM - B @ dC) / D, dC])

    x = np.zeros(A.shape[1])
    g = A.T @ b
    x = step(g[:ne], g[ne:])
    dx = np.inf
    for _ in range(refine):
        g = A.T @ (b - A @ x) - lam * x
        d = step(g[:ne], g[ne:])
        x = x + d
        dx = float(np.abs(d).max())
    r = b - A @ x
    g = A.T @ r - lam * x
    anorm = float(np.sqrt(A.multiply(A).sum()))
    rel = float(np.linalg.norm(g) / (anorm * np.linalg.norm(r)))
    info = dict(method="exact", refine_step=dx, normal_res_rel=rel, istop=None, itn=None,
                eig_min=[float(v) for v in w[:3]])
    if not (dx < REFINE_TOL and rel < NORMAL_TOL):
        raise NotConverged(f"exact solve: refinement step {dx:.2e} (limit {REFINE_TOL:.0e}), "
                           f"relative normal-equation residual {rel:.2e} (limit {NORMAL_TOL:.0e})")
    return x, info


def solve_lsqr(s, damp=DAMP, iter_lim=LSQR_ITER, tol=LSQR_TOL, check=True, scaled=True):
    """LSQR on the phase19 system. scaled: columns divided by their data-row norms (Jacobi
    preconditioning; the damping is carried as an explicit diagonal block so that the
    objective is unchanged). check: raise NotConverged unless istop is 1 or 2."""
    A, b, lam = s["A"], s["b"], damp ** 2
    if scaled:
        n = s["n"]
        cn = np.sqrt(np.asarray(A[:n].multiply(A[:n]).sum(0)).ravel())
        As = sparse.vstack([A @ sparse.diags(1.0 / cn), sparse.diags(damp / cn)]).tocsr()
        r = lsqr(As, np.concatenate([b, np.zeros(A.shape[1])]), damp=0.0, atol=tol, btol=tol,
                 iter_lim=iter_lim)
        x = r[0] / cn
    else:
        r = lsqr(A, b, damp=damp, atol=tol, btol=tol, iter_lim=iter_lim)
        x = r[0]
    istop, itn = int(r[1]), int(r[2])
    res = b - A @ x
    g = A.T @ res - lam * x
    anorm = float(np.sqrt(A.multiply(A).sum()))
    info = dict(method=("lsqr" + ("_scaled" if scaled else "")) if check else "legacy3000", istop=istop,
                itn=itn, normal_res_rel=float(np.linalg.norm(g) / (anorm * np.linalg.norm(res))),
                refine_step=None, eig_min=None)
    if check and istop not in (1, 2):
        raise NotConverged(f"LSQR stopped with istop {istop} after {itn} iterations "
                           f"(iter_lim {iter_lim}); not converged")
    return x, info


def solve(d, method="exact"):
    """station ML = M_i + C_jp on the pick table d (event_id, sp, m). Returns M (Series by
    event), C (Series by station-phase), per-pick residuals and the info dict."""
    s = build(d)
    comp = components(s)
    if comp["near_events_outside_main"]:
        raise NotConverged(f"{comp['near_events_outside_main']} events with >= 3 picks lie in a component "
                           f"not linked to the main network: their level is set by the damping only")
    if method == "exact":
        x, info = solve_exact(s)
    elif method == "lsqr":
        x, info = solve_lsqr(s)
    elif method == "legacy3000":
        x, info = solve_lsqr(s, iter_lim=LEGACY_ITER, tol=1e-8, check=False, scaled=False)
    else:
        raise ValueError(f"unknown method {method!r}")
    ne = s["ne"]
    res = s["m"] - x[s["e"]] - x[ne + s["j"]]
    info.update(comp, n_picks=s["n"], n_events=ne, n_terms=s["ns"], r_norm=float(np.linalg.norm(res)))
    return pd.Series(x[:ne], index=s["ev_ids"]), pd.Series(x[ne:], index=s["sp_ids"]), res, info

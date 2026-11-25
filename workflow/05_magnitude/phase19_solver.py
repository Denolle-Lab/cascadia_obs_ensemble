"""Exact solver for the phase19 station-term inversion (decision D-B091).

phase19 fits station ML_ijp = M_i + C_jp to the Hutton-Boore station magnitudes within
150 km, with damping DAMP = 1e-3 on every unknown and gauge rows of weight GAUGE_W = 1000:

    sum_k (m_k - M_e(k) - C_j(k))^2 + GAUGE_W^2 sum_g (sum_{j in g} C_j)^2
        + DAMP^2 (|M|^2 + |C|^2).

Gauge (D-B091, verifier must-fix 1). The data fix every M_i + C_jp but not a common shift:
on each connected piece of the event / station-phase graph, all event terms +delta and
all station terms -delta leave the misfit unchanged. The number of these null directions
is the number of connected pieces, and the gauge needs exactly one row per null
direction. More rows than null directions do not fix a gauge: they constrain the data
fit.
  * The stored runs used two rows, sum of the P terms = 0 and sum of the S terms = 0.
    The graph had two pieces, the main network and one isolated S term
    (7D.J35C@2013-08-09|S, 9 picks on 9 one-pick events). The S row was absorbed by that
    term, which ended at C = -303 (v4.1) and -295 (v5.1). The two rows therefore fixed
    the two null directions, and the P - S level difference of the main network stayed
    free (mean S term - mean P term +0.454).
  * Without the isolated term, the same two rows would have forced mean S = mean P on the
    main network. The misfit then rises by 2,938, and regional ML moves by up to -0.22
    (Mendocino, offshore forearc, Gorda).
  * Now: picks not linked to the main network are dropped (main_component(); their
    events have fewer than 3 picks, so no ML is lost). ONE gauge row is used, the sum of
    the P terms = 0, which keeps the P terms and the main-network S terms of the stored
    solution. check_gauge() verifies that the gauge rows number exactly the null
    directions, counted from the graph and numerically from the eigenvalues of the
    reduced system, and that they are not orthogonal to the null space. It raises
    NotConverged otherwise. The ML of the stored converged solution is unchanged
    (max |dML| 0.0015, from the damping). The anchor offset absorbs any choice of a
    single gauge row, so ML does not depend on which one.

Until D-B091 the inversion was solved with scipy LSQR stopped at iter_lim 3000, which
ends at istop 7 (iteration limit) far from the minimum. The directions in which a whole
ocean-bottom deployment moves against its events cost little misfit, and a Krylov
solver resolves them last. Where it stops depends on rounding, i.e. on the BLAS thread
count.

method "exact" (default) solves the objective directly. The normal equations are

    [ D    B ] [M]   [r_M]      D = diag(picks per event) + DAMP^2
    [ B'   H ] [C] = [r_C]      B = event x station-phase pick counts
                                H = diag(picks per term) + DAMP^2 + GAUGE_W^2 G G'

Eliminating the events (D is diagonal) leaves the Schur complement
S = H - B' D^-1 B, of size n_terms x n_terms (1,337 for v4.1). It is solved by a
symmetric eigendecomposition, then M = D^-1 (r_M - B C). Two steps of iterative
refinement on the full sparse system follow. The solve stops with NotConverged unless
the last refinement step is below REFINE_TOL and the relative normal-equation residual
|A'(b - Ax) - DAMP^2 x| / (|A|_F |b - Ax|) is below NORMAL_TOL. The result does not
depend on the starting point or on where an iteration stops.

method "lsqr" runs scipy LSQR on the same objective with the columns scaled by their
data-row norms (Jacobi preconditioning), iter_lim LSQR_ITER and atol = btol = LSQR_TOL.
It raises NotConverged unless istop is 1 or 2, and serves as the independent check of
"exact" (v4.1: istop 2 after 238 and 240 iterations, identical output file). istop 2 bounds the
normal-equation residual, not the error: in a weak direction the error is that residual
divided by a small eigenvalue, so the tolerance must be tight. With atol = btol = 1e-8
the scaled LSQR reports istop 2 after 176 iterations while 0.5 away from the minimum.

method "legacy3000" is the pre-D-B091 solver (two gauge rows, no component filter, LSQR,
iter_lim 3000, atol = btol = 1e-8, no convergence or gauge check). It is used only to
reproduce the stored v4.1 / v5.1 values.

Thread count: the exact solve is deterministic up to rounding (~1e-12). Pin BLAS threads
with OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS before numpy is imported
(the drivers set 12).

info (returned by solve) holds:
  method, n_picks, n_events, n_terms;
  n_components: connected pieces of the system solved (1 after main_component);
  n_null_numerical: eigenvalues of the reduced system below NULL_TOL x the largest;
  n_gauge_rows;
  gauge_null_sv: smallest singular value of G' V_null, relative to |G|;
  r_norm: residual norm of the data rows;
  normal_res_rel, refine_step, istop, itn;
  eig_min: three smallest eigenvalues of S.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr

DAMP = 1e-3
GAUGE_W = 1000.0
GAUGE = "P"            # one row: sum of the P terms = 0. "PS" = the stored two rows (legacy only)
LEGACY_ITER = 3000
LSQR_ITER = 200_000
LSQR_TOL = 1e-12
REFINE_TOL = 1e-8
NORMAL_TOL = 1e-10
NULL_TOL = 1e-9


class NotConverged(RuntimeError):
    """The station-term inversion did not reach its convergence or gauge test."""


def _graph(d):
    e, ev_ids = pd.factorize(d.event_id)
    j, sp_ids = pd.factorize(d.sp)
    ne, ns = len(ev_ids), len(sp_ids)
    g = sparse.coo_matrix((np.ones(len(d)), (e, ne + j)), shape=(ne + ns, ne + ns)).tocsr()
    nc, lab = connected_components(g, directed=False)
    return e, ev_ids, j, sp_ids, nc, lab


def main_component(d, method="exact", verbose=True):
    """The picks of the largest connected piece of the event / station-phase graph.
    Picks outside it have a level set by the damping only; they are dropped. Raises if
    an event with 3 or more picks lies outside, since its ML would be lost. No-op for
    legacy3000, which reproduces the stored runs."""
    if method == "legacy3000":
        return d
    e, ev_ids, j, sp_ids, nc, lab = _graph(d)
    if nc == 1:
        return d
    ne = len(ev_ids)
    main = np.argmax(np.bincount(lab))
    keep = lab[e] == main
    cnt = np.bincount(e, minlength=ne)
    lost = np.unique(e[~keep])
    if (cnt[lost] >= 3).any():
        raise NotConverged(f"{int((cnt[lost] >= 3).sum())} events with >= 3 picks are not linked to the main network")
    if verbose:
        terms = sorted(set(sp_ids[j[~keep]]))
        print(f"  main_component: dropped {int((~keep).sum())} picks of {len(lost)} events on {len(terms)} "
              f"station-phase term(s) not linked to the main network: {terms}", flush=True)
    return d[keep]


def build(d, gauge=None):
    """Factorise events and station-phase terms of the pick table d (columns event_id,
    sp, m) and build the phase19 design matrix with its gauge rows ("P": one row, sum of
    the P terms; "PS": the stored two rows)."""
    gauge = gauge or GAUGE
    e, ev_ids = pd.factorize(d.event_id)
    j, sp_ids = pd.factorize(d.sp)
    n, ne, ns = len(d), len(ev_ids), len(sp_ids)
    m = d.m.to_numpy(dtype=float)
    ph = np.array([k.split("|")[1] for k in sp_ids])
    groups = [np.flatnonzero(ph == p) for p in gauge]
    rows, cols, vals = [np.arange(n)] * 2, [e, ne + j], [np.ones(n)] * 2
    for i, sel in enumerate(groups):
        rows.append(np.full(len(sel), n + i))
        cols.append(ne + sel)
        vals.append(np.full(len(sel), GAUGE_W))
    A = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n + len(groups), ne + ns)).tocsr()
    b = np.concatenate([m, np.zeros(len(groups))])
    G = np.zeros((ns, len(groups)))
    for i, sel in enumerate(groups):
        G[sel, i] = GAUGE_W
    return dict(e=e, j=j, ev_ids=ev_ids, sp_ids=sp_ids, ph=ph, n=n, ne=ne, ns=ns, m=m, A=A, b=b, G=G, gauge=gauge)


def components(s):
    """Connected components of the bipartite event / station-phase graph of a built system."""
    n, ne, ns = s["n"], s["ne"], s["ns"]
    g = sparse.coo_matrix((np.ones(n), (s["e"], ne + s["j"])), shape=(ne + ns, ne + ns)).tocsr()
    nc, lab = connected_components(g, directed=False)
    main = np.argmax(np.bincount(lab))
    out_ev = lab[:ne] != main
    out_sp = lab[ne:] != main
    cnt = np.bincount(s["e"], minlength=ne)
    return dict(n_components=int(nc), terms_outside_main=list(s["sp_ids"][out_sp]),
                near_events_outside_main=int((out_ev & (cnt >= 3)).sum()))


def schur_blocks(s, damp=DAMP):
    """Normal-equation blocks with the events eliminated: S0 (no gauge, no damping), the
    damped S with the gauge rows, and the pieces to recover M."""
    lam = damp ** 2
    ne, ns, e, j = s["ne"], s["ns"], s["e"], s["j"]
    cnt_e = np.bincount(e, minlength=ne).astype(float)
    cnt_j = np.bincount(j, minlength=ns).astype(float)
    B = sparse.coo_matrix((np.ones(s["n"]), (e, j)), shape=(ne, ns)).tocsr()
    S0 = np.diag(cnt_j) - (B.T @ sparse.diags(1.0 / cnt_e) @ B).toarray()
    D = cnt_e + lam
    S = np.diag(cnt_j + lam) - (B.T @ sparse.diags(1.0 / D) @ B).toarray() + s["G"] @ s["G"].T
    return S0, S, D, B


def check_gauge(s, S0, raise_on_fail=True):
    """Gauge rows must equal the null directions of the data part: counted from the graph
    (connected pieces) and numerically (eigenvalues of S0 below NULL_TOL x the largest),
    and the gauge rows must not be orthogonal to the null space."""
    comp = components(s)
    w0, V0 = np.linalg.eigh(S0)
    null = w0 < NULL_TOL * w0[-1]
    n_null = int(null.sum())
    n_g = s["G"].shape[1]
    sv = (np.linalg.svd(s["G"].T @ V0[:, null], compute_uv=False).min() / np.linalg.norm(s["G"])
          if n_null and n_null == n_g else np.nan)
    info = dict(n_components=comp["n_components"], terms_outside_main=comp["terms_outside_main"],
                n_null_numerical=n_null, n_gauge_rows=n_g, gauge_null_sv=float(sv))
    ok = comp["n_components"] == n_null == n_g and sv > 1e-6
    if raise_on_fail and not ok:
        raise NotConverged(f"gauge: {n_g} gauge row(s) but {comp['n_components']} connected piece(s) and {n_null} "
                           f"numerical null direction(s) (gauge/null-space singular value {sv:.2e}); extra rows "
                           f"would constrain the fit, missing rows leave a level free. Use main_component() and "
                           f"one gauge row.")
    return info


def solve_exact(s, damp=DAMP, refine=2, check=True):
    lam = damp ** 2
    A, b, ne = s["A"], s["b"], s["ne"]
    S0, S, D, B = schur_blocks(s, damp)
    ginfo = check_gauge(s, S0, raise_on_fail=check)
    w, V = np.linalg.eigh(S)
    if w[0] <= 0:
        raise NotConverged(f"Schur complement not positive definite (smallest eigenvalue {w[0]:.3e})")

    def step(gM, gC):
        dC = V @ ((V.T @ (gC - B.T @ (gM / D))) / w)
        return np.concatenate([(gM - B @ dC) / D, dC])

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
                eig_min=[float(v) for v in w[:3]], **ginfo)
    if not (dx < REFINE_TOL and rel < NORMAL_TOL):
        raise NotConverged(f"exact solve: refinement step {dx:.2e} (limit {REFINE_TOL:.0e}), "
                           f"relative normal-equation residual {rel:.2e} (limit {NORMAL_TOL:.0e})")
    return x, info


def solve_lsqr(s, damp=DAMP, iter_lim=LSQR_ITER, tol=LSQR_TOL, check=True, scaled=True):
    """LSQR on the phase19 system. scaled: columns divided by their data-row norms (Jacobi
    preconditioning; the damping is carried as an explicit diagonal block so that the
    objective is unchanged). check: gauge test, and NotConverged unless istop is 1 or 2."""
    A, b, lam = s["A"], s["b"], damp ** 2
    ginfo = {}
    if check:
        ginfo = check_gauge(s, schur_blocks(s, damp)[0])
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
                refine_step=None, eig_min=None, **ginfo)
    if check and istop not in (1, 2):
        raise NotConverged(f"LSQR stopped with istop {istop} after {itn} iterations "
                           f"(iter_lim {iter_lim}); not converged")
    return x, info


def solve(d, method="exact"):
    """station ML = M_i + C_jp on the pick table d (event_id, sp, m), which must be one
    connected piece for "exact" and "lsqr" (main_component). Returns M (Series by event),
    C (Series by station-phase), per-pick residuals and the info dict."""
    if method == "legacy3000":
        s = build(d, gauge="PS")
        x, info = solve_lsqr(s, iter_lim=LEGACY_ITER, tol=1e-8, check=False, scaled=False)
        info.update(components(s), n_gauge_rows=2)
    else:
        s = build(d)
        if method == "exact":
            x, info = solve_exact(s)
        elif method == "lsqr":
            x, info = solve_lsqr(s)
        else:
            raise ValueError(f"unknown method {method!r}")
    ne = s["ne"]
    res = s["m"] - x[s["e"]] - x[ne + s["j"]]
    info.update(n_picks=s["n"], n_events=ne, n_terms=s["ns"], gauge=s["gauge"], r_norm=float(np.linalg.norm(res)))
    return pd.Series(x[:ne], index=s["ev_ids"]), pd.Series(x[ne:], index=s["sp_ids"]), res, info

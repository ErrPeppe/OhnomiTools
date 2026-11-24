# src/ohnomitools/stats.py
from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True)
def _seed_numba(seed: int) -> None:
    np.random.seed(seed)


@njit(cache=True)
def _bootstrap_means_numba(arr: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Genera `n_boot` medie bootstrap non parametriche in O(n_boot * n) con memoria O(n_boot)."""
    _seed_numba(seed)
    n = arr.shape[0]
    out = np.empty(n_boot, dtype=np.float64)
    inv_n = 1.0 / n
    for b in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += arr[np.random.randint(0, n)]
        out[b] = s * inv_n
    return out


def bootstrap_mean_ci(
    arr: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Restituisce (media, ci_low, ci_high) tramite bootstrap percentile non parametrico."""
    vals = np.asarray(arr, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean_val = float(np.mean(vals))
    if n == 1 or float(np.std(vals)) <= 1e-15:
        return mean_val, mean_val, mean_val
    boot_means = _bootstrap_means_numba(vals, n_boot, seed)
    lo, hi = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return mean_val, float(lo), float(hi)


@njit(cache=True)
def rewire_bipartite_maslov_sneppen(
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    offsets: np.ndarray,
    n_attempts: int,
) -> int:
    """Double-edge swap microcanonico degree-preserving su grafo bipartito CSR."""
    n_edges = edge_u.shape[0]
    if n_edges < 2:
        return 0

    accepted = 0
    for _ in range(n_attempts):
        e1 = np.random.randint(0, n_edges)
        e2 = np.random.randint(0, n_edges)
        u1 = edge_u[e1]
        u2 = edge_u[e2]
        if u1 == u2:
            continue

        v1 = edge_v[e1]
        v2 = edge_v[e2]
        if v1 == v2:
            continue

        s1 = offsets[u1]
        end1 = offsets[u1 + 1]
        dup = False
        for idx in range(s1, end1):
            if edge_v[idx] == v2:
                dup = True
                break
        if dup:
            continue

        s2 = offsets[u2]
        end2 = offsets[u2 + 1]
        for idx in range(s2, end2):
            if edge_v[idx] == v1:
                dup = True
                break
        if dup:
            continue

        edge_v[e1] = v2
        edge_v[e2] = v1
        accepted += 1

    return accepted


@njit(cache=True)
def evaluate_pair_motifs_numba(
    offsets: np.ndarray,
    edge_v: np.ndarray,
    n_v: int,
    pair_u: np.ndarray,
    pair_w: np.ndarray,
    ppi_mat: np.ndarray,
    dup_mat: np.ndarray,
    use_ppi: bool,
    use_dup: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Calcola per ciascuna coppia (u, w):
    v_motifs, dice, jaccard, generic_k22_bifans, dup_target_bifans,
    ppi_complex_bifans (STRING), ppi_dup_bifans (STRING & paraloghi).
    """
    n_pairs = pair_u.shape[0]
    v_motifs = np.zeros(n_pairs, dtype=np.int64)
    dice = np.zeros(n_pairs, dtype=np.float64)
    jaccard = np.zeros(n_pairs, dtype=np.float64)
    generic_k22_bifans = np.zeros(n_pairs, dtype=np.int64)
    dup_target_bifans = np.zeros(n_pairs, dtype=np.int64)
    ppi_complex_bifans = np.zeros(n_pairs, dtype=np.int64)
    ppi_dup_bifans = np.zeros(n_pairs, dtype=np.int64)

    seen = np.zeros(n_v, dtype=np.int32)
    common_buf = np.empty(n_v, dtype=np.int32)
    stamp = 0

    for p in range(n_pairs):
        u = pair_u[p]
        w = pair_w[p]
        s1 = offsets[u]
        e1 = offsets[u + 1]
        s2 = offsets[w]
        e2 = offsets[w + 1]
        d1 = e1 - s1
        d2 = e2 - s2
        if d1 == 0 or d2 == 0:
            continue

        stamp += 1
        if stamp >= 2_000_000_000:
            seen[:] = 0
            stamp = 1

        for i in range(s1, e1):
            seen[edge_v[i]] = stamp

        c = 0
        for j in range(s2, e2):
            t = edge_v[j]
            if seen[t] == stamp:
                common_buf[c] = t
                c += 1

        v_motifs[p] = c
        if c > 0:
            dice[p] = (2.0 * c) / (d1 + d2)
            jaccard[p] = c / (d1 + d2 - c)
            if c >= 2:
                generic_k22_bifans[p] = (c * (c - 1)) // 2
                if use_ppi or use_dup:
                    pcb = 0
                    db = 0
                    pdb = 0
                    for a in range(c - 1):
                        ta = common_buf[a]
                        for b in range(a + 1, c):
                            tb = common_buf[b]
                            is_ppi = use_ppi and (ppi_mat[ta, tb] == 1)
                            is_dup = use_dup and (dup_mat[ta, tb] == 1)
                            if is_ppi:
                                pcb += 1
                            if is_dup:
                                db += 1
                            if is_ppi and is_dup:
                                pdb += 1
                    ppi_complex_bifans[p] = pcb
                    dup_target_bifans[p] = db
                    ppi_dup_bifans[p] = pdb

    return (
        v_motifs,
        dice,
        jaccard,
        generic_k22_bifans,
        dup_target_bifans,
        ppi_complex_bifans,
        ppi_dup_bifans,
    )


@njit(cache=True)
def evaluate_single_node_ppi_delta_numba(
    offsets: np.ndarray,
    edge_v: np.ndarray,
    nodes: np.ndarray,
    ppi_mat: np.ndarray,
    dup_mat: np.ndarray,
    use_dup: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcola per ciascun singolo nodo `u`:
    - ppi_complex_delta: coppie di bersagli connessi in STRING
    - ppi_dup_delta: coppie di bersagli sia connessi in STRING sia paraloghi
    """
    n_nodes = nodes.shape[0]
    out_complex = np.zeros(n_nodes, dtype=np.int64)
    out_dup = np.zeros(n_nodes, dtype=np.int64)

    for idx in range(n_nodes):
        u = nodes[idx]
        s = offsets[u]
        e = offsets[u + 1]
        if e - s < 2:
            continue
        cnt_c = 0
        cnt_d = 0
        for i in range(s, e - 1):
            ta = edge_v[i]
            for j in range(i + 1, e):
                tb = edge_v[j]
                if ppi_mat[ta, tb] == 1:
                    cnt_c += 1
                    if use_dup and dup_mat[ta, tb] == 1:
                        cnt_d += 1
        out_complex[idx] = cnt_c
        out_dup[idx] = cnt_d

    return out_complex, out_dup


@njit(cache=True)
def run_null_ensemble_pairs_numba(
    edge_u: np.ndarray,
    edge_v_orig: np.ndarray,
    offsets: np.ndarray,
    n_v: int,
    pair_u: np.ndarray,
    pair_w: np.ndarray,
    ppi_mat: np.ndarray,
    dup_mat: np.ndarray,
    use_ppi: bool,
    use_dup: bool,
    n_null: int,
    burn_in_factor: int,
    step_factor: int,
    seed: int,
    track_per_pair: bool,
    independent_replicas: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Ensemble nullo di Maslov-Sneppen su coppie di nodi con aggregazione Welford per coppia.
    """
    _seed_numba(seed)
    n_edges = edge_u.shape[0]
    n_pairs = pair_u.shape[0]

    edge_v = edge_v_orig.copy()

    null_v = np.zeros(n_null, dtype=np.float64)
    null_dice = np.zeros(n_null, dtype=np.float64)
    null_gk = np.zeros(n_null, dtype=np.float64)
    null_db = np.zeros(n_null, dtype=np.float64)
    null_pcb = np.zeros(n_null, dtype=np.float64)
    null_pdb = np.zeros(n_null, dtype=np.float64)

    p_len = n_pairs if track_per_pair else 1
    pv_mean = np.zeros(p_len, dtype=np.float64)
    pv_m2 = np.zeros(p_len, dtype=np.float64)
    ppcb_mean = np.zeros(p_len, dtype=np.float64)
    ppcb_m2 = np.zeros(p_len, dtype=np.float64)
    ppdb_mean = np.zeros(p_len, dtype=np.float64)
    ppdb_m2 = np.zeros(p_len, dtype=np.float64)

    if not independent_replicas:
        rewire_bipartite_maslov_sneppen(edge_u, edge_v, offsets, burn_in_factor * n_edges)

    for r in range(n_null):
        if independent_replicas:
            edge_v = edge_v_orig.copy()
            rewire_bipartite_maslov_sneppen(edge_u, edge_v, offsets, burn_in_factor * n_edges)
        elif r > 0:
            rewire_bipartite_maslov_sneppen(edge_u, edge_v, offsets, step_factor * n_edges)

        vm, dc, _, gk, db, pcb, pdb = evaluate_pair_motifs_numba(
            offsets, edge_v, n_v, pair_u, pair_w, ppi_mat, dup_mat, use_ppi, use_dup
        )

        null_v[r] = float(vm.sum())
        null_dice[r] = float(dc.mean()) if n_pairs > 0 else 0.0
        null_gk[r] = float(gk.sum())
        null_db[r] = float(db.sum())
        null_pcb[r] = float(pcb.sum())
        null_pdb[r] = float(pdb.sum())

        if track_per_pair:
            inv_cnt = 1.0 / (r + 1)
            for p in range(n_pairs):
                fv = float(vm[p])
                dv = fv - pv_mean[p]
                pv_mean[p] += dv * inv_cnt
                pv_m2[p] += dv * (fv - pv_mean[p])

                fc = float(pcb[p])
                dc_ = fc - ppcb_mean[p]
                ppcb_mean[p] += dc_ * inv_cnt
                ppcb_m2[p] += dc_ * (fc - ppcb_mean[p])

                fd = float(pdb[p])
                dd = fd - ppdb_mean[p]
                ppdb_mean[p] += dd * inv_cnt
                ppdb_m2[p] += dd * (fd - ppdb_mean[p])

    return (
        null_v, null_dice, null_gk, null_db, null_pcb, null_pdb,
        pv_mean, pv_m2, ppcb_mean, ppcb_m2, ppdb_mean, ppdb_m2,
    )


@njit(cache=True)
def run_null_ensemble_ppi_delta_numba(
    edge_u: np.ndarray,
    edge_v_orig: np.ndarray,
    offsets: np.ndarray,
    nodes: np.ndarray,
    ppi_mat: np.ndarray,
    dup_mat: np.ndarray,
    use_dup: bool,
    n_null: int,
    burn_in_factor: int,
    step_factor: int,
    seed: int,
    independent_replicas: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble nullo di Maslov-Sneppen per PPI-Complex Delta e PPI-Dup Delta (Via 3)."""
    _seed_numba(seed)
    n_edges = edge_u.shape[0]
    edge_v = edge_v_orig.copy()
    null_complex = np.zeros(n_null, dtype=np.float64)
    null_dup = np.zeros(n_null, dtype=np.float64)

    if not independent_replicas:
        rewire_bipartite_maslov_sneppen(edge_u, edge_v, offsets, burn_in_factor * n_edges)

    for r in range(n_null):
        if independent_replicas:
            edge_v = edge_v_orig.copy()
            rewire_bipartite_maslov_sneppen(edge_u, edge_v, offsets, burn_in_factor * n_edges)
        elif r > 0:
            rewire_bipartite_maslov_sneppen(edge_u, edge_v, offsets, step_factor * n_edges)

        cnt_c, cnt_d = evaluate_single_node_ppi_delta_numba(
            offsets, edge_v, nodes, ppi_mat, dup_mat, use_dup
        )
        null_complex[r] = float(cnt_c.sum())
        null_dup[r] = float(cnt_d.sum())

    return null_complex, null_dup


def compute_zscore(obs: float, null_arr: np.ndarray) -> tuple[float, float, float, float]:
    """
    Restituisce (null_mean, null_std, z_score, emp_p_value_two_sided).
    Se sigma_null == 0: z = 0.0 quando obs == mean, altrimenti NaN (Agasso et al., 2025).
    """
    mean = float(np.mean(null_arr))
    std = float(np.std(null_arr, ddof=1)) if len(null_arr) > 1 else 0.0
    if std > 1e-12:
        z = (obs - mean) / std
    elif abs(obs - mean) <= 1e-12:
        z = 0.0
    else:
        z = float("nan")
    p_ge = (float(np.sum(null_arr >= obs)) + 1.0) / (len(null_arr) + 1.0)
    p_le = (float(np.sum(null_arr <= obs)) + 1.0) / (len(null_arr) + 1.0)
    p_emp = min(1.0, 2.0 * min(p_ge, p_le))
    return mean, std, z, p_emp

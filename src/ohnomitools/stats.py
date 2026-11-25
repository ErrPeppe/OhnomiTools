# src/ohnomitools/stats.py
from __future__ import annotations
import numpy as np
from numba import njit


@njit
def _seed_numba(seed: int) -> None:
    np.random.seed(seed)


@njit
def _bootstrap_means_numba(arr: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Genera `n_boot` medie bootstrap non parametriche in O(n_boot * n)."""
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


@njit
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


@njit
def rewire_bipartite_maslov_sneppen_dense(
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    adj: np.ndarray,
    n_attempts: int,
) -> int:
    """Double-edge swap microcanonico O(1) per reti compatte (es. layer RBP)."""
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

        if adj[u1, v2] != 0 or adj[u2, v1] != 0:
            continue

        adj[u1, v1] = 0
        adj[u2, v2] = 0
        adj[u1, v2] = 1
        adj[u2, v1] = 1
        edge_v[e1] = v2
        edge_v[e2] = v1
        accepted += 1

    return accepted


@njit
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


@njit
def evaluate_single_node_ppi_delta_numba(
    offsets: np.ndarray,
    edge_v: np.ndarray,
    nodes: np.ndarray,
    ppi_mat: np.ndarray,
    dup_mat: np.ndarray,
    use_dup: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcola ppi_complex_delta (STRING) e ppi_dup_delta (STRING & paraloghi) per nodo."""
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


@njit
def run_null_ensemble_pairs_numba(
    edge_u: np.ndarray,
    edge_v_orig: np.ndarray,
    offsets: np.ndarray,
    n_v: int,
    pair_u: np.ndarray,
    pair_w: np.ndarray,
    n_wgd: int,
    ppi_mat: np.ndarray,
    dup_mat: np.ndarray,
    use_ppi: bool,
    use_dup: bool,
    n_null: int,
    burn_in_factor: int,
    step_factor: int,
    seed: int,
    track_wgd: bool,
    track_ssd: bool,
    independent_replicas: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensemble nullo condiviso WGD + SSD in singolo passaggio con Welford online.
    Indici metriche in null_wgd/null_ssd (6, n_null):
      0=v_motifs, 1=mean_dice, 2=generic_k22_bifans, 3=dup_target_bifans,
      4=ppi_complex_bifans, 5=ppi_dup_bifans.
    Indici in pair_mean/pair_m2 (3, n_total):
      0=v_motifs, 1=ppi_complex_bifans, 2=ppi_dup_bifans.
    """
    _seed_numba(seed)
    n_edges = edge_u.shape[0]
    n_u = offsets.shape[0] - 1
    n_total = pair_u.shape[0]
    n_ssd = n_total - n_wgd

    edge_v = edge_v_orig.copy()
    use_dense = (n_u * n_v) <= 20_000_000
    if use_dense:
        adj = np.zeros((n_u, n_v), dtype=np.uint8)
        for k in range(n_edges):
            adj[edge_u[k], edge_v[k]] = 1
    else:
        adj = np.zeros((1, 1), dtype=np.uint8)

    null_wgd = np.zeros((6, n_null), dtype=np.float64)
    null_ssd = np.zeros((6, n_null), dtype=np.float64)

    p_len = n_total if (track_wgd or track_ssd) else 1
    pair_mean = np.zeros((3, p_len), dtype=np.float64)
    pair_m2 = np.zeros((3, p_len), dtype=np.float64)

    if not independent_replicas:
        if use_dense:
            rewire_bipartite_maslov_sneppen_dense(edge_u, edge_v, adj, burn_in_factor * n_edges)
        else:
            rewire_bipartite_maslov_sneppen(edge_u, edge_v, offsets, burn_in_factor * n_edges)

    for r in range(n_null):
        if independent_replicas:
            if r > 0:
                if use_dense:
                    for k in range(n_edges):
                        adj[edge_u[k], edge_v[k]] = 0
                edge_v[:] = edge_v_orig
                if use_dense:
                    for k in range(n_edges):
                        adj[edge_u[k], edge_v[k]] = 1
            if use_dense:
                rewire_bipartite_maslov_sneppen_dense(edge_u, edge_v, adj, burn_in_factor * n_edges)
            else:
                rewire_bipartite_maslov_sneppen(edge_u, edge_v, offsets, burn_in_factor * n_edges)
        elif r > 0:
            if use_dense:
                rewire_bipartite_maslov_sneppen_dense(edge_u, edge_v, adj, step_factor * n_edges)
            else:
                rewire_bipartite_maslov_sneppen(edge_u, edge_v, offsets, step_factor * n_edges)

        vm, dc, _, gk, db, pcb, pdb = evaluate_pair_motifs_numba(
            offsets, edge_v, n_v, pair_u, pair_w, ppi_mat, dup_mat, use_ppi, use_dup
        )

        if n_wgd > 0:
            sv = sdc = sgk = sdb = spcb = spdb = 0.0
            for p in range(n_wgd):
                sv += vm[p]
                sdc += dc[p]
                sgk += gk[p]
                sdb += db[p]
                spcb += pcb[p]
                spdb += pdb[p]
            null_wgd[0, r] = sv
            null_wgd[1, r] = sdc / n_wgd
            null_wgd[2, r] = sgk
            null_wgd[3, r] = sdb
            null_wgd[4, r] = spcb
            null_wgd[5, r] = spdb

        if n_ssd > 0:
            sv = sdc = sgk = sdb = spcb = spdb = 0.0
            for p in range(n_wgd, n_total):
                sv += vm[p]
                sdc += dc[p]
                sgk += gk[p]
                sdb += db[p]
                spcb += pcb[p]
                spdb += pdb[p]
            null_ssd[0, r] = sv
            null_ssd[1, r] = sdc / n_ssd
            null_ssd[2, r] = sgk
            null_ssd[3, r] = sdb
            null_ssd[4, r] = spcb
            null_ssd[5, r] = spdb

        inv_cnt = 1.0 / (r + 1)
        if track_wgd and n_wgd > 0:
            for p in range(n_wgd):
                fv = float(vm[p])
                dv = fv - pair_mean[0, p]
                pair_mean[0, p] += dv * inv_cnt
                pair_m2[0, p] += dv * (fv - pair_mean[0, p])

                fc = float(pcb[p])
                dc_ = fc - pair_mean[1, p]
                pair_mean[1, p] += dc_ * inv_cnt
                pair_m2[1, p] += dc_ * (fc - pair_mean[1, p])

                fd = float(pdb[p])
                dd = fd - pair_mean[2, p]
                pair_mean[2, p] += dd * inv_cnt
                pair_m2[2, p] += dd * (fd - pair_mean[2, p])

        if track_ssd and n_ssd > 0:
            for p in range(n_wgd, n_total):
                fv = float(vm[p])
                dv = fv - pair_mean[0, p]
                pair_mean[0, p] += dv * inv_cnt
                pair_m2[0, p] += dv * (fv - pair_mean[0, p])

                fc = float(pcb[p])
                dc_ = fc - pair_mean[1, p]
                pair_mean[1, p] += dc_ * inv_cnt
                pair_m2[1, p] += dc_ * (fc - pair_mean[1, p])

                fd = float(pdb[p])
                dd = fd - pair_mean[2, p]
                pair_mean[2, p] += dd * inv_cnt
                pair_m2[2, p] += dd * (fd - pair_mean[2, p])

    return null_wgd, null_ssd, pair_mean, pair_m2


@njit
def run_null_ensemble_ppi_delta_multi_numba(
    edge_u: np.ndarray,
    edge_v_orig: np.ndarray,
    offsets: np.ndarray,
    n_v: int,
    all_nodes: np.ndarray,
    cohort_offsets: np.ndarray,
    ppi_mat: np.ndarray,
    dup_mat: np.ndarray,
    use_dup: bool,
    n_null: int,
    burn_in_factor: int,
    step_factor: int,
    seed: int,
    independent_replicas: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble nullo condiviso per PPI-Complex Delta e PPI-Dup Delta su tutte le coorti Via 3."""
    _seed_numba(seed)
    n_edges = edge_u.shape[0]
    n_u = offsets.shape[0] - 1
    n_cohorts = cohort_offsets.shape[0] - 1

    edge_v = edge_v_orig.copy()
    use_dense = (n_u * n_v) <= 20_000_000
    if use_dense:
        adj = np.zeros((n_u, n_v), dtype=np.uint8)
        for k in range(n_edges):
            adj[edge_u[k], edge_v[k]] = 1
    else:
        adj = np.zeros((1, 1), dtype=np.uint8)

    null_complex = np.zeros((n_cohorts, n_null), dtype=np.float64)
    null_dup = np.zeros((n_cohorts, n_null), dtype=np.float64)

    if not independent_replicas:
        if use_dense:
            rewire_bipartite_maslov_sneppen_dense(edge_u, edge_v, adj, burn_in_factor * n_edges)
        else:
            rewire_bipartite_maslov_sneppen(edge_u, edge_v, offsets, burn_in_factor * n_edges)

    for r in range(n_null):
        if independent_replicas:
            if r > 0:
                if use_dense:
                    for k in range(n_edges):
                        adj[edge_u[k], edge_v[k]] = 0
                edge_v[:] = edge_v_orig
                if use_dense:
                    for k in range(n_edges):
                        adj[edge_u[k], edge_v[k]] = 1
            if use_dense:
                rewire_bipartite_maslov_sneppen_dense(edge_u, edge_v, adj, burn_in_factor * n_edges)
            else:
                rewire_bipartite_maslov_sneppen(edge_u, edge_v, offsets, burn_in_factor * n_edges)
        elif r > 0:
            if use_dense:
                rewire_bipartite_maslov_sneppen_dense(edge_u, edge_v, adj, step_factor * n_edges)
            else:
                rewire_bipartite_maslov_sneppen(edge_u, edge_v, offsets, step_factor * n_edges)

        cnt_c, cnt_d = evaluate_single_node_ppi_delta_numba(
            offsets, edge_v, all_nodes, ppi_mat, dup_mat, use_dup
        )
        for c in range(n_cohorts):
            s = cohort_offsets[c]
            e = cohort_offsets[c + 1]
            sc = sd = 0.0
            for idx in range(s, e):
                sc += cnt_c[idx]
                sd += cnt_d[idx]
            null_complex[c, r] = sc
            null_dup[c, r] = sd

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

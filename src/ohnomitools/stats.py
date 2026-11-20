# src/ohnomitools/stats.py
from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True)
def _seed_numba(seed: int) -> None:
    np.random.seed(seed)


@njit(cache=True)
def rewire_bipartite_maslov_sneppen(
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    offsets: np.ndarray,
    n_attempts: int,
) -> int:
    """
    Algoritmo microcanonico di Maslov-Sneppen (double-edge swap) su grafo bipartito.
    `edge_u` deve essere ordinato in modo crescente per nodo sorgente `u`, così che
    `edge_v[offsets[u]:offsets[u+1]]` contenga sempre esattamente i vicini correnti di `u`.
    Preserva esattamente i gradi di tutti i nodi sorgente e di tutti i nodi bersaglio.
    Restituisce il numero di scambi accettati.
    """
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

        # Verifica che (u1, v2) non esista già
        s1 = offsets[u1]
        end1 = offsets[u1 + 1]
        dup = False
        for idx in range(s1, end1):
            if edge_v[idx] == v2:
                dup = True
                break
        if dup:
            continue

        # Verifica che (u2, v1) non esista già
        s2 = offsets[u2]
        end2 = offsets[u2 + 1]
        for idx in range(s2, end2):
            if edge_v[idx] == v1:
                dup = True
                break
        if dup:
            continue

        # Scambio in-place O(1)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcola in un singolo passaggio O(sum(d_u + d_w)) per ciascuna coppia (u, w):
    - v_motifs: numero di partner condivisi |N(u) ∩ N(w)|
    - dice: coefficiente di Sørensen-Dice = 2|N(u)∩N(w)| / (|N(u)| + |N(w)|)
    - jaccard: indice di Jaccard = |N(u)∩N(w)| / |N(u)∪N(w)|
    - simple_bifans: C(|N(u)∩N(w)|, 2)
    - dup_bifans: coppie di partner condivisi che sono anche paraloghi tra loro
    - ppi_bifans: coppie di partner condivisi che interagiscono fisicamente in STRING
    """
    n_pairs = pair_u.shape[0]
    v_motifs = np.zeros(n_pairs, dtype=np.int64)
    dice = np.zeros(n_pairs, dtype=np.float64)
    jaccard = np.zeros(n_pairs, dtype=np.float64)
    simple_bifans = np.zeros(n_pairs, dtype=np.int64)
    dup_bifans = np.zeros(n_pairs, dtype=np.int64)
    ppi_bifans = np.zeros(n_pairs, dtype=np.int64)

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
                simple_bifans[p] = (c * (c - 1)) // 2
                if use_ppi or use_dup:
                    pb = 0
                    db = 0
                    for a in range(c - 1):
                        ta = common_buf[a]
                        for b in range(a + 1, c):
                            tb = common_buf[b]
                            if use_ppi and ppi_mat[ta, tb] == 1:
                                pb += 1
                            if use_dup and dup_mat[ta, tb] == 1:
                                db += 1
                    ppi_bifans[p] = pb
                    dup_bifans[p] = db

    return v_motifs, dice, jaccard, simple_bifans, dup_bifans, ppi_bifans


@njit(cache=True)
def evaluate_single_node_ppi_delta_numba(
    offsets: np.ndarray,
    edge_v: np.ndarray,
    nodes: np.ndarray,
    ppi_mat: np.ndarray,
) -> np.ndarray:
    """
    Calcola il numero di motivi PPI-Delta per ciascun singolo lncRNA `u` in `nodes`:
    numero di coppie di bersagli (t_a, t_b) legati da `u` che interagiscono fisicamente in STRING.
    """
    n_nodes = nodes.shape[0]
    out = np.zeros(n_nodes, dtype=np.int64)
    for idx in range(n_nodes):
        u = nodes[idx]
        s = offsets[u]
        e = offsets[u + 1]
        d = e - s
        if d < 2:
            continue
        cnt = 0
        for i in range(s, e - 1):
            ta = edge_v[i]
            for j in range(i + 1, e):
                if ppi_mat[ta, edge_v[j]] == 1:
                    cnt += 1
        out[idx] = cnt
    return out


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Esegue un ensemble di `n_null` reti randomizzate di Maslov-Sneppen (con burn-in
    iniziale `burn_in_factor * E` e thinning `step_factor * E` tra repliche).
    Restituisce:
    - distribuzioni nulle globali (shape `(n_null,)`): V-motifs, Mean Dice, Simple Bifans, Dup Bifans, PPI Bifans
    - somme e somme dei quadrati per singola coppia (per Z-score per coppia stile Agasso Fig. 3) se `track_per_pair=True`.
    """
    _seed_numba(seed)
    n_edges = edge_u.shape[0]
    n_pairs = pair_u.shape[0]

    edge_v = edge_v_orig.copy()

    null_v = np.zeros(n_null, dtype=np.float64)
    null_dice = np.zeros(n_null, dtype=np.float64)
    null_sb = np.zeros(n_null, dtype=np.float64)
    null_db = np.zeros(n_null, dtype=np.float64)
    null_pb = np.zeros(n_null, dtype=np.float64)

    p_len = n_pairs if track_per_pair else 1
    pair_v_sum = np.zeros(p_len, dtype=np.float64)
    pair_v_sq = np.zeros(p_len, dtype=np.float64)
    pair_pb_sum = np.zeros(p_len, dtype=np.float64)
    pair_pb_sq = np.zeros(p_len, dtype=np.float64)

    # Burn-in iniziale della catena di Markov
    rewire_bipartite_maslov_sneppen(edge_u, edge_v, offsets, burn_in_factor * n_edges)

    for r in range(n_null):
        if r > 0:
            rewire_bipartite_maslov_sneppen(edge_u, edge_v, offsets, step_factor * n_edges)

        vm, dc, _, sb, db, pb = evaluate_pair_motifs_numba(
            offsets, edge_v, n_v, pair_u, pair_w, ppi_mat, dup_mat, use_ppi, use_dup
        )

        null_v[r] = float(vm.sum())
        null_dice[r] = float(dc.mean()) if n_pairs > 0 else 0.0
        null_sb[r] = float(sb.sum())
        null_db[r] = float(db.sum())
        null_pb[r] = float(pb.sum())

        if track_per_pair:
            for p in range(n_pairs):
                fv = float(vm[p])
                fpb = float(pb[p])
                pair_v_sum[p] += fv
                pair_v_sq[p] += fv * fv
                pair_pb_sum[p] += fpb
                pair_pb_sq[p] += fpb * fpb

    return (
        null_v, null_dice, null_sb, null_db, null_pb,
        pair_v_sum, pair_v_sq, pair_pb_sum, pair_pb_sq,
    )


@njit(cache=True)
def run_null_ensemble_ppi_delta_numba(
    edge_u: np.ndarray,
    edge_v_orig: np.ndarray,
    offsets: np.ndarray,
    nodes: np.ndarray,
    ppi_mat: np.ndarray,
    n_null: int,
    burn_in_factor: int,
    step_factor: int,
    seed: int,
) -> np.ndarray:
    """
    Ensemble nullo di Maslov-Sneppen per il totale di motivi PPI-Delta su una coorte di nodi (Via 3).
    """
    _seed_numba(seed)
    n_edges = edge_u.shape[0]
    edge_v = edge_v_orig.copy()
    null_totals = np.zeros(n_null, dtype=np.float64)

    rewire_bipartite_maslov_sneppen(edge_u, edge_v, offsets, burn_in_factor * n_edges)
    for r in range(n_null):
        if r > 0:
            rewire_bipartite_maslov_sneppen(edge_u, edge_v, offsets, step_factor * n_edges)
        counts = evaluate_single_node_ppi_delta_numba(offsets, edge_v, nodes, ppi_mat)
        null_totals[r] = float(counts.sum())

    return null_totals


def compute_zscore(obs: float, null_arr: np.ndarray) -> tuple[float, float, float, float]:
    """Restituisce (null_mean, null_std, z_score, emp_p_value_two_sided)."""
    mean = float(np.mean(null_arr))
    std = float(np.std(null_arr, ddof=1)) if len(null_arr) > 1 else 0.0
    z = (obs - mean) / std if std > 1e-12 else (0.0 if abs(obs - mean) < 1e-12 else float(np.sign(obs - mean) * 999.0))
    p_ge = (float(np.sum(null_arr >= obs)) + 1.0) / (len(null_arr) + 1.0)
    p_le = (float(np.sum(null_arr <= obs)) + 1.0) / (len(null_arr) + 1.0)
    p_emp = min(1.0, 2.0 * min(p_ge, p_le))
    return mean, std, z, p_emp

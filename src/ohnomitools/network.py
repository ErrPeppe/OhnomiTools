# src/ohnomitools/network.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging
import numpy as np
import polars as pl
from scipy.stats import ks_2samp, mannwhitneyu

from .config import PATHS, ProjectPaths
from .stats import (
    bootstrap_mean_ci,
    compute_zscore,
    evaluate_pair_motifs_numba,
    evaluate_single_node_ppi_delta_numba,
    run_null_ensemble_pairs_numba,
    run_null_ensemble_ppi_delta_multi_numba,
)

logger = logging.getLogger("OhnomiTools.network")


def _compute_pair_z_array(
    obs: np.ndarray,
    null_mean: np.ndarray,
    null_m2: np.ndarray,
    n_null: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Z-score per coppia con gestione esatta di sigma_null == 0 (Agasso et al., 2025):
    NaN se sigma_null == 0 e obs != mean; 0.0 solo se sigma_null == 0 e obs == mean.
    """
    std = np.sqrt(np.clip(null_m2 / max(1, n_null - 1), 0.0, None))
    z = np.full(len(obs), np.nan, dtype=np.float64)
    has_var = std > 1e-12
    zero_both = (~has_var) & (np.abs(obs - null_mean) <= 1e-12)
    z[has_var] = (obs[has_var] - null_mean[has_var]) / std[has_var]
    z[zero_both] = 0.0
    undefined_mask = (~has_var) & (~zero_both)
    return z, undefined_mask


@dataclass
class BipartiteNetwork:
    """Rappresentazione CSR compatta di un grafo bipartito (lncRNA -> target)."""
    layer_name: str
    u_names: list[str]
    v_names: list[str]
    u_to_idx: dict[str, int]
    v_to_idx: dict[str, int]
    edge_u: np.ndarray
    edge_v: np.ndarray
    offsets: np.ndarray
    degrees_u: np.ndarray
    degrees_v: np.ndarray
    ppi_mat: np.ndarray
    dup_mat: np.ndarray
    use_ppi: bool
    use_dup: bool
    n_ppi_edges_mapped: int
    n_dup_target_edges_mapped: int


class NetworkMotifAnalyzer:
    """Fase 3: gradi k_out, ridondanza, 4 famiglie di motivi e modelli nulli Maslov-Sneppen."""

    def __init__(self, paths: ProjectPaths = PATHS) -> None:
        self.paths = paths

    def _load_bipartite_edges(
        self,
        layer: str,
        min_rbp_sites: int = 1,
        min_rbp_cell_lines: int = 1,
    ) -> pl.DataFrame:
        if layer == "rbp":
            pq_path = self.paths.silver_lncbook_rbp_int
            if not pq_path.exists():
                raise FileNotFoundError(f"File Silver RBP mancante: {pq_path}")
            lf = pl.scan_parquet(pq_path)
            cols = lf.collect_schema().names()
            gene_col = "Gene ID" if "Gene ID" in cols else cols[0]
            prot_col = "Protein" if "Protein" in cols else cols[3]
            cell_col = "Cell Line" if "Cell Line" in cols else None

            aggs = [pl.len().alias("n_sites")]
            if cell_col:
                aggs.append(pl.col(cell_col).n_unique().alias("n_cell_lines"))

            df = (
                lf.select([
                    pl.col(gene_col).cast(pl.Utf8).str.strip_chars().alias("lncrna_id"),
                    pl.col(prot_col).cast(pl.Utf8).str.strip_chars().alias("target_id"),
                    *([pl.col(cell_col).cast(pl.Utf8)] if cell_col else []),
                ])
                .filter(
                    pl.col("lncrna_id").is_not_null()
                    & pl.col("target_id").is_not_null()
                    & (pl.col("lncrna_id") != "")
                    & (pl.col("target_id") != "")
                    & (pl.col("target_id") != "-")
                )
                .group_by(["lncrna_id", "target_id"])
                .agg(aggs)
                .filter(pl.col("n_sites") >= min_rbp_sites)
                .collect()
            )
            if cell_col and min_rbp_cell_lines > 1:
                df = df.filter(pl.col("n_cell_lines") >= min_rbp_cell_lines)
            return df.select(["lncrna_id", "target_id", "n_sites"])

        if layer == "mirna":
            pq_path = self.paths.silver_lncbook_mirna_int
            if not pq_path.exists():
                raise FileNotFoundError(f"File Silver miRNA mancante: {pq_path}")
            df_head = pl.read_parquet(pq_path, n_rows=20)
            cols = df_head.columns

            if "gene_id" in cols and "mirna_id" in cols:
                gene_col, mir_col = "gene_id", "mirna_id"
            else:
                gene_col = next(
                    (c for c in cols if df_head[c].cast(pl.Utf8).str.starts_with("HSALNG").any()),
                    cols[0],
                )
                mir_col = next(
                    (
                        c for c in cols
                        if df_head[c].cast(pl.Utf8).str.contains(r"(?i)^hsa-(mir|let)-").any()
                    ),
                    cols[-1],
                )

            return (
                pl.scan_parquet(pq_path)
                .select([
                    pl.col(gene_col).cast(pl.Utf8).str.strip_chars().alias("lncrna_id"),
                    pl.col(mir_col).cast(pl.Utf8).str.strip_chars().alias("target_id"),
                ])
                .filter(
                    pl.col("lncrna_id").is_not_null()
                    & pl.col("target_id").is_not_null()
                    & (pl.col("lncrna_id") != "")
                    & (pl.col("target_id") != "")
                    & (pl.col("target_id") != "-")
                )
                .group_by(["lncrna_id", "target_id"])
                .agg(pl.len().alias("n_sites"))
                .collect()
            )

        raise ValueError(f"Layer non riconosciuto: '{layer}'. Usa 'rbp' oppure 'mirna'.")

    def build_network(
        self,
        layer: str = "rbp",
        ppi_min_score: int = 700,
        min_rbp_sites: int = 1,
        min_rbp_cell_lines: int = 1,
    ) -> BipartiteNetwork:
        df_edges = self._load_bipartite_edges(
            layer=layer,
            min_rbp_sites=min_rbp_sites,
            min_rbp_cell_lines=min_rbp_cell_lines,
        )

        u_names = sorted(df_edges["lncrna_id"].unique().to_list())
        v_names = sorted(df_edges["target_id"].unique().to_list())
        u_to_idx = {name: i for i, name in enumerate(u_names)}
        v_to_idx = {name: i for i, name in enumerate(v_names)}
        v_to_idx_upper = {name.upper(): i for i, name in enumerate(v_names)}

        u_map_df = pl.DataFrame({"lncrna_id": u_names, "u_idx": np.arange(len(u_names), dtype=np.int32)})
        v_map_df = pl.DataFrame({"target_id": v_names, "v_idx": np.arange(len(v_names), dtype=np.int32)})

        df_idx = (
            df_edges.join(u_map_df, on="lncrna_id", how="inner")
            .join(v_map_df, on="target_id", how="inner")
            .sort(["u_idx", "v_idx"])
        )

        edge_u = df_idx["u_idx"].to_numpy().astype(np.int32)
        edge_v = df_idx["v_idx"].to_numpy().astype(np.int32)

        n_u = len(u_names)
        n_v = len(v_names)
        degrees_u = np.bincount(edge_u, minlength=n_u).astype(np.int32)
        degrees_v = np.bincount(edge_v, minlength=n_v).astype(np.int32)
        offsets = np.zeros(n_u + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(degrees_u)

        use_ppi = False
        use_dup = False
        n_ppi_mapped = 0
        n_dup_mapped = 0
        ppi_mat = np.zeros((1, 1), dtype=np.uint8)
        dup_mat = np.zeros((1, 1), dtype=np.uint8)

        if layer == "rbp" and n_v <= 6000:
            if self.paths.silver_string_ppi.exists():
                ppi_mat = np.zeros((n_v, n_v), dtype=np.uint8)
                df_ppi = (
                    pl.scan_parquet(self.paths.silver_string_ppi)
                    .filter(pl.col("combined_score") >= ppi_min_score)
                    .select(["protein_1", "protein_2"])
                    .collect()
                )
                for p1, p2 in df_ppi.iter_rows():
                    i1 = v_to_idx.get(p1)
                    if i1 is None and p1:
                        i1 = v_to_idx_upper.get(p1.upper())
                    i2 = v_to_idx.get(p2)
                    if i2 is None and p2:
                        i2 = v_to_idx_upper.get(p2.upper())
                    if i1 is not None and i2 is not None and i1 != i2 and ppi_mat[i1, i2] == 0:
                        ppi_mat[i1, i2] = 1
                        ppi_mat[i2, i1] = 1
                        n_ppi_mapped += 1
                use_ppi = n_ppi_mapped > 0

            dup_mat = np.zeros((n_v, n_v), dtype=np.uint8)
            for dup_pq in [self.paths.silver_wgd_pairs, self.paths.silver_ssd_pairs]:
                if dup_pq.exists():
                    df_d = pl.read_parquet(dup_pq).select(["gene_name_1", "gene_name_2"]).drop_nulls()
                    for g1, g2 in df_d.iter_rows():
                        i1 = v_to_idx.get(g1)
                        if i1 is None and g1:
                            i1 = v_to_idx_upper.get(g1.upper())
                        i2 = v_to_idx.get(g2)
                        if i2 is None and g2:
                            i2 = v_to_idx_upper.get(g2.upper())
                        if i1 is not None and i2 is not None and i1 != i2 and dup_mat[i1, i2] == 0:
                            dup_mat[i1, i2] = 1
                            dup_mat[i2, i1] = 1
                            n_dup_mapped += 1
            use_dup = n_dup_mapped > 0

        logger.info(
            f"Rete [{layer.upper()}] costruita: {n_u:,} lncRNA | {n_v:,} target | "
            f"{len(edge_u):,} archi unici | PPI STRING (>={ppi_min_score}): {n_ppi_mapped:,} | "
            f"Target paraloghi: {n_dup_mapped:,}"
        )

        return BipartiteNetwork(
            layer_name=layer,
            u_names=u_names,
            v_names=v_names,
            u_to_idx=u_to_idx,
            v_to_idx=v_to_idx,
            edge_u=edge_u,
            edge_v=edge_v,
            offsets=offsets,
            degrees_u=degrees_u,
            degrees_v=degrees_v,
            ppi_mat=ppi_mat,
            dup_mat=dup_mat,
            use_ppi=use_ppi,
            use_dup=use_dup,
            n_ppi_edges_mapped=n_ppi_mapped,
            n_dup_target_edges_mapped=n_dup_mapped,
        )

    def _load_lane_pairs(
        self,
        lane: str,
        dup_type: str,
        strict_1to1_only: bool = False,
        dedup_lncrna_pair_key: bool = True,
    ) -> pl.DataFrame:
        if lane == "via1":
            pq = self.paths.silver_lane1_wgd_pairs if dup_type == "WGD" else self.paths.silver_lane1_ssd_pairs
            df = pl.read_parquet(pq)
            if dup_type == "SSD" and "is_ancient_ssd" in df.columns:
                df = df.filter(pl.col("is_ancient_ssd") == True)
        elif lane == "via2":
            pq = self.paths.silver_lane2_wgd_pairs if dup_type == "WGD" else self.paths.silver_lane2_ssd_pairs
            df = pl.read_parquet(pq)
        else:
            raise ValueError(f"Lane non valida: '{lane}' (usa 'via1' o 'via2').")

        if strict_1to1_only and "is_strict_1to1_host" in df.columns:
            df = df.filter(pl.col("is_strict_1to1_host") == True)

        if dedup_lncrna_pair_key and "lncrna_pair_key" in df.columns:
            df = df.unique(subset=["lncrna_pair_key"])

        return df

    def analyze_degree_distributions(self, net: BipartiteNetwork) -> dict[str, pl.DataFrame]:
        """Distribuzioni k_out sulle 3 Vie con disgiunzione stretta WGD vs SSD."""
        df_deg = pl.DataFrame({
            "lncrna_id": net.u_names,
            "k_out": net.degrees_u.astype(np.int64),
        })

        def _extract_ids(df_p: pl.DataFrame) -> set[str]:
            return set(df_p["lncrna_1"].to_list()) | set(df_p["lncrna_2"].to_list())

        wgd1_ids = _extract_ids(self._load_lane_pairs("via1", "WGD"))
        ssd1_raw = _extract_ids(self._load_lane_pairs("via1", "SSD"))
        ssd1_ids = ssd1_raw - wgd1_ids

        wgd2_ids = _extract_ids(self._load_lane_pairs("via2", "WGD"))
        ssd2_raw = _extract_ids(self._load_lane_pairs("via2", "SSD"))
        ssd2_ids = ssd2_raw - wgd2_ids

        logger.info(
            f"[{net.layer_name.upper()}] Disgiunzione WGD/SSD: rimossi {len(ssd1_raw) - len(ssd1_ids):,} "
            f"geni sovrapposti da SSD Via 1 e {len(ssd2_raw) - len(ssd2_ids):,} da SSD Via 2."
        )

        df_via3 = pl.read_parquet(self.paths.silver_lane3_euteleostomi)
        df_phylo = pl.read_parquet(self.paths.silver_phylostratigraphy)

        eut_wgd_ids = set(df_via3.filter(pl.col("genomic_context") == "intragenic_WGD_host")["lncrna_gene_id"].to_list())
        eut_oth_ids = set(df_via3.filter(pl.col("genomic_context") == "intragenic_other_host")["lncrna_gene_id"].to_list())
        eut_linc_ids = set(df_via3.filter(pl.col("genomic_context") == "intergenic_lincRNA")["lncrna_gene_id"].to_list())
        eutheria_ids = set(df_phylo.filter(pl.col("age_node_num") == 12)["lncrna_gene_id"].to_list())
        homo_ids = set(df_phylo.filter(pl.col("age_node_num") == 1)["lncrna_gene_id"].to_list())
        all_net_ids = set(net.u_names)

        cohorts: list[tuple[str, str, set[str]]] = [
            ("Via 1 (Ramo A)", "WGD_intragenic", wgd1_ids),
            ("Via 1 (Ramo A)", "SSD_ancient_intragenic", ssd1_ids),
            ("Via 2 (A ∩ B >=15)", "WGD_AB_ancient", wgd2_ids),
            ("Via 2 (A ∩ B >=15)", "SSD_AB_ancient", ssd2_ids),
            ("Via 3 (Ramo B Eut.)", "Eut_intragenic_WGD", eut_wgd_ids),
            ("Via 3 (Ramo B Eut.)", "Eut_intragenic_other", eut_oth_ids),
            ("Via 3 (Ramo B Eut.)", "Eut_intergenic_lincRNA", eut_linc_ids),
            ("Controllo Età", "12. Eutheria", eutheria_ids),
            ("Controllo Età", "1. Homo", homo_ids),
            ("Background", "All_lncRNAs_in_net", all_net_ids),
        ]

        summary_rows = []
        deg_arrays: dict[str, np.ndarray] = {}

        for lane_lbl, grp_lbl, id_set in cohorts:
            sub = df_deg.filter(pl.col("lncrna_id").is_in(list(id_set)))
            arr = sub["k_out"].to_numpy()
            deg_arrays[grp_lbl] = arr
            n_tot = len(id_set)
            n_in_net = len(arr)
            summary_rows.append({
                "Layer": net.layer_name.upper(),
                "Via": lane_lbl,
                "Coorte": grp_lbl,
                "N_geni_coorte": n_tot,
                "N_in_rete": n_in_net,
                "Copertura_pct": round(100.0 * n_in_net / n_tot, 2) if n_tot else 0.0,
                "k_mediana": float(np.median(arr)) if n_in_net else 0.0,
                "k_media": round(float(np.mean(arr)), 3) if n_in_net else 0.0,
                "k_p90": round(float(np.quantile(arr, 0.90)), 2) if n_in_net else 0.0,
                "k_max": int(np.max(arr)) if n_in_net else 0,
            })

        comparisons = [
            ("Via 1: WGD vs SSD_ancient", "WGD_intragenic", "SSD_ancient_intragenic"),
            ("Via 2 (A∩B): WGD vs SSD_ancient", "WGD_AB_ancient", "SSD_AB_ancient"),
            ("Via 3: Eut_WGD vs Eut_other", "Eut_intragenic_WGD", "Eut_intragenic_other"),
            ("Via 3: Eut_WGD vs Eut_lincRNA", "Eut_intragenic_WGD", "Eut_intergenic_lincRNA"),
            ("Via 3: Eut_lincRNA vs 1. Homo", "Eut_intergenic_lincRNA", "1. Homo"),
            ("Via 3: Eut_WGD vs 12. Eutheria", "Eut_intragenic_WGD", "12. Eutheria"),
        ]
        test_rows = []
        for label, g1, g2 in comparisons:
            a1, a2 = deg_arrays[g1], deg_arrays[g2]
            if len(a1) >= 5 and len(a2) >= 5:
                ks_stat, ks_p = ks_2samp(a1, a2)
                _, mw_p = mannwhitneyu(a1, a2, alternative="two-sided")
            else:
                ks_stat, ks_p, mw_p = float("nan"), float("nan"), float("nan")
            test_rows.append({
                "Layer": net.layer_name.upper(),
                "Confronto": label,
                "Mediana_1": float(np.median(a1)) if len(a1) else 0.0,
                "Mediana_2": float(np.median(a2)) if len(a2) else 0.0,
                "Media_1": round(float(np.mean(a1)), 3) if len(a1) else 0.0,
                "Media_2": round(float(np.mean(a2)), 3) if len(a2) else 0.0,
                "KS_stat": round(float(ks_stat), 4),
                "KS_p_value": float(ks_p),
                "MW_p_value": float(mw_p),
            })

        df_summary = pl.DataFrame(summary_rows)
        df_tests = pl.DataFrame(test_rows)
        df_summary.write_csv(self.paths.gold / f"phase3_{net.layer_name}_degree_summary.csv")
        df_tests.write_csv(self.paths.gold / f"phase3_{net.layer_name}_degree_tests.csv")
        return {"summary": df_summary, "tests": df_tests}

    def analyze_pair_motifs_and_null_models(
        self,
        net: BipartiteNetwork,
        lane: str = "via2",
        n_null: int = 1000,
        burn_in_factor: int = 10,
        step_factor: int = 10,
        seed: int = 42,
        strict_1to1_only: bool = False,
        max_ssd_pairs_for_null: int | None = None,
        independent_replicas: bool = False,
        save_outputs: bool = True,
        output_tag: str | None = None,
    ) -> dict[str, pl.DataFrame | dict]:
        """Analisi motivi a coppie e modello nullo condiviso WGD + SSD (4 famiglie Bifans)."""
        prepped: dict[str, tuple[pl.DataFrame, np.ndarray, np.ndarray, int]] = {}

        for dup_type in ("WGD", "SSD"):
            df_raw = self._load_lane_pairs(lane, dup_type, strict_1to1_only=strict_1to1_only)
            n_total_pairs = len(df_raw)

            u_idx_list = []
            w_idx_list = []
            keep_mask = []
            for r1, r2 in df_raw.select(["lncrna_1", "lncrna_2"]).iter_rows():
                i1 = net.u_to_idx.get(r1)
                i2 = net.u_to_idx.get(r2)
                if i1 is not None and i2 is not None and i1 != i2:
                    u_idx_list.append(i1)
                    w_idx_list.append(i2)
                    keep_mask.append(True)
                else:
                    keep_mask.append(False)

            df_in = df_raw.filter(pl.Series(keep_mask))
            if dup_type == "SSD" and max_ssd_pairs_for_null is not None and len(df_in) > max_ssd_pairs_for_null:
                rng = np.random.default_rng(seed)
                sample_idx = np.sort(rng.choice(len(df_in), size=max_ssd_pairs_for_null, replace=False))
                df_in = df_in[sample_idx]
                u_idx_list = [u_idx_list[i] for i in sample_idx]
                w_idx_list = [w_idx_list[i] for i in sample_idx]

            pair_u = np.array(u_idx_list, dtype=np.int32)
            pair_w = np.array(w_idx_list, dtype=np.int32)
            if len(pair_u) == 0:
                logger.warning(f"[{net.layer_name.upper()} | {lane} | {dup_type}] Nessuna coppia in rete.")
            prepped[dup_type] = (df_in, pair_u, pair_w, n_total_pairs)

        wgd_u, wgd_w = prepped["WGD"][1], prepped["WGD"][2]
        ssd_u, ssd_w = prepped["SSD"][1], prepped["SSD"][2]
        n_wgd = len(wgd_u)
        n_ssd = len(ssd_u)

        if n_wgd == 0 and n_ssd == 0:
            return {
                "z_summary": pl.DataFrame(),
                "wgd_vs_ssd": pl.DataFrame(),
                "pair_tables": {},
                "null_dists": {},
                "undefined_z_pairs": {},
            }

        all_u = np.concatenate([wgd_u, ssd_u]).astype(np.int32)
        all_w = np.concatenate([wgd_w, ssd_w]).astype(np.int32)

        obs_all = evaluate_pair_motifs_numba(
            net.offsets, net.edge_v, len(net.v_names),
            all_u, all_w, net.ppi_mat, net.dup_mat, net.use_ppi, net.use_dup
        )

        track_wgd = 0 < n_wgd <= 50_000
        track_ssd = 0 < n_ssd <= 50_000
        null_wgd, null_ssd, pair_mean, pair_m2 = run_null_ensemble_pairs_numba(
            net.edge_u, net.edge_v, net.offsets, len(net.v_names),
            all_u, all_w, n_wgd,
            net.ppi_mat, net.dup_mat, net.use_ppi, net.use_dup,
            n_null, burn_in_factor, step_factor, seed,
            track_wgd, track_ssd, independent_replicas,
        )

        results_rows: list[dict] = []
        pair_tables: dict[str, pl.DataFrame] = {}
        null_dists: dict[str, dict[str, np.ndarray]] = {}
        undefined_z_pairs: dict[str, pl.DataFrame] = {}

        slices = [
            ("WGD", 0, n_wgd, null_wgd, track_wgd),
            ("SSD", n_wgd, n_wgd + n_ssd, null_ssd, track_ssd),
        ]

        for dup_type, s_idx, e_idx, null_mat, is_tracked in slices:
            n_in_net = e_idx - s_idx
            if n_in_net == 0:
                continue

            df_in, pair_u, pair_w, n_total_pairs = prepped[dup_type]
            obs_v = obs_all[0][s_idx:e_idx]
            obs_dice = obs_all[1][s_idx:e_idx]
            obs_jacc = obs_all[2][s_idx:e_idx]
            obs_gk = obs_all[3][s_idx:e_idx]
            obs_db = obs_all[4][s_idx:e_idx]
            obs_pcb = obs_all[5][s_idx:e_idx]
            obs_pdb = obs_all[6][s_idx:e_idx]

            deg1 = net.degrees_u[pair_u].astype(np.int64)
            deg2 = net.degrees_u[pair_w].astype(np.int64)
            delta_deg = np.abs(deg1 - deg2)
            norm_delta_deg = delta_deg / (deg1 + deg2)

            null_dists[dup_type] = {
                "v_motifs": null_mat[0],
                "mean_dice": null_mat[1],
                "generic_k22_bifans": null_mat[2],
                "dup_target_bifans": null_mat[3],
                "ppi_complex_bifans": null_mat[4],
                "ppi_dup_bifans": null_mat[5],
            }

            if is_tracked:
                pair_v_z, undef_v = _compute_pair_z_array(
                    obs_v.astype(np.float64), pair_mean[0, s_idx:e_idx], pair_m2[0, s_idx:e_idx], n_null
                )
                pair_pcb_z, undef_pcb = _compute_pair_z_array(
                    obs_pcb.astype(np.float64), pair_mean[1, s_idx:e_idx], pair_m2[1, s_idx:e_idx], n_null
                )
                pair_pdb_z, undef_pdb = _compute_pair_z_array(
                    obs_pdb.astype(np.float64), pair_mean[2, s_idx:e_idx], pair_m2[2, s_idx:e_idx], n_null
                )
                any_undef = (
                    undef_v
                    | (undef_pcb if net.use_ppi else False)
                    | (undef_pdb if (net.use_ppi and net.use_dup) else False)
                )
            else:
                pair_v_z = np.full(n_in_net, np.nan)
                pair_pcb_z = np.full(n_in_net, np.nan)
                pair_pdb_z = np.full(n_in_net, np.nan)
                any_undef = np.zeros(n_in_net, dtype=bool)

            keep_cols = [
                c for c in [
                    "pair_key", "lncrna_pair_key", "gene_name_1", "gene_name_2",
                    "lncrna_1", "lncrna_2", "lncrna_age_1", "lncrna_age_2",
                    "rel_strand_1", "is_strict_1to1_host", "wgd_round"
                ] if c in df_in.columns
            ]
            df_pairs_out = df_in.select(keep_cols).with_columns([
                pl.Series("k_1", deg1),
                pl.Series("k_2", deg2),
                pl.Series("delta_k", delta_deg),
                pl.Series("norm_delta_k", np.round(norm_delta_deg, 4)),
                pl.Series("v_motifs", obs_v),
                pl.Series("sorensen_dice", np.round(obs_dice, 5)),
                pl.Series("jaccard", np.round(obs_jacc, 5)),
                pl.Series("generic_k22_bifans", obs_gk),
                pl.Series("dup_target_bifans", obs_db),
                pl.Series("ppi_complex_bifans", obs_pcb),
                pl.Series("ppi_dup_bifans", obs_pdb),
                pl.Series("v_motif_pair_z", np.round(pair_v_z, 3)),
                pl.Series("ppi_complex_bifan_pair_z", np.round(pair_pcb_z, 3)),
                pl.Series("ppi_dup_bifan_pair_z", np.round(pair_pdb_z, 3)),
            ])
            pair_tables[dup_type] = df_pairs_out

            df_undef = df_pairs_out.filter(pl.Series(any_undef))
            undefined_z_pairs[dup_type] = df_undef
            if len(df_undef) > 0:
                logger.info(
                    f"[{net.layer_name.upper()} | {lane} | {dup_type}] "
                    f"{len(df_undef)} coppie con sigma_null=0 e obs>0 impostate a NaN."
                )

            metric_specs = [
                ("V-Motifs (totale)", float(obs_v.sum()), null_mat[0], True),
                ("Mean Sørensen-Dice", float(obs_dice.mean()), null_mat[1], True),
                ("Generic K2,2 Bifans", float(obs_gk.sum()), null_mat[2], True),
                ("Duplicated-Target Bifans", float(obs_db.sum()), null_mat[3], net.use_dup),
                ("PPI-Complex Bifans (STRING)", float(obs_pcb.sum()), null_mat[4], net.use_ppi),
                ("PPI-Dup Bifans (STRING+Paralog)", float(obs_pdb.sum()), null_mat[5], net.use_ppi and net.use_dup),
            ]
            for m_name, obs_val, n_arr, active in metric_specs:
                if not active:
                    continue
                n_mean, n_std, z_val, p_emp = compute_zscore(obs_val, n_arr)
                results_rows.append({
                    "Layer": net.layer_name.upper(),
                    "Via": lane,
                    "Tipo": dup_type,
                    "Coppie_totali": n_total_pairs,
                    "Coppie_in_rete": n_in_net,
                    "Frac_V_pos_pct": round(100.0 * float(np.mean(obs_v > 0)), 2),
                    "Frac_PPI_Complex_pos_pct": round(100.0 * float(np.mean(obs_pcb > 0)), 2) if net.use_ppi else 0.0,
                    "Frac_PPI_Dup_pos_pct": round(100.0 * float(np.mean(obs_pdb > 0)), 2) if (net.use_ppi and net.use_dup) else 0.0,
                    "Metrica": m_name,
                    "Osservato": round(obs_val, 5),
                    "Null_Mean": round(n_mean, 5),
                    "Null_Std": round(n_std, 5),
                    "Z_score": round(z_val, 2) if np.isfinite(z_val) else float("nan"),
                    "p_empirico": round(p_emp, 5),
                })

        df_z_summary = pl.DataFrame(results_rows)
        suffix = f"{net.layer_name}_{lane}" + ("_1to1" if strict_1to1_only else "")
        if output_tag:
            suffix = f"{suffix}_{output_tag}"

        if save_outputs:
            df_z_summary.write_csv(self.paths.gold / f"phase3_{suffix}_null_zscores.csv")
            for dt, df_u in undefined_z_pairs.items():
                if len(df_u) > 0:
                    df_u.write_csv(self.paths.gold / f"phase3_{suffix}_{dt}_undefined_sigma0_pairs.csv")

        comp_rows = []
        if "WGD" in pair_tables and "SSD" in pair_tables:
            tw, ts = pair_tables["WGD"], pair_tables["SSD"]
            metric_cols = [
                ("sorensen_dice", True),
                ("jaccard", True),
                ("v_motifs", True),
                ("generic_k22_bifans", True),
                ("norm_delta_k", True),
                ("dup_target_bifans", net.use_dup),
                ("ppi_complex_bifans", net.use_ppi),
                ("ppi_dup_bifans", net.use_ppi and net.use_dup),
            ]
            for col, active in metric_cols:
                if not active:
                    continue
                aw = tw[col].to_numpy().astype(np.float64)
                as_ = ts[col].to_numpy().astype(np.float64)
                ks_s, ks_p = ks_2samp(aw, as_)
                _, mw_p = mannwhitneyu(aw, as_, alternative="two-sided")
                comp_rows.append({
                    "Layer": net.layer_name.upper(),
                    "Via": lane,
                    "Metrica": col,
                    "WGD_n": len(aw),
                    "WGD_media": round(float(np.mean(aw)), 5),
                    "WGD_mediana": round(float(np.median(aw)), 5),
                    "SSD_n": len(as_),
                    "SSD_media": round(float(np.mean(as_)), 5),
                    "SSD_mediana": round(float(np.median(as_)), 5),
                    "KS_stat": round(float(ks_s), 4),
                    "KS_p": float(ks_p),
                    "MW_p": float(mw_p),
                })

            if save_outputs:
                self.paths.save_parquet_safe(tw, self.paths.silver / f"phase3_{suffix}_wgd_pair_metrics.parquet")

        df_comp = pl.DataFrame(comp_rows)
        if save_outputs and not df_comp.is_empty():
            df_comp.write_csv(self.paths.gold / f"phase3_{suffix}_wgd_vs_ssd_divergence.csv")

        return {
            "z_summary": df_z_summary,
            "wgd_vs_ssd": df_comp,
            "pair_tables": pair_tables,
            "null_dists": null_dists,
            "undefined_z_pairs": undefined_z_pairs,
        }

    def compare_host_aggregated_pair_metrics(
        self,
        pair_tables: dict[str, pl.DataFrame],
        layer_name: str,
        lane: str,
        agg: str = "median",
    ) -> pl.DataFrame:
        """Controllo anti-pseudoreplicazione: 1 valore per singola pair_key di host."""
        if "WGD" not in pair_tables or "SSD" not in pair_tables:
            return pl.DataFrame()
        if agg not in {"median", "mean"}:
            raise ValueError("`agg` deve essere 'median' oppure 'mean'.")

        candidate_cols = [
            "sorensen_dice", "jaccard", "v_motifs",
            "generic_k22_bifans", "norm_delta_k",
            "ppi_complex_bifans", "ppi_dup_bifans",
        ]
        tw_raw, ts_raw = pair_tables["WGD"], pair_tables["SSD"]
        cols = [c for c in candidate_cols if c in tw_raw.columns and c in ts_raw.columns]
        agg_exprs = [
            (pl.col(c).median() if agg == "median" else pl.col(c).mean()).alias(c)
            for c in cols
        ]
        tw = tw_raw.group_by("pair_key").agg(agg_exprs)
        ts = ts_raw.group_by("pair_key").agg(agg_exprs)

        rows = []
        for col in cols:
            aw = tw[col].drop_nulls().to_numpy().astype(np.float64)
            as_ = ts[col].drop_nulls().to_numpy().astype(np.float64)
            if len(aw) < 3 or len(as_) < 3:
                continue
            ks_s, ks_p = ks_2samp(aw, as_)
            _, mw_p = mannwhitneyu(aw, as_, alternative="two-sided")
            rows.append({
                "Layer": layer_name.upper(),
                "Via": lane,
                "Aggregazione": f"host_pair_{agg}",
                "Metrica": col,
                "WGD_hosts_n": len(aw),
                "WGD_media": round(float(np.mean(aw)), 5),
                "WGD_mediana": round(float(np.median(aw)), 5),
                "SSD_hosts_n": len(as_),
                "SSD_media": round(float(np.mean(as_)), 5),
                "SSD_mediana": round(float(np.median(as_)), 5),
                "KS_stat": round(float(ks_s), 4),
                "KS_p": float(ks_p),
                "MW_p": float(mw_p),
            })
        return pl.DataFrame(rows)

    def analyze_via3_ppi_delta(
        self,
        net: BipartiteNetwork,
        n_null: int = 1000,
        burn_in_factor: int = 10,
        step_factor: int = 10,
        seed: int = 123,
        independent_replicas: bool = False,
        save_outputs: bool = True,
        output_tag: str | None = None,
    ) -> pl.DataFrame:
        """Calcola PPI-Complex Delta (STRING) e PPI-Dup Delta (STRING+Paralog) su Via 3."""
        if not net.use_ppi:
            raise ValueError("PPI-Delta richiede un layer proteico con matrice STRING PPI attiva.")

        df_via3 = pl.read_parquet(self.paths.silver_lane3_euteleostomi)
        df_phylo = pl.read_parquet(self.paths.silver_phylostratigraphy)

        cohorts_raw = [
            ("Eut_intragenic_WGD", df_via3.filter(pl.col("genomic_context") == "intragenic_WGD_host")["lncrna_gene_id"].to_list()),
            ("Eut_intragenic_other", df_via3.filter(pl.col("genomic_context") == "intragenic_other_host")["lncrna_gene_id"].to_list()),
            ("Eut_intergenic_lincRNA", df_via3.filter(pl.col("genomic_context") == "intergenic_lincRNA")["lncrna_gene_id"].to_list()),
            ("Control_12_Eutheria", df_phylo.filter(pl.col("age_node_num") == 12)["lncrna_gene_id"].to_list()),
            ("Control_1_Homo", df_phylo.filter(pl.col("age_node_num") == 1)["lncrna_gene_id"].to_list()),
        ]

        active_labels: list[str] = []
        node_arrays: list[np.ndarray] = []
        offsets_list = [0]
        for label, id_list in cohorts_raw:
            arr = np.array(
                [net.u_to_idx[g] for g in set(id_list) if g in net.u_to_idx],
                dtype=np.int32,
            )
            if len(arr) > 0:
                active_labels.append(label)
                node_arrays.append(arr)
                offsets_list.append(offsets_list[-1] + len(arr))

        if not node_arrays:
            return pl.DataFrame()

        all_nodes = np.concatenate(node_arrays).astype(np.int32)
        cohort_offsets = np.array(offsets_list, dtype=np.int32)

        obs_c_all, obs_d_all = evaluate_single_node_ppi_delta_numba(
            net.offsets, net.edge_v, all_nodes, net.ppi_mat, net.dup_mat, net.use_dup
        )
        null_c_all, null_d_all = run_null_ensemble_ppi_delta_multi_numba(
            net.edge_u, net.edge_v, net.offsets, len(net.v_names),
            all_nodes, cohort_offsets,
            net.ppi_mat, net.dup_mat, net.use_dup,
            n_null, burn_in_factor, step_factor, seed, independent_replicas,
        )

        rows = []
        for idx_c, label in enumerate(active_labels):
            s = cohort_offsets[idx_c]
            e = cohort_offsets[idx_c + 1]
            obs_c = obs_c_all[s:e]
            obs_d = obs_d_all[s:e]

            specs = [("ppi_complex_delta", obs_c, null_c_all[idx_c])]
            if net.use_dup:
                specs.append(("ppi_dup_delta", obs_d, null_d_all[idx_c]))

            for m_type, obs_arr, null_arr in specs:
                obs_tot = float(obs_arr.sum())
                n_mean, n_std, z_val, p_emp = compute_zscore(obs_tot, null_arr)
                rows.append({
                    "Layer": net.layer_name.upper(),
                    "Tipo_Motivo": m_type,
                    "Coorte_Via3": label,
                    "N_in_rete": len(obs_arr),
                    "Frac_PPI_Delta_pos_pct": round(100.0 * float(np.mean(obs_arr > 0)), 2),
                    "Media_PPI_Delta_per_gene": round(float(obs_arr.mean()), 3),
                    "Totale_osservato": int(obs_tot),
                    "Null_Mean": round(n_mean, 2),
                    "Null_Std": round(n_std, 2),
                    "Z_score": round(z_val, 2) if np.isfinite(z_val) else float("nan"),
                    "p_empirico": round(p_emp, 5),
                })

        df_out = pl.DataFrame(rows)
        if save_outputs:
            suffix = f"_{output_tag}" if output_tag else ""
            df_out.write_csv(self.paths.gold / f"phase3_{net.layer_name}_via3_ppi_delta{suffix}.csv")
        return df_out

    def plot_phase3_comparison(
        self,
        net: BipartiteNetwork,
        res_via1: dict,
        res_via2: dict,
        n_boot: int = 10_000,
        boot_seed: int = 42,
        save_fig: bool = True,
    ) -> tuple[Path | None, Path | None]:
        """Figura riassuntiva Fase 3 con CI bootstrap non parametrici (B=10,000) su S_SD."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), dpi=140)
        fig.patch.set_facecolor("white")

        ax = axes[0]
        labels_a = ["Via 1 (Ramo A)\nWGD", "Via 1 (Ramo A)\nSSD ant.", "Via 2 (A ∩ B)\nWGD", "Via 2 (A ∩ B)\nSSD ant."]
        vals_a, err_lo_a, err_hi_a, counts_a = [], [], [], []
        colors_a = ["#D62728", "#1F77B4", "#B2182B", "#4393C3"]
        b_idx = 0
        for res in (res_via1, res_via2):
            for dt in ("WGD", "SSD"):
                if dt in res["pair_tables"] and len(res["pair_tables"][dt]) > 0:
                    arr = res["pair_tables"][dt]["sorensen_dice"].to_numpy()
                    m_val, ci_lo, ci_hi = bootstrap_mean_ci(arr, n_boot=n_boot, seed=boot_seed + b_idx)
                    vals_a.append(m_val)
                    err_lo_a.append(max(0.0, m_val - ci_lo))
                    err_hi_a.append(max(0.0, ci_hi - m_val))
                    counts_a.append(len(arr))
                else:
                    vals_a.append(0.0)
                    err_lo_a.append(0.0)
                    err_hi_a.append(0.0)
                    counts_a.append(0)
                b_idx += 1

        x_a = np.arange(len(labels_a))
        yerr_a = np.vstack([err_lo_a, err_hi_a])
        ax.bar(x_a, vals_a, yerr=yerr_a, capsize=4, color=colors_a, alpha=0.88, edgecolor="black", linewidth=0.7)
        x_labels_n = [f"{lbl}\n(n={n})" for lbl, n in zip(labels_a, counts_a)]
        ax.set_xticks(x_a)
        ax.set_xticklabels(x_labels_n, fontsize=8.3)
        ax.set_ylim(bottom=0.0)
        ax.set_ylabel("Indice di Sørensen–Dice medio (CI 95% bootstrap)", fontsize=9.5)
        ax.set_title(f"A) Ridondanza Target — Layer {net.layer_name.upper()}\n(WGD vs SSD Antiche)", fontsize=10.2, fontweight="bold")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)

        ax = axes[1]
        df_z2: pl.DataFrame = res_via2["z_summary"]
        metrics_order = df_z2["Metrica"].unique(maintain_order=True).to_list() if not df_z2.is_empty() else []
        y_b = np.arange(len(metrics_order))
        z_wgd, z_ssd = [], []
        for m in metrics_order:
            w_row = df_z2.filter((pl.col("Metrica") == m) & (pl.col("Tipo") == "WGD"))
            s_row = df_z2.filter((pl.col("Metrica") == m) & (pl.col("Tipo") == "SSD"))
            vw = float(w_row["Z_score"][0]) if len(w_row) else float("nan")
            vs = float(s_row["Z_score"][0]) if len(s_row) else float("nan")
            z_wgd.append(vw if np.isfinite(vw) else 0.0)
            z_ssd.append(vs if np.isfinite(vs) else 0.0)

        h = 0.35
        ax.barh(y_b - h / 2, z_wgd, height=h, color="#D62728", alpha=0.9, label="WGD (Via 2: A ∩ B)")
        ax.barh(y_b + h / 2, z_ssd, height=h, color="#1F77B4", alpha=0.85, label="SSD Antiche (Via 2)")
        ax.axvline(5.0, color="black", linestyle="--", linewidth=1.0, label="Soglia Mottes (|Z| = 5)")
        ax.axvline(-5.0, color="black", linestyle="--", linewidth=1.0)
        ax.axvline(0.0, color="gray", linestyle="-", linewidth=0.8)
        ax.set_yticks(y_b)
        ax.set_yticklabels(metrics_order, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("Z-score (Modello Nullo di Maslov–Sneppen)", fontsize=9.5)
        ax.set_title(f"B) Arricchimento Motivi (Via 2: A ∩ B)\nLayer {net.layer_name.upper()}", fontsize=10.2, fontweight="bold")
        ax.grid(True, axis="x", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc="best")

        ax = axes[2]
        if "WGD" in res_via2["pair_tables"] and len(res_via2["pair_tables"]["WGD"]) > 0:
            tw2 = res_via2["pair_tables"]["WGD"]
            x_c = tw2["sorensen_dice"].to_numpy()
            y_c = tw2["norm_delta_k"].to_numpy()
            c_val = tw2["ppi_complex_bifans"].to_numpy() if net.use_ppi else tw2["v_motifs"].to_numpy()
            sc = ax.scatter(
                x_c, y_c, c=np.log1p(c_val), cmap="viridis",
                s=34, alpha=0.82, edgecolors="k", linewidths=0.35
            )
            cbar_lbl = "log(1 + PPI-Complex Bifans)" if net.use_ppi else "log(1 + V-Motifs)"
            fig.colorbar(sc, ax=ax, label=cbar_lbl)
        else:
            ax.text(
                0.5, 0.5, "Nessuna coppia WGD con entrambi\ni membri mappati in questo layer",
                ha="center", va="center", fontsize=9.5, color="gray", transform=ax.transAxes
            )
        ax.set_xlabel("Similarità Target (Sørensen–Dice)", fontsize=9.5)
        ax.set_ylabel("Asimmetria di Grado $|k_1 - k_2| / (k_1 + k_2)$", fontsize=9.5)
        ax.set_title(f"C) Ridondanza vs Neofunzionalizzazione\n(Coppie WGD Via 2 — {net.layer_name.upper()})", fontsize=10.2, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()

        png_path = pdf_path = None
        if save_fig:
            png_path = self.paths.gold / f"phase3_{net.layer_name}_motifs_summary.png"
            pdf_path = self.paths.gold / f"phase3_{net.layer_name}_motifs_summary.pdf"
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")
            logger.info(f"Figure Fase 3 salvate in Gold: {png_path.name}, {pdf_path.name}")

        plt.show()
        return png_path, pdf_path

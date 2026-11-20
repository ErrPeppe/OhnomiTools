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
    compute_zscore,
    evaluate_pair_motifs_numba,
    evaluate_single_node_ppi_delta_numba,
    run_null_ensemble_pairs_numba,
    run_null_ensemble_ppi_delta_numba,
)

logger = logging.getLogger("OhnomiTools.network")


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
    """
    Fase 3: Analisi topologica, ridondanza regolatoria (Sørensen-Dice, Jaccard, ΔDegree),
    motivi di rete (V-motifs, Bifans, Duplicated Bifans, PPI-Bifans, PPI-Delta) e modelli
    nulli di Maslov-Sneppen su Layer RBP (+ STRING PPI) e Layer miRNA per tutte e 3 le Vie.
    """

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
                # Fallback sicuro basato sui valori reali (mai sui nomi colonna)
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

            df = (
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
            return df

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
                    i2 = v_to_idx.get(p2)
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
                        i2 = v_to_idx.get(g2)
                        if i1 is not None and i2 is not None and i1 != i2 and dup_mat[i1, i2] == 0:
                            dup_mat[i1, i2] = 1
                            dup_mat[i2, i1] = 1
                            n_dup_mapped += 1
            use_dup = n_dup_mapped > 0

        logger.info(
            f"Rete [{layer.upper()}] costruita: {n_u:,} lncRNA | {n_v:,} target | "
            f"{len(edge_u):,} archi unici | PPI STRING mappati (>={ppi_min_score}): {n_ppi_mapped:,} | "
            f"Coppie target paraloghi: {n_dup_mapped:,}"
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
            raise ValueError(f"Lane non valida per analisi a coppie: '{lane}' (usa 'via1' o 'via2').")

        if strict_1to1_only and "is_strict_1to1_host" in df.columns:
            df = df.filter(pl.col("is_strict_1to1_host") == True)

        if dedup_lncrna_pair_key and "lncrna_pair_key" in df.columns:
            df = df.unique(subset=["lncrna_pair_key"])

        return df

    def analyze_degree_distributions(self, net: BipartiteNetwork) -> dict[str, pl.DataFrame]:
        df_deg = pl.DataFrame({
            "lncrna_id": net.u_names,
            "k_out": net.degrees_u.astype(np.int64),
        })

        def _extract_ids(df_p: pl.DataFrame) -> set[str]:
            return set(df_p["lncrna_1"].to_list()) | set(df_p["lncrna_2"].to_list())

        wgd1_ids = _extract_ids(self._load_lane_pairs("via1", "WGD"))
        ssd1_ids = _extract_ids(self._load_lane_pairs("via1", "SSD"))
        wgd2_ids = _extract_ids(self._load_lane_pairs("via2", "WGD"))
        ssd2_ids = _extract_ids(self._load_lane_pairs("via2", "SSD"))

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
        n_null: int = 100,
        burn_in_factor: int = 6,
        step_factor: int = 2,
        seed: int = 42,
        strict_1to1_only: bool = False,
        max_ssd_pairs_for_null: int | None = None,
    ) -> dict[str, pl.DataFrame | dict]:
        results_rows: list[dict] = []
        pair_tables: dict[str, pl.DataFrame] = {}
        null_dists: dict[str, dict[str, np.ndarray]] = {}

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
            n_in_net = len(pair_u)

            if n_in_net == 0:
                logger.warning(f"[{net.layer_name.upper()} | {lane} | {dup_type}] Nessuna coppia con entrambi i nodi in rete.")
                continue

            obs_v, obs_dice, obs_jacc, obs_sb, obs_db, obs_pb = evaluate_pair_motifs_numba(
                net.offsets, net.edge_v, len(net.v_names),
                pair_u, pair_w, net.ppi_mat, net.dup_mat, net.use_ppi, net.use_dup
            )

            deg1 = net.degrees_u[pair_u].astype(np.int64)
            deg2 = net.degrees_u[pair_w].astype(np.int64)
            delta_deg = np.abs(deg1 - deg2)
            norm_delta_deg = delta_deg / (deg1 + deg2)

            track_pp = n_in_net <= 25000
            (
                null_v, null_dice, null_sb, null_db, null_pb,
                pv_sum, pv_sq, ppb_sum, ppb_sq,
            ) = run_null_ensemble_pairs_numba(
                net.edge_u, net.edge_v, net.offsets, len(net.v_names),
                pair_u, pair_w, net.ppi_mat, net.dup_mat, net.use_ppi, net.use_dup,
                n_null, burn_in_factor, step_factor,
                seed + (0 if dup_type == "WGD" else 1000),
                track_pp,
            )

            null_dists[dup_type] = {
                "v_motifs": null_v,
                "mean_dice": null_dice,
                "simple_bifans": null_sb,
                "dup_bifans": null_db,
                "ppi_bifans": null_pb,
            }

            if track_pp:
                pv_mean = pv_sum / n_null
                pv_std = np.sqrt(np.clip((pv_sq - n_null * pv_mean * pv_mean) / max(1, n_null - 1), 0.0, None))
                pair_v_z = np.zeros(n_in_net, dtype=np.float64)
                np.divide(obs_v - pv_mean, pv_std, out=pair_v_z, where=pv_std > 1e-12)

                ppb_mean = ppb_sum / n_null
                ppb_std = np.sqrt(np.clip((ppb_sq - n_null * ppb_mean * ppb_mean) / max(1, n_null - 1), 0.0, None))
                pair_pb_z = np.zeros(n_in_net, dtype=np.float64)
                np.divide(obs_pb - ppb_mean, ppb_std, out=pair_pb_z, where=ppb_std > 1e-12)
            else:
                pair_v_z = np.full(n_in_net, np.nan)
                pair_pb_z = np.full(n_in_net, np.nan)

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
                pl.Series("simple_bifans", obs_sb),
                pl.Series("dup_bifans", obs_db),
                pl.Series("ppi_bifans", obs_pb),
                pl.Series("v_motif_pair_z", np.round(pair_v_z, 3)),
                pl.Series("ppi_bifan_pair_z", np.round(pair_pb_z, 3)),
            ])
            pair_tables[dup_type] = df_pairs_out

            metric_specs = [
                ("V-Motifs (totale)", float(obs_v.sum()), null_v),
                ("Mean Sørensen-Dice", float(obs_dice.mean()), null_dice),
                ("Simple Bifans (totale)", float(obs_sb.sum()), null_sb),
                ("Duplicated Bifans", float(obs_db.sum()), null_db),
                ("PPI-Bifans (STRING)", float(obs_pb.sum()), null_pb),
            ]
            for m_name, obs_val, n_arr in metric_specs:
                if m_name == "PPI-Bifans (STRING)" and not net.use_ppi:
                    continue
                if m_name == "Duplicated Bifans" and not net.use_dup:
                    continue
                n_mean, n_std, z_val, p_emp = compute_zscore(obs_val, n_arr)
                results_rows.append({
                    "Layer": net.layer_name.upper(),
                    "Via": lane,
                    "Tipo": dup_type,
                    "Coppie_totali": n_total_pairs,
                    "Coppie_in_rete": n_in_net,
                    "Frac_V_pos_pct": round(100.0 * float(np.mean(obs_v > 0)), 2),
                    "Frac_PPI_Bifan_pos_pct": round(100.0 * float(np.mean(obs_pb > 0)), 2) if net.use_ppi else 0.0,
                    "Metrica": m_name,
                    "Osservato": round(obs_val, 5),
                    "Null_Mean": round(n_mean, 5),
                    "Null_Std": round(n_std, 5),
                    "Z_score": round(z_val, 2),
                    "p_empirico": round(p_emp, 5),
                })

        df_z_summary = pl.DataFrame(results_rows)
        suffix = f"{net.layer_name}_{lane}" + ("_1to1" if strict_1to1_only else "")
        df_z_summary.write_csv(self.paths.gold / f"phase3_{suffix}_null_zscores.csv")

        comp_rows = []
        if "WGD" in pair_tables and "SSD" in pair_tables:
            tw, ts = pair_tables["WGD"], pair_tables["SSD"]
            for col in ["sorensen_dice", "jaccard", "v_motifs", "simple_bifans", "norm_delta_k", "ppi_bifans"]:
                if col == "ppi_bifans" and not net.use_ppi:
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

            self.paths.save_parquet_safe(tw, self.paths.silver / f"phase3_{suffix}_wgd_pair_metrics.parquet")

        df_comp = pl.DataFrame(comp_rows)
        if not df_comp.is_empty():
            df_comp.write_csv(self.paths.gold / f"phase3_{suffix}_wgd_vs_ssd_divergence.csv")

        return {
            "z_summary": df_z_summary,
            "wgd_vs_ssd": df_comp,
            "pair_tables": pair_tables,
            "null_dists": null_dists,
        }

    def analyze_via3_ppi_delta(
        self,
        net: BipartiteNetwork,
        n_null: int = 100,
        burn_in_factor: int = 6,
        step_factor: int = 2,
        seed: int = 123,
    ) -> pl.DataFrame:
        if not net.use_ppi:
            raise ValueError("PPI-Delta richiede un layer proteico con matrice STRING PPI attiva.")

        df_via3 = pl.read_parquet(self.paths.silver_lane3_euteleostomi)
        df_phylo = pl.read_parquet(self.paths.silver_phylostratigraphy)

        cohorts = [
            ("Eut_intragenic_WGD", df_via3.filter(pl.col("genomic_context") == "intragenic_WGD_host")["lncrna_gene_id"].to_list()),
            ("Eut_intragenic_other", df_via3.filter(pl.col("genomic_context") == "intragenic_other_host")["lncrna_gene_id"].to_list()),
            ("Eut_intergenic_lincRNA", df_via3.filter(pl.col("genomic_context") == "intergenic_lincRNA")["lncrna_gene_id"].to_list()),
            ("Control_12_Eutheria", df_phylo.filter(pl.col("age_node_num") == 12)["lncrna_gene_id"].to_list()),
            ("Control_1_Homo", df_phylo.filter(pl.col("age_node_num") == 1)["lncrna_gene_id"].to_list()),
        ]

        rows = []
        for idx_c, (label, id_list) in enumerate(cohorts):
            node_idxs = np.array(
                [net.u_to_idx[g] for g in set(id_list) if g in net.u_to_idx],
                dtype=np.int32,
            )
            if len(node_idxs) == 0:
                continue

            obs_counts = evaluate_single_node_ppi_delta_numba(net.offsets, net.edge_v, node_idxs, net.ppi_mat)
            obs_tot = float(obs_counts.sum())
            null_arr = run_null_ensemble_ppi_delta_numba(
                net.edge_u, net.edge_v, net.offsets, node_idxs, net.ppi_mat,
                n_null, burn_in_factor, step_factor, seed + idx_c * 17
            )
            n_mean, n_std, z_val, p_emp = compute_zscore(obs_tot, null_arr)
            rows.append({
                "Layer": net.layer_name.upper(),
                "Coorte_Via3": label,
                "N_in_rete": len(node_idxs),
                "Frac_PPI_Delta_pos_pct": round(100.0 * float(np.mean(obs_counts > 0)), 2),
                "Media_PPI_Delta_per_gene": round(float(obs_counts.mean()), 3),
                "Totale_osservato": int(obs_tot),
                "Null_Mean": round(n_mean, 2),
                "Null_Std": round(n_std, 2),
                "Z_score": round(z_val, 2),
                "p_empirico": round(p_emp, 5),
            })

        df_out = pl.DataFrame(rows)
        df_out.write_csv(self.paths.gold / f"phase3_{net.layer_name}_via3_ppi_delta.csv")
        return df_out

    def plot_phase3_comparison(
        self,
        net: BipartiteNetwork,
        res_via1: dict,
        res_via2: dict,
        save_fig: bool = True,
    ) -> tuple[Path | None, Path | None]:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), dpi=140)
        fig.patch.set_facecolor("white")

        # --- Pannello A: Sørensen-Dice medio WGD vs SSD (Via 1 e Via 2) ---
        ax = axes[0]
        labels_a = ["Via 1 (Ramo A)\nWGD", "Via 1 (Ramo A)\nSSD ant.", "Via 2 (A ∩ B)\nWGD", "Via 2 (A ∩ B)\nSSD ant."]
        vals_a = []
        errs_a = []
        counts_a = []
        colors_a = ["#D62728", "#1F77B4", "#B2182B", "#4393C3"]
        for res in (res_via1, res_via2):
            for dt in ("WGD", "SSD"):
                if dt in res["pair_tables"]:
                    arr = res["pair_tables"][dt]["sorensen_dice"].to_numpy()
                    vals_a.append(float(np.mean(arr)))
                    errs_a.append(float(1.96 * np.std(arr) / np.sqrt(max(1, len(arr)))))
                    counts_a.append(len(arr))
                else:
                    vals_a.append(0.0)
                    errs_a.append(0.0)
                    counts_a.append(0)

        x_a = np.arange(len(labels_a))
        ax.bar(x_a, vals_a, yerr=errs_a, capsize=4, color=colors_a, alpha=0.88, edgecolor="black", linewidth=0.7)
        x_labels_n = [f"{lbl}\n(n={n})" for lbl, n in zip(labels_a, counts_a)]
        ax.set_xticks(x_a)
        ax.set_xticklabels(x_labels_n, fontsize=8.3)
        ax.set_ylabel("Indice di Sørensen–Dice medio (± CI 95%)", fontsize=9.5)
        ax.set_title(f"A) Ridondanza Target — Layer {net.layer_name.upper()}\n(WGD vs SSD Antiche)", fontsize=10.2, fontweight="bold")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)

        # --- Pannello B: Z-score Maslov-Sneppen su Via 2 (A ∩ B) ---
        ax = axes[1]
        df_z2: pl.DataFrame = res_via2["z_summary"]
        metrics_order = df_z2["Metrica"].unique(maintain_order=True).to_list()
        y_b = np.arange(len(metrics_order))
        z_wgd = []
        z_ssd = []
        for m in metrics_order:
            w_row = df_z2.filter((pl.col("Metrica") == m) & (pl.col("Tipo") == "WGD"))
            s_row = df_z2.filter((pl.col("Metrica") == m) & (pl.col("Tipo") == "SSD"))
            z_wgd.append(float(w_row["Z_score"][0]) if len(w_row) else 0.0)
            z_ssd.append(float(s_row["Z_score"][0]) if len(s_row) else 0.0)

        h = 0.35
        ax.barh(y_b - h / 2, z_wgd, height=h, color="#D62728", alpha=0.9, label="WGD (Via 2: A ∩ B)")
        ax.barh(y_b + h / 2, z_ssd, height=h, color="#1F77B4", alpha=0.85, label="SSD Antiche (Via 2)")
        ax.axvline(5.0, color="black", linestyle="--", linewidth=1.0, label="Soglia Mottes (|Z| = 5)")
        ax.axvline(-5.0, color="black", linestyle="--", linewidth=1.0)
        ax.axvline(0.0, color="gray", linestyle="-", linewidth=0.8)
        ax.set_yticks(y_b)
        ax.set_yticklabels(metrics_order, fontsize=8.8)
        ax.invert_yaxis()
        ax.set_xlabel("Z-score (Modello Nullo di Maslov–Sneppen)", fontsize=9.5)
        ax.set_title(f"B) Arricchimento Motivi (Via 2: A ∩ B)\nLayer {net.layer_name.upper()}", fontsize=10.2, fontweight="bold")
        ax.grid(True, axis="x", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc="best")

        # --- Pannello C: Scatter Topologico Via 2 WGD (Sørensen-Dice vs ΔDegree norm.) ---
        ax = axes[2]
        if "WGD" in res_via2["pair_tables"] and len(res_via2["pair_tables"]["WGD"]) > 0:
            tw2 = res_via2["pair_tables"]["WGD"]
            x_c = tw2["sorensen_dice"].to_numpy()
            y_c = tw2["norm_delta_k"].to_numpy()
            c_val = tw2["ppi_bifans"].to_numpy() if net.use_ppi else tw2["v_motifs"].to_numpy()
            sc = ax.scatter(
                x_c, y_c, c=np.log1p(c_val), cmap="viridis",
                s=34, alpha=0.82, edgecolors="k", linewidths=0.35
            )
            cbar_lbl = "log(1 + PPI-Bifans)" if net.use_ppi else "log(1 + V-Motifs)"
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

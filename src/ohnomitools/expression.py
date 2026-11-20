# src/ohnomitools/expression.py
from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
import polars as pl
from scipy.stats import fisher_exact, ks_2samp, mannwhitneyu

from .config import PATHS, ProjectPaths

logger = logging.getLogger("OhnomiTools.expression")

EXPRESSION_CONTEXT_COLS: list[str] = [
    "Normal Tissue/Cell Line",
    "Organ Development",
    "Preimplantation Embryo",
    "Cell Differentiation",
    "Subcellular Localization",
    "Exosome",
    "Cancer Cell Line",
    "Virus Infection",
    "Circadian Rhythm",
]


def _ordinal_expr(col_name: str) -> pl.Expr:
    """Codifica ordinale standard LncBook v2.0: HC=3, MC=2, LC=1, NE/altro=0."""
    cleaned = pl.col(col_name).cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    return (
        pl.when(cleaned == "HC").then(3)
        .when(cleaned == "MC").then(2)
        .when(cleaned == "LC").then(1)
        .otherwise(0)
        .cast(pl.Int8)
    )


class MultiOmicsExpressionAnalyzer:
    """
    Fase 4: Analisi della dinamica di espressione multi-contesto (LncBook v2.0),
    divergenza di espressione nelle coppie duplicate (Cosine Distance 1 - cos e
    Manhattan normalizzata), profili epigenetici di metilazione in patologia e
    potenziale codificante nascosto (Small Proteins / SmProt) sulle 3 Vie analitiche.
    """

    def __init__(self, paths: ProjectPaths = PATHS) -> None:
        self.paths = paths

    def build_gene_multiomics_table(self, force_rebuild: bool = False) -> pl.DataFrame:
        """
        Unifica a livello di singolo gene lncRNA (`lncrna_gene_id`) le informazioni di:
        - Espressione meta-categoriale (9 contesti codificati 0..3, breadth, score medio,
          Featured Expression, Expression Capacity globale)
        - Metilazione differenziale in 16 patologie (promoter/body, hyper/hypo)
        - Small Proteins (SmProt: numero micropeptidi, evidenza Mass Spec / Ribo-seq)
        """
        out_pq = self.paths.silver / "lncbook_gene_multiomics_features.parquet"
        if out_pq.exists() and not force_rebuild:
            return pl.read_parquet(out_pq)

        # Base: tutti i geni lncRNA canonici del GTF LncBook v2.0
        df_base = pl.read_parquet(self.paths.silver_lncrna_genes).select([
            pl.col("gene_id").alias("lncrna_gene_id"),
            pl.col("chr"),
            pl.col("strand"),
            (pl.col("end") - pl.col("start")).alias("lnc_length_bp"),
            pl.col("n_isoforms"),
        ]).unique(subset=["lncrna_gene_id"])

        # 1. Espressione LncBook v2.0
        df_expr_raw = pl.read_parquet(self.paths.silver_lncbook_expr).unique(subset=["Gene ID"])
        avail_ctx = [c for c in EXPRESSION_CONTEXT_COLS if c in df_expr_raw.columns]

        ctx_exprs = [_ordinal_expr(c).alias(f"ctx_{i}") for i, c in enumerate(avail_ctx)]
        df_expr = (
            df_expr_raw.select([
                pl.col("Gene ID").cast(pl.Utf8).str.strip_chars().alias("lncrna_gene_id"),
                pl.col("Symbol").cast(pl.Utf8).alias("expr_symbol"),
                (
                    pl.col("Featured Expression").cast(pl.Utf8).str.strip_chars().str.to_lowercase() == "yes"
                ).alias("is_featured_expr"),
                _ordinal_expr("Expression Capacity").alias("global_expr_capacity"),
                *ctx_exprs,
            ])
            .with_columns([
                pl.sum_horizontal([pl.col(f"ctx_{i}") for i in range(len(avail_ctx))]).alias("expr_sum_score"),
                pl.sum_horizontal([(pl.col(f"ctx_{i}") > 0).cast(pl.Int8) for i in range(len(avail_ctx))]).alias("n_ctx_expressed"),
                pl.sum_horizontal([(pl.col(f"ctx_{i}") >= 2).cast(pl.Int8) for i in range(len(avail_ctx))]).alias("n_ctx_hc_mc"),
                pl.sum_horizontal([(pl.col(f"ctx_{i}") == 3).cast(pl.Int8) for i in range(len(avail_ctx))]).alias("n_ctx_hc"),
                pl.lit(True).alias("in_expr_db"),
            ])
            .with_columns(
                (pl.col("expr_sum_score") / float(max(1, len(avail_ctx)))).alias("expr_mean_score")
            )
        )

        # Rinomina leggibile anche per i singoli contesti
        ctx_rename = {f"ctx_{i}": f"expr__{c}" for i, c in enumerate(avail_ctx)}
        df_expr = df_expr.rename(ctx_rename)

        df_merged = df_base.join(df_expr, on="lncrna_gene_id", how="left").with_columns(
            pl.col("in_expr_db").fill_null(False)
        )

        # 2. Metilazione LncBook v2.0 (aggregata per Gene ID su tutte le regioni/patologie)
        if self.paths.silver_lncbook_methylation.exists():
            df_m_raw = pl.read_parquet(self.paths.silver_lncbook_methylation)
            meta_cols = {"Gene ID", "Symbol", "Methylation Region"}
            disease_cols = [c for c in df_m_raw.columns if c not in meta_cols]

            hyper_flags = [
                pl.col(c).cast(pl.Utf8).str.to_lowercase().str.contains("hyper").fill_null(False).cast(pl.Int16)
                for c in disease_cols
            ]
            hypo_flags = [
                pl.col(c).cast(pl.Utf8).str.to_lowercase().str.contains("hypo").fill_null(False).cast(pl.Int16)
                for c in disease_cols
            ]

            df_meth = (
                df_m_raw.with_columns([
                    pl.col("Gene ID").cast(pl.Utf8).str.strip_chars().alias("lncrna_gene_id"),
                    pl.col("Methylation Region").cast(pl.Utf8).str.to_lowercase().alias("meth_region"),
                    pl.sum_horizontal(hyper_flags).alias("_n_hyper_row"),
                    pl.sum_horizontal(hypo_flags).alias("_n_hypo_row"),
                ])
                .group_by("lncrna_gene_id")
                .agg([
                    pl.lit(True).alias("has_diff_methylation"),
                    (pl.col("meth_region").str.contains("promoter")).any().alias("has_promoter_meth"),
                    (pl.col("meth_region").str.contains("body")).any().alias("has_body_meth"),
                    pl.col("_n_hyper_row").sum().alias("n_meth_hyper_events"),
                    pl.col("_n_hypo_row").sum().alias("n_meth_hypo_events"),
                ])
                .with_columns(
                    (pl.col("n_meth_hyper_events") + pl.col("n_meth_hypo_events")).alias("n_meth_total_events")
                )
            )
            df_merged = df_merged.join(df_meth, on="lncrna_gene_id", how="left").with_columns([
                pl.col("has_diff_methylation").fill_null(False),
                pl.col("has_promoter_meth").fill_null(False),
                pl.col("has_body_meth").fill_null(False),
                pl.col("n_meth_hyper_events").fill_null(0),
                pl.col("n_meth_hypo_events").fill_null(0),
                pl.col("n_meth_total_events").fill_null(0),
            ])

        # 3. Small Proteins (SmProt)
        if self.paths.silver_lncbook_sprotein.exists():
            df_sp_raw = pl.read_parquet(self.paths.silver_lncbook_sprotein)
            df_sp = (
                df_sp_raw.with_columns([
                    pl.col("Gene ID").cast(pl.Utf8).str.strip_chars().alias("lncrna_gene_id"),
                    pl.col("Experimental Evidence").cast(pl.Utf8).str.to_lowercase().alias("_ev"),
                ])
                .group_by("lncrna_gene_id")
                .agg([
                    pl.lit(True).alias("has_sprotein"),
                    pl.col("SmProt ID").n_unique().alias("n_sproteins"),
                    pl.col("_ev").str.contains("mass").any().alias("has_sprotein_ms"),
                    pl.col("_ev").str.contains("ribo").any().alias("has_sprotein_riboseq"),
                ])
            )
            df_merged = df_merged.join(df_sp, on="lncrna_gene_id", how="left").with_columns([
                pl.col("has_sprotein").fill_null(False),
                pl.col("n_sproteins").fill_null(0),
                pl.col("has_sprotein_ms").fill_null(False),
                pl.col("has_sprotein_riboseq").fill_null(False),
            ])

        self.paths.save_parquet_safe(df_merged, out_pq)
        logger.info(
            f"Tabella multi-omica costruita per {len(df_merged):,} geni lncRNA "
            f"(in_expr_db={df_merged['in_expr_db'].sum():,})."
        )
        return df_merged

    def _get_three_lane_cohorts(self) -> list[tuple[str, str, set[str]]]:
        """Restituisce esattamente le stesse coorti delle 3 Vie usate nella Fase 3."""
        def _pair_ids(pq: Path, ancient_ssd: bool = False) -> set[str]:
            df = pl.read_parquet(pq)
            if ancient_ssd and "is_ancient_ssd" in df.columns:
                df = df.filter(pl.col("is_ancient_ssd") == True)
            return set(df["lncrna_1"].to_list()) | set(df["lncrna_2"].to_list())

        wgd1 = _pair_ids(self.paths.silver_lane1_wgd_pairs)
        ssd1 = _pair_ids(self.paths.silver_lane1_ssd_pairs, ancient_ssd=True)
        wgd2 = _pair_ids(self.paths.silver_lane2_wgd_pairs)
        ssd2 = _pair_ids(self.paths.silver_lane2_ssd_pairs)

        df_via3 = pl.read_parquet(self.paths.silver_lane3_euteleostomi)
        df_phylo = pl.read_parquet(self.paths.silver_phylostratigraphy)
        all_ids = set(pl.read_parquet(self.paths.silver_lncrna_genes)["gene_id"].to_list())

        eut_wgd = set(df_via3.filter(pl.col("genomic_context") == "intragenic_WGD_host")["lncrna_gene_id"].to_list())
        eut_oth = set(df_via3.filter(pl.col("genomic_context") == "intragenic_other_host")["lncrna_gene_id"].to_list())
        eut_linc = set(df_via3.filter(pl.col("genomic_context") == "intergenic_lincRNA")["lncrna_gene_id"].to_list())
        eutheria = set(df_phylo.filter(pl.col("age_node_num") == 12)["lncrna_gene_id"].to_list())
        homo = set(df_phylo.filter(pl.col("age_node_num") == 1)["lncrna_gene_id"].to_list())

        return [
            ("Via 1 (Ramo A)", "WGD_intragenic", wgd1),
            ("Via 1 (Ramo A)", "SSD_ancient_intragenic", ssd1),
            ("Via 2 (A ∩ B >=15)", "WGD_AB_ancient", wgd2),
            ("Via 2 (A ∩ B >=15)", "SSD_AB_ancient", ssd2),
            ("Via 3 (Ramo B Eut.)", "Eut_intragenic_WGD", eut_wgd),
            ("Via 3 (Ramo B Eut.)", "Eut_intragenic_other", eut_oth),
            ("Via 3 (Ramo B Eut.)", "Eut_intergenic_lincRNA", eut_linc),
            ("Controllo Età", "12. Eutheria", eutheria),
            ("Controllo Età", "1. Homo", homo),
            ("Background", "All_LncBook_v2", all_ids),
        ]

    def analyze_expression_and_multiomics_cohorts(
        self,
        df_mo: pl.DataFrame | None = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Analizza a livello di singolo gene su tutte e 3 le Vie:
        1. Capacità di espressione, ampiezza (`n_ctx_expressed`, `n_ctx_hc_mc`),
           `Featured Expression`, `has_diff_methylation`, `has_sprotein`.
        2. Test statistici (Mann-Whitney U, KS, Fisher exact) sui confronti chiave.
        3. Profilo dettagliato di espressione alta (`HC` / `HC+MC`) nei 9 contesti biologici.
        """
        if df_mo is None:
            df_mo = self.build_gene_multiomics_table(force_rebuild=False)

        cohorts = self._get_three_lane_cohorts()
        summary_rows = []
        context_rows = []
        cohort_dfs: dict[str, pl.DataFrame] = {}

        avail_ctx = [c for c in EXPRESSION_CONTEXT_COLS if f"expr__{c}" in df_mo.columns]

        for lane_lbl, grp_lbl, id_set in cohorts:
            sub = df_mo.filter(pl.col("lncrna_gene_id").is_in(list(id_set)))
            sub_expr = sub.filter(pl.col("in_expr_db"))
            cohort_dfs[grp_lbl] = sub

            n_tot = len(sub)
            n_exp = len(sub_expr)
            summary_rows.append({
                "Via": lane_lbl,
                "Coorte": grp_lbl,
                "N_geni": n_tot,
                "In_Expr_DB_pct": round(100.0 * n_exp / n_tot, 2) if n_tot else 0.0,
                "Featured_Expr_pct": round(100.0 * float(sub_expr["is_featured_expr"].mean()), 2) if n_exp else 0.0,
                "Global_HC_pct": round(100.0 * float((sub_expr["global_expr_capacity"] == 3).mean()), 2) if n_exp else 0.0,
                "Mean_Score_0_3": round(float(sub_expr["expr_mean_score"].mean()), 3) if n_exp else 0.0,
                "Median_Ctx_Expr": float(sub_expr["n_ctx_expressed"].median()) if n_exp else 0.0,
                "Mean_Ctx_HC_MC": round(float(sub_expr["n_ctx_hc_mc"].mean()), 3) if n_exp else 0.0,
                "Diff_Meth_pct": round(100.0 * float(sub["has_diff_methylation"].mean()), 2) if (n_tot and "has_diff_methylation" in sub.columns) else 0.0,
                "Promoter_Meth_pct": round(100.0 * float(sub["has_promoter_meth"].mean()), 2) if (n_tot and "has_promoter_meth" in sub.columns) else 0.0,
                "SmProt_pct": round(100.0 * float(sub["has_sprotein"].mean()), 2) if (n_tot and "has_sprotein" in sub.columns) else 0.0,
                "SmProt_MS_pct": round(100.0 * float(sub["has_sprotein_ms"].mean()), 2) if (n_tot and "has_sprotein_ms" in sub.columns) else 0.0,
            })

            ctx_row: dict[str, str | float] = {"Via": lane_lbl, "Coorte": grp_lbl}
            for c in avail_ctx:
                col = f"expr__{c}"
                ctx_row[f"{c}_HC_MC_pct"] = (
                    round(100.0 * float((sub_expr[col] >= 2).mean()), 2) if n_exp else 0.0
                )
            context_rows.append(ctx_row)

        # Test statistici sui confronti chiave
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
            d1 = cohort_dfs[g1]
            d2 = cohort_dfs[g2]
            e1 = d1.filter(pl.col("in_expr_db"))
            e2 = d2.filter(pl.col("in_expr_db"))

            # Mann-Whitney su score medio di espressione (0..3)
            s1 = e1["expr_mean_score"].to_numpy()
            s2 = e2["expr_mean_score"].to_numpy()
            _, mw_expr_p = mannwhitneyu(s1, s2, alternative="two-sided") if (len(s1) >= 5 and len(s2) >= 5) else (np.nan, np.nan)

            # Fisher su Featured Expression
            fe1 = int(e1["is_featured_expr"].sum())
            fe2 = int(e2["is_featured_expr"].sum())
            or_fe, p_fe = fisher_exact([[fe1, len(e1) - fe1], [fe2, len(e2) - fe2]])

            # Fisher su Metilazione Differenziale
            m1 = int(d1["has_diff_methylation"].sum()) if "has_diff_methylation" in d1.columns else 0
            m2 = int(d2["has_diff_methylation"].sum()) if "has_diff_methylation" in d2.columns else 0
            or_m, p_m = fisher_exact([[m1, len(d1) - m1], [m2, len(d2) - m2]])

            # Fisher su Small Proteins (SmProt)
            sp1 = int(d1["has_sprotein"].sum()) if "has_sprotein" in d1.columns else 0
            sp2 = int(d2["has_sprotein"].sum()) if "has_sprotein" in d2.columns else 0
            or_sp, p_sp = fisher_exact([[sp1, len(d1) - sp1], [sp2, len(d2) - sp2]])

            test_rows.append({
                "Confronto": label,
                "ExprScore_1": round(float(np.mean(s1)), 3),
                "ExprScore_2": round(float(np.mean(s2)), 3),
                "Expr_MW_p": float(mw_expr_p),
                "Featured_OR": round(float(or_fe), 3),
                "Featured_p": float(p_fe),
                "Meth_OR": round(float(or_m), 3),
                "Meth_p": float(p_m),
                "SmProt_OR": round(float(or_sp), 3),
                "SmProt_p": float(p_sp),
            })

        df_summary = pl.DataFrame(summary_rows)
        df_contexts = pl.DataFrame(context_rows)
        df_tests = pl.DataFrame(test_rows)

        df_summary.write_csv(self.paths.gold / "phase4_multiomics_cohorts_summary.csv")
        df_contexts.write_csv(self.paths.gold / "phase4_expression_contexts_profile.csv")
        df_tests.write_csv(self.paths.gold / "phase4_multiomics_cohorts_tests.csv")

        return {"summary": df_summary, "contexts": df_contexts, "tests": df_tests}

    def analyze_pairwise_expression_divergence(
        self,
        df_mo: pl.DataFrame | None = None,
    ) -> dict[str, pl.DataFrame | dict]:
        """
        Calcola la divergenza di espressione tra i due membri di ciascuna coppia
        duplicata (Via 1 Ramo A e Via 2 A ∩ B, confrontando WGD vs SSD antiche):
        - Distanza Coseno: d_cos = 1 - (e1 · e2) / (||e1|| ||e2||) (stile Agasso et al. Fig. 5D)
        - Distanza di Manhattan normalizzata: d_L1 = sum(|e1 - e2|) / (3 * 9) in [0, 1]
        - Asimmetria di espressione: |sum(e1) - sum(e2)| / (sum(e1) + sum(e2))
        - Concordanza esatta dei contesti: frazione di contesti con uguale livello (0..3)
        """
        if df_mo is None:
            df_mo = self.build_gene_multiomics_table(force_rebuild=False)

        avail_ctx = [f"expr__{c}" for c in EXPRESSION_CONTEXT_COLS if f"expr__{c}" in df_mo.columns]
        n_ctx = len(avail_ctx)

        df_e = df_mo.filter(pl.col("in_expr_db")).select(["lncrna_gene_id", *avail_ctx])
        gene_ids = df_e["lncrna_gene_id"].to_list()
        gene_to_idx = {g: i for i, g in enumerate(gene_ids)}
        E_mat = df_e.select(avail_ctx).to_numpy().astype(np.float64)

        summary_rows = []
        test_rows = []
        pair_outputs: dict[str, pl.DataFrame] = {}

        for lane in ("via1", "via2"):
            lane_tables: dict[str, pl.DataFrame] = {}

            for dup_type in ("WGD", "SSD"):
                if lane == "via1":
                    pq = self.paths.silver_lane1_wgd_pairs if dup_type == "WGD" else self.paths.silver_lane1_ssd_pairs
                    df_p = pl.read_parquet(pq)
                    if dup_type == "SSD" and "is_ancient_ssd" in df_p.columns:
                        df_p = df_p.filter(pl.col("is_ancient_ssd") == True)
                else:
                    pq = self.paths.silver_lane2_wgd_pairs if dup_type == "WGD" else self.paths.silver_lane2_ssd_pairs
                    df_p = pl.read_parquet(pq)

                df_p = df_p.unique(subset=["lncrna_pair_key"])

                i1_list = []
                i2_list = []
                mask = []
                for g1, g2 in df_p.select(["lncrna_1", "lncrna_2"]).iter_rows():
                    idx1 = gene_to_idx.get(g1)
                    idx2 = gene_to_idx.get(g2)
                    if idx1 is not None and idx2 is not None and idx1 != idx2:
                        i1_list.append(idx1)
                        i2_list.append(idx2)
                        mask.append(True)
                    else:
                        mask.append(False)

                df_in = df_p.filter(pl.Series(mask))
                v1 = E_mat[np.array(i1_list, dtype=np.int32)]
                v2 = E_mat[np.array(i2_list, dtype=np.int32)]

                # Manhattan normalizzata in [0, 1]
                manhattan_norm = np.sum(np.abs(v1 - v2), axis=1) / (3.0 * n_ctx)

                # Cosine Distance (solo se entrambi i vettori sono non nulli)
                n1 = np.linalg.norm(v1, axis=1)
                n2 = np.linalg.norm(v2, axis=1)
                denom = n1 * n2
                both_active = denom > 1e-12
                cos_sim = np.full(len(df_in), np.nan, dtype=np.float64)
                cos_sim[both_active] = np.sum(v1[both_active] * v2[both_active], axis=1) / denom[both_active]
                cos_dist = 1.0 - np.clip(cos_sim, 0.0, 1.0)

                # Asimmetria di ampiezza
                s1 = np.sum(v1, axis=1)
                s2 = np.sum(v2, axis=1)
                asym = np.zeros(len(df_in), dtype=np.float64)
                np.divide(np.abs(s1 - s2), s1 + s2, out=asym, where=(s1 + s2) > 0)

                # Frazione di contesti identici
                exact_match = np.mean(v1 == v2, axis=1)

                keep_cols = [
                    c for c in [
                        "pair_key", "lncrna_pair_key", "gene_name_1", "gene_name_2",
                        "lncrna_1", "lncrna_2", "lncrna_age_1", "lncrna_age_2",
                        "rel_strand_1", "is_strict_1to1_host", "wgd_round"
                    ] if c in df_in.columns
                ]
                df_res = df_in.select(keep_cols).with_columns([
                    pl.Series("expr_sum_1", s1.astype(np.int32)),
                    pl.Series("expr_sum_2", s2.astype(np.int32)),
                    pl.Series("both_expressed", both_active),
                    pl.Series("cosine_distance", np.round(cos_dist, 5)),
                    pl.Series("manhattan_norm", np.round(manhattan_norm, 5)),
                    pl.Series("expr_asymmetry", np.round(asym, 5)),
                    pl.Series("exact_ctx_match", np.round(exact_match, 4)),
                ])
                lane_tables[dup_type] = df_res
                pair_outputs[f"{lane}_{dup_type}"] = df_res

                valid_cd = cos_dist[both_active]
                summary_rows.append({
                    "Via": lane,
                    "Tipo": dup_type,
                    "N_coppie_in_Expr": len(df_res),
                    "Both_Expressed_pct": round(100.0 * float(np.mean(both_active)), 2),
                    "CosDist_media": round(float(np.mean(valid_cd)), 4) if len(valid_cd) else np.nan,
                    "CosDist_mediana": round(float(np.median(valid_cd)), 4) if len(valid_cd) else np.nan,
                    "Manhattan_media": round(float(np.mean(manhattan_norm)), 4),
                    "Manhattan_mediana": round(float(np.median(manhattan_norm)), 4),
                    "Asimmetria_media": round(float(np.mean(asym)), 4),
                    "ExactMatch_media": round(float(np.mean(exact_match)), 4),
                })

                if dup_type == "WGD":
                    self.paths.save_parquet_safe(
                        df_res, self.paths.silver / f"phase4_{lane}_wgd_expression_pairs.parquet"
                    )

            # Test KS e Mann-Whitney WGD vs SSD antiche per questa Via
            tw, ts = lane_tables["WGD"], lane_tables["SSD"]
            for metric in ("cosine_distance", "manhattan_norm", "expr_asymmetry", "exact_ctx_match"):
                aw = tw[metric].drop_nulls().drop_nans().to_numpy()
                as_ = ts[metric].drop_nulls().drop_nans().to_numpy()
                ks_s, ks_p = ks_2samp(aw, as_)
                _, mw_p = mannwhitneyu(aw, as_, alternative="two-sided")
                test_rows.append({
                    "Via": lane,
                    "Metrica": metric,
                    "WGD_n": len(aw),
                    "WGD_media": round(float(np.mean(aw)), 4),
                    "WGD_mediana": round(float(np.median(aw)), 4),
                    "SSD_n": len(as_),
                    "SSD_media": round(float(np.mean(as_)), 4),
                    "SSD_mediana": round(float(np.median(as_)), 4),
                    "KS_stat": round(float(ks_s), 4),
                    "KS_p": float(ks_p),
                    "MW_p": float(mw_p),
                })

        df_sum = pl.DataFrame(summary_rows)
        df_tests = pl.DataFrame(test_rows)
        df_sum.write_csv(self.paths.gold / "phase4_pairwise_expression_summary.csv")
        df_tests.write_csv(self.paths.gold / "phase4_pairwise_expression_tests.csv")

        return {"summary": df_sum, "tests": df_tests, "pair_tables": pair_outputs}

    def plot_phase4_summary(
        self,
        cohort_res: dict,
        pair_res: dict,
        save_fig: bool = True,
    ) -> tuple[Path | None, Path | None]:
        """
        Genera la figura riassuntiva a 3 pannelli della Fase 4 (salvata in `gold/` .png/.pdf):
        - A) Heatmap/profilo % espressione alta/media (`HC+MC`) nei 9 contesti biologici
             attraverso le coorti chiave delle 3 Vie.
        - B) Distanza di espressione a coppie (Cosine Distance 1-cos e Manhattan normalizzata)
             confrontando WGD vs SSD antiche nella Via 1 e nella Via 2.
        - C) Firma Multi-Omica sulle 3 Vie: % Featured Expression, % Metilazione Differenziale
             in patologia e % Micropeptidi (SmProt).
        """
        import matplotlib.pyplot as plt

        df_ctx: pl.DataFrame = cohort_res["contexts"]
        df_sum: pl.DataFrame = cohort_res["summary"]

        fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.6), dpi=140)
        fig.patch.set_facecolor("white")

        # --- Pannello A: Profilo % HC+MC sui 9 contesti biologici ---
        ax = axes[0]
        target_cohorts = [
            ("WGD_AB_ancient", "Via 2: WGD (A∩B)", "#D62728", "o-"),
            ("SSD_AB_ancient", "Via 2: SSD ant.", "#1F77B4", "s--"),
            ("Eut_intragenic_WGD", "Via 3: Eut. WGD", "#E31A1C", "^:"),
            ("Eut_intergenic_lincRNA", "Via 3: Eut. lincRNA", "#2CA02C", "D-"),
            ("1. Homo", "Controllo: 1. Homo", "#7F7F7F", "x-."),
        ]
        ctx_cols = [c for c in df_ctx.columns if c.endswith("_HC_MC_pct")]
        short_ctx = [
            c.replace("_HC_MC_pct", "")
            .replace("Normal Tissue/Cell Line", "Normal Tissue")
            .replace("Preimplantation Embryo", "Preimpl. Embryo")
            .replace("Subcellular Localization", "Subcell. Loc.")
            .replace("Cell Differentiation", "Cell Diff.")
            for c in ctx_cols
        ]
        x_ctx = np.arange(len(ctx_cols))
        for cid, lbl, col, fmt in target_cohorts:
            row = df_ctx.filter(pl.col("Coorte") == cid)
            if len(row) > 0:
                vals = [float(row[c][0]) for c in ctx_cols]
                ax.plot(x_ctx, vals, fmt, color=col, linewidth=2.0, markersize=5.5, label=lbl)

        ax.set_xticks(x_ctx)
        ax.set_xticklabels(short_ctx, rotation=38, ha="right", fontsize=8.3)
        ax.set_ylabel("Geni con capacità Alta/Media (HC+MC, %)", fontsize=9.5)
        ax.set_title("A) Ampiezza Espressione nei 9 Contesti Biologici\n(Confronto sulle 3 Vie)", fontsize=10.2, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc="best")

        # --- Pannello B: Divergenza di Espressione a Coppie (WGD vs SSD antiche) ---
        ax = axes[1]
        groups_b = [
            ("via1_WGD", "Via 1\nWGD", "#D62728"),
            ("via1_SSD", "Via 1\nSSD ant.", "#1F77B4"),
            ("via2_WGD", "Via 2\nWGD", "#B2182B"),
            ("via2_SSD", "Via 2\nSSD ant.", "#4393C3"),
        ]
        x_b = np.arange(len(groups_b))
        cos_m, cos_e, man_m, man_e = [], [], [], []
        for key, _, _ in groups_b:
            t = pair_res["pair_tables"][key]
            cd = t["cosine_distance"].drop_nulls().drop_nans().to_numpy()
            mn = t["manhattan_norm"].drop_nulls().drop_nans().to_numpy()
            cos_m.append(float(np.mean(cd)))
            cos_e.append(float(1.96 * np.std(cd) / np.sqrt(max(1, len(cd)))))
            man_m.append(float(np.mean(mn)))
            man_e.append(float(1.96 * np.std(mn) / np.sqrt(max(1, len(mn)))))

        w = 0.35
        ax.bar(x_b - w / 2, cos_m, width=w, yerr=cos_e, capsize=4, color="#FF7F0E", alpha=0.88, edgecolor="black", linewidth=0.7, label="Cosine Distance (1 − cos)")
        ax.bar(x_b + w / 2, man_m, width=w, yerr=man_e, capsize=4, color="#6A3D9A", alpha=0.85, edgecolor="black", linewidth=0.7, label="Manhattan Norm. (L1)")
        ax.set_xticks(x_b)
        ax.set_xticklabels([g[1] for g in groups_b], fontsize=8.8)
        ax.set_ylabel("Distanza di Espressione Media (± CI 95%)", fontsize=9.5)
        ax.set_title("B) Divergenza vs Co-espressione nelle Coppie\n(WGD vs SSD Antiche — Via 1 e Via 2)", fontsize=10.2, fontweight="bold")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8.3, loc="best")

        # --- Pannello C: Firma Multi-Omica (Featured Expr, Metilazione, SmProt) ---
        ax = axes[2]
        c_order = [
            ("WGD_intragenic", "Via 1 WGD"),
            ("SSD_ancient_intragenic", "Via 1 SSD"),
            ("WGD_AB_ancient", "Via 2 WGD"),
            ("SSD_AB_ancient", "Via 2 SSD"),
            ("Eut_intragenic_WGD", "Via 3 Eut-WGD"),
            ("Eut_intergenic_lincRNA", "Via 3 Eut-linc"),
            ("1. Homo", "1. Homo"),
        ]
        y_c = np.arange(len(c_order))
        fe_v, me_v, sp_v = [], [], []
        for cid, _ in c_order:
            r = df_sum.filter(pl.col("Coorte") == cid)
            fe_v.append(float(r["Featured_Expr_pct"][0]) if len(r) else 0.0)
            me_v.append(float(r["Diff_Meth_pct"][0]) if len(r) else 0.0)
            sp_v.append(float(r["SmProt_pct"][0]) if len(r) else 0.0)

        h = 0.25
        ax.barh(y_c - h, fe_v, height=h, color="#1B9E77", label="Featured Expression (%)")
        ax.barh(y_c, me_v, height=h, color="#D95F02", label="Metilazione Diff. Patologie (%)")
        ax.barh(y_c + h, sp_v, height=h, color="#7570B3", label="Micropeptidi SmProt (%)")
        ax.set_yticks(y_c)
        ax.set_yticklabels([c[1] for c in c_order], fontsize=8.6)
        ax.invert_yaxis()
        ax.set_xlabel("Percentuale di geni nella coorte (%)", fontsize=9.5)
        ax.set_title("C) Profilo Multi-Omico Integrato\n(Espressione, Epigenetica, Micropeptidi)", fontsize=10.2, fontweight="bold")
        ax.grid(True, axis="x", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc="lower right")

        plt.tight_layout()

        png_path = pdf_path = None
        if save_fig:
            png_path = self.paths.gold / "phase4_expression_multiomics_summary.png"
            pdf_path = self.paths.gold / "phase4_expression_multiomics_summary.pdf"
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")
            logger.info(f"Figure Fase 4 salvate in Gold: {png_path.name}, {pdf_path.name}")

        plt.show()
        return png_path, pdf_path
